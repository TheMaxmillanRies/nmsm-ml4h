from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionalLinear(nn.Module):
    'Linear output head that cleanly supports an empty dataset-specific target.'

    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()
        self.output_features = int(output_features)
        self.linear = (
            nn.Linear(input_features, output_features) if output_features else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            return inputs.new_empty((*inputs.shape[:-1], 0))
        return self.linear(inputs)


@dataclass(frozen=True)
class ModelConfig:
    num_states: int
    num_features: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    max_time: int
    context: str = "full"
    boolean_causal_mask: bool = False
    categorical_cardinalities: tuple[int, ...] = ()
    use_dwell: bool = False
    num_dynamic_continuous: int = 0
    num_observation_processes: int = 0
    num_dynamic_binary: int = 0
    covariate_parameterization: str = "absolute"
    condition_covariates_on_next_state: bool = True
    time_encoding: str = "discrete_absolute"
    time2vec_dim: int = 8


class Time2Vec(nn.Module):
    'LANTERN-compatible backward-gap encoding.'

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 2:
            raise ValueError("Time2Vec dimension must be at least two")
        self.linear = nn.Linear(1, 1)
        self.frequency = nn.Linear(1, dimension - 1)

    def forward(self, elapsed: torch.Tensor) -> torch.Tensor:
        transformed = torch.log1p(elapsed.clamp_min(0.0).unsqueeze(-1))
        return torch.cat(
            [self.linear(transformed), torch.sin(self.frequency(transformed))], dim=-1
        )


def allowed_transitions(
    num_states: int,
    terminals: tuple[int, ...],
    edges: tuple[tuple[int, int], ...] = (),
) -> torch.Tensor:
    allowed = (
        torch.zeros((num_states, num_states), dtype=torch.bool)
        if edges
        else torch.ones((num_states, num_states), dtype=torch.bool)
    )
    for source, destination in edges:
        if not (0 <= source < num_states and 0 <= destination < num_states):
            raise ValueError(f"Transition edge {(source, destination)} is outside the state space")
        allowed[source, destination] = True
    for state in terminals:
        allowed[state] = False
        allowed[state, state] = True
    if not torch.all(allowed.any(dim=1)):
        raise ValueError("Every state must have at least one permitted destination")
    return allowed


def context_attention_mask(length: int, context: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    row = torch.arange(length, device=device).unsqueeze(1)
    col = torch.arange(length, device=device).unsqueeze(0)
    permitted = col <= row
    if context in {"snapshot", "semi_markov"}:
        permitted &= col == row
    elif context.startswith("window"):
        width = int(context.removeprefix("window"))
        permitted &= col >= (row - width + 1)
    elif context != "full":
        raise ValueError(f"Unknown context: {context}")
    mask = torch.zeros((length, length), device=device, dtype=dtype)
    return mask.masked_fill(~permitted, float("-inf"))


class JointCausalTransformer(nn.Module):
    'Autoregressive nMSM with state-conditioned stochastic covariate continuation.'

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_states = config.num_states
        self.max_time = config.max_time
        self.state_emb = nn.Embedding(config.num_states, config.d_model)
        self.cov_proj = nn.Linear(config.num_features, config.d_model)
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, config.d_model)
            for cardinality in config.categorical_cardinalities
        )
        if config.time_encoding == "discrete_absolute":
            self.time_emb: nn.Module | None = nn.Embedding(
                config.max_time + 1, config.d_model
            )
            self.time2vec = None
            self.visit_process_proj = None
        elif config.time_encoding == "lantern_delta":
            self.time_emb = None
            self.time2vec = nn.Sequential(
                Time2Vec(config.time2vec_dim),
                nn.Linear(config.time2vec_dim, config.d_model),
            )


            self.visit_process_proj = nn.Linear(3, config.d_model)
        else:
            raise ValueError(f"Unknown time encoding: {config.time_encoding}")
        self.dwell_proj = (
            nn.Linear(1, config.d_model, bias=False) if config.use_dwell else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,


            enable_nested_tensor=config.context == 'full',
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.state_head = nn.Linear(config.d_model, config.num_states)
        if config.covariate_parameterization not in {"absolute", "residual"}:
            raise ValueError("covariate_parameterization must be 'absolute' or 'residual'")
        joint_width = 2 * config.d_model
        self.covariate_trunk = nn.Sequential(
            nn.Linear(joint_width, config.d_model),
            nn.SiLU(),
            nn.LayerNorm(config.d_model),
        )
        self.covariate_location_head = OptionalLinear(
            config.d_model, config.num_dynamic_continuous
        )
        self.covariate_scale_head = OptionalLinear(
            config.d_model, config.num_dynamic_continuous
        )
        self.observation_head = OptionalLinear(
            config.d_model, config.num_observation_processes
        )
        self.binary_head = OptionalLinear(config.d_model, config.num_dynamic_binary)

    def embed_inputs(
        self,
        states: torch.Tensor,
        covariates: torch.Tensor,
        times: torch.Tensor,
        categories: torch.Tensor | None = None,
        dwell: torch.Tensor | None = None,
        elapsed_times: torch.Tensor | None = None,
        visit_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        'Embed tokens without applying the causal transformer.'
        if self.config.time_encoding == "discrete_absolute":
            assert self.time_emb is not None
            time_token = self.time_emb(times.clamp(0, self.max_time))
        else:
            assert self.time2vec is not None and self.visit_process_proj is not None
            if elapsed_times is None:
                elapsed_times = torch.zeros_like(times, dtype=covariates.dtype)
                if times.shape[1] > 1:
                    elapsed_times[:, 1:] = (
                        times[:, 1:] - times[:, :-1]
                    ).clamp_min(0).to(covariates.dtype)
            else:
                elapsed_times = elapsed_times.to(covariates.dtype)
            if visit_indices is None:
                visit_indices = torch.arange(
                    times.shape[1], device=times.device, dtype=covariates.dtype
                ).unsqueeze(0).expand_as(elapsed_times)
            else:
                visit_indices = visit_indices.to(covariates.dtype)
            first = (visit_indices == 0).to(covariates.dtype)
            process = torch.stack(
                [torch.log1p(elapsed_times.clamp_min(0.0)), first, visit_indices],
                dim=-1,
            )
            time_token = self.time2vec(elapsed_times) + self.visit_process_proj(process)
        x = self.state_emb(states) + self.cov_proj(covariates) + time_token
        if self.dwell_proj is not None:
            if dwell is None or dwell.shape != states.shape:
                raise ValueError("Dwell-aware model requires one dwell value per state token")
            scaled_dwell = torch.log1p(dwell.clamp_min(0.0)) / torch.log1p(
                dwell.new_tensor(float(max(self.max_time, 1)))
            )
            x = x + self.dwell_proj(scaled_dwell.unsqueeze(-1))
        if self.category_embeddings:
            if categories is None or categories.shape[-1] != len(self.category_embeddings):
                raise ValueError("Categorical inputs do not match the fitted feature schema")
            category_sum = torch.zeros_like(x)
            for index, embedding in enumerate(self.category_embeddings):
                category_sum = category_sum + embedding(categories[..., index])
            x = x + category_sum / (len(self.category_embeddings) ** 0.5)
        return x

    def encode(
        self,
        states: torch.Tensor,
        covariates: torch.Tensor,
        times: torch.Tensor,
        valid_mask: torch.Tensor | None,
        categories: torch.Tensor | None = None,
        dwell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_inputs(states, covariates, times, categories, dwell)
        mask = context_attention_mask(x.shape[1], self.config.context, x.device, x.dtype)
        if self.config.boolean_causal_mask:
            mask = torch.isneginf(mask)


        padding = (
            ~valid_mask
            if valid_mask is not None and self.config.context == 'full'
            else None
        )
        if padding is not None and not self.config.boolean_causal_mask:
            padding_float = torch.zeros_like(padding, dtype=x.dtype)
            padding = padding_float.masked_fill(padding, float("-inf"))
        return self.norm(self.encoder(x, mask=mask, src_key_padding_mask=padding))

    @staticmethod
    def _project_key_value(
        layer: nn.TransformerEncoderLayer, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        'Project layer inputs into per-head key/value tensors.'
        attention = layer.self_attn
        if not attention._qkv_same_embed_dim or attention.in_proj_weight is None:
            raise RuntimeError("Incremental inference requires standard self-attention projections")
        dimension = attention.embed_dim
        weight = attention.in_proj_weight
        bias = attention.in_proj_bias
        key = F.linear(
            inputs,
            weight[dimension : 2 * dimension],
            None if bias is None else bias[dimension : 2 * dimension],
        )
        value = F.linear(
            inputs,
            weight[2 * dimension :],
            None if bias is None else bias[2 * dimension :],
        )
        batch, length, _ = key.shape
        heads = attention.num_heads
        head_dimension = dimension // heads
        key = key.view(batch, length, heads, head_dimension).transpose(1, 2)
        value = value.view(batch, length, heads, head_dimension).transpose(1, 2)
        return key, value

    @staticmethod
    def repeat_incremental_cache(
        cache: list[tuple[torch.Tensor, torch.Tensor]], repeats: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        return [
            (key.repeat_interleave(repeats, dim=0), value.repeat_interleave(repeats, dim=0))
            for key, value in cache
        ]

    def initialize_incremental(
        self,
        states: torch.Tensor,
        covariates: torch.Tensor,
        times: torch.Tensor,
        categories: torch.Tensor | None = None,
        dwell: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        'Encode an unpadded observed history and cache projected K/V tensors.'
        incremental_context = self.config.context == "full" or self.config.context.startswith(
            "window"
        )
        if not incremental_context:
            raise ValueError(
                "Incremental inference requires full-history or fixed-window attention"
            )
        if states.ndim != 2 or states.shape[1] == 0:
            raise ValueError("Incremental histories must be non-empty [batch,time] tensors")
        x = self.embed_inputs(
            states, covariates, times, categories=categories, dwell=dwell
        )
        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        length = x.shape[1]
        mask = context_attention_mask(
            length, self.config.context, x.device, x.dtype
        )
        if self.config.boolean_causal_mask:
            mask = torch.isneginf(mask)
        for layer in self.encoder.layers:
            cache.append(self._project_key_value(layer, x))
            x = layer(x, src_mask=mask)
        if self.config.context.startswith("window"):
            width = int(self.config.context.removeprefix("window"))
            keep = max(width - 1, 0)
            cache = [
                (
                    key[:, :, -keep:, :] if keep else key[:, :, :0, :],
                    value[:, :, -keep:, :] if keep else value[:, :, :0, :],
                )
                for key, value in cache
            ]
        return self.norm(x[:, -1, :]), cache

    def incremental_step(
        self,
        states: torch.Tensor,
        covariates: torch.Tensor,
        times: torch.Tensor,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        categories: torch.Tensor | None = None,
        dwell: torch.Tensor | None = None,
        elapsed_times: torch.Tensor | None = None,
        visit_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        'Append one token using an exact projected-key/value causal cache.'
        if states.ndim != 2 or states.shape[1] != 1:
            raise ValueError("incremental_step appends exactly one token")
        if len(cache) != len(self.encoder.layers):
            raise ValueError("Incremental cache does not match the transformer depth")
        current = self.embed_inputs(
            states,
            covariates,
            times,
            categories=categories,
            dwell=dwell,
            elapsed_times=elapsed_times,
            visit_indices=visit_indices,
        )
        updated: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, (key_history, value_history) in zip(self.encoder.layers, cache):
            attention = layer.self_attn
            if layer.norm_first:
                raise RuntimeError("Incremental inference currently requires post-norm layers")
            dimension = attention.embed_dim
            weight = attention.in_proj_weight
            bias = attention.in_proj_bias
            assert weight is not None
            query = F.linear(
                current,
                weight[:dimension],
                None if bias is None else bias[:dimension],
            )
            key_new, value_new = self._project_key_value(layer, current)
            batch = current.shape[0]
            heads = attention.num_heads
            head_dimension = dimension // heads
            query = query.view(batch, 1, heads, head_dimension).transpose(1, 2)
            if self.config.context.startswith("window"):
                width = int(self.config.context.removeprefix("window"))
                keep = max(width - 1, 0)
                key_history = (
                    key_history[:, :, -keep:, :]
                    if keep
                    else key_history[:, :, :0, :]
                )
                value_history = (
                    value_history[:, :, -keep:, :]
                    if keep
                    else value_history[:, :, :0, :]
                )
            elif self.config.context != "full":
                raise ValueError(
                    "Incremental inference requires full-history or fixed-window attention"
                )
            key = torch.cat([key_history, key_new], dim=2)
            value = torch.cat([value_history, value_new], dim=2)
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=attention.dropout if self.training else 0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).contiguous().view(batch, 1, dimension)
            attended = attention.out_proj(attended)
            current = layer.norm1(current + layer.dropout1(attended))
            feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(current))))
            current = layer.norm2(current + layer.dropout2(feedforward))
            updated.append((key, value))
        return self.norm(current[:, 0, :]), updated

    def covariate_outputs(
        self,
        hidden: torch.Tensor,
        next_states: torch.Tensor,
        current_continuous: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        'Parameters of p(M,X | history, just-realized next state).'
        if hidden.shape[:-1] != next_states.shape:
            raise ValueError("Next-state conditioning shape does not match hidden states")
        state_condition = self.state_emb(
            next_states.clamp(0, self.num_states - 1)
        )
        if not self.config.condition_covariates_on_next_state:


            state_condition = torch.zeros_like(state_condition)
        conditioned = self.covariate_trunk(
            torch.cat([hidden, state_condition], dim=-1)
        )
        location = self.covariate_location_head(conditioned)
        if self.config.covariate_parameterization == "residual":
            if current_continuous is None or current_continuous.shape != location.shape:
                raise ValueError("Residual covariate head requires aligned current values")
            location = current_continuous + location
        raw_scale = self.covariate_scale_head(conditioned).clamp(-8.0, 6.0)
        scale = F.softplus(raw_scale) + 1e-3
        return {
            "covariate_location": location,
            "covariate_log_scale": torch.log(scale),
            "observation_logits": self.observation_head(conditioned),
            "binary_logits": self.binary_head(conditioned),
        }

    def forward(
        self,
        states: torch.Tensor,
        covariates: torch.Tensor,
        times: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        categories: torch.Tensor | None = None,
        dwell: torch.Tensor | None = None,
        next_states: torch.Tensor | None = None,
        current_continuous: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        hidden = self.encode(states, covariates, times, valid_mask, categories, dwell)
        joint = (
            self.covariate_outputs(hidden, next_states, current_continuous)
            if next_states is not None
            else {
                "covariate_location": None,
                "covariate_log_scale": None,
                "observation_logits": None,
                "binary_logits": None,
            }
        )
        return {
            "state_logits": self.state_head(hidden),
            "hidden": hidden,
            **joint,
        }
