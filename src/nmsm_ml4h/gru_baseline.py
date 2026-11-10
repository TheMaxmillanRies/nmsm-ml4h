from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import FeatureEncoder, TrajectoryDataset, add_history_summaries, collate_trajectories
from .losses import LossWeights, compute_loss
from .model import OptionalLinear, allowed_transitions
from .rollout import internal_frame, sample_joint_covariates, validate_probability
from .splits import split_train_validation
from .targets import KEYS
from .training import fit_feature_encoder, move_batch, record_feature_encoder, set_seed


@dataclass(frozen=True)
class GRUConfig:
    num_states: int
    num_features: int
    categorical_cardinalities: tuple[int, ...]
    hidden_size: int
    layers: int
    dropout: float
    max_time: int
    num_dynamic_continuous: int
    num_observation_processes: int
    num_dynamic_binary: int
    covariate_parameterization: str


class AutoregressiveGRU(nn.Module):
    'Matched recurrent baseline with the same joint next-token factorization.'

    def __init__(self, config: GRUConfig) -> None:
        super().__init__()
        if config.covariate_parameterization not in {"absolute", "residual"}:
            raise ValueError("Unknown covariate parameterization")
        self.config = config
        self.num_states = config.num_states
        self.state_embedding = nn.Embedding(config.num_states, config.hidden_size)
        self.feature_projection = nn.Linear(config.num_features, config.hidden_size)
        self.time_embedding = nn.Embedding(config.max_time + 1, config.hidden_size)
        self.dwell_projection = nn.Linear(1, config.hidden_size, bias=False)
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, config.hidden_size)
            for cardinality in config.categorical_cardinalities
        )
        self.gru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            num_layers=config.layers,
            dropout=config.dropout if config.layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.state_head = nn.Linear(config.hidden_size, config.num_states)
        self.covariate_trunk = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.covariate_location_head = OptionalLinear(
            config.hidden_size, config.num_dynamic_continuous
        )
        self.covariate_scale_head = OptionalLinear(
            config.hidden_size, config.num_dynamic_continuous
        )
        self.observation_head = OptionalLinear(
            config.hidden_size, config.num_observation_processes
        )
        self.binary_head = OptionalLinear(config.hidden_size, config.num_dynamic_binary)

    def embed(
        self,
        states: torch.Tensor,
        values: torch.Tensor,
        times: torch.Tensor,
        categories: torch.Tensor | None,
        dwell: torch.Tensor,
    ) -> torch.Tensor:
        scaled_dwell = torch.log1p(dwell.clamp_min(0.0)) / torch.log1p(
            dwell.new_tensor(float(max(self.config.max_time, 1)))
        )
        embedded = (
            self.state_embedding(states)
            + self.feature_projection(values)
            + self.time_embedding(times.clamp(0, self.config.max_time))
            + self.dwell_projection(scaled_dwell.unsqueeze(-1))
        )
        if self.category_embeddings:
            if categories is None or categories.shape[-1] != len(self.category_embeddings):
                raise ValueError("Categorical inputs do not match the GRU schema")
            category_sum = torch.zeros_like(embedded)
            for index, embedding in enumerate(self.category_embeddings):
                category_sum += embedding(categories[..., index])
            embedded += category_sum / (len(self.category_embeddings) ** 0.5)
        return embedded

    def encode(
        self,
        states: torch.Tensor,
        values: torch.Tensor,
        times: torch.Tensor,
        categories: torch.Tensor | None,
        dwell: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, recurrent_state = self.gru(
            self.embed(states, values, times, categories, dwell), recurrent_state
        )
        return self.norm(output), recurrent_state

    def covariate_outputs(
        self,
        hidden: torch.Tensor,
        next_states: torch.Tensor,
        current_continuous: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden.shape[:-1] != next_states.shape:
            raise ValueError("Next-state conditioning does not match GRU hidden states")
        conditioned = self.covariate_trunk(
            torch.cat([hidden, self.state_embedding(next_states)], dim=-1)
        )
        location = self.covariate_location_head(conditioned)
        if self.config.covariate_parameterization == "residual":
            if current_continuous is None or current_continuous.shape != location.shape:
                raise ValueError("Residual GRU head requires current continuous values")
            location = current_continuous + location
        scale = F.softplus(self.covariate_scale_head(conditioned).clamp(-8.0, 6.0)) + 1e-3
        return {
            "covariate_location": location,
            "covariate_log_scale": torch.log(scale),
            "observation_logits": self.observation_head(conditioned),
            "binary_logits": self.binary_head(conditioned),
        }

    def forward(
        self,
        states: torch.Tensor,
        values: torch.Tensor,
        times: torch.Tensor,
        categories: torch.Tensor | None,
        dwell: torch.Tensor,
        next_states: torch.Tensor | None = None,
        current_continuous: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        hidden, recurrent_state = self.encode(
            states, values, times, categories, dwell, recurrent_state
        )
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
        return {"state_logits": self.state_head(hidden), "hidden": hidden, **joint}, recurrent_state


def _forward_batch(model: AutoregressiveGRU, batch: dict) -> dict:
    categories = batch["input_categories"]
    outputs, _ = model(
        batch["input_states"],
        batch["input_values"],
        batch["input_times"],
        categories if categories.shape[-1] else None,
        batch["input_dwell"],
        batch["target_states"].clamp_min(0),
        batch["current_continuous_values"],
    )
    return outputs


def _objective_parts(
    model: AutoregressiveGRU,
    batch: dict,
    graph: torch.Tensor,
    weights: LossWeights,
    objective: str,
) -> dict[str, torch.Tensor]:
    'Use the same faithful mean/scale separation as the primary nMSM.'
    if objective not in {"nll", "mean_focused"}:
        raise ValueError("Unknown GRU objective")
    outputs = _forward_batch(model, batch)
    parts = compute_loss(outputs, batch, graph, weights)
    residual = batch["target_continuous_values"] - outputs["covariate_location"]
    valid = batch["continuous_value_mask"]
    denominator = valid.sum().clamp_min(1)
    mean_mse = torch.where(valid, residual.square(), 0.0).sum() / denominator
    with torch.no_grad():
        condition = model.state_embedding(
            batch["target_states"].clamp(0, model.num_states - 1)
        )
        trunk = model.covariate_trunk(
            torch.cat((outputs["hidden"].detach(), condition), dim=-1)
        )
    sigma = F.softplus(model.covariate_scale_head(trunk).clamp(-8.0, 6.0)) + 1e-3
    scale_nll = (
        0.5 * (residual.detach() / sigma).square()
        + sigma.log()
        + 0.5 * math.log(2.0 * math.pi)
    )
    scale_nll = torch.where(valid, scale_nll, 0.0).sum() / denominator
    parts["mean_mse"] = mean_mse
    parts["scale_nll"] = scale_nll
    parts["original_total"] = parts["total"]
    if objective == "mean_focused":
        parts["total"] = parts["state"] + weights.covariates * (
            mean_mse + scale_nll + parts["observations"] + parts["binary_covariates"]
        )
    return parts


@torch.no_grad()
def _validation(
    model: AutoregressiveGRU,
    loader: DataLoader,
    graph: torch.Tensor,
    weights: LossWeights,
    device: torch.device,
    objective: str = "nll",
) -> dict[str, float]:
    model.eval()
    numerators = {"total": 0.0, "state": 0.0, "mean_mse": 0.0, "scale_nll": 0.0,
                  "observations": 0.0, "binary_covariates": 0.0}
    denominator = 0
    for raw in loader:
        batch = move_batch(raw, device)
        count = int(batch["next_mask"].sum().item())
        if count:
            parts = _objective_parts(model, batch, graph, weights, objective)
            for key in numerators:
                numerators[key] += float(parts[key].item()) * count
            denominator += count
    if not denominator:
        raise RuntimeError("GRU validation split has no transition targets")
    values = {key: value / denominator for key, value in numerators.items()}
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError("Non-finite GRU validation objective")
    return values


def train_gru_fold(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
    output_dir: Path,
    epochs: int,
    seed: int,
    patience: int = 8,
    device: torch.device | None = None,
    objective: str = "nll",
    split_seed: int | None = None,
) -> tuple[AutoregressiveGRU, FeatureEncoder]:
    set_seed(seed + fold)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_training = folds[folds["outer_fold"] != fold]
    split_seed = seed if split_seed is None else split_seed
    train_ids, validation_ids = split_train_validation(outer_training, seed=split_seed + fold)
    train = frame[frame["id"].isin(train_ids)].copy()
    validation = frame[frame["id"].isin(validation_ids)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = fit_feature_encoder(train, experiment)
    record_feature_encoder(encoder, output_dir, f"{experiment.name}/gru/fold_{fold}", train)
    train_data = TrajectoryDataset(
        train,
        experiment.feature_cols,
        experiment.num_states,
        simulation_state_map=experiment.name == "simulation",
        feature_encoder=encoder,
    )
    validation_data = TrajectoryDataset(
        validation,
        experiment.feature_cols,
        experiment.num_states,
        simulation_state_map=experiment.name == "simulation",
        feature_encoder=encoder,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=experiment.batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=experiment.batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
        num_workers=0,
    )
    architecture = GRUConfig(
        experiment.num_states,
        len(encoder.continuous_feature_names),
        encoder.categorical_cardinalities,
        min(experiment.d_model, 256),
        2,
        experiment.dropout,
        experiment.max_time,
        len(encoder.dynamic_continuous_columns),
        len(encoder.observation_numeric_columns),
        len(encoder.binary_dynamic_columns),
        experiment.covariate_parameterization,
    )
    model = AutoregressiveGRU(architecture).to(device)
    graph = allowed_transitions(
        experiment.num_states, experiment.terminal_states, experiment.allowed_edges
    ).to(device)
    weights = LossWeights(experiment.lambda_cov)
    if objective not in {"nll", "mean_focused"}:
        raise ValueError("Unknown GRU objective")
    optimizer_class = torch.optim.Adam if objective == "mean_focused" else torch.optim.AdamW
    optimizer = optimizer_class(
        model.parameters(),
        lr=experiment.learning_rate,
        weight_decay=experiment.weight_decay,
    )
    best = float("inf")
    stale = 0
    history: list[dict] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        gradient_norms: list[float] = []
        for raw in train_loader:
            batch = move_batch(raw, device)
            parts = _objective_parts(model, batch, graph, weights, objective)
            if not torch.isfinite(parts["total"]):
                raise FloatingPointError("Non-finite GRU training objective")
            optimizer.zero_grad(set_to_none=True)
            parts["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite GRU gradient")
            optimizer.step()
            if any(not torch.isfinite(value).all() for value in model.parameters()):
                raise FloatingPointError("Non-finite GRU parameter")
            losses.append(float(parts["total"].item()))
            gradient_norms.append(float(gradient_norm.item()))
        validation = _validation(model, validation_loader, graph, weights, device, objective)
        row = {
            "epoch": epoch,
            "train_total": float(np.mean(losses)),
            "train_gradient_norm": float(np.mean(gradient_norms)),
            "validation_total": validation["total"],
            "validation_state": validation["state"],
            "validation_mean_rmse": float(np.sqrt(validation["mean_mse"])),
        }
        history.append(row)
        selection_value = validation["state"] if objective == "mean_focused" else validation["total"]
        if selection_value < best - 1e-5:
            best = selection_value
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": asdict(architecture),
                    "feature_encoder": encoder.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "validation": validation,
                    "selection_value": selection_value,
                    "objective": objective,
                    "config": asdict(experiment),
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        print(
            f"gru fold={fold} epoch={epoch}/{epochs} train={row['train_total']:.6f} "
            f"val_state={validation['state']:.6f} val_total={validation['total']:.6f} "
            f"grad={row['train_gradient_norm']:.4f} stale={stale}",
            flush=True,
        )
        if stale >= patience:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    (output_dir / "training.json").write_text(
        json.dumps(
            {
                "best_epoch": checkpoint["epoch"],
                "best_validation": checkpoint["validation"],
                "objective": objective,
                "checkpoint_metric": "state" if objective == "mean_focused" else "total",
                "lambda_cov": experiment.lambda_cov,
                "covariate_parameterization": experiment.covariate_parameterization,
                "epochs_run": len(history),
                "maximum_epochs": epochs,
                "early_stopping_patience": patience,
                "stopped_by_patience_before_epoch_cap": len(history) < epochs,
                "best_at_training_boundary": int(checkpoint["epoch"]) == len(history),
                "runtime_seconds": time.time() - started,
                "history": history,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return model, encoder


@torch.no_grad()
def predict_gru_fold(
    model: AutoregressiveGRU,
    encoder: FeatureEncoder,
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    experiment: ExperimentConfig,
    simulations: int,
    seed: int,
    simulation_batch_size: int = 32,
    patient_batch_size: int = 2,
    gpu_yield_seconds: float = 0.0,
    max_concurrent_paths: int = 128,
    covariate_mode: str = "joint",
    state_temperature: float = 1.0,
    progress_every: int = 0,
) -> pd.DataFrame:
    if simulations <= 0 or simulation_batch_size <= 0 or patient_batch_size <= 0:
        raise ValueError("Simulation counts must be positive")
    if patient_batch_size * simulation_batch_size > max_concurrent_paths:
        raise ValueError("Unsafe GRU rollout concurrency")
    if covariate_mode not in {"joint", "locf"}:
        raise ValueError("covariate_mode must be 'joint' or 'locf'")
    if not np.isfinite(state_temperature) or state_temperature <= 0:
        raise ValueError("state_temperature must be finite and positive")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    graph = allowed_transitions(
        experiment.num_states, experiment.terminal_states, experiment.allowed_edges
    ).to(device)
    terminals = frozenset(experiment.terminal_states)
    raw = internal_frame(frame[frame["id"].isin(targets["id"].unique())], experiment.name == "simulation")
    raw = add_history_summaries(raw).reset_index(drop=True)
    all_values, all_categories = encoder.encode_many(raw)
    groups = {int(pid): group for pid, group in raw.groupby("id", sort=False)}
    encoded = {pid: (group, all_values[group.index], all_categories[group.index])
               for pid, group in groups.items()}
    pairs = targets[["id", "landmark", "fold"]].drop_duplicates().sort_values(["landmark", "id"])
    rows: list[dict] = []
    model.eval()
    max_simplex_error = 0.0
    max_continuous_abs = 0.0
    max_scale = 0.0
    rollout_started = time.monotonic()
    finished_pairs = 0
    next_progress = progress_every
    for landmark, landmark_pairs in pairs.groupby("landmark", sort=True):
        landmark = int(landmark)
        records = list(landmark_pairs.itertuples(index=False))
        for start in range(0, len(records), patient_batch_size):
            selected = records[start : start + patient_batch_size]
            histories = [
                groups[int(record.id)][groups[int(record.id)]["t"] <= landmark]
                for record in selected
            ]
            if len({len(history) for history in histories}) != 1:
                raise ValueError("Batched GRU rollout requires aligned observed grids")
            state_np = np.stack([history["state"].to_numpy(np.int64) for history in histories])
            time_np = np.stack([history["t"].to_numpy(np.int64) for history in histories])
            dwell_np = np.stack([history["time_in_state"].to_numpy(np.float32) for history in histories])
            value_np = np.stack([
                encoded[int(record.id)][1][encoded[int(record.id)][0]["t"].to_numpy(int) <= landmark]
                for record in selected
            ])
            category_np = np.stack([
                encoded[int(record.id)][2][encoded[int(record.id)][0]["t"].to_numpy(int) <= landmark]
                for record in selected
            ])
            states = torch.as_tensor(state_np, dtype=torch.long, device=device)
            times = torch.as_tensor(time_np, dtype=torch.long, device=device)
            dwells = torch.as_tensor(dwell_np, dtype=torch.float32, device=device)
            values = torch.as_tensor(value_np, dtype=torch.float32, device=device)
            categories = torch.as_tensor(category_np, dtype=torch.long, device=device)
            hidden_history, recurrent = model.encode(
                states,
                values,
                times,
                categories if categories.shape[-1] else None,
                dwells,
            )
            base_hidden = hidden_history[:, -1, :]
            origin = states[:, -1]
            base_values = values[:, -1, :]
            base_categories = categories[:, -1, :]
            base_dwell = dwells[:, -1]
            batch = len(selected)
            sums = torch.zeros(
                (batch, experiment.max_horizon, experiment.num_states), dtype=torch.float64
            )
            completed = 0
            while completed < simulations:
                chunk = min(simulation_batch_size, simulations - completed)
                current_hidden = base_hidden.repeat_interleave(chunk, dim=0)
                current_recurrent = recurrent.repeat_interleave(chunk, dim=1)
                current_state = origin.repeat_interleave(chunk)
                current_values = base_values.repeat_interleave(chunk, dim=0)
                current_categories = base_categories.repeat_interleave(chunk, dim=0)
                current_dwell = base_dwell.repeat_interleave(chunk)
                for horizon in range(1, experiment.max_horizon + 1):
                    probability = torch.softmax(
                        model.state_head(current_hidden).masked_fill(~graph[current_state], -1e9)
                        / state_temperature,
                        dim=-1,
                    )
                    max_simplex_error = max(
                        max_simplex_error,
                        validate_probability(probability, "GRU transition probability"),
                    )
                    sums[:, horizon - 1] += (
                        probability.view(batch, chunk, experiment.num_states)
                        .sum(dim=1)
                        .to(dtype=torch.float64, device="cpu")
                    )
                    next_state = torch.multinomial(probability, 1, generator=generator).squeeze(1)
                    if horizon == experiment.max_horizon:
                        continue
                    if covariate_mode == "locf":
                        next_values = encoder.advance_locf(current_values)
                    else:
                        next_values, diagnostic = sample_joint_covariates(
                            model,
                            encoder,
                            current_hidden,
                            current_values,
                            next_state,
                            terminals,
                            generator,
                        )
                        max_continuous_abs = max(max_continuous_abs, diagnostic["continuous_abs_max"])
                        max_scale = max(max_scale, diagnostic["scale_max"])
                    next_dwell = torch.where(
                        next_state == current_state,
                        current_dwell + 1.0,
                        torch.zeros_like(current_dwell),
                    )
                    next_time = torch.full(
                        (batch * chunk, 1), landmark + horizon, dtype=torch.long, device=device
                    )
                    output, current_recurrent = model(
                        next_state[:, None],
                        next_values[:, None, :],
                        next_time,
                        current_categories[:, None, :] if current_categories.shape[-1] else None,
                        next_dwell[:, None],
                        recurrent_state=current_recurrent,
                    )
                    current_hidden = output["hidden"][:, -1, :]
                    current_state = next_state
                    current_values = next_values
                    current_dwell = next_dwell
                completed += chunk
                if gpu_yield_seconds:
                    time.sleep(gpu_yield_seconds)
            probabilities = sums.numpy() / simulations
            validate_probability(torch.as_tensor(probabilities), "GRU occupation probability")
            for patient_index, record in enumerate(selected):
                for horizon in range(1, experiment.max_horizon + 1):
                    row = {
                        "id": int(record.id),
                        "fold": int(record.fold),
                        "landmark": landmark,
                        "horizon": horizon,
                        "variant": "gru",
                        "n_sims": simulations,
                        "covariate_mode": covariate_mode,
                        "state_temperature": float(state_temperature),
                    }
                    row.update(
                        {
                            f"p_state_{state}": float(probabilities[patient_index, horizon - 1, state])
                            for state in range(experiment.num_states)
                        }
                    )
                    rows.append(row)
            finished_pairs += len(selected)
            if progress_every and (finished_pairs >= next_progress or finished_pairs == len(pairs)):
                elapsed = time.monotonic() - rollout_started
                print(f"GRU_PROGRESS pairs={finished_pairs}/{len(pairs)} sims={simulations} "
                      f"elapsed_seconds={elapsed:.1f} pairs_per_second={finished_pairs/max(elapsed,1e-9):.3f}", flush=True)
                next_progress = finished_pairs + progress_every
    result = pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
    print(
        f"GRU ROLLOUT PASS rows={len(result)} mode={covariate_mode} "
        f"simplex_error={max_simplex_error:.3g} max_abs_covariate={max_continuous_abs:.3g} "
        f"max_scale={max_scale:.3g}",
        flush=True,
    )
    return result
