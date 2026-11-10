from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .baseline_common import (
    baseline_feature_indices,
    destination_map,
    encode_frame,
    fit_baseline_encoder,
    nonself_edges,
)
from .config import ExperimentConfig
from .idnetwork_general import _spells
from .splits import split_train_validation
from .targets import KEYS
from .training import set_seed


class Encoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        latent_features: int = 20,
        hidden_features: int = 800,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, latent_features),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ODEFunction(nn.Module):
    def __init__(
        self,
        num_states: int,
        edges: tuple[tuple[int, int], ...],
        input_features: int,
        latent_features: int = 20,
        hidden_features: int = 1000,
    ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_probabilities = num_states * num_states
        self.num_edges = len(edges)
        self.latent_features = latent_features
        self.register_buffer("edge_origin", torch.tensor([edge[0] for edge in edges]))
        self.register_buffer("edge_destination", torch.tensor([edge[1] for edge in edges]))
        input_size = (
            2 * self.num_probabilities
            + self.num_edges
            + 2 * latent_features
            + 1
        )
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, self.num_edges + latent_features),
        )
        self.initial_latent: torch.Tensor | None = None
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.zeros_(module.bias)

    def set_context(self, initial_latent: torch.Tensor) -> None:
        self.initial_latent = initial_latent

    def generator(self, edge_rates: torch.Tensor) -> torch.Tensor:
        batch = len(edge_rates)
        generator = edge_rates.new_zeros(batch, self.num_states, self.num_states)
        generator[:, self.edge_origin, self.edge_destination] = edge_rates
        diagonal = -generator.sum(dim=2)
        index = torch.arange(self.num_states, device=edge_rates.device)
        generator[:, index, index] = diagonal
        return generator

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.initial_latent is None:
            raise RuntimeError("SurvNODE ODE context has not been set")
        batch = len(state)
        time_column = t.reshape(1, 1).expand(batch, 1)
        output = self.network(
            torch.cat(
                [state, self.initial_latent, time_column], dim=1
            )
        )
        edge_rates = torch.nn.functional.softplus(output[:, : self.num_edges])
        generator = self.generator(edge_rates)
        forward_start = 0
        backward_start = self.num_probabilities
        cumulative_start = 2 * self.num_probabilities
        forward = state[:, forward_start:backward_start].reshape(
            batch, self.num_states, self.num_states
        )
        backward = state[:, backward_start:cumulative_start].reshape(
            batch, self.num_states, self.num_states
        )
        forward_derivative = torch.bmm(forward, generator)
        backward_derivative = -torch.bmm(generator, backward)
        return torch.cat(
            [
                forward_derivative.flatten(1),
                backward_derivative.flatten(1),
                edge_rates,
                output[:, self.num_edges :],
            ],
            dim=1,
        )


class SurvNODE(nn.Module):
    def __init__(
        self,
        num_states: int,
        edges: tuple[tuple[int, int], ...],
        input_features: int,
        latent_features: int = 20,
        encoder_hidden_features: int = 800,
        ode_hidden_features: int = 1000,
        solver_method: str = "dopri5",
        solver_rtol: float = 1e-8,
        solver_atol: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_probabilities = num_states * num_states
        self.edges = edges
        self.solver_method = solver_method
        self.solver_rtol = solver_rtol
        self.solver_atol = solver_atol
        self.encoder = Encoder(
            input_features, latent_features, encoder_hidden_features
        )
        self.ode_function = ODEFunction(
            num_states,
            edges,
            input_features,
            latent_features,
            ode_hidden_features,
        )
        edge_lookup = torch.full((num_states, num_states), -1, dtype=torch.long)
        for index, (origin, destination) in enumerate(edges):
            edge_lookup[origin, destination] = index
        self.register_buffer("edge_lookup", edge_lookup)

    def _integrate(
        self, features: torch.Tensor, requested_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from torchdiffeq import odeint
        except ImportError as error:
            raise ImportError(
                "SurvNODE requires torchdiffeq==0.2.5; install the pinned "
                "requirements before running this baseline"
            ) from error
        latent = self.encoder(features)
        batch = len(features)
        identity = torch.eye(
            self.num_states, device=features.device, dtype=features.dtype
        ).flatten().expand(batch, -1)
        cumulative = features.new_zeros(batch, len(self.edges))
        initial = torch.cat([identity, identity, cumulative, latent], dim=1)
        self.ode_function.set_context(latent)
        trajectory = odeint(
            self.ode_function,
            initial,
            requested_times,
            method=self.solver_method,
            rtol=self.solver_rtol,
            atol=self.solver_atol,
        )
        return trajectory, latent

    def interval_likelihood(
        self,
        features: torch.Tensor,
        start: torch.Tensor,
        stop: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        event: torch.Tensor,
        *,
        integration_times: torch.Tensor | None = None,
        regularizer_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if integration_times is None:
            zero = features.new_zeros(1)
            times = torch.unique(torch.cat([zero, start, stop]), sorted=True)
        else:
            times = integration_times
        trajectory, latent = self._integrate(features, times)
        start_index = torch.searchsorted(times, start)
        stop_index = torch.searchsorted(times, stop)
        batch_index = torch.arange(len(features), device=features.device)
        cumulative_start = 2 * self.num_probabilities
        cumulative_stop = cumulative_start + len(self.edges)
        cumulative = trajectory[:, :, cumulative_start:cumulative_stop]
        increment = (
            cumulative[stop_index, batch_index]
            - cumulative[start_index, batch_index]
        )
        origin_mask = (
            self.ode_function.edge_origin[None, :] == origin[:, None]
        ).to(features.dtype)
        survival = torch.exp(-(increment * origin_mask).sum(dim=1))
        state_at_stop = trajectory[stop_index, batch_index]
        network_input = torch.cat(
            [
                state_at_stop,
                latent,
                stop[:, None],
            ],
            dim=1,
        )
        network_output = self.ode_function.network(network_input)
        rates = torch.nn.functional.softplus(network_output[:, : len(self.edges)])
        edge_index = self.edge_lookup[origin, destination].clamp_min(0)
        hazard = rates[batch_index, edge_index]
        final_time = stop.max() if regularizer_time is None else regularizer_time
        final_index = int(torch.searchsorted(times, final_time))
        final_state = trajectory[final_index]
        final_output = self.ode_function.network(
            torch.cat(
                [final_state, latent, final_time.reshape(1, 1).expand(len(features), 1)],
                dim=1,
            )
        )
        final_rates = torch.nn.functional.softplus(
            final_output[:, : len(self.edges)]
        )
        regularizer_output = torch.cat(
            [final_rates, final_output[:, len(self.edges) :]], dim=1
        )
        return survival, hazard, regularizer_output

    @torch.no_grad()
    def occupation(
        self,
        features: torch.Tensor,
        landmark: float,
        horizons: torch.Tensor,
        origin: torch.Tensor,
    ) -> torch.Tensor:
        requested = torch.unique(
            torch.cat([features.new_tensor([0.0, landmark]), horizons]), sorted=True
        )
        trajectory, _ = self._integrate(features, requested)
        landmark_index = int(torch.searchsorted(requested, features.new_tensor(landmark)))
        backward = trajectory[
            landmark_index,
            :,
            self.num_probabilities : 2 * self.num_probabilities,
        ].reshape(len(features), self.num_states, self.num_states)
        outputs = []
        batch_index = torch.arange(len(features), device=features.device)
        for horizon in horizons:
            index = int(torch.searchsorted(requested, horizon))
            forward = trajectory[
                index, :, : self.num_probabilities
            ].reshape(len(features), self.num_states, self.num_states)
            conditional = torch.bmm(backward, forward)
            probability = conditional[batch_index, origin]
            if not torch.isfinite(probability).all():
                raise FloatingPointError("SurvNODE occupation contains non-finite values")
            minimum = float(probability.min().item())
            if minimum < -1e-8:
                raise FloatingPointError(
                    f"SurvNODE ODE produced materially negative occupation mass: {minimum}"
                )
            probability = torch.where(
                probability < 0, torch.zeros_like(probability), probability
            )
            row_sum = probability.sum(dim=1, keepdim=True)
            row_error = float(torch.max(torch.abs(row_sum - 1.0)).item())


            if torch.any(row_sum <= 0) or row_error > 1e-3:
                raise FloatingPointError(
                    "SurvNODE occupation does not conserve probability within ODE "
                    f"tolerance; max_row_sum_error={row_error:.3e}"
                )
            probability = probability / row_sum
            outputs.append(probability)
        return torch.stack(outputs, dim=1)


def _step_dataset(
    encoded,
    patient_ids: set[int],
    experiment: ExperimentConfig,
    edges: tuple[tuple[int, int], ...],
) -> TensorDataset:


    spells = _spells(encoded, patient_ids, destination_map(experiment))
    if spells.empty:
        return TensorDataset(
            torch.empty((0, encoded.design.shape[1]), dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
        )
    first_destination = {
        origin: next(destination for source, destination in edges if source == origin)
        for origin in sorted({source for source, _ in edges})
    }
    observed = spells["event"].to_numpy(dtype=bool)
    origin = spells["origin"].to_numpy(dtype=np.int64)
    destination = spells["destination"].to_numpy(dtype=np.int64)
    destination[~observed] = [
        first_destination[int(value)] for value in origin[~observed]
    ]
    feature_index = spells["feature_index"].to_numpy(dtype=np.int64)
    start = spells["start_time"].to_numpy(dtype=np.float64)
    stop = start + spells["duration"].to_numpy(dtype=np.float64)
    scale = max(float(experiment.max_time), 1.0)
    return TensorDataset(
        torch.as_tensor(encoded.design[feature_index], dtype=torch.float32),
        torch.as_tensor(start / scale, dtype=torch.float32),
        torch.as_tensor(stop / scale, dtype=torch.float32),
        torch.as_tensor(origin),
        torch.as_tensor(destination),
        torch.as_tensor(observed.astype(np.float32)),
    )


def _loss(
    model: SurvNODE,
    batch: tuple[torch.Tensor, ...],
    penalty: float,
    *,
    integration_times: torch.Tensor | None = None,
    regularizer_time: torch.Tensor | None = None,
) -> torch.Tensor:
    survival, hazard, latent_output = model.interval_likelihood(
        *batch,
        integration_times=integration_times,
        regularizer_time=regularizer_time,
    )
    event = batch[-1]
    if (
        not torch.isfinite(survival).all()
        or not torch.isfinite(hazard).all()
        or torch.any(survival <= 0)
        or torch.any(survival > 1.0 + 1e-10)
        or torch.any(hazard <= 0)
    ):
        raise FloatingPointError("SurvNODE produced invalid survival or hazard values")
    log_likelihood = event * torch.log(hazard)
    log_likelihood = log_likelihood + torch.log(survival)
    regularizer = torch.linalg.vector_norm(latent_output, dim=1).mean()
    return -log_likelihood.mean() + penalty * regularizer


@torch.no_grad()
def _evaluate(model: SurvNODE, loader: DataLoader, device: torch.device, penalty: float) -> float:
    model.eval()
    total = 0.0
    count = 0
    for raw in loader:
        batch = tuple(value.to(device) for value in raw)
        value = _loss(model, batch, penalty)
        total += float(value.item()) * len(batch[0])
        count += len(batch[0])
    if count == 0:
        raise RuntimeError("SurvNODE validation loader is empty")
    return total / count


def _subsample_training(
    outer_training: pd.DataFrame,
    candidate_ids: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    candidates = outer_training[outer_training["id"].isin(candidate_ids)].copy()
    requested = max(2, int(round(fraction * len(candidates))))
    requested = min(requested, len(candidates))
    if requested == len(candidates):
        return candidates["id"].to_numpy()
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(candidates), size=requested, replace=False))
    return candidates.iloc[selected]["id"].to_numpy()


def _release_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _backward_logical_batch(
    model: SurvNODE,
    raw: tuple[torch.Tensor, ...] | list[torch.Tensor],
    device: torch.device,
    penalty: float,
    microbatch_size: int,
    fold: int,
    epoch: int,
    batch_index: int,
) -> float:
    logical_batch_size = len(raw[0])
    logical_loss = 0.0
    logical_start = raw[1].to(device)
    logical_stop = raw[2].to(device)
    logical_times = torch.unique(
        torch.cat([logical_start.new_zeros(1), logical_start, logical_stop]),
        sorted=True,
    )
    logical_regularizer_time = logical_stop.max()
    for micro_start in range(0, logical_batch_size, microbatch_size):
        micro_stop = min(micro_start + microbatch_size, logical_batch_size)
        batch = tuple(value[micro_start:micro_stop].to(device) for value in raw)
        loss = _loss(
            model,
            batch,
            penalty,
            integration_times=logical_times,
            regularizer_time=logical_regularizer_time,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "SurvNODE non-finite loss "
                f"fold={fold} epoch={epoch} batch={batch_index}"
            )
        weight = len(batch[0]) / logical_batch_size
        (loss * weight).backward()
        logical_loss += float(loss.item()) * weight
        if micro_stop < logical_batch_size:


            del loss, batch
            _release_cuda_cache(device)
    return logical_loss


def run_survnode_fold(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    targets: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
    output_dir: Path,
    epochs: int = 500,
    patience: int = 20,
    training_fraction: float = 1.0,
    seed: int = 2026,
    batch_size: int = 512,
    microbatch_size: int | None = None,
    penalty: float = 1e-4,
    learning_rate: float = 1e-3,
    encoder_hidden_features: int = 800,
    ode_hidden_features: int = 1000,
    solver_method: str = "dopri5",
    solver_rtol: float = 1e-8,
    solver_atol: float = 1e-8,
    gradient_clip_norm: float | None = None,
    optimizer_name: str = "adam",
    device: torch.device | None = None,
) -> pd.DataFrame:
    if not 0 < training_fraction <= 1:
        raise ValueError("SurvNODE training_fraction must be in (0,1]")
    if epochs <= 0 or patience <= 0:
        raise ValueError("SurvNODE epochs and patience must be positive")
    if batch_size <= 0:
        raise ValueError("SurvNODE batch_size must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("SurvNODE learning_rate must be finite and positive")
    if min(encoder_hidden_features, ode_hidden_features) <= 0:
        raise ValueError("SurvNODE hidden widths must be positive")
    if solver_method not in {"dopri5", "rk4"}:
        raise ValueError("SurvNODE solver_method must be dopri5 or rk4")
    if min(solver_rtol, solver_atol) <= 0:
        raise ValueError("SurvNODE solver tolerances must be positive")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("SurvNODE gradient_clip_norm must be positive")
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("SurvNODE optimizer_name must be adam or adamw")
    microbatch_size = batch_size if microbatch_size is None else microbatch_size
    if microbatch_size <= 0 or microbatch_size > batch_size:
        raise ValueError(
            "SurvNODE microbatch_size must be positive and no larger than batch_size"
        )
    set_seed(seed + fold)
    device = device or torch.device("cpu")
    started = time.time()
    outer_training = folds[folds["outer_fold"] != fold].copy()
    candidate_ids, validation_ids = split_train_validation(
        outer_training, seed=seed + fold
    )
    train_ids = _subsample_training(
        outer_training, candidate_ids, training_fraction, seed + fold
    )
    fit_frame = frame[frame["id"].isin(train_ids)]
    encoder = fit_baseline_encoder(fit_frame, experiment)
    encoded = encode_frame(frame, encoder)
    baseline_index = baseline_feature_indices(encoded)
    edges = nonself_edges(experiment)
    train_data = _step_dataset(encoded, set(map(int, train_ids)), experiment, edges)
    validation_data = _step_dataset(
        encoded, set(map(int, validation_ids)), experiment, edges
    )
    if min(len(train_data), len(validation_data)) == 0:
        raise ValueError(f"Empty SurvNODE partition in fold {fold}")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_data, batch_size=microbatch_size, shuffle=False
    )
    model = SurvNODE(
        experiment.num_states,
        edges,
        encoded.design.shape[1],
        encoder_hidden_features=encoder_hidden_features,
        ode_hidden_features=ode_hidden_features,
        solver_method=solver_method,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
    ).to(device=device, dtype=torch.float32)
    optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_index, raw in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            logical_loss = _backward_logical_batch(
                model,
                raw,
                device,
                penalty,
                microbatch_size,
                fold,
                epoch,
                batch_index,
            )
            if gradient_clip_norm is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(
                        "SurvNODE produced a non-finite gradient norm "
                        f"fold={fold} epoch={epoch} batch={batch_index}"
                    )
            optimizer.step()
            losses.append(logical_loss)
            print(
                f"survnode fold={fold} epoch={epoch}/{epochs} "
                f"batch={batch_index}/{len(train_loader)} loss={logical_loss:.7f}",
                flush=True,
            )
        validation_nll = _evaluate(model, validation_loader, device, penalty)
        train_nll = float(np.mean(losses))
        history.append(
            {"epoch": epoch, "train_nll": train_nll, "validation_nll": validation_nll}
        )
        if epoch == 1 or validation_nll < best - 1e-6:
            best = validation_nll
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, output_dir / "best.pt")
        else:
            stale += 1
        print(
            f"survnode fold={fold} epoch={epoch}/{epochs} DONE "
            f"train={train_nll:.7f} val={validation_nll:.7f}",
            flush=True,
        )
        if stale >= patience:
            print(
                f"survnode fold={fold} early_stop epoch={epoch} patience={patience}",
                flush=True,
            )
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])


    model = model.to(dtype=torch.float64)
    model.eval()

    test = targets[targets["fold"] == fold].sort_values(KEYS).copy()
    probability = np.empty((len(test), experiment.num_states), dtype=np.float64)
    row_lookup = {
        tuple(map(int, key)): index for index, key in enumerate(test[KEYS].to_numpy())
    }
    scale = max(float(experiment.max_time), 1.0)
    with torch.no_grad():
        for landmark, landmark_rows in test.groupby("landmark", sort=True):
            pairs = landmark_rows[["id", "origin_state"]].drop_duplicates()
            prediction_batch_size = min(256, microbatch_size)
            for start in range(0, len(pairs), prediction_batch_size):
                batch_pairs = pairs.iloc[start : start + prediction_batch_size]
                indices = [
                    baseline_index[int(row.id)]
                    for row in batch_pairs.itertuples(index=False)
                ]
                features = torch.as_tensor(
                    encoded.design[indices], dtype=torch.float64, device=device
                )
                origins = torch.as_tensor(
                    batch_pairs["origin_state"].to_numpy(dtype=np.int64),
                    device=device,
                )
                horizons = torch.arange(
                    int(landmark) + 1,
                    int(landmark) + experiment.max_horizon + 1,
                    dtype=torch.float64,
                    device=device,
                ) / scale
                occupation = model.occupation(
                    features, float(landmark) / scale, horizons, origins
                ).cpu().numpy()
                for pair_index, row in enumerate(batch_pairs.itertuples(index=False)):
                    for horizon in range(1, experiment.max_horizon + 1):
                        key = (int(row.id), fold, int(landmark), horizon)
                        probability[row_lookup[key]] = occupation[pair_index, horizon - 1]
    predictions = test[KEYS].reset_index(drop=True)
    for state in range(experiment.num_states):
        predictions[f"p_state_{state}"] = probability[:, state]
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "training.json").write_text(
        json.dumps(
            {
                "method": "SurvNODE neural generator ODE",
                "training_fraction_requested": training_fraction,
                "outer_training_patients": int(len(outer_training)),
                "model_training_patients": int(len(train_ids)),
                "validation_patients": int(len(validation_ids)),
                "candidate_training_patients_before_subsample": int(len(candidate_ids)),
                "training_fraction_realized_of_candidate_training": float(
                    len(train_ids) / len(candidate_ids)
                ),
                "training_fraction_realized_of_outer_training": float(
                    len(train_ids) / len(outer_training)
                ),
                "subsample_rule": "uniform random patient sample with frozen seed",
                "best_epoch": int(checkpoint["epoch"]),
                "best_validation_nll": best,
                "epochs_run": len(history),
                "maximum_epochs": epochs,
                "early_stopping_patience": patience,
                "logical_training_batch_size": batch_size,
                "microbatch_size": microbatch_size,
                "learning_rate": learning_rate,
                "encoder_hidden_features": encoder_hidden_features,
                "ode_hidden_features": ode_hidden_features,
                "solver_method": solver_method,
                "solver_rtol": solver_rtol,
                "solver_atol": solver_atol,
                "gradient_clip_norm": gradient_clip_norm,
                "optimizer_name": optimizer_name,
                "parameter_count": int(sum(p.numel() for p in model.parameters())),
                "prediction_batch_size": min(256, microbatch_size),
                "microbatch_semantics": (
                    "weighted gradient accumulation; one optimizer step per logical batch"
                ),
                "stopped_by_patience_before_epoch_cap": len(history) < epochs,
                "best_at_training_boundary": int(checkpoint["epoch"]) == len(history),
                "runtime_seconds": time.time() - started,
                "reference": "Groha, Schmon, and Gusev 2020/2021; official SurvNODE repository",
                "reference_url": "https://github.com/stefangroha/SurvNODE",
                "repository_commit": "ef65d782cc3bf7a4347e5ce76af8de55253af328",
                "implementation_status": (
                    "official-code reproduction with the minimum recurrent-graph "
                    "likelihood adapter"
                ),
                "occupation_prediction_dtype": "float64",
                "recurrent_graph_adapter": (
                    "the official progressive-graph loss uses P_ii(s,t), which equals "
                    "no-transition survival only when leaving-and-returning paths are "
                    "impossible; recurrent graphs use the same learned generator and "
                    "integrate its outgoing hazards over each observed spell"
                ),
                "numerical_probability_policy": (
                    "fail if row-sum error exceeds 1e-3; only negative roundoff "
                    "within 1e-8 is zeroed and float32 forward/backward composition "
                    "error below 1e-3 is renormalized"
                ),
                "training_rows": "one counting-process interval per state spell",
                "covariates": "cohort-entry baseline only",
                "architecture": (
                    f"encoder {encoder_hidden_features}-{encoder_hidden_features}, "
                    f"latent 20, ODE network {ode_hidden_features}-"
                    f"{ode_hidden_features}-{ode_hidden_features}, no dropout"
                ),
                "optimizer": (
                    f"{optimizer_class.__name__} lr={learning_rate:g} "
                    f"weight_decay=1e-6; gradient_clip_norm={gradient_clip_norm}"
                ),
                "solver": (
                    f"torchdiffeq {solver_method}, "
                    f"rtol={solver_rtol:g}, atol={solver_atol:g}"
                ),
                "time_scale": scale,
                "edges": edges,
                "feature_encoder": encoder.state_dict(),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return predictions
