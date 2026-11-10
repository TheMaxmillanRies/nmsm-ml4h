from __future__ import annotations

from dataclasses import dataclass
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
    current_dwell,
    destination_map,
    encode_frame,
    fit_baseline_encoder,
)
from .config import ExperimentConfig
from .splits import split_train_validation
from .targets import KEYS
from .training import set_seed


@dataclass(frozen=True)
class IDArchitecture:
    input_features: int
    hidden_size: int
    max_duration: int
    destinations: dict[int, tuple[int, ...]]


class GeneralIDNetwork(nn.Module):
    def __init__(self, architecture: IDArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        hidden = architecture.hidden_size
        self.trunk = nn.Sequential(
            nn.Linear(architecture.input_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )


        self.branches = nn.ModuleDict(
            {
                f"{origin}_{destination}": nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU()
                )
                for origin, destinations in architecture.destinations.items()
                for destination in destinations
            }
        )
        self.heads = nn.ModuleDict(
            {
                f"{origin}_{destination}": nn.Linear(
                    hidden, architecture.max_duration + 1
                )
                for origin, destinations in architecture.destinations.items()
                for destination in destinations
            }
        )


        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def logits(self, features: torch.Tensor, origin: int) -> torch.Tensor:
        shared = self.trunk(features)
        return torch.stack(
            [
                self.heads[f"{origin}_{destination}"](
                    self.branches[f"{origin}_{destination}"](shared)
                )
                for destination in self.architecture.destinations[origin]
            ],
            dim=1,
        )

    def mass(self, features: torch.Tensor, origin: int) -> torch.Tensor:
        logits = self.logits(features, origin)
        return torch.softmax(logits.flatten(1), dim=1).reshape_as(logits)

    def smoothness(self) -> torch.Tensor:
        penalty = next(self.parameters()).new_zeros(())
        for origin, destinations in self.architecture.destinations.items():
            for destination in destinations:
                head = self.heads[f"{origin}_{destination}"]
                penalty = penalty + (head.weight[1:] - head.weight[:-1]).abs().sum()
                penalty = penalty + (head.bias[1:] - head.bias[:-1]).abs().sum()
        return penalty

    def output_l1(self) -> torch.Tensor:


        return sum(head.weight.abs().sum() for head in self.heads.values())

    def hidden_l2(self) -> torch.Tensor:
        'L2 penalty on shared and transition-specific hidden layers only.'
        penalty = next(self.parameters()).new_zeros(())
        for module in (self.trunk, self.branches):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    penalty = penalty + layer.weight.square().sum()
        return penalty


@dataclass(frozen=True)
class IDNetworkConfig:
    'Compatibility configuration for the general graph IDNetwork adapter.'

    num_states: int
    transient_states: tuple[int, ...]
    num_features: int
    categorical_cardinalities: tuple[int, ...]
    time_bins: int
    hidden: int = 64
    dropout: float = 0.0


class IDNetworkMultiState(nn.Module):
    'Graph descriptor used by the exact residual-time propagation routine.'

    def __init__(self, config: IDNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.destinations = {
            origin: tuple(
                destination
                for destination in range(config.num_states)
                if destination != origin
            )
            for origin in config.transient_states
        }


def exact_residual_occupation(
    model: IDNetworkMultiState,
    masses: dict[int, np.ndarray],
    origins: np.ndarray,
    dwells: np.ndarray,
    horizons: int,
) -> np.ndarray:
    'Propagate joint destination/duration masses conditional on current dwell.'
    origins = np.asarray(origins, dtype=np.int64)
    dwells = np.asarray(dwells, dtype=np.int64)
    output = np.empty((len(origins), horizons, model.config.num_states), dtype=np.float64)
    terminals = tuple(
        state
        for state in range(model.config.num_states)
        if state not in model.config.transient_states
    )
    for patient, (origin, dwell) in enumerate(zip(origins, dwells)):
        patient_masses = {
            state: np.asarray(values[patient], dtype=np.float64)
            for state, values in masses.items()
        }
        output[patient] = _occupation(
            patient_masses,
            model.destinations,
            int(origin),
            int(dwell),
            horizons,
            model.config.num_states,
            terminals,
        )
    return output


def _spells(
    encoded,
    patient_ids: set[int],
    destinations: dict[int, tuple[int, ...]],
) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    selected = encoded.frame[encoded.frame["id"].isin(patient_ids)]
    baseline_index = baseline_feature_indices(encoded)
    for patient, group in selected.groupby("id", sort=False):
        group = group.sort_values("t")
        records = list(group.itertuples())
        next_states = pd.to_numeric(group["next_state"], errors="coerce").to_numpy(dtype=float)
        cursor = 0
        while cursor < len(records):
            first = records[cursor]
            origin = int(first.state)
            if origin not in destinations:
                break
            start_time = int(first.t)
            end = cursor
            event = 0
            destination = -1
            censor_time = start_time
            while end < len(records) and int(records[end].state) == origin:
                record = records[end]
                raw_next = next_states[end]
                if np.isfinite(raw_next) and int(raw_next) >= 0:
                    censor_time = int(record.t) + 1
                    if int(raw_next) != origin:
                        event = 1
                        destination = int(raw_next)
                        break
                else:
                    censor_time = int(record.t)
                end += 1
            duration = int(censor_time - start_time)
            if event and destination not in destinations[origin]:
                raise ValueError(
                    f"IDNetwork found forbidden transition {origin}->{destination}"
                )
            if duration > 0:
                rows.append(
                    {
                        "id": int(patient),
                        "start_time": start_time,
                        "origin": origin,
                        "duration": duration,
                        "event": event,
                        "destination": destination,


                        "feature_index": baseline_index[int(patient)],
                    }
                )
            if event:
                cursor = end + 1
            else:
                cursor = max(cursor + 1, end)
    return pd.DataFrame(rows)


def stepwise_to_spells(
    frame: pd.DataFrame,
    patient_ids: set[int],
    encoder,
    terminal_states: tuple[int, ...],
    baseline_covariates: bool = False,
) -> pd.DataFrame:
    'Create recurrent counting-process spells with causal feature encoding.'
    encoded = encode_frame(frame, encoder)
    terminals = frozenset(map(int, terminal_states))
    destinations = {
        origin: tuple(
            destination
            for destination in range(int(frame["state"].max()) + 1)
            if destination != origin
        )
        for origin in range(int(frame["state"].max()) + 1)
        if origin not in terminals
    }
    spells = _spells(encoded, set(map(int, patient_ids)), destinations)
    if spells.empty:
        return spells.assign(values=[], categories=[])
    if baseline_covariates:
        first = baseline_feature_indices(encoded)
        indices = spells["id"].map(first).to_numpy(dtype=np.int64)
    else:


        indices = np.asarray(
            [encoded.lookup[(int(row.id), int(row.start_time))] for row in spells.itertuples(index=False)],
            dtype=np.int64,
        )
    spells = spells.copy()
    spells["values"] = list(encoded.continuous[indices].astype(np.float32))
    spells["categories"] = list(encoded.categories[indices].astype(np.int64))
    return spells


def _landmark_dwell(
    frame: pd.DataFrame, pairs: pd.DataFrame
) -> dict[tuple[int, int], int]:
    ordered = frame[["id", "t", "state"]].sort_values(["id", "t"])
    ids, times, states = (ordered[column].to_numpy(dtype=np.int64) for column in ("id", "t", "state"))
    boundary = np.r_[True, (ids[1:] != ids[:-1]) | (states[1:] != states[:-1])
                     | (times[1:] != times[:-1] + 1)]
    starts = np.maximum.accumulate(np.where(boundary, np.arange(len(ordered)), 0))
    index = pd.MultiIndex.from_arrays([ids, times])
    requested = pairs[["id", "landmark"]].drop_duplicates()
    positions = index.get_indexer(pd.MultiIndex.from_frame(requested))
    if np.any(positions < 0):
        raise KeyError("Missing requested IDNetwork landmark")
    dwell = times - times[starts]
    return {tuple(map(int, key)): int(dwell[position])
            for key, position in zip(requested.to_numpy(), positions)}


def _dataset(
    spells: pd.DataFrame,
    design: np.ndarray,
    destinations: dict[int, tuple[int, ...]],
) -> TensorDataset:
    destination_index = np.full(len(spells), -1, dtype=np.int64)
    for index, row in enumerate(spells.itertuples(index=False)):
        if int(row.event):
            destination_index[index] = destinations[int(row.origin)].index(
                int(row.destination)
            )
    return TensorDataset(
        torch.as_tensor(
            design[spells["feature_index"].to_numpy(dtype=np.int64)],
            dtype=torch.float32,
        ),
        torch.as_tensor(spells["origin"].to_numpy(dtype=np.int64)),
        torch.as_tensor(spells["duration"].to_numpy(dtype=np.int64)),
        torch.as_tensor(spells["event"].to_numpy(dtype=np.int64)),
        torch.as_tensor(destination_index),
    )


def idnetwork_nll(
    model: GeneralIDNetwork,
    batch: tuple[torch.Tensor, ...],
    smoothness: float,
    output_l1: float = 1e-5,
    hidden_l2: float = 1e-5,
) -> torch.Tensor:
    features, origins, durations, events, destination_indices = batch
    total = features.new_zeros(())
    count = 0
    maximum = model.architecture.max_duration
    for origin in model.architecture.destinations:
        selected = origins == origin
        if not torch.any(selected):
            continue
        logits = model.logits(features[selected], origin)
        log_mass = torch.log_softmax(logits.flatten(1), dim=1).reshape_as(logits)
        local_duration = durations[selected]
        local_event = events[selected].bool()
        local_destination = destination_indices[selected]
        nll = features.new_zeros(int(selected.sum()))
        observed = local_event
        if torch.any(observed):
            if torch.any(local_duration[observed] > maximum):
                raise ValueError("Observed IDNetwork duration exceeds configured grid")
            row = torch.arange(len(nll), device=features.device)[observed]
            duration_index = local_duration[observed] - 1
            nll[observed] = -log_mass[
                row, local_destination[observed], duration_index
            ]
        censored = ~observed
        if torch.any(censored):
            for row in torch.nonzero(censored, as_tuple=False).flatten():
                survived = min(int(local_duration[row].item()), maximum)
                nll[row] = -torch.logsumexp(log_mass[row, :, survived:], dim=(0, 1))
        total = total + nll.sum()
        count += len(nll)
    if count == 0:
        raise RuntimeError("IDNetwork batch contains no supported origin state")
    return (
        total / count
        + smoothness * model.smoothness()
        + output_l1 * model.output_l1()
        + hidden_l2 * model.hidden_l2()
    )


@torch.no_grad()
def _validation_loss(
    model: GeneralIDNetwork,
    loader: DataLoader,
    device: torch.device,
    smoothness: float,
    output_l1: float,
    hidden_l2: float,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for raw in loader:
        batch = tuple(value.to(device) for value in raw)
        loss = idnetwork_nll(model, batch, smoothness, output_l1, hidden_l2)
        size = len(batch[0])
        total += float(loss.item()) * size
        count += size
    if count == 0:
        raise RuntimeError("IDNetwork validation loader is empty")
    return total / count


def _occupation(
    masses: dict[int, np.ndarray],
    destinations: dict[int, tuple[int, ...]],
    origin: int,
    dwell: int,
    horizons: int,
    num_states: int,
    terminal_states: tuple[int, ...],
) -> np.ndarray:
    maximum = next(iter(masses.values())).shape[1] - 1
    if dwell < 0 or dwell >= maximum:
        raise ValueError(
            f"IDNetwork dwell={dwell} is outside the supported recursion grid [0,{maximum})"
        )
    expanded = np.zeros((num_states, maximum + 1), dtype=np.float64)
    expanded[origin, dwell] = 1.0
    output = np.empty((horizons, num_states), dtype=np.float64)
    terminals = frozenset(terminal_states)
    for step in range(horizons):
        updated = np.zeros_like(expanded)
        for state in range(num_states):
            for age in np.flatnonzero(expanded[state] > 0):
                weight = expanded[state, age]
                if state in terminals:
                    updated[state, min(age + 1, maximum)] += weight
                    continue
                if age >= maximum:
                    raise FloatingPointError(
                        "IDNetwork nonterminal path reached the duration-grid boundary"
                    )
                mass = masses[state]
                survival = float(mass[:, age:].sum())
                if not np.isfinite(survival) or survival <= 0:
                    raise FloatingPointError(
                        f"IDNetwork state={state} age={age} has invalid residual survival={survival}"
                    )
                event_probability = mass[:, age] / survival
                if not np.isfinite(event_probability).all():
                    raise FloatingPointError("IDNetwork produced non-finite conditional event mass")
                event_total = float(event_probability.sum())
                if event_total < -1e-12 or event_total > 1.0 + 1e-12:
                    raise FloatingPointError(
                        f"IDNetwork conditional event mass is outside [0,1]: {event_total}"
                    )
                event_total = min(1.0, max(0.0, event_total))
                stay = 1.0 - event_total
                updated[state, age + 1] += weight * stay
                for probability, destination in zip(
                    event_probability, destinations[state]
                ):
                    updated[destination, 0] += weight * float(probability)
        total = float(updated.sum())
        if total <= 0 or not np.isfinite(total):
            raise FloatingPointError("IDNetwork occupation recursion lost probability mass")


        if not np.isclose(total, 1.0, atol=1e-7, rtol=0.0):
            raise FloatingPointError(
                f"IDNetwork occupation recursion does not conserve mass: {total}"
            )
        expanded = updated / total
        output[step] = expanded.sum(axis=1)
    return output


def run_idnetwork_general_fold(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    targets: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
    output_dir: Path,
    epochs: int = 50,
    patience: int = 8,
    seed: int = 2026,
    smoothness: float = 1e-3,
    output_l1: float = 1e-5,
    hidden_l2: float = 1e-5,
    device: torch.device | None = None,
) -> pd.DataFrame:
    set_seed(seed + fold)
    device = device or torch.device("cpu")
    started = time.time()
    outer_training = folds[folds["outer_fold"] != fold]
    train_ids, validation_ids = split_train_validation(
        outer_training, seed=seed + fold
    )
    fit_frame = frame[frame["id"].isin(train_ids)]
    encoder = fit_baseline_encoder(fit_frame, experiment)
    encoded = encode_frame(frame, encoder)
    destinations = destination_map(experiment)
    train_spells = _spells(encoded, set(map(int, train_ids)), destinations)
    validation_spells = _spells(
        encoded, set(map(int, validation_ids)), destinations
    )
    if train_spells.empty or validation_spells.empty:
        raise ValueError(f"Empty IDNetwork spell partition in fold {fold}")
    observed_duration = train_spells.loc[
        train_spells["event"] == 1, "duration"
    ].max()
    observed_duration = int(observed_duration) if np.isfinite(observed_duration) else 1


    maximum = max(
        experiment.max_horizon + 1,
        int(experiment.max_time) + experiment.max_horizon + 1,
        observed_duration + experiment.max_horizon + 1,
    )
    architecture = IDArchitecture(
        encoded.design.shape[1], 50, maximum, destinations
    )
    model = GeneralIDNetwork(architecture).to(device)
    train_loader = DataLoader(
        _dataset(train_spells, encoded.design, destinations),
        batch_size=128,
        shuffle=True,
    )
    validation_loader = DataLoader(
        _dataset(validation_spells, encoded.design, destinations),
        batch_size=2048,
        shuffle=False,
    )


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for raw in train_loader:
            batch = tuple(value.to(device) for value in raw)
            loss = idnetwork_nll(
                model, batch, smoothness, output_l1, hidden_l2
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation_nll = _validation_loss(
            model,
            validation_loader,
            device,
            smoothness,
            output_l1,
            hidden_l2,
        )
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
            f"idnetwork fold={fold} epoch={epoch}/{epochs} train={train_nll:.7f} "
            f"val={validation_nll:.7f} stale={stale}",
            flush=True,
        )
        if stale >= patience:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test = targets[targets["fold"] == fold].sort_values(KEYS).copy()
    probability = np.empty((len(test), experiment.num_states), dtype=np.float64)
    patient_baseline_index = baseline_feature_indices(encoded)
    row_lookup = {
        tuple(map(int, key)): index for index, key in enumerate(test[KEYS].to_numpy())
    }
    with torch.no_grad():
        for (patient, landmark), group in test.groupby(["id", "landmark"], sort=False):
            feature_index = patient_baseline_index[int(patient)]
            feature = torch.as_tensor(
                encoded.design[feature_index : feature_index + 1],
                dtype=torch.float32,
                device=device,
            )
            masses = {
                origin: model.mass(feature, origin)[0].cpu().numpy()
                for origin in destinations
            }
            occupation = _occupation(
                masses,
                destinations,
                int(group["origin_state"].iloc[0]),
                current_dwell(frame, int(patient), int(landmark)),
                experiment.max_horizon,
                experiment.num_states,
                experiment.terminal_states,
            )
            for horizon in range(1, experiment.max_horizon + 1):
                key = (int(patient), fold, int(landmark), horizon)
                probability[row_lookup[key]] = occupation[horizon - 1]
    predictions = test[KEYS].reset_index(drop=True)
    for state in range(experiment.num_states):
        predictions[f"p_state_{state}"] = probability[:, state]
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    origin_counts = train_spells["origin"].value_counts().sort_index().to_dict()
    event_counts = (
        train_spells[train_spells["event"] == 1]
        .groupby(["origin", "destination"])
        .size()
        .to_dict()
    )
    (output_dir / "training.json").write_text(
        json.dumps(
            {
                "method": "IDNetwork joint destination/waiting-time distribution",
                "reference": "Cottin et al. 2021, PMLR 146",
                "reference_url": "https://proceedings.mlr.press/v146/cottin21a.html",
                "implementation_status": "paper reconstruction; no author code located",
                "repository_search": "no author repository located as of 2026-08-27",
                "adaptation": (
                    "published baseline-covariate architecture generalized from the "
                    "three-state illness-death graph to the prespecified graph"
                ),
                "reconstruction_limit": (
                    "the paper reports dataset-specific random search but not the "
                    "selected settings; the fixed architecture and penalties are "
                    "inside its published search spaces"
                ),
                "covariates": "cohort-entry baseline only",
                "architecture": "2x50 shared ReLU plus 1x50 per transition",
                "learning_rate": 1e-4,
                "learning_rate_decay_factor": 1.0,
                "optimizer_weight_decay": 0.0,
                "best_epoch": int(checkpoint["epoch"]),
                "best_validation_nll": best,
                "runtime_seconds": time.time() - started,
                "maximum_duration": maximum,
                "smoothness": smoothness,
                "output_layer_l1": output_l1,
                "hidden_layer_l2": hidden_l2,
                "epochs_run": len(history),
                "maximum_epochs": epochs,
                "early_stopping_patience": patience,
                "stopped_by_patience_before_epoch_cap": len(history) < epochs,
                "best_at_training_boundary": int(checkpoint["epoch"]) == len(history),
                "destinations": {str(key): value for key, value in destinations.items()},
                "training_spells_by_origin": origin_counts,
                "training_events_by_edge": {
                    f"{key[0]}->{key[1]}": value for key, value in event_counts.items()
                },
                "feature_encoder": encoder.state_dict(),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return predictions
