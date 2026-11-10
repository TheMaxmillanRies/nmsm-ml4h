from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from .config import ExperimentConfig
from .data import FeatureEncoder, add_history_summaries
from .model import JointCausalTransformer, allowed_transitions
from .rollout import internal_frame
from .training import target_at


@dataclass(frozen=True)
class PathExample:
    patient: int
    landmark: int
    origin_state: int
    states: torch.Tensor
    covariates: torch.Tensor
    categories: torch.Tensor
    times: torch.Tensor
    dwell: torch.Tensor
    score_start: int
    targets: torch.Tensor


def make_path_examples(
    scaler: FeatureEncoder,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
) -> list[PathExample]:
    'Build teacher-forced future paths under a declared LOCF covariate rule.'

    simulation = config.name == "simulation"
    raw = internal_frame(test_df, simulation)
    terminals = frozenset(config.terminal_states)
    raw_groups = {
        int(patient): group.sort_values("t")
        for patient, group in raw.groupby("id", sort=False)
    }
    examples: list[PathExample] = []
    for patient in sorted(raw_groups):
        raw_group = raw_groups[patient]
        group = add_history_summaries(raw_group)
        encoded_values, encoded_categories = scaler.encode(group)
        encoded_by_time = {int(time): index for index, time in enumerate(group["t"].to_numpy(dtype=int))}
        for landmark in config.landmarks:
            landmark_rows = group[group["t"] == landmark]
            if landmark_rows.empty:
                continue
            origin = int(landmark_rows.iloc[0]["state"])
            if origin in terminals:
                continue
            history = group[group["t"] <= landmark]
            targets: list[int] = []
            for horizon in range(1, config.max_horizon + 1):
                target, known, _ = target_at(raw_group, landmark + horizon, terminals)
                if not known:
                    break
                targets.append(int(target))
            if not targets:
                continue
            history_states = history["state"].to_numpy(dtype=np.int64)
            history_indices = np.array([encoded_by_time[int(time)] for time in history["t"]], dtype=int)
            history_covariates = encoded_values[history_indices]
            history_categories = encoded_categories[history_indices]
            history_times = history["t"].to_numpy(dtype=np.int64)
            history_dwell = history["time_in_state"].to_numpy(dtype=np.float32)

            future_inputs = np.asarray(targets[:-1], dtype=np.int64)
            if len(future_inputs):
                states = np.concatenate([history_states, future_inputs])
                future_covariate_rows = []
                current_covariates = torch.as_tensor(history_covariates[-1:])
                for _ in future_inputs:
                    current_covariates = scaler.advance_locf(current_covariates)
                    future_covariate_rows.append(current_covariates[0].numpy())
                future_covariates = np.asarray(future_covariate_rows, dtype=np.float32)
                covariates = np.concatenate([history_covariates, future_covariates], axis=0)
                future_categories = np.repeat(
                    history_categories[-1][None, :], len(future_inputs), axis=0
                )
                categories = np.concatenate([history_categories, future_categories], axis=0)
                times = np.concatenate(
                    [
                        history_times,
                        np.arange(landmark + 1, landmark + len(targets), dtype=np.int64),
                    ]
                )
                dwell_values = list(history_dwell)
                previous_state = int(history_states[-1])
                previous_dwell = float(history_dwell[-1])
                for next_input_state in future_inputs:
                    next_input_state = int(next_input_state)
                    previous_dwell = (
                        previous_dwell + 1.0
                        if next_input_state == previous_state
                        else 0.0
                    )
                    dwell_values.append(previous_dwell)
                    previous_state = next_input_state
                dwell = np.asarray(dwell_values, dtype=np.float32)
            else:
                states = history_states
                covariates = history_covariates
                categories = history_categories
                times = history_times
                dwell = history_dwell
            examples.append(
                PathExample(
                    patient=patient,
                    landmark=landmark,
                    origin_state=origin,
                    states=torch.as_tensor(states, dtype=torch.long),
                    covariates=torch.as_tensor(covariates, dtype=torch.float32),
                    categories=torch.as_tensor(categories, dtype=torch.long),
                    times=torch.as_tensor(times, dtype=torch.long),
                    dwell=torch.as_tensor(dwell, dtype=torch.float32),
                    score_start=len(history_states) - 1,
                    targets=torch.as_tensor(targets, dtype=torch.long),
                )
            )
    return examples


def masked_transition_nll(
    logits: torch.Tensor,
    current_states: torch.Tensor,
    targets: torch.Tensor,
    graph: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    'Negative log probability of observed graph-constrained transitions.'

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    permitted = graph[current_states]
    calibrated = logits.masked_fill(~permitted, -1e9) / temperature
    log_probabilities = torch.log_softmax(calibrated, dim=-1)
    return -log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def score_path_examples(
    model: JointCausalTransformer,
    examples: list[PathExample],
    config: ExperimentConfig,
    variant: str,
    fold: int,
    batch_size: int = 32,
    temperature: float = 1.0,
) -> pd.DataFrame:
    'Score exact held-out future paths with one causal forward pass each.'

    if model.config.context == "snapshot":
        raise ValueError("Snapshot controls require a separate path scorer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    graph = allowed_transitions(
        config.num_states, config.terminal_states, config.allowed_edges
    ).to(device)
    model.eval()
    rows: list[dict[str, float | int | str]] = []
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset : offset + batch_size]
        states = pad_sequence([example.states for example in batch], batch_first=True).to(device)
        covariates = pad_sequence(
            [example.covariates for example in batch], batch_first=True
        ).to(device)
        categories = pad_sequence(
            [example.categories for example in batch], batch_first=True
        ).to(device)
        times = pad_sequence([example.times for example in batch], batch_first=True).to(device)
        dwell = pad_sequence([example.dwell for example in batch], batch_first=True).to(device)
        lengths = torch.as_tensor([len(example.states) for example in batch], device=device)
        valid = torch.arange(states.shape[1], device=device)[None, :] < lengths[:, None]
        hidden = model.encode(
            states,
            covariates,
            times,
            valid,
            categories if categories.shape[-1] else None,
            dwell if model.config.use_dwell else None,
        )
        logits = model.state_head(hidden)
        for index, example in enumerate(batch):
            horizons = len(example.targets)
            selected = logits[index, example.score_start : example.score_start + horizons]
            current = states[index, example.score_start : example.score_start + horizons]
            targets = example.targets.to(device)
            transition_nll = masked_transition_nll(
                selected, current, targets, graph, temperature
            ).cpu().numpy()
            cumulative = np.cumsum(transition_nll)
            for horizon in range(1, horizons + 1):
                rows.append(
                    {
                        "id": example.patient,
                        "fold": fold,
                        "variant": variant,
                        "landmark": example.landmark,
                        "horizon": horizon,
                        "origin_state": example.origin_state,
                        "current_state": int(current[horizon - 1].item()),
                        "target_state": int(targets[horizon - 1].item()),
                        "transition_nll": float(transition_nll[horizon - 1]),
                        "cumulative_path_nll": float(cumulative[horizon - 1]),
                        "mean_path_nll": float(cumulative[horizon - 1] / horizon),
                        "joint_log_probability": float(-cumulative[horizon - 1]),
                        "covariate_mode": "locf",
                        "state_temperature": float(temperature),
                    }
                )
    return pd.DataFrame(rows)
