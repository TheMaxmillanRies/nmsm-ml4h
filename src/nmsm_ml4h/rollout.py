from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import FeatureEncoder, add_history_summaries
from .ipcw import conditional_ipcw, fit_censoring_km
from .model import JointCausalTransformer, allowed_transitions
from .splits import patient_table
from .training import target_at


def internal_frame(frame: pd.DataFrame, simulation: bool) -> pd.DataFrame:
    result = frame.copy()
    if simulation:
        mapping = {1: 0, 2: 1, 3: 2}
        result["state"] = result["state"].map(mapping)
        valid = result["next_state"].notna() & (result["next_state"] >= 0)
        result.loc[valid, "next_state"] = result.loc[valid, "next_state"].map(mapping)
    return result


def sample_joint_covariates(
    model: JointCausalTransformer,
    encoder: FeatureEncoder,
    hidden: torch.Tensor,
    current_values: torch.Tensor,
    next_state: torch.Tensor,
    terminals: frozenset[int],
    generator: torch.Generator,
    observation_mode: str = "learned",
    state_observation_rates: torch.Tensor | None = None,
    continuous_mode: str = "sample",
) -> tuple[torch.Tensor, dict[str, float]]:
    if observation_mode not in {"learned", "state_rate", "always_observed"}:
        raise ValueError("Unknown observation-process rollout mode")
    if continuous_mode not in {"sample", "mean"}:
        raise ValueError("continuous_mode must be 'sample' or 'mean'")
    continuous_indices = torch.as_tensor(
        encoder.dynamic_continuous_indices, dtype=torch.long, device=current_values.device
    )
    current_continuous = current_values.index_select(1, continuous_indices)
    parameters = model.covariate_outputs(hidden, next_state, current_continuous)
    location = parameters["covariate_location"]
    scale = torch.exp(parameters["covariate_log_scale"])
    if continuous_mode == "mean":
        continuous = location
    else:
        continuous = location + scale * torch.randn(
            location.shape,
            dtype=location.dtype,
            device=location.device,
            generator=generator,
        )
    learned_observation_probability = torch.sigmoid(
        parameters["observation_logits"]
    )
    if observation_mode == "learned":
        observation_probability = learned_observation_probability
    elif observation_mode == "always_observed":
        observation_probability = torch.ones_like(learned_observation_probability)
    else:
        expected_shape = (model.num_states, learned_observation_probability.shape[-1])
        if state_observation_rates is None or tuple(state_observation_rates.shape) != expected_shape:
            raise ValueError(
                "state_rate rollout requires a [num_states,num_observation_processes] "
                "training-fold rate matrix"
            )
        observation_probability = state_observation_rates.to(
            device=hidden.device, dtype=hidden.dtype
        )[next_state]
    observations = torch.rand(
        observation_probability.shape,
        dtype=observation_probability.dtype,
        device=observation_probability.device,
        generator=generator,
    ) < observation_probability
    binary_probability = torch.sigmoid(parameters["binary_logits"])
    binary = (
        torch.rand(
            binary_probability.shape,
            dtype=binary_probability.dtype,
            device=binary_probability.device,
            generator=generator,
        )
        < binary_probability
    ).to(current_values.dtype)
    generated = encoder.advance_joint(
        current_values, continuous, observations, binary
    )
    if terminals:
        terminal = torch.zeros_like(next_state, dtype=torch.bool)
        for state in terminals:
            terminal |= next_state == state
        carried = encoder.advance_locf(current_values)
        generated = torch.where(terminal[:, None], carried, generated)
    tensors = (location, scale, observation_probability, binary_probability, generated)
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("Non-finite stochastic covariate rollout")
    diagnostics = {
        "continuous_abs_max": float(continuous.abs().max().item()) if continuous.numel() else 0.0,
        "scale_max": float(scale.max().item()) if scale.numel() else 0.0,
        "observation_probability_min": float(observation_probability.min().item()) if observation_probability.numel() else 1.0,
        "observation_probability_max": float(observation_probability.max().item()) if observation_probability.numel() else 1.0,
    }
    return generated, diagnostics


def fit_state_observation_rates(
    training_df: pd.DataFrame,
    observation_columns: tuple[str, ...],
    num_states: int,
    simulation: bool = False,
) -> np.ndarray:
    'Estimate a leakage-free state-only observation-process control.'
    if not observation_columns:
        return np.empty((num_states, 0), dtype=np.float32)
    missing = set(observation_columns).difference(training_df.columns)
    if missing:
        raise ValueError(f"Training data lack observation columns: {sorted(missing)}")
    frame = internal_frame(training_df.copy(), simulation)
    numerators = np.ones((num_states, len(observation_columns)), dtype=np.float64)
    denominators = np.full((num_states, len(observation_columns)), 2.0, dtype=np.float64)
    for _, group in frame.sort_values(["id", "t"]).groupby("id", sort=False):
        if len(group) < 2:
            continue
        next_rows = group.iloc[1:]
        states = next_rows["state"].to_numpy(dtype=np.int64)
        if np.any((states < 0) | (states >= num_states)):
            raise ValueError("Outer-training state is outside the configured state space")
        observed = next_rows.loc[:, observation_columns].notna().to_numpy(dtype=np.float64)
        for state in range(num_states):
            selected = states == state
            if selected.any():
                numerators[state] += observed[selected].sum(axis=0)
                denominators[state] += float(selected.sum())
    rates = numerators / denominators
    if not np.isfinite(rates).all() or np.any((rates <= 0) | (rates >= 1)):
        raise FloatingPointError("Invalid state-only observation rates")
    return rates.astype(np.float32)


def validate_probability(probability: torch.Tensor, label: str) -> float:
    if not torch.isfinite(probability).all():
        raise FloatingPointError(f"{label} contains NaN or infinity")
    minimum = float(probability.min().item())
    maximum = float(probability.max().item())
    error = float((probability.sum(dim=-1) - 1.0).abs().max().item())
    if minimum < -1e-7 or maximum > 1.0 + 1e-7 or error > 1e-5:
        raise FloatingPointError(
            f"{label} is not a probability simplex: min={minimum} max={maximum} sum_error={error}"
        )
    return error


def scramble_observed_history_content(
    states: np.ndarray,
    values: np.ndarray,
    categories: np.ndarray,
    dwell: np.ndarray,
    patient_ids: list[int],
    landmark: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    'Break temporal order while preserving token content and the current token.'
    arrays = tuple(np.asarray(value).copy() for value in (states, values, categories, dwell))
    if any(value.shape[:2] != arrays[0].shape[:2] for value in arrays[1:]):
        raise ValueError("History arrays do not share batch and time dimensions")
    if len(patient_ids) != arrays[0].shape[0]:
        raise ValueError("One patient ID is required per history row")
    history_length = arrays[0].shape[1]
    if history_length <= 2:
        return arrays
    for row, patient in enumerate(patient_ids):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(patient), int(landmark)])
        )
        permutation = rng.permutation(history_length - 1)
        if np.array_equal(permutation, np.arange(history_length - 1)):
            permutation = np.roll(permutation, 1)
        for value in arrays:
            value[row, :-1] = value[row, permutation]
    return arrays


@torch.no_grad()
def rollout_predictions(
    model: JointCausalTransformer,
    encoder: FeatureEncoder,
    test_df: pd.DataFrame,
    training_df: pd.DataFrame,
    config: ExperimentConfig,
    variant: str,
    fold: int,
    n_sims: int,
    seed: int,
    max_patients: int | None = None,
    simulation_batch_size: int = 64,
    covariate_mode: str = "joint",
    observation_mode: str = "learned",
    continuous_mode: str = "sample",
    state_temperature: float = 1.0,
    patient_batch_size: int = 1,
    late_landmark_patient_batch_size: int | None = None,
    gpu_yield_seconds: float = 0.0,
    max_concurrent_paths: int = 128,
    history_order: str = "ordered",
    history_order_seed: int = 1729,
    history_mode: str = "dense",
) -> tuple[pd.DataFrame, float]:
    if model.config.context == "snapshot":
        raise ValueError("Use the dedicated snapshot/current+dwell control rollout")
    if n_sims <= 0 or simulation_batch_size <= 0 or patient_batch_size <= 0:
        raise ValueError("Simulation and patient batch sizes must be positive")
    if late_landmark_patient_batch_size is not None and late_landmark_patient_batch_size <= 0:
        raise ValueError("Late-landmark patient batch size must be positive")
    largest_patient_batch = max(
        patient_batch_size, late_landmark_patient_batch_size or patient_batch_size
    )
    if largest_patient_batch * simulation_batch_size > max_concurrent_paths:
        raise ValueError(
            "Unsafe rollout: patient_batch_size * simulation_batch_size exceeds "
            f"{max_concurrent_paths}"
        )
    if covariate_mode not in {"joint", "locf"}:
        raise ValueError("covariate_mode must be 'joint' or 'locf'")
    if covariate_mode == "locf" and (
        observation_mode != "learned" or continuous_mode != "sample"
    ):
        raise ValueError("Observation/value rollout controls apply only to joint continuation")
    if observation_mode not in {"learned", "state_rate", "always_observed"}:
        raise ValueError("Unknown observation-process rollout mode")
    if continuous_mode not in {"sample", "mean"}:
        raise ValueError("continuous_mode must be 'sample' or 'mean'")
    if not np.isfinite(state_temperature) or state_temperature <= 0:
        raise ValueError("state_temperature must be finite and positive")
    if not np.isfinite(gpu_yield_seconds) or gpu_yield_seconds < 0:
        raise ValueError("gpu_yield_seconds must be finite and nonnegative")
    if history_order not in {"ordered", "scrambled"}:
        raise ValueError("history_order must be 'ordered' or 'scrambled'")
    if history_mode not in {"dense", "state_events"}:
        raise ValueError("history_mode must be 'dense' or 'state_events'")

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    simulation = config.name == "simulation"
    eligible_ids = sorted(test_df["id"].astype(int).unique())
    if max_patients is not None:
        rng = np.random.default_rng(seed)
        eligible_ids = sorted(
            rng.choice(
                eligible_ids,
                size=min(max_patients, len(eligible_ids)),
                replace=False,
            )
        )
    raw = internal_frame(test_df[test_df["id"].isin(eligible_ids)], simulation)
    censoring_columns = [
        column for column in ("id", "t", "state", "next_state") if column in training_df
    ]
    training = internal_frame(training_df.loc[:, censoring_columns], simulation)
    km = fit_censoring_km(patient_table(training, config.terminal_states))
    graph = allowed_transitions(
        config.num_states, config.terminal_states, config.allowed_edges
    ).to(device)
    terminals = frozenset(config.terminal_states)
    observation_rates = (
        fit_state_observation_rates(
            training_df,
            encoder.observation_numeric_columns,
            config.num_states,
            simulation=simulation,
        )
        if observation_mode == "state_rate"
        else None
    )
    observation_rates_tensor = (
        torch.as_tensor(observation_rates, dtype=torch.float32, device=device)
        if observation_rates is not None
        else None
    )
    raw = add_history_summaries(raw).reset_index(drop=True)
    all_values, all_categories = encoder.encode_many(raw)
    grouped = {int(pid): group for pid, group in raw.groupby("id", sort=False)}
    encoded = {pid: (group, all_values[group.index], all_categories[group.index])
               for pid, group in grouped.items()}
    rows: list[dict] = []
    started = time.time()
    pair_count = 0
    max_simplex_error = 0.0
    max_continuous_abs = 0.0
    max_scale = 0.0
    model.eval()

    for landmark in config.landmarks:
        active_patient_batch_size = (
            late_landmark_patient_batch_size
            if late_landmark_patient_batch_size is not None
            and landmark != config.landmarks[0]
            else patient_batch_size
        )
        eligible: list[tuple[int, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]] = []
        for pid, (group, values, categories) in encoded.items():
            at_landmark = group[group["t"] == landmark]
            if at_landmark.empty or int(at_landmark.iloc[0]["state"]) in terminals:
                continue
            times = group["t"].to_numpy(dtype=int)
            history_indices = np.flatnonzero(times <= landmark)
            if history_mode == "state_events":
                states = group["state"].to_numpy(dtype=int)
                changed = np.ones(len(group), dtype=bool)
                changed[1:] = states[1:] != states[:-1]
                keep = (times < landmark) & changed
                keep |= times == landmark
                keep[0] = True
                history_indices = np.flatnonzero(keep)
            eligible.append((pid, group, values, categories, history_indices))


        eligible.sort(key=lambda item: (len(item[4]), item[0]))
        batches: list[list[tuple[int, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]]] = []
        cursor = 0
        while cursor < len(eligible):
            length = len(eligible[cursor][4])
            stop = cursor
            while stop < len(eligible) and len(eligible[stop][4]) == length:
                stop += 1
            for start in range(cursor, stop, active_patient_batch_size):
                batches.append(eligible[start : min(start + active_patient_batch_size, stop)])
            cursor = stop
        for items in batches:
            histories = [item[1].iloc[item[4]] for item in items]
            state_np = np.stack([history["state"].to_numpy(np.int64) for history in histories])
            time_np = np.stack([history["t"].to_numpy(np.int64) for history in histories])
            dwell_np = np.stack([history["time_in_state"].to_numpy(np.float32) for history in histories])
            value_np = np.stack([item[2][item[4]] for item in items])
            category_np = np.stack([item[3][item[4]] for item in items])
            if history_order == "scrambled":
                state_np, value_np, category_np, dwell_np = scramble_observed_history_content(
                    state_np,
                    value_np,
                    category_np,
                    dwell_np,
                    [int(item[0]) for item in items],
                    int(landmark),
                    int(history_order_seed),
                )
            history_states = torch.as_tensor(state_np, dtype=torch.long, device=device)
            history_times = torch.as_tensor(time_np, dtype=torch.long, device=device)
            history_dwell = torch.as_tensor(dwell_np, dtype=torch.float32, device=device)
            history_values = torch.as_tensor(value_np, dtype=torch.float32, device=device)
            history_categories = torch.as_tensor(category_np, dtype=torch.long, device=device)
            categories_argument = history_categories if history_categories.shape[-1] else None
            initial_hidden, base_cache = model.initialize_incremental(
                history_states,
                history_values,
                history_times,
                categories_argument,
                dwell=history_dwell,
            )
            origins = history_states[:, -1]
            last_values = history_values[:, -1, :]
            last_categories = history_categories[:, -1, :]
            last_dwell = history_dwell[:, -1]
            occupation_sums = torch.zeros(
                (len(items), config.max_horizon, config.num_states), dtype=torch.float64
            )
            completed = 0
            while completed < n_sims:
                chunk = min(simulation_batch_size, n_sims - completed)
                hidden = initial_hidden.repeat_interleave(chunk, dim=0)
                cache = model.repeat_incremental_cache(base_cache, chunk)
                current_state = origins.repeat_interleave(chunk, dim=0)
                current_values = last_values.repeat_interleave(chunk, dim=0)
                current_categories = last_categories.repeat_interleave(chunk, dim=0)
                current_dwell = last_dwell.repeat_interleave(chunk, dim=0)
                for horizon in range(1, config.max_horizon + 1):
                    logits = model.state_head(hidden)
                    permitted = graph[current_state]
                    probability = torch.softmax(
                        logits.masked_fill(~permitted, -1e9) / state_temperature,
                        dim=-1,
                    )
                    max_simplex_error = max(
                        max_simplex_error, validate_probability(probability, "transition probability")
                    )
                    occupation_sums[:, horizon - 1, :] += (
                        probability.view(len(items), chunk, config.num_states)
                        .sum(dim=1)
                        .to(dtype=torch.float64, device="cpu")
                    )
                    next_state = torch.multinomial(
                        probability, 1, generator=generator
                    ).squeeze(1)
                    if horizon == config.max_horizon:
                        continue
                    if covariate_mode == "locf":
                        next_values = encoder.advance_locf(current_values)
                    else:
                        next_values, diagnostic = sample_joint_covariates(
                            model,
                            encoder,
                            hidden,
                            current_values,
                            next_state,
                            terminals,
                            generator,
                            observation_mode=observation_mode,
                            state_observation_rates=observation_rates_tensor,
                            continuous_mode=continuous_mode,
                        )
                        max_continuous_abs = max(
                            max_continuous_abs, diagnostic["continuous_abs_max"]
                        )
                        max_scale = max(max_scale, diagnostic["scale_max"])
                    next_dwell = torch.where(
                        next_state == current_state,
                        current_dwell + 1.0,
                        torch.zeros_like(current_dwell),
                    )
                    next_time = torch.full(
                        (len(next_state), 1),
                        landmark + horizon,
                        dtype=torch.long,
                        device=device,
                    )
                    next_categories = (
                        current_categories[:, None, :]
                        if current_categories.shape[-1]
                        else None
                    )
                    hidden, cache = model.incremental_step(
                        next_state[:, None],
                        next_values[:, None, :],
                        next_time,
                        cache,
                        next_categories,
                        dwell=next_dwell[:, None],
                        elapsed_times=torch.ones(
                            (len(next_state), 1), dtype=torch.float32, device=device
                        ),
                        visit_indices=torch.full(
                            (len(next_state), 1),
                            history_states.shape[1] + horizon - 1,
                            dtype=torch.float32,
                            device=device,
                        ),
                    )
                    current_state = next_state
                    current_values = next_values
                    current_dwell = next_dwell
                completed += chunk
                if gpu_yield_seconds:
                    time.sleep(gpu_yield_seconds)
            probabilities = occupation_sums.numpy() / n_sims
            validate_probability(
                torch.as_tensor(probabilities), "occupation probability"
            )
            for item_index, (pid, raw_group, _, _, _) in enumerate(items):
                origin = int(origins[item_index].item())
                for horizon in range(1, config.max_horizon + 1):
                    target, known, observation_time = target_at(
                        raw_group, landmark + horizon, terminals
                    )
                    row = {
                        "id": pid,
                        "fold": fold,
                        "variant": variant,
                        "n_sims": n_sims,
                        "covariate_mode": covariate_mode,
                        "observation_mode": observation_mode,
                        "continuous_mode": continuous_mode,
                        "state_temperature": float(state_temperature),
                        "history_order": history_order,
                        "history_order_seed": int(history_order_seed),
                        "history_mode": history_mode,
                        "landmark": landmark,
                        "horizon": horizon,
                        "target_time": landmark + horizon,
                        "origin_state": origin,
                        "target_state": target,
                        "known": known,
                        "observation_time": observation_time,
                    }
                    row.update({
                        f"p_state_{state}": float(probabilities[item_index, horizon - 1, state])
                        for state in range(config.num_states)
                    })
                    rows.append(row)
            pair_count += len(items)
            if pair_count % 200 < len(items):
                print(
                    f"rollout patient_landmarks={pair_count} sims={n_sims} "
                    f"elapsed_min={(time.time()-started)/60:.1f} pairs_per_sec={pair_count/max(time.time()-started,1e-6):.2f} "
                    f"simplex_error={max_simplex_error:.3g} "
                    f"max_abs_covariate={max_continuous_abs:.3g} max_scale={max_scale:.3g}",
                    flush=True,
                )
    result = pd.DataFrame(rows)
    if len(result):
        result["ipcw"] = conditional_ipcw(
            km,
            result["landmark"].to_numpy(dtype=float),
            result["observation_time"].to_numpy(dtype=float),
            result["known"].to_numpy(dtype=bool),
        )
    print(
        f"ROLLOUT PASS mode={covariate_mode} observation={observation_mode} "
        f"continuous={continuous_mode} rows={len(result)} "
        f"simplex_error={max_simplex_error:.3g} max_abs_covariate={max_continuous_abs:.3g} "
        f"max_scale={max_scale:.3g}",
        flush=True,
    )
    return result, time.time() - started
