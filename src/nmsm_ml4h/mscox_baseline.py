from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp
import torch

from .baseline_common import (
    advance_continuous,
    combine_design,
    design_names,
    encode_frame,
    nonself_edges,
)
from .config import ExperimentConfig
from .targets import KEYS
from .splits import split_train_validation
from .training import fit_feature_encoder


MSCOX_RIDGE_GRID = (
    0.0,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
)


def _cox_objective(
    beta: np.ndarray,
    risk_groups: list[np.ndarray],
    event_groups: list[np.ndarray],
    event_count: int,
    ridge: float,
) -> tuple[float, np.ndarray]:
    log_likelihood = 0.0
    gradient = np.zeros_like(beta)
    for risk_x, event_x in zip(risk_groups, event_groups):
        if not len(event_x):
            continue
        eta = risk_x @ beta
        shift = float(np.max(eta))
        weights = np.exp(eta - shift)
        denominator = float(weights.sum())
        deaths = len(event_x)
        log_likelihood += float((event_x @ beta).sum())
        log_likelihood -= deaths * (shift + np.log(denominator))
        gradient += event_x.sum(axis=0)
        gradient -= deaths * (weights[:, None] * risk_x).sum(axis=0) / denominator
    scale = max(int(event_count), 1)
    loss = -log_likelihood / scale + 0.5 * ridge * float(beta @ beta)
    grad = -gradient / scale + ridge * beta
    return float(loss), grad


def _make_cox_objective(risk_groups, event_groups, event_count, ridge):
    'Cache Breslow sufficient statistics; keep the exact same objective.'
    sizes = np.asarray([len(group) for group in risk_groups], dtype=np.int64)
    if not len(sizes) or np.any(sizes == 0):
        raise ValueError("Cox objective requires non-empty risk groups")
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    design = np.concatenate(risk_groups, axis=0)
    group_index = np.repeat(np.arange(len(sizes)), sizes)
    events = np.asarray([len(group) for group in event_groups], dtype=np.float64)
    event_sum = np.sum([group.sum(axis=0) for group in event_groups], axis=0)
    scale = max(int(event_count), 1)

    def objective(beta):
        eta = design @ beta
        shifts = np.maximum.reduceat(eta, starts)
        weights = np.exp(eta - shifts[group_index])
        denominators = np.add.reduceat(weights, starts)
        log_likelihood = beta @ event_sum - events @ (shifts + np.log(denominators))
        gradient = event_sum - design.T @ (weights * (events / denominators)[group_index])
        return (float(-log_likelihood / scale + 0.5 * ridge * (beta @ beta)),
                -gradient / scale + ridge * beta)

    return objective


def _profile_separated_axes(
    risk_groups: list[np.ndarray],
    event_groups: list[np.ndarray],
    times: np.ndarray,
) -> tuple[list[np.ndarray], list[dict], int]:
    'Return the exact extended-MLE risk sets for axis separation.'
    if not risk_groups:
        return risk_groups, [], 0
    feature_count = risk_groups[0].shape[1]
    profiled = risk_groups
    rules: list[dict] = []
    removed_total = 0
    used: set[int] = set()
    while True:
        negative = np.ones(feature_count, dtype=bool)
        positive = np.ones(feature_count, dtype=bool)
        strict = np.zeros(feature_count, dtype=bool)
        minimums: list[np.ndarray] = []
        maximums: list[np.ndarray] = []
        for risk_x, event_x in zip(profiled, event_groups):
            risk_min = risk_x.min(axis=0)
            risk_max = risk_x.max(axis=0)
            event_min = event_x.min(axis=0)
            event_max = event_x.max(axis=0)
            minimums.append(risk_min)
            maximums.append(risk_max)
            negative &= (event_min == risk_min) & (event_max == risk_min)
            positive &= (event_min == risk_max) & (event_max == risk_max)
            strict |= risk_max > risk_min
        available = np.asarray(
            [index not in used for index in range(feature_count)], dtype=bool
        )
        negative_indices = np.flatnonzero(negative & strict & available)
        positive_indices = np.flatnonzero(positive & strict & available)
        if not len(negative_indices) and not len(positive_indices):
            break
        for index in negative_indices:
            rules.append(
                {
                    "feature_index": int(index),
                    "coefficient_limit": "-infinity",
                    "zero_hazard_when": "above_boundary",
                    "boundary_by_time": {
                        int(time): float(minimums[group][index])
                        for group, time in enumerate(times)
                    },
                }
            )
            used.add(int(index))
        for index in positive_indices:
            rules.append(
                {
                    "feature_index": int(index),
                    "coefficient_limit": "+infinity",
                    "zero_hazard_when": "below_boundary",
                    "boundary_by_time": {
                        int(time): float(maximums[group][index])
                        for group, time in enumerate(times)
                    },
                }
            )
            used.add(int(index))
        next_profiled: list[np.ndarray] = []
        removed = 0
        for group, (risk_x, event_x) in enumerate(zip(profiled, event_groups)):
            keep = np.ones(len(risk_x), dtype=bool)
            for index in negative_indices:
                keep &= risk_x[:, index] == minimums[group][index]
                if not np.all(event_x[:, index] == minimums[group][index]):
                    raise AssertionError("Extended-MLE profiling would remove an event")
            for index in positive_indices:
                keep &= risk_x[:, index] == maximums[group][index]
                if not np.all(event_x[:, index] == maximums[group][index]):
                    raise AssertionError("Extended-MLE profiling would remove an event")
            next_profiled.append(risk_x[keep])
            removed += int((~keep).sum())
        if removed == 0:
            raise AssertionError("Strict separation did not remove any risk rows")
        profiled = next_profiled
        removed_total += removed
    return profiled, rules, removed_total


def _identifiable_whitening(
    risk_groups: list[np.ndarray],
    event_groups: list[np.ndarray],
    event_count: int,
) -> tuple[np.ndarray, int, float]:
    'Whiten the identifiable Cox subspace without changing its predictors.'
    feature_count = risk_groups[0].shape[1]
    information = np.zeros((feature_count, feature_count), dtype=np.float64)
    for risk_x, event_x in zip(risk_groups, event_groups):
        deaths = len(event_x)
        mean = risk_x.mean(axis=0)
        information += deaths * (
            risk_x.T @ risk_x / len(risk_x) - np.outer(mean, mean)
        ) / max(event_count, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    scale = float(max(np.max(np.abs(eigenvalues)), 1.0))
    tolerance = np.finfo(np.float64).eps * max(information.shape) * scale
    identifiable = eigenvalues > tolerance
    positive = eigenvalues[identifiable]
    if not len(positive):
        return np.empty((feature_count, 0), dtype=np.float64), 0, float("inf")


    transform = eigenvectors[:, identifiable] / np.sqrt(positive)[None, :]
    return transform, int(len(positive)), float(positive.max() / positive.min())


def _zero_hazard_mask(
    model: dict,
    design: np.ndarray,
    event_time: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(design), dtype=bool)
    times = np.asarray(event_time, dtype=np.int64)
    for rule in model.get("zero_hazard_rules", ()):
        boundaries = rule["boundary_by_time"]
        boundary = np.fromiter(
            (
                float(
                    boundaries.get(
                        int(time), boundaries.get(str(int(time)), np.nan)
                    )
                )
                for time in times
            ),
            dtype=np.float64,
            count=len(times),
        )
        active = np.isfinite(boundary)
        values = design[:, int(rule["feature_index"])]
        if rule["zero_hazard_when"] == "above_boundary":
            result |= active & (values > boundary)
        elif rule["zero_hazard_when"] == "below_boundary":
            result |= active & (values < boundary)
        else:
            raise ValueError("Unknown msCox extended-MLE zero-hazard rule")
    return result


def _fit_edge(
    design: np.ndarray,
    origin: np.ndarray,
    destination: np.ndarray,
    stop_time: np.ndarray,
    edge: tuple[int, int],
    ridge: float,
) -> dict:
    source, target = edge
    risk_mask = origin == source
    event_mask = risk_mask & (destination == target)
    event_count = int(event_mask.sum())
    feature_count = design.shape[1]
    if event_count == 0:
        return {
            "edge": edge,
            "beta": np.zeros(feature_count, dtype=np.float64),
            "log_baseline": {},
            "events": 0,
            "converged": True,
            "objective": 0.0,
            "zero_hazard_rules": [],
            "profiled_risk_rows": 0,
        }
    times = np.unique(stop_time[event_mask])
    risk_groups: list[np.ndarray] = []
    event_groups: list[np.ndarray] = []
    for event_time in times:
        risk_groups.append(design[risk_mask & (stop_time == event_time)])
        event_groups.append(design[event_mask & (stop_time == event_time)])
    zero_hazard_rules: list[dict] = []
    profiled_risk_rows = 0
    information_rank = feature_count
    information_condition = 1.0
    if ridge == 0.0:
        fit_risk_groups, zero_hazard_rules, profiled_risk_rows = (
            _profile_separated_axes(risk_groups, event_groups, times)
        )
        transform, information_rank, information_condition = _identifiable_whitening(
            fit_risk_groups, event_groups, event_count
        )

        initial = np.zeros(information_rank, dtype=np.float64)
    else:
        fit_risk_groups = risk_groups
        transform = np.eye(feature_count, dtype=np.float64)
        initial = np.zeros(feature_count, dtype=np.float64)
    cached_objective = _make_cox_objective(fit_risk_groups, event_groups, event_count, ridge)

    def objective(theta):
        loss, gradient = cached_objective(transform @ theta)
        return loss, transform.T @ gradient
    if information_rank == 0:
        value, gradient = objective(initial)
        result = OptimizeResult(
            x=initial,
            fun=value,
            jac=gradient,
            success=True,
            status=0,
            message="All Cox directions are non-identifiable after exact profiling",
            nit=0,
            nfev=1,
        )
    else:
        result = minimize(
            fun=objective,
            x0=initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-7},
        )


    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        result = minimize(
            fun=objective,
            x0=np.asarray(result.x, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
        )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(
            f"msCox failed to converge for edge {source}->{target}: {result.message}"
        )
    beta = transform @ np.asarray(result.x, dtype=np.float64)
    if not np.isfinite(beta).all():
        raise RuntimeError(f"msCox produced non-finite coefficients for edge {edge}")
    log_baseline: dict[int, float] = {}
    for event_time, risk_x, event_x in zip(times, fit_risk_groups, event_groups):
        value = float(np.log(len(event_x)) - logsumexp(risk_x @ beta))
        if not np.isfinite(value):
            raise FloatingPointError(f"msCox non-finite Breslow log increment for edge {edge}")
        log_baseline[int(event_time)] = value
    return {
        "edge": edge,
        "beta": beta,
        "log_baseline": log_baseline,
        "events": event_count,
        "converged": bool(result.success),
        "objective": float(result.fun),
        "message": str(result.message),
        "iterations": int(result.nit),
        "zero_hazard_rules": zero_hazard_rules,
        "profiled_risk_rows": profiled_risk_rows,
        "information_rank": information_rank,
        "information_nullity": feature_count - information_rank,
        "information_condition_before_whitening": information_condition,
    }


def _transition_arrays(encoded, experiment: ExperimentConfig) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rows = encoded.frame
    raw_next = pd.to_numeric(rows["next_state"], errors="coerce")
    use = (
        raw_next.notna()
        & (raw_next >= 0)
        & ~rows["state"].isin(experiment.terminal_states)
    ).to_numpy(dtype=bool)
    return (
        encoded.design[use],
        rows.loc[use, "state"].to_numpy(dtype=np.int64),
        raw_next.loc[use].to_numpy(dtype=np.int64),
        rows.loc[use, "t"].to_numpy(dtype=np.int64) + 1,
    )


def _fit_transition_models(
    design: np.ndarray,
    origin: np.ndarray,
    destination: np.ndarray,
    stop_time: np.ndarray,
    permitted: set[tuple[int, int]],
    ridge: float,
) -> dict[tuple[int, int], dict]:
    return {
        edge: _fit_edge(
            design, origin, destination, stop_time, edge, ridge
        )
        for edge in sorted(permitted)
    }


def _training_aj_validity(
    fitted: dict[tuple[int, int], dict],
    design: np.ndarray,
    origin: np.ndarray,
    stop_time: np.ndarray,
    num_states: int,
) -> tuple[bool, float, str]:
    'Check observed training risk rows without clipping hazard jumps.'
    maximum_jump = 0.0
    for source in range(num_states):
        models = [
            (target, model)
            for (local_source, target), model in fitted.items()
            if local_source == source
        ]
        if not models:
            continue
        selected = origin == source
        source_design = design[selected]
        source_time = stop_time[selected]
        event_times = sorted(
            {
                int(time)
                for _, model in models
                for time in model["log_baseline"]
            }
        )
        for event_time in event_times:
            risk_x = source_design[source_time == event_time]
            if not len(risk_x):
                continue
            total = np.zeros(len(risk_x), dtype=np.float64)
            for _, model in models:
                log_increment = model["log_baseline"].get(event_time)
                if log_increment is None:
                    continue
                log_hazard = float(log_increment) + risk_x @ model["beta"]
                log_hazard[
                    _zero_hazard_mask(
                        model,
                        risk_x,
                        np.full(len(risk_x), event_time, dtype=np.int64),
                    )
                ] = -np.inf
                if np.any(log_hazard > np.log(np.finfo(np.float64).max)):
                    return False, float("inf"), "non-finite training hazard"
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    hazard = np.exp(log_hazard)
                if not np.isfinite(hazard).all():
                    return False, float("inf"), "non-finite training hazard"
                total += hazard
            local_maximum = float(total.max(initial=0.0))
            maximum_jump = max(maximum_jump, local_maximum)
            if local_maximum > 1.0 + 1e-12:
                return (
                    False,
                    maximum_jump,
                    f"negative training AJ diagonal at state={source} time={event_time}",
                )
    return True, maximum_jump, "valid"


def _target_probabilities(
    fitted: dict[tuple[int, int], dict],
    encoded,
    encoder,
    targets: pd.DataFrame,
    experiment: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    ordered_targets = targets.sort_values(KEYS).reset_index(drop=True)
    pairs = (
        ordered_targets[["id", "landmark", "origin_state"]]
        .drop_duplicates(["id", "landmark"])
        .sort_values(["id", "landmark"])
        .reset_index(drop=True)
    )
    encoded_indices = np.asarray(
        [
            encoded.lookup[(int(row.id), int(row.landmark))]
            for row in pairs.itertuples(index=False)
        ],
        dtype=np.int64,
    )
    continuous = encoded.continuous[encoded_indices].copy()
    categories = encoded.categories[encoded_indices]
    state_probability = np.eye(experiment.num_states, dtype=np.float64)[
        pairs["origin_state"].to_numpy(dtype=np.int64)
    ]
    occupation = np.empty(
        (len(pairs), experiment.max_horizon, experiment.num_states),
        dtype=np.float64,
    )
    landmarks = pairs["landmark"].to_numpy(dtype=np.int64)
    for horizon in range(1, experiment.max_horizon + 1):
        current_design = combine_design(continuous, categories, encoder)
        transition = _prediction_matrices(
            fitted,
            current_design,
            landmarks + horizon,
            experiment.num_states,
            device,
        )
        state_probability = np.einsum(
            "bi,bij->bj", state_probability, transition, optimize=True
        )
        occupation[:, horizon - 1] = state_probability
        continuous = _advance_continuous_batch(continuous, encoder)
    pair_lookup = {
        (int(row.id), int(row.landmark)): index
        for index, row in enumerate(pairs.itertuples(index=False))
    }
    pair_index = np.fromiter(
        (
            pair_lookup[(int(row.id), int(row.landmark))]
            for row in ordered_targets.itertuples(index=False)
        ),
        dtype=np.int64,
        count=len(ordered_targets),
    )
    probability = occupation[
        pair_index,
        ordered_targets["horizon"].to_numpy(dtype=np.int64) - 1,
    ]
    if not np.isfinite(probability).all():
        raise FloatingPointError("msCox produced non-finite state probabilities")
    if np.min(probability) < -1e-10 or np.max(probability) > 1.0 + 1e-10:
        raise FloatingPointError("msCox state probabilities left the simplex")
    return probability


def _validation_partial_likelihood(
    fitted: dict[tuple[int, int], dict],
    design: np.ndarray,
    origin: np.ndarray,
    destination: np.ndarray,
    stop_time: np.ndarray,
) -> tuple[float, int]:
    'Return event-weighted held-out Cox NLL from validation risk sets.'
    negative_log_likelihood = 0.0
    total_events = 0
    for edge, model in sorted(fitted.items()):
        source, target = edge
        risk_mask = origin == source
        event_mask = risk_mask & (destination == target)
        event_times = np.unique(stop_time[event_mask])
        for event_time in event_times:
            risk_x = design[risk_mask & (stop_time == event_time)]
            event_x = design[event_mask & (stop_time == event_time)]
            deaths = len(event_x)
            if not deaths:
                continue
            for rule in model.get("zero_hazard_rules", ()):
                boundaries = rule["boundary_by_time"]
                if (
                    int(event_time) not in boundaries
                    and str(int(event_time)) not in boundaries
                ):
                    raise FloatingPointError(
                        "extended msCox MLE is not scorable at an unseen "
                        f"validation event time for edge {source}->{target}"
                    )
            risk_zero = _zero_hazard_mask(
                model,
                risk_x,
                np.full(len(risk_x), event_time, dtype=np.int64),
            )
            event_zero = _zero_hazard_mask(
                model,
                event_x,
                np.full(deaths, event_time, dtype=np.int64),
            )
            if event_zero.any():
                raise FloatingPointError(
                    f"msCox assigns zero hazard to validation event {source}->{target}"
                )
            risk_x = risk_x[~risk_zero]
            if not len(risk_x):
                raise FloatingPointError(
                    f"msCox has an empty validation risk set for edge {source}->{target}"
                )
            beta = np.asarray(model["beta"], dtype=np.float64)
            risk_eta = risk_x @ beta
            event_eta = event_x @ beta
            if not np.isfinite(risk_eta).all() or not np.isfinite(event_eta).all():
                raise FloatingPointError("msCox validation linear predictor is non-finite")
            negative_log_likelihood += float(
                deaths * logsumexp(risk_eta) - event_eta.sum()
            )
            total_events += deaths
    if total_events == 0:
        raise RuntimeError("msCox inner validation has no transition events")
    score = negative_log_likelihood / total_events
    if not np.isfinite(score):
        raise FloatingPointError("msCox validation partial likelihood is non-finite")
    return float(score), int(total_events)


def _choose_ridge(audits: list[dict]) -> float:
    valid = [audit for audit in audits if audit["status"] == "valid"]
    if not valid:
        raise RuntimeError("msCox ridge selection found no AJ-valid candidate")
    selected = min(
        valid,
        key=lambda audit: (
            float(audit["inner_validation_partial_nll"]),
            float(audit["ridge"]),
        ),
    )
    return float(selected["ridge"])


def _select_ridge(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    targets: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
    permitted: set[tuple[int, int]],
    candidates: tuple[float, ...],
    seed: int,
    device: torch.device,
) -> tuple[float, dict]:
    outer_patients = folds.loc[folds["outer_fold"] != fold].copy()
    inner_ids, validation_ids = split_train_validation(
        outer_patients, seed=seed + fold
    )
    inner = frame[frame["id"].isin(set(map(int, inner_ids)))].copy()
    validation = frame[
        frame["id"].isin(set(map(int, validation_ids)))
    ].copy()
    encoder = fit_feature_encoder(inner, experiment)
    encoded_inner = encode_frame(inner, encoder)
    encoded_validation = encode_frame(validation, encoder)
    inner_design, inner_origin, inner_destination, inner_stop = _transition_arrays(
        encoded_inner, experiment
    )
    (
        validation_design,
        validation_origin,
        validation_destination,
        validation_stop,
    ) = _transition_arrays(encoded_validation, experiment)
    validation_targets = targets[
        targets["id"].isin(set(map(int, validation_ids)))
    ].copy()
    if validation_targets.empty:
        raise RuntimeError("msCox inner validation has no target rows")
    ordered_validation = validation_targets.sort_values(KEYS).reset_index(drop=True)
    audits: list[dict] = []
    for candidate in candidates:
        audit = {"ridge": float(candidate), "status": "rejected"}
        try:
            fitted = _fit_transition_models(
                inner_design,
                inner_origin,
                inner_destination,
                inner_stop,
                permitted,
                float(candidate),
            )
            valid, maximum_jump, reason = _training_aj_validity(
                fitted,
                inner_design,
                inner_origin,
                inner_stop,
                experiment.num_states,
            )
            audit["inner_training_max_aj_jump"] = maximum_jump
            if not valid:
                raise FloatingPointError(reason)
            probability = _target_probabilities(
                fitted,
                encoded_validation,
                encoder,
                ordered_validation,
                experiment,
                device,
            )
            partial_nll, validation_events = _validation_partial_likelihood(
                fitted,
                validation_design,
                validation_origin,
                validation_destination,
                validation_stop,
            )
            audit.update(
                {
                    "status": "valid",
                    "inner_validation_partial_nll": partial_nll,
                    "inner_validation_events": validation_events,
                }
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            audit["reason"] = str(error)
        audits.append(audit)
        print(
            f"mscox fold={fold} ridge_candidate={candidate:.4g} "
            f"status={audit['status']} "
            f"partial_nll={audit.get('inner_validation_partial_nll', float('nan')):.7f} "
            f"reason={audit.get('reason', '')}",
            flush=True,
        )
    selected_ridge = _choose_ridge(audits)
    selection = {
        "protocol": (
            "predeclared event-normalized ridge grid; reject optimizer/non-finite/"
            "negative-AJ candidates on inner training or validation; minimize "
            "event-weighted inner-validation transition Cox partial NLL"
        ),
        "grid": [float(value) for value in candidates],
        "seed": int(seed + fold),
        "inner_training_patients": int(len(inner_ids)),
        "inner_validation_patients": int(len(validation_ids)),
        "candidates": audits,
        "selected_ridge": selected_ridge,
    }
    del (
        encoded_inner,
        encoded_validation,
        inner_design,
        validation_design,
        inner,
        validation,
    )
    gc.collect()
    return selected_ridge, selection


def _prediction_matrix(
    fitted: dict[tuple[int, int], dict],
    design: np.ndarray,
    event_time: int,
    num_states: int,
    prediction_method: str = "product_integral",
) -> np.ndarray:
    generator = np.zeros((num_states, num_states), dtype=np.float64)
    for (origin, destination), model in fitted.items():
        if "log_baseline" in model:
            log_increment = model["log_baseline"].get(int(event_time))
        else:
            increment = float(model["baseline"].get(int(event_time), 0.0))
            if increment < 0 or not np.isfinite(increment):
                raise FloatingPointError("msCox baseline increment is invalid")
            log_increment = None if increment == 0.0 else float(np.log(increment))
        if log_increment is None:
            continue
        if _zero_hazard_mask(
            model,
            np.asarray(design, dtype=np.float64)[None, :],
            np.asarray([event_time], dtype=np.int64),
        )[0]:
            continue
        log_hazard = float(log_increment + design @ model["beta"])
        with np.errstate(over="raise", invalid="raise"):
            try:
                hazard = float(np.exp(log_hazard))
            except FloatingPointError as error:
                raise FloatingPointError(
                    "msCox predicted hazard overflow; refusing to clip the linear predictor"
                ) from error
        if not np.isfinite(hazard):
            raise FloatingPointError("msCox predicted a non-finite hazard increment")
        generator[origin, destination] = hazard
    for state in range(num_states):
        generator[state, state] = -float(generator[state].sum())
    if prediction_method == "matrix_exponential":
        return _exponential_transition(generator[None])[0]
    if prediction_method != "product_integral":
        raise ValueError("Unknown msCox prediction method")


    transition = np.eye(num_states, dtype=np.float64) + generator
    if np.min(transition) < -1e-12:
        raise FloatingPointError(
            "msCox Aalen-Johansen factor I+dA has a negative entry; "
            "refusing to replace the published product-integral update with "
            "an artificial sub-jump or matrix-exponential approximation"
        )
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise FloatingPointError("msCox Aalen-Johansen factor does not conserve mass")
    return transition


def _prediction_matrices(
    fitted: dict[tuple[int, int], dict],
    design: np.ndarray,
    event_time: np.ndarray,
    num_states: int,
    device: torch.device,
    batch_size: int = 4096,
    prediction_method: str = "product_integral",
) -> np.ndarray:
    'Batched equivalent of :func:`_prediction_matrix`.'
    result = np.empty((len(design), num_states, num_states), dtype=np.float64)
    times = np.asarray(event_time, dtype=np.int64)
    for start in range(0, len(design), batch_size):
        stop = min(start + batch_size, len(design))
        local_design = design[start:stop]
        local_times = times[start:stop]
        generator = np.zeros((stop - start, num_states, num_states), dtype=np.float64)
        for (origin, destination), model in fitted.items():
            if "log_baseline" in model:
                log_increments = np.fromiter(
                    (float(model["log_baseline"].get(int(value), -np.inf)) for value in local_times),
                    dtype=np.float64,
                    count=len(local_times),
                )
            else:
                increments = np.fromiter(
                    (float(model["baseline"].get(int(value), 0.0)) for value in local_times),
                    dtype=np.float64,
                    count=len(local_times),
                )
                if np.any(increments < 0) or not np.isfinite(increments).all():
                    raise FloatingPointError("msCox baseline increments are invalid")
                log_increments = np.full_like(increments, -np.inf)
                positive = increments > 0
                log_increments[positive] = np.log(increments[positive])
            log_hazard = log_increments + local_design @ model["beta"]
            log_hazard[
                _zero_hazard_mask(model, local_design, local_times)
            ] = -np.inf
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                try:
                    hazards = np.exp(log_hazard)
                except FloatingPointError as error:
                    raise FloatingPointError(
                        "msCox predicted hazard overflow; refusing to clip the linear predictor"
                    ) from error
            if not np.isfinite(hazards).all():
                raise FloatingPointError("msCox predicted non-finite hazard increments")
            generator[:, origin, destination] = hazards
        diagonal = -generator.sum(axis=2)
        index = np.arange(num_states)
        generator[:, index, index] = diagonal
        if prediction_method == "matrix_exponential":
            result[start:stop] = _exponential_transition(generator)
            continue
        if prediction_method != "product_integral":
            raise ValueError("Unknown msCox prediction method")
        transition = np.eye(num_states, dtype=np.float64)[None, :, :] + generator
        if np.min(transition) < -1e-12:
            raise FloatingPointError(
                "msCox Aalen-Johansen factor I+dA has a negative entry; "
                "refusing to alter the published product-integral estimator"
            )
        if not np.allclose(
            transition.sum(axis=2), 1.0, atol=1e-12, rtol=0.0
        ):
            raise FloatingPointError("msCox Aalen-Johansen factors do not conserve mass")
        result[start:stop] = transition
    return result


def _exponential_transition(generator):
    'Explicit Cox stype=2 prediction, not a clipped Aalen--Johansen factor.'
    if not np.isfinite(generator).all():
        raise FloatingPointError("Non-finite Cox generator")
    transition = torch.matrix_exp(torch.from_numpy(generator)).numpy()
    if (not np.isfinite(transition).all() or transition.min() < -1e-10
            or not np.allclose(transition.sum(axis=-1), 1.0, atol=1e-10, rtol=0)):
        raise FloatingPointError("Cox matrix exponential failed probability validation")
    return transition


def _advance_continuous_batch(
    continuous: np.ndarray, encoder
) -> np.ndarray:
    output = np.asarray(continuous, dtype=np.float64).copy()
    encoder._check_fitted()
    assert encoder.stds is not None
    for column, raw_increment in encoder.known_dynamic_raw_increments:
        index = encoder.numeric_columns.index(column)
        output[:, index] += float(raw_increment) / float(encoder.stds.loc[column])
    if not encoder.include_missingness:
        return output
    n_numeric = len(encoder.numeric_columns)
    dynamic = set(encoder.observation_numeric_columns)
    for index, column in enumerate(encoder.numeric_columns):
        if column in dynamic:
            output[:, n_numeric + index] = 0.0
    if encoder.observation_numeric_columns:
        delta_start = 2 * n_numeric
        output[:, delta_start:] = np.minimum(
            1.0,
            output[:, delta_start:] + 1.0 / max(float(encoder.max_delta), 1.0),
        )
    return output


def run_mscox_fold(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    targets: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
    output_dir: Path,
    ridge: float | None = None,
    ridge_grid: tuple[float, ...] = MSCOX_RIDGE_GRID,
    selection_seed: int = 2026,
    device: torch.device | None = None,
    prediction_method: str = "product_integral",
    resume_edges: bool = False,
) -> pd.DataFrame:
    started = time.time()
    device = device or torch.device("cpu")
    permitted = set(nonself_edges(experiment))
    if prediction_method not in {"product_integral", "matrix_exponential"}:
        raise ValueError("Unknown msCox prediction method")
    if prediction_method == "matrix_exponential" and ridge is None:
        raise ValueError("Explicit matrix-exponential Cox requires a fixed ridge, not AJ-based selection")
    print(f"mscox fold={fold} PREPARE prediction={prediction_method} ridge={ridge}", flush=True)
    if ridge is None:
        selected_ridge, ridge_selection = _select_ridge(
            frame,
            folds,
            targets,
            fold,
            experiment,
            permitted,
            ridge_grid,
            selection_seed,
            device,
        )
    else:
        selected_ridge = float(ridge)
        ridge_selection = {
            "protocol": "fixed ridge supplied explicitly",
            "grid": [selected_ridge],
            "selected_ridge": selected_ridge,
        }
    print(
        f"mscox fold={fold} selected_ridge={selected_ridge:.4g}", flush=True
    )
    training_ids = set(map(int, folds.loc[folds["outer_fold"] != fold, "id"]))
    training = frame[frame["id"].isin(training_ids)].copy()


    encoder = fit_feature_encoder(training, experiment)
    print(f"mscox fold={fold} ENCODE patients={len(training_ids)} rows={len(training)}", flush=True)
    encoded_training = encode_frame(training, encoder)
    design, origin, destination, stop_time = _transition_arrays(
        encoded_training, experiment
    )
    observed_edges = set(
        zip(origin[origin != destination].tolist(), destination[origin != destination].tolist())
    )
    invalid = sorted(observed_edges.difference(permitted))
    if invalid:
        raise ValueError(f"msCox found transitions outside the frozen graph: {invalid}")
    fitted: dict[tuple[int, int], dict] = {}
    feature_names = design_names(encoder)
    cache = output_dir / "edge_checkpoints_INTERNAL"
    if resume_edges:
        cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        for array in (design, origin, destination, stop_time):
            digest.update(np.ascontiguousarray(array).tobytes())

        digest.update(Path(__file__).read_bytes())
        digest.update(json.dumps([selected_ridge, sorted(permitted), feature_names]).encode())
        contract = cache / "contract.json"
        signature = digest.hexdigest()
        if contract.exists() and json.loads(contract.read_text())["signature"] != signature:
            raise ValueError("msCox edge checkpoint contract changed; use a fresh run directory")
        contract.write_text(json.dumps({"signature": signature}), encoding="utf-8")
    for edge in sorted(permitted):
        edge_path = cache / f"edge_{edge[0]}_{edge[1]}.json"
        if resume_edges and edge_path.exists():
            restored = json.loads(edge_path.read_text())
            restored["edge"] = edge
            restored["beta"] = np.asarray(restored["beta"], dtype=np.float64)
            restored["log_baseline"] = {int(key): value for key, value in restored["log_baseline"].items()}
            for rule in restored.get("zero_hazard_rules", ()):
                rule["feature_name"] = feature_names[int(rule["feature_index"])]
            fitted[edge] = restored
            print(f"mscox fold={fold} edge={edge[0]}->{edge[1]} REUSE completed fit", flush=True)
            continue
        edge_started = time.time()
        print(f"mscox fold={fold} edge={edge[0]}->{edge[1]} START", flush=True)
        fitted[edge] = _fit_edge(
            design,
            origin,
            destination,
            stop_time,
            edge,
            selected_ridge,
        )
        info = fitted[edge]
        if resume_edges:
            saved = {**info, "beta": info["beta"].tolist()}
            pending = edge_path.with_suffix(".tmp")
            pending.write_text(json.dumps(saved), encoding="utf-8")
            pending.replace(edge_path)
        for rule in info.get("zero_hazard_rules", ()):
            rule["feature_name"] = feature_names[int(rule["feature_index"])]
        print(
            f"mscox fold={fold} edge={edge[0]}->{edge[1]} DONE "
            f"events={info['events']} converged={info['converged']} "
            f"objective={info['objective']:.7f} extended_rules={len(info.get('zero_hazard_rules', ()))} "
            f"rank={info.get('information_rank', design.shape[1])} seconds={time.time()-edge_started:.1f}",
            flush=True,
        )

    outer_valid, outer_maximum_jump, outer_reason = _training_aj_validity(
        fitted,
        design,
        origin,
        stop_time,
        experiment.num_states,
    )
    if not outer_valid and prediction_method == "product_integral":
        raise FloatingPointError(
            "msCox selected ridge is AJ-invalid after outer-training refit: "
            f"{outer_reason}"
        )
    if prediction_method == "matrix_exponential":
        print(f"mscox fold={fold} AJ_DIAGNOSTIC valid={outer_valid} max_jump={outer_maximum_jump:.6g}; "
              "using predeclared matrix-exponential prediction", flush=True)

    test = targets[targets["fold"] == fold].sort_values(KEYS).copy()
    encoded_all = encode_frame(frame[frame["id"].isin(test["id"].unique())], encoder)
    pairs = (
        test[["id", "landmark", "origin_state"]]
        .drop_duplicates(["id", "landmark"])
        .sort_values(["id", "landmark"])
        .reset_index(drop=True)
    )
    encoded_indices = np.asarray(
        [
            encoded_all.lookup[(int(row.id), int(row.landmark))]
            for row in pairs.itertuples(index=False)
        ],
        dtype=np.int64,
    )
    continuous = encoded_all.continuous[encoded_indices].copy()
    categories = encoded_all.categories[encoded_indices]
    state_probability = np.eye(experiment.num_states, dtype=np.float64)[
        pairs["origin_state"].to_numpy(dtype=np.int64)
    ]
    occupation = np.empty(
        (len(pairs), experiment.max_horizon, experiment.num_states),
        dtype=np.float64,
    )
    landmarks = pairs["landmark"].to_numpy(dtype=np.int64)
    for horizon in range(1, experiment.max_horizon + 1):
        print(
            f"mscox fold={fold} prediction_horizon={horizon}/{experiment.max_horizon}",
            flush=True,
        )
        current_design = combine_design(continuous, categories, encoder)
        transition = _prediction_matrices(
            fitted,
            current_design,
            landmarks + horizon,
            experiment.num_states,
            device,
            prediction_method=prediction_method,
        )
        state_probability = np.einsum(
            "bi,bij->bj", state_probability, transition, optimize=True
        )
        occupation[:, horizon - 1] = state_probability
        continuous = _advance_continuous_batch(continuous, encoder)
    pair_lookup = {
        (int(row.id), int(row.landmark)): index
        for index, row in enumerate(pairs.itertuples(index=False))
    }
    pair_index = np.fromiter(
        (
            pair_lookup[(int(row.id), int(row.landmark))]
            for row in test.itertuples(index=False)
        ),
        dtype=np.int64,
        count=len(test),
    )
    probability = occupation[
        pair_index, test["horizon"].to_numpy(dtype=np.int64) - 1
    ]

    predictions = test[KEYS].reset_index(drop=True)
    for state in range(experiment.num_states):
        predictions[f"p_state_{state}"] = probability[:, state]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    serializable = {
        f"{origin}->{destination}": {
            **{key: value for key, value in model.items() if key not in {"edge", "beta", "log_baseline"}},
            "coefficients": dict(zip(feature_names, model["beta"].tolist())),
            "log_baseline_hazard_increment": model["log_baseline"],
        }
        for (origin, destination), model in fitted.items()
    }
    (output_dir / "training.json").write_text(
        json.dumps(
            {
                "method": (
                    "ridge-regularized transition-specific multi-state Cox with "
                    "Breslow baseline hazards"
                ),
                "reference_implementation": ("R survival::survfit.coxph, stype=2"
                    if prediction_method == "matrix_exponential" else "mstate R package"),
                "reference_url": ("https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.coxph.html"
                    if prediction_method == "matrix_exponential" else "https://github.com/hputter/mstate"),
                "implementation_status": "independent Python implementation; see explicit prediction and ridge settings",
                "implementation_note": (
                    "Transition-specific Cox/Breslow fitting. Prediction method and regularisation "
                    "are explicitly recorded; no clipping or silent prediction fallback. "
                    "Matrix-exponential prediction is not the direct Aalen-Johansen estimator."
                ),
                "time_scale": "discrete clock-forward time since cohort entry",
                "time_unit": {
                    "mimic": "hour",
                    "simulation": "simulation step",
                    "sail": "quarter",
                }.get(experiment.name, "dataset step"),
                "ridge": selected_ridge,
                "ridge_selection": ridge_selection,
                "outer_training_max_aj_jump": outer_maximum_jump,
                "prediction_method": prediction_method,
                "prediction_update": ("P <- P exp(dA)" if prediction_method == "matrix_exponential"
                                      else "Aalen-Johansen product integral P <- P(I + dA)"),
                "array_device": str(device),
                "training_patients": len(training_ids),
                "runtime_seconds": time.time() - started,
                "feature_encoder": encoder.state_dict(),
                "transition_models": serializable,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return predictions
