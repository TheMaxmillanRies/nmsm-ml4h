from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import attach_targets, probability_columns, probability_semantics
from .targets import TargetProtocol


def stratified_patient_bootstrap_counts(
    patient_folds: np.ndarray, replicates: int, seed: int
) -> np.ndarray:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    patient_folds = np.asarray(patient_folds, dtype=int)
    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates, len(patient_folds)), dtype=np.int16)
    for fold in np.unique(patient_folds):
        indices = np.flatnonzero(patient_folds == fold)
        for replicate in range(replicates):
            selected = rng.choice(indices, size=len(indices), replace=True)
            counts[replicate] += np.bincount(selected, minlength=len(patient_folds)).astype(np.int16)
    return counts


def _integrate_cells(values: np.ndarray, protocol: TargetProtocol) -> np.ndarray:
    expected_cells = len(protocol.landmarks) * protocol.max_horizon
    if values.shape[-1] != expected_cells:
        raise ValueError("Cell matrix does not match the frozen landmark/horizon grid")
    reshaped = values.reshape(values.shape[0], len(protocol.landmarks), protocol.max_horizon)
    if protocol.max_horizon == 1:
        within_landmark = reshaped[..., 0]
    else:
        within_landmark = np.trapezoid(
            reshaped, x=np.arange(1, protocol.max_horizon + 1), axis=-1
        ) / (protocol.max_horizon - 1)
    return np.mean(within_landmark, axis=1)


def patient_cell_score_matrices(
    joined: pd.DataFrame, protocol: TargetProtocol
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    patient_table = joined[["id", "fold"]].drop_duplicates().sort_values(["fold", "id"]).reset_index(drop=True)
    patient_lookup = pd.Series(np.arange(len(patient_table)), index=patient_table["id"])
    patient_index = joined["id"].map(patient_lookup).to_numpy(dtype=int)
    landmark_lookup = {landmark: index for index, landmark in enumerate(protocol.landmarks)}
    landmark_index = joined["landmark"].map(landmark_lookup).to_numpy(dtype=int)
    cell_index = landmark_index * protocol.max_horizon + joined["horizon"].to_numpy(dtype=int) - 1
    cells = len(protocol.landmarks) * protocol.max_horizon
    flat_index = patient_index * cells + cell_index
    weights = joined["ipcw"].to_numpy(dtype=float)
    target = joined["target_state"].to_numpy(dtype=int)
    known = joined["known"].to_numpy(dtype=bool)
    valid = known & (target >= 0) & (weights > 0)
    pcols = probability_columns(protocol.num_states)
    probability = joined[pcols].to_numpy(dtype=float)
    one_hot = np.eye(protocol.num_states)[np.maximum(target, 0)]
    selected_probability = probability[np.arange(len(joined)), np.maximum(target, 0)]
    losses = {
        "ibs": np.square(probability - one_hot).mean(axis=1),
        "integrated_multiclass_brier": np.square(probability - one_hot).sum(axis=1),
        "integrated_mae": np.abs(probability - one_hot).mean(axis=1),
    }
    if probability_semantics(joined) == "categorical":
        losses["integrated_nll"] = -np.log(
            np.clip(selected_probability, 1e-12, 1.0)
        )
    denominator = np.bincount(
        flat_index[valid], weights=weights[valid], minlength=len(patient_table) * cells
    ).reshape(len(patient_table), cells)
    numerators = {
        name: np.bincount(
            flat_index[valid],
            weights=weights[valid] * loss[valid],
            minlength=len(patient_table) * cells,
        ).reshape(len(patient_table), cells)
        for name, loss in losses.items()
    }
    return patient_index, denominator, numerators, patient_table


def bootstrap_integrated_scores(
    joined: pd.DataFrame,
    protocol: TargetProtocol,
    counts: np.ndarray,
    expected_patients: pd.DataFrame | None = None,
    block_size: int = 50,
) -> dict[str, np.ndarray]:
    _, denominator, numerators, patients = patient_cell_score_matrices(joined, protocol)
    if expected_patients is not None and not patients.equals(expected_patients):
        raise ValueError("Paired models do not contain the same eligible patients/folds")
    output = {name: np.empty(len(counts), dtype=float) for name in numerators}
    for start in range(0, len(counts), block_size):
        stop = min(start + block_size, len(counts))
        block = counts[start:stop].astype(float)
        denominator_total = block @ denominator
        for name, numerator in numerators.items():
            numerator_total = block @ numerator
            cell_score = np.divide(
                numerator_total,
                denominator_total,
                out=np.full_like(numerator_total, np.nan),
                where=denominator_total > 0,
            )
            output[name][start:stop] = _integrate_cells(cell_score, protocol)
    return output


def _binary_auc_influence(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    'Weighted AUROC and its observation-weight influence contributions.'
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    influence = np.zeros(len(labels), dtype=np.float64)
    use = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    selected = np.flatnonzero(use)
    if not len(selected):
        return float("nan"), influence
    local_labels = labels[selected]
    local_scores = scores[selected]
    local_weights = weights[selected]
    case_total = float(local_weights[local_labels].sum())
    control_total = float(local_weights[~local_labels].sum())
    if case_total <= 0 or control_total <= 0:
        return float("nan"), influence
    order = np.argsort(local_scores, kind="mergesort")
    sorted_labels = local_labels[order]
    sorted_scores = local_scores[order]
    sorted_weights = local_weights[order]
    group = np.cumsum(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]) - 1
    cases = np.bincount(group, weights=sorted_weights * sorted_labels)
    controls = np.bincount(group, weights=sorted_weights * (~sorted_labels))
    controls_before = np.cumsum(controls) - controls
    cases_after = case_total - np.cumsum(cases)
    numerator = float(np.sum(cases * (controls_before + 0.5 * controls)))
    auc = numerator / (case_total * control_total)
    local_influence = np.empty(len(order), dtype=np.float64)
    case_value = controls_before[group] + 0.5 * controls[group]
    control_value = cases_after[group] + 0.5 * cases[group]
    local_influence[sorted_labels] = (
        sorted_weights[sorted_labels]
        / case_total
        * (case_value[sorted_labels] / control_total - auc)
    )
    local_influence[~sorted_labels] = (
        sorted_weights[~sorted_labels]
        / control_total
        * (control_value[~sorted_labels] / case_total - auc)
    )
    unsorted = np.empty_like(local_influence)
    unsorted[order] = local_influence
    influence[selected] = unsorted
    return float(auc), influence


def integrated_auc_influence(
    joined: pd.DataFrame,
    protocol: TargetProtocol,
    expected_patients: pd.DataFrame | None = None,
) -> tuple[float, np.ndarray, pd.DataFrame]:
    'Integrated macro AUROC and patient-cluster influence contributions.'
    patients = (
        joined[["id", "fold"]]
        .drop_duplicates()
        .sort_values(["fold", "id"])
        .reset_index(drop=True)
    )
    if expected_patients is not None and not patients.equals(expected_patients):
        raise ValueError("Paired models do not contain the same eligible patients/folds")
    patient_lookup = pd.Series(np.arange(len(patients)), index=patients["id"])
    landmarks = {value: index for index, value in enumerate(protocol.landmarks)}
    point = np.full((len(protocol.landmarks), protocol.max_horizon), np.nan)
    cluster = np.zeros(
        (len(protocol.landmarks), protocol.max_horizon, len(patients)),
        dtype=np.float64,
    )
    for (landmark, horizon), group in joined.groupby(
        ["landmark", "horizon"], sort=True
    ):
        target = group["target_state"].to_numpy(dtype=int)
        weight = group["ipcw"].to_numpy(dtype=float)
        valid = group["known"].to_numpy(dtype=bool) & (target >= 0) & (weight > 0)
        patient_index = group["id"].map(patient_lookup).to_numpy(dtype=int)
        state_points = []
        state_influences = []
        for state in range(protocol.num_states):
            auc, row_influence = _binary_auc_influence(
                target == state,
                group[f"p_state_{state}"].to_numpy(dtype=float),
                np.where(valid, weight, 0.0),
            )
            if not np.isfinite(auc):
                continue
            state_points.append(auc)
            state_influences.append(
                np.bincount(
                    patient_index,
                    weights=row_influence,
                    minlength=len(patients),
                )
            )
        if not state_points:
            continue
        landmark_index = landmarks[int(landmark)]
        horizon_index = int(horizon) - 1
        point[landmark_index, horizon_index] = float(np.mean(state_points))
        cluster[landmark_index, horizon_index] = np.mean(state_influences, axis=0)

    integrated_point = 0.0
    integrated_influence = np.zeros(len(patients), dtype=np.float64)
    valid_landmarks = 0
    for landmark_index in range(len(protocol.landmarks)):
        use = np.flatnonzero(np.isfinite(point[landmark_index]))
        if not len(use):
            continue
        valid_landmarks += 1
        if len(use) == 1:
            coefficients = np.array([1.0])
        else:
            x = use.astype(float) + 1.0
            widths = np.diff(x)
            coefficients = np.zeros(len(use), dtype=float)
            coefficients[:-1] += 0.5 * widths
            coefficients[1:] += 0.5 * widths
            coefficients /= x[-1] - x[0]
        integrated_point += float(coefficients @ point[landmark_index, use])
        integrated_influence += np.einsum(
            "h,hp->p", coefficients, cluster[landmark_index, use]
        )
    if valid_landmarks == 0:
        return float("nan"), integrated_influence, patients
    integrated_point /= valid_landmarks
    integrated_influence /= valid_landmarks
    return float(integrated_point), integrated_influence, patients


def bootstrap_integrated_auc(
    joined: pd.DataFrame,
    protocol: TargetProtocol,
    counts: np.ndarray,
    expected_patients: pd.DataFrame,
) -> np.ndarray:
    point, influence, _ = integrated_auc_influence(
        joined, protocol, expected_patients
    )


    return point + counts.astype(np.float64) @ influence


def paired_score_intervals(
    predictions: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    protocol: TargetProtocol,
    reference: str,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reference not in predictions:
        raise ValueError("Reference model is not in predictions")
    joined = {
        name: attach_targets(frame, targets, protocol)
        for name, frame in predictions.items()
    }
    patients = joined[reference][["id", "fold"]].drop_duplicates().sort_values(["fold", "id"]).reset_index(drop=True)
    counts = stratified_patient_bootstrap_counts(
        patients["fold"].to_numpy(dtype=int), replicates, seed
    )
    samples = {
        name: bootstrap_integrated_scores(frame, protocol, counts, patients)
        for name, frame in joined.items()
    }
    all_once = np.ones((1, len(patients)), dtype=np.int16)
    points = {
        name: {
            metric: float(values[0])
            for metric, values in bootstrap_integrated_scores(
                frame, protocol, all_once, patients
            ).items()
        }
        for name, frame in joined.items()
    }
    for name, frame in joined.items():
        auc_point, _, _ = integrated_auc_influence(frame, protocol, patients)
        points[name]["integrated_auc"] = float(auc_point)
        samples[name]["integrated_auc"] = bootstrap_integrated_auc(
            frame, protocol, counts, patients
        )
    rows = []
    deltas = []
    for name, metrics in samples.items():
        for metric, values in metrics.items():
            if not np.isfinite(values).all():
                raise FloatingPointError(
                    f"Non-finite bootstrap values for model={name}, metric={metric}"
                )
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "point_estimate": points[name][metric],
                    "bootstrap_mean": float(np.mean(values)),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "replicates": replicates,
                }
            )
            if name != reference:
                if metric not in samples[reference]:
                    raise ValueError(
                        f"Reference model has no comparable bootstrap metric {metric}"
                    )
                difference = values - samples[reference][metric]
                deltas.append(
                    {
                        "model": name,
                        "reference": reference,
                        "metric": metric,
                        "point_delta": points[name][metric] - points[reference][metric],
                        "delta_mean": float(np.mean(difference)),
                        "ci_low": float(np.quantile(difference, 0.025)),
                        "ci_high": float(np.quantile(difference, 0.975)),
                        "probability_delta_below_zero": float(np.mean(difference < 0)),
                        "replicates": replicates,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(deltas)
