from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    use = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(use):
        return float("nan")
    return float(np.sum(values[use] * weights[use]) / np.sum(weights[use]))


def weighted_binary_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    'IPCW-weighted one-vs-rest AUROC with exact half credit for ties.'
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    use = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    labels, scores, weights = labels[use], scores[use], weights[use]
    case_total = float(weights[labels].sum())
    control_total = float(weights[~labels].sum())
    if case_total <= 0 or control_total <= 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    labels, scores, weights = labels[order], scores[order], weights[order]
    group = np.cumsum(np.r_[True, scores[1:] != scores[:-1]]) - 1
    case_by_group = np.bincount(group, weights=weights * labels)
    control_by_group = np.bincount(group, weights=weights * (~labels))
    controls_before = np.cumsum(control_by_group) - control_by_group
    numerator = float(np.sum(case_by_group * (controls_before + 0.5 * control_by_group)))
    return numerator / (case_total * control_total)


def weighted_binary_average_precision(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float:
    'IPCW-weighted average precision, grouping tied scores exactly.'
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    use = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    labels, scores, weights = labels[use], scores[use], weights[use]
    case_total = float(weights[labels].sum())
    if case_total <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels, scores, weights = labels[order], scores[order], weights[order]
    group = np.cumsum(np.r_[True, scores[1:] != scores[:-1]]) - 1
    true_by_group = np.bincount(group, weights=weights * labels)
    total_by_group = np.bincount(group, weights=weights)
    cumulative_true = np.cumsum(true_by_group)
    cumulative_total = np.cumsum(total_by_group)
    precision = cumulative_true / cumulative_total
    recall_increment = true_by_group / case_total
    return float(np.sum(recall_increment * precision))


def metric_rows(predictions: pd.DataFrame, num_states: int) -> pd.DataFrame:
    rows = []
    pcols = [f"p_state_{state}" for state in range(num_states)]
    for (landmark, horizon), group in predictions.groupby(["landmark", "horizon"], sort=True):
        p = group[pcols].to_numpy(dtype=float)
        y = group["target_state"].to_numpy(dtype=int)
        w = group["ipcw"].to_numpy(dtype=float)
        known = group["known"].to_numpy(dtype=bool)
        valid = known & (y >= 0) & np.isfinite(p).all(axis=1) & (w > 0)
        if not np.any(valid):
            continue
        p = p[valid]
        y = y[valid]
        w = w[valid]
        one_hot = np.eye(num_states, dtype=float)[y]


        squared_error = np.square(p - one_hot)
        brier = squared_error.mean(axis=1)
        multiclass_brier = squared_error.sum(axis=1)
        mae = np.abs(p - one_hot).mean(axis=1)
        semantics = (
            str(group["probability_semantics"].iloc[0])
            if "probability_semantics" in group
            else "categorical"
        )
        nll = (
            -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))
            if semantics == "categorical"
            else np.full(len(y), np.nan, dtype=float)
        )
        rows.append(
            {
                "landmark": int(landmark),
                "horizon": int(horizon),
                "n": int(len(y)),
                "weight_sum": float(w.sum()),
                "effective_n": float(w.sum() ** 2 / np.square(w).sum()),
                "brier": weighted_mean(brier, w),
                "multiclass_brier": weighted_mean(multiclass_brier, w),
                "mae": weighted_mean(mae, w),
                "nll": weighted_mean(nll, w),
                **{
                    f"brier_state_{state}": weighted_mean(
                        squared_error[:, state], w
                    )
                    for state in range(num_states)
                },
                **{
                    f"mae_state_{state}": weighted_mean(
                        np.abs(p[:, state] - one_hot[:, state]), w
                    )
                    for state in range(num_states)
                },
                **{
                    f"auc_state_{state}": weighted_binary_auc(y == state, p[:, state], w)
                    for state in range(num_states)
                },
                **{
                    f"average_precision_state_{state}": weighted_binary_average_precision(
                        y == state, p[:, state], w
                    )
                    for state in range(num_states)
                },
                **{
                    f"calibration_bias_state_{state}": weighted_mean(
                        p[:, state] - (y == state).astype(float), w
                    )
                    for state in range(num_states)
                },
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        auc_columns = [f"auc_state_{state}" for state in range(num_states)]
        ap_columns = [f"average_precision_state_{state}" for state in range(num_states)]
        calibration_columns = [f"calibration_bias_state_{state}" for state in range(num_states)]
        result["macro_auc"] = result[auc_columns].mean(axis=1, skipna=True)
        result["macro_average_precision"] = result[ap_columns].mean(axis=1, skipna=True)
        result["macro_abs_calibration_bias"] = result[calibration_columns].abs().mean(
            axis=1, skipna=True
        )
    return result


def metric_rows_by_origin(predictions: pd.DataFrame, num_states: int) -> pd.DataFrame:
    'Dynamic transition-probability scores stratified by landmark origin.'
    pieces = []
    for origin, group in predictions.groupby("origin_state", sort=True):
        rows = metric_rows(group, num_states)
        rows.insert(0, "origin_state", int(origin))
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def integrated_metrics(rows: pd.DataFrame) -> dict[str, float]:
    def integrate(column: str) -> float:
        landmark_values = []
        for _, group in rows.groupby("landmark", sort=True):
            group = group.sort_values("horizon")
            x = group["horizon"].to_numpy(dtype=float)
            y = group[column].to_numpy(dtype=float)
            use = np.isfinite(x) & np.isfinite(y)
            x, y = x[use], y[use]
            if len(y) == 1:
                landmark_values.append(float(y[0]))
            elif len(y) > 1 and x[-1] > x[0]:
                landmark_values.append(float(np.trapezoid(y, x) / (x[-1] - x[0])))
        return float(np.mean(landmark_values)) if landmark_values else float("nan")

    result = {
        "ibs": integrate("brier"),
        "integrated_multiclass_brier": integrate("multiclass_brier")
        if "multiclass_brier" in rows
        else float("nan"),
        "integrated_mae": integrate("mae"),
        "integrated_nll": integrate("nll"),
    }
    if "macro_auc" in rows:
        result["integrated_auc"] = integrate("macro_auc")
    if "macro_average_precision" in rows:
        result["integrated_average_precision"] = integrate("macro_average_precision")
    if "macro_abs_calibration_bias" in rows:
        result["integrated_abs_calibration_bias"] = integrate(
            "macro_abs_calibration_bias"
        )
    for column in rows.columns:
        if column.startswith(("brier_state_", "mae_state_", "auc_state_", "average_precision_state_")):
            result[f"integrated_{column}"] = integrate(column)
    return result


def integrated_metrics_by_origin(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    'Integrate within origin first, then macro-average origin summaries.'
    summaries = []
    for origin, group in rows.groupby("origin_state", sort=True):
        summary = integrated_metrics(group.drop(columns="origin_state"))
        summaries.append({"origin_state": int(origin), **summary})
    frame = pd.DataFrame(summaries)
    macro = {
        f"origin_macro_{column}": float(frame[column].mean(skipna=True))
        for column in frame.columns
        if column != "origin_state"
    }
    return frame, macro


def calibration_rows(predictions: pd.DataFrame, num_states: int, bins: int = 10) -> pd.DataFrame:
    output = []
    for (landmark, horizon), group in predictions.groupby(["landmark", "horizon"], sort=True):
        for state in range(num_states):
            p = group[f"p_state_{state}"].to_numpy(dtype=float)
            y = (group["target_state"].to_numpy(dtype=int) == state).astype(float)
            w = group["ipcw"].to_numpy(dtype=float)
            known = group["known"].to_numpy(dtype=bool)


            finite_indices = np.flatnonzero(np.isfinite(p))
            order = finite_indices[np.argsort(p[finite_indices], kind="mergesort")]
            for bin_index, all_selected in enumerate(np.array_split(order, bins)):
                selected = all_selected[known[all_selected] & (w[all_selected] > 0)]
                if len(selected) == 0:
                    continue
                ww = w[selected]
                output.append(
                    {
                        "landmark": int(landmark),
                        "horizon": int(horizon),
                        "state": state,
                        "bin": bin_index,
                        "n_eligible": len(all_selected),
                        "n": len(selected),
                        "events": int(y[selected].sum()),
                        "weight_sum": float(ww.sum()),
                        "mean_probability": weighted_mean(p[selected], ww),
                        "observed_frequency": weighted_mean(y[selected], ww),
                    }
                )
    return pd.DataFrame(output)


def calibration_rows_by_origin(
    predictions: pd.DataFrame, num_states: int, bins: int = 10
) -> pd.DataFrame:
    pieces = []
    for origin, group in predictions.groupby("origin_state", sort=True):
        rows = calibration_rows(group, num_states, bins)
        rows.insert(0, "origin_state", int(origin))
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def ipcw_loess_calibration_rows(
    predictions: pd.DataFrame,
    num_states: int,
    horizons: tuple[int, ...] | None = None,
    grid_points: int = 25,
    span: float = 0.5,
) -> pd.DataFrame:
    'Origin-stratified BLR-IPCW moderate-calibration curves.'
    if not 0 < span <= 1:
        raise ValueError("LOESS span must lie in (0, 1]")
    selected_horizons = (
        set(map(int, horizons))
        if horizons is not None
        else {int(predictions["horizon"].max())}
    )
    output: list[dict[str, float | int | bool]] = []
    grouped = predictions[predictions["horizon"].isin(selected_horizons)].groupby(
        ["origin_state", "landmark", "horizon"], sort=True
    )
    for (origin, landmark, horizon), group in grouped:
        target = group["target_state"].to_numpy(dtype=int)
        known = group["known"].to_numpy(dtype=bool)
        weight = group["ipcw"].to_numpy(dtype=float)
        for state in range(num_states):
            probability = group[f"p_state_{state}"].to_numpy(dtype=float)
            valid = known & (target >= 0) & np.isfinite(probability) & (weight > 0)
            p = probability[valid]
            y = (target[valid] == state).astype(float)
            w = weight[valid]
            if len(p) < 20 or len(np.unique(y)) < 2:
                continue
            lower, upper = np.quantile(p, [0.01, 0.99])
            if not np.isfinite(lower + upper) or upper <= lower:
                continue
            grid = np.linspace(lower, upper, grid_points)
            neighbours = max(20, int(np.ceil(span * len(p))))
            for point in grid:
                distance = np.abs(p - point)
                bandwidth = np.partition(distance, neighbours - 1)[neighbours - 1]
                if bandwidth <= 0:
                    positive = distance[distance > 0]
                    bandwidth = float(positive.min()) if len(positive) else 1.0
                ratio = np.minimum(distance / bandwidth, 1.0)
                kernel = np.power(1.0 - np.power(ratio, 3.0), 3.0)
                local_weight = w * kernel
                use = local_weight > 0
                if not use.any():


                    continue
                design = np.column_stack(
                    [np.ones(int(use.sum())), p[use] - point]
                )
                weighted_design = design * local_weight[use, None]
                gram = design.T @ weighted_design
                rhs = weighted_design.T @ y[use]
                beta = np.linalg.pinv(gram, rcond=1e-10) @ rhs


                positive_weight = local_weight[use]
                scaled_weight = positive_weight / positive_weight.max()
                effective = float(
                    scaled_weight.sum() ** 2 / np.square(scaled_weight).sum()
                )
                output.append(
                    {
                        "origin_state": int(origin),
                        "landmark": int(landmark),
                        "horizon": int(horizon),
                        "state": state,
                        "predicted_probability": float(point),
                        "observed_probability": float(np.clip(beta[0], 0.0, 1.0)),
                        "effective_n": effective,
                        "n": int(len(p)),
                        "span": float(span),
                    }
                )
    return pd.DataFrame(output)


def ipcw_multinomial_calibration_rows(
    predictions: pd.DataFrame,
    num_states: int,
    horizons: tuple[int, ...] | None = None,
    n_knots: int = 4,
    degree: int = 3,
    max_points_per_cell: int | None = 5000,
) -> pd.DataFrame:
    'Origin-stratified MLR-IPCW moderate-calibration scatter data.'
    if n_knots < 3 or degree < 1:
        raise ValueError("MLR-IPCW requires n_knots >= 3 and degree >= 1")
    selected_horizons = (
        set(map(int, horizons))
        if horizons is not None
        else {int(predictions["horizon"].max())}
    )
    pcols = [f"p_state_{state}" for state in range(num_states)]
    output: list[dict[str, float | int | bool]] = []
    grouped = predictions[predictions["horizon"].isin(selected_horizons)].groupby(
        ["origin_state", "landmark", "horizon"], sort=True
    )
    for (origin, landmark, horizon), group in grouped:
        if (
            "probability_semantics" in group
            and str(group["probability_semantics"].iloc[0]) != "categorical"
        ):
            continue
        probability = group[pcols].to_numpy(dtype=float)
        target = group["target_state"].to_numpy(dtype=int)
        known = group["known"].to_numpy(dtype=bool)
        weight = group["ipcw"].to_numpy(dtype=float)
        valid = (
            known
            & (target >= 0)
            & (target < num_states)
            & np.isfinite(probability).all(axis=1)
            & np.isfinite(weight)
            & (weight > 0)
        )
        probability, target, weight = (
            probability[valid],
            target[valid],
            weight[valid],
        )
        classes, counts = np.unique(target, return_counts=True)
        if len(target) < 50 or len(classes) < 2 or np.min(counts) < 5:
            continue

        probability = np.clip(probability, 1e-8, 1.0)
        probability /= probability.sum(axis=1, keepdims=True)


        reference = int(classes[np.argmax(counts)])
        ratio_states = [state for state in range(num_states) if state != reference]
        log_ratios = np.log(probability[:, ratio_states]) - np.log(
            probability[:, [reference]]
        )
        spline = SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            include_bias=False,
            extrapolation="linear",
        )
        design = spline.fit_transform(log_ratios)
        model = LogisticRegression(


            C=100.0,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=2000,
            tol=1e-8,
        )
        model.fit(design, target, sample_weight=weight)
        fitted_active = model.predict_proba(design)
        fitted = np.zeros((len(target), num_states), dtype=float)
        fitted[:, model.classes_.astype(int)] = fitted_active
        converged = bool(np.max(model.n_iter_) < model.max_iter)
        selected = np.arange(len(target))
        if max_points_per_cell is not None and len(selected) > max_points_per_cell:


            order = np.lexsort(probability.T[::-1])
            positions = np.linspace(
                0, len(order) - 1, max_points_per_cell, dtype=int
            )
            selected = order[positions]
        for row_index in selected:
            for state in range(num_states):
                output.append(
                    {
                        "origin_state": int(origin),
                        "landmark": int(landmark),
                        "horizon": int(horizon),
                        "state": state,
                        "scatter_index": row_index,
                        "predicted_probability": float(probability[row_index, state]),
                        "observed_probability": float(fitted[row_index, state]),
                        "reference_state": reference,
                        "n": int(len(target)),
                        "n_knots": int(n_knots),
                        "degree": int(degree),
                        "converged": converged,
                    }
                )
    return pd.DataFrame(output)


def expected_calibration_error(calibration: pd.DataFrame) -> float:
    gap = np.abs(calibration["mean_probability"] - calibration["observed_frequency"])
    weight = calibration["weight_sum"].to_numpy(dtype=float)
    return weighted_mean(gap.to_numpy(dtype=float), weight)


def integrated_expected_calibration_error(calibration: pd.DataFrame) -> float:
    'Macro-state ECE integrated over horizons, then averaged over landmarks.'
    if calibration.empty:
        return float("nan")
    state_rows = []
    for (landmark, horizon, state), group in calibration.groupby(
        ["landmark", "horizon", "state"], sort=True
    ):
        state_rows.append(
            {
                "landmark": int(landmark),
                "horizon": int(horizon),
                "state": int(state),
                "ece": expected_calibration_error(group),
            }
        )
    state_frame = pd.DataFrame(state_rows)
    cell = (
        state_frame.groupby(["landmark", "horizon"], as_index=False)["ece"]
        .mean()
        .sort_values(["landmark", "horizon"])
    )
    landmark_values = []
    for _, group in cell.groupby("landmark", sort=True):
        x = group["horizon"].to_numpy(dtype=float)
        y = group["ece"].to_numpy(dtype=float)
        use = np.isfinite(x) & np.isfinite(y)
        x, y = x[use], y[use]
        if len(y) == 1:
            landmark_values.append(float(y[0]))
        elif len(y) > 1 and x[-1] > x[0]:
            landmark_values.append(float(np.trapezoid(y, x) / (x[-1] - x[0])))
    return float(np.mean(landmark_values)) if landmark_values else float("nan")


def weighted_logistic_calibration(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> tuple[float, float, bool]:
    'Weighted calibration intercept/slope for logit(p), with light stabilization.'
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    use = (
        np.isfinite(labels)
        & np.isfinite(probabilities)
        & np.isfinite(weights)
        & (weights > 0)
    )
    labels, probabilities, weights = labels[use], probabilities[use], weights[use]
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), False
    x = np.log(np.clip(probabilities, 1e-8, 1 - 1e-8)) - np.log1p(
        -np.clip(probabilities, 1e-8, 1 - 1e-8)
    )
    design = np.column_stack([np.ones(len(x)), x])
    scale = float(weights.sum())

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = np.clip(design @ beta, -35.0, 35.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        nll = np.sum(weights * (np.logaddexp(0.0, linear) - labels * linear)) / scale

        penalty = 1e-8 * np.square(beta).sum()
        gradient = design.T @ (weights * (fitted - labels)) / scale + 2e-8 * beta
        return float(nll + penalty), gradient

    prevalence = np.clip(np.sum(weights * labels) / scale, 1e-6, 1 - 1e-6)
    initial = np.array([np.log(prevalence / (1 - prevalence)), 1.0])
    fitted = minimize(
        lambda beta: objective(beta)[0],
        initial,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
    )
    return float(fitted.x[0]), float(fitted.x[1]), bool(fitted.success)


def calibration_model_rows(predictions: pd.DataFrame, num_states: int) -> pd.DataFrame:
    'Statewise calibration intercept/slope at each frozen landmark and horizon.'
    output = []
    for (landmark, horizon), group in predictions.groupby(
        ["landmark", "horizon"], sort=True
    ):
        target = group["target_state"].to_numpy(dtype=int)
        known = group["known"].to_numpy(dtype=bool)
        weights = group["ipcw"].to_numpy(dtype=float)
        for state in range(num_states):
            probability = group[f"p_state_{state}"].to_numpy(dtype=float)
            valid = known & (target >= 0) & np.isfinite(probability) & (weights > 0)
            intercept, slope, converged = weighted_logistic_calibration(
                target[valid] == state, probability[valid], weights[valid]
            )
            output.append(
                {
                    "landmark": int(landmark),
                    "horizon": int(horizon),
                    "state": state,
                    "n": int(valid.sum()),
                    "events": int(np.sum(target[valid] == state)),
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "converged": converged,
                }
            )
    return pd.DataFrame(output)


def calibration_model_rows_by_origin(
    predictions: pd.DataFrame, num_states: int
) -> pd.DataFrame:
    pieces = []
    for origin, group in predictions.groupby("origin_state", sort=True):
        rows = calibration_model_rows(group, num_states)
        rows.insert(0, "origin_state", int(origin))
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
