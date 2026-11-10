from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import (
    calibration_model_rows_by_origin,
    calibration_model_rows,
    calibration_rows_by_origin,
    calibration_rows,
    expected_calibration_error,
    integrated_expected_calibration_error,
    integrated_metrics_by_origin,
    integrated_metrics,
    metric_rows_by_origin,
    metric_rows,
)
from .targets import KEYS, TargetProtocol, validate_target_registry


def probability_columns(num_states: int) -> list[str]:
    return [f"p_state_{state}" for state in range(num_states)]


def probability_semantics(predictions: pd.DataFrame) -> str:
    if "probability_semantics" not in predictions:
        return "categorical"
    values = predictions["probability_semantics"].dropna().unique()
    if len(values) != 1 or values[0] not in {"categorical", "independent_statewise"}:
        raise ValueError(
            "probability_semantics must be one constant value: categorical or "
            "independent_statewise"
        )
    return str(values[0])


def validate_predictions(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    protocol: TargetProtocol,
    atol: float = 1e-5,
) -> None:
    validate_target_registry(targets, protocol)
    pcols = probability_columns(protocol.num_states)
    missing = set(KEYS + pcols).difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    if predictions.duplicated(KEYS).any():
        raise ValueError("Prediction file has duplicate comparison keys")
    expected = targets[KEYS].sort_values(KEYS).reset_index(drop=True)
    actual = predictions[KEYS].sort_values(KEYS).reset_index(drop=True)
    if not expected.equals(actual):
        missing_keys = len(expected.merge(actual, on=KEYS, how="left", indicator=True).query("_merge == 'left_only'"))
        extra_keys = len(actual.merge(expected, on=KEYS, how="left", indicator=True).query("_merge == 'left_only'"))
        raise ValueError(
            f"Prediction keys do not exactly match the target registry: "
            f"missing={missing_keys}, extra={extra_keys}"
        )
    values = predictions[pcols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinite probabilities")
    if np.any(values < -atol) or np.any(values > 1 + atol):
        raise ValueError("Predicted probabilities leave [0,1]")
    if probability_semantics(predictions) == "categorical":
        sums = values.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=atol, rtol=0.0):
            raise ValueError(
                f"Predicted probabilities do not sum to one; max error={np.max(np.abs(sums-1))}"
            )


def attach_targets(
    predictions: pd.DataFrame, targets: pd.DataFrame, protocol: TargetProtocol
) -> pd.DataFrame:
    validate_predictions(predictions, targets, protocol)
    pcols = probability_columns(protocol.num_states)
    metadata = [
        *KEYS,
        "target_time",
        "origin_state",
        "target_state",
        "known",
        "observation_time",
        "ipcw",
    ]
    prediction_columns = KEYS + pcols
    if "probability_semantics" in predictions:
        prediction_columns.append("probability_semantics")
    return targets[metadata].merge(
        predictions[prediction_columns], on=KEYS, validate="one_to_one"
    )


def score_predictions(
    predictions: pd.DataFrame, targets: pd.DataFrame, protocol: TargetProtocol
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    joined = attach_targets(predictions, targets, protocol)
    rows = metric_rows(joined, protocol.num_states)
    calibration = calibration_rows(joined, protocol.num_states)
    calibration_models = calibration_model_rows(joined, protocol.num_states)
    summary = integrated_metrics(rows)
    summary["ece"] = integrated_expected_calibration_error(calibration)
    summary["pooled_ece"] = expected_calibration_error(calibration)
    summary["patients"] = int(joined["id"].nunique())
    summary["prediction_rows"] = int(len(joined))
    summary["known_fraction"] = float(joined["known"].mean())
    weights = joined.loc[joined["known"], "ipcw"].to_numpy(dtype=float)
    summary["max_ipcw"] = float(weights.max()) if len(weights) else float("nan")
    summary["effective_n_all_rows"] = (
        float(weights.sum() ** 2 / np.square(weights).sum()) if len(weights) else float("nan")
    )
    return rows, calibration, calibration_models, summary


def score_predictions_by_origin(
    predictions: pd.DataFrame, targets: pd.DataFrame, protocol: TargetProtocol
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    'Score P_jk(s,t) separately for every observed landmark origin j.'
    joined = attach_targets(predictions, targets, protocol)
    rows = metric_rows_by_origin(joined, protocol.num_states)
    calibration = calibration_rows_by_origin(joined, protocol.num_states)
    calibration_models = calibration_model_rows_by_origin(
        joined, protocol.num_states
    )
    summaries, macro = integrated_metrics_by_origin(rows)
    return rows, calibration, calibration_models, summaries, macro


def assert_common_prediction_keys(
    predictions: dict[str, pd.DataFrame], targets: pd.DataFrame, protocol: TargetProtocol
) -> None:
    if not predictions:
        raise ValueError("At least one prediction frame is required")
    for frame in predictions.values():
        validate_predictions(frame, targets, protocol)
