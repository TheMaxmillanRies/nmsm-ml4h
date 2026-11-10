from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KaplanMeierCensoring:
    times: np.ndarray
    survival_after: np.ndarray

    def at_right(self, query: np.ndarray | float | int) -> np.ndarray:
        'Return G(t), after censoring events recorded at t.'
        values = np.asarray(query, dtype=float)
        index = np.searchsorted(self.times, values, side="right") - 1
        result = np.ones_like(values, dtype=float)
        use = index >= 0
        result[use] = self.survival_after[index[use]]
        return result

    def at_left(self, query: np.ndarray | float | int) -> np.ndarray:
        'Return G(t-)=P(C>=t), before censoring events recorded at t.'
        values = np.asarray(query, dtype=float)
        index = np.searchsorted(self.times, values, side="left") - 1
        result = np.ones_like(values, dtype=float)
        use = index >= 0
        result[use] = self.survival_after[index[use]]
        return result

    def at(self, query: np.ndarray | float | int) -> np.ndarray:
        'Backward-compatible alias for the explicit right-continuous value.'
        return self.at_right(query)


def fit_censoring_km(patient_df: pd.DataFrame) -> KaplanMeierCensoring:
    'Reverse Kaplan--Meier: censoring is the event, terminal outcome is competing/non-event.'
    required = {"last_time", "censored"}
    missing = required.difference(patient_df.columns)
    if missing:
        raise ValueError(f"Censoring table is missing columns: {sorted(missing)}")
    times = patient_df["last_time"].to_numpy(dtype=float)
    events = patient_df["censored"].to_numpy(dtype=bool)
    if len(times) == 0 or not np.isfinite(times).all():
        raise ValueError("Censoring times must be non-empty and finite")
    unique = np.sort(np.unique(times))
    survival = []
    current = 1.0
    for time in unique:
        at_risk = int(np.sum(times >= time))
        censored = int(np.sum((times == time) & events))
        if at_risk:
            current *= 1.0 - censored / at_risk
        survival.append(current)
    fitted = KaplanMeierCensoring(unique, np.asarray(survival, dtype=float))
    if np.any(np.diff(fitted.survival_after) > 1e-12):
        raise AssertionError("Kaplan--Meier censoring survival must be non-increasing")
    if np.any((fitted.survival_after < 0) | (fitted.survival_after > 1)):
        raise AssertionError("Kaplan--Meier censoring survival left [0,1]")
    return fitted


def conditional_ipcw(
    km: KaplanMeierCensoring,
    landmark: np.ndarray,
    observation_time: np.ndarray,
    known: np.ndarray,
    max_weight: float | None = None,
) -> np.ndarray:
    landmark = np.asarray(landmark, dtype=float)
    observation_time = np.asarray(observation_time, dtype=float)
    known = np.asarray(known, dtype=bool)
    if not (landmark.shape == observation_time.shape == known.shape):
        raise ValueError("landmark, observation_time and known must have identical shapes")
    if np.any(known & (observation_time < landmark)):
        raise ValueError("Known post-landmark outcomes cannot precede their landmark")


    numerator = km.at_left(landmark)
    denominator = km.at_left(observation_time)
    weights = np.zeros_like(denominator, dtype=float)
    use = known & (denominator > 0)
    weights[use] = numerator[use] / denominator[use]
    if max_weight is not None:
        weights = np.minimum(weights, float(max_weight))
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise AssertionError("IPCW weights must be finite and non-negative")
    return weights
