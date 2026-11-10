from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ipcw import conditional_ipcw, fit_censoring_km
from .splits import patient_table


KEYS = ["id", "fold", "landmark", "horizon"]


@dataclass(frozen=True)
class TargetProtocol:
    landmarks: tuple[int, ...]
    max_horizon: int
    terminal_states: tuple[int, ...]
    num_states: int


def remap_states(frame: pd.DataFrame, state_map: dict[int, int] | None) -> pd.DataFrame:
    if state_map is None:
        return frame.copy()
    result = frame.copy()
    result["state"] = result["state"].map(state_map)
    if result["state"].isna().any():
        raise ValueError("state_map does not cover every observed state")
    if "next_state" in result:
        raw = pd.to_numeric(result["next_state"], errors="coerce")
        known = raw.notna() & (raw >= 0)
        result.loc[known, "next_state"] = raw.loc[known].astype(int).map(state_map)
        if result.loc[known, "next_state"].isna().any():
            raise ValueError("state_map does not cover every known next_state")
    return result


def target_at(
    group: pd.DataFrame, target_time: int, terminals: frozenset[int]
) -> tuple[int, bool, int]:
    'Return occupancy target, observability, and time at which it became known.'
    exact = group[group["t"] == target_time]
    if len(exact) > 1:
        raise ValueError("Patient trajectory has duplicate time rows")
    if len(exact):
        return int(exact.iloc[0]["state"]), True, int(target_time)
    last = group.iloc[-1]
    last_time = int(last["t"])
    last_state = int(last["state"])
    if last_state in terminals and target_time > last_time:
        return last_state, True, last_time
    raw_next = pd.to_numeric(pd.Series([last.get("next_state", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(raw_next) and int(raw_next) >= 0:
        next_state = int(raw_next)
        if target_time == last_time + 1:
            return next_state, True, last_time + 1
        if next_state in terminals and target_time > last_time + 1:
            return next_state, True, last_time + 1
        observation_time = last_time + 1
    else:
        observation_time = last_time
    return -1, False, observation_time


def validate_target_registry(targets: pd.DataFrame, protocol: TargetProtocol) -> None:
    required = {
        *KEYS,
        "target_time",
        "origin_state",
        "target_state",
        "known",
        "observation_time",
        "ipcw",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Target registry is missing columns: {sorted(missing)}")
    if targets.duplicated(KEYS).any():
        raise ValueError("Target registry has duplicate model-comparison keys")
    if not np.array_equal(
        targets["target_time"].to_numpy(dtype=int),
        (targets["landmark"] + targets["horizon"]).to_numpy(dtype=int),
    ):
        raise ValueError("target_time must equal landmark+horizon")
    if not targets["horizon"].between(1, protocol.max_horizon).all():
        raise ValueError("Target registry contains the landmark or an unsupported horizon")
    expected = set(range(1, protocol.max_horizon + 1))
    observed = targets.groupby(["id", "fold", "landmark"])["horizon"].agg(lambda x: set(map(int, x)))
    if not observed.map(lambda x: x == expected).all():
        raise ValueError("Every eligible patient-landmark must have the exact common horizon grid")
    if targets["origin_state"].isin(protocol.terminal_states).any():
        raise ValueError("Terminal landmark origins are ineligible")
    known = targets["known"].to_numpy(dtype=bool)
    target = targets["target_state"].to_numpy(dtype=int)
    if np.any(known & ((target < 0) | (target >= protocol.num_states))):
        raise ValueError("Known targets must be valid states")
    if np.any(~known & (targets["ipcw"].to_numpy(dtype=float) != 0)):
        raise ValueError("Unknown targets must have zero IPCW")
    weights = targets["ipcw"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("IPCW weights must be finite and non-negative")


def build_target_registry(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    protocol: TargetProtocol,
    state_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    work = remap_states(frame, state_map).sort_values(["id", "t"]).copy()
    if work.duplicated(["id", "t"]).any():
        raise ValueError("Trajectory data have duplicate patient/time rows")
    if folds.duplicated("id").any() or set(work["id"].unique()) != set(folds["id"].unique()):
        raise ValueError("Fold table must assign every trajectory patient exactly once")
    terminals = frozenset(protocol.terminal_states)
    rows: list[dict[str, int | bool]] = []
    for fold in sorted(folds["outer_fold"].unique()):
        training_ids = folds.loc[folds["outer_fold"] != fold, "id"]
        test_ids = set(map(int, folds.loc[folds["outer_fold"] == fold, "id"]))
        training = work[work["id"].isin(training_ids)]
        km = fit_censoring_km(patient_table(training, protocol.terminal_states))
        test = work[work["id"].isin(test_ids)]
        for pid, group in test.groupby("id", sort=False):
            group = group.sort_values("t")
            times = group["t"].to_numpy(dtype=int)
            states = group["state"].to_numpy(dtype=int)
            state_at_time = dict(zip(times.tolist(), states.tolist()))
            last_time = int(times[-1])
            last_state = int(states[-1])
            raw_next = pd.to_numeric(
                pd.Series([group.iloc[-1].get("next_state", np.nan)]), errors="coerce"
            ).iloc[0]
            next_known = bool(np.isfinite(raw_next) and int(raw_next) >= 0)
            final_next = int(raw_next) if next_known else -1
            for landmark in protocol.landmarks:
                if landmark not in state_at_time:
                    continue
                origin = int(state_at_time[landmark])
                if origin in terminals:
                    continue
                for horizon in range(1, protocol.max_horizon + 1):
                    query = landmark + horizon
                    if query in state_at_time:
                        target, known, observation_time = state_at_time[query], True, query
                    elif last_state in terminals and query > last_time:
                        target, known, observation_time = last_state, True, last_time
                    elif next_known and query == last_time + 1:
                        target, known, observation_time = final_next, True, last_time + 1
                    elif next_known and final_next in terminals and query > last_time + 1:
                        target, known, observation_time = final_next, True, last_time + 1
                    else:
                        target, known = -1, False
                        observation_time = last_time + 1 if next_known else last_time
                    rows.append(
                        {
                            "id": int(pid),
                            "fold": int(fold),
                            "landmark": int(landmark),
                            "horizon": int(horizon),
                            "target_time": int(landmark + horizon),
                            "origin_state": origin,
                            "target_state": int(target),
                            "known": bool(known),
                            "observation_time": int(observation_time),
                        }
                    )
        fold_rows = [row for row in rows if row["fold"] == int(fold)]
        if fold_rows:
            selected = pd.DataFrame(fold_rows)
            selected["ipcw"] = conditional_ipcw(
                km,
                selected["landmark"].to_numpy(dtype=float),
                selected["observation_time"].to_numpy(dtype=float),
                selected["known"].to_numpy(dtype=bool),
            )
            for row, weight in zip(fold_rows, selected["ipcw"].to_numpy(dtype=float)):
                row["ipcw"] = float(weight)
    result = pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
    validate_target_registry(result, protocol)
    return result
