'Time-inhomogeneous Markov and dwell-time semi-Markov controls.'

from __future__ import annotations

import numpy as np
import pandas as pd


DWELL_CUTS = np.array([1, 2, 3, 6, 12, 24], dtype=int)


def add_dwell(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["id", "t"]).copy()
    previous = ordered.groupby("id", sort=False)["state"].shift()
    run = ordered["state"].ne(previous).groupby(ordered["id"], sort=False).cumsum()
    ordered["dwell"] = ordered.groupby(["id", run], sort=False).cumcount() + 1
    return ordered


def dwell_bucket(dwell: np.ndarray | int) -> np.ndarray:
    return np.searchsorted(DWELL_CUTS, np.asarray(dwell), side="left")


def _normalize_rows(counts: np.ndarray) -> np.ndarray:
    denominator = counts.sum(axis=-1, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError("Every transition row must have positive support")
    return counts / denominator


def estimate_processes(
    training: pd.DataFrame,
    num_states: int,
    terminal_states: tuple[int, ...],
    max_time: int,
    time_shrinkage: float,
    dwell_shrinkage: float,
    allowed_edges: tuple[tuple[int, int], ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    valid = training["next_state"].notna() & training["next_state"].between(0, num_states - 1)
    rows = training.loc[valid, ["t", "state", "next_state", "dwell"]]
    rows = rows[rows["t"].between(0, max_time)]
    current = rows["state"].to_numpy(int)
    destination = rows["next_state"].to_numpy(int)
    times = rows["t"].to_numpy(int)
    allowed = (
        np.zeros((num_states, num_states), dtype=bool)
        if allowed_edges
        else np.ones((num_states, num_states), dtype=bool)
    )
    for source, target in allowed_edges:
        allowed[source, target] = True
    for terminal in terminal_states:
        allowed[terminal] = False
        allowed[terminal, terminal] = True
    global_counts = 0.5 * allowed.astype(float)
    np.add.at(global_counts, (current, destination), 1.0)
    global_counts[~allowed] = 0.0
    global_probability = _normalize_rows(global_counts)
    time_counts = np.zeros((max_time + 1, num_states, num_states), dtype=float)
    np.add.at(time_counts, (times, current, destination), 1.0)
    time_counts += time_shrinkage * global_probability[None, :, :]
    time_counts[:, ~allowed] = 0.0
    time_probability = _normalize_rows(time_counts)
    buckets = len(DWELL_CUTS) + 1
    semi_counts = np.zeros((max_time + 1, num_states, buckets, num_states), dtype=float)
    np.add.at(
        semi_counts,
        (times, current, dwell_bucket(rows["dwell"].to_numpy(int)), destination),
        1.0,
    )
    semi_counts += dwell_shrinkage * time_probability[:, :, None, :]
    semi_counts *= allowed[None, :, None, :]
    return time_probability, _normalize_rows(semi_counts)


def _markov_occupations(
    probabilities: np.ndarray, origin: int, landmark: int, horizons: int
) -> np.ndarray:
    occupation = np.eye(probabilities.shape[1], dtype=float)[origin]
    output = []
    for horizon in range(1, horizons + 1):
        time = min(landmark + horizon - 1, probabilities.shape[0] - 1)
        occupation = occupation @ probabilities[time]
        output.append(occupation.copy())
    return np.asarray(output)


def _semi_markov_occupations(
    probabilities: np.ndarray,
    origin: int,
    initial_dwell: int,
    landmark: int,
    horizons: int,
    dwell_cap: int,
) -> np.ndarray:
    num_states = probabilities.shape[1]
    augmented = np.zeros((num_states, dwell_cap), dtype=float)
    augmented[origin, min(initial_dwell, dwell_cap) - 1] = 1.0
    output = []
    for horizon in range(1, horizons + 1):
        time = min(landmark + horizon - 1, probabilities.shape[0] - 1)
        updated = np.zeros_like(augmented)
        for state in range(num_states):
            for dwell_index in np.flatnonzero(augmented[state] > 0):
                mass, dwell = augmented[state, dwell_index], dwell_index + 1
                kernel = probabilities[time, state, int(dwell_bucket(dwell))]
                for destination, probability in enumerate(kernel):
                    if probability <= 0:
                        continue
                    next_dwell = min(dwell + 1, dwell_cap) if destination == state else 1
                    updated[destination, next_dwell - 1] += mass * probability
        augmented = updated
        output.append(augmented.sum(axis=1))
    return np.asarray(output)


def make_predictions(
    reference: pd.DataFrame,
    dwell_lookup: dict[tuple[int, int], int],
    markov: np.ndarray,
    semi_markov: np.ndarray,
    num_states: int,
    max_horizon: int,
    dwell_cap: int,
) -> dict[str, pd.DataFrame]:


    reference = reference.reset_index(drop=True)
    probability_columns = [f"p_state_{state}" for state in range(num_states)]
    metadata = reference.drop(columns=[column for column in probability_columns if column in reference]).copy()
    results = {"empirical_markov": metadata.copy(), "empirical_semi_markov": metadata.copy()}
    markov_values = np.zeros((len(reference), num_states), dtype=float)
    semi_values = np.zeros((len(reference), num_states), dtype=float)
    markov_cache, semi_cache = {}, {}
    for (patient, landmark), indices in reference.groupby(["id", "landmark"], sort=False).groups.items():
        selected = reference.loc[indices].sort_values("horizon")
        origin = int(selected.iloc[0]["origin_state"])
        dwell = dwell_lookup[(int(patient), int(landmark))]
        markov_key = (int(landmark), origin)
        if markov_key not in markov_cache:
            markov_cache[markov_key] = _markov_occupations(markov, origin, int(landmark), max_horizon)
        semi_key = (int(landmark), origin, min(int(dwell), dwell_cap))
        if semi_key not in semi_cache:
            semi_cache[semi_key] = _semi_markov_occupations(
                semi_markov, origin, dwell, int(landmark), max_horizon, dwell_cap
            )
        horizons = selected["horizon"].to_numpy(int) - 1
        markov_values[selected.index] = markov_cache[markov_key][horizons]
        semi_values[selected.index] = semi_cache[semi_key][horizons]
    for state, column in enumerate(probability_columns):
        results["empirical_markov"][column] = markov_values[:, state]
        results["empirical_semi_markov"][column] = semi_values[:, state]
    return results
