from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import FeatureEncoder
from .training import fit_feature_encoder


@dataclass(frozen=True)
class EncodedFrame:
    frame: pd.DataFrame
    continuous: np.ndarray
    categories: np.ndarray
    design: np.ndarray
    lookup: dict[tuple[int, int], int]


def fit_baseline_encoder(frame: pd.DataFrame, config: ExperimentConfig) -> FeatureEncoder:
    'Fit preprocessing on one cohort-entry row per training patient.'
    return fit_feature_encoder(baseline_rows(frame), config)


def baseline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    'Return the first observed row for each patient, preserving all columns.'
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(["id", "t"])
    result = ordered.groupby("id", sort=False, as_index=False).head(1).copy()
    if result["id"].duplicated().any():
        raise AssertionError("Cohort-entry extraction did not produce one row per patient")
    return result


def _one_hot(categories: np.ndarray, cardinalities: tuple[int, ...]) -> np.ndarray:
    if not cardinalities:
        return np.empty((len(categories), 0), dtype=np.float64)
    pieces: list[np.ndarray] = []
    for column, cardinality in enumerate(cardinalities):
        values = categories[:, column]
        if np.any((values < 0) | (values >= cardinality)):
            raise ValueError("Categorical baseline input is outside its fitted cardinality")
        encoded = np.zeros((len(values), cardinality), dtype=np.float64)
        encoded[np.arange(len(values)), values] = 1.0
        pieces.append(encoded)
    return np.concatenate(pieces, axis=1)


def combine_design(
    continuous: np.ndarray,
    categories: np.ndarray,
    encoder: FeatureEncoder,
) -> np.ndarray:
    return np.concatenate(
        [continuous.astype(np.float64), _one_hot(categories, encoder.categorical_cardinalities)],
        axis=1,
    )


def encode_frame(frame: pd.DataFrame, encoder: FeatureEncoder) -> EncodedFrame:
    ordered = frame.sort_values(["id", "t"]).reset_index(drop=True).copy()
    continuous, categories = encoder.encode_many(ordered)
    continuous = continuous.astype(np.float64)
    design = combine_design(continuous, categories, encoder)
    lookup = {
        (int(row.id), int(row.t)): index
        for index, row in enumerate(ordered[["id", "t"]].itertuples(index=False))
    }
    if len(lookup) != len(ordered):
        raise ValueError("Baseline encoding requires unique patient/time rows")
    return EncodedFrame(ordered, continuous, categories, design, lookup)


def baseline_feature_indices(encoded: EncodedFrame) -> dict[int, int]:
    'Map each patient to the encoded index of their cohort-entry row.'
    first = (
        encoded.frame.reset_index()
        .sort_values(["id", "t"])
        .groupby("id", sort=False, as_index=False)
        .head(1)
    )
    mapping = {
        int(row.id): int(row.index)
        for row in first[["index", "id"]].itertuples(index=False)
    }
    if len(mapping) != encoded.frame["id"].nunique():
        raise AssertionError("Missing cohort-entry feature index")
    return mapping


def baseline_snapshot_features(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    encoder: FeatureEncoder,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    'Encode cohort-entry covariates for patient-landmark prediction pairs.'
    encoded = encode_frame(baseline_rows(frame), encoder)
    first = baseline_feature_indices(encoded)
    ordered = pairs[["id", "landmark"]].drop_duplicates().sort_values(
        ["id", "landmark"]
    )
    keys = [
        (int(row.id), int(row.landmark))
        for row in ordered.itertuples(index=False)
    ]
    indices = np.asarray([first[patient] for patient, _ in keys], dtype=np.int64)
    lookup = {key: index for index, key in enumerate(keys)}
    return encoded.continuous[indices], encoded.categories[indices], lookup


def design_names(encoder: FeatureEncoder) -> list[str]:
    names = list(encoder.continuous_feature_names)
    assert encoder.category_levels is not None
    for column, cardinality in zip(
        encoder.categorical_columns, encoder.categorical_cardinalities
    ):
        levels = ["unseen"] + [str(value) for value in encoder.category_levels[column]]
        if len(levels) != cardinality:
            raise AssertionError("Categorical design-name cardinality mismatch")
        names.extend(f"category__{column}__{level}" for level in levels)
    return names


def advance_continuous(continuous: np.ndarray, encoder: FeatureEncoder) -> np.ndarray:
    output = np.asarray(continuous, dtype=np.float64).copy()
    encoder._check_fitted()
    assert encoder.stds is not None
    for column, raw_increment in encoder.known_dynamic_raw_increments:
        index = encoder.numeric_columns.index(column)
        output[index] += float(raw_increment) / float(encoder.stds.loc[column])
    if not encoder.include_missingness:
        return output
    n_numeric = len(encoder.numeric_columns)
    dynamic = set(encoder.observation_numeric_columns)
    for index, column in enumerate(encoder.numeric_columns):
        if column in dynamic:
            output[n_numeric + index] = 0.0
    if encoder.observation_numeric_columns:
        delta_start = 2 * n_numeric
        output[delta_start:] = np.minimum(
            1.0,
            output[delta_start:] + 1.0 / max(float(encoder.max_delta), 1.0),
        )
    return output


def nonself_edges(config: ExperimentConfig) -> tuple[tuple[int, int], ...]:
    if config.allowed_edges:
        edges = tuple((int(origin), int(destination)) for origin, destination in config.allowed_edges)
    else:
        terminals = frozenset(config.terminal_states)
        edges = tuple(
            (origin, destination)
            for origin in range(config.num_states)
            if origin not in terminals
            for destination in range(config.num_states)
        )
    return tuple(edge for edge in edges if edge[0] != edge[1])


def destination_map(config: ExperimentConfig) -> dict[int, tuple[int, ...]]:
    result: dict[int, list[int]] = {
        state: [] for state in range(config.num_states) if state not in config.terminal_states
    }
    for origin, destination in nonself_edges(config):
        result[origin].append(destination)
    final = {origin: tuple(sorted(set(destinations))) for origin, destinations in result.items()}
    empty = [origin for origin, destinations in final.items() if not destinations]
    if empty:
        raise ValueError(f"Nonterminal states have no permitted destination: {empty}")
    return final


def current_dwell(frame: pd.DataFrame, patient: int, landmark: int) -> int:
    group = frame[(frame["id"] == patient) & (frame["t"] <= landmark)].sort_values("t")
    if group.empty or int(group.iloc[-1]["t"]) != landmark:
        raise KeyError(f"No landmark row for patient={patient}, landmark={landmark}")
    state = int(group.iloc[-1]["state"])
    times = group["t"].to_numpy(dtype=int)
    states = group["state"].to_numpy(dtype=int)
    start = len(group) - 1
    while start > 0 and states[start - 1] == state and times[start - 1] == times[start] - 1:
        start -= 1
    return int(landmark - times[start])
