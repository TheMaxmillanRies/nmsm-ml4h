from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def patient_table(df: pd.DataFrame, terminal_states: tuple[int, ...]) -> pd.DataFrame:
    ordered = df.sort_values(["id", "t"])
    first = ordered.groupby("id", sort=False).head(1).set_index("id")
    last = ordered.groupby("id", sort=False).tail(1).set_index("id")
    tab = pd.DataFrame(index=last.index)
    tab["last_state"] = last["state"].astype(int)
    tab["last_time"] = last["t"].astype(int)
    if "next_state" in last.columns:
        final_next = pd.to_numeric(last["next_state"], errors="coerce").reindex(tab.index)
        known_next = final_next.notna() & (final_next >= 0) & ~tab["last_state"].isin(terminal_states)
        tab.loc[known_next, "last_state"] = final_next.loc[known_next].astype(int)
        tab.loc[known_next, "last_time"] = tab.loc[known_next, "last_time"] + 1
    tab["censored"] = ~tab["last_state"].isin(terminal_states)
    tab["first_state"] = first.loc[tab.index, "state"].astype(int)

    ranks = tab["last_time"].rank(method="first")
    tab["length_bin"] = pd.qcut(ranks, q=4, labels=False, duplicates="drop").astype(int)
    tab["stratum"] = (
        tab["last_state"].astype(str)
        + "_"
        + tab["censored"].astype(int).astype(str)
        + "_"
        + tab["length_bin"].astype(str)
    )

    counts = tab["stratum"].value_counts()
    small = tab["stratum"].map(counts) < 5
    tab.loc[small, "stratum"] = tab.loc[small, "last_state"].astype(str) + "_" + tab.loc[small, "censored"].astype(int).astype(str)
    return tab.reset_index()


def make_outer_folds(
    df: pd.DataFrame,
    terminal_states: tuple[int, ...],
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    patients = patient_table(df, terminal_states)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment = np.full(len(patients), -1, dtype=int)
    for fold, (_, test_index) in enumerate(splitter.split(patients["id"], patients["stratum"])):
        assignment[test_index] = fold
    patients["outer_fold"] = assignment
    return patients


def split_train_validation(
    outer_training_patients: pd.DataFrame,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    total = len(outer_training_patients)
    if total < 2:
        raise ValueError("At least two outer-training patients are required")
    requested = min(total - 1, max(1, int(round(validation_fraction * total))))
    groups = outer_training_patients.groupby("stratum", sort=True).indices
    ideal = {key: requested * len(index) / total for key, index in groups.items()}
    allocation = {key: int(np.floor(value)) for key, value in ideal.items()}
    remaining = requested - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda key: (-(ideal[key] - allocation[key]), str(key)),
    )
    for key in order[:remaining]:
        allocation[key] += 1
    rng = np.random.default_rng(seed)
    validation_indices: list[int] = []
    for key in sorted(groups, key=str):
        positions = np.asarray(groups[key], dtype=np.int64)
        take = allocation[key]
        if take:
            validation_indices.extend(
                rng.choice(positions, size=take, replace=False).tolist()
            )
    validation_index = np.sort(np.asarray(validation_indices, dtype=np.int64))
    if len(validation_index) != requested:
        raise AssertionError("Proportional validation allocation is inconsistent")
    training_mask = np.ones(total, dtype=bool)
    training_mask[validation_index] = False
    ids = outer_training_patients["id"].to_numpy()
    return ids[training_mask], ids[validation_index]


def audit_folds(folds: pd.DataFrame, expected_patients: int, n_splits: int = 5) -> None:
    if len(folds) != expected_patients or folds["id"].nunique() != expected_patients:
        raise AssertionError("Every patient must occur exactly once in the fold table")
    if set(folds["outer_fold"].unique()) != set(range(n_splits)):
        raise AssertionError("Outer fold labels are incomplete")
