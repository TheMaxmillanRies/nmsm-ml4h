'msPseudo LMAJ adapter with documented sparse-validation handling.'

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .baseline_common import (
    baseline_feature_indices,
    encode_frame,
    fit_baseline_encoder,
)
from .config import ExperimentConfig
from .splits import split_train_validation
from .targets import KEYS
from .training import set_seed


def _mspseudo_split(targets, folds, fold, seed):
    'Keep the outer folds fixed; ensure estimable inner-training strata.'
    outer = folds[folds.outer_fold != fold]
    train, validation = split_train_validation(outer, seed=seed + fold)
    train, validation = set(map(int, train)), set(map(int, validation))
    population = targets.loc[targets.id.isin(set(outer.id)), ["id", "landmark", "origin_state"]].drop_duplicates()
    moved = []
    for _, group in population.groupby(["landmark", "origin_state"], sort=True):
        members = set(map(int, group.id))
        needed = max(0, 2 - len(members & train)) if len(members) >= 2 else 0
        selected = sorted(members & validation)[:needed]
        train.update(selected)
        validation.difference_update(selected)
        moved.extend(selected)
    return train, validation, sorted(set(moved))


def preflight_mspseudo_support(targets, folds, requested_folds, experiment, seed=2026):
    'Fail before expensive fitting if an evaluated task is not estimable.'
    population = targets[["id", "fold", "landmark", "origin_state"]].drop_duplicates()
    failures = []
    for fold in requested_folds:
        train, validation, moved = _mspseudo_split(population, folds, fold, seed)
        counts = population[population.id.isin(train)].groupby(["landmark", "origin_state"]).id.nunique()
        wanted = population[population.fold == fold][["landmark", "origin_state"]].drop_duplicates()
        for key in wanted.itertuples(index=False, name=None):
            number = int(counts.get(key, 0))
            if number < 2:
                failures.append({"fold": int(fold), "landmark": int(key[0]),
                                 "origin_state": int(key[1]), "training_patients": number})
        print(f"MSPSEUDO SUPPORT fold={fold} validation_to_training_moves={len(moved)}", flush=True)
    if failures:
        raise ValueError("msPseudo cannot estimate these requested landmark/origin tasks with fewer than "
                         "two outer-training patients. No test patients were dropped or used for fitting. "
                         f"A different predeclared baseline adaptation would be required: {failures}")


def _ij_group(
    rows: pd.DataFrame,
    num_states: int,
    max_horizon: int,
    terminal_states: tuple[int, ...],
) -> tuple[pd.DataFrame, list[dict[str, float | int]]]:
    'Compute exact landmark-AJ delete-one pseudo-values for one stratum.'
    rows = rows.sort_values(["id", "horizon"]).copy()
    counts = rows.groupby("id", sort=False)["horizon"].count()
    if not (counts == max_horizon).all():
        bad = counts[counts != max_horizon].head().to_dict()
        raise ValueError(f"Incomplete target horizon grid: {bad}")
    ids = rows["id"].drop_duplicates().to_numpy(dtype=np.int64)
    n = len(ids)
    if n < 2:
        raise ValueError("Delete-one pseudo-values require at least two patients")
    origins = (
        rows.groupby("id", sort=False)["origin_state"].first().to_numpy(dtype=np.int64)
    )
    if len(np.unique(origins)) != 1:
        raise ValueError("Pseudo-value group must contain one landmark origin")
    origin = int(origins[0])
    target = rows["target_state"].to_numpy(dtype=np.int64).reshape(n, max_horizon)
    known = rows["known"].to_numpy(dtype=bool).reshape(n, max_horizon)
    previous = np.full((n, max_horizon), -1, dtype=np.int64)
    previous[:, 0] = origin
    if max_horizon > 1:
        previous[:, 1:] = np.where(known[:, :-1], target[:, :-1], -1)

    occupation = np.eye(num_states, dtype=np.float64)[origin]
    leave_one_occupation = np.broadcast_to(occupation, (n, num_states)).copy()
    pseudo = np.empty((n, max_horizon, num_states), dtype=np.float32)
    audit: list[dict[str, float | int]] = []
    terminals = frozenset(map(int, terminal_states))

    for step in range(max_horizon):
        transition = np.eye(num_states, dtype=np.float64)
        omitted_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        for state in range(num_states):
            if state in terminals:
                continue
            at_risk = known[:, step] & (previous[:, step] == state)
            indices = np.flatnonzero(at_risk)
            if not len(indices):
                continue
            destinations = target[indices, step]
            if np.any((destinations < 0) | (destinations >= num_states)):
                raise ValueError("Known transition has an invalid destination")
            counts_to = np.bincount(destinations, minlength=num_states).astype(np.float64)
            transition[state] = counts_to / float(len(indices))
            if len(indices) == 1:
                leave_one = np.zeros((1, num_states), dtype=np.float64)
                leave_one[0, state] = 1.0
            else:
                leave_one_counts = np.broadcast_to(
                    counts_to, (len(indices), num_states)
                ).copy()
                leave_one_counts[np.arange(len(indices)), destinations] -= 1.0
                leave_one = leave_one_counts / float(len(indices) - 1)
            omitted_rows.append((state, indices, leave_one))

        transition_minus = np.broadcast_to(
            transition, (n, num_states, num_states)
        ).copy()
        for state, indices, leave_one in omitted_rows:
            transition_minus[indices, state] = leave_one
        occupation = occupation @ transition
        leave_one_occupation = np.einsum(
            "ni,nij->nj", leave_one_occupation, transition_minus
        )
        values = n * occupation[None] - (n - 1) * leave_one_occupation
        pseudo[:, step] = values.astype(np.float32)
        mean_error = float(np.max(np.abs(values.mean(axis=0) - occupation)))
        simplex_error = float(np.max(np.abs(values.sum(axis=1) - 1.0)))
        if simplex_error > 2e-5:
            raise AssertionError(
                f"Delete-one simplex identity failed at horizon {step + 1}: "
                f"error={simplex_error}"
            )
        audit.append(
            {
                "horizon": step + 1,
                "patients": n,
                "jackknife_mean_minus_aj_max_abs": mean_error,
                "pseudo_simplex_sum_max_abs": simplex_error,
                "aj_min": float(occupation.min()),
                "aj_max": float(occupation.max()),
            }
        )

    result = pd.DataFrame(
        {
            "id": np.repeat(ids, max_horizon),
            "landmark": int(rows["landmark"].iloc[0]),
            "origin_state": origin,
            "horizon": np.tile(
                np.arange(1, max_horizon + 1, dtype=np.int16), n
            ),
        }
    )
    flat = pseudo.reshape(n * max_horizon, num_states)
    for state in range(num_states):
        result[f"pseudo_state_{state}"] = flat[:, state]
    return result, audit


def _pseudo_for_ids(
    targets: pd.DataFrame,
    patient_ids: set[int],
    experiment: ExperimentConfig,
    label: str,
    reference_targets: pd.DataFrame | None = None,
    required_strata: set[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, float | int | str]]]:
    selected = targets[targets["id"].isin(patient_ids)]
    if selected.empty and label != "validation":
        raise ValueError(f"No target rows for msPseudo {label} cohort")
    if reference_targets is not None and reference_targets.id.isin(patient_ids).any():
        raise ValueError("msPseudo validation reference overlaps validation patients")
    pieces: list[pd.DataFrame] = []
    audits: list[dict[str, float | int | str]] = []
    for (landmark, origin), group in selected.groupby(
        ["landmark", "origin_state"], sort=True
    ):
        key = (int(landmark), int(origin))
        if required_strata is not None and key not in required_strata:
            continue
        n = int(group.id.nunique())
        if n < 2 and label == "inner_training":
            audits.append({"scope": label, "landmark": key[0], "origin_state": key[1],
                           "patients": n, "status": "not_estimable_no_training_pseudovalue"})
            continue
        reference_count = 0
        computation = group
        if n == 1:
            if reference_targets is None:
                raise ValueError("Singleton validation requires a disjoint inner-training reference")
            reference = reference_targets[(reference_targets.landmark == landmark)
                                          & (reference_targets.origin_state == origin)]
            reference_count = int(reference.id.nunique())
            if reference_count < 2:
                raise ValueError("Singleton validation lacks two inner-training reference patients")


            computation = pd.concat([reference, group], ignore_index=True)
        print(
            f"MSPSEUDO DELETE1 START scope={label} landmark={int(landmark)} "
            f"origin={int(origin)} patients={group['id'].nunique()}",
            flush=True,
        )
        piece, rows = _ij_group(
            computation,
            experiment.num_states,
            experiment.max_horizon,
            experiment.terminal_states,
        )
        if reference_count:
            piece = piece[piece.id.isin(set(group.id))].copy()
        pieces.append(piece)
        audits.extend(
            {
                "scope": label,
                "landmark": int(landmark),
                "origin_state": int(origin),
                **row,
                "validation_reference_patients": reference_count,
                "target_cohort_patients": n,
                "status": "training_reference_delete_one" if reference_count else "separate_cohort_delete_one",
            }
            for row in rows
        )
        print(
            f"MSPSEUDO DELETE1 DONE scope={label} landmark={int(landmark)} "
            f"origin={int(origin)}",
            flush=True,
        )
    if not pieces:
        empty = pd.DataFrame(columns=["id", "landmark", "origin_state", "horizon",
                                      *[f"pseudo_state_{s}" for s in range(experiment.num_states)]])
        return empty, audits
    return (
        pd.concat(pieces, ignore_index=True).sort_values(
            ["id", "landmark", "horizon"]
        ),
        audits,
    )


def generate_fold_pseudovalues(
    targets: pd.DataFrame,
    folds: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
    output_path: Path,
    audit_path: Path,
    seed: int = 2026,
) -> pd.DataFrame:
    train_set, validation_set, moved = _mspseudo_split(targets, folds, fold, seed)
    if train_set & validation_set:
        raise AssertionError("msPseudo inner training and validation overlap")
    result, audits = _pseudo_for_ids(
        targets, train_set, experiment, "inner_training"
    )
    for row in audits:
        row["fold"] = fold
        row["inner_training_patients"] = len(train_set)
        row["validation_patients_excluded"] = len(validation_set)
        row["validation_to_training_moves"] = len(moved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    return result


class PublishedMSPseudoMLP(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden: int,
        horizons: int,
        states: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizons = horizons
        self.states = states
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizons * states),
        )

    def forward(self, design: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(design)).reshape(
            len(design), self.horizons, self.states
        )


def _stratum_arrays(
    rows: pd.DataFrame,
    design: np.ndarray,
    baseline_index: dict[int, int],
    max_horizon: int,
    num_states: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = rows.sort_values(["id", "horizon"])
    counts = ordered.groupby("id", sort=False)["horizon"].count()
    if not (counts == max_horizon).all():
        raise ValueError("msPseudo stratum has an incomplete horizon grid")
    ids = ordered["id"].drop_duplicates().to_numpy(dtype=np.int64)
    features = design[
        [baseline_index[int(patient)] for patient in ids]
    ].astype(np.float32)
    columns = [f"pseudo_state_{state}" for state in range(num_states)]
    values = ordered[columns].to_numpy(dtype=np.float32).reshape(
        len(ids), max_horizon, num_states
    )
    return ids, features, values


def run_mspseudo_fold(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    targets: pd.DataFrame,
    pseudo: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
    output_dir: Path,
    epochs: int = 10000,
    patience: int = 100,
    dropout: float = 0.1,
    seed: int = 2026,
    device: torch.device | None = None,
) -> pd.DataFrame:
    set_seed(seed + fold)
    device = device or torch.device("cpu")
    started = time.time()
    preflight_mspseudo_support(targets, folds, [fold], experiment, seed)
    train_set, validation_set, moved = _mspseudo_split(targets, folds, fold, seed)
    training_targets = targets[targets.id.isin(train_set)]
    training_counts = training_targets.groupby(["landmark", "origin_state"]).id.nunique()
    estimable = set(training_counts[training_counts >= 2].index)
    eligible = pd.MultiIndex.from_frame(training_targets[["landmark", "origin_state"]]).isin(estimable)
    expected_ids = set(map(int, training_targets.loc[eligible, "id"].unique()))
    if set(map(int, pseudo["id"].unique())) != expected_ids:
        raise ValueError("msPseudo targets are not the exact inner-training cohort")
    if pseudo["id"].isin(validation_set).any():
        raise ValueError("msPseudo training targets contain validation patients")

    encoder = fit_baseline_encoder(frame[frame["id"].isin(train_set)], experiment)
    encoded = encode_frame(frame, encoder)
    baseline_index = baseline_feature_indices(encoded)
    test = targets[targets["fold"] == fold].sort_values(KEYS).copy()
    required = set(test[["landmark", "origin_state"]].itertuples(index=False, name=None))
    validation, validation_audit = _pseudo_for_ids(
        targets, validation_set, experiment, "validation",
        reference_targets=training_targets, required_strata=required,
    )
    if min(len(pseudo), len(test)) == 0:
        raise ValueError(f"Empty msPseudo partition in fold {fold}")

    output_dir.mkdir(parents=True, exist_ok=True)
    probability = np.full(
        (len(test), experiment.num_states), np.nan, dtype=np.float64
    )
    test_row = {
        tuple(map(int, key)): index
        for index, key in enumerate(test[KEYS].to_numpy())
    }
    strata: list[dict[str, object]] = []

    validation_keys = set(validation[["landmark", "origin_state"]].itertuples(index=False, name=None))
    groups = list(pseudo.groupby(["landmark", "origin_state"], sort=True))


    groups.sort(key=lambda item: (item[0] not in validation_keys, *item[0]))
    for (landmark, origin), train_group in groups:
        landmark, origin = int(landmark), int(origin)
        test_group = test[
            (test["landmark"] == landmark) & (test["origin_state"] == origin)
        ]
        if test_group.empty:
            continue
        ids, train_x, train_y = _stratum_arrays(
            train_group,
            encoded.design,
            baseline_index,
            experiment.max_horizon,
            experiment.num_states,
        )
        validation_group = validation[
            (validation["landmark"] == landmark)
            & (validation["origin_state"] == origin)
        ]
        has_validation = not validation_group.empty
        epoch_limit, budget_source = epochs, "validation_early_stopping"
        if has_validation:
            _, validation_x, validation_y = _stratum_arrays(
                validation_group, encoded.design, baseline_index,
                experiment.max_horizon, experiment.num_states,
            )
        else:
            donors = [row for row in strata if row["validation_patients"] > 0]
            local = [row for row in donors if row["landmark"] == landmark]
            selected = local or donors
            epoch_limit = min(epochs, max(1, int(np.ceil(np.median(
                [row["best_epoch"] for row in selected]
            ))))) if selected else min(epochs, max(1, patience))
            budget_source = ("median_validation_selected_same_landmark" if local else
                             "median_validation_selected_same_fold" if donors else
                             "predeclared_patience_sized_budget_no_validation_tasks")
            print(f"MSPSEUDO NO_VALIDATION fold={fold} landmark={landmark} origin={origin} "
                  f"fixed_epochs={epoch_limit} source={budget_source}; no test outcomes used", flush=True)
        model = PublishedMSPseudoMLP(
            encoded.design.shape[1],
            64,
            experiment.max_horizon,
            experiment.num_states,
            dropout,
        ).to(device)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(train_x), torch.as_tensor(train_y)),
            batch_size=256,
            shuffle=True,
        )
        if has_validation:
            validation_x_t = torch.as_tensor(validation_x, device=device)
            validation_y_t = torch.as_tensor(validation_y, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best = float("inf")
        stale = 0
        best_state = None
        best_epoch = 0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, epoch_limit + 1):
            model.train()
            losses = []
            for batch_x, batch_y in loader:
                loss = F.mse_loss(
                    model(batch_x.to(device)), batch_y.to(device)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            if not losses:
                raise RuntimeError("msPseudo training loader is empty")
            model.eval()
            validation_mse = None
            if has_validation:
                with torch.no_grad():
                    validation_mse = float(
                        F.mse_loss(model(validation_x_t), validation_y_t).item()
                    )
            if has_validation and not np.isfinite(validation_mse):
                raise FloatingPointError("msPseudo validation MSE is non-finite")
            history.append(
                {
                    "epoch": epoch,
                    "train_pseudovalue_mse": float(np.mean(losses)),
                    "validation_pseudovalue_mse": validation_mse,
                }
            )
            if not has_validation or validation_mse < best - 1e-7:
                best = validation_mse if has_validation else None
                stale = 0
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                stale += 1
            validation_label = f"{validation_mse:.7f}" if has_validation else "NA_fixed_epoch_budget"
            print(
                f"mspseudo fold={fold} landmark={landmark} origin={origin} "
                f"epoch={epoch}/{epoch_limit} train={np.mean(losses):.7f} "
                f"val_pseudo_mse={validation_label} stale={stale}",
                flush=True,
            )
            if has_validation and stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("msPseudo produced no checkpoint")
        model.load_state_dict(best_state)
        model.eval()

        test_ids = test_group["id"].drop_duplicates().to_numpy(dtype=np.int64)
        test_x = torch.as_tensor(
            encoded.design[
                [baseline_index[int(patient)] for patient in test_ids]
            ].astype(np.float32),
            device=device,
        )
        with torch.no_grad():
            test_values = model(test_x).cpu().numpy()
        patient_lookup = {
            int(patient): index for index, patient in enumerate(test_ids)
        }
        for row in test_group.itertuples(index=False):
            key = (int(row.id), fold, landmark, int(row.horizon))
            probability[test_row[key]] = test_values[
                patient_lookup[int(row.id)], int(row.horizon) - 1
            ]
        torch.save(
            {
                "model": best_state,
                "landmark": landmark,
                "origin_state": origin,
                "best_epoch": best_epoch,
            },
            output_dir / f"landmark_{landmark}_origin_{origin}.pt",
        )
        strata.append(
            {
                "landmark": landmark,
                "origin_state": origin,
                "train_patients": int(len(ids)),
                "validation_patients": int(validation_group["id"].nunique()),
                "test_patients": int(test_group["id"].nunique()),
                "best_epoch": best_epoch,
                "best_validation_pseudovalue_mse": best,
                "epochs_run": len(history),
                "maximum_epochs": epoch_limit,
                "configured_maximum_epochs": epochs,
                "epoch_budget_source": budget_source,
                "early_stopping_patience": patience,
                "stopped_by_patience_before_epoch_cap": has_validation and len(history) < epoch_limit,
                "best_at_training_boundary": best_epoch == len(history),
                "history": history,
            }
        )

    if not np.isfinite(probability).all():
        missing = int(np.sum(~np.isfinite(probability).all(axis=1)))
        raise RuntimeError(f"msPseudo left {missing} test rows without predictions")
    predictions = test[KEYS].reset_index(drop=True)
    for state in range(experiment.num_states):
        predictions[f"p_state_{state}"] = probability[:, state]
    predictions["probability_semantics"] = "independent_statewise"
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "training.json").write_text(
        json.dumps(
            {
                "method": "msPseudo (LMAJ) K-by-M independent-sigmoid regression",
                "reference": "Rahman and Purushotham, SIAM SDM 2023",
                "reference_url": "https://doi.org/10.1137/1.9781611977653.ch85",
                "implementation_status": "paper reconstruction; no author code located",
                "repository_search": "no author repository located as of 2026-08-27",
                "pseudo_values": "exact delete-one landmark Aalen-Johansen",
                "published_estimator_branch": "dynamic SOP with LMAJ",
                "markov_test_note": (
                    "the paper selects AJ versus LMAJ using a Markov-property test; "
                    "because no author code or reproducible test specification was "
                    "available, this comparator is explicitly the published LMAJ branch"
                ),
                "pseudo_value_scope": "inner-training-only targets; separate validation cohorts when n>=2; singleton validation uses its exact delete-one value in training plus that held-out subject",
                "checkpoint_metric": "validation pseudo-value MSE when available; transferred epoch budget otherwise",
                "sparse_validation_policy": "V6 sparse-stratum repair, explicitly an implementation adaptation, not an author-prescribed rule",
                "inner_split_repair_patients": len(moved),
                "inner_split_repair_basis": "landmark origins only; whole validation patients moved to retain two training patients where possible; outer folds unchanged",
                "covariates": "cohort-entry baseline only",
                "adaptation": (
                    "one model per landmark/origin dynamic task; published "
                    "independent sigmoid outputs retained without masking or normalization"
                ),
                "architecture": (
                    f"two hidden ReLU layers, 64 units each; dropout={dropout} "
                    "after the first hidden layer"
                ),
                "architecture_note": (
                    "the paper reports hyperparameter tuning but no selected architecture "
                    "or author code; this fixed architecture is disclosed, not claimed exact"
                ),
                "optimizer": "Adam lr=1e-3",
                "maximum_epochs": epochs,
                "early_stopping_patience": patience,
                "validation_pseudovalue_audit": validation_audit,
                "runtime_seconds": time.time() - started,
                "strata": strata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return predictions
