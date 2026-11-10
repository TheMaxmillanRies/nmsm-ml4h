'Maystre et al. (UAI 2022) GDirMixDTMC on the common MIMIC protocol.'

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch

from .baseline_common import baseline_snapshot_features, fit_baseline_encoder
from .config import ExperimentConfig
from .mixmarkov_baseline import generalized_dirichlet_loglike
from .model import allowed_transitions
from .targets import KEYS
from .training import set_seed


def _flat_features(values: np.ndarray, categories: np.ndarray, cardinalities: tuple[int, ...]) -> np.ndarray:
    parts = [np.ones((len(values), 1), dtype=np.float64), values.astype(np.float64)]
    for index, cardinality in enumerate(cardinalities):

        for level in range(1, cardinality):
            parts.append((categories[:, index:index + 1] == level).astype(np.float64))
    return np.concatenate(parts, axis=1)


def _counts(group: pd.DataFrame, landmark: int, states: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = np.zeros((states, states), dtype=np.float64)
    suffix = np.zeros((states, states), dtype=np.float64)
    for row in group.sort_values("t").itertuples(index=False):
        destination = int(row.next_state)
        if destination < 0:
            continue
        source = int(row.state)
        if not (0 <= source < states and 0 <= destination < states):
            raise ValueError("MixMarkov encountered an invalid transition")
        (prefix if int(row.t) < landmark else suffix)[source, destination] += 1.0
    return prefix, suffix


def _records(
    frame: pd.DataFrame,
    patient_ids: set[int],
    landmark: int,
    encoder,
    experiment: ExperimentConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    pairs = frame[
        frame["id"].isin(patient_ids)
        & (frame["t"] == landmark)
        & ~frame["state"].isin(experiment.terminal_states)
    ][["id", "state"]].rename(columns={"state": "origin_state"})
    pairs = pairs.drop_duplicates("id").sort_values("id")
    selected = frame[frame["id"].isin(pairs["id"])].sort_values(["id", "t"])
    values, categories, lookup = baseline_snapshot_features(
        selected, pairs.assign(landmark=landmark), encoder
    )
    order = [None] * len(lookup)
    for key, index in lookup.items():
        order[index] = key
    record = pd.DataFrame(order, columns=["id", "landmark"])
    origins = pairs.set_index("id")["origin_state"].to_dict()
    record["origin_state"] = record["id"].map(origins).astype(int)
    x = _flat_features(values, categories, encoder.categorical_cardinalities)
    groups = {int(pid): group for pid, group in selected.groupby("id", sort=False)}
    prefix, suffix = [], []
    for pid in record["id"]:
        before, after = _counts(groups[int(pid)], landmark, experiment.num_states)
        prefix.append(before)
        suffix.append(after)
    return record, x, np.asarray(prefix), np.asarray(suffix)


def _fit(
    train_x: np.ndarray,
    train_prefix: np.ndarray,
    train_suffix: np.ndarray,
    mask: torch.Tensor,
    max_iterations: int | None,
    l2: float,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, object]]:
    "Fit the authors' full-batch penalized likelihood with Newton-CG."
    dtype = torch.float64
    x_train = torch.as_tensor(train_x, dtype=dtype)
    k_train, o_train = torch.as_tensor(train_suffix, dtype=dtype), torch.as_tensor(train_prefix, dtype=dtype)
    shape = (train_x.shape[1], mask.shape[0], mask.shape[1] - 1, 2)
    history: list[dict[str, float]] = []

    def tensor_parameter(vector: np.ndarray) -> torch.Tensor:
        return torch.tensor(vector.reshape(shape), dtype=dtype, requires_grad=True)

    def fun_and_jac(vector: np.ndarray) -> tuple[float, np.ndarray]:
        parameter = tensor_parameter(vector)
        loss = -generalized_dirichlet_loglike(
            parameter, x_train, k_train, mask, o_train, l2
        )
        gradient = torch.autograd.grad(loss, parameter)[0]
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError("MixMarkov Newton-CG objective is non-finite")
        return float(loss.detach()), gradient.detach().numpy().ravel()

    def hessp(vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
        parameter = tensor_parameter(vector)
        loss = -generalized_dirichlet_loglike(
            parameter, x_train, k_train, mask, o_train, l2
        )
        gradient = torch.autograd.grad(loss, parameter, create_graph=True)[0]
        direction_tensor = torch.as_tensor(direction.reshape(shape), dtype=dtype)
        product = torch.autograd.grad(
            (gradient * direction_tensor).sum(), parameter
        )[0]
        if not torch.isfinite(product).all():
            raise FloatingPointError("MixMarkov Newton-CG Hessian product is non-finite")
        return product.detach().numpy().ravel()

    def callback(vector: np.ndarray) -> None:
        objective, _ = fun_and_jac(vector)
        history.append(
            {"iteration": len(history) + 1, "penalized_negative_log_likelihood": objective}
        )
        limit = "unlimited" if max_iterations is None else str(max_iterations)
        print(
            f"mixmarkov newton_cg iteration={len(history)}/{limit} "
            f"objective={objective:.6f}",
            flush=True,
        )

    options = {"disp": False}
    if max_iterations is not None:
        if max_iterations <= 0:
            raise ValueError("MixMarkov max_iterations must be positive or None")
        options["maxiter"] = int(max_iterations)
    result = minimize(
        fun=fun_and_jac,
        x0=np.zeros(int(np.prod(shape)), dtype=np.float64),
        method="Newton-CG",
        jac=True,
        hessp=hessp,
        callback=callback,
        tol=1e-5,
        options=options,
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"MixMarkov Newton-CG failed: {result.message}")
    if not history:
        history.append(
            {"iteration": 0, "penalized_negative_log_likelihood": float(result.fun)}
        )
    optimizer_audit = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "finite_result_accepted_like_official_repository": True,
    }
    return result.x.reshape(shape), history, optimizer_audit


def _predict(
    params: np.ndarray,
    x: np.ndarray,
    records: pd.DataFrame,
    offsets: np.ndarray,
    mask: np.ndarray,
    horizon: int,
    samples: int,
    seed: int,
    chunk: int = 256,
) -> np.ndarray:
    linear = np.einsum("md,dijq->mijq", x, params)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            concentration = np.exp(linear)
        except FloatingPointError as error:
            raise FloatingPointError(
                "MixMarkov concentration overflow; refusing to clip fitted log-concentrations"
            ) from error
    if not np.isfinite(concentration).all() or np.any(concentration <= 0):
        raise FloatingPointError("MixMarkov produced invalid generalized-Dirichlet concentrations")
    alpha, beta = concentration[..., 0], concentration[..., 1]
    aggregate = np.flip(np.cumsum(np.flip(offsets, axis=-1), axis=-1), axis=-1)
    alpha += offsets[..., :-1]
    beta += aggregate[..., 1:]
    tail = np.flip(np.cumsum(np.flip(mask.astype(int), axis=1), axis=1), axis=1).astype(bool)
    gd_mask = mask[:, :-1] & tail[:, 1:]
    last = (mask.shape[1] - 1) - np.argmax(mask[:, ::-1], axis=1)
    rng = np.random.default_rng(seed)
    output = np.zeros((len(records), horizon, mask.shape[0]), dtype=np.float64)
    for start in range(0, len(records), chunk):
        stop = min(start + chunk, len(records))
        a, b = alpha[start:stop], beta[start:stop]
        draws = rng.beta(a[None], b[None], size=(samples, *a.shape))
        draws *= gd_mask[None, None]
        transition = np.zeros((samples, stop - start, mask.shape[0], mask.shape[1]), dtype=np.float64)
        remaining = np.ones((samples, stop - start, mask.shape[0]), dtype=np.float64)
        for destination in range(mask.shape[1] - 1):
            value = draws[..., destination] * remaining
            transition[..., destination] = value
            remaining -= value
        for source, destination in enumerate(last):
            transition[:, :, source, destination] = remaining[:, :, source]
        origin = records.iloc[start:stop]["origin_state"].to_numpy(dtype=int)
        distribution = np.broadcast_to(
            np.eye(mask.shape[0], dtype=np.float64)[origin],
            (samples, stop - start, mask.shape[0]),
        ).copy()
        for step in range(horizon):
            distribution = np.einsum("smi,smij->smj", distribution, transition)
            output[start:stop, step] = distribution.mean(axis=0)
    return output


def run_mixmarkov_fold(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    targets: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
    output_dir: Path,
    max_iterations: int | None = None,
    samples: int = 100,
    seed: int = 2026,
) -> pd.DataFrame:
    set_seed(seed + fold)
    outer_training = folds[folds["outer_fold"] != fold]
    train_ids = set(map(int, outer_training["id"]))
    test_ids = set(map(int, folds.loc[folds["outer_fold"] == fold, "id"]))
    encoder = fit_baseline_encoder(frame[frame["id"].isin(train_ids)], experiment)
    mask = allowed_transitions(experiment.num_states, experiment.terminal_states, experiment.allowed_edges)
    rows, audits = [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()


    train_record, train_x, _, train_counts = _records(
        frame, train_ids, 0, encoder, experiment
    )
    zero_train = np.zeros_like(train_counts)
    params, history, optimizer_audit = _fit(
        train_x,
        zero_train,
        train_counts,
        mask,
        max_iterations,
        0.0,
    )
    np.savez_compressed(output_dir / "gdir_mix_dtmc.npz", params=params)

    for landmark in experiment.landmarks:
        print(f"MIXMARKOV START fold={fold} landmark={landmark}", flush=True)
        test_record, test_x, test_prefix, _ = _records(frame, test_ids, landmark, encoder, experiment)
        probability = _predict(
            params, test_x, test_record, test_prefix, mask.numpy(),
            experiment.max_horizon, samples, seed + 1000 * fold + landmark,
        )
        lookup = {int(pid): index for index, pid in enumerate(test_record["id"])}
        requested = targets[(targets["fold"] == fold) & (targets["landmark"] == landmark)]
        for row in requested.itertuples(index=False):
            result = {"id": int(row.id), "fold": fold, "landmark": landmark, "horizon": int(row.horizon)}
            values = probability[lookup[int(row.id)], int(row.horizon) - 1]
            result.update({f"p_state_{state}": float(values[state]) for state in range(experiment.num_states)})
            rows.append(result)
        audits.append(
            {
                "landmark": landmark, "test_patients": len(test_record),
            }
        )
        print(f"MIXMARKOV DONE fold={fold} landmark={landmark}", flush=True)
    predictions = pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "training.json").write_text(
        json.dumps(
            {
                "method": "Maystre et al. 2022 GDirMixDTMC",
                "reference_url": "https://proceedings.mlr.press/v180/maystre22a.html",
                "repository_url": "https://github.com/spotify-research/mixmarkov",
                "reference_commit": "spotify-research/mixmarkov@80e173143138ba58e3e28211f8ec7627c1a55d59",
                "implementation_status": (
                    "official objective and prediction algorithm; PyTorch exact "
                    "derivatives replace JAX without changing SciPy Newton-CG"
                ),
                "adaptation": (
                    "published discrete-time generalized-Dirichlet mixture; one "
                    "sequence-level model, with observed prefixes used only as "
                    "posterior prediction offsets"
                ),
                "covariates": "cohort-entry baseline only",
                "optimizer": "SciPy Newton-CG with exact PyTorch gradient and Hessian-vector product",
                "l2": 0.0,
                "l2_note": (
                    "official GDirMixDTMC fit default; penalties reported in "
                    "other repository notebooks belong to different CTMC or "
                    "finite-mixture estimators and are not transferred here"
                ),
                "train_patients": len(train_record),
                "optimizer_iterations": len(history),
                "maximum_optimizer_iterations": max_iterations,
                "optimizer_iteration_cap_note": (
                    "None matches the official scipy.optimize.minimize call; "
                    "a finite value is permitted only for explicit smoke tests"
                ),
                "optimizer_tolerance": 1e-5,
                "optimizer_result": optimizer_audit,
                "optimizer_result_note": (
                    "The official repository stores scipy OptimizeResult.x "
                    "without requiring OptimizeResult.success. This adapter "
                    "therefore accepts the same finite Newton-CG result while "
                    "recording the status and still rejects non-finite output."
                ),
                "history": history,
                "samples": samples,
                "samples_note": "100 is the official GDirMixDTMC prediction default",
                "runtime_seconds": time.time() - started, "landmarks": audits,
            }, indent=2,
        ), encoding="utf-8",
    )
    return predictions
