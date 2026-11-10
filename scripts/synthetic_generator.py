'ICML Supplementary: Synthetic multi-state simulator with history-dependent risk ("burden")'

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def make_baseline_series(max_time: int, lam0: float, lam_slope: float = 0.0) -> pd.Series:
    'Create a simple time-varying baseline hazard series indexed by integer time t.'
    idx = pd.Index(np.arange(max_time + 1), name="t")
    frac = idx.to_numpy(dtype=float) / max(1.0, float(max_time))
    vals = np.clip(lam0 + lam_slope * frac, 0.0, None)
    return pd.Series(vals, index=idx)


def _safe_baseline_at(bh: pd.Series, t: int) -> float:
    'Return baseline at time t, clamped to the series boundary if t is out of range.'
    t = int(t)
    if t in bh.index:
        return float(bh.loc[t])
    if t < int(bh.index.min()):
        return float(bh.iloc[0])
    return float(bh.iloc[-1])


def competing_risks_probs(hazards: Dict[int, float]) -> Dict[int | str, float]:
    'Convert competing risks hazards into one-step transition probabilities.'
    H = float(sum(max(0.0, float(v)) for v in hazards.values()))
    if H <= 0.0:
        return {**{int(k): 0.0 for k in hazards.keys()}, "event": 0.0, "stay": 1.0}

    p_event = 1.0 - np.exp(-H)
    out: Dict[int | str, float] = {}
    for k, h in hazards.items():
        h = max(0.0, float(h))
        out[int(k)] = (h / H) * p_event
    out["event"] = float(p_event)
    out["stay"] = float(1.0 - p_event)
    return out


def sample_next_state(
    rng: np.random.Generator, current_state: int, probs: Dict[int | str, float], targets: List[int]
) -> int:
    "Sample next state among 'targets' using probs[target], with remaining mass treated as 'stay'."
    u = float(rng.uniform(0.0, 1.0))
    cum = 0.0
    for tgt in targets:
        cum += float(probs.get(int(tgt), 0.0))
        if u < cum:
            return int(tgt)
    return int(current_state)


def softplus(x: float) -> float:
    'Numerically stable softplus for scalar inputs.'
    x = float(x)
    if x > 30.0:
        return x
    if x < -30.0:
        return float(np.exp(x))
    return float(np.log1p(np.exp(x)))


def sigmoid(x: float) -> float:
    'Numerically stable sigmoid for scalar inputs.'
    x = float(np.clip(x, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def simulate_icml_pomdp_burden(
    n: int = 8000,
    max_time: int = 50,
    seed: int = 42,

    I_rho: float = 0.95,
    I_sigma: float = 0.35,

    meas_sd: float = 1.0,

    z_fast_sd: float = 1.0,

    burden_rho: float = 0.92,
    burden_alpha: float = 0.25,
    burden_clip: float = 8.0,

    frailty_sd: float = 0.12,

    death_b0: float = 2.2,
    death_k: float = 2.2,
    export_oracle: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    'Simulate n independent subjects.'
    rng = np.random.default_rng(seed)

    terminals = {3}
    outgoing = {1: [2, 3], 2: [1, 3], 3: [3]}


    bh_12 = make_baseline_series(max_time, lam0=0.018, lam_slope=0.000)
    bh_21 = make_baseline_series(max_time, lam0=0.070, lam_slope=0.000)
    bh_13 = make_baseline_series(max_time, lam0=0.0006, lam_slope=0.0012)
    bh_23 = make_baseline_series(max_time, lam0=0.0040, lam_slope=0.0030)

    rows: List[dict] = []
    oracle_rows: Optional[List[dict]] = [] if export_oracle else None

    for sid in range(int(n)):

        x1 = int(rng.integers(0, 2))
        x2 = float(rng.normal(0.0, 1.0))


        state = 1
        I = float(rng.normal(0.0, 1.0))


        z2 = float(rng.normal(0.0, z_fast_sd))
        z3 = float(rng.normal(0.0, z_fast_sd))


        burden = 0.0
        frailty = float(rng.normal(0.0, frailty_sd))


        cum_sick = 0
        relapses = 0

        for t in range(int(max_time)):

            z1 = float(I + rng.normal(0.0, meas_sd))


            if state in terminals:
                rows.append(
                    dict(
                        id=sid,
                        t=t,
                        state=3,
                        next_state=3,
                        x1=x1,
                        x2=x2,
                        z1=z1,
                        z2=z2,
                        z3=z3,
                        p_event=0.0,
                    )
                )
                if export_oracle and oracle_rows is not None:
                    oracle_rows.append(
                        dict(
                            id=sid,
                            t=t,
                            I_oracle=float(I),
                            burden_oracle=float(burden),
                            frailty_oracle=float(frailty),
                            cum_sick_oracle=int(cum_sick),
                            relapses_oracle=int(relapses),
                        )
                    )
                break

            hazards: Dict[int, float] = {}


            death_boost = sigmoid(death_k * (burden - death_b0))
            death_boost = 1.0 + 6.0 * death_boost

            if state == 1:

                eta_12 = (
                    0.03 * x1
                    + 0.01 * x2
                    + 0.02 * z1
                    - 0.01 * z2
                    + 0.90 * burden
                    + 0.05 * np.log1p(relapses)
                    + 0.40 * frailty
                )
                eta_12 = float(np.clip(eta_12, -6.0, 6.0))
                hazards[2] = _safe_baseline_at(bh_12, t) * float(np.exp(eta_12))


                eta_13 = 0.25 * frailty + 0.10 * np.log1p(burden)
                eta_13 = float(np.clip(eta_13, -6.0, 6.0))
                hazards[3] = _safe_baseline_at(bh_13, t) * float(np.exp(eta_13)) * float(death_boost)

            elif state == 2:

                eta_21 = (
                    0.08 * x1
                    - 0.20 * z1
                    + 0.05 * (-z2)
                    - 0.70 * burden
                    - 0.05 * np.log1p(cum_sick)
                    - 0.20 * frailty
                )
                eta_21 = float(np.clip(eta_21, -6.0, 6.0))
                hazards[1] = _safe_baseline_at(bh_21, t) * float(np.exp(eta_21))


                eta_23 = (
                    0.02 * x2
                    + 1.05 * burden
                    + 0.08 * np.log1p(cum_sick)
                    + 0.06 * relapses
                    + 0.30 * frailty
                )
                eta_23 = float(np.clip(eta_23, -6.0, 6.0))
                hazards[3] = _safe_baseline_at(bh_23, t) * float(np.exp(eta_23)) * float(death_boost)

            probs = competing_risks_probs(hazards)
            next_state = sample_next_state(rng, state, probs, outgoing[state])

            rows.append(
                dict(
                    id=sid,
                    t=t,
                    state=int(state),
                    next_state=int(next_state),
                    x1=x1,
                    x2=x2,
                    z1=z1,
                    z2=z2,
                    z3=z3,
                    p_event=float(probs.get("event", 0.0)),
                )
            )
            if export_oracle and oracle_rows is not None:
                oracle_rows.append(
                    dict(
                        id=sid,
                        t=t,
                        I_oracle=float(I),
                        burden_oracle=float(burden),
                        frailty_oracle=float(frailty),
                        cum_sick_oracle=int(cum_sick),
                        relapses_oracle=int(relapses),
                    )
                )


            if state == 2:
                cum_sick += 1
            if state == 1 and next_state == 2:
                relapses += 1


            exposure = softplus(I)
            burden = float(burden_rho * burden + burden_alpha * exposure)
            burden = float(np.clip(burden, 0.0, burden_clip))


            I = float(I_rho * I + rng.normal(0.0, I_sigma))


            z2 = float(0.60 * z2 + rng.normal(0.0, z_fast_sd * 0.50))
            z3 = float(0.60 * z3 + rng.normal(0.0, z_fast_sd * 0.50))
            z2 = float(np.clip(z2, -4.0, 4.0))
            z3 = float(np.clip(z3, -4.0, 4.0))

            state = int(next_state)

    df = pd.DataFrame(rows)
    oracle_df = pd.DataFrame(oracle_rows) if export_oracle and oracle_rows is not None else None
    return df, oracle_df


def print_landmark_survival(df: pd.DataFrame, landmarks: Iterable[int] = (10, 20, 30, 40)) -> None:
    'Simple diagnostics: how many subjects are alive (non-terminal) at selected landmarks.'
    print("\n--- Landmark survival diagnostics ---")
    print("max t in df:", int(df["t"].max()))
    for L in landmarks:
        n_rows = int((df["t"] == L).sum())
        n_alive = int(df[(df["t"] == L) & (df["state"] != 3)]["id"].nunique())
        print(f"t=={int(L)}: rows={n_rows} | alive subjects={n_alive}")


def make_external_test_and_5fold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    test_frac: float = 0.20,
    seed: int = 123,
) -> Dict[str, object]:
    'Create subject-level splits to prevent temporal leakage.'
    if "id" not in df.columns:
        raise ValueError("df must contain an 'id' column")

    rng = np.random.default_rng(seed)

    all_ids = np.array(sorted(df["id"].unique()), dtype=int)
    if len(all_ids) < n_folds:
        raise ValueError(f"Need at least {n_folds} unique ids for {n_folds}-fold CV")

    rng.shuffle(all_ids)

    n_test = int(np.floor(test_frac * len(all_ids)))
    n_test = max(1, n_test)
    test_ids = all_ids[:n_test]
    trainval_ids = all_ids[n_test:]

    rng.shuffle(trainval_ids)
    folds_ids = np.array_split(trainval_ids, n_folds)

    folds: List[dict] = []
    for k in range(n_folds):
        val_ids = folds_ids[k]
        train_ids = np.concatenate([folds_ids[j] for j in range(n_folds) if j != k]) if n_folds > 1 else trainval_ids
        folds.append(
            dict(
                fold=int(k),
                train_ids=np.array(sorted(train_ids), dtype=int),
                val_ids=np.array(sorted(val_ids), dtype=int),
            )
        )

    return dict(
        test_ids=np.array(sorted(test_ids), dtype=int),
        trainval_ids=np.array(sorted(trainval_ids), dtype=int),
        folds=folds,
    )


def slice_df_by_ids(df: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    'Return a copy of df containing only rows whose id is in ids.'
    ids_set = set(int(x) for x in ids.tolist())
    return df[df["id"].astype(int).isin(ids_set)].copy()


if __name__ == "__main__":
    df, oracle = simulate_icml_pomdp_burden(n=8000, max_time=50, seed=42, export_oracle=True)
    print(f"Generated rows={len(df)} | subjects={df['id'].nunique()}")
    print_landmark_survival(df)


    splits = make_external_test_and_5fold_splits(df, n_folds=5, test_frac=0.20, seed=2026)

    df_test = slice_df_by_ids(df, splits["test_ids"])
    print(f"\nExternal test: ids={len(splits['test_ids'])} | rows={len(df_test)}")

    df_test.to_csv("simulated_icml_pomdp_burden_test.csv", index=False)
    print("Wrote: simulated_icml_pomdp_burden_test.csv")

    for fold in splits["folds"]:
        k = int(fold["fold"])
        df_train = slice_df_by_ids(df, fold["train_ids"])
        df_val = slice_df_by_ids(df, fold["val_ids"])


        assert set(df_train["id"].unique()).isdisjoint(set(df_val["id"].unique()))
        assert set(df_train["id"].unique()).isdisjoint(set(df_test["id"].unique()))
        assert set(df_val["id"].unique()).isdisjoint(set(df_test["id"].unique()))

        train_path = f"simulated_icml_pomdp_burden_fold{k}_train.csv"
        val_path = f"simulated_icml_pomdp_burden_fold{k}_val.csv"
        df_train.to_csv(train_path, index=False)
        df_val.to_csv(val_path, index=False)
        print(f"Wrote: {train_path} | {val_path}")

    df.to_csv("simulated_icml_pomdp_burden.csv", index=False)
    print("\nWrote: simulated_icml_pomdp_burden.csv")

    if oracle is not None:
        oracle.to_csv("simulated_icml_pomdp_burden_oracle.csv", index=False)
        print("Wrote (diagnostics only): simulated_icml_pomdp_burden_oracle.csv")
