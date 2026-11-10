from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nmsm_ml4h.splits import audit_folds, make_outer_folds
from nmsm_ml4h.targets import TargetProtocol, build_target_registry


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mimic", "synthetic"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--folds",
        type=Path,
        help="Optional frozen outer_folds.csv. Required for paired observation-layer variants.",
    )
    args = parser.parse_args()

    frame = load_frame(args.data)
    if args.dataset == "mimic":
        protocol = TargetProtocol((24, 48, 84), 48, (4, 5), 6)
        split_terminals = (4, 5)
        state_map = None
    else:
        protocol = TargetProtocol((15, 24, 37), 12, (2,), 3)
        split_terminals = (3,)
        state_map = {1: 0, 2: 1, 3: 2}
    folds = (
        pd.read_csv(args.folds)
        if args.folds is not None
        else make_outer_folds(frame, split_terminals, seed=args.seed)
    )
    audit_folds(folds, frame["id"].nunique())
    targets = build_target_registry(frame, folds, protocol, state_map=state_map)

    args.output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output / "outer_folds.csv", index=False)
    targets.to_parquet(args.output / "targets.parquet", index=False)
    by_fold = targets.groupby("fold").agg(
        patients=("id", "nunique"), rows=("id", "size"), known_fraction=("known", "mean"),
        max_ipcw=("ipcw", "max"),
    )
    audit = {
        "dataset": args.dataset,
        "source": str(args.data.resolve()),
        "patients": int(frame["id"].nunique()),
        "rows": int(len(frame)),
        "seed": args.seed,
        "fold_source": str(args.folds.resolve()) if args.folds is not None else "generated",
        "landmarks": list(protocol.landmarks),
        "horizons": [1, protocol.max_horizon],
        "terminal_states_internal": list(protocol.terminal_states),
        "target_rows": int(len(targets)),
        "eligible_patient_landmarks": int(len(targets) / protocol.max_horizon),
        "by_fold": by_fold.reset_index().to_dict(orient="records"),
        "ipcw_convention": "outer-training reverse KM, conditional G(L-)/G(U-)",
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
