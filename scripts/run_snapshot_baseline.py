from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nmsm_ml4h.config import MIMIC_CONFIG, SIMULATION_CONFIG
from nmsm_ml4h.evaluation import score_predictions
from nmsm_ml4h.snapshot_baseline import run_snapshot_fold
from nmsm_ml4h.targets import TargetProtocol


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mimic", "synthetic"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fold-table", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    frame = load_frame(args.data)
    folds = pd.read_csv(args.fold_table)
    targets = pd.read_parquet(args.targets)
    config = MIMIC_CONFIG if args.dataset == "mimic" else SIMULATION_CONFIG
    protocol = (
        TargetProtocol((24, 48, 84), 48, (4, 5), 6)
        if args.dataset == "mimic"
        else TargetProtocol((15, 24, 37), 12, (2,), 3)
    )
    for fold in args.folds:
        fold_dir = args.output / f"fold_{fold}"
        prediction = run_snapshot_fold(
            frame, folds, targets, fold, config, fold_dir, args.epochs, args.seed
        )
        fold_target = targets[targets["fold"] == fold]
        metrics, calibration, calibration_models, summary = score_predictions(
            prediction, fold_target, protocol
        )
        metrics.to_csv(fold_dir / "metrics.csv", index=False)
        calibration.to_csv(fold_dir / "calibration.csv", index=False)
        calibration_models.to_csv(fold_dir / "calibration_intercept_slope.csv", index=False)
        (fold_dir / "summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

