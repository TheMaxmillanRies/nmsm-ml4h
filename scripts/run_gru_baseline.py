from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from nmsm_ml4h.config import MIMIC_CONFIG, SIMULATION_CONFIG
from nmsm_ml4h.evaluation import score_predictions
from nmsm_ml4h.gru_likelihood_baseline import predict_gru_fold, train_gru_fold
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
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--simulation-batch-size", type=int, default=32)
    parser.add_argument("--patient-batch-size", type=int, default=2)
    parser.add_argument("--gpu-yield-ms", type=float, default=0.0)
    parser.add_argument("--max-concurrent-paths", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lambda-cov", type=float)
    parser.add_argument(
        "--covariate-parameterization",
        choices=("absolute", "residual"),
        default="absolute",
    )
    args = parser.parse_args()

    frame = load_frame(args.data)
    folds = pd.read_csv(args.fold_table)
    all_targets = pd.read_parquet(args.targets)
    if args.dataset == "mimic":
        config = MIMIC_CONFIG
        protocol = TargetProtocol((24, 48, 84), 48, (4, 5), 6)
        epochs = args.epochs or 40
    else:
        config = SIMULATION_CONFIG
        protocol = TargetProtocol((15, 24, 37), 12, (2,), 3)
        epochs = args.epochs or 30
    if args.lambda_cov is not None:
        config = replace(config, lambda_cov=float(args.lambda_cov))
    config = replace(
        config, covariate_parameterization=args.covariate_parameterization
    )
    for fold in args.folds:
        fold_dir = args.output / f"fold_{fold}"
        model, encoder = train_gru_fold(
            frame,
            folds,
            fold,
            config,
            fold_dir,
            epochs,
            args.seed,
        )
        target = all_targets[all_targets["fold"] == fold].copy()
        predictions = predict_gru_fold(
            model,
            encoder,
            frame,
            target,
            config,
            args.simulations,
            args.seed + 1000 * fold,
            args.simulation_batch_size,
            args.patient_batch_size,
            args.gpu_yield_ms / 1000.0,
            args.max_concurrent_paths,
        )
        predictions.to_parquet(fold_dir / "predictions.parquet", index=False)
        metrics, calibration, calibration_models, summary = score_predictions(
            predictions, target, protocol
        )
        metrics.to_csv(fold_dir / "metrics.csv", index=False)
        calibration.to_csv(fold_dir / "calibration.csv", index=False)
        calibration_models.to_csv(fold_dir / "calibration_intercept_slope.csv", index=False)
        (fold_dir / "summary.json").write_text(
            pd.Series(summary).to_json(indent=2), encoding="utf-8"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
