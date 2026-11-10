'Run identical baseline implementations on MIMIC, simulation, or SAIL.'

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from nmsm_ml4h.baseline_registry import (
    PRIMARY_BASELINES,
    BaselineSettings,
    run_baseline_fold,
)
from nmsm_ml4h.config import (
    MIMIC_CONFIG,
    SIMULATION_CONFIG,
    sail_config_from_manifest,
)
from nmsm_ml4h.evaluation import (
    attach_targets,
    score_predictions,
    score_predictions_by_origin,
    validate_predictions,
)
from nmsm_ml4h.metrics import (
    ipcw_loess_calibration_rows,
    ipcw_multinomial_calibration_rows,
)
from nmsm_ml4h.targets import KEYS, TargetProtocol, validate_target_registry


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _configuration(dataset: str, manifest: Path | None):
    if dataset == "mimic":
        return MIMIC_CONFIG
    if dataset == "synthetic":
        return SIMULATION_CONFIG
    if manifest is None:
        raise ValueError("--manifest is required for SAIL")
    return sail_config_from_manifest(manifest)


def _save_score(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    protocol: TargetProtocol,
    output: Path,
) -> None:
    predictions = predictions.sort_values(KEYS).reset_index(drop=True)
    targets = targets.sort_values(KEYS).reset_index(drop=True)
    validate_predictions(predictions, targets, protocol)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    metrics, calibration, calibration_models, summary = score_predictions(
        predictions, targets, protocol
    )
    (
        origin_metrics,
        origin_calibration,
        origin_calibration_models,
        origin_summary,
        origin_macro,
    ) = score_predictions_by_origin(predictions, targets, protocol)
    joined = attach_targets(predictions, targets, protocol)
    selected_horizons = tuple(
        sorted({1, max(1, protocol.max_horizon // 2), protocol.max_horizon})
    )
    moderate_calibration = ipcw_loess_calibration_rows(
        joined, protocol.num_states, horizons=selected_horizons
    )
    multinomial_calibration = ipcw_multinomial_calibration_rows(
        joined,
        protocol.num_states,
        horizons=selected_horizons,
    )
    summary.update(origin_macro)
    metrics.to_csv(output / "metrics.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    calibration_models.to_csv(
        output / "calibration_intercept_slope.csv", index=False
    )
    origin_metrics.to_csv(
        output / "metrics_by_origin_landmark_horizon.csv", index=False
    )
    origin_calibration.to_csv(output / "calibration_by_origin.csv", index=False)
    origin_calibration_models.to_csv(
        output / "calibration_intercept_slope_by_origin.csv", index=False
    )
    origin_summary.to_csv(output / "summary_by_origin.csv", index=False)
    moderate_calibration.to_csv(
        output / "moderate_calibration_blr_ipcw.csv", index=False
    )
    multinomial_calibration.to_csv(
        output / "moderate_calibration_mlr_ipcw.csv", index=False
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mimic", "synthetic", "sail"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fold-table", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=PRIMARY_BASELINES, default=list(PRIMARY_BASELINES))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs-survnode", type=int, default=500)
    parser.add_argument("--epochs-idnetwork", type=int, default=50)
    parser.add_argument("--epochs-mspseudo", type=int, default=10000)
    parser.add_argument(
        "--mixmarkov-max-iterations",
        type=int,
        default=None,
        help=(
            "Optional smoke-test cap. Production default is uncapped, matching "
            "the official scipy.optimize.minimize call."
        ),
    )
    parser.add_argument("--patience-idnetwork", type=int, default=8)
    parser.add_argument("--patience-mspseudo", type=int, default=100)
    parser.add_argument("--patience-survnode", type=int, default=20)
    parser.add_argument(
        "--mixmarkov-samples",
        type=int,
        default=100,
        help="Posterior transition-matrix draws; 100 matches the official default.",
    )
    parser.add_argument("--survnode-training-fraction", type=float, default=1.0)
    parser.add_argument("--survnode-batch-size", type=int, default=512)
    parser.add_argument("--survnode-microbatch-size", type=int, default=512)
    parser.add_argument("--survnode-learning-rate", type=float, default=1e-3)
    parser.add_argument("--survnode-encoder-width", type=int, default=800)
    parser.add_argument("--survnode-ode-width", type=int, default=1000)
    parser.add_argument("--survnode-solver", choices=("dopri5", "rk4"), default="dopri5")
    parser.add_argument("--survnode-solver-rtol", type=float, default=1e-8)
    parser.add_argument("--survnode-solver-atol", type=float, default=1e-8)
    parser.add_argument("--survnode-gradient-clip-norm", type=float)
    parser.add_argument("--survnode-optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help=(
            "Write only the requested per-fold outputs. This permits a guarded "
            "outer launcher to resume at fold boundaries and combine once all "
            "five fresh folds are complete."
        ),
    )
    args = parser.parse_args()

    if args.dataset != "sail" and args.survnode_training_fraction != 1.0:
        parser.error("MIMIC and synthetic SurvNODE must use the full outer-training partition")
    if args.dataset == "sail" and args.survnode_training_fraction not in (0.2, 1.0):
        parser.error("SAIL SurvNODE must explicitly use either the documented 20% or full training")
    if not set(args.folds).issubset(set(range(5))):
        parser.error("--folds must be drawn from 0 1 2 3 4")

    config = _configuration(args.dataset, args.manifest)
    protocol = TargetProtocol(
        config.landmarks,
        config.max_horizon,
        config.terminal_states,
        config.num_states,
    )
    frame = _load_frame(args.data)
    folds = pd.read_csv(args.fold_table)
    targets = pd.read_parquet(args.targets)
    validate_target_registry(targets, protocol)
    settings = BaselineSettings(
        epochs_survnode=args.epochs_survnode,
        epochs_idnetwork=args.epochs_idnetwork,
        epochs_mspseudo=args.epochs_mspseudo,
        mixmarkov_max_iterations=args.mixmarkov_max_iterations,
        patience_idnetwork=args.patience_idnetwork,
        patience_mspseudo=args.patience_mspseudo,
        patience_survnode=args.patience_survnode,
        mixmarkov_samples=args.mixmarkov_samples,
        survnode_training_fraction=args.survnode_training_fraction,
        survnode_batch_size=args.survnode_batch_size,
        survnode_microbatch_size=args.survnode_microbatch_size,
        survnode_learning_rate=args.survnode_learning_rate,
        survnode_encoder_width=args.survnode_encoder_width,
        survnode_ode_width=args.survnode_ode_width,
        survnode_solver=args.survnode_solver,
        survnode_solver_rtol=args.survnode_solver_rtol,
        survnode_solver_atol=args.survnode_solver_atol,
        survnode_gradient_clip_norm=args.survnode_gradient_clip_norm,
        survnode_optimizer=args.survnode_optimizer,
        seed=args.seed,
    )
    device = torch.device(
        "cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda"
    )
    print(
        f"COMMON_BASELINES dataset={args.dataset} methods={','.join(args.methods)} "
        f"folds={args.folds} device={device}",
        flush=True,
    )
    for method in args.methods:
        for fold in args.folds:
            output = args.output / method / f"fold_{fold}"
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    f"Fresh runner refuses non-empty output directory: {output}"
                )
            output.mkdir(parents=True, exist_ok=True)
            print(f"METHOD START method={method} fold={fold}", flush=True)
            predictions = run_baseline_fold(
                method,
                frame,
                folds,
                targets,
                fold,
                config,
                output,
                settings,
                device,
            )
            fold_targets = targets[targets["fold"] == fold]
            _save_score(predictions, fold_targets, protocol, output)
            print(f"METHOD DONE method={method} fold={fold}", flush=True)
        if not args.skip_combine:
            selected = [
                args.output / method / f"fold_{fold}" / "predictions.parquet"
                for fold in args.folds
            ]
            combined = pd.concat(
                [pd.read_parquet(path) for path in selected], ignore_index=True
            )
            combined_output = args.output / method / "combined_selected_folds"
            combined_output.mkdir(parents=True, exist_ok=False)
            selected_targets = targets[targets["fold"].isin(args.folds)]
            _save_score(combined, selected_targets, protocol, combined_output)
            print(f"COMBINE DONE method={method}", flush=True)


if __name__ == "__main__":
    main()
