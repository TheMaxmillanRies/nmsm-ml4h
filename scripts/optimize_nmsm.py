from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from nmsm_ml4h.config import MIMIC_CONFIG, SIMULATION_CONFIG, ExperimentConfig
from nmsm_ml4h.data import FeatureEncoder
from nmsm_ml4h.evaluation import score_predictions, validate_predictions
from nmsm_ml4h.metrics import integrated_metrics
from nmsm_ml4h.model import JointCausalTransformer, ModelConfig
from nmsm_ml4h.rollout import rollout_predictions
from nmsm_ml4h.splits import audit_folds, split_train_validation
from nmsm_ml4h.targets import KEYS, TargetProtocol
from nmsm_ml4h.training import train_fold


TARGET_COLUMNS = [
    *KEYS,
    "target_time",
    "origin_state",
    "target_state",
    "known",
    "observation_time",
    "ipcw",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    learning_rate: float
    weight_decay: float


def candidates(dataset: str) -> list[Candidate]:
    if dataset == "mimic":
        return [
            Candidate("canonical", 256, 8, 4, 256, 0.10, 1e-4, 1e-5),
            Candidate("compact", 128, 8, 3, 256, 0.10, 2e-4, 1e-5),
            Candidate("wider_ff", 256, 8, 3, 512, 0.10, 1e-4, 1e-5),
            Candidate("regularized", 256, 8, 4, 256, 0.20, 1e-4, 1e-4),
        ]
    return [
        Candidate("canonical", 64, 4, 2, 128, 0.10, 1e-3, 0.0),
        Candidate("small", 32, 4, 2, 64, 0.10, 1e-3, 0.0),
        Candidate("wide_slow", 128, 4, 2, 256, 0.10, 5e-4, 0.0),
        Candidate("regularized", 64, 4, 2, 128, 0.20, 1e-3, 1e-4),
    ]


def dataset_paths(dataset: str) -> tuple[Path, Path, Path]:
    if dataset == "mimic":
        root = PROJECT / "data" / "mimic"
        return root / "mimic_clean.parquet", root / "protocol" / "outer_folds.csv", root / "protocol" / "targets.parquet"
    root = PROJECT / "data" / "synthetic"
    if dataset == "synthetic_full":
        return root / "simulated_icml_pomdp_burden.csv", root / "protocol_full" / "outer_folds.csv", root / "protocol_full" / "targets.parquet"
    return root / "simulated_icml_pomdp_burden_censored.parquet", root / "protocol_censored" / "outer_folds.csv", root / "protocol_censored" / "targets.parquet"


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def protocol(config: ExperimentConfig) -> TargetProtocol:
    return TargetProtocol(config.landmarks, config.max_horizon, config.terminal_states, config.num_states)


def configured(base: ExperimentConfig, candidate: Candidate) -> ExperimentConfig:
    return replace(
        base,
        lambda_cov=0.1,
        covariate_parameterization="absolute",
        d_model=candidate.d_model,
        nhead=candidate.nhead,
        num_layers=candidate.num_layers,
        dim_feedforward=candidate.dim_feedforward,
        dropout=candidate.dropout,
        learning_rate=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )


def load_checkpoint(directory: Path, device: torch.device) -> tuple[JointCausalTransformer, FeatureEncoder]:
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    model = JointCausalTransformer(ModelConfig(**checkpoint["architecture"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, FeatureEncoder.from_state_dict(checkpoint["feature_encoder"])


def score_and_save(predictions: pd.DataFrame, targets: pd.DataFrame, config: ExperimentConfig, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    validate_predictions(predictions, targets, protocol(config))
    rows, calibration, calibration_models, summary = score_predictions(predictions, targets, protocol(config))
    predictions.to_parquet(output / "predictions_internal.parquet", index=False)
    rows.to_csv(output / "metrics_by_horizon.csv", index=False)
    by_landmark = []
    labels = dict(zip(config.landmarks, ("Q25", "Q50", "Q75")))
    for landmark, group in rows.groupby("landmark", sort=True):
        by_landmark.append({"landmark_label": labels[int(landmark)], "landmark": int(landmark), **integrated_metrics(group)})
    pd.DataFrame(by_landmark).to_csv(output / "results_by_landmark.csv", index=False)
    calibration.to_csv(output / "calibration_bins.csv", index=False)
    calibration_models.to_csv(output / "calibration_models.csv", index=False)
    dump_json(output / "summary.json", summary)
    return summary


def rank(summary: dict) -> tuple[float, float, float]:
    values = (float(summary["ibs"]), float(summary["integrated_nll"]), -float(summary["integrated_auc"]))
    if not np.isfinite(values).all():
        raise FloatingPointError(f"Non-finite validation metrics: {summary}")
    return values


def rollout_and_score(
    model: JointCausalTransformer,
    encoder: FeatureEncoder,
    evaluation: pd.DataFrame,
    training: pd.DataFrame,
    config: ExperimentConfig,
    fold: int,
    simulations: int,
    temperature: float,
    seed: int,
    output: Path,
    targets: pd.DataFrame | None = None,
) -> dict:
    summary_path = output / "summary.json"
    prediction_path = output / "predictions_internal.parquet"
    if summary_path.exists() and prediction_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    predictions, runtime = rollout_predictions(
        model,
        encoder,
        evaluation,
        training,
        config,
        "original",
        fold,
        n_sims=simulations,
        seed=seed,
        simulation_batch_size=min(128, simulations),
        patient_batch_size=4,
        covariate_mode="joint",
        observation_mode="learned",
        continuous_mode="sample",
        state_temperature=temperature,
        max_concurrent_paths=512,
    )
    predictions["selected_temperature"] = float(temperature)
    predictions["runtime_seconds"] = float(runtime)
    if targets is None:
        targets = predictions[TARGET_COLUMNS].copy()
    return score_and_save(predictions, targets, config, output)


def run_fold(args: argparse.Namespace) -> None:
    data_path, folds_path, targets_path = dataset_paths(args.dataset)
    for path in (data_path, folds_path, targets_path):
        if not path.exists():
            raise FileNotFoundError(path)
    base = MIMIC_CONFIG if args.dataset == "mimic" else SIMULATION_CONFIG
    frame = load_frame(data_path)
    folds = pd.read_csv(folds_path)
    all_targets = pd.read_parquet(targets_path)
    audit_folds(folds, frame["id"].nunique())
    outer_training = folds[folds["outer_fold"] != args.fold]
    test_ids = folds.loc[folds["outer_fold"] == args.fold, "id"].to_numpy()
    train_ids, validation_ids = split_train_validation(outer_training, seed=args.seed + args.fold)
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
        raise RuntimeError("Patient leakage between inner training, inner validation and outer test")
    train = frame[frame["id"].isin(train_ids)].copy()
    validation = frame[frame["id"].isin(validation_ids)].copy()
    outer_test = frame[frame["id"].isin(test_ids)].copy()
    fold_dir = args.output / args.dataset / f"fold_{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        fold_dir / "split_audit.json",
        {
            "fold": args.fold,
            "inner_training_patients": int(len(train_ids)),
            "inner_validation_patients": int(len(validation_ids)),
            "outer_test_patients": int(len(test_ids)),
            "outer_test_used_for_selection": False,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_names = set(args.candidates or [item.name for item in candidates(args.dataset)])
    candidate_set = [item for item in candidates(args.dataset) if item.name in selected_names]
    if len(candidate_set) != len(selected_names):
        known = {item.name for item in candidates(args.dataset)}
        raise ValueError(f"Unknown candidates: {sorted(selected_names - known)}")
    results: dict[str, dict] = {}
    for index, candidate in enumerate(candidate_set, start=1):
        config = configured(base, candidate)
        directory = fold_dir / "candidates" / candidate.name
        spec = {
            "candidate": asdict(candidate),
            "lambda_cov": 0.1,
            "covariate_targets": "absolute",
            "checkpoint_metric": "state",
            "seed": args.seed + 1000 * args.fold,
            "epochs": args.epochs,
            "patience": args.patience,
        }
        spec_path = directory / "training_spec.json"
        if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise RuntimeError(f"Cached run has a different specification: {directory}")
        dump_json(spec_path, spec)
        print(f"CANDIDATE {index}/{len(candidate_set)} START dataset={args.dataset} fold={args.fold} name={candidate.name}", flush=True)
        if not (directory / "best.pt").exists() or not (directory / "training.json").exists():
            train_fold(
                train,
                validation,
                config,
                "original",
                directory,
                seed=args.seed + 1000 * args.fold,
                epochs=args.epochs,
                device=device,
                patience=args.patience,
                checkpoint_metric="state",
            )
        model, encoder = load_checkpoint(directory, device)
        summary = rollout_and_score(
            model,
            encoder,
            validation,
            train,
            config,
            args.fold,
            args.selection_sims,
            1.0,
            args.seed + 1000 * args.fold + 101,
            directory / "validation_T1",
        )
        results[candidate.name] = {"candidate": asdict(candidate), "validation": summary}
        dump_json(directory / "candidate_result.json", results[candidate.name])
        print(f"CANDIDATE DONE name={candidate.name} IBS={summary['ibs']:.6f} NLL={summary['integrated_nll']:.6f} iAUC={summary['integrated_auc']:.6f}", flush=True)

    winner_name = min(results, key=lambda name: rank(results[name]["validation"]))
    winner = next(item for item in candidate_set if item.name == winner_name)
    winner_config = configured(base, winner)
    winner_dir = fold_dir / "candidates" / winner_name
    model, encoder = load_checkpoint(winner_dir, device)
    temperatures = []
    for temperature in args.temperatures:
        if np.isclose(temperature, 1.0):
            summary = results[winner_name]["validation"]
        else:
            summary = rollout_and_score(
                model,
                encoder,
                validation,
                train,
                winner_config,
                args.fold,
                args.selection_sims,
                temperature,
                args.seed + 1000 * args.fold + 101,
                winner_dir / f"validation_T{temperature:g}",
            )
        temperatures.append({"temperature": float(temperature), "validation": summary})
    temperature_choice = min(temperatures, key=lambda item: rank(item["validation"]))
    temperature = float(temperature_choice["temperature"])
    selection = {
        "selection_data": "inner validation only",
        "outer_test_used_for_selection": False,
        "selection_order": ["IBS", "NLL", "negative_iAUC"],
        "selected_candidate": asdict(winner),
        "selected_temperature": temperature,
        "candidate_results": results,
        "temperature_results": temperatures,
    }
    dump_json(fold_dir / "selection.json", selection)
    print(f"SELECTED dataset={args.dataset} fold={args.fold} candidate={winner_name} temperature={temperature:g}", flush=True)
    outer_summary = rollout_and_score(
        model,
        encoder,
        outer_test,
        frame[frame["id"].isin(outer_training["id"])].copy(),
        winner_config,
        args.fold,
        args.outer_sims,
        temperature,
        args.seed + 2000 * args.fold,
        fold_dir / "outer_test_selected",
        all_targets[all_targets["fold"] == args.fold].copy(),
    )
    dump_json(fold_dir / "fold_result.json", {"selection": selection, "outer_test": outer_summary})
    print(f"FOLD DONE dataset={args.dataset} fold={args.fold} IBS={outer_summary['ibs']:.6f} NLL={outer_summary['integrated_nll']:.6f} iAUC={outer_summary['integrated_auc']:.6f}", flush=True)


def combine(args: argparse.Namespace) -> None:
    _, _, targets_path = dataset_paths(args.dataset)
    base = MIMIC_CONFIG if args.dataset == "mimic" else SIMULATION_CONFIG
    root = args.output / args.dataset
    paths = [root / f"fold_{fold}" / "outer_test_selected" / "predictions_internal.parquet" for fold in range(5)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete folds: {missing}")
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    targets = pd.read_parquet(targets_path)
    summary = score_and_save(predictions, targets, base, root / "combined_selected_folds")
    selections = []
    for fold in range(5):
        result = json.loads((root / f"fold_{fold}" / "fold_result.json").read_text(encoding="utf-8"))
        selected = result["selection"]["selected_candidate"]
        selections.append({"fold": fold, **selected, "temperature": result["selection"]["selected_temperature"], **result["outer_test"]})
    pd.DataFrame(selections).to_csv(root / "selected_models_and_outer_scores.csv", index=False)
    print(f"COMBINED DONE dataset={args.dataset} IBS={summary['ibs']:.6f} NLL={summary['integrated_nll']:.6f} iAUC={summary['integrated_auc']:.6f} iAP={summary['integrated_average_precision']:.6f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free nMSM-only nested tuning")
    parser.add_argument("--dataset", choices=("mimic", "synthetic_full", "synthetic_censored"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--candidates", nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--selection-sims", type=int, default=100)
    parser.add_argument("--outer-sims", type=int, default=500)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.9, 1.0, 1.1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.combine == (args.fold is not None):
        raise ValueError("Specify exactly one of --fold or --combine")
    if args.epochs is None:
        args.epochs = 60 if args.dataset == "mimic" else 30
    if args.epochs < 1 or args.patience < 1 or args.selection_sims < 1 or args.outer_sims < 1:
        raise ValueError("Epoch, patience and simulation counts must be positive")
    if any(not np.isfinite(value) or value <= 0 for value in args.temperatures):
        raise ValueError("Temperatures must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    if args.combine:
        combine(args)
    else:
        run_fold(args)


if __name__ == "__main__":
    main()
