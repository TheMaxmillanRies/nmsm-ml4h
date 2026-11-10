'Reproduce the matched MIMIC GRU; inherit primary validation-selected weight/temperature.'
from pathlib import Path
import argparse
import gc
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
import pandas as pd
import torch
from nmsm_ml4h.data import FeatureEncoder
from nmsm_ml4h.gru_baseline import AutoregressiveGRU, GRUConfig, train_gru_fold, predict_gru_fold
from nmsm_ml4h.evaluation import score_predictions, validate_predictions
from nmsm_ml4h.mean_training import ComputeBudget
from nmsm_ml4h.targets import TargetProtocol
from train_main_mean_focused import load_protocol, configuration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--primary-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--stage", choices=("fit", "evaluate", "all"), default="all")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    guard = ComputeBudget(args.device, min_gpu_free_gib=6, max_gpu_gib=12, min_ram_free_gib=8, max_ram_gib=20)
    guard.initialize()
    protocol = load_protocol(ROOT/"configs/main_mean_focused_20260906.json")
    choice_file = args.primary_root/f"mimic_fold_{args.fold}/mimic/fold_{args.fold}/selection.json"
    selected = json.loads(choice_file.read_text())["methods"]["mean_focused"]
    config = configuration(protocol, "mimic", args.fold, selected["weight"])
    folder = args.output/f"fold_{args.fold}"
    checkpoint = folder/"best.pt"
    frame = pd.read_parquet(args.data_root/"mimic/mimic_clean.parquet")
    folds = pd.read_csv(args.data_root/"mimic/protocol/outer_folds.csv")
    spec = {"fold": args.fold, "weight": selected["weight"], "temperature": selected["temperature"], "objective": "mean_focused"}
    spec_file = folder/"release_spec.json"
    if args.stage in ("fit", "all"):
        if folder.exists() and any(folder.iterdir()):
            if not ((folder/"training.json").exists() and checkpoint.exists() and spec_file.exists() and json.loads(spec_file.read_text()) == spec):
                raise FileExistsError("Incomplete or mismatched GRU output; use a new directory, do not overwrite a fit")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            spec_file.write_text(json.dumps(spec, indent=2))
            model, encoder = train_gru_fold(frame, folds, args.fold, config, folder, config.epochs,
                protocol["fit_seed"], protocol["patience"], guard.device,
                objective="mean_focused", split_seed=protocol["split_seed"])
            del model
            gc.collect()
    if args.stage == "fit":
        return
    if not spec_file.exists() or json.loads(spec_file.read_text()) != spec:
        raise ValueError("Saved GRU and selected primary settings differ")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = AutoregressiveGRU(GRUConfig(**payload["architecture"]))
    model.load_state_dict(payload["model"])
    model.to(guard.device).eval()
    encoder = FeatureEncoder.from_state_dict(payload["feature_encoder"])
    target = pd.read_parquet(args.data_root/"mimic/protocol/targets.parquet", filters=[("fold", "=", args.fold)])
    output = folder/"outer_test"
    if output.exists():
        raise FileExistsError("Outer evaluation exists; use the saved predictions, or explicitly choose a fresh output")
    started = time.monotonic()
    hook = model.state_head.register_forward_pre_hook(guard.hook)
    try:
        prediction = predict_gru_fold(model, encoder, frame[frame.id.isin(target.id.unique())], target,
            config, 500, protocol["outer_seed"]+1000*args.fold,
            simulation_batch_size=500, patient_batch_size=16, max_concurrent_paths=8000,
            state_temperature=selected["temperature"])
    finally:
        hook.remove()
    target_protocol = TargetProtocol(config.landmarks, config.max_horizon, config.terminal_states, config.num_states)
    validate_predictions(prediction, target, target_protocol)
    metrics, calibration, calibration_models, summary = score_predictions(prediction, target, target_protocol)
    output.mkdir()
    prediction.to_parquet(output/"predictions_internal.parquet", index=False)
    metrics.to_csv(output/"metrics_by_horizon.csv", index=False)
    calibration.to_csv(output/"calibration.csv", index=False)
    calibration_models.to_csv(output/"calibration_models.csv", index=False)
    (output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"elapsed_seconds": time.monotonic()-started, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
