'Prepare by default. Only --execute can fit models or evaluate real datasets.'
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
try:
    import psutil
except ImportError:
    psutil = None
import torch
from torch.utils.data import DataLoader

from nmsm_ml4h.config import MIMIC_CONFIG, SIMULATION_CONFIG
from nmsm_ml4h.data import FeatureEncoder, collate_trajectories
from nmsm_ml4h.evaluation import score_predictions, validate_predictions
from nmsm_ml4h.mean_training import (
    ComputeBudget, ResourcePause, evaluate_factual, file_hash, fingerprint, train_from_scratch, write_json,
)
from nmsm_ml4h.model import JointCausalTransformer, ModelConfig, allowed_transitions
from nmsm_ml4h.rollout import rollout_predictions
from nmsm_ml4h.splits import audit_folds, split_train_validation
from nmsm_ml4h.targets import TargetProtocol
from nmsm_ml4h.training import make_dataset
from optimize_nmsm import candidates, configured, TARGET_COLUMNS, rank

METHODS = ("nll", "mean_focused")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly start the requested stage; absent means print plan only")
    parser.add_argument("--stage", choices=("fit", "evaluate", "all"), default="fit")
    parser.add_argument("--datasets", nargs="+", choices=("synthetic_full", "mimic"), default=["synthetic_full", "mimic"])
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--objectives", nargs="+", choices=METHODS, default=list(METHODS),
                        help="Run a prespecified subset; use mean_focused alone when existing NLL folds are the control")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/main_mean_focused_20260906.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/main_mean_focused_full_20260906")
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--path-batch-size", type=int, default=16)
    parser.add_argument("--patient-batch-size", type=int, default=1)
    parser.add_argument("--max-concurrent-paths", type=int, default=16)
    parser.add_argument("--cpu-affinity", type=str, default="0,1",
                        help="Comma-separated logical CPUs; concurrent fold shards must use disjoint sets")
    parser.add_argument("--min-gpu-free-gib", type=float, default=6.)
    parser.add_argument("--max-gpu-gib", type=float, default=6.)
    parser.add_argument("--min-ram-free-gib", type=float, default=12.)
    parser.add_argument("--max-process-ram-gib", type=float, default=12.)
    args = parser.parse_args(argv)
    if (len(args.datasets) != len(set(args.datasets)) or len(args.folds) != len(set(args.folds))
            or len(args.objectives) != len(set(args.objectives))):
        parser.error("Duplicate datasets/folds/objectives are not allowed")
    try:
        args.cpu_affinity = tuple(int(value) for value in args.cpu_affinity.split(","))
    except ValueError:
        parser.error("--cpu-affinity must be comma-separated nonnegative integers")
    if not args.cpu_affinity or len(args.cpu_affinity) != len(set(args.cpu_affinity)) or min(args.cpu_affinity) < 0:
        parser.error("--cpu-affinity must contain unique nonnegative logical CPUs")
    if (args.microbatch_size < 1 or not 1 <= args.path_batch_size <= 128
            or not 1 <= args.patient_batch_size <= 8
            or args.max_concurrent_paths < args.path_batch_size * args.patient_batch_size):
        parser.error("Invalid batching: path batch <=128, patient batch <=8, and their product must fit the concurrency cap")
    limits = (args.min_gpu_free_gib, args.max_gpu_gib, args.min_ram_free_gib, args.max_process_ram_gib)
    if not all(np.isfinite(x) and x > 0 for x in limits) or args.min_gpu_free_gib < 6:
        parser.error("Resource limits must be positive and the GPU reserve must be at least 6 GiB")
    return args


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol["version"] != 1 or protocol["objectives"] != list(METHODS):
        raise ValueError("Protocol must contain both matched objectives")
    weights = protocol["covariate_weights"]
    if not weights or len(weights) != len(set(weights)) or any(not np.isfinite(x) or x <= 0 for x in weights):
        raise ValueError("Invalid weight grid")
    temperatures = protocol["temperatures"]
    if 1. not in temperatures or len(temperatures) != len(set(temperatures)) or any(not np.isfinite(x) or x <= 0 for x in temperatures):
        raise ValueError("Temperature grid must be unique, positive and include 1")
    if protocol["checkpoint_metric"] != "state":
        raise ValueError("Both objectives must use the prespecified state-CE checkpoint rule")
    for key in ("patience", "selection_paths", "outer_paths"):
        if type(protocol[key]) is not int or protocol[key] <= 0:
            raise ValueError("Invalid positive integer: " + key)
    for dataset in ("synthetic_full", "mimic"):
        names = protocol["selected_candidates"][dataset]
        if len(names) != 5 or set(names) - {c.name for c in candidates(dataset)}:
            raise ValueError("Invalid frozen architecture choices")
        if type(protocol["epochs"][dataset]) is not int or protocol["epochs"][dataset] < 1:
            raise ValueError("Invalid epoch cap")
    return protocol


def paths(data_root, dataset):
    if dataset == "mimic":
        root = data_root / "mimic"
        return root / "mimic_clean.parquet", root / "protocol/outer_folds.csv", root / "protocol/targets.parquet"
    root = data_root / "synthetic"
    return root / "simulated_icml_pomdp_burden.csv", root / "protocol_full/outer_folds.csv", root / "protocol_full/targets.parquet"


def configuration(protocol, dataset, fold, weight):
    candidate_name = protocol["selected_candidates"][dataset][fold]
    candidate = next(c for c in candidates(dataset) if c.name == candidate_name)
    base = MIMIC_CONFIG if dataset == "mimic" else SIMULATION_CONFIG
    return replace(configured(base, candidate), lambda_cov=weight, epochs=protocol["epochs"][dataset])


def make_plan(args, protocol):
    jobs = [{"dataset": dataset, "fold": fold,
             "architecture": protocol["selected_candidates"][dataset][fold],
             "objective": method, "covariate_weight": weight,
             "epochs": protocol["epochs"][dataset]}
            for dataset in args.datasets for fold in args.folds
            for method in args.objectives for weight in protocol["covariate_weights"]]
    return {"execution_enabled": args.execute, "stage": args.stage,
            "fits": len(jobs), "jobs": jobs, "sequential": True,
            "initialization": "from scratch; no pretrained weights", "cpu_threads": 2,
            "data_root": str(args.data_root.resolve()), "output": str(args.output.resolve()),
            "device": args.device, "microbatch_size": args.microbatch_size,
            "objectives": args.objectives, "path_batch_size": args.path_batch_size,
            "patient_batch_size": args.patient_batch_size,
            "max_concurrent_paths": args.max_concurrent_paths,
            "cpu_affinity": args.cpu_affinity,
            "gpu_free_reserve_gib": args.min_gpu_free_gib, "gpu_allocator_cap_gib": args.max_gpu_gib,
            "outer_outcomes_used_by_fit_stage": False,
            "note": "No training, data loading, output writing, GPU initialization, remote access or scheduling without --execute."}


def partition(frame, folds, fold, seed):
    audit_folds(folds, frame.id.nunique())
    if set(frame.id) != set(folds.id):
        raise ValueError("Cohort and fold IDs differ")
    training_ids, validation_ids = split_train_validation(folds[folds.outer_fold != fold], seed=seed + fold)
    test_ids = folds.loc[folds.outer_fold == fold, "id"].to_numpy()
    sets = [set(training_ids), set(validation_ids), set(test_ids)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Patient leakage")
    return training_ids, validation_ids, test_ids


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "RUNNING.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("Output is locked. Check RUNNING.lock before resuming; never start concurrent folds in this root.") from exc
    with handle:
        handle.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
    try:
        yield
    finally:
        lock.unlink()


def load_model(path, guard):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = JointCausalTransformer(ModelConfig(**checkpoint["architecture"]))
    model.load_state_dict(checkpoint["model"])
    guard.check(force=True)
    model.to(guard.device).eval()
    return model, FeatureEncoder.from_state_dict(checkpoint["feature_encoder"])


def forecast(checkpoint_path, evaluation, censoring_reference, config, fold, sims, temperature,
             seed, output, guard, path_batch_size, run_id, targets=None,
             patient_batch_size=1, max_concurrent_paths=16):
    spec = {"run_id": run_id, "checkpoint_sha256": file_hash(checkpoint_path), "paths": sims,
            "temperature": temperature, "seed": seed, "path_batch_size": path_batch_size,
            "patient_batch_size": patient_batch_size, "max_concurrent_paths": max_concurrent_paths}
    marker = output / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["spec"] != spec or saved["prediction_sha256"] != file_hash(output / "predictions_internal.parquet"):
            raise ValueError("Cached forecast differs from frozen specification")
        return saved["summary"]
    model, encoder = load_model(checkpoint_path, guard)
    hook = model.state_head.register_forward_pre_hook(guard.hook)
    try:
        predictions, seconds = rollout_predictions(
            model, encoder, evaluation, censoring_reference, config, "original", fold,
            n_sims=sims, seed=seed, simulation_batch_size=path_batch_size,
            patient_batch_size=patient_batch_size, max_concurrent_paths=max_concurrent_paths,
            covariate_mode="joint", observation_mode="learned",
            continuous_mode="sample", state_temperature=temperature, gpu_yield_seconds=.002)
    finally:
        hook.remove()
    target_protocol = TargetProtocol(config.landmarks, config.max_horizon, config.terminal_states, config.num_states)
    if targets is None:
        targets = predictions[TARGET_COLUMNS].copy()
    validate_predictions(predictions, targets, target_protocol)
    metrics, calibration, calibration_models, summary = score_predictions(predictions, targets, target_protocol)


    rank(summary)
    undefined = [key for key, value in summary.items()
                 if isinstance(value, (float, np.floating)) and not np.isfinite(value)]
    summary = {key: None if key in undefined else value for key, value in summary.items()}
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "predictions_internal.parquet", index=False)
    metrics.to_csv(output / "metrics_by_horizon.csv", index=False)
    calibration.to_csv(output / "calibration_bins.csv", index=False)
    calibration_models.to_csv(output / "calibration_models.csv", index=False)
    write_json(marker, {"spec": spec, "summary": summary, "undefined_metrics": undefined, "runtime_seconds": seconds,
                       "prediction_sha256": file_hash(output / "predictions_internal.parquet")})
    return summary


def clear_unused(guard):
    gc.collect()
    if guard.device.type == "cuda":
        torch.cuda.empty_cache()
    guard.check(force=True)


def fit_and_select(args, protocol, dataset, fold, train, validation, folder, guard, run_id):
    results = {}
    for method in args.objectives:
        results[method] = []
        for weight in protocol["covariate_weights"]:
            config = configuration(protocol, dataset, fold, weight)
            directory = folder / method / f"lambda_{weight:g}"
            fitted = train_from_scratch(train, validation, config, method, directory,
                protocol["fit_seed"] + 1000 * fold, args.microbatch_size, protocol["patience"], guard, run_id)
            clear_unused(guard)
            summary = forecast(directory / "best.pt", validation, train, config, fold,
                protocol["selection_paths"], 1., protocol["selection_seed"] + 1000 * fold,
                directory / "validation_T1", guard, args.path_batch_size, run_id,
                patient_batch_size=args.patient_batch_size,
                max_concurrent_paths=args.max_concurrent_paths)
            results[method].append({"weight": weight, "fit": fitted, "validation": summary})
            clear_unused(guard)

    fits = [r["fit"] for rows in results.values() for r in rows]
    if len({fit["initial_state_sha256"] for fit in fits}) != 1:
        raise AssertionError("Unmatched random initialization")
    for epoch in range(min(fit["epochs_run"] for fit in fits)):
        if len({fit["history"][epoch]["batch_order_sha256"] for fit in fits}) != 1:
            raise AssertionError("Unmatched minibatch order")
    selections = {}
    for method in args.objectives:
        best = min(results[method], key=lambda row: (*rank(row["validation"]), row["weight"]))
        weight = best["weight"]
        config = configuration(protocol, dataset, fold, weight)
        directory = folder / method / f"lambda_{weight:g}"
        temperatures = [{"temperature": 1., "validation": best["validation"]}]
        for temperature in protocol["temperatures"]:
            if temperature == 1.:
                continue
            summary = forecast(directory / "best.pt", validation, train, config, fold,
                protocol["selection_paths"], temperature, protocol["selection_seed"] + 1000 * fold,
                directory / f"validation_T{temperature:g}", guard, args.path_batch_size, run_id,
                patient_batch_size=args.patient_batch_size,
                max_concurrent_paths=args.max_concurrent_paths)
            temperatures.append({"temperature": temperature, "validation": summary})
            clear_unused(guard)
        chosen = min(temperatures, key=lambda row: (*rank(row["validation"]), abs(row["temperature"] - 1.)))
        selections[method] = {"weight": weight, "temperature": chosen["temperature"],
            "checkpoint": f"{method}/lambda_{weight:g}/best.pt", "checkpoint_sha256": file_hash(directory / "best.pt"),
            "validation": chosen["validation"], "weight_results": results[method], "temperature_results": temperatures}
    selection = {"run_id": run_id, "outer_test_used_for_selection": False,
                 "selection_order": "state-CE checkpoint; weight by T1 validation IBS/NLL/-iAUC; temperature on validation",
                 "matched_initialization_and_minibatches": len(fits) > 1, "methods": selections}
    write_json(folder / "selection.json", selection)
    return selection


def evaluate_selected(args, protocol, dataset, fold, frame, folds, test_ids, folder, guard, run_id):
    selection = json.loads((folder / "selection.json").read_text())
    if selection["run_id"] != run_id or selection["outer_test_used_for_selection"] is not False:
        raise ValueError("Both within-fold selections must be frozen before evaluation")
    _, _, target_path = paths(args.data_root, dataset)
    targets = pd.read_parquet(target_path)
    targets = targets[targets.fold == fold].copy()
    test = frame[frame.id.isin(test_ids)].copy()
    reference = frame[frame.id.isin(folds.loc[folds.outer_fold != fold, "id"])].copy()
    outputs = []
    for method in args.objectives:
        chosen = selection["methods"][method]
        checkpoint = folder / chosen["checkpoint"]
        if file_hash(checkpoint) != chosen["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint was modified")
        config = configuration(protocol, dataset, fold, chosen["weight"])
        output = folder / method / "outer_test"
        summary = forecast(checkpoint, test, reference, config, fold, protocol["outer_paths"],
            chosen["temperature"], protocol["outer_seed"] + 1000 * fold,
            output, guard, args.path_batch_size, run_id, targets=targets,
            patient_batch_size=args.patient_batch_size,
            max_concurrent_paths=args.max_concurrent_paths)
        clear_unused(guard)
        model, encoder = load_model(checkpoint, guard)
        loader = DataLoader(make_dataset(test, config, encoder, "original"), batch_size=args.microbatch_size,
                            shuffle=False, collate_fn=collate_trajectories, num_workers=0)
        graph = allowed_transitions(config.num_states, config.terminal_states, config.allowed_edges).to(guard.device)
        factual, variable = evaluate_factual(model, loader, graph, guard.device, config.lambda_cov, guard)
        write_json(output / "factual.json", factual)
        rows = []
        for j, name in enumerate(encoder.dynamic_continuous_columns):
            count, bias, sse, covered = variable[:, j]
            rows.append({"covariate": name, "count": count, "standardized_bias": bias / count if count else None,
                         "standardized_rmse": np.sqrt(sse / count) if count else None,
                         "coverage95": covered / count if count else None})
        pd.DataFrame(rows).to_csv(output / "factual_covariates.csv", index=False)
        outputs.append({"dataset": dataset, "fold": fold, "objective": method,
                        "lambda_cov": chosen["weight"], "temperature": chosen["temperature"], **summary,
                        **{f"factual_{key}": value for key, value in factual.items()}})
        del model, encoder, loader, graph
        clear_unused(guard)
    pd.DataFrame(outputs).to_csv(folder / "comparison.csv", index=False)


def code_hashes():
    files = [*sorted((ROOT / "src/nmsm_ml4h").glob("*.py")), Path(__file__), ROOT / "scripts/optimize_nmsm.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path) for path in files}


def execute(args, protocol):
    torch.set_num_threads(len(args.cpu_affinity))
    torch.set_num_interop_threads(1)
    import pyarrow as pa
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    if psutil is not None:
        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else max(process.nice(), 10))
        if hasattr(process, "cpu_affinity"):
            available = set(process.cpu_affinity())
            if not set(args.cpu_affinity) <= available:
                raise ResourcePause(f"Requested CPU affinity {args.cpu_affinity} is unavailable; have {sorted(available)}")
            process.cpu_affinity(list(args.cpu_affinity))
    elif os.name != "nt":
        os.nice(10)
        if hasattr(os, "sched_getaffinity"):
            available_cpus = set(os.sched_getaffinity(0))
            if not set(args.cpu_affinity) <= available_cpus:
                raise ResourcePause(f"Requested CPU affinity {args.cpu_affinity} is unavailable; have {sorted(available_cpus)}")
            os.sched_setaffinity(0, set(args.cpu_affinity))
    else:
        raise RuntimeError("psutil is required for Windows resource controls")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ResourcePause("Requested CUDA device is unavailable; no automatic CPU fallback")
    guard = ComputeBudget(args.device, min_gpu_free_gib=args.min_gpu_free_gib, max_gpu_gib=args.max_gpu_gib,
                          min_ram_free_gib=args.min_ram_free_gib, max_ram_gib=args.max_process_ram_gib)
    guard.initialize()
    source_hashes = code_hashes()
    with output_lock(args.output):
        try:
            for dataset in args.datasets:
                data_path, fold_path, target_path = paths(args.data_root, dataset)
                for path in (data_path, fold_path, target_path):
                    if not path.is_file():
                        raise FileNotFoundError("Required existing benchmark input: " + str(path))


                manifest = {"protocol": protocol, "code_sha256": source_hashes,
                    "inputs_sha256": {"data": file_hash(data_path), "folds": file_hash(fold_path), "targets": file_hash(target_path)},
                    "dataset": dataset, "microbatch_size": args.microbatch_size,
                    "path_batch_size": args.path_batch_size,
                    "patient_batch_size": args.patient_batch_size,
                    "max_concurrent_paths": args.max_concurrent_paths,
                    "objectives": args.objectives, "folds": args.folds,
                    "device": args.device,
                    "torch_version": str(torch.__version__)}
                run_id = fingerprint(manifest)
                manifest_path = args.output / dataset / "manifest.json"
                if manifest_path.exists() and fingerprint(json.loads(manifest_path.read_text())) != run_id:
                    raise ValueError("Data/code/protocol changed: use a new output directory")
                write_json(manifest_path, manifest)
                frame = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
                folds = pd.read_csv(fold_path)
                for fold in args.folds:
                    guard.check(force=True)
                    train_ids, valid_ids, test_ids = partition(frame, folds, fold, protocol["split_seed"])
                    folder = args.output / dataset / f"fold_{fold}"
                    write_json(folder / "split_audit_INTERNAL.json", {"training_ids": list(map(int, train_ids)),
                        "validation_ids": list(map(int, valid_ids)), "outer_test_ids": list(map(int, test_ids)),
                        "feature_encoder_fit_scope": "inner training only", "disjoint": True})
                    if args.stage in ("fit", "all"):
                        train = frame[frame.id.isin(train_ids)].copy()
                        validation = frame[frame.id.isin(valid_ids)].copy()
                        fit_and_select(args, protocol, dataset, fold, train, validation, folder, guard, run_id)
                        del train, validation
                        clear_unused(guard)
                    if args.stage in ("evaluate", "all"):
                        evaluate_selected(args, protocol, dataset, fold, frame, folds, test_ids, folder, guard, run_id)
                    write_json(folder / f"{args.stage}_complete.json", {"run_id": run_id, "status": "complete"})
                comparison_paths = sorted((args.output / dataset).glob("fold_*/comparison.csv"))
                if comparison_paths:
                    combined = pd.concat([pd.read_csv(path) for path in comparison_paths], ignore_index=True)
                    combined.to_csv(args.output / dataset / "comparison_by_fold.csv", index=False)
                    numeric = combined.select_dtypes(include="number").columns.difference(["fold", "lambda_cov", "temperature"])
                    combined.groupby("objective")[list(numeric)].agg(["mean", "std"]).to_csv(args.output / dataset / "comparison_mean_sd.csv")
                del frame, folds
                clear_unused(guard)
            if code_hashes() != source_hashes:
                raise RuntimeError("Source code changed while running")
            write_json(args.output / "status.json", {"status": "complete", "stage": args.stage,
                        "datasets": args.datasets, "folds": args.folds, "resources": guard.summary()})
        except (ResourcePause, torch.cuda.OutOfMemoryError, KeyboardInterrupt) as exc:
            write_json(args.output / "status.json", {"status": "paused", "reason": str(exc),
                        "resume": "Repeat the identical command after compute is available; incomplete epochs replay from last.pt",
                        "resources": guard.summary()})
            print("PAUSED: " + str(exc), flush=True)
            return 75
    return 0


def main(argv=None):
    args = parse_args(argv)
    protocol = load_protocol(args.protocol)
    print(json.dumps(make_plan(args, protocol), indent=2), flush=True)
    if not args.execute:
        return 0
    try:
        return execute(args, protocol)
    except ResourcePause as exc:
        print("PAUSED BEFORE START: " + str(exc), flush=True)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
