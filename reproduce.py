'Portable primary-paper entry point. Commands are previews unless --execute is given.'
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
METHODS = ("nmsm_t", "nmsm_gru", "direct_mlp", "mscox", "survnode", "idnetwork", "mspseudo", "mixmarkov_2022")


def commands(args):
    data = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    if args.command == "prepare-synthetic":
        return [["scripts/prepare_synthetic.py", "--output", str(data/"synthetic")]]
    if args.command == "prepare-mimic":
        if not args.raw_root:
            raise ValueError("--raw-root must point at the credentialed respiratory-support dataset's files directory")
        raw, clean = data/"mimic/mimic_all_patients_t200.parquet", data/"mimic/mimic_clean.parquet"
        return [
            ["scripts/build_mimic_cohort.py", "--raw-root", str(Path(args.raw_root).resolve()), "--output", str(raw),
             "--encodings", str(data/"mimic/encodings.json"), "--audit", str(data/"mimic/cohort_audit.json"), "--workers", "2"],
            ["scripts/clean_mimic_features.py", "--input", str(raw), "--output", str(clean), "--audit", str(data/"mimic/cleaning_audit.json")],
            ["scripts/build_protocol.py", "--dataset", "mimic", "--data", str(clean), "--output", str(data/"mimic/protocol")],
        ]
    ds = args.dataset
    data_file = data/("mimic/mimic_clean.parquet" if ds == "mimic" else "synthetic/simulated_icml_pomdp_burden.csv")
    protocol = data/("mimic/protocol" if ds == "mimic" else "synthetic/protocol_full")
    common = ["--dataset", "mimic" if ds == "mimic" else "synthetic", "--data", str(data_file),
              "--fold-table", str(protocol/"outer_folds.csv"), "--targets", str(protocol/"targets.parquet")]
    reference = json.loads((ROOT/"configs/reported_selections.json").read_text())
    jobs = []
    for method in args.methods:
        for fold in args.folds:
            if method == "nmsm_t":
                jobs.append(["scripts/train_main_mean_focused.py", "--execute", "--stage", args.stage,
                    "--objectives", "mean_focused", "--datasets", ds, "--folds", str(fold),
                    "--data-root", str(data), "--output", str(output/"primary"/f"{ds}_fold_{fold}"),
                    "--device", args.device, "--microbatch-size", "16", "--path-batch-size", "128",
                    "--patient-batch-size", "4", "--max-concurrent-paths", "512",
                    "--cpu-affinity", args.cpu_affinity])
            elif method == "nmsm_gru" and ds == "mimic":
                jobs.append(["scripts/train_mimic_gru.py", "--data-root", str(data), "--output", str(output/"mimic/nmsm_gru"),
                    "--primary-root", str(output/"primary"), "--fold", str(fold), "--device", args.device, "--stage", args.stage])
            elif method == "nmsm_gru":
                if args.stage != "all":
                    raise ValueError("Synthetic GRU and baselines have a single train-and-evaluate stage; use --stage all")
                jobs.append(["scripts/run_gru_baseline.py", *common, "--output", str(output/ds/"nmsm_gru"),
                    "--folds", str(fold), "--epochs", "30", "--seed", "2026", "--lambda-cov", "0.1",
                    "--simulations", "500", "--simulation-batch-size", "128", "--patient-batch-size", "4", "--max-concurrent-paths", "512"])
            elif method == "direct_mlp":
                if args.stage != "all":
                    raise ValueError("Baselines require --stage all")
                jobs.append(["scripts/run_snapshot_baseline.py", *common, "--output", str(output/ds/method),
                    "--folds", str(fold), "--epochs", "60" if ds == "mimic" else "30", "--seed", "2026"])
            else:
                if args.stage != "all":
                    raise ValueError("Baselines require --stage all")
                job = ["scripts/run_common_baselines.py", *common, "--output", str(output/ds/"baselines"),
                    "--methods", method, "--folds", str(fold), "--seed", "2026", "--skip-combine"]
                if args.device == "cpu":
                    job.append("--cpu-only")
                if method == "survnode":
                    s = reference["survnode"][ds][fold]
                    mapping = {"epochs-survnode": "maximum_epochs", "patience-survnode": "early_stopping_patience",
                        "survnode-batch-size": "logical_training_batch_size", "survnode-microbatch-size": "microbatch_size",
                        "survnode-learning-rate": "learning_rate", "survnode-encoder-width": "encoder_hidden_features",
                        "survnode-ode-width": "ode_hidden_features", "survnode-solver": "solver_method",
                        "survnode-solver-rtol": "solver_rtol", "survnode-solver-atol": "solver_atol",
                        "survnode-gradient-clip-norm": "gradient_clip_norm", "survnode-optimizer": "optimizer_name"}
                    for flag, key in mapping.items():
                        job += ["--"+flag, str(s[key])]
                jobs.append(job)
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare-synthetic", "prepare-mimic", "train"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dataset", choices=("synthetic_full", "mimic"), default="synthetic_full")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    p.add_argument("--stage", choices=("fit", "evaluate", "all"), default="all")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--data-root", default=str(ROOT/"data"))
    p.add_argument("--output", default=str(ROOT/"runs"))
    p.add_argument("--raw-root")
    p.add_argument("--cpu-affinity", default="0,1")
    args = p.parse_args()
    jobs = commands(args)
    print(json.dumps({"execute": args.execute, "sequential": True, "commands": jobs}, indent=2), flush=True)
    if not args.execute:
        return
    if args.command == "prepare-mimic" and (Path(args.data_root)/"mimic").exists():
        raise FileExistsError("MIMIC output directory already exists; choose a fresh --data-root instead of overwriting it")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT/"src") + os.pathsep + str(ROOT/"scripts")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
        env[key] = "2"
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    for job in jobs:
        subprocess.run([sys.executable, *job], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
