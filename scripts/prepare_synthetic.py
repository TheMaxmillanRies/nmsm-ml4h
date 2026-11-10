"Generate the paper cohort; do not use the generator's legacy external split."
from pathlib import Path
import argparse
import subprocess
import sys
from synthetic_generator import simulate_icml_pomdp_burden


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    destination = args.output/"simulated_icml_pomdp_burden.csv"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    frame, _ = simulate_icml_pomdp_burden(n=8000, max_time=50, seed=42, export_oracle=False)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    subprocess.run([sys.executable, str(Path(__file__).with_name("build_protocol.py")),
        "--dataset", "synthetic", "--data", str(destination), "--output", str(args.output/"protocol_full"), "--seed", "2026"], check=True)


if __name__ == "__main__":
    main()
