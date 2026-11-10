from __future__ import annotations

import argparse
from pathlib import Path

from nmsm_ml4h.mimic_cohort import build_mimic_cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/mimic/mimic_all_patients_t200.parquet"))
    parser.add_argument("--encodings", type=Path, default=Path("data/mimic/encodings.json"))
    parser.add_argument("--audit", type=Path, default=Path("reports/mimic_cohort_audit.json"))
    parser.add_argument("--censor-time", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(build_mimic_cohort(args.raw_root, args.output, args.encodings, args.audit, args.censor_time, args.workers))


if __name__ == "__main__":
    main()
