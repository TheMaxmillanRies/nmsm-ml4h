from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nmsm_ml4h.mimic_cleaning import clean_mimic_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    cleaned, audit = clean_mimic_features(frame)
    key = ["id", "t", "state", "next_state"]
    if not frame[key].equals(cleaned[key]):
        raise AssertionError("Feature cleaning changed trajectory keys or outcomes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.output, index=False)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        f"MIMIC_CLEAN patients={audit['patients']} rows={audit['rows']} "
        f"set_missing={audit['total_set_missing']} clamped={audit['total_clamped']}",
        flush=True,
    )
    for row in audit["feature_audit"]:
        print(
            f"  {row['feature']}: observed={row['observed_before']} "
            f"missing={row['set_missing']} low={row['clamped_low']} "
            f"high={row['clamped_high']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
