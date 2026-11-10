from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nmsm_ml4h.evaluation import (
    attach_targets,
    score_predictions,
    score_predictions_by_origin,
)
from nmsm_ml4h.metrics import (
    ipcw_loess_calibration_rows,
    ipcw_multinomial_calibration_rows,
)
from nmsm_ml4h.targets import TargetProtocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mimic", "synthetic"), required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-moderate-calibration",
        action="store_true",
        help="Write core/IPCW metrics only; final submission runs must omit this flag.",
    )
    args = parser.parse_args()
    protocol = (
        TargetProtocol((24, 48, 84), 48, (4, 5), 6)
        if args.dataset == "mimic"
        else TargetProtocol((15, 24, 37), 12, (2,), 3)
    )
    targets = pd.read_parquet(args.targets)
    predictions = pd.read_parquet(args.predictions)
    rows, calibration, calibration_models, summary = score_predictions(predictions, targets, protocol)
    (
        origin_rows,
        origin_calibration,
        origin_calibration_models,
        origin_summaries,
        origin_macro,
    ) = score_predictions_by_origin(predictions, targets, protocol)
    summary.update(origin_macro)
    joined = attach_targets(predictions, targets, protocol)
    calibration_horizons = tuple(
        sorted({1, max(1, protocol.max_horizon // 2), protocol.max_horizon})
    )
    if args.skip_moderate_calibration:
        moderate_calibration = pd.DataFrame()
        multinomial_calibration = pd.DataFrame()
    else:
        moderate_calibration = ipcw_loess_calibration_rows(
            joined,
            protocol.num_states,
            horizons=calibration_horizons,
        )
        multinomial_calibration = ipcw_multinomial_calibration_rows(
            joined,
            protocol.num_states,
            horizons=calibration_horizons,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output / "metrics_by_landmark_horizon.csv", index=False)
    calibration.to_csv(args.output / "calibration.csv", index=False)
    calibration_models.to_csv(args.output / "calibration_intercept_slope.csv", index=False)
    origin_rows.to_csv(
        args.output / "metrics_by_origin_landmark_horizon.csv", index=False
    )
    origin_calibration.to_csv(
        args.output / "calibration_by_origin.csv", index=False
    )
    origin_calibration_models.to_csv(
        args.output / "calibration_intercept_slope_by_origin.csv", index=False
    )
    origin_summaries.to_csv(args.output / "summary_by_origin.csv", index=False)
    moderate_calibration.to_csv(
        args.output / "moderate_calibration_blr_ipcw.csv", index=False
    )
    multinomial_calibration.to_csv(
        args.output / "moderate_calibration_mlr_ipcw.csv", index=False
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
