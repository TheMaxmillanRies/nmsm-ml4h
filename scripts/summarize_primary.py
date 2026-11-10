'Score complete primary runs.'
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import pandas as pd
from scipy.stats import t
from nmsm_ml4h.evaluation import score_predictions, validate_predictions
from nmsm_ml4h.metrics import integrated_expected_calibration_error
from nmsm_ml4h.targets import TargetProtocol

METHODS = {"nmsm": "nMSM-T", "gru": "nMSM-GRU", "snapshot": "Direct MLP", "mscox": "msCox",
           "survnode": "SurvNODE", "idnetwork": "IDNetwork", "mspseudo": "msPseudo", "mixmarkov_2022": "MixMarkov"}
METRICS = ("ibs", "integrated_auc", "integrated_nll", "integrated_mae", "ece")


def prediction_path(runs, dataset, method, fold):
    if method == "nmsm":
        return runs/f"primary/{dataset}_fold_{fold}/{dataset}/fold_{fold}/mean_focused/outer_test/predictions_internal.parquet"
    if method == "gru":
        return runs/f"{dataset}/nmsm_gru/fold_{fold}"/("outer_test/predictions_internal.parquet" if dataset == "mimic" else "predictions.parquet")
    if method == "snapshot":
        return runs/f"{dataset}/direct_mlp/fold_{fold}/predictions.parquet"
    return runs/f"{dataset}/baselines/{method}/fold_{fold}/predictions.parquet"


def from_runs(runs, data):
    paths = [prediction_path(runs, ds, method, fold) for ds in ("synthetic_full", "mimic") for method in METHODS for fold in range(5)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing primary predictions (no reference-score fallback):\n"+"\n".join(missing))
    rows = []
    for ds, landmarks, horizon, terminals, states, target in [
        ("synthetic_full", (15,24,37),12,(2,),3,"synthetic/protocol_full/targets.parquet"),
        ("mimic", (24,48,84),48,(4,5),6,"mimic/protocol/targets.parquet")]:
        targets = pd.read_parquet(data/target)
        protocol = TargetProtocol(landmarks,horizon,terminals,states)
        for method in METHODS:
            for fold in range(5):
                prediction = pd.read_parquet(prediction_path(runs, ds, method, fold))
                truth = targets[targets.fold.eq(fold)]
                validate_predictions(prediction, truth, protocol)
                for label, landmark in zip(("Q25", "Q50", "Q75"),landmarks):
                    pred, tr = prediction[prediction.landmark.eq(landmark)], truth[truth.landmark.eq(landmark)]
                    _, calibration, _, summary = score_predictions(pred, tr, TargetProtocol((landmark,),horizon,terminals,states))
                    row = {"dataset": ds, "method": method, "fold": fold, "landmark": label, "time": landmark}
                    row.update({key: summary.get(key, np.nan) for key in METRICS})
                    row["ece"] = integrated_expected_calibration_error(calibration)
                    if method == "mspseudo":
                        row["integrated_nll"] = np.nan
                    rows.append(row)
    return pd.DataFrame(rows)


def statistics(frame):
    result = []
    for (ds,lm),group in frame.groupby(["dataset","landmark"]):
        for metric in METRICS:
            means = group.groupby("method")[metric].mean().dropna()
            best = means.idxmax() if metric == "integrated_auc" else means.idxmin()
            reference = group[group.method.eq(best)].sort_values("fold")[metric].to_numpy()
            family = []
            for method in means.index:
                if method == best:
                    continue
                values = group[group.method.eq(method)].sort_values("fold")[metric].to_numpy()
                difference = values-reference
                se = np.sqrt(.45*np.var(difference,ddof=1))
                p = float(2*t.sf(abs(difference.mean()/se),4)) if se else float(difference.mean()==0)
                family.append({"dataset": ds, "landmark": lm, "metric": metric, "best": best,
                    "method": method, "difference": float(difference.mean()), "p": p})
            previous = 0.
            for i,row in enumerate(sorted(family,key=lambda x:x["p"])):
                previous = max(previous,min(1.,(len(family)-i)*row["p"]))
                row["p_holm"] = previous
                row["not_significantly_different"] = previous >= .05
                result.append(row)
    return pd.DataFrame(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=ROOT/"runs")
    p.add_argument("--data-root", type=Path, default=ROOT/"data")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frame = from_runs(args.runs,args.data_root)
    if len(frame)!=240 or frame.duplicated(["dataset","method","fold","landmark"]).any():
        raise ValueError("Expected both datasets, eight methods, five folds and three landmarks")
    args.output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(args.output/"fold_metrics.csv",index=False)
    frame.groupby(["dataset","landmark","method"])[list(METRICS)].agg(["mean","std"]).to_csv(args.output/"mean_sd.csv")
    statistics(frame).to_csv(args.output/"corrected_tests.csv",index=False)
    (args.output/"STATUS.json").write_text(json.dumps({"source": "new predictions", "rows": len(frame)},indent=2))
    print(f"Wrote {len(frame)} fold/landmark rows")


if __name__ == "__main__":
    main()
