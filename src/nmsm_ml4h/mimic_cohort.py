from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MIMIC_FEATURES


CATEGORICAL = ("gender", "first_careunit", "race", "insurance", "marital_status", "language")
NUMERIC = tuple(x for x in MIMIC_FEATURES if not x.endswith("_code"))


def normalize_category(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip()
    return text if text else "UNKNOWN"


def construct_state(frame: pd.DataFrame) -> np.ndarray:
    'Exact submitted priority: death > ICU discharge > INV > NIV > HFNC > none.'
    state = np.zeros(len(frame), dtype=np.int8)
    state[pd.to_numeric(frame["highflow"], errors="coerce").fillna(0).to_numpy() == 1] = 1
    state[pd.to_numeric(frame["noninvasive"], errors="coerce").fillna(0).to_numpy() == 1] = 2
    state[pd.to_numeric(frame["invasive"], errors="coerce").fillna(0).to_numpy() == 1] = 3
    state[pd.to_numeric(frame["icuouttime_outcome"], errors="coerce").fillna(0).to_numpy() == 1] = 4
    state[pd.to_numeric(frame["death_outcome"], errors="coerce").fillna(0).to_numpy() == 1] = 5
    return state


def process_patient(path: Path, censor_time: int) -> pd.DataFrame:
    required = {
        "subject_id", "invasive", "noninvasive", "highflow",
        "icuouttime_outcome", "death_outcome", *CATEGORICAL, *NUMERIC,
    }
    wanted = required | {"hr"}


    frame = pd.read_csv(path, usecols=lambda column: column in wanted, nrows=censor_time + 1)


    if "hr" in frame and len(frame) and int(pd.to_numeric(frame["hr"], errors="raise").iloc[0]) != 0:
        frame = pd.read_csv(path, usecols=lambda column: column in wanted)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "hr" not in frame:
        frame["hr"] = np.arange(len(frame), dtype=int)
    else:
        frame["hr"] = pd.to_numeric(frame["hr"], errors="raise")
    frame = frame.sort_values("hr", kind="stable")
    frame = frame[frame["hr"] <= censor_time].copy()
    if frame.empty:
        raise ValueError(f"{path} contains no rows at or before {censor_time}")
    state = construct_state(frame)
    terminal = np.flatnonzero((state == 4) | (state == 5))
    if len(terminal):
        end = int(terminal[0])
        frame = frame.iloc[: end + 1].copy()
        state = state[: end + 1]
        censored = False
    else:
        censored = True
    output = pd.DataFrame(
        {
            "id": pd.to_numeric(frame["subject_id"], errors="raise").astype(np.int64),
            "t": pd.to_numeric(frame["hr"], errors="raise").astype(np.int16),
            "state": state.astype(np.int8),
        }
    )
    next_state = np.roll(state, -1).astype(np.int16)
    next_state[-1] = -1 if censored else int(state[-1])
    output["next_state"] = next_state
    for column in CATEGORICAL:
        output[column] = frame[column].map(normalize_category).to_numpy()
    for column in NUMERIC:
        output[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
    output["censored_trajectory"] = censored
    return output


def build_mimic_cohort(
    raw_root: Path,
    output_path: Path,
    encoding_path: Path,
    audit_path: Path,
    censor_time: int = 200,
    workers: int = 8,
) -> dict:
    files = sorted(raw_root.glob("[1-9][0-9][0-9]/*.csv"))
    if not files:
        raise FileNotFoundError(f"No per-patient CSV files found under {raw_root}")
    pieces = []
    process = partial(process_patient, censor_time=censor_time)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, piece in enumerate(executor.map(process, files), start=1):
            pieces.append(piece)
            if index % 1000 == 0:
                print(f"cohort files={index}/{len(files)}", flush=True)
    cohort = pd.concat(pieces, ignore_index=True)
    encodings: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL:
        values = sorted(set(cohort[column].astype(str).unique()) | {"UNKNOWN"})
        mapping = {value: index for index, value in enumerate(values)}
        encodings[column] = mapping
        cohort[f"{column}_code"] = cohort[column].map(mapping).astype(np.int16)
    cohort = cohort.drop(columns=list(CATEGORICAL))
    ordered = ["id", "t", "state", "next_state", *MIMIC_FEATURES, "censored_trajectory"]
    cohort = cohort.loc[:, ordered].sort_values(["id", "t"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(output_path, index=False)
    encoding_path.parent.mkdir(parents=True, exist_ok=True)
    encoding_path.write_text(json.dumps(encodings, indent=2, sort_keys=True), encoding="utf-8")
    last = cohort.groupby("id", sort=False).tail(1)
    audit = {
        "source_root": str(raw_root.resolve()),
        "source_files": len(files),
        "patients": int(cohort["id"].nunique()),
        "rows": int(len(cohort)),
        "administrative_censor_time": censor_time,
        "endpoint_counts": {
            "icu_discharge": int((last["state"] == 4).sum()),
            "death": int((last["state"] == 5).sum()),
            "censored": int(last["censored_trajectory"].sum()),
        },
        "censored_final_next_state_minus_one": bool(
            (last.loc[last["censored_trajectory"], "next_state"] == -1).all()
        ),
        "all_source_patients_retained": int(cohort["id"].nunique()) == len(files),
        "max_time": int(cohort["t"].max()),
    }
    if not audit["all_source_patients_retained"]:
        raise AssertionError("Cohort construction dropped source patients")
    if not audit["censored_final_next_state_minus_one"]:
        raise AssertionError("Censored endpoints must not be labeled as observed self-transitions")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit

