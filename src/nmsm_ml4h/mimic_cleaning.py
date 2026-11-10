'Prespecified, non-data-adaptive cleaning for MIMIC physiological inputs.'

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


RANGE_SOURCE = (
    "https://github.com/MLforHealth/MIMIC_Extract/blob/master/"
    "resources/variable_ranges.csv"
)


@dataclass(frozen=True)
class RangeRule:
    outlier_low: float
    valid_low: float
    valid_high: float
    outlier_high: float


MIMIC_RANGE_RULES: dict[str, RangeRule] = {
    "heart_rate": RangeRule(0.0, 0.0, 350.0, 390.0),
    "rr": RangeRule(0.0, 0.0, 300.0, 330.0),
    "sbp": RangeRule(0.0, 0.0, 375.0, 375.0),
    "sbp_ni": RangeRule(0.0, 0.0, 375.0, 375.0),
    "dbp": RangeRule(0.0, 0.0, 375.0, 375.0),
    "dbp_ni": RangeRule(0.0, 0.0, 375.0, 375.0),
    "mbp": RangeRule(0.0, 14.0, 330.0, 375.0),
    "mbp_ni": RangeRule(0.0, 14.0, 330.0, 375.0),
    "temperature": RangeRule(14.2, 26.0, 45.0, 47.0),
    "spo2": RangeRule(0.0, 0.0, 100.0, 150.0),
    "so2": RangeRule(0.0, 0.0, 100.0, 150.0),
    "glucose": RangeRule(0.0, 33.0, 2000.0, 2200.0),
    "calculated_bicarbonate": RangeRule(0.0, 0.0, 60.0, 66.0),
    "pCO2": RangeRule(0.0, 0.0, 200.0, 220.0),
    "pO2": RangeRule(0.0, 32.0, 700.0, 770.0),
    "pH": RangeRule(6.3, 6.3, 8.4, 10.0),
    "height_inch": RangeRule(0.0, 0.0, 240.0 / 2.54, 275.0 / 2.54),
}


def clean_mimic_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    'Apply fixed physiological limits without using outcome or test-fold data.'
    cleaned = frame.copy()
    rows = []
    height_outside_valid = None
    for column, rule in MIMIC_RANGE_RULES.items():
        if column not in cleaned:
            raise ValueError(f"MIMIC frame is missing range-checked feature {column!r}")
        values = pd.to_numeric(cleaned[column], errors="coerce").astype(float)
        observed_before = values.notna()
        strict_outlier = observed_before & (
            (values < rule.outlier_low) | (values > rule.outlier_high)
        )
        valid_low = observed_before & ~strict_outlier & (values < rule.valid_low)
        valid_high = observed_before & ~strict_outlier & (values > rule.valid_high)
        if column == "height_inch":
            height_outside_valid = strict_outlier | valid_low | valid_high
        values.loc[strict_outlier] = np.nan
        values.loc[valid_low] = rule.valid_low
        values.loc[valid_high] = rule.valid_high
        cleaned[column] = values.astype(np.float32)
        rows.append(
            {
                "feature": column,
                "observed_before": int(observed_before.sum()),
                "set_missing": int(strict_outlier.sum()),
                "clamped_low": int(valid_low.sum()),
                "clamped_high": int(valid_high.sum()),
                **asdict(rule),
            }
        )


    if "pbw_kg" not in cleaned:
        raise ValueError("MIMIC frame is missing derived feature 'pbw_kg'")
    assert height_outside_valid is not None
    pbw = pd.to_numeric(cleaned["pbw_kg"], errors="coerce").astype(float)
    pbw_strict_outlier = pbw.notna() & ((pbw < 0.0) | (pbw > 250.0))
    pbw_invalid = height_outside_valid | pbw_strict_outlier
    observed_pbw = pbw.notna()
    pbw.loc[pbw_invalid] = np.nan
    cleaned["pbw_kg"] = pbw.astype(np.float32)
    rows.append(
        {
            "feature": "pbw_kg",
            "observed_before": int(observed_pbw.sum()),
            "set_missing": int((observed_pbw & pbw_invalid).sum()),
            "clamped_low": 0,
            "clamped_high": 0,
            "outlier_low": 0.0,
            "valid_low": 0.0,
            "valid_high": 250.0,
            "outlier_high": 250.0,
        }
    )

    audit_rows = pd.DataFrame(rows)
    audit = {
        "range_source": RANGE_SOURCE,
        "policy": (
            "MIMIC-Extract strict outliers -> missing; values between valid and "
            "strict limits -> clamped; limits are fixed before model fitting"
        ),
        "patients": int(cleaned["id"].nunique()),
        "rows": int(len(cleaned)),
        "feature_audit": audit_rows.to_dict(orient="records"),
        "total_set_missing": int(audit_rows["set_missing"].sum()),
        "total_clamped": int(
            audit_rows["clamped_low"].sum() + audit_rows["clamped_high"].sum()
        ),
    }
    return cleaned, audit
