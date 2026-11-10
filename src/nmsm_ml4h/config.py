from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path


SIMULATION_FEATURES = ("x1", "x2", "z1", "z2", "z3")
MIMIC_FEATURES = (
    "gender_code",
    "first_careunit_code",
    "race_code",
    "insurance_code",
    "marital_status_code",
    "language_code",
    "anchor_age",
    "height_inch",
    "pbw_kg",
    "heart_rate",
    "rr",
    "sbp",
    "dbp",
    "mbp",
    "sbp_ni",
    "dbp_ni",
    "mbp_ni",
    "temperature",
    "spo2",
    "glucose",
    "calculated_bicarbonate",
    "pCO2",
    "pH",
    "pO2",
    "so2",
)


MIMIC_CATEGORICAL_FEATURES = (
    "gender_code",
    "first_careunit_code",
    "race_code",
    "insurance_code",
    "marital_status_code",
    "language_code",
)

MIMIC_NUMERIC_FEATURES = tuple(
    column for column in MIMIC_FEATURES if column not in MIMIC_CATEGORICAL_FEATURES
)

MIMIC_STATIC_NUMERIC_FEATURES = (
    "anchor_age",
    "height_inch",
    "pbw_kg",
)

MIMIC_DYNAMIC_NUMERIC_FEATURES = tuple(
    column for column in MIMIC_NUMERIC_FEATURES if column not in MIMIC_STATIC_NUMERIC_FEATURES
)

SIMULATION_STATIC_FEATURES = ("x1", "x2")
MIMIC_STATIC_FEATURES = (
    "gender_code",
    "first_careunit_code",
    "race_code",
    "insurance_code",
    "marital_status_code",
    "language_code",
    "anchor_age",
    "height_inch",
    "pbw_kg",
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    num_states: int
    terminal_states: tuple[int, ...]
    feature_cols: tuple[str, ...]
    landmarks: tuple[int, ...]
    max_horizon: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    max_time: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    lambda_cov: float
    static_feature_cols: tuple[str, ...] = ()
    numeric_feature_cols: tuple[str, ...] = ()
    categorical_feature_cols: tuple[str, ...] = ()
    dynamic_numeric_feature_cols: tuple[str, ...] = ()
    known_dynamic_numeric_feature_cols: tuple[str, ...] = ()
    known_dynamic_raw_increments: tuple[tuple[str, float], ...] = ()
    observation_numeric_feature_cols: tuple[str, ...] = ()
    binary_dynamic_feature_cols: tuple[str, ...] = ()
    monotone_binary_feature_cols: tuple[str, ...] = ()
    covariate_parameterization: str = "absolute"
    include_missingness: bool = False
    allowed_edges: tuple[tuple[int, int], ...] = ()


SIMULATION_CONFIG = ExperimentConfig(
    name="simulation",
    num_states=3,
    terminal_states=(2,),
    feature_cols=SIMULATION_FEATURES,
    landmarks=(15, 24, 37),
    max_horizon=12,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    max_time=100,
    epochs=30,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=0.0,


    lambda_cov=0.1,
    static_feature_cols=SIMULATION_STATIC_FEATURES,
    numeric_feature_cols=SIMULATION_FEATURES,
    dynamic_numeric_feature_cols=("z1", "z2", "z3"),
    observation_numeric_feature_cols=(),
)


MIMIC_CONFIG = ExperimentConfig(
    name="mimic",
    num_states=6,
    terminal_states=(4, 5),
    feature_cols=MIMIC_FEATURES,
    landmarks=(24, 48, 84),
    max_horizon=48,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=256,
    dropout=0.1,
    max_time=260,
    epochs=60,
    batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-5,
    lambda_cov=0.1,
    static_feature_cols=MIMIC_STATIC_FEATURES,
    numeric_feature_cols=MIMIC_NUMERIC_FEATURES,
    categorical_feature_cols=MIMIC_CATEGORICAL_FEATURES,
    dynamic_numeric_feature_cols=MIMIC_DYNAMIC_NUMERIC_FEATURES,
    observation_numeric_feature_cols=MIMIC_DYNAMIC_NUMERIC_FEATURES,
    include_missingness=True,
)


SAIL_FEATURES = ("age", "efi", "care_home", "sex_code")
SAIL_NUMERIC_FEATURES = ("age", "efi", "care_home")
SAIL_CATEGORICAL_FEATURES = ("sex_code",)
SAIL_DYNAMIC_NUMERIC_FEATURES = ("age", "efi", "care_home")


SAIL_BURDEN_DEFINITION = "hospital admissions within the preceding completed three-month interval"
SAIL_PREPROCESSING_VERSION = "nmsm_ml4h_sail_v5_recurrent_quarterly"


SAIL_ALLOWED_EDGES = tuple(
    (source, destination) for source in range(4) for destination in range(5)
) + ((4, 4),)


SAIL_CONFIG = ExperimentConfig(
    name="sail",
    num_states=5,
    terminal_states=(4,),
    feature_cols=SAIL_FEATURES,


    landmarks=(8, 16, 24),
    max_horizon=6,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    max_time=256,
    epochs=30,
    batch_size=128,
    learning_rate=5e-4,
    weight_decay=1e-5,
    lambda_cov=0.1,
    static_feature_cols=(),
    numeric_feature_cols=SAIL_NUMERIC_FEATURES,
    categorical_feature_cols=SAIL_CATEGORICAL_FEATURES,
    dynamic_numeric_feature_cols=SAIL_DYNAMIC_NUMERIC_FEATURES,


    known_dynamic_numeric_feature_cols=("age",),
    known_dynamic_raw_increments=(("age", 0.25),),


    observation_numeric_feature_cols=("efi",),
    binary_dynamic_feature_cols=("care_home",),
    monotone_binary_feature_cols=("care_home",),
    include_missingness=True,
    allowed_edges=SAIL_ALLOWED_EDGES,
)


def sail_config_from_manifest(path: str | Path) -> ExperimentConfig:
    'Load the cohort-frozen landmarks and graph into the SAIL model config.'
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    landmarks = tuple(map(int, manifest["landmarks"]))
    edges = tuple(tuple(map(int, edge)) for edge in manifest["allowed_edges"])
    if edges != SAIL_ALLOWED_EDGES:
        raise ValueError(
            "SAIL manifest graph is not the recurrent quarterly-admission graph. "
            "Rebuild quarterly data, targets and all models in a fresh work directory; "
            "old cumulative-admission results are not compatible."
        )
    if (manifest.get("burden_definition") != SAIL_BURDEN_DEFINITION
            or manifest.get("preprocessing_version") != SAIL_PREPROCESSING_VERSION):
        raise ValueError(
            "SAIL manifest does not use verified recurrent quarterly admissions; "
            "rebuild preprocessing and targets before training."
        )
    maximum = int(manifest["max_observed_quarter"]) + int(manifest["horizon_quarters"]) + 1
    return replace(
        SAIL_CONFIG,
        landmarks=landmarks,
        max_horizon=int(manifest["horizon_quarters"]),
        max_time=max(maximum, max(landmarks) + int(manifest["horizon_quarters"]) + 1),
        allowed_edges=edges,
    )


VARIANTS = (
    "original",
    "ar_window6",
    "ar_window24",
    "ar_no_missingness",
    "ar_no_dwell",
    "ar_unconditioned_continuation",
    "process_snapshot",
    "process_semi_markov",
)


def variant_context(variant: str) -> str:
    if variant in {"process_snapshot", "process_semi_markov"}:
        return "snapshot"
    if variant == "ar_window6":
        return "window6"
    if variant == "ar_window24":
        return "window24"
    if variant in {
        "original",
        "ar_no_missingness",
        "ar_no_dwell",
        "ar_unconditioned_continuation",
    }:
        return "full"
    raise ValueError(f"Unknown variant: {variant}")
