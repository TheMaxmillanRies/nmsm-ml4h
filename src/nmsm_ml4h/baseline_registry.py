'Canonical baseline entry points for every dataset.'

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig
from .empirical_process import DWELL_CUTS, add_dwell, estimate_processes, make_predictions
from .idnetwork_general import run_idnetwork_general_fold
from .mixmarkov_general import run_mixmarkov_fold
from .mscox_baseline import run_mscox_fold
from .mspseudo_general import generate_fold_pseudovalues, run_mspseudo_fold
from .survnode_baseline import run_survnode_fold
from .targets import remap_states


PRIMARY_BASELINES = (
    "mscox",
    "survnode",
    "idnetwork",
    "mspseudo",
    "mixmarkov_2022",
    "empirical_markov",
    "empirical_semi_markov",
)

SIMULATION_STATE_MAP = {1: 0, 2: 1, 3: 2}


@dataclass(frozen=True)
class BaselineSettings:
    epochs_survnode: int = 500
    epochs_idnetwork: int = 50
    epochs_mspseudo: int = 10000
    mixmarkov_max_iterations: int | None = None
    patience_idnetwork: int = 8
    patience_mspseudo: int = 100
    patience_survnode: int = 20
    mixmarkov_samples: int = 100
    survnode_training_fraction: float = 1.0
    survnode_batch_size: int = 512
    survnode_microbatch_size: int = 512
    survnode_learning_rate: float = 1e-3
    survnode_encoder_width: int = 800
    survnode_ode_width: int = 1000
    survnode_solver: str = "dopri5"
    survnode_solver_rtol: float = 1e-8
    survnode_solver_atol: float = 1e-8
    survnode_gradient_clip_norm: float | None = None
    survnode_optimizer: str = "adam"
    seed: int = 2026


def internalize_baseline_frame(
    frame: pd.DataFrame, experiment: ExperimentConfig
) -> pd.DataFrame:
    'Return the state coding used by the model and target registry.'
    if experiment.name != "simulation":
        return frame.copy()
    observed = set(map(int, frame["state"].dropna().unique()))
    if not observed.issubset(SIMULATION_STATE_MAP):
        raise ValueError(
            "Synthetic baseline input must use the published raw state IDs "
            f"1/2/3; observed={sorted(observed)}"
        )
    return remap_states(frame, SIMULATION_STATE_MAP)


def _empirical_pair(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    fold_targets: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
) -> dict[str, pd.DataFrame]:
    work = add_dwell(frame[["id", "t", "state", "next_state"]])
    training_ids = folds.loc[folds["outer_fold"] != fold, "id"]
    training = work[work["id"].isin(training_ids)]
    max_time = int(max(training["t"].max(), fold_targets["target_time"].max()))
    markov, semi_markov = estimate_processes(
        training,
        experiment.num_states,
        experiment.terminal_states,
        max_time,
        time_shrinkage=20.0,
        dwell_shrinkage=20.0,
        allowed_edges=experiment.allowed_edges,
    )
    dwell_lookup = {
        (int(row.id), int(row.t)): int(row.dwell)
        for row in work[["id", "t", "dwell"]].itertuples(index=False)
    }
    return make_predictions(
        fold_targets,
        dwell_lookup,
        markov,
        semi_markov,
        experiment.num_states,
        experiment.max_horizon,
        dwell_cap=max(25, int(DWELL_CUTS[-1]) + 1),
    )


def run_baseline_fold(
    method: str,
    raw_frame: pd.DataFrame,
    folds: pd.DataFrame,
    all_targets: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
    output_dir: Path,
    settings: BaselineSettings = BaselineSettings(),
    device: torch.device | None = None,
) -> pd.DataFrame:
    'Fit one canonical baseline and return predictions on common keys.'
    if method not in PRIMARY_BASELINES:
        raise ValueError(f"Unknown canonical baseline: {method}")
    frame = internalize_baseline_frame(raw_frame, experiment)
    fold_targets = all_targets[all_targets["fold"] == fold].copy()
    if fold_targets.empty:
        raise ValueError(f"No target rows for fold {fold}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if method == "mscox":
        return run_mscox_fold(
            frame,
            folds,
            all_targets,
            fold,
            experiment,
            output_dir,
            ridge=None,
            selection_seed=settings.seed,
            device=device,
        )
    if method == "survnode":
        return run_survnode_fold(
            frame,
            folds,
            fold_targets,
            fold,
            experiment,
            output_dir,
            epochs=settings.epochs_survnode,
            patience=settings.patience_survnode,
            training_fraction=settings.survnode_training_fraction,
            batch_size=settings.survnode_batch_size,
            microbatch_size=settings.survnode_microbatch_size,
            learning_rate=settings.survnode_learning_rate,
            encoder_hidden_features=settings.survnode_encoder_width,
            ode_hidden_features=settings.survnode_ode_width,
            solver_method=settings.survnode_solver,
            solver_rtol=settings.survnode_solver_rtol,
            solver_atol=settings.survnode_solver_atol,
            gradient_clip_norm=settings.survnode_gradient_clip_norm,
            optimizer_name=settings.survnode_optimizer,
            seed=settings.seed,
            device=device,
        )
    if method == "idnetwork":
        return run_idnetwork_general_fold(
            frame,
            folds,
            fold_targets,
            fold,
            experiment,
            output_dir,
            epochs=settings.epochs_idnetwork,
            patience=settings.patience_idnetwork,
            seed=settings.seed,
            device=device,
        )
    if method == "mspseudo":
        pseudo = generate_fold_pseudovalues(
            all_targets,
            folds,
            fold,
            experiment,
            output_dir / "training_pseudovalues.parquet",
            output_dir / "pseudovalue_audit.csv",
            seed=settings.seed,
        )
        return run_mspseudo_fold(
            frame,
            folds,
            all_targets,
            pseudo,
            fold,
            experiment,
            output_dir,
            epochs=settings.epochs_mspseudo,
            patience=settings.patience_mspseudo,
            seed=settings.seed,
            device=device,
        )
    if method == "mixmarkov_2022":
        return run_mixmarkov_fold(
            frame,
            folds,
            fold_targets,
            fold,
            experiment,
            output_dir,
            max_iterations=settings.mixmarkov_max_iterations,
            samples=settings.mixmarkov_samples,
            seed=settings.seed,
        )
    return _empirical_pair(frame, folds, fold_targets, fold, experiment)[method]
