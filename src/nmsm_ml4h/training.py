from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, variant_context
from .data import (
    FeatureEncoder,
    LandmarkForecastDataset,
    TrajectoryDataset,
    collate_trajectories,
)
from .losses import LossWeights, compute_loss
from .model import JointCausalTransformer, ModelConfig, allowed_transitions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_config(
    config: ExperimentConfig, encoder: FeatureEncoder, variant: str
) -> ModelConfig:
    context = variant_context(variant)
    return ModelConfig(
        num_states=config.num_states,
        num_features=len(encoder.continuous_feature_names),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_time=config.max_time,
        context=context,
        boolean_causal_mask=True,
        categorical_cardinalities=encoder.categorical_cardinalities,
        use_dwell=variant not in {"process_snapshot", "ar_no_dwell"},
        num_dynamic_continuous=len(encoder.dynamic_continuous_columns),
        num_observation_processes=len(encoder.observation_numeric_columns),
        num_dynamic_binary=len(encoder.binary_dynamic_columns),
        covariate_parameterization=config.covariate_parameterization,
        condition_covariates_on_next_state=(
            variant != "ar_unconditioned_continuation"
        ),
    )


def make_dataset(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    encoder: FeatureEncoder,
    variant: str,
) -> TrajectoryDataset:
    return TrajectoryDataset(
        frame,
        config.feature_cols,
        config.num_states,
        simulation_state_map=config.name == "simulation",
        feature_encoder=encoder,
    )


def fit_feature_encoder(
    train_df: pd.DataFrame, config: ExperimentConfig
) -> FeatureEncoder:
    numeric = config.numeric_feature_cols or config.feature_cols
    dynamic = config.dynamic_numeric_feature_cols or tuple(
        column for column in config.feature_cols if column not in config.static_feature_cols
    )
    return FeatureEncoder(
        numeric_columns=numeric,
        categorical_columns=config.categorical_feature_cols,
        dynamic_numeric_columns=dynamic,
        include_missingness=config.include_missingness,
        max_delta=float(config.max_time),
        observation_numeric_columns=config.observation_numeric_feature_cols,
        binary_dynamic_columns=config.binary_dynamic_feature_cols,
        monotone_binary_columns=config.monotone_binary_feature_cols,
        known_dynamic_columns=config.known_dynamic_numeric_feature_cols,
        known_dynamic_raw_increments=config.known_dynamic_raw_increments,
    ).fit(train_df)


def record_feature_encoder(
    encoder: FeatureEncoder,
    output_dir: Path,
    label: str,
    training_frame: pd.DataFrame,
) -> None:
    state = encoder.state_dict()
    audit = {
        "fit_scope": "inner-training patients only",
        "training_patients": int(training_frame["id"].nunique()),
        "training_rows": int(len(training_frame)),
        **state,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_encoder.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"{label} FEATURE_ENCODER FIT scope=inner-training-only "
        f"patients={audit['training_patients']} rows={audit['training_rows']} "
        f"dynamic_continuous={list(encoder.dynamic_continuous_columns)} "
        f"observation_processes={list(encoder.observation_numeric_columns)} "
        f"dynamic_binary={list(encoder.binary_dynamic_columns)}",
        flush=True,
    )


def forward_batch(model: JointCausalTransformer, batch: dict) -> dict:
    categories = batch["input_categories"]
    return model(
        batch["input_states"],
        batch["input_values"],
        batch["input_times"],
        batch["valid_mask"],
        categories if categories.shape[-1] else None,
        dwell=batch["input_dwell"] if model.config.use_dwell else None,
        next_states=batch["target_states"].clamp_min(0),
        current_continuous=batch["current_continuous_values"],
    )


def _component_count(batch: dict, key: str) -> int:
    return int(batch[key].sum().item())


@torch.no_grad()
def validation_loss(
    model: JointCausalTransformer,
    loader: DataLoader,
    allowed: torch.Tensor,
    weights: LossWeights,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {"state": 0.0, "covariate_values": 0.0, "observations": 0.0, "binary_covariates": 0.0}
    counts = {"state": 0, "covariate_values": 0, "observations": 0, "binary_covariates": 0}
    mask_for = {
        "state": "next_mask",
        "covariate_values": "continuous_value_mask",
        "observations": "observation_loss_mask",
        "binary_covariates": "binary_loss_mask",
    }
    for raw in loader:
        batch = move_batch(raw, device)
        parts = compute_loss(forward_batch(model, batch), batch, allowed, weights)
        for component, mask in mask_for.items():
            count = _component_count(batch, mask)
            if count:
                sums[component] += float(parts[component].item()) * count
                counts[component] += count
    if not counts["state"]:
        raise RuntimeError("Validation partition contains no transition targets")
    means = {
        key: sums[key] / counts[key] if counts[key] else 0.0 for key in sums
    }
    covariates = (
        means["covariate_values"]
        + means["observations"]
        + means["binary_covariates"]
    )
    total = means["state"] + weights.covariates * covariates
    if not np.isfinite(total):
        raise FloatingPointError("Non-finite validation objective")
    return {**means, "covariates": covariates, "total": total, **{f"n_{k}": float(v) for k, v in counts.items()}}


def _assert_finite_parameters(model: JointCausalTransformer) -> None:
    invalid = [name for name, value in model.named_parameters() if not torch.isfinite(value).all()]
    if invalid:
        raise FloatingPointError(f"Non-finite model parameters: {invalid[:5]}")


def train_fold(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ExperimentConfig,
    variant: str,
    output_dir: Path,
    seed: int,
    epochs: int | None = None,
    device: torch.device | None = None,
    patience: int | None = 8,
    landmark_history_mode: str | None = None,
    time_encoding: str = "discrete_absolute",
    checkpoint_metric: str = "total",
) -> tuple[JointCausalTransformer, FeatureEncoder, dict]:
    if not np.isfinite(config.lambda_cov) or config.lambda_cov < 0:
        raise ValueError("lambda_cov must be finite and nonnegative")
    if checkpoint_metric not in {"total", "state"}:
        raise ValueError("checkpoint_metric must be 'total' or 'state'")
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = fit_feature_encoder(train_df, config)
    record_feature_encoder(encoder, output_dir, f"{config.name}/{variant}", train_df)
    def build_dataset(frame: pd.DataFrame):
        if landmark_history_mode is None:
            return make_dataset(frame, config, encoder, variant)
        return LandmarkForecastDataset(
            frame,
            config.feature_cols,
            config.num_states,
            config.landmarks,
            config.max_horizon,
            config.terminal_states,
            landmark_history_mode,
            simulation_state_map=config.name == "simulation",
            feature_encoder=encoder,
        )

    train_loader = DataLoader(
        build_dataset(train_df),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
        num_workers=0,
    )
    validation_loader = DataLoader(
        build_dataset(validation_df),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
        num_workers=0,
    )
    architecture = replace(
        model_config(config, encoder, variant), time_encoding=time_encoding
    )
    model = JointCausalTransformer(architecture).to(device)
    allowed = allowed_transitions(
        config.num_states, config.terminal_states, config.allowed_edges
    ).to(device)
    weights = LossWeights(covariates=config.lambda_cov)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best = float("inf")
    history: list[dict] = []
    stale = 0
    started = time.time()
    n_epochs = int(epochs or config.epochs)
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_parts: dict[str, list[float]] = {
            "total": [], "state": [], "covariates": [], "covariate_values": [],
            "observations": [], "binary_covariates": [], "gradient_norm": []
        }
        for raw in train_loader:
            batch = move_batch(raw, device)
            parts = compute_loss(forward_batch(model, batch), batch, allowed, weights)
            if not torch.isfinite(parts["total"]):
                raise FloatingPointError("Non-finite training objective")
            optimizer.zero_grad(set_to_none=True)
            parts["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            _assert_finite_parameters(model)
            for key in epoch_parts:
                value = gradient_norm if key == "gradient_norm" else parts[key]
                epoch_parts[key].append(float(value.item()))
        validation = validation_loss(model, validation_loader, allowed, weights, device)
        train_summary = {key: float(np.mean(value)) for key, value in epoch_parts.items()}
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_summary.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        current = validation[checkpoint_metric]
        if current < best - 1e-5:
            best = current
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": asdict(architecture),
                    "feature_encoder": encoder.state_dict(),
                    "variant": variant,
                    "landmark_history_mode": landmark_history_mode,
                    "config": asdict(config),
                    "epoch": epoch,
                    "validation": validation,
                    "checkpoint_metric": checkpoint_metric,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        print(
            f"epoch={epoch:03d}/{n_epochs} train={train_summary['total']:.6f} "
            f"state={train_summary['state']:.6f} cov={train_summary['covariates']:.6f} "
            f"val={current:.6f} val_state={validation['state']:.6f} "
            f"val_cov={validation['covariates']:.6f} grad={train_summary['gradient_norm']:.4f} "
            f"elapsed_min={(time.time() - started) / 60:.1f}",
            flush=True,
        )
        if patience is not None and patience > 0 and stale >= patience:
            print(f"early_stop epoch={epoch} patience={patience}", flush=True)
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    _assert_finite_parameters(model)
    summary = {
        "variant": variant,
        "covariate_parameterization": config.covariate_parameterization,
        "lambda_cov": config.lambda_cov,
        "landmark_history_mode": landmark_history_mode,
        "time_encoding": time_encoding,
        "checkpoint_metric": checkpoint_metric,
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["validation"],
        "maximum_epochs": n_epochs,
        "early_stopping_patience": patience,
        "epochs_run": len(history),
        "stopped_by_patience_before_epoch_cap": len(history) < n_epochs,
        "best_at_training_boundary": int(checkpoint["epoch"]) == len(history),
        "runtime_seconds": time.time() - started,
        "history": history,
    }
    (output_dir / "training.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return model, encoder, summary


def target_at(
    group: pd.DataFrame, target_time: int, terminals: frozenset[int]
) -> tuple[int, bool, int]:
    exact = group[group["t"] == target_time]
    if len(exact):
        return int(exact.iloc[0]["state"]), True, target_time
    last = group.iloc[-1]
    last_time = int(last["t"])
    last_state = int(last["state"])
    if last_state in terminals and target_time > last_time:
        return last_state, True, last_time
    return -1, False, last_time
