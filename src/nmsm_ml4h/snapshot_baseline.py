from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .data import FeatureEncoder
from .ipcw import conditional_ipcw, fit_censoring_km
from .splits import patient_table, split_train_validation
from .targets import KEYS
from .training import fit_feature_encoder, set_seed


@dataclass(frozen=True)
class SnapshotConfig:
    num_states: int
    num_features: int
    categorical_cardinalities: tuple[int, ...]
    hidden_size: int
    max_time: int
    max_horizon: int
    dropout: float


class SnapshotOccupationMLP(nn.Module):
    def __init__(self, config: SnapshotConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_projection = nn.Linear(config.num_features, config.hidden_size)
        self.state_embedding = nn.Embedding(config.num_states, config.hidden_size)
        self.landmark_embedding = nn.Embedding(config.max_time + 1, config.hidden_size)
        self.horizon_embedding = nn.Embedding(config.max_horizon + 1, config.hidden_size)
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, config.hidden_size)
            for cardinality in config.categorical_cardinalities
        )
        self.network = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_states),
        )

    def forward(
        self,
        values: torch.Tensor,
        categories: torch.Tensor,
        origin: torch.Tensor,
        landmark: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self.feature_projection(values)
            + self.state_embedding(origin)
            + self.landmark_embedding(landmark.clamp(0, self.config.max_time))
            + self.horizon_embedding(horizon)
        )
        if self.category_embeddings:
            category_sum = torch.zeros_like(hidden)
            for index, embedding in enumerate(self.category_embeddings):
                category_sum = category_sum + embedding(categories[:, index])
            hidden = hidden + category_sum / (len(self.category_embeddings) ** 0.5)
        return self.network(hidden)


class SnapshotDataset(Dataset):
    def __init__(
        self,
        target_rows: pd.DataFrame,
        pair_values: np.ndarray,
        pair_categories: np.ndarray,
        pair_lookup: dict[tuple[int, int], int],
    ) -> None:
        self.values = torch.as_tensor(pair_values, dtype=torch.float32)
        self.categories = torch.as_tensor(pair_categories, dtype=torch.long)
        self.pair_index = torch.as_tensor(
            [pair_lookup[(int(row.id), int(row.landmark))] for row in target_rows.itertuples(index=False)],
            dtype=torch.long,
        )
        self.origin = torch.as_tensor(target_rows["origin_state"].to_numpy(dtype=np.int64))
        self.landmark = torch.as_tensor(target_rows["landmark"].to_numpy(dtype=np.int64))
        self.horizon = torch.as_tensor(target_rows["horizon"].to_numpy(dtype=np.int64))
        self.target = torch.as_tensor(target_rows["target_state"].to_numpy(dtype=np.int64))
        self.weight = torch.as_tensor(target_rows["ipcw"].to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pair = self.pair_index[index]
        return (
            self.values[pair],
            self.categories[pair],
            self.origin[index],
            self.landmark[index],
            self.horizon[index],
            self.target[index],
            self.weight[index],
        )


def snapshot_features(
    frame: pd.DataFrame,
    requested_pairs: pd.DataFrame,
    encoder: FeatureEncoder,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    ordered = frame.sort_values(["id", "t"])
    values, categories = encoder.encode_many(ordered)
    requested = requested_pairs[["id", "landmark"]].drop_duplicates().sort_values(["id", "landmark"])
    positions = pd.MultiIndex.from_frame(ordered[["id", "t"]]).get_indexer(
        pd.MultiIndex.from_frame(requested))
    if np.any(positions < 0):
        raise KeyError("Missing requested snapshot landmark")
    lookup = {tuple(map(int, pair)): index for index, pair in enumerate(requested.to_numpy())}
    return values[positions], categories[positions], lookup


def weighted_nll(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, reduction="none")
    return torch.sum(loss * weight) / weight.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_loader(
    model: SnapshotOccupationMLP, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    for batch in loader:
        values, categories, origin, landmark, horizon, target, weight = [value.to(device) for value in batch]
        logits = model(values, categories, origin, landmark, horizon)
        loss = F.cross_entropy(logits, target, reduction="none")
        numerator += float(torch.sum(loss * weight).item())
        denominator += float(weight.sum().item())
    return numerator / max(denominator, 1.0)


def run_snapshot_fold(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    all_targets: pd.DataFrame,
    fold: int,
    experiment: ExperimentConfig,
    output_dir: Path,
    epochs: int,
    seed: int,
    patience: int = 8,
) -> pd.DataFrame:
    set_seed(seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_training = folds[folds["outer_fold"] != fold]
    train_ids, validation_ids = split_train_validation(outer_training, seed=seed + fold)
    test_ids = folds.loc[folds["outer_fold"] == fold, "id"].to_numpy()
    fit_frame = frame[frame["id"].isin(train_ids)]
    encoder = fit_feature_encoder(fit_frame, experiment)


    censoring = fit_censoring_km(
        patient_table(fit_frame[["id", "t", "state", "next_state"]], experiment.terminal_states)
    )
    eligible = all_targets[all_targets["known"]].copy()
    train_rows = eligible[eligible["id"].isin(train_ids)].copy()
    validation_rows = eligible[eligible["id"].isin(validation_ids)].copy()
    for selected in (train_rows, validation_rows):
        selected["ipcw"] = conditional_ipcw(
            censoring,
            selected["landmark"].to_numpy(dtype=float),
            selected["observation_time"].to_numpy(dtype=float),
            selected["known"].to_numpy(dtype=bool),
        )
    train_rows = train_rows[train_rows["ipcw"] > 0].copy()
    validation_rows = validation_rows[validation_rows["ipcw"] > 0].copy()
    test_rows = all_targets[(all_targets["fold"] == fold) & all_targets["id"].isin(test_ids)].copy()
    requested_pairs = pd.concat(
        [train_rows[["id", "landmark"]], validation_rows[["id", "landmark"]], test_rows[["id", "landmark"]]]
    ).drop_duplicates()
    feature_frame = frame[frame["id"].isin(requested_pairs["id"].unique())]
    values, categories, lookup = snapshot_features(feature_frame, requested_pairs, encoder)
    train_data = SnapshotDataset(train_rows, values, categories, lookup)
    validation_data = SnapshotDataset(validation_rows, values, categories, lookup)
    test_data = SnapshotDataset(test_rows.assign(target_state=0, ipcw=1.0), values, categories, lookup)
    train_loader = DataLoader(train_data, batch_size=2048, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=4096, shuffle=False)
    architecture = SnapshotConfig(
        experiment.num_states,
        len(encoder.continuous_feature_names),
        encoder.categorical_cardinalities,
        min(experiment.d_model, 256),
        experiment.max_time,
        experiment.max_horizon,
        experiment.dropout,
    )
    model = SnapshotOccupationMLP(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            values_batch, categories_batch, origin, landmark, horizon, target, weight = [value.to(device) for value in batch]
            logits = model(values_batch, categories_batch, origin, landmark, horizon)
            loss = weighted_nll(logits, target, weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation_nll = evaluate_loader(model, validation_loader, device)
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses)), "validation_nll": validation_nll})
        if validation_nll < best - 1e-5:
            best = validation_nll
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": asdict(architecture),
                    "feature_encoder": encoder.state_dict(),
                    "epoch": epoch,
                    "validation_nll": validation_nll,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        print(
            f"snapshot fold={fold} epoch={epoch}/{epochs} train={np.mean(losses):.6f} "
            f"val={validation_nll:.6f} stale={stale}",
            flush=True,
        )
        if stale >= patience:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    probability_parts = []
    with torch.no_grad():
        for batch in test_loader:
            values_batch, categories_batch, origin, landmark, horizon, _, _ = [value.to(device) for value in batch]
            probability_parts.append(
                torch.softmax(model(values_batch, categories_batch, origin, landmark, horizon), dim=-1).cpu().numpy()
            )
    probability = np.concatenate(probability_parts)
    predictions = test_rows[KEYS].reset_index(drop=True)
    for state in range(experiment.num_states):
        predictions[f"p_state_{state}"] = probability[:, state]
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "training.json").write_text(
        json.dumps(
            {
                "best_epoch": checkpoint["epoch"],
                "best_validation_nll": checkpoint["validation_nll"],
                "maximum_epochs": epochs,
                "early_stopping_patience": patience,
                "epochs_run": len(history),
                "stopped_by_patience_before_epoch_cap": len(history) < epochs,
                "runtime_seconds": time.time() - started,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return predictions
