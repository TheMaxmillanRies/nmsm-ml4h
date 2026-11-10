from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class FeatureEncoder:
    'Fold-fitted numeric/missingness encoder with categorical index embeddings.'

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...] = ()
    dynamic_numeric_columns: tuple[str, ...] = ()
    include_missingness: bool = False
    max_delta: float = 200.0
    observation_numeric_columns: tuple[str, ...] = ()
    binary_dynamic_columns: tuple[str, ...] = ()
    monotone_binary_columns: tuple[str, ...] = ()
    known_dynamic_columns: tuple[str, ...] = ()
    known_dynamic_raw_increments: tuple[tuple[str, float], ...] = ()
    means: pd.Series | None = None
    stds: pd.Series | None = None
    category_levels: dict[str, tuple[int, ...]] | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        self.means = df.loc[:, self.numeric_columns].apply(pd.to_numeric, errors="coerce").mean()
        self.stds = df.loc[:, self.numeric_columns].apply(pd.to_numeric, errors="coerce").std(ddof=1)
        self.stds = self.stds.mask((~np.isfinite(self.stds)) | (self.stds < 1e-6), 1.0)


        for column in self.binary_dynamic_columns:
            if column not in self.numeric_columns:
                raise ValueError(f"Binary dynamic column is not numeric: {column}")
            self.means.loc[column] = 0.0
            self.stds.loc[column] = 1.0
        invalid_observation = set(self.observation_numeric_columns).difference(
            self.dynamic_numeric_columns
        )
        if invalid_observation:
            raise ValueError(
                "Observation-process columns must be dynamic numeric columns: "
                f"{sorted(invalid_observation)}"
            )
        invalid_binary = set(self.binary_dynamic_columns).difference(
            self.dynamic_numeric_columns
        )
        if invalid_binary:
            raise ValueError(
                f"Binary columns must be dynamic: {sorted(invalid_binary)}"
            )
        if not set(self.monotone_binary_columns).issubset(self.binary_dynamic_columns):
            raise ValueError("Monotone binary columns must also be binary dynamic columns")
        invalid_known = set(self.known_dynamic_columns).difference(
            self.dynamic_numeric_columns
        )
        if invalid_known:
            raise ValueError(f"Known dynamic columns must be dynamic: {sorted(invalid_known)}")
        overlap = set(self.known_dynamic_columns).intersection(self.binary_dynamic_columns)
        if overlap:
            raise ValueError(f"Known dynamic columns cannot also be binary: {sorted(overlap)}")
        increment_columns = [column for column, _ in self.known_dynamic_raw_increments]
        if len(increment_columns) != len(set(increment_columns)):
            raise ValueError("Known dynamic increments contain duplicate columns")
        if set(increment_columns) != set(self.known_dynamic_columns):
            raise ValueError("Every known dynamic column requires exactly one raw-scale increment")
        if any(not np.isfinite(value) for _, value in self.known_dynamic_raw_increments):
            raise ValueError("Known dynamic increments must be finite")
        self.category_levels = {}
        for column in self.categorical_columns:
            raw = pd.to_numeric(df[column], errors="coerce")
            self.category_levels[column] = tuple(sorted(map(int, raw.dropna().unique())))
        return self

    def _check_fitted(self) -> None:
        if self.means is None or self.stds is None or self.category_levels is None:
            raise RuntimeError("FeatureEncoder must be fitted before use")

    @property
    def continuous_feature_names(self) -> tuple[str, ...]:
        values = tuple(f"value__{column}" for column in self.numeric_columns)
        if not self.include_missingness:
            return values
        masks = tuple(f"observed__{column}" for column in self.numeric_columns)
        deltas = tuple(f"delta__{column}" for column in self.observation_numeric_columns)
        return values + masks + deltas

    @property
    def dynamic_continuous_columns(self) -> tuple[str, ...]:
        excluded = set(self.binary_dynamic_columns) | set(self.known_dynamic_columns)
        return tuple(column for column in self.dynamic_numeric_columns if column not in excluded)

    @property
    def dynamic_continuous_indices(self) -> tuple[int, ...]:
        return tuple(self.numeric_columns.index(column) for column in self.dynamic_continuous_columns)

    @property
    def dynamic_binary_indices(self) -> tuple[int, ...]:
        return tuple(self.numeric_columns.index(column) for column in self.binary_dynamic_columns)

    @property
    def observation_indices(self) -> tuple[int, ...]:
        return tuple(self.numeric_columns.index(column) for column in self.observation_numeric_columns)

    @property
    def observation_to_continuous(self) -> tuple[int, ...]:
        continuous = self.dynamic_continuous_columns
        return tuple(continuous.index(column) for column in self.observation_numeric_columns)

    @property
    def monotone_binary_positions(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, column in enumerate(self.binary_dynamic_columns)
            if column in self.monotone_binary_columns
        )

    @property
    def categorical_cardinalities(self) -> tuple[int, ...]:
        self._check_fitted()
        assert self.category_levels is not None

        return tuple(len(self.category_levels[column]) + 1 for column in self.categorical_columns)

    def encode(self, group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self._encode(group, np.zeros(len(group), dtype=np.int64))

    def encode_many(self, ordered: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        'Encode patient-contiguous rows in one pass, resetting LOCF and gaps.'
        ids = ordered["id"].to_numpy()
        if not len(ids):
            return (np.empty((0, len(self.continuous_feature_names)), dtype=np.float32),
                    np.empty((0, len(self.categorical_columns)), dtype=np.int64))
        boundary = np.r_[True, ids[1:] != ids[:-1]]
        if len(np.unique(ids[boundary])) != int(boundary.sum()):
            raise ValueError("Batched encoding requires contiguous patient rows")
        starts = np.maximum.accumulate(np.where(boundary, np.arange(len(ids)), 0))
        return self._encode(ordered, starts)

    def _encode(self, group, patient_starts):
        self._check_fitted()
        assert self.means is not None and self.stds is not None and self.category_levels is not None
        raw = group.loc[:, self.numeric_columns].to_numpy(dtype=float, na_value=np.nan)
        observed = np.isfinite(raw)
        means = self.means.loc[list(self.numeric_columns)].to_numpy(dtype=float)
        stds = self.stds.loc[list(self.numeric_columns)].to_numpy(dtype=float)
        values = np.where(observed, (raw - means) / stds, 0.0).astype(np.float32)
        numeric_index = {column: index for index, column in enumerate(self.numeric_columns)}


        for column in self.dynamic_numeric_columns:
            column_index = numeric_index[column]
            indices = np.arange(len(group), dtype=int)
            last_index = np.maximum.accumulate(
                np.where(observed[:, column_index], indices, -1)
            )
            carry = last_index >= patient_starts
            values[carry, column_index] = values[last_index[carry], column_index]
        continuous = [values]
        if self.include_missingness:
            continuous.append(observed.astype(np.float32))
            times = group["t"].to_numpy(dtype=float)
            deltas = np.empty((len(group), len(self.observation_numeric_columns)), dtype=np.float32)
            cap = max(float(self.max_delta), 1.0)
            for output_index, column in enumerate(self.observation_numeric_columns):
                mask = observed[:, numeric_index[column]]
                indices = np.arange(len(group), dtype=int)
                last_index = np.maximum.accumulate(np.where(mask, indices, -1))
                last_time = np.where(
                    last_index >= patient_starts,
                    times[np.maximum(last_index, 0)],
                    times[patient_starts] - 1.0,
                )
                gap = np.clip(times - last_time, 0.0, cap)
                deltas[:, output_index] = (gap / cap).astype(np.float32)
            continuous.append(deltas)
        categories = np.empty((len(group), len(self.categorical_columns)), dtype=np.int64)
        for output_index, column in enumerate(self.categorical_columns):
            raw_category = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            levels = np.asarray(self.category_levels[column], dtype=int)
            encoded = np.zeros(len(raw_category), dtype=np.int64)
            finite = np.isfinite(raw_category)
            if len(levels) and np.any(finite):
                integer = raw_category[finite].astype(int)
                positions = np.searchsorted(levels, integer)
                valid = positions < len(levels)
                valid[valid] &= levels[positions[valid]] == integer[valid]
                finite_indices = np.flatnonzero(finite)
                encoded[finite_indices[valid]] = positions[valid] + 1
            categories[:, output_index] = encoded
        return np.concatenate(continuous, axis=1), categories

    def advance_locf(self, values: torch.Tensor) -> torch.Tensor:
        'Advance one unobserved future hour under the declared LOCF rule.'
        output = values.clone()
        self._check_fitted()
        assert self.stds is not None
        for column, raw_increment in self.known_dynamic_raw_increments:
            index = self.numeric_columns.index(column)
            standardized_increment = float(raw_increment) / float(self.stds.loc[column])
            output[:, index] = output[:, index] + standardized_increment
        if not self.include_missingness:
            return output
        n_numeric = len(self.numeric_columns)
        observed_process = set(self.observation_numeric_columns)
        for index, column in enumerate(self.numeric_columns):
            if column in observed_process:
                output[:, n_numeric + index] = 0.0
        delta_start = 2 * n_numeric
        increment = 1.0 / max(float(self.max_delta), 1.0)
        if self.observation_numeric_columns:
            output[:, delta_start:] = torch.clamp(output[:, delta_start:] + increment, max=1.0)
        return output

    def advance_joint(
        self,
        values: torch.Tensor,
        continuous_values: torch.Tensor,
        observations: torch.Tensor,
        binary_values: torch.Tensor,
    ) -> torch.Tensor:
        'Advance one generated token under the fitted observation process.'
        expected = (values.shape[0], len(self.dynamic_continuous_columns))
        if tuple(continuous_values.shape) != expected:
            raise ValueError(
                f"Continuous draws have shape {tuple(continuous_values.shape)}, expected {expected}"
            )
        if tuple(observations.shape) != (values.shape[0], len(self.observation_numeric_columns)):
            raise ValueError("Observation draws do not match the fitted observation schema")
        if tuple(binary_values.shape) != (values.shape[0], len(self.binary_dynamic_columns)):
            raise ValueError("Binary draws do not match the fitted binary schema")
        output = self.advance_locf(values)
        n_numeric = len(self.numeric_columns)
        observation_position = {
            column: index for index, column in enumerate(self.observation_numeric_columns)
        }
        for draw_index, column in enumerate(self.dynamic_continuous_columns):
            numeric_index = self.numeric_columns.index(column)
            if column in observation_position:
                observed_index = observation_position[column]
                selected = observations[:, observed_index].to(torch.bool)
            else:
                observed_index = None
                selected = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
            output[:, numeric_index] = torch.where(
                selected, continuous_values[:, draw_index], output[:, numeric_index]
            )
            if self.include_missingness:
                output[:, n_numeric + numeric_index] = selected.to(output.dtype)
                if observed_index is not None:
                    output[:, 2 * n_numeric + observed_index] = torch.where(
                        selected,
                        torch.zeros((), dtype=output.dtype, device=output.device),
                        output[:, 2 * n_numeric + observed_index],
                    )
        monotone = set(self.monotone_binary_positions)
        for draw_index, column in enumerate(self.binary_dynamic_columns):
            numeric_index = self.numeric_columns.index(column)
            generated = binary_values[:, draw_index].to(output.dtype)
            if draw_index in monotone:
                generated = torch.maximum(values[:, numeric_index], generated)
            output[:, numeric_index] = generated
            if self.include_missingness:
                output[:, n_numeric + numeric_index] = 1.0
        return output

    def state_dict(self) -> dict:
        self._check_fitted()
        assert self.means is not None and self.stds is not None and self.category_levels is not None
        return {
            "type": "FeatureEncoder",
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "dynamic_numeric_columns": list(self.dynamic_numeric_columns),
            "include_missingness": self.include_missingness,
            "max_delta": self.max_delta,
            "observation_numeric_columns": list(self.observation_numeric_columns),
            "binary_dynamic_columns": list(self.binary_dynamic_columns),
            "monotone_binary_columns": list(self.monotone_binary_columns),
            "known_dynamic_columns": list(self.known_dynamic_columns),
            "known_dynamic_raw_increments": [list(value) for value in self.known_dynamic_raw_increments],
            "means": self.means.to_dict(),
            "stds": self.stds.to_dict(),
            "category_levels": {key: list(value) for key, value in self.category_levels.items()},
            "continuous_feature_names": list(self.continuous_feature_names),
            "categorical_cardinalities": list(self.categorical_cardinalities),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureEncoder":
        encoder = cls(
            tuple(state["numeric_columns"]),
            tuple(state.get("categorical_columns", ())),
            tuple(state.get("dynamic_numeric_columns", ())),
            bool(state.get("include_missingness", False)),
            float(state.get("max_delta", 200.0)),
            tuple(state.get("observation_numeric_columns", ())),
            tuple(state.get("binary_dynamic_columns", ())),
            tuple(state.get("monotone_binary_columns", ())),
            tuple(state.get("known_dynamic_columns", ())),
            tuple(
                (str(column), float(value))
                for column, value in state.get("known_dynamic_raw_increments", ())
            ),
        )
        encoder.means = pd.Series(state["means"], dtype=float)
        encoder.stds = pd.Series(state["stds"], dtype=float)
        encoder.category_levels = {
            key: tuple(map(int, value)) for key, value in state.get("category_levels", {}).items()
        }
        return encoder


def add_history_summaries(df: pd.DataFrame, patient_col: str = "id") -> pd.DataFrame:
    'Derived C1 variables: duration, previous state, and prior transition count.'
    out = df.sort_values([patient_col, "t"]).copy()
    if out.empty:
        for column in ("time_in_state", "previous_state", "transition_count"):
            out[column] = np.empty(0, dtype=np.float32)
        return out
    ids = out[patient_col].to_numpy()
    states = out["state"].to_numpy(dtype=np.int64)
    times = out["t"].to_numpy(dtype=np.int64)
    index = np.arange(len(out))
    boundary = np.r_[True, ids[1:] != ids[:-1]]
    changes = np.r_[False, states[1:] != states[:-1]] & ~boundary
    starts = np.maximum.accumulate(np.where(boundary | changes, index, 0))
    patient_starts = np.maximum.accumulate(np.where(boundary, index, 0))
    previous = np.r_[states[0], states[:-1]]
    previous[boundary] = states[boundary]
    accumulated = np.cumsum(changes)
    out["time_in_state"] = (times - times[starts]).astype(np.float32)
    out["previous_state"] = previous.astype(np.float32)
    out["transition_count"] = (accumulated - accumulated[patient_starts]).astype(np.float32)
    return out


class TrajectoryDataset(Dataset):
    'One item per patient with causal dwell and joint next-token targets.'

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        num_states: int,
        simulation_state_map: bool = False,
        feature_encoder: FeatureEncoder | None = None,
    ) -> None:
        self.feature_cols = tuple(feature_cols)
        self.num_states = int(num_states)
        work = df.sort_values(["id", "t"]).copy()
        if simulation_state_map:
            state_map = {1: 0, 2: 1, 3: 2}
            work["state"] = work["state"].map(state_map)
            valid_next = work["next_state"].notna() & (work["next_state"] >= 0)
            work.loc[valid_next, "next_state"] = work.loc[valid_next, "next_state"].map(state_map)


        work = add_history_summaries(work)
        encoded_many = feature_encoder.encode_many(work) if feature_encoder is not None else None
        encoded_offset = 0
        self.patient_ids: list[int] = []
        self.data: dict[int, dict[str, torch.Tensor | int]] = {}
        for pid, g in work.groupby("id", sort=False):
            states = g["state"].to_numpy(dtype=np.int64)
            raw_next = g["next_state"].to_numpy()
            next_states = np.where(pd.isna(raw_next), -1, raw_next).astype(np.int64)
            times = g["t"].to_numpy(dtype=np.int64)
            if feature_encoder is None:
                covs = g.loc[:, self.feature_cols].to_numpy(dtype=np.float32)
                categories = np.empty((len(g), 0), dtype=np.int64)
            else:
                covs = encoded_many[0][encoded_offset:encoded_offset + len(g)]
                categories = encoded_many[1][encoded_offset:encoded_offset + len(g)]
            encoded_offset += len(g)
            cov_targets = np.zeros_like(covs)
            if len(covs) > 1:
                cov_targets[:-1] = covs[1:]
            cov_targets[-1] = covs[-1]
            next_mask = next_states >= 0


            cov_mask = np.zeros(len(g), dtype=bool)
            if len(g) > 1:
                cov_mask[:-1] = np.diff(times) == 1
            if feature_encoder is not None:
                continuous_indices = feature_encoder.dynamic_continuous_indices
                binary_indices = feature_encoder.dynamic_binary_indices
                observation_columns = feature_encoder.observation_numeric_columns
                continuous_columns = feature_encoder.dynamic_continuous_columns
                binary_columns = feature_encoder.binary_dynamic_columns
            else:
                continuous_indices = tuple(range(covs.shape[1]))
                binary_indices = ()
                observation_columns = ()
                continuous_columns = tuple(self.feature_cols)
                binary_columns = ()
            target_continuous = np.zeros(
                (len(g), len(continuous_indices)), dtype=np.float32
            )
            current_continuous = np.zeros_like(target_continuous)
            if continuous_indices:
                current_continuous = covs[:, continuous_indices].astype(np.float32)
                target_continuous[:-1] = covs[1:, continuous_indices]
                target_continuous[-1] = current_continuous[-1]
            target_observed = np.zeros(
                (len(g), len(observation_columns)), dtype=np.float32
            )
            if len(g) > 1:
                for column_index, column in enumerate(observation_columns):
                    target_observed[:-1, column_index] = pd.to_numeric(
                        g[column].iloc[1:], errors="coerce"
                    ).notna().to_numpy(dtype=np.float32)
            observation_loss_mask = cov_mask[:, None] & np.ones_like(
                target_observed, dtype=bool
            )
            continuous_value_mask = cov_mask[:, None] & np.ones_like(
                target_continuous, dtype=bool
            )
            observation_position = {
                column: index for index, column in enumerate(observation_columns)
            }
            for column_index, column in enumerate(continuous_columns):
                if column in observation_position:
                    continuous_value_mask[:, column_index] &= target_observed[
                        :, observation_position[column]
                    ].astype(bool)
            target_binary = np.zeros((len(g), len(binary_columns)), dtype=np.float32)
            current_binary = np.zeros_like(target_binary)
            for column_index, column in enumerate(binary_columns):
                raw = pd.to_numeric(g[column], errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(raw).all() or not np.isin(raw, (0.0, 1.0)).all():
                    raise ValueError(f"Binary process column {column} must contain only 0/1")
                current_binary[:, column_index] = raw.astype(np.float32)
                if len(g) > 1:
                    target_binary[:-1, column_index] = raw[1:].astype(np.float32)
                target_binary[-1, column_index] = raw[-1]
            binary_loss_mask = cov_mask[:, None] & np.ones_like(
                target_binary, dtype=bool
            )
            if feature_encoder is not None:
                for binary_position in feature_encoder.monotone_binary_positions:
                    binary_loss_mask[:, binary_position] &= current_binary[
                        :, binary_position
                    ] < 0.5

            item: dict[str, torch.Tensor | int] = {
                "id": int(pid),
                "input_states": torch.as_tensor(states, dtype=torch.long),
                "input_values": torch.as_tensor(covs, dtype=torch.float32),
                "input_categories": torch.as_tensor(categories, dtype=torch.long),
                "input_times": torch.as_tensor(times, dtype=torch.long),
                "input_dwell": torch.as_tensor(
                    g["time_in_state"].to_numpy(dtype=np.float32), dtype=torch.float32
                ),
                "target_states": torch.as_tensor(next_states, dtype=torch.long),
                "next_mask": torch.as_tensor(next_mask, dtype=torch.bool),
                "current_continuous_values": torch.as_tensor(
                    current_continuous, dtype=torch.float32
                ),
                "target_continuous_values": torch.as_tensor(
                    target_continuous, dtype=torch.float32
                ),
                "continuous_value_mask": torch.as_tensor(
                    continuous_value_mask, dtype=torch.bool
                ),
                "target_observed": torch.as_tensor(
                    target_observed, dtype=torch.float32
                ),
                "observation_loss_mask": torch.as_tensor(
                    observation_loss_mask, dtype=torch.bool
                ),
                "current_binary_values": torch.as_tensor(
                    current_binary, dtype=torch.float32
                ),
                "target_binary_values": torch.as_tensor(
                    target_binary, dtype=torch.float32
                ),
                "binary_loss_mask": torch.as_tensor(
                    binary_loss_mask, dtype=torch.bool
                ),
            }
            self.patient_ids.append(int(pid))
            self.data[int(pid)] = item

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict:
        return self.data[self.patient_ids[index]]


class LandmarkForecastDataset(Dataset):
    'Matched landmark forecast windows with dense or event-only prehistory.'

    _LOSS_MASKS = (
        "next_mask",
        "continuous_value_mask",
        "observation_loss_mask",
        "binary_loss_mask",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        num_states: int,
        landmarks: Sequence[int],
        max_horizon: int,
        terminal_states: Sequence[int],
        history_mode: str,
        simulation_state_map: bool = False,
        feature_encoder: FeatureEncoder | None = None,
    ) -> None:
        if history_mode not in {"dense", "state_events"}:
            raise ValueError("history_mode must be 'dense' or 'state_events'")
        self.history_mode = history_mode
        self.landmarks = tuple(map(int, landmarks))
        self.max_horizon = int(max_horizon)
        self.terminals = frozenset(map(int, terminal_states))
        self.base = TrajectoryDataset(
            df,
            feature_cols,
            num_states,
            simulation_state_map=simulation_state_map,
            feature_encoder=feature_encoder,
        )
        self.items: list[tuple[int, int]] = []
        for pid in self.base.patient_ids:
            item = self.base.data[pid]
            times = item["input_times"]
            states = item["input_states"]
            next_mask = item["next_mask"]
            assert (
                torch.is_tensor(times)
                and torch.is_tensor(states)
                and torch.is_tensor(next_mask)
            )
            for landmark in self.landmarks:
                positions = torch.nonzero(times == landmark, as_tuple=False).flatten()
                if len(positions) != 1:
                    continue
                position = int(positions[0])
                forecast_window = (times >= landmark) & (
                    times < landmark + self.max_horizon
                )
                has_state_target = bool((next_mask & forecast_window).any())
                if (
                    int(states[position]) not in self.terminals
                    and has_state_target
                ):
                    self.items.append((pid, landmark))
        if not self.items:
            raise ValueError("No eligible patient-landmark forecast windows")

    @staticmethod
    def selected_indices(item: dict, landmark: int, horizon: int, mode: str) -> torch.Tensor:
        times = item["input_times"]
        states = item["input_states"]
        assert torch.is_tensor(times) and torch.is_tensor(states)
        before = times < landmark
        after = (times >= landmark) & (times < landmark + horizon)
        if mode == "dense":
            keep = before | after
        else:
            changed = torch.ones_like(states, dtype=torch.bool)
            changed[1:] = states[1:] != states[:-1]
            keep = (before & changed) | after
            keep[0] = True
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        if not len(indices) or not bool((times[indices] == landmark).any()):
            raise ValueError("Landmark window lost its cutoff token")
        return indices

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        pid, landmark = self.items[index]
        source = self.base.data[pid]
        indices = self.selected_indices(
            source, landmark, self.max_horizon, self.history_mode
        )
        output: dict[str, torch.Tensor | int] = {"id": pid, "landmark": landmark}
        for key, value in source.items():
            if key == "id":
                continue
            assert torch.is_tensor(value)
            output[key] = value[indices].clone()
        selected_times = output["input_times"]
        assert torch.is_tensor(selected_times)
        pre_landmark = selected_times < landmark
        for key in self._LOSS_MASKS:
            mask = output[key]
            assert torch.is_tensor(mask)
            mask[pre_landmark] = False
        if not bool(output["next_mask"].any()):
            raise ValueError("Eligible landmark window has no hourly state target")
        return output


def collate_trajectories(batch: list[dict]) -> dict:
    ids = [int(x["id"]) for x in batch]
    fields = {
        "input_states": (0, torch.long),
        "input_values": (0.0, torch.float32),
        "input_categories": (0, torch.long),
        "input_times": (0, torch.long),
        "input_dwell": (0.0, torch.float32),
        "target_states": (-1, torch.long),
        "next_mask": (False, torch.bool),
        "current_continuous_values": (0.0, torch.float32),
        "target_continuous_values": (0.0, torch.float32),
        "continuous_value_mask": (False, torch.bool),
        "target_observed": (0.0, torch.float32),
        "observation_loss_mask": (False, torch.bool),
        "current_binary_values": (0.0, torch.float32),
        "target_binary_values": (0.0, torch.float32),
        "binary_loss_mask": (False, torch.bool),
    }
    out: dict[str, object] = {"ids": ids}
    for key, (padding, _) in fields.items():
        out[key] = pad_sequence([x[key] for x in batch], batch_first=True, padding_value=padding)
    lengths = torch.as_tensor([len(x["input_states"]) for x in batch], dtype=torch.long)
    width = int(out["input_states"].shape[1])
    out["lengths"] = lengths
    out["valid_mask"] = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
    return out
