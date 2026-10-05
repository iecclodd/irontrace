"""Context-only nearest-neighbor donor draws for hypothetical panel outcomes."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .policies import Panel, panel_schema


class VisibleCase(Protocol):
    observed_groups: tuple[str, ...]
    values: dict[str, float]


class DonorSampler:
    def __init__(self, context_X: object, panels: object):
        matrix = np.asarray(context_X, dtype=np.float64)
        if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
            raise ValueError("context_X must be a nonempty finite matrix")
        self.context_X = matrix
        self.panels: tuple[Panel, ...] = panel_schema(panels, matrix.shape[1])
        self.by_id = {panel.id: panel for panel in self.panels}
        self.medians = np.median(matrix, axis=0)
        q75, q25 = np.percentile(matrix, [75, 25], axis=0)
        self.scales = q75 - q25
        self.scales[self.scales <= 0] = 1.0

    def observed_row(self, observed: VisibleCase) -> np.ndarray:
        if len(set(observed.observed_groups)) != len(observed.observed_groups):
            raise ValueError("observed groups cannot repeat")
        row = np.full(self.context_X.shape[1], np.nan, dtype=np.float64)
        allowed: set[str] = set()
        for group_id in observed.observed_groups:
            if group_id not in self.by_id:
                raise ValueError(f"unknown observed group: {group_id}")
            panel = self.by_id[group_id]
            for feature, column in zip(panel.features, panel.columns):
                allowed.add(feature)
                if feature not in observed.values:
                    raise ValueError(f"missing visible value: {feature}")
                value = float(observed.values[feature])
                if not np.isfinite(value):
                    raise ValueError("visible values must be finite")
                row[column] = value
        if set(observed.values) != allowed:
            raise ValueError("observed values contain unobserved or unknown features")
        return row

    def _neighbors(self, observed: VisibleCase, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        panel_distances = []
        for group_id in observed.observed_groups:
            columns = self.by_id[group_id].columns
            standardized = (self.context_X[:, columns] - row[list(columns)]) / self.scales[list(columns)]
            panel_distances.append(np.mean(standardized**2, axis=1))
        if not panel_distances:
            return np.arange(self.context_X.shape[0]), np.zeros(self.context_X.shape[0])
        distances = np.mean(panel_distances, axis=0)
        order = np.lexsort((np.arange(len(distances)), distances))
        indices = order[: min(32, len(order))]
        return indices, distances[indices]

    def sample(
        self, observed: VisibleCase, candidate_id: str, seed: int, draws: int = 16
    ) -> tuple[np.ndarray, dict[str, object]]:
        if not isinstance(draws, int) or draws <= 0:
            raise ValueError("draws must be a positive integer")
        if candidate_id not in self.by_id:
            raise ValueError(f"unknown candidate panel: {candidate_id}")
        if candidate_id in observed.observed_groups:
            raise ValueError("candidate panel has already been observed")
        row = self.observed_row(observed)
        neighbors, distances = self._neighbors(observed, row)
        rng = np.random.default_rng(seed)
        donor_indices = rng.choice(neighbors, size=draws, replace=True)
        rows = np.repeat(row[None, :], draws, axis=0)
        candidate_columns = list(self.by_id[candidate_id].columns)
        rows[:, candidate_columns] = self.context_X[donor_indices][:, candidate_columns]
        diagnostics = {
            "neighborhood_size": int(len(neighbors)),
            "unique_donor_count": int(len(set(donor_indices.tolist()))),
            "mean_distance": float(np.mean(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "neighbor_distances": [float(value) for value in distances],
            "seed": int(seed),
            "draws": draws,
        }
        return rows, diagnostics
