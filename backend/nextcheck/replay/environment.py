"""Hidden historical row; only charged, validated groups become visible."""

from __future__ import annotations

import math
import numpy as np

from nextcheck.acquisition.policies import panel_schema
from nextcheck.agent.state import ObservedCase


class ReplayEnvironment:
    def __init__(
        self, row: object, label: int, manifest: object,
        initial_groups: tuple[str, ...] = ("initial",),
    ) -> None:
        values = np.asarray(row, dtype=np.float64)
        if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError("replay row must be a finite one-dimensional vector")
        self.__row = values.copy()
        self.__label = int(label)
        self.panels = panel_schema(manifest, len(values))
        self.by_id = {panel.id: panel for panel in self.panels}
        if len(set(initial_groups)) != len(initial_groups):
            raise ValueError("initial groups cannot repeat")
        if any(group_id not in self.by_id for group_id in initial_groups):
            raise ValueError("unknown initial group")
        self.initial_groups = tuple(initial_groups)

    def initial_observation(self) -> ObservedCase:
        visible = {}
        for group_id in self.initial_groups:
            panel = self.by_id[group_id]
            visible.update({name: float(self.__row[column]) for name, column in zip(panel.features, panel.columns)})
        return ObservedCase(self.initial_groups, visible)

    def reveal(
        self, observed: ObservedCase, group_id: str, remaining_budget: float
    ) -> tuple[ObservedCase, float]:
        """Return a new visible case and exact charge after checking the action."""
        if group_id not in self.by_id:
            raise ValueError("unknown panel")
        if group_id in observed.observed_groups:
            return observed, 0.0  # A retry never reveals or charges twice.
        if not math.isfinite(remaining_budget) or remaining_budget < 0:
            raise ValueError("invalid remaining budget")
        panel = self.by_id[group_id]
        if panel.cost > remaining_budget:
            raise ValueError("panel exceeds remaining budget")
        if len(observed.observed_groups) >= len(self.panels):
            raise ValueError("all panels have already been observed")
        visible = dict(observed.values)
        visible.update({name: float(self.__row[column]) for name, column in zip(panel.features, panel.columns)})
        return ObservedCase(observed.observed_groups + (group_id,), visible), panel.cost

    def completed_label(self, *, completed: bool) -> int:
        """Only a terminal report caller should request the retrospective label."""
        if not completed:
            raise ValueError("label access requires an explicitly completed report")
        return self.__label
