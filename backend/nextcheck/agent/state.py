"""The only case state that may cross into the acquisition engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservedCase:
    observed_groups: tuple[str, ...]
    values: dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_groups", tuple(self.observed_groups))
        object.__setattr__(self, "values", dict(self.values))
