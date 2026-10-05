"""Public panel schema and deterministic policy helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Panel:
    id: str
    cost: float
    features: tuple[str, ...]
    columns: tuple[int, ...]


def panel_schema(manifest: object, width: int) -> tuple[Panel, ...]:
    """Resolve manifest feature names against their flattened declared order."""
    if isinstance(manifest, Mapping):
        panel_data = manifest.get("panels")
    else:
        panel_data = manifest
    if not isinstance(panel_data, Sequence) or isinstance(panel_data, (str, bytes)):
        raise ValueError("manifest must contain a panels sequence")
    raw: list[tuple[str, float, tuple[str, ...], tuple[int, ...] | None]] = []
    flattened: list[str] = []
    for item in panel_data:
        if not isinstance(item, Mapping):
            raise ValueError("each panel must be a mapping")
        group_id = str(item["id"])
        cost = float(item.get("cost", 0))
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"invalid cost for panel {group_id}")
        entries = tuple(item["features"])
        if not entries:
            raise ValueError(f"panel {group_id} has no features")
        if all(isinstance(feature, (int,)) for feature in entries):
            columns = tuple(int(feature) for feature in entries)
            features = tuple(str(feature) for feature in entries)
        elif all(isinstance(feature, str) for feature in entries):
            columns = None
            features = tuple(entries)
            flattened.extend(features)
        else:
            raise ValueError("panel features must be all names or all column indices")
        raw.append((group_id, cost, features, columns))
    if len({entry[0] for entry in raw}) != len(raw):
        raise ValueError("panel ids must be unique")
    if flattened and len(flattened) != width:
        raise ValueError("flattened manifest feature count does not match context width")
    if len(set(flattened)) != len(flattened):
        raise ValueError("manifest feature names must be unique")
    index = {name: column for column, name in enumerate(flattened)}
    panels = tuple(
        Panel(group_id, cost, features, columns if columns is not None else tuple(index[name] for name in features))
        for group_id, cost, features, columns in raw
    )
    all_columns = [column for panel in panels for column in panel.columns]
    if len(all_columns) != width or sorted(all_columns) != list(range(width)):
        raise ValueError("panel columns must partition all feature columns exactly once")
    return panels


def score_per_cost(score: float, cost: float) -> float:
    """Positive free information has priority; never divide by zero."""
    if cost == 0:
        return math.inf if score > 0 else score
    return score / cost
