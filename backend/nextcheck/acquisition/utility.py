"""Offline one-step value model. Runtime prediction consumes visible diagnostics only."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


GROUPS = ("pressure", "flow", "vibration", "power", "temperature")
FEATURE_NAMES = (
    "information", "residual", "entropy", "margin", "observed_count",
    *(f"observed_{g}" for g in GROUPS),
    *(f"candidate_{g}" for g in GROUPS),
    "donor_count", "effective_donors", "mean_distance", "min_distance",
)
NO_RESIDUAL_FEATURES = tuple(name for name in FEATURE_NAMES if name != "residual")


def _finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def feature_vector(
    candidate: dict[str, Any], observed_groups: Iterable[str], probabilities: Iterable[float],
    *, include_residual: bool = True,
) -> np.ndarray:
    """Feature order is fixed; costs, labels and hidden readings never enter it."""
    observed = set(observed_groups)
    p = np.asarray(tuple(probabilities), dtype=np.float64)
    if p.shape != (3,) or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("expected a finite three-class probability distribution")
    descending = np.sort(p)[::-1]
    sampler = candidate.get("sampler") or {}
    if not isinstance(sampler, dict):
        sampler = {}
    group = candidate.get("group_id")
    if group not in GROUPS:
        raise ValueError(f"unknown candidate group: {group!r}")
    by_name = {
        "information": _finite_number(candidate.get("information")),
        "residual": _finite_number(candidate.get("residual")),
        "entropy": float(-np.sum(p * np.log(np.clip(p, 1e-15, 1)))),
        "margin": float(descending[0] - descending[1]),
        "observed_count": float(len(observed.intersection(GROUPS))),
        **{f"observed_{g}": float(g in observed) for g in GROUPS},
        **{f"candidate_{g}": float(g == group) for g in GROUPS},
        "donor_count": _finite_number(sampler.get("neighborhood_size", sampler.get("donor_count", sampler.get("n_donors")))),
        "effective_donors": _finite_number(sampler.get("unique_donor_count", sampler.get("effective_donors", sampler.get("effective_sample_size")))),
        "mean_distance": _finite_number(sampler.get("mean_distance")),
        "min_distance": _finite_number(sampler.get("min_distance")),
    }
    names = FEATURE_NAMES if include_residual else NO_RESIDUAL_FEATURES
    return np.asarray([by_name[name] for name in names], dtype=np.float64)


@dataclass
class UtilityModel:
    """Standardized ridge regressor, serialized as inspectable JSON."""

    alpha: float
    include_residual: bool
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    fitting_rows: tuple[int, ...]
    training_metadata: dict[str, Any] | None = None

    @classmethod
    def fit(
        cls, examples: list[dict[str, Any]], *, alpha: float,
        include_residual: bool = True,
    ) -> "UtilityModel":
        if not examples or alpha < 0 or not math.isfinite(alpha):
            raise ValueError("utility fit requires examples and a finite nonnegative alpha")
        x = np.vstack([
            feature_vector(e["candidate"], e["observed_groups"], e["probabilities"], include_residual=include_residual)
            for e in examples
        ])
        y = np.asarray([e["delta"] for e in examples], dtype=np.float64)
        if not np.all(np.isfinite(y)):
            raise ValueError("nonfinite utility targets")
        scaler = StandardScaler().fit(x)
        ridge = Ridge(alpha=alpha).fit(scaler.transform(x), y)
        rows = tuple(sorted({int(e["row_index"]) for e in examples}))
        return cls(float(alpha), include_residual, scaler.mean_, scaler.scale_, ridge.coef_, float(ridge.intercept_), rows)

    def predict(self, candidate: dict[str, Any], observed_groups: Iterable[str], probabilities: Iterable[float]) -> float:
        x = feature_vector(candidate, observed_groups, probabilities, include_residual=self.include_residual)
        return float(np.dot((x - self.mean) / self.scale, self.coefficients) + self.intercept)

    def predict_examples(self, examples: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([self.predict(e["candidate"], e["observed_groups"], e["probabilities"]) for e in examples])

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1, "alpha": self.alpha, "include_residual": self.include_residual,
            "feature_names": list(FEATURE_NAMES if self.include_residual else NO_RESIDUAL_FEATURES),
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(), "intercept": self.intercept,
            "fitting_rows": list(self.fitting_rows),
            "training_metadata": self.training_metadata or {},
        }
        target.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UtilityModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported utility model schema")
        include_residual = bool(payload["include_residual"])
        expected = FEATURE_NAMES if include_residual else NO_RESIDUAL_FEATURES
        if tuple(payload["feature_names"]) != expected:
            raise ValueError("utility model feature schema mismatch")
        vectors = [np.asarray(payload[key], dtype=float) for key in ("mean", "scale", "coefficients")]
        if any(v.shape != (len(expected),) or not np.all(np.isfinite(v)) for v in vectors) or np.any(vectors[1] <= 0):
            raise ValueError("invalid utility model coefficients")
        return cls(float(payload["alpha"]), include_residual, *vectors, float(payload["intercept"]), tuple(payload["fitting_rows"]), payload.get("training_metadata", {}))
