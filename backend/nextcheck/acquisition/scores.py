"""Finite-sample acquisition diagnostics for class-probability predictions."""

from __future__ import annotations

import numpy as np


EPSILON = 1e-8


def normalize_probabilities(values: object, *, ndim: int) -> np.ndarray:
    """Clip and renormalize one or more probability vectors in float64."""
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != ndim or probabilities.shape[-1] < 2:
        raise ValueError(f"expected a {ndim}-D array of class probabilities")
    if probabilities.size == 0 or not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite and nonempty")
    if np.any(probabilities < 0):
        raise ValueError("probabilities cannot be negative")
    sums = probabilities.sum(axis=-1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("each probability vector must have positive mass")
    probabilities = np.clip(probabilities / sums, EPSILON, None)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def diagnostics(q0: object, qm: object) -> dict[str, float]:
    """Calculate KL decomposition and entropy change with shared hypothetical draws."""
    current = normalize_probabilities(q0, ndim=1)
    hypothetical = normalize_probabilities(qm, ndim=2)
    if hypothetical.shape[1] != current.shape[0]:
        raise ValueError("hypothetical and current predictions have different classes")
    mean_prediction = hypothetical.mean(axis=0)
    log_hypothetical = np.log(hypothetical)
    raw = np.mean(np.sum(hypothetical * (log_hypothetical - np.log(current)), axis=1))
    information = np.mean(
        np.sum(hypothetical * (log_hypothetical - np.log(mean_prediction)), axis=1)
    )
    residual = np.sum(mean_prediction * (np.log(mean_prediction) - np.log(current)))
    # Equal distributions can produce roundoff-sized nonzero KL values.
    information = 0.0 if abs(information) < 1e-14 else information
    residual = 0.0 if abs(residual) < 1e-14 else residual
    raw = information + residual if abs(raw - information - residual) < 1e-14 else raw
    entropy_drop = -np.sum(current * np.log(current)) + np.mean(
        np.sum(hypothetical * log_hypothetical, axis=1)
    )
    return {
        "raw_kl": float(raw),
        "information": float(information),
        "residual": float(residual),
        "entropy_drop": float(entropy_drop),
    }
