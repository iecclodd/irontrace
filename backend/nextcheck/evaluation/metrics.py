"""Outcome metrics for a frozen, three-class acquisition benchmark."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("cannot summarize zero cases")
    y = np.asarray([case["true_label"] for case in cases], dtype=int)
    p = np.asarray([case["probabilities"] for case in cases], dtype=float)
    if p.shape != (len(cases), 3) or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("invalid multiclass probabilities")
    if np.any((y < 0) | (y > 2)):
        raise ValueError("labels must be 0, 1, or 2")
    losses = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    one_hot = np.eye(3)[y]
    predicted = np.argmax(p, axis=1)
    costs = np.asarray([case["cost"] for case in cases], dtype=float)
    purchases = np.asarray([case["purchases"] for case in cases], dtype=float)
    elapsed = np.asarray([case["wall_time_s"] for case in cases], dtype=float)
    if not np.all(np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("invalid acquisition costs")
    return {
        "n_cases": len(cases),
        "class_support": {str(i): int(np.sum(y == i)) for i in range(3)},
        "log_loss": float(np.mean(losses)),
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "mean_cost": float(np.mean(costs)),
        "mean_purchases": float(np.mean(purchases)),
        "mean_wall_time_s": float(np.mean(elapsed)),
        "mean_execution_wall_time_s": float(np.mean([case.get("execution_wall_time_s", case["wall_time_s"]) for case in cases])),
        "cache_reused_cases": int(sum(bool(case.get("cache_reused", False)) for case in cases)),
        "prediction_calls": int(sum(case["prediction_calls"] for case in cases)),
        "prediction_query_rows": int(sum(case["prediction_query_rows"] for case in cases)),
        "hypothetical_query_rows": int(sum(case["hypothetical_query_rows"] for case in cases)),
        "physical_prediction_calls": int(sum(case.get("physical_prediction_calls", case["prediction_calls"]) for case in cases)),
        "physical_prediction_query_rows": int(sum(case.get("physical_prediction_query_rows", case["prediction_query_rows"]) for case in cases)),
        "physical_hypothetical_query_rows": int(sum(case.get("physical_hypothetical_query_rows", case["hypothetical_query_rows"]) for case in cases)),
    }


def objective(cases: list[dict[str, Any]], lambda_cost: float) -> float:
    metrics = summarize(cases)
    return float(metrics["log_loss"] + lambda_cost * metrics["mean_cost"])
