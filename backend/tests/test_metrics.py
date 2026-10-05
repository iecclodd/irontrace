import numpy as np
import pytest

from nextcheck.evaluation.metrics import objective, summarize


def test_multiclass_metrics_and_costs():
    cases = [
        {"true_label": 0, "probabilities": [0.8, 0.1, 0.1], "cost": 0,
         "purchases": 0, "prediction_calls": 1, "prediction_query_rows": 1,
         "hypothetical_query_rows": 0, "wall_time_s": 0.1},
        {"true_label": 1, "probabilities": [0.1, 0.7, 0.2], "cost": 3,
         "purchases": 1, "prediction_calls": 2, "prediction_query_rows": 17,
         "hypothetical_query_rows": 16, "wall_time_s": 0.2},
    ]
    metrics = summarize(cases)
    assert metrics["log_loss"] == pytest.approx((-np.log(0.8) - np.log(0.7)) / 2)
    assert metrics["accuracy"] == 1
    assert metrics["balanced_accuracy"] == 1
    assert metrics["brier"] == pytest.approx((0.06 + 0.14) / 2)
    assert metrics["mean_cost"] == 1.5
    assert metrics["prediction_calls"] == 3
    assert metrics["hypothetical_query_rows"] == 16
    assert objective(cases, 0.02) == pytest.approx(metrics["log_loss"] + 0.03)


def test_invalid_probabilities_rejected():
    case = {"true_label": 0, "probabilities": [0.4, 0.4, 0.4], "cost": 0,
            "purchases": 0, "prediction_calls": 0, "prediction_query_rows": 0,
            "hypothetical_query_rows": 0, "wall_time_s": 0}
    with pytest.raises(ValueError):
        summarize([case])
