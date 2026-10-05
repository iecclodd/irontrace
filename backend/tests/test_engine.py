import json

import numpy as np
import pytest

from nextcheck.agent.loop import AcquisitionEngine
from nextcheck.agent.state import ObservedCase
from nextcheck.replay.environment import ReplayEnvironment


PANELS = [
    {"id": "initial", "cost": 0, "features": ["sensor0_mean"]},
    {"id": "a", "cost": 1, "features": ["sensor1_mean"]},
    {"id": "b", "cost": 2, "features": ["sensor2_mean"]},
]
CONTEXT = np.array([[0.0, -2.0, 2.0], [0.0, 2.0, -2.0], [1.0, 0.0, 0.0]])


class FixturePredictor:
    """Explicit deterministic fixture; never used as a production model."""

    def __init__(self, constant=False):
        self.constant = constant
        self.calls = []

    def predict_proba(self, X):
        self.calls.append(np.asarray(X).copy())
        probabilities = []
        for row in X:
            if self.constant or (np.isnan(row[1]) and np.isnan(row[2])):
                probabilities.append([0.2, 0.6, 0.2])
            elif not np.isnan(row[1]):
                probabilities.append([0.7 if row[1] > 0 else 0.1, 0.2, 0.1 if row[1] > 0 else 0.7])
            else:
                probabilities.append([0.1, 0.3 if row[2] > 0 else 0.8, 0.6 if row[2] > 0 else 0.1])
        return np.asarray(probabilities)


class FixtureUtility:
    def __init__(self, score):
        self.score = score

    def predict(self, candidate, observed_groups, probabilities):
        assert candidate["sampler"]["draws"] == 16
        assert probabilities.shape == (3,)
        return self.score


def test_engine_batches_queries_and_returns_ranked_finite_diagnostics():
    predictor = FixturePredictor()
    engine = AcquisitionEngine(predictor, CONTEXT, {"panels": PANELS})
    observed = ObservedCase(("initial",), {"sensor0_mean": 0.0})
    result = engine.analyze(observed, 3, lambda_cost=0, seed=12)
    assert result["prediction_calls"] == 2
    assert [len(query) for query in predictor.calls] == [1, 32]
    assert result["hypothetical_query_rows"] == 32
    assert len(result["candidates"]) == 2
    assert result["selected_action"] in {"a", "b"}
    assert sum(result["probabilities"]) == pytest.approx(1)
    assert all(np.isfinite(result["probabilities"]))
    for candidate in result["candidates"]:
        assert candidate["raw_kl"] == pytest.approx(candidate["information"] + candidate["residual"], abs=1e-10)
        assert candidate["sampler"]["neighborhood_size"] == 3


def test_nonpositive_value_stops_and_zero_cost_is_safe():
    panels = [dict(panel) for panel in PANELS]
    panels[1]["cost"] = 0
    engine = AcquisitionEngine(FixturePredictor(), CONTEXT, panels, FixtureUtility(-0.01))
    observed = ObservedCase(("initial",), {"sensor0_mean": 0.0})
    stopped = engine.analyze(observed, 0, policy="value")
    assert stopped["selected_action"] is None
    assert stopped["stop_reason"] == "no_predicted_net_value"
    assert stopped["candidates"][0]["group_id"] == "a"
    assert stopped["candidates"][0]["affordable"] is True
    assert stopped["candidates"][1]["affordable"] is False
    engine = AcquisitionEngine(FixturePredictor(), CONTEXT, panels, FixtureUtility(0.01))
    selected = engine.analyze(observed, 0, policy="value")
    assert selected["selected_action"] == "a"


def test_hidden_row_and_label_invariance_before_acquisition():
    first = ReplayEnvironment([0, 1, 2], label=0, manifest=PANELS)
    changed = ReplayEnvironment([0, -999, 999], label=2, manifest=PANELS)
    assert first.initial_observation() == changed.initial_observation()
    engine = AcquisitionEngine(FixturePredictor(), CONTEXT, PANELS)
    a = engine.analyze(first.initial_observation(), 3, seed=4)
    b = engine.analyze(changed.initial_observation(), 3, seed=4)
    assert a["state_hash"] == b["state_hash"]
    assert a["selected_action"] == b["selected_action"]
    assert a["probabilities"] == b["probabilities"]
    assert a["candidates"] == b["candidates"]
    serialized = json.dumps({key: a[key] for key in ("state_hash", "selected_action", "probabilities", "candidates")})
    assert "sensor1_mean\": -999" not in serialized
    assert "sensor2_mean\": 999" not in serialized


def test_budget_and_no_groups_stop_reasons():
    engine = AcquisitionEngine(FixturePredictor(), CONTEXT, PANELS)
    initial = ObservedCase(("initial",), {"sensor0_mean": 0.0})
    assert engine.analyze(initial, 0)["stop_reason"] == "budget_exhausted"
    complete = ObservedCase(("initial", "a", "b"), {"sensor0_mean": 0, "sensor1_mean": 1, "sensor2_mean": 2})
    assert engine.analyze(complete, 0)["stop_reason"] == "no_groups_remaining"


def test_constant_fixture_stops_on_zero_information():
    engine = AcquisitionEngine(FixturePredictor(constant=True), CONTEXT, PANELS)
    state = ObservedCase(("initial",), {"sensor0_mean": 0.0})
    result = engine.analyze(state, 3, lambda_cost=0)
    assert result["selected_action"] is None
    assert result["stop_reason"] == "no_predicted_net_value"


@pytest.mark.parametrize("policy", ["random", "initial"])
def test_simple_policies_do_not_query_hypothetical_rows(policy):
    predictor = FixturePredictor()
    engine = AcquisitionEngine(predictor, CONTEXT, PANELS)
    state = ObservedCase(("initial",), {"sensor0_mean": 0.0})
    result = engine.analyze(state, 3, policy=policy, seed=19)
    assert [len(query) for query in predictor.calls] == [1]
    assert result["prediction_calls"] == 1
    assert result["hypothetical_query_rows"] == 0
    assert all(candidate["information"] is None and candidate["sampler"] is None for candidate in result["candidates"])
    assert result["selected_action"] in {"a", "b"} if policy == "random" else result["selected_action"] is None
    assert engine.analyze(state, 3, policy=policy, seed=19)["selected_action"] == result["selected_action"]
