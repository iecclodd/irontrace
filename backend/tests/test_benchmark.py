from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from nextcheck.acquisition.utility import GROUPS
from nextcheck.evaluation.benchmark import (
    CountingPredictor, DiagnosticCache, all_subsets, mask_row, panel_columns,
    run_actions, score_outcomes, validate_splits,
)


def test_split_rejects_shared_blocks_and_accounts_for_unused():
    splits = {"context": [0, 1], "utility": [2], "selection": [3], "test": [4],
              "unused": [5], "block_ids": [0, 0, 1, 2, 3, 4]}
    validate_splits(splits, 6)
    splits["block_ids"][2] = 0
    with pytest.raises(ValueError, match="block"):
        validate_splits(splits, 6)
    splits["block_ids"][2] = 1
    splits["test"] = [2, 4]
    with pytest.raises(ValueError, match="row"):
        validate_splits(splits, 6)


def test_exactly_32_static_subsets_and_atomic_mask():
    names = ["TS1__mean", "PS1__mean", "FS1__mean", "VS1__mean", "EPS1__mean", "TS2__mean"]
    manifest = {"panels": [{"id": name, "features": [feature], "cost": 0 if name == "initial" else 1}
                           for name, feature in zip(("initial", *GROUPS), names)]}
    columns = panel_columns(manifest, names)
    masked = mask_row(np.arange(6.0), ("flow",), columns)
    np.testing.assert_equal(masked[[0, 2]], [0, 2])
    assert np.isnan(masked[[1, 3, 4, 5]]).all()
    assert len(all_subsets()) == 32


class FakePredictor:
    provenance = {"fixture": True}

    def predict_proba(self, rows):
        result = []
        for row in rows:
            value = 0 if np.isnan(row[0]) else row[0]
            result.append([0.5 + value * 0.01, 0.3, 0.2 - value * 0.01])
        return np.asarray(result)


@dataclass
class FakeObservedCase:
    observed_groups: tuple[str, ...]
    values: dict[str, float]


class FakeEngine:
    def __init__(self, predictor):
        self.predictor = predictor

    def analyze(self, observed, remaining_budget, policy, seed):
        assert "PS1__mean" not in observed.values
        # The fixture mimics the real engine boundary: only visible values.
        probabilities = self.predictor.predict_proba([[observed.values["TS1__mean"], np.nan]])[0]
        return {"probabilities": probabilities.tolist(), "candidates": [
            {"group_id": "pressure", "raw_kl": 0.1, "information": 0.1,
             "residual": 0.0, "entropy_drop": 0.1, "sampler": {}}
        ]}


class NegativeValue:
    def predict(self, candidate, observed_groups, probabilities):
        return -0.1


def test_nonpositive_value_stops_without_hidden_reveal(monkeypatch):
    agent_package = types.ModuleType("nextcheck.agent")
    agent_package.__path__ = []
    state_module = types.ModuleType("nextcheck.agent.state")
    state_module.ObservedCase = FakeObservedCase
    monkeypatch.setitem(sys.modules, "nextcheck.agent", agent_package)
    monkeypatch.setitem(sys.modules, "nextcheck.agent.state", state_module)
    columns = {"initial": [0], "pressure": [1], **{g: [] for g in GROUPS if g != "pressure"}}
    names = ["TS1__mean", "PS1__mean"]
    x = np.asarray([[1.0, 10.0], [1.0, -10.0]])
    predictor = CountingPredictor(FakePredictor())
    cache = DiagnosticCache(FakeEngine(predictor), x, columns, names)
    costs = {"initial": 0, **{g: 1 for g in GROUPS}}
    outcomes = [run_actions(index=i, row=x[i], policy="value", budget=1,
        lambda_cost=0.02, threshold=0, random_seed=1, predictor=predictor,
        cache=cache, columns=columns, costs=costs, models={"value": NegativeValue()},
        prior=np.array([0.5, 0.3, 0.2])) for i in range(2)]
    assert outcomes[0]["probabilities"] == outcomes[1]["probabilities"]
    assert all(item["cost"] == 0 and item["purchases"] == 0 for item in outcomes)
    assert outcomes[0]["trace"] == []
    assert score_outcomes(outcomes, np.array([0, 2]))[1]["true_label"] == 2


def test_random_uses_no_hypothetical_queries():
    columns = {"initial": [0], "pressure": [1], **{g: [] for g in GROUPS if g != "pressure"}}
    costs = {"initial": 0, "pressure": 4, "flow": 3, "vibration": 1, "power": 2, "temperature": 1}
    class NoDiagnostics:
        def get(self, *_):
            raise AssertionError("random acquisition must not compute diagnostics")
    outcome = run_actions(index=0, row=np.array([1.0, 10.0]), policy="random", budget=6,
        lambda_cost=0.02, threshold=0, random_seed=1, predictor=CountingPredictor(FakePredictor()),
        cache=NoDiagnostics(), columns=columns, costs=costs, models={}, prior=np.array([0.5, 0.3, 0.2]))
    assert outcome["cost"] <= 6
    assert outcome["prediction_calls"] == 1
    assert outcome["prediction_query_rows"] == 1
    assert outcome["hypothetical_query_rows"] == 0
