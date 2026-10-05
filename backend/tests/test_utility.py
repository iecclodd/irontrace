from __future__ import annotations

import numpy as np
import pytest

from nextcheck.acquisition.utility import UtilityModel, feature_vector
from nextcheck.evaluation.benchmark import partial_states, select_utility_models


def example(row: int, residual: float, delta: float):
    return {"row_index": row, "candidate": {"group_id": "flow", "information": 0.2,
            "residual": residual, "cost": 1000, "hidden_value": 999,
            "sampler": {"unique_donor_count": 3, "mean_distance": 0.5}},
        "observed_groups": ["initial"], "probabilities": [0.2, 0.3, 0.5], "delta": delta}


def test_value_features_exclude_cost_labels_hidden_values():
    item = example(1, 0.3, 0.4)
    before = feature_vector(item["candidate"], item["observed_groups"], item["probabilities"])
    item["candidate"].update(cost=1, hidden_value=-4, true_label=2)
    after = feature_vector(item["candidate"], item["observed_groups"], item["probabilities"])
    np.testing.assert_array_equal(before, after)
    without = feature_vector(item["candidate"], item["observed_groups"], item["probabilities"], include_residual=False)
    assert len(before) == len(without) + 1


def test_utility_fit_save_load_and_ablation(tmp_path):
    fitting = [example(i, float(i), 2.0 * i + 1.0) for i in range(8)]
    selection = [example(i, float(i), 2.0 * i + 1.0) for i in range(8, 10)]
    models, tuning = select_utility_models(fitting, selection, alphas=(0.01, 1.0))
    assert set(models) == {"value", "value_no_residual"}
    assert tuning["value"]["selection_mse"] < tuning["value_no_residual"]["selection_mse"]
    path = tmp_path / "value.json"
    models["value"].save(path)
    loaded = UtilityModel.load(path)
    assert loaded.predict(selection[0]["candidate"], ["initial"], [0.2, 0.3, 0.5]) == pytest.approx(models["value"].predict_examples(selection)[0])
    assert len(loaded.fitting_rows) == 8


def test_four_nonterminal_states_per_row():
    states = partial_states(100)
    assert len(states) == 4
    assert states[0] == ()
    assert [len(s) for s in states] == [0, 1, 2, 3]
    assert states == partial_states(100)
    assert all(len(set(s)) == len(s) for s in states)


def test_selection_row_overlap_rejected():
    with pytest.raises(ValueError, match="overlap"):
        select_utility_models([example(1, 0, 0)], [example(1, 0, 0)])
