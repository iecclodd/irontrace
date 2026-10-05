import numpy as np
import pytest

from nextcheck.acquisition.sampler import DonorSampler
from nextcheck.agent.state import ObservedCase


PANELS = [
    {"id": "initial", "cost": 0, "features": ["a", "b"]},
    {"id": "pair", "cost": 1, "features": ["c", "d"]},
]
CONTEXT = np.array([
    [0, 0, 10, 100],
    [1, 1, 20, 200],
    [2, 2, 30, 300],
    [3, 3, 40, 400],
], dtype=float)


def test_joint_donors_and_nan_mask():
    sampler = DonorSampler(CONTEXT, PANELS)
    observed = ObservedCase(("initial",), {"a": 1.0, "b": 1.0})
    rows, info = sampler.sample(observed, "pair", seed=7, draws=100)
    assert rows.shape == (100, 4)
    assert np.all(rows[:, :2] == [1, 1])
    assert all(tuple(pair) in {(10, 100), (20, 200), (30, 300), (40, 400)} for pair in rows[:, 2:])
    assert info["unique_donor_count"] <= info["neighborhood_size"]


def test_unobserved_values_cannot_affect_neighbors():
    sampler = DonorSampler(CONTEXT, PANELS)
    observed = ObservedCase(("initial",), {"a": 1.0, "b": 1.0})
    rows1, info1 = sampler.sample(observed, "pair", seed=19)
    altered = CONTEXT.copy()
    altered[:, 2:] *= 1000
    rows2, info2 = DonorSampler(altered, PANELS).sample(observed, "pair", seed=19)
    assert info1["neighbor_distances"] == info2["neighbor_distances"]
    assert np.all(rows2[:, 2:] == rows1[:, 2:] * 1000)


def test_visible_validation_rejects_hidden_feature():
    sampler = DonorSampler(CONTEXT, PANELS)
    with pytest.raises(ValueError):
        sampler.sample(ObservedCase(("initial",), {"a": 1, "b": 1, "c": 2}), "pair", 1)
