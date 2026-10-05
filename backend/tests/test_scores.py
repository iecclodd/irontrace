from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nextcheck.acquisition.scores import diagnostics, normalize_probabilities


VECTORS = json.loads((Path(__file__).parents[2] / "score_test_vectors.json").read_text())


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["name"])
def test_exact_finite_sample_identity(case):
    result = diagnostics(case["q0"], case["qm"])
    assert result["raw_kl"] == pytest.approx(result["information"] + result["residual"], abs=1e-10)
    assert all(np.isfinite(value) for value in result.values())
    if case.get("expect_information_zero"):
        assert result["information"] == pytest.approx(0, abs=1e-12)
    if case.get("expect_information_positive"):
        assert result["information"] > 0
    if case.get("expect_residual_zero"):
        assert result["residual"] == pytest.approx(0, abs=1e-12)
    if case.get("expect_residual_positive"):
        assert result["residual"] > 0
    if "true_label" in case:
        assert np.argmax(case["q0"]) != case["true_label"]


def test_random_normalized_distributions_preserve_identity():
    rng = np.random.default_rng(20261004)
    for _ in range(100):
        q0 = rng.dirichlet(np.ones(3))
        qm = rng.dirichlet(np.ones(3), size=16)
        result = diagnostics(q0, qm)
        assert abs(result["raw_kl"] - result["information"] - result["residual"]) < 1e-10


def test_invalid_probabilities_rejected_and_zeros_clipped():
    with pytest.raises(ValueError):
        normalize_probabilities([0.0, 0.0, 0.0], ndim=1)
    with pytest.raises(ValueError):
        normalize_probabilities([0.5, float("nan"), 0.5], ndim=1)
    normalized = normalize_probabilities([1.0, 0.0, 0.0], ndim=1)
    assert normalized.sum() == pytest.approx(1)
    assert np.all(normalized > 0)
