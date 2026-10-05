"""Explicit real-checkpoint integration test; no fixture model or substitute."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from nextcheck.model.tabpfn35 import EXPECTED_CHECKPOINT, TabPFN35Predictor


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_real_tabpfn35_held_out_prediction():
    artifacts = Path(os.getenv("NEXTCHECK_ARTIFACTS", ROOT / "artifacts"))
    data_path = artifacts / "data" / "features.npz"
    split_path = artifacts / "data" / "splits.json"
    if not data_path.exists() or not split_path.exists():
        pytest.skip("Official UCI data unavailable; this is not a passed integration test")
    with np.load(data_path) as data:
        X, y = data["X"], data["y"]
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    context = splits["context"]
    query = np.full((1, 70), np.nan)
    query[0, :5] = X[splits["utility"][0], :5]
    predictor = TabPFN35Predictor(X[context], y[context])
    assert predictor.provenance["ready"] is False
    probabilities = predictor.predict_proba(query)
    assert predictor.provenance["ready"] is True
    assert Path(predictor.provenance["checkpoint_path"]).name == EXPECTED_CHECKPOINT
    assert len(predictor.provenance["checkpoint_sha256"]) == 64
    assert predictor.provenance["classes_fitted"] == [0, 1, 2]
    assert probabilities.shape == (1, 3)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0], atol=1e-6)
