import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from nextcheck.data.download import DataValidationError, extract_required
from nextcheck.data.extract import STATS, summarize
from nextcheck.data.split import CAPS, make_splits


ROOT = Path(__file__).resolve().parents[2]


def test_manifest_covers_exactly_70_physical_features():
    manifest = json.loads((ROOT / "demo_manifest.json").read_text(encoding="utf-8"))
    panels = manifest["panels"]
    assert [panel["id"] for panel in panels] == ["initial", "pressure", "flow", "vibration", "power", "temperature"]
    assert sum(panel["cost"] for panel in panels) == 11
    feature_names = [name for panel in panels for name in panel["features"]]
    assert len(feature_names) == len(set(feature_names)) == 70
    assert all(name.rsplit("__", 1)[1] in STATS for name in feature_names)
    assert all(name.split("__")[0] not in {"CE", "CP", "SE", "profile", "cycle_id"} for name in feature_names)
    assert panels[0]["sensors"] == ["TS1"]


def test_row_local_population_statistics_and_time_slope():
    matrix = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 4.0, 4.0, 4.0]])
    result = summarize(matrix, sample_rate_hz=2)
    np.testing.assert_allclose(result[0], [2.5, np.sqrt(1.25), 1, 4, 2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result[1], [4, 0, 4, 4, 0], rtol=0, atol=1e-12)
    with pytest.raises(DataValidationError):
        summarize(np.ones((2, 1)), sample_rate_hz=1)


def test_group_splits_keep_all_five_partitions_disjoint():
    y = np.arange(2205) % 3
    result = make_splits(y)
    block_ids = np.asarray(result["block_ids"])
    parts = ["context", "utility", "selection", "test", "unused"]
    assert sorted(row for name in parts for row in result[name]) == list(range(len(y)))
    groups = [set(block_ids[result[name]]) for name in parts]
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert a.isdisjoint(b)
    assert all(0 < len(result[name]) <= CAPS[name] for name in CAPS)
    assert result == make_splits(y)


def test_zip_path_traversal_is_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("../profile.txt", "bad")
    with pytest.raises(DataValidationError, match="Unsafe ZIP path"):
        extract_required(archive, tmp_path / "raw")


def test_real_prepared_data_if_available():
    import os

    artifacts = Path(os.getenv("NEXTCHECK_ARTIFACTS", ROOT / "artifacts"))
    path = artifacts / "data" / "features.npz"
    if not path.exists():
        pytest.skip("Official UCI data has not been prepared in this checkout")
    with np.load(path) as data:
        X, y, names, cycles = (data[key] for key in ("X", "y", "feature_names", "cycle_ids"))
        assert X.shape == (2205, 70)
        assert y.shape == cycles.shape == (2205,)
        assert np.isfinite(X).all()
        manifest = json.loads((ROOT / "demo_manifest.json").read_text(encoding="utf-8"))
        assert names.tolist() == [name for panel in manifest["panels"] for name in panel["features"]]
        assert set(y.tolist()) == {0, 1, 2}
