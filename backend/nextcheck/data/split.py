"""Predeclared grouped partitions and resource caps; test labels never tune splits."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from .download import DataValidationError

SEED = 20261004
BLOCK_SIZE = 20
CAPS = {"context": 256, "utility": 96, "selection": 96, "test": 128}
FRACTIONS = {"context": 0.50, "utility": 0.20, "selection": 0.15, "test": 0.15}


def make_splits(y: np.ndarray, seed: int = SEED) -> dict:
    """Split contiguous 20-cycle blocks, then select capped rows within each pool.

    This block ID is a declared proxy for within-rig temporal dependence, not a
    documented run ID. Unused rows remain out of every model and donor pool.
    """
    y = np.asarray(y)
    if y.ndim != 1 or len(y) < 100 or not np.isin(y, [0, 1, 2]).all():
        raise DataValidationError("Expected a one-dimensional 0/1/2 target with at least 100 rows")
    n = len(y)
    block_ids = np.arange(n, dtype=np.int32) // BLOCK_SIZE
    blocks = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    randomized_blocks = rng.permutation(blocks)
    n_context = round(len(blocks) * FRACTIONS["context"])
    n_utility = round(len(blocks) * FRACTIONS["utility"])
    n_selection = round(len(blocks) * FRACTIONS["selection"])
    group_counts = (n_context, n_utility, n_selection, len(blocks) - n_context - n_utility - n_selection)
    result: dict = {"block_ids": block_ids.tolist()}
    offset = 0
    chosen = []
    assigned_groups = {}
    for name, count in zip(CAPS, group_counts, strict=True):
        group_pool = randomized_blocks[offset:offset + count]
        offset += count
        assigned_groups[name] = [int(x) for x in group_pool]
        # Keep whole groups. A resource cap may leave a few slots unused.
        selected = []
        for group in group_pool:
            rows = np.flatnonzero(block_ids == group).astype(int).tolist()
            if len(selected) + len(rows) > CAPS[name]:
                break
            selected.extend(rows)
        result[name] = selected
        chosen.extend(selected)
    result["unused"] = np.setdiff1d(np.arange(n), chosen, assume_unique=False).astype(int).tolist()
    selected_groups = {name: set(block_ids[result[name]]) for name in CAPS}
    selected_groups["unused"] = set(block_ids[result["unused"]])
    for a, a_groups in selected_groups.items():
        for b, b_groups in selected_groups.items():
            if a != b and a_groups & b_groups:
                raise AssertionError("Original cycle block crossed partitions")
    coverage = {name: {str(label): int(np.sum(y[result[name]] == label)) for label in [0, 1, 2]} for name in ("context", "utility", "selection")}
    if any(count == 0 for classes in coverage.values() for count in classes.values()):
        raise DataValidationError(f"Predeclared split lacks development class coverage: {coverage}")
    result["metadata"] = {
        "seed": seed,
        "method": "random permutation of contiguous 20-cycle proxy blocks; capped group-order pools",
        "group_source": "original row index // 20; no documented run/batch IDs available",
        "block_size": BLOCK_SIZE,
        "fractions": FRACTIONS,
        "caps": CAPS,
        "assigned_groups": assigned_groups,
        "development_class_counts": coverage,
        "selected_counts": {name: len(result[name]) for name in CAPS},
        "unused_count": len(result["unused"]),
        "split_code_sha256": hashlib.sha256(inspect.getsource(make_splits).encode()).hexdigest(),
        "test_class_counts_withheld": True,
    }
    return result


def save_splits(y: np.ndarray, path: Path, seed: int = SEED) -> dict:
    result = make_splits(y, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
