"""Row-local feature extraction from original UCI hydraulic sensor matrices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .download import DataValidationError, SENSORS, sha256_file

RATES_HZ = {sensor: 100 for sensor in ("PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1")}
RATES_HZ.update({sensor: 10 for sensor in ("FS1", "FS2")})
RATES_HZ.update({sensor: 1 for sensor in ("TS1", "TS2", "TS3", "TS4", "VS1")})
STATS = ("mean", "std", "min", "max", "slope")
EXPECTED_CYCLES = 2205


def summarize(matrix: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise DataValidationError("Each cycle needs at least two samples")
    if not np.isfinite(matrix).all():
        raise DataValidationError("Sensor matrix contains nonfinite values")
    t = np.arange(matrix.shape[1], dtype=np.float64) / sample_rate_hz
    centered_t = t - t.mean()
    slope = (matrix @ centered_t) / np.dot(centered_t, centered_t)
    output = np.column_stack((matrix.mean(axis=1), matrix.std(axis=1, ddof=0), matrix.min(axis=1), matrix.max(axis=1), slope))
    if not np.isfinite(output).all():
        raise DataValidationError("Extracted sensor summaries are nonfinite")
    return output


def _raw_file(raw_dir: Path, name: str) -> Path:
    matches = list(raw_dir.rglob(name))
    if len(matches) != 1:
        raise DataValidationError(f"Expected exactly one {name}, found {len(matches)}")
    return matches[0]


def prepare(raw_dir: Path, output_dir: Path) -> dict:
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = _raw_file(raw_dir, "profile.txt")
    profile = np.loadtxt(profile_path, dtype=np.float64, ndmin=2)
    if profile.shape != (EXPECTED_CYCLES, 5) or not np.isfinite(profile).all():
        raise DataValidationError(f"Expected {EXPECTED_CYCLES} profile rows and five finite columns")
    target = profile[:, 2]
    if not np.isin(target, [0, 1, 2]).all():
        raise DataValidationError("Pump leakage target must contain only labels 0, 1, 2")
    columns = []
    names = []
    raw_hashes = {"profile.txt": sha256_file(profile_path)}
    for sensor in SENSORS:
        source = _raw_file(raw_dir, f"{sensor}.txt")
        raw_hashes[source.name] = sha256_file(source)
        matrix = np.loadtxt(source, dtype=np.float64, ndmin=2)
        expected_samples = RATES_HZ[sensor] * 60
        if matrix.shape != (EXPECTED_CYCLES, expected_samples):
            raise DataValidationError(f"{sensor}: expected {(EXPECTED_CYCLES, expected_samples)}, got {matrix.shape}")
        columns.append(summarize(matrix, RATES_HZ[sensor]))
        names.extend(f"{sensor}__{stat}" for stat in STATS)
        del matrix
    X = np.hstack(columns)
    if X.shape != (EXPECTED_CYCLES, 70) or len(set(names)) != 70:
        raise DataValidationError("Feature matrix must have 70 unique physical-sensor features")
    cycle_ids = np.arange(EXPECTED_CYCLES, dtype=np.int32)
    feature_names = np.asarray(names, dtype="<U24")
    feature_path = output_dir / "features.npz"
    np.savez_compressed(feature_path, X=X, y=target.astype(np.int8), cycle_ids=cycle_ids, feature_names=feature_names)
    metadata = {
        "source": "UCI Condition monitoring of hydraulic systems, dataset 447",
        "source_url": "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems",
        "doi": "10.24432/C5CW21",
        "citation": "Helwig, N., Pignanelli, E., & Schütze, A. (2015). Condition monitoring of hydraulic systems [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21",
        "license": "CC BY 4.0",
        "target": "profile.txt column 2 (zero-based): internal pump leakage",
        "rows": EXPECTED_CYCLES,
        "features": len(names),
        "raw_sha256": raw_hashes,
        "features_sha256": sha256_file(feature_path),
        "feature_names": names,
        "sample_rates_hz": RATES_HZ,
        "feature_definition": "per-cycle mean, population std, min, max, least-squares slope against seconds",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata
