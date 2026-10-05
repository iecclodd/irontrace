"""Execute a real base TabPFN 3.5 inference on a held-out development row."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from nextcheck.model.tabpfn35 import ModelUnavailable, TabPFN35Predictor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path(os.getenv("NEXTCHECK_ARTIFACTS", ROOT / "artifacts")))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output = args.artifacts / "model_status.json"
    args.artifacts.mkdir(parents=True, exist_ok=True)
    status: dict = {"ready": False, "model": "TabPFN 3.5 base", "blockers": []}
    try:
        data_dir = args.artifacts / "data"
        with np.load(data_dir / "features.npz") as data:
            X, y = data["X"], data["y"]
            names = data["feature_names"].tolist()
        splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
        if names[:5] != [f"TS1__{stat}" for stat in ("mean", "std", "min", "max", "slope")]:
            raise ValueError("Initial panel feature order does not match manifest")
        context = np.asarray(splits["context"], dtype=int)
        held_out = int(splits["utility"][0])
        query = np.full((1, X.shape[1]), np.nan, dtype=np.float64)
        query[0, :5] = X[held_out, :5]
        start = time.perf_counter()
        predictor = TabPFN35Predictor(X[context], y[context], device=args.device)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        probabilities = predictor.predict_proba(query)
        predict_seconds = time.perf_counter() - start
        status.update({
            "ready": True,
            "provenance": predictor.provenance,
            "smoke": {"partition": "utility", "cycle_id": held_out, "observed_panel": "initial", "probabilities_0_1_2": probabilities[0].tolist(), "fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "test_label_accessed": False},
        })
    except (OSError, ValueError, ModelUnavailable) as exc:
        status["blockers"].append(f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        status["blockers"].append(f"Unexpected {type(exc).__name__}: {exc}")
    output.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
