"""Explicit official base TabPFN 3.5 classifier with verifiable provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np

from nextcheck.data.download import sha256_file

EXPECTED_CHECKPOINT = "tabpfn-v3.5-20260909.safetensors"
LABELS = (0, 1, 2)


class ModelUnavailable(RuntimeError):
    """The requested real 3.5 checkpoint could not fit or predict."""


def _context_hash(X: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (X, y):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


class TabPFN35Predictor:
    """Fit once on declared context rows, then predict aligned three-class outputs.

    Authentication/license access is handled only by the official TabPFN package.
    A missing checkpoint or access error is surfaced as ModelUnavailable.
    """

    def __init__(
        self,
        context_X: np.ndarray,
        context_y: np.ndarray,
        *,
        device: str = "auto",
        n_estimators: int = 4,
        random_state: int = 20261004,
    ) -> None:
        X = np.asarray(context_X, dtype=np.float64)
        y = np.asarray(context_y, dtype=np.int64)
        if X.ndim != 2 or X.shape[1] != 70 or y.shape != (X.shape[0],):
            raise ValueError("TabPFN context requires X[n,70] and y[n]")
        if X.shape[0] < 3 or not np.isin(y, LABELS).all() or set(np.unique(y)) != set(LABELS):
            raise ValueError("TabPFN context must cover all three pump leakage labels")
        if not np.isfinite(X).all():
            raise ValueError("Context feature matrix contains nonfinite values")
        self.provenance: dict = {
            "ready": False,
            "model": "TabPFN 3.5 base",
            "model_version": "v3.5",
            "expected_checkpoint": EXPECTED_CHECKPOINT,
            "device_requested": device,
            "n_estimators_requested": n_estimators,
            "random_state": random_state,
            "training_rows": X.shape[0],
            "context_sha256": _context_hash(X, y),
            "fit_mode": "fit_preprocessors",
        }
        try:
            import torch
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion

            self.provenance["tabpfn_version"] = importlib.metadata.version("tabpfn")
            self.provenance["torch_version"] = torch.__version__
            self.provenance["cuda_available"] = bool(torch.cuda.is_available())
            model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V3_5,
                device=device,
                n_estimators=n_estimators,
                random_state=random_state,
                fit_mode="fit_preprocessors",
                show_progress_bar=False,
            )
            checkpoint = Path(model.model_path).expanduser().resolve()
            if checkpoint.name != EXPECTED_CHECKPOINT:
                raise ModelUnavailable(f"Unexpected checkpoint selected: {checkpoint.name}")
            self.provenance["checkpoint_path"] = str(checkpoint)
            if not checkpoint.is_file():
                raise ModelUnavailable(
                    f"Official base TabPFN 3.5 checkpoint unavailable at {checkpoint}; "
                    "license acceptance or authentication may be required"
                )
            self.provenance["checkpoint_sha256"] = sha256_file(checkpoint)
            self.provenance["checkpoint_bytes"] = checkpoint.stat().st_size
            model.fit(X, y.astype(np.int64))
            classes = np.asarray(model.classes_)
            if set(classes.tolist()) != set(LABELS):
                raise ModelUnavailable(f"Fitted classes {classes.tolist()} do not match 0,1,2")
            self._model = model
            self._class_indices = [int(np.flatnonzero(classes == label)[0]) for label in LABELS]
            self.provenance["classes_fitted"] = classes.tolist()
            self.provenance["device_resolved"] = [str(item) for item in model.devices_]
            self.provenance["n_estimators_resolved"] = int(getattr(model, "n_estimators_", n_estimators))
        except ModelUnavailable:
            raise
        except Exception as exc:
            raise ModelUnavailable(f"TabPFN 3.5 fit failed: {type(exc).__name__}: {exc}") from exc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        query = np.asarray(X, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 70:
            raise ValueError("TabPFN query requires X[n,70]")
        if np.isinf(query).any() or query.shape[0] == 0:
            raise ValueError("TabPFN query has infinity or zero rows")
        try:
            raw = np.asarray(self._model.predict_proba(query), dtype=np.float64)
        except Exception as exc:
            raise ModelUnavailable(f"TabPFN 3.5 prediction failed: {type(exc).__name__}: {exc}") from exc
        if raw.shape != (len(query), 3):
            raise ModelUnavailable(f"TabPFN returned probability shape {raw.shape}, expected {(len(query), 3)}")
        aligned = raw[:, self._class_indices]
        if not np.isfinite(aligned).all() or (aligned < 0).any() or not np.allclose(aligned.sum(axis=1), 1.0, atol=1e-6):
            raise ModelUnavailable("TabPFN returned invalid class probabilities")
        self.provenance["ready"] = True
        self.provenance["first_prediction_rows"] = self.provenance.get("first_prediction_rows", len(query))
        return aligned
