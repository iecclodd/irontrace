"""Single-step, visible-state panel recommendation engine."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from time import perf_counter
from typing import Protocol

import numpy as np

from nextcheck.acquisition.policies import score_per_cost
from nextcheck.acquisition.sampler import DonorSampler
from nextcheck.acquisition.scores import diagnostics, normalize_probabilities
from .state import ObservedCase


class Predictor(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class UtilityPredictor(Protocol):
    def predict(
        self, candidate: dict[str, object], observed_groups: tuple[str, ...], probabilities: np.ndarray
    ) -> float: ...


DIAGNOSTIC_POLICIES = {
    "raw_kl": "raw_kl",
    "information": "information",
    "entropy_drop": "entropy_drop",
}
POLICIES = set(DIAGNOSTIC_POLICIES) | {
    "value", "value_no_residual", "random", "static", "all", "initial"
}


class AcquisitionEngine:
    def __init__(
        self, predictor: Predictor, context_X: object, manifest: object,
        utility_model: UtilityPredictor | None = None,
    ) -> None:
        self.predictor = predictor
        self.sampler = DonorSampler(context_X, manifest)
        self.panels = self.sampler.panels
        self.utility_model = utility_model

    def _predict(self, rows: np.ndarray) -> np.ndarray:
        output = np.asarray(self.predictor.predict_proba(rows), dtype=np.float64)
        if output.ndim != 2 or output.shape[0] != len(rows) or output.shape[1] != 3:
            raise ValueError("predictor must return one aligned three-class probability row per query")
        return normalize_probabilities(output, ndim=2)

    def analyze(
        self, observed: ObservedCase, remaining_budget: float, policy: str = "information",
        lambda_cost: float = 0.02, seed: int = 20261004,
    ) -> dict[str, object]:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        if policy in {"value", "value_no_residual"} and self.utility_model is None:
            raise ValueError(f"{policy} policy requires a fitted utility model")
        if not math.isfinite(remaining_budget) or remaining_budget < 0:
            raise ValueError("remaining_budget must be nonnegative and finite")
        if not math.isfinite(lambda_cost) or lambda_cost < 0:
            raise ValueError("lambda_cost must be nonnegative and finite")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        start = perf_counter()
        current_row = self.sampler.observed_row(observed)
        state = {
            "observed_groups": observed.observed_groups,
            "values": {key: observed.values[key] for key in sorted(observed.values)},
            "remaining_budget": remaining_budget,
            "panels": [
                {"id": panel.id, "cost": panel.cost, "features": panel.features}
                for panel in self.panels
            ],
        }
        state_hash = sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        current = self._predict(current_row[None, :])[0]
        current_entropy = float(-np.sum(current * np.log(current)))
        margin = float(np.sort(current)[-1] - np.sort(current)[-2])
        remaining = [panel for panel in self.panels if panel.id not in observed.observed_groups]
        candidates: list[dict[str, object]] = []
        batches: list[np.ndarray] = []
        sampled: list[tuple[int, dict[str, object]]] = []
        for position, panel in enumerate(remaining):
            affordable = panel.cost <= remaining_budget
            candidate: dict[str, object] = {
                "group_id": panel.id,
                "cost": panel.cost,
                "affordable": affordable,
                "raw_kl": None,
                "information": None,
                "residual": None,
                "entropy_drop": None,
                "current_entropy": current_entropy,
                "margin": margin,
                "predicted_delta": None,
                "net_value": None,
                "sampler": None,
            }
            candidates.append(candidate)
            if affordable and policy not in {"random", "initial"}:
                candidate_seed = int.from_bytes(
                    sha256(f"{seed}:{state_hash}:{panel.id}".encode()).digest()[:8], "big"
                )
                rows, donor_diagnostics = self.sampler.sample(
                    observed, panel.id, candidate_seed, draws=16
                )
                sampled.append((position, donor_diagnostics))
                batches.append(rows)
        prediction_calls = 1
        hypothetical_query_rows = sum(len(batch) for batch in batches)
        if batches:
            predicted = self._predict(np.concatenate(batches, axis=0))
            prediction_calls += 1
            offset = 0
            for (position, donor_diagnostics), batch in zip(sampled, batches):
                group_predictions = predicted[offset:offset + len(batch)]
                offset += len(batch)
                candidate = candidates[position]
                candidate.update(diagnostics(current, group_predictions))
                candidate["sampler"] = donor_diagnostics
                if self.utility_model is not None:
                    predicted_delta = float(
                        self.utility_model.predict(candidate, observed.observed_groups, current)
                    )
                    if not math.isfinite(predicted_delta):
                        raise ValueError("utility model returned a nonfinite prediction")
                    candidate["predicted_delta"] = predicted_delta
                score = (
                    float(candidate["predicted_delta"])
                    if policy in {"value", "value_no_residual"}
                    else float(candidate[DIAGNOSTIC_POLICIES.get(policy, "information")])
                )
                candidate["net_value"] = score - lambda_cost * float(candidate["cost"])

        def rank(candidate: dict[str, object]) -> tuple[float, float, int]:
            if not candidate["affordable"]:
                return (-math.inf, -math.inf, -self._panel_order(candidate["group_id"]))
            if policy in {"value", "value_no_residual"}:
                primary = float(candidate["net_value"])
            elif policy in DIAGNOSTIC_POLICIES:
                primary = score_per_cost(float(candidate[policy]), float(candidate["cost"]))
            else:
                primary = float(candidate["net_value"])
            return (primary, float(candidate["net_value"]), -self._panel_order(candidate["group_id"]))

        if policy == "random":
            # Random baselines use the same fixed state seed, without consulting hidden values.
            rng_seed = int.from_bytes(sha256(f"{seed}:{state_hash}:random".encode()).digest()[:8], "big")
            shuffled = np.random.default_rng(rng_seed).permutation(len(candidates))
            random_order = {int(index): position for position, index in enumerate(shuffled)}
            candidates.sort(key=lambda candidate: (not candidate["affordable"], random_order[self._remaining_order(candidate["group_id"], remaining)]))
        elif policy in {"static", "all", "initial"}:
            candidates.sort(key=lambda candidate: (not candidate["affordable"], self._panel_order(candidate["group_id"])))
        else:
            candidates.sort(key=rank, reverse=True)

        affordable_candidates = [candidate for candidate in candidates if candidate["affordable"]]
        if not remaining:
            selected_action = None
            stop_reason = "no_groups_remaining"
        elif not affordable_candidates:
            selected_action = None
            stop_reason = "budget_exhausted"
        elif policy == "initial":
            selected_action = None
            stop_reason = "no_predicted_net_value"
        elif policy in {"random", "static", "all"}:
            selected_action = affordable_candidates[0]["group_id"]
            stop_reason = None
        elif float(affordable_candidates[0]["net_value"]) > 0:
            selected_action = affordable_candidates[0]["group_id"]
            stop_reason = None
        else:
            selected_action = None
            stop_reason = "no_predicted_net_value"
        return {
            "probabilities": current.tolist(),
            "candidates": candidates,
            "selected_action": selected_action,
            "stop_reason": stop_reason,
            "inference_ms": (perf_counter() - start) * 1000,
            "state_hash": state_hash,
            "prediction_calls": prediction_calls,
            "hypothetical_query_rows": hypothetical_query_rows,
        }

    def _panel_order(self, group_id: object) -> int:
        return next(index for index, panel in enumerate(self.panels) if panel.id == group_id)

    @staticmethod
    def _remaining_order(group_id: object, remaining: list[object]) -> int:
        return next(index for index, panel in enumerate(remaining) if panel.id == group_id)
