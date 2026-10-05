"""Frozen offline evaluation. The action path receives no labels or hidden values."""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nextcheck.acquisition.utility import GROUPS, UtilityModel
from nextcheck.evaluation.metrics import summarize


SEED = 20261004
BUDGETS = (0, 2, 4, 6, 8, 11)
LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
THRESHOLDS = (0.0, 0.005, 0.02)
RANDOM_SEEDS = (20261004, 20261005, 20261006)
ADAPTIVE = ("random", "raw_kl", "information", "entropy_drop", "value", "value_no_residual")


def validate_splits(splits: dict[str, Any], n_rows: int) -> None:
    partitions = ("context", "utility", "selection", "test", "unused")
    seen: set[int] = set()
    block_ids = np.asarray(splits["block_ids"])
    if block_ids.shape != (n_rows,):
        raise ValueError("block_ids must cover the full extracted table")
    seen_blocks: set[Any] = set()
    for name in partitions:
        rows = [int(index) for index in splits.get(name, [])]
        if len(rows) != len(set(rows)) or any(index < 0 or index >= n_rows for index in rows):
            raise ValueError(f"invalid {name} row indices")
        if seen.intersection(rows):
            raise ValueError("original row crosses partitions")
        seen.update(rows)
        blocks = set(block_ids[rows].tolist())
        if seen_blocks.intersection(blocks):
            raise ValueError("original block crosses partitions")
        seen_blocks.update(blocks)
    if seen != set(range(n_rows)):
        raise ValueError("split partitions do not account for every row")
    for name in ("context", "utility", "selection", "test"):
        if not splits.get(name):
            raise ValueError(f"empty {name} partition")


def panel_columns(manifest: dict[str, Any], feature_names: list[str]) -> dict[str, list[int]]:
    positions = {name: i for i, name in enumerate(feature_names)}
    if len(positions) != len(feature_names):
        raise ValueError("duplicate feature names")
    panels: dict[str, list[int]] = {}
    assigned: set[int] = set()
    for panel in manifest["panels"]:
        group = panel["id"]
        columns = [positions[name] for name in panel["features"]]
        if assigned.intersection(columns):
            raise ValueError("feature assigned to multiple panels")
        assigned.update(columns)
        panels[group] = columns
    if set(panels) != {"initial", *GROUPS} or assigned != set(range(len(feature_names))):
        raise ValueError("manifest does not cover exactly the expected panels and features")
    return panels


def panel_costs(manifest: dict[str, Any]) -> dict[str, float]:
    costs = {panel["id"]: float(panel["cost"]) for panel in manifest["panels"]}
    if costs.get("initial") != 0 or set(costs) != {"initial", *GROUPS} or any(not math.isfinite(c) or c < 0 for c in costs.values()):
        raise ValueError("invalid panel costs")
    return costs


def mask_row(row: np.ndarray, groups: tuple[str, ...], columns: dict[str, list[int]]) -> np.ndarray:
    masked = np.full_like(row, np.nan, dtype=float)
    for group in ("initial", *groups):
        masked[columns[group]] = row[columns[group]]
    return masked


def observed_case(row: np.ndarray, groups: tuple[str, ...], columns: dict[str, list[int]], names: list[str]):
    from nextcheck.agent.state import ObservedCase

    indices = [index for group in ("initial", *groups) for index in columns[group]]
    values = {names[index]: float(row[index]) for index in indices}
    return ObservedCase(observed_groups=("initial", *groups), values=values)


def state_seed(row_index: int, groups: tuple[str, ...], seed: int = SEED) -> int:
    digest = hashlib.sha256(f"{seed}:{row_index}:{','.join(groups)}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass
class CountingPredictor:
    predictor: Any
    prediction_calls: int = 0
    query_rows: int = 0
    hypothetical_rows: int = 0

    @property
    def provenance(self):
        return self.predictor.provenance

    def predict_proba(self, x):
        array = np.asarray(x)
        if array.ndim != 2:
            raise ValueError("prediction requires a 2D matrix")
        self.prediction_calls += 1
        self.query_rows += len(array)
        # In the engine, one current state is a singleton; hypothetical draws
        # are batched. The latter are counted separately for cost transparency.
        if len(array) > 1:
            self.hypothetical_rows += len(array)
        return self.predictor.predict_proba(array)

    def counts(self) -> tuple[int, int, int]:
        return self.prediction_calls, self.query_rows, self.hypothetical_rows


@dataclass
class DiagnosticCache:
    engine: Any
    x: np.ndarray
    columns: dict[str, list[int]]
    names: list[str]
    values: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = field(default_factory=dict)

    def get(self, index: int, groups: tuple[str, ...]) -> dict[str, Any]:
        groups = tuple(group for group in GROUPS if group in groups)
        key = (int(index), groups)
        if key in self.values:
            return {**self.values[key], "cache_reused": True}
        else:
            case = observed_case(self.x[index], groups, self.columns, self.names)
            counter = self.engine.predictor
            before = counter.counts()
            start = time.perf_counter()
            analysis = self.engine.analyze(case, remaining_budget=11, policy="information", seed=state_seed(index, groups))
            after = counter.counts()
            self.values[key] = {
                "analysis": analysis,
                "counts": tuple(a - b for a, b in zip(after, before)),
                "elapsed_s": time.perf_counter() - start,
            }
        return {**self.values[key], "cache_reused": False}


def partial_states(row_index: int, seed: int = SEED) -> tuple[tuple[str, ...], ...]:
    rng = np.random.default_rng(seed + int(row_index))
    order = tuple(str(group) for group in rng.permutation(GROUPS))
    return ((), order[:1], order[:2], order[:3])


def generate_utility_examples(
    x: np.ndarray, y: np.ndarray, rows: list[int], engine: Any, predictor: Any,
    manifest: dict[str, Any], names: list[str], *, seed: int = SEED,
) -> list[dict[str, Any]]:
    """Four states per original row; labels enter only after diagnostics are made."""
    columns = panel_columns(manifest, names)
    costs = panel_costs(manifest)
    examples: list[dict[str, Any]] = []
    for index in rows:
        index = int(index)
        for groups in partial_states(index, seed):
            visible = observed_case(x[index], groups, columns, names)
            spent = sum(costs[g] for g in groups)
            remaining = 11 - spent
            analysis = engine.analyze(visible, remaining_budget=remaining, policy="information", seed=state_seed(index, groups, seed))
            q0 = np.asarray(analysis["probabilities"], dtype=float)
            candidates = [candidate for candidate in analysis["candidates"] if candidate["group_id"] not in groups and costs[candidate["group_id"]] <= remaining]
            if not candidates:
                continue
            counterfactuals = np.vstack([mask_row(x[index], (*groups, candidate["group_id"]), columns) for candidate in candidates])
            after = np.asarray(predictor.predict_proba(counterfactuals), dtype=float)
            if after.shape != (len(candidates), 3):
                raise ValueError("counterfactual prediction shape mismatch")
            label = int(y[index])
            before_loss = -math.log(max(float(q0[label]), 1e-15))
            for candidate, q_after in zip(candidates, after):
                after_loss = -math.log(max(float(q_after[label]), 1e-15))
                examples.append({
                    "row_index": index, "observed_groups": ["initial", *groups],
                    "budget": 11, "remaining_budget": remaining,
                    "candidate": candidate, "probabilities": q0.tolist(),
                    "delta": before_loss - after_loss,
                    "loss_before": before_loss, "loss_after": after_loss,
                })
    return examples


def select_utility_models(
    fitting: list[dict[str, Any]], selection: list[dict[str, Any]],
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[dict[str, UtilityModel], dict[str, Any]]:
    if set(e["row_index"] for e in fitting).intersection(e["row_index"] for e in selection):
        raise ValueError("utility and selection rows overlap")
    models = {}
    record = {}
    actual = np.asarray([e["delta"] for e in selection], dtype=float)
    for name, residual in (("value", True), ("value_no_residual", False)):
        candidates = [(float(np.mean((model.predict_examples(selection) - actual) ** 2)), alpha, model)
                      for alpha in alphas for model in [UtilityModel.fit(fitting, alpha=alpha, include_residual=residual)]]
        mse, alpha, model = min(candidates, key=lambda item: (item[0], item[1]))
        models[name] = model
        record[name] = {"alpha": alpha, "selection_mse": mse, "selection_examples": len(selection), "fitting_examples": len(fitting)}
    return models, record


def all_subsets() -> tuple[tuple[str, ...], ...]:
    return tuple(subset for count in range(len(GROUPS) + 1) for subset in itertools.combinations(GROUPS, count))


def _single_prediction(predictor: CountingPredictor, masked: np.ndarray) -> tuple[list[float], float]:
    start = time.perf_counter()
    probabilities = np.asarray(predictor.predict_proba(masked[None, :])[0], dtype=float)
    if probabilities.shape != (3,) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("prediction backend returned invalid probabilities")
    return probabilities.tolist(), time.perf_counter() - start


def _choose(
    policy: str, analysis: dict[str, Any], groups: tuple[str, ...], remaining: float,
    costs: dict[str, float], lambda_cost: float, threshold: float,
    models: dict[str, UtilityModel], rng: np.random.Generator,
) -> str | None:
    feasible = [candidate for candidate in analysis["candidates"] if candidate["group_id"] not in groups and costs[candidate["group_id"]] <= remaining]
    if not feasible:
        return None
    if policy == "random":
        return str(feasible[int(rng.integers(len(feasible)))]["group_id"])
    if policy in ("value", "value_no_residual"):
        model = models[policy]
        scored = [(model.predict(candidate, ("initial", *groups), analysis["probabilities"]) - lambda_cost * costs[candidate["group_id"]], candidate["group_id"])
                  for candidate in feasible]
        best = max(scored, key=lambda pair: (pair[0], pair[1]))
        return str(best[1]) if best[0] > 0 else None
    if policy not in ("raw_kl", "information", "entropy_drop"):
        raise ValueError(f"unsupported adaptive policy {policy}")
    scored = [(float(candidate[policy]) / max(costs[candidate["group_id"]], 1e-12), candidate["group_id"]) for candidate in feasible]
    best = max(scored, key=lambda pair: (pair[0], pair[1]))
    return str(best[1]) if best[0] > threshold else None


def run_actions(
    *, index: int, row: np.ndarray, policy: str, budget: float,
    lambda_cost: float, threshold: float, random_seed: int,
    predictor: CountingPredictor, cache: DiagnosticCache,
    columns: dict[str, list[int]], costs: dict[str, float],
    models: dict[str, UtilityModel], prior: np.ndarray,
    static_subset: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Choose actions without a label argument. Hidden row is revealed only after choice."""
    groups: tuple[str, ...] = ()
    elapsed = 0.0
    calls = query_rows = hypothetical_rows = 0
    execution_elapsed = 0.0
    physical_calls = physical_query_rows = physical_hypothetical_rows = 0
    cache_reused = False
    rng = np.random.default_rng(random_seed + index)
    trace: list[dict[str, Any]] = []
    stop_reason: str
    if policy == "prior":
        probabilities = prior.tolist()
        stop_reason = "no_measurement"
    elif policy in ("initial", "static", "all"):
        groups = static_subset if policy == "static" else (GROUPS if policy == "all" else ())
        probabilities, elapsed = _single_prediction(predictor, mask_row(row, groups, columns))
        calls = query_rows = 1
        physical_calls = physical_query_rows = 1
        execution_elapsed = elapsed
        stop_reason = {"initial": "no_measurement", "static": "fixed_subset_complete", "all": "all_panels_reference"}[policy]
    elif policy == "random":
        # Random needs only the public costs and mask. It must not run the
        # diagnostic sampler or pay for hypothetical model queries.
        for _ in range(len(GROUPS)):
            remaining = budget - sum(costs[group] for group in groups)
            feasible = [group for group in GROUPS if group not in groups and costs[group] <= remaining]
            if not feasible:
                break
            action = str(feasible[int(rng.integers(len(feasible)))])
            groups = tuple(group for group in GROUPS if group in (*groups, action))
            trace.append({"group_id": action, "cost": costs[action], "remaining_budget_after": budget - sum(costs[group] for group in groups)})
        probabilities, elapsed = _single_prediction(predictor, mask_row(row, groups, columns))
        calls = query_rows = 1
        physical_calls = physical_query_rows = 1
        execution_elapsed = elapsed
        stop_reason = "no_groups_remaining" if len(groups) == len(GROUPS) else "budget_exhausted"
    else:
        probabilities = []
        stop_reason = "no_predicted_net_value"
        for _ in range(len(GROUPS) + 1):
            remaining = budget - sum(costs[group] for group in groups)
            feasible = [group for group in GROUPS if group not in groups and costs[group] <= remaining]
            if not feasible:
                stop_reason = "no_groups_remaining" if len(groups) == len(GROUPS) else "budget_exhausted"
                probabilities, seconds = _single_prediction(predictor, mask_row(row, groups, columns))
                elapsed += seconds
                calls += 1
                query_rows += 1
                execution_elapsed += seconds
                physical_calls += 1
                physical_query_rows += 1
                break
            cached = cache.get(index, groups)
            analysis = cached["analysis"]
            probabilities = list(analysis["probabilities"])
            elapsed += cached["elapsed_s"]
            calls += cached["counts"][0]
            query_rows += cached["counts"][1]
            hypothetical_rows += cached["counts"][2]
            if cached["cache_reused"]:
                cache_reused = True
            else:
                execution_elapsed += cached["elapsed_s"]
                physical_calls += cached["counts"][0]
                physical_query_rows += cached["counts"][1]
                physical_hypothetical_rows += cached["counts"][2]
            action = _choose(policy, analysis, groups, remaining, costs, lambda_cost, threshold, models, rng)
            if action is None:
                break
            # This is the only reveal. No label or hidden value enters _choose.
            groups = tuple(group for group in GROUPS if group in (*groups, action))
            trace.append({"group_id": action, "cost": costs[action], "remaining_budget_after": budget - sum(costs[group] for group in groups)})
        else:
            raise RuntimeError("acquisition exceeded five purchases")
    spent = sum(costs[group] for group in groups)
    if policy != "all" and spent > budget + 1e-9:
        raise RuntimeError("policy overspent")
    return {
        "row_index": index, "policy": policy, "budget": budget,
        "lambda_cost": lambda_cost, "observed_groups": ["initial", *groups],
        "probabilities": probabilities, "cost": spent, "purchases": len(groups),
        "stop_reason": stop_reason,
        "prediction_calls": calls, "prediction_query_rows": query_rows,
        "hypothetical_query_rows": hypothetical_rows,
        "physical_prediction_calls": physical_calls,
        "physical_prediction_query_rows": physical_query_rows,
        "physical_hypothetical_query_rows": physical_hypothetical_rows,
        "wall_time_s": elapsed, "execution_wall_time_s": execution_elapsed,
        "wall_time_basis": "measured_compute_reference" if cache_reused else "measured_this_execution",
        "cache_reused": cache_reused, "trace": trace,
    }


def score_outcomes(outcomes: list[dict[str, Any]], labels: np.ndarray) -> list[dict[str, Any]]:
    """Access labels only after run_actions has returned every decision."""
    scored = []
    for outcome in outcomes:
        index = int(outcome["row_index"])
        scored.append({**outcome, "true_label": int(labels[index])})
    return scored


def evaluate_configuration(
    *, rows: list[int], x: np.ndarray, y: np.ndarray, policy: str,
    budget: float, lambda_cost: float, threshold: float,
    random_seed: int, predictor: CountingPredictor, cache: DiagnosticCache,
    columns: dict[str, list[int]], costs: dict[str, float],
    models: dict[str, UtilityModel], prior: np.ndarray,
    static_subset: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outcomes = [run_actions(
        index=int(index), row=x[int(index)], policy=policy, budget=budget,
        lambda_cost=lambda_cost, threshold=threshold, random_seed=random_seed,
        predictor=predictor, cache=cache, columns=columns, costs=costs,
        models=models, prior=prior, static_subset=static_subset,
    ) for index in rows]
    cases = score_outcomes(outcomes, y)
    return cases, summarize(cases)
