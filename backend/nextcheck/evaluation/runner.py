"""Artifact-producing utility preparation and frozen benchmark orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nextcheck.acquisition.utility import GROUPS, UtilityModel
from nextcheck.evaluation.benchmark import (
    ADAPTIVE, ALPHAS, BUDGETS, LAMBDAS, RANDOM_SEEDS, SEED, THRESHOLDS,
    CountingPredictor, DiagnosticCache, all_subsets, evaluate_configuration,
    generate_utility_examples, panel_columns, panel_costs, select_utility_models,
    validate_splits,
)
from nextcheck.evaluation.provenance import environment_info, sha256, write_json
from nextcheck.evaluation.metrics import summarize


def artifact_root(repository: Path) -> Path:
    return Path(os.environ.get("NEXTCHECK_ARTIFACTS", repository / "artifacts"))


def load_prepared(repository: Path, data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any], dict[str, Any], Path]:
    data_dir = data_dir or artifact_root(repository) / "data"
    features_path = data_dir / "features.npz"
    splits_path = data_dir / "splits.json"
    manifest_path = repository / "demo_manifest.json"
    with np.load(features_path, allow_pickle=False) as package:
        x = np.asarray(package["X"], dtype=float)
        y = np.asarray(package["y"], dtype=int)
        names = [str(name) for name in package["feature_names"].tolist()]
    if x.ndim != 2 or x.shape[0] != len(y) or x.shape[1] != len(names) or not np.all(np.isfinite(x)):
        raise ValueError("invalid prepared feature table")
    if not set(np.unique(y)).issubset({0, 1, 2}):
        raise ValueError("unexpected labels")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    validate_splits(splits, len(y))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    panel_columns(manifest, names)
    panel_costs(manifest)
    return x, y, names, splits, manifest, data_dir


def build_predictor(x: np.ndarray, y: np.ndarray, context: list[int]):
    from nextcheck.model.tabpfn35 import TabPFN35Predictor

    indices = np.asarray(context, dtype=int)
    if len(indices) == 0 or len(set(y[indices].tolist())) < 3:
        raise ValueError("context partition lacks three-class coverage")
    return CountingPredictor(TabPFN35Predictor(x[indices], y[indices]))


def context_hash(x: np.ndarray, y: np.ndarray, context: list[int]) -> str:
    rows = np.asarray(context, dtype=int)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x[rows], dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(y[rows], dtype="<i8").tobytes())
    return digest.hexdigest()


def build_engine(predictor: CountingPredictor, x: np.ndarray, context: list[int], manifest: dict[str, Any]):
    from nextcheck.agent.loop import AcquisitionEngine

    return AcquisitionEngine(predictor, x[np.asarray(context, dtype=int)], manifest)


def capped_rows(splits: dict[str, Any], partition: str, limit: int | None) -> list[int]:
    rows = [int(row) for row in splits[partition]]
    if limit is not None:
        if limit <= 0:
            raise ValueError("row limits must be positive")
        if limit < len(rows):
            # Predeclared positions spread smoke cases across the fixed partition.
            # This selection sees neither features nor test labels.
            positions = np.linspace(0, len(rows) - 1, limit, dtype=int)
            rows = [rows[int(position)] for position in positions]
    return rows


def prepare_utility(
    repository: Path, *, data_dir: Path | None = None, output_dir: Path | None = None,
    context_limit: int | None = None, utility_limit: int | None = None,
    selection_limit: int | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or artifact_root(repository) / "utility"
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "execution_status.json"
    write_json(status_path, {"status": "running", "stage": "loading_prepared_data"})
    try:
        x, y, names, splits, manifest, prepared_dir = load_prepared(repository, data_dir)
        context = capped_rows(splits, "context", context_limit)
        utility = capped_rows(splits, "utility", utility_limit)
        selection = capped_rows(splits, "selection", selection_limit)
        if not set(context).isdisjoint(utility) or not set(utility).isdisjoint(selection):
            raise ValueError("development partitions overlap")
        predictor = build_predictor(x, y, context)
        engine = build_engine(predictor, x, context, manifest)
        write_json(status_path, {"status": "running", "stage": "generating_utility_examples"})
        fitting = generate_utility_examples(x, y, utility, engine, predictor, manifest, names)
        write_json(status_path, {"status": "running", "stage": "generating_selection_examples", "fitting_examples": len(fitting)})
        selected = generate_utility_examples(x, y, selection, engine, predictor, manifest, names)
        models, tuning = select_utility_models(fitting, selected)
        hashes = {"features_sha256": sha256(prepared_dir / "features.npz"),
                  "splits_sha256": sha256(prepared_dir / "splits.json"),
                  "manifest_sha256": sha256(repository / "demo_manifest.json")}
        identity = {"context_hash": context_hash(x, y, context), "context_row_indices": context,
                    "model_provenance": predictor.provenance, "input_hashes": hashes}
        for name, model in models.items():
            model.training_metadata = identity
            model.save(output_dir / f"{name}.json")
        write_json(output_dir / "fitting_examples.json", fitting)
        write_json(output_dir / "selection_examples.json", selected)
        metadata = {
            "status": "complete", "scope": "full" if all(limit is None for limit in (context_limit, utility_limit, selection_limit)) else "smoke",
            "seed": SEED, "counts": {"context": len(context), "utility": len(utility), "selection": len(selection), "test": len(splits["test"]), "unused": len(splits["unused"])},
            "example_counts": {"fitting": len(fitting), "selection": len(selected)},
            "states_per_row": 4, "alphas": list(ALPHAS), "tuning": tuning,
            "fitting_row_indices": utility, "selection_row_indices": selection,
            "context_row_indices": context,
            "model_provenance": predictor.provenance,
            "context_hash": identity["context_hash"],
            "inputs": hashes,
            "environment": environment_info(),
        }
        write_json(output_dir / "metadata.json", metadata)
        write_json(status_path, {"status": "complete", "stage": "utility_models_saved", "model_provenance": predictor.provenance})
        return metadata
    except Exception as exc:
        write_json(status_path, {"status": "blocked", "stage": "utility_preparation", "error_type": type(exc).__name__, "blockers": [str(exc)]})
        raise


def _prediction_metrics(
    *, rows: list[int], x: np.ndarray, y: np.ndarray, policy: str,
    subset: tuple[str, ...], predictor: CountingPredictor,
    columns: dict[str, list[int]], costs: dict[str, float], prior: np.ndarray,
) -> dict[str, Any]:
    from nextcheck.evaluation.benchmark import mask_row

    if policy == "prior":
        probabilities = np.tile(prior, (len(rows), 1))
    else:
        masked = np.vstack([mask_row(x[index], subset, columns) for index in rows])
        probabilities = predictor.predict_proba(masked)
    labels = y[np.asarray(rows, dtype=int)]
    loss = -np.log(np.clip(probabilities[np.arange(len(rows)), labels], 1e-15, 1))
    return {"mean_log_loss": float(np.mean(loss)), "cost": float(sum(costs[g] for g in subset))}


def select_frozen_policies(
    *, rows: list[int], x: np.ndarray, y_dev: np.ndarray,
    predictor: CountingPredictor, engine: Any, manifest: dict[str, Any], names: list[str],
    models: dict[str, UtilityModel], prior: np.ndarray,
    budgets: tuple[float, ...], lambdas: tuple[float, ...],
    thresholds: tuple[float, ...] = THRESHOLDS, random_seeds: tuple[int, ...] = RANDOM_SEEDS,
) -> dict[str, Any]:
    columns = panel_columns(manifest, names)
    costs = panel_costs(manifest)
    cache = DiagnosticCache(engine, x, columns, names)
    fixed = {subset: _prediction_metrics(rows=rows, x=x, y=y_dev, policy="static", subset=subset, predictor=predictor, columns=columns, costs=costs, prior=prior)
             for subset in all_subsets()}
    prior_result = _prediction_metrics(rows=rows, x=x, y=y_dev, policy="prior", subset=(), predictor=predictor, columns=columns, costs=costs, prior=prior)
    choices: dict[str, Any] = {}
    static_candidates: list[dict[str, Any]] = []
    for budget in budgets:
        for lambda_cost in lambdas:
            key = f"budget={budget:g};lambda={lambda_cost:g}"
            best_subset = min((subset for subset in all_subsets() if fixed[subset]["cost"] <= budget),
                              key=lambda subset: (fixed[subset]["mean_log_loss"] + lambda_cost * fixed[subset]["cost"], fixed[subset]["cost"], subset))
            static_candidates.extend({"budget": budget, "lambda_cost": lambda_cost, "groups": list(subset), **fixed[subset], "feasible": fixed[subset]["cost"] <= budget}
                                     for subset in all_subsets())
            configurations: dict[str, Any] = {
                "prior": {"objective": prior_result["mean_log_loss"], "configuration": {}},
                "initial": {"objective": fixed[()]["mean_log_loss"], "configuration": {}},
                "static": {"objective": fixed[best_subset]["mean_log_loss"] + lambda_cost * fixed[best_subset]["cost"], "configuration": {"groups": list(best_subset)}},
            }
            if budget >= sum(costs[g] for g in GROUPS):
                configurations["all"] = {"objective": fixed[GROUPS]["mean_log_loss"] + lambda_cost * fixed[GROUPS]["cost"], "configuration": {}}
            for policy in ADAPTIVE:
                candidates = []
                thresholds_to_try = thresholds if policy in ("raw_kl", "information", "entropy_drop") else (0.0,)
                seeds_to_try = random_seeds if policy == "random" else (random_seeds[0],)
                for threshold in thresholds_to_try:
                    objectives = []
                    for random_seed in seeds_to_try:
                        _, metrics = evaluate_configuration(rows=rows, x=x, y=y_dev, policy=policy, budget=budget,
                            lambda_cost=lambda_cost, threshold=threshold, random_seed=random_seed,
                            predictor=predictor, cache=cache, columns=columns, costs=costs,
                            models=models, prior=prior)
                        objectives.append(metrics["log_loss"] + lambda_cost * metrics["mean_cost"])
                    candidates.append((float(np.mean(objectives)), threshold, objectives))
                objective, threshold, seed_objectives = min(candidates, key=lambda item: (item[0], item[1]))
                configurations[policy] = {"objective": objective, "configuration": {"threshold": threshold, "random_seeds": list(random_seeds) if policy == "random" else None}, "seed_objectives": seed_objectives}
            default = min(configurations, key=lambda policy: (configurations[policy]["objective"], policy))
            choices[key] = {"budget": budget, "lambda_cost": lambda_cost, "default_policy": default, "policies": configurations}
    return {
        "schema_version": 1, "selection_rows": rows, "budgets": list(budgets), "lambda_grid": list(lambdas),
        "threshold_grid": list(thresholds), "random_seeds": list(random_seeds),
        "static_subsets_evaluated": 32, "static_candidates": static_candidates,
        "choices": choices, "selection_diagnostic_states": len(cache.values),
    }


def _null_labels(y: np.ndarray, splits: dict[str, Any]) -> np.ndarray:
    shuffled = y.copy()
    rng = np.random.default_rng(SEED + 999)
    for partition in ("context", "utility", "selection", "test"):
        rows = np.asarray(splits[partition], dtype=int)
        shuffled[rows] = rng.permutation(shuffled[rows])
    return shuffled


def _plot_value_scatter(points: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not points:
        return
    actual = np.asarray([point["actual_delta"] for point in points], dtype=float)
    predicted = (
        np.asarray([point["predicted_delta"] for point in points], dtype=float),
        np.asarray([point["predicted_without_residual"] for point in points], dtype=float),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    bounds = np.concatenate((actual, *predicted))
    lower, upper = np.quantile(bounds, [0.01, 0.99])
    if lower == upper:
        lower, upper = lower - 1, upper + 1
    for axis, values, title in zip(axes, predicted, ("Value model", "No-residual ablation")):
        axis.scatter(values, actual, s=12, alpha=0.45)
        axis.plot([lower, upper], [lower, upper], color="gray", linewidth=1)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_title(title)
        axis.set_xlabel("Predicted one-step loss improvement (nats)")
    axes[0].set_ylabel("Realized one-step loss improvement (nats)")
    figure.suptitle(f"Frozen test counterfactuals · {len(points)} state-action examples")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_null_task(
    *, x: np.ndarray, y: np.ndarray, splits: dict[str, Any], names: list[str],
    manifest: dict[str, Any], context: list[int], utility: list[int],
    selection: list[int], test: list[int], budget: float, lambda_cost: float,
) -> dict[str, Any]:
    null_y = _null_labels(y, splits)
    predictor = build_predictor(x, null_y, context)
    engine = build_engine(predictor, x, context, manifest)
    fitting = generate_utility_examples(x, null_y, utility, engine, predictor, manifest, names, seed=SEED + 999)
    selected = generate_utility_examples(x, null_y, selection, engine, predictor, manifest, names, seed=SEED + 999)
    models, tuning = select_utility_models(fitting, selected)
    columns = panel_columns(manifest, names)
    costs = panel_costs(manifest)
    prior = np.bincount(null_y[np.asarray(context, dtype=int)], minlength=3).astype(float)
    prior /= prior.sum()
    cache = DiagnosticCache(engine, x, columns, names)
    # Null labels are read only after the action sequence is fixed.
    cases, metrics = evaluate_configuration(rows=test, x=x, y=null_y, policy="value", budget=budget,
        lambda_cost=lambda_cost, threshold=0.0, random_seed=SEED + 999,
        predictor=predictor, cache=cache, columns=columns, costs=costs,
        models=models, prior=prior)
    return {
        "status": "complete", "seed": SEED + 999,
        "method": "independent seeded permutation of labels within each predeclared partition; fit a separate real predictor and utility model",
        "budget": budget, "lambda_cost": lambda_cost, "policy": "value",
        "model_provenance": predictor.provenance, "utility_tuning": tuning,
        "metrics": metrics, "cases": cases,
    }


def evaluate_frozen(
    repository: Path, *, data_dir: Path | None = None, utility_dir: Path | None = None,
    output_dir: Path | None = None, context_limit: int | None = None,
    utility_limit: int | None = None, selection_limit: int | None = None,
    test_limit: int | None = None, budgets: tuple[float, ...] = BUDGETS,
    lambdas: tuple[float, ...] = LAMBDAS, include_null: bool = True,
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = output_dir or artifact_root(repository) / f"run_{run_id}"
    utility_dir = utility_dir or artifact_root(repository) / "utility"
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "execution_status.json"
    write_json(status_path, {"status": "running", "stage": "loading_prepared_data", "run_id": run_id})
    scope = "full" if include_null and all(limit is None for limit in (context_limit, utility_limit, selection_limit, test_limit)) and tuple(budgets) == BUDGETS and tuple(lambdas) == LAMBDAS else "smoke"
    try:
        x, y, names, splits, manifest, prepared_dir = load_prepared(repository, data_dir)
        context = capped_rows(splits, "context", context_limit)
        utility = capped_rows(splits, "utility", utility_limit)
        selection = capped_rows(splits, "selection", selection_limit)
        test = capped_rows(splits, "test", test_limit)
        metadata = json.loads((utility_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "complete" or metadata["context_row_indices"] != context or metadata["fitting_row_indices"] != utility or metadata["selection_row_indices"] != selection:
            raise ValueError("utility model was fit on different development rows; rerun prepare_utility with matching limits")
        if metadata["inputs"] != {
            "features_sha256": sha256(prepared_dir / "features.npz"),
            "splits_sha256": sha256(prepared_dir / "splits.json"),
            "manifest_sha256": sha256(repository / "demo_manifest.json"),
        }:
            raise ValueError("prepared data changed since utility fit")
        predictor = build_predictor(x, y, context)
        engine = build_engine(predictor, x, context, manifest)
        models = {name: UtilityModel.load(utility_dir / f"{name}.json") for name in ("value", "value_no_residual")}
        if any(set(model.fitting_rows) != set(utility) for model in models.values()):
            raise ValueError("utility model fitting rows mismatch")
        expected_context_hash = context_hash(x, y, context)
        if metadata.get("context_hash") != expected_context_hash or any(model.training_metadata.get("context_hash") != expected_context_hash for model in models.values()):
            raise ValueError("utility model context hash mismatch")
        prior = np.bincount(y[np.asarray(context, dtype=int)], minlength=3).astype(float)
        prior /= prior.sum()
        # Make misuse of test labels during selection a hard failure.
        y_dev = y.copy()
        y_dev[np.asarray(splits["test"], dtype=int)] = -1
        write_json(status_path, {"status": "running", "stage": "policy_selection", "run_id": run_id})
        choices = select_frozen_policies(rows=selection, x=x, y_dev=y_dev, predictor=predictor,
            engine=engine, manifest=manifest, names=names, models=models, prior=prior,
            budgets=budgets, lambdas=lambdas)
        choices["input_hashes"] = metadata["inputs"]
        choices["model_provenance"] = predictor.provenance
        choices["utility_model_hashes"] = {name: sha256(utility_dir / f"{name}.json") for name in models}
        choices["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        frozen_path = output_dir / "frozen_choices.json"
        write_json(frozen_path, choices)
        frozen_hash = sha256(frozen_path)
        # The first access to final-test labels is below, after frozen_choices.json exists.
        write_json(status_path, {"status": "running", "stage": "frozen_test_evaluation", "run_id": run_id, "frozen_choices_sha256": frozen_hash})
        columns = panel_columns(manifest, names)
        costs = panel_costs(manifest)
        cache = DiagnosticCache(engine, x, columns, names)
        outcomes: list[dict[str, Any]] = []
        result_summaries: list[dict[str, Any]] = []
        fixed_test_cache: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
        for key, choice in choices["choices"].items():
            budget = float(choice["budget"])
            lambda_cost = float(choice["lambda_cost"])
            for policy in ("prior", "initial", "static", *ADAPTIVE, "all"):
                configuration = choice["policies"].get(policy, {}).get("configuration", {})
                threshold = float(configuration.get("threshold", 0.0))
                seeds = RANDOM_SEEDS if policy == "random" else (RANDOM_SEEDS[0],)
                per_seed = []
                for random_seed in seeds:
                    subset = tuple(configuration.get("groups", ()))
                    fixed_key = (policy, subset)
                    if policy in ("prior", "initial", "static", "all") and fixed_key in fixed_test_cache:
                        cases = [{**case, "budget": budget, "lambda_cost": lambda_cost}
                                 for case in fixed_test_cache[fixed_key]]
                        metrics = summarize(cases)
                    else:
                        cases, metrics = evaluate_configuration(rows=test, x=x, y=y, policy=policy,
                            budget=budget, lambda_cost=lambda_cost, threshold=threshold,
                            random_seed=random_seed, predictor=predictor, cache=cache,
                            columns=columns, costs=costs, models=models, prior=prior,
                            static_subset=subset)
                        if policy in ("prior", "initial", "static", "all"):
                            fixed_test_cache[fixed_key] = cases
                    for case in cases:
                        outcomes.append({**case, "random_seed": random_seed if policy == "random" else None,
                                         "frozen_choices_sha256": frozen_hash})
                    per_seed.append(metrics)
                    result_summaries.append({"policy": policy, "budget": budget, "lambda_cost": lambda_cost,
                        "random_seed": random_seed if policy == "random" else None,
                        "feasible": policy != "all" or budget >= sum(costs[g] for g in GROUPS),
                        "configuration": configuration, "metrics": metrics})
                if policy == "random":
                    numeric = ("log_loss", "accuracy", "balanced_accuracy", "brier", "mean_cost", "mean_purchases", "mean_wall_time_s")
                    result_summaries.append({"policy": "random_mean", "budget": budget, "lambda_cost": lambda_cost,
                        "random_seeds": list(seeds), "feasible": True,
                        "metrics": {name: float(np.mean([item[name] for item in per_seed])) for name in numeric},
                        "n_cases_per_seed": len(test)})
        write_json(output_dir / "per_case.json", outcomes)
        write_json(output_dir / "aggregate.json", result_summaries)
        # Test value diagnostics are generated only after frozen decisions.
        test_examples = generate_utility_examples(x, y, test, engine, predictor, manifest, names)
        value_scatter = [{"row_index": example["row_index"], "observed_groups": example["observed_groups"],
                          "group_id": example["candidate"]["group_id"], "actual_delta": example["delta"],
                          "predicted_delta": models["value"].predict(example["candidate"], example["observed_groups"], example["probabilities"]),
                          "predicted_without_residual": models["value_no_residual"].predict(example["candidate"], example["observed_groups"], example["probabilities"])}
                         for example in test_examples]
        write_json(output_dir / "value_scatter.json", value_scatter)
        _plot_value_scatter(value_scatter, output_dir / "value_scatter.png")
        null_result = None
        blockers: list[str] = []
        if include_null:
            write_json(status_path, {"status": "running", "stage": "null_task", "run_id": run_id, "frozen_choices_sha256": frozen_hash})
            try:
                null_result = evaluate_null_task(x=x, y=y, splits=splits, names=names, manifest=manifest,
                    context=context, utility=utility, selection=selection, test=test,
                    budget=6 if 6 in budgets else max(budgets), lambda_cost=0.02)
            except Exception as exc:
                blockers = [f"Null task failed: {type(exc).__name__}: {exc}"]
                null_result = {"status": "blocked", "blockers": blockers}
            write_json(output_dir / "null_task.json", null_result)
        summary = {
            "run_id": run_id, "status": "partial" if blockers else "complete", "scope": scope,
            "budgets": list(budgets), "lambda_grid": list(lambdas),
            "model_provenance": predictor.provenance,
            "split_counts": {name: len(splits[name]) for name in ("context", "utility", "selection", "test", "unused")},
            "evaluated_counts": {"context": len(context), "utility": len(utility), "selection": len(selection), "test": len(test)},
            "frozen_choices_sha256": frozen_hash,
            "selected_defaults": {key: choice["default_policy"] for key, choice in choices["choices"].items()},
            "results": result_summaries,
            "null_task": None if null_result is None else {"status": null_result["status"], "metrics": null_result.get("metrics"), "blockers": null_result.get("blockers", [])},
            "files": {"frozen_choices": "frozen_choices.json", "per_case": "per_case.json", "aggregate": "aggregate.json", "value_scatter": "value_scatter.json", "value_scatter_plot": "value_scatter.png", "null_task": "null_task.json" if include_null else None},
            "blockers": blockers,
        }
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "environment.json", environment_info())
        write_json(output_dir / "provenance.json", {"inputs": metadata["inputs"], "model": predictor.provenance,
            "utility_training": metadata["tuning"], "frozen_choices_sha256": frozen_hash})
        write_json(status_path, {"status": summary["status"], "run_id": run_id, "scope": scope,
            "blockers": blockers, "frozen_choices_sha256": frozen_hash,
            "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return summary
    except Exception as exc:
        write_json(status_path, {"status": "blocked", "run_id": run_id,
            "error_type": type(exc).__name__, "blockers": [str(exc)]})
        write_json(output_dir / "summary.json", {"run_id": run_id, "status": "blocked", "scope": scope,
            "results": [], "blockers": [str(exc)]})
        raise
