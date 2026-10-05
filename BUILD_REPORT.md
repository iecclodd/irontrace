# IRONTRACE implementation and verification

Completed 4 October 2026, America/Chicago. Application URL: http://127.0.0.1:8000. Launch with `./start.ps1`; exact clean setup, data preparation, model doctor, test, utility, evaluation, backend and frontend commands are in `README.md`.

This report records a historical local validation run under the original NextCheck name. The public product is now IRONTRACE, by azaan noman. Paths under `artifacts/` below are reproducible local outputs, not files included in a GitHub clone. Current publication checks are recorded in [docs/PUBLIC_RELEASE.md](docs/PUBLIC_RELEASE.md).

## Delivered and running

The local FastAPI/SQLite backend and React console are connected to **actual base TabPFN 3.5**. Inspection sessions expose only purchased readings, charge panels once, reject stale mutations, support bounded autopilot/cancellation, and reveal the label only in a terminal report. The application includes an acquisition audit, empirical ridge value model, no-residual ablation, development-selected defaults, static/random/prior/initial/all-panels references, immutable real recorded replay, and actual-result benchmark/value plots.

The checkpoint is `tabpfn-v3.5-20260909.safetensors`, SHA-256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`. It ran on `cuda:0` with four estimators, TabPFN 9.1.0, PyTorch 2.11.0+cu128, and Python 3.11.15. No fallback model or fixture powers the application. The checkpoint remains in the existing external cache and is not redistributed.

## Executed checks

| Check | Result |
|---|---|
| Official UCI download, archive validation, extraction | Passed; 2,205 cycles, 70 finite features from 14 sensors |
| Four disjoint whole-block partitions | 240 context / 80 utility / 80 selection / 120 test; 1,685 unused |
| `python scripts/doctor.py` | Passed with real held-out prediction and checkpoint hash |
| `python -m pytest backend/tests -q` | **48 passed**, including real model integration; no skipped integration |
| `npm.cmd --prefix frontend ci` | Passed; audit reported 0 vulnerabilities |
| `npm.cmd --prefix frontend run build` | TypeScript and Vite production build passed |
| Locked dependency installation dry run | All 68 packages satisfied; no changes needed |
| PowerShell launch/setup syntax; Python compileall; Git whitespace checks | Passed |
| `python scripts/verify_live.py` | **8 live acceptance groups passed**, actual model |
| Browser inspection | Start, recommend, audit, purchase, probabilities, budget, stop, completed report, autopilot, benchmark, replay passed |
| Browser unavailable/empty states | Explicit disabled-model state and genuine empty benchmark passed |
| Browser responsive check | 390 px viewport; 390 px document width; no error overlay |
| Browser console | No application errors reported |

Tests cover the float64 score identity, degenerate and coherent-but-wrong fixtures, joint context donors, observed-only invariance, split isolation, exact feature schema, no overspending, zero budget, idempotency, stale requests, terminal reports, model failures, artifact changes, interrupted reveals, invalid utility models, replay allowlisting and cache identity. Real live API checks additionally exercised these boundaries against actual model predictions. Cold model initialization and first-query timings are retained separately in `artifacts/model_status.json`; session inference times are measured warm calls.

Screenshots are from the running app: `inspection-live.png`, `audit-live.png`, `benchmark-results.png`, `recorded-replay.png`, `benchmark-empty.png`, and `mobile-check.png`. Some inspection screenshots use different illustrative development sessions; none are synthetic model outputs.

## Actual evaluation

Run: `artifacts/run_20261005T043553Z_fa25deea/`. Status **complete**, scope **smoke**, no blockers. The run uses 240 context rows, **6 utility rows, 6 policy-selection rows and 8 final-test rows**. All 6 budgets × 6 lambda values were evaluated. Three acquisition seeds were included for random. There are **468 summary records and 3,456 per-case policy outcomes**; these are repeated policies/configurations on eight cases, not 3,456 independent examples.

Four states per development row produced 84 candidate examples for fitting and 84 for regularization selection. Both ridge models selected alpha 100.0. Thirty-two fixed panel subsets were evaluated on selection data. Frozen choices were written before final-test scoring and hashed as `0bf400493a505b0ec0c0f93ea0c90ee930e4efa127a6d1592faa13abbb7357d8`. Test support was 3 no-leakage, 4 weak-leakage, and 1 severe-leakage case.

At **budget 6, lambda 0.02**, the following are measured final-test results:

| Policy | Log loss (nats) | Mean simulated cost | Accuracy |
|---|---:|---:|---:|
| Class prior | 1.0085 | 0.000 | 37.5% |
| TS1 only | 0.9059 | 0.000 | 75.0% |
| Development-selected static subset (Flow + additional temperatures) | 0.1508 | 4.000 | 100.0% |
| Random, mean across 3 seeds | 0.2482 | 5.625 | 91.7% |
| Raw KL | 0.0728 | 4.375 | 100.0% |
| Centered information | 0.0748 | 4.375 | 100.0% |
| Entropy drop | 0.1133 | 4.750 | 100.0% |
| Empirical value with residual | 0.1723 | 5.625 | 100.0% |
| Empirical value without residual | 0.1454 | 5.750 | 100.0% |
| All panels (infeasible at budget 6) | 0.0018 | 11.000 | 100.0% |

**Entropy drop was selected as the default on development data** for this scenario; the test table did not change that choice. The residual-aware value model had worse test log loss than its ablation and the static reference. This run does **not** demonstrate an advantage from the residual feature or establish the stretch savings target. Its perfect accuracies on several policies have only eight-case support.

The separate real-model shuffled-label null task completed. Its learned value policy spent **0 units** on eight cases; log loss was **1.0844**, accuracy **37.5%**, and balanced accuracy **33.3%**. That is an observed outcome of this finite run, not a statistical guarantee against needless measurements.

Artifacts include immutable split/input hashes, model provenance, utility fitting/selection examples, frozen choices, all per-case outcomes, all summary metrics, class support, query counts, timing/cache metadata, predicted-versus-realized value JSON/PNG, null results and execution status. Earlier interrupted runs are preserved with explicit reasons. Implementation corrections were driven by code review before inspecting their test outcomes; no target/features/defaults were chosen to improve final-test performance.

## Remaining scope and uncertainty

The full capped 80-utility/80-selection/120-test evaluation has **not** been executed. The delivered full-run commands implement it. The completed smoke grid establishes functioning end-to-end numerical paths, not a finished high-power scientific study. Temporal holdout, larger-draw sensitivity, optional bootstrap ensembles, cross-machine validation and commercial/hardware deployment are not completed or claimed. The one-rig dataset and pragmatic block grouping limit generalization.

No unresolved application blocker was observed in the executed checks. The final screenshots and browser checks use the built app, not a mockup. Source, package locks, commands and artifacts are preserved locally; model weights remain separately licensed.

## Engineering work and delegation

All application files were newly created in this repository. Main areas are `backend/nextcheck/{data,model,acquisition,agent,replay,evaluation}`, the FastAPI/runtime/storage/jobs modules, `backend/tests`, `frontend/src`, scripts, manifest/test vectors, launch/setup scripts and project documentation. The coordinator integrated disjoint commits and fixed the independent review findings.

Five child roles were used: data/model (GPT-6 Sol High), numerics (GPT-6 Sol High), evaluation (GPT-6 Sol High), frontend (GPT-5.6 Sol Medium), and independent reviewer (GPT-5.6 Sol High). They separated downloads/model access, numerical logic, research evaluation, UI, and independent failure analysis. **Effective routing was not exposed by runtime metadata and remains unverified.** No grandchildren were used. Detailed ownership and routing evidence is in `docs/codex-agent-setup.md`.
