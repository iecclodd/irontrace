# NextCheck

**Know what to measure next.** A local hydraulic inspection replay application with real TabPFN 3.5, five purchasable sensor panels, a transparent acquisition audit, and a development-fitted empirical value model with a no-residual ablation.

The target is internal pump leakage (no/weak/severe), not overall machine health. Historical recordings simulate buying access to hidden measurements. Costs are editable scenario units. This application does not operate machinery or provide repair or safety decisions.

## Run the delivered application

In PowerShell, from this directory:

```powershell
.\start.ps1
```

Open http://127.0.0.1:8000. Startup loads one checkpoint into one model worker. The built React app is served by FastAPI. `start.ps1` uses this delivered workspace's tested Python environment when a project `.venv` does not yet exist. Stop the server with Ctrl+C. Do not launch several backend workers.

## Reproduce setup on Windows

Requires `uv`, Node.js/npm, Python 3.11 provisioned by uv, and authorized access to the official model checkpoint. The verified machine has an RTX 4060 Ti with 8 GB; the lock includes CUDA 12.8 PyTorch wheels.

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe scripts\prepare_data.py
.\.venv\Scripts\python.exe scripts\doctor.py
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

`backend/requirements.lock` and `frontend/package-lock.json` record installed package versions. Torch's CUDA wheel index is explicit in setup. No weights or credentials are included. If the base checkpoint is not cached, the doctor reports its required location. Obtain `tabpfn-v3.5-20260909.safetensors` through the official model repository using your authorized access, review the applicable terms yourself, and rerun the doctor. No other model is substituted. `NEXTCHECK_DISABLE_MODEL=1` explicitly disables live inference for availability testing.

Local raw data is also supported:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --raw-dir C:\path\to\raw-sensors
```

## Numerical pipeline

```powershell
# Small, explicitly declared smoke development fit and full-grid smoke evaluation
.\.venv\Scripts\python.exe scripts\prepare_utility.py --smoke --context-limit 240 --utility-limit 6 --selection-limit 6
.\.venv\Scripts\python.exe scripts\evaluate.py --smoke --context-limit 240 --utility-limit 6 --selection-limit 6 --test-limit 8 --budgets 0 2 4 6 8 11 --lambdas 0 .005 .01 .02 .05 .1

# Full capped dataset evaluation; this is a longer run
.\.venv\Scripts\python.exe scripts\prepare_utility.py
.\.venv\Scripts\python.exe scripts\evaluate.py

# Record a real development example while the server is running
.\.venv\Scripts\python.exe scripts\record_demo.py
```

Use `../../work/nextcheck-venv/Scripts/python.exe` in place of `.venv/Scripts/python.exe` to reuse the already-tested environment in this delivered workspace. Run `--help` on each script for available options. Restart the app after utility preparation so its active model identities are revalidated.

For frontend development, run the backend in one terminal and `npm.cmd --prefix frontend run dev` in another. Vite uses http://127.0.0.1:5173 and proxies `/api` to port 8000. Production build: `npm.cmd --prefix frontend run build`.

## What is implemented

- Official bounded UCI archive download, path validation, row-local 70-feature extraction, and whole 20-cycle block splits. All 2,205 cycles are retained: 240 context, 80 utility, 80 selection, 120 test, and 1,685 unused. Smoke caps are separately reported.
- Explicit base TabPFN 3.5, fixed context and NaN observation masks, class alignment, checkpoint hashing, real readiness checks, and serialized inference.
- Context-only joint donor sampling, float64 KL decomposition, information/raw/entropy policies, value and ablation policies, random/static/prior/all-panels references, and bounded stopping.
- Versioned SQLite sessions, atomic single charges, purchase idempotency, cancellation, explicit error termination, artifact identity checks, and hidden readings omitted from active responses. Completed reports reveal labels and cannot resume.
- Ridge value fitting on utility rows, regularization and policy choices on separate selection rows, frozen choices before test scoring, metrics for every policy, full cost/lambda grid, null-task evaluation, and predicted-versus-realized value artifacts.
- Live inspection, immutable recorded replay, audit drawer, actual-result benchmark charts and tables, honest empty/unavailable states, and trace export.

## Evidence and limitations

The real checkpoint is `tabpfn-v3.5-20260909.safetensors`, SHA-256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`; installed TabPFN is 9.1.0. See `artifacts/model_status.json` and `BUILD_REPORT.md` for executed checks and final measured results. A mathematical fixture is used only in tests, never in production UI or real benchmark files.

Smoke results are pipeline checks with small sample support. They do not establish generalization, calibration, or an adaptive-policy advantage. A single rig and pragmatic block grouping do not prove independent samples. Native missing-value predictions and nearest-context donors are approximations; residual disagreement is a diagnostic, not evidence that a prediction is wrong. Reported cached timings distinguish reference computation from physical execution. Full capped evaluation, temporal stress testing, larger-draw sensitivity, and optional bootstrap sensitivity are separate additional runs.

Artifacts live under `artifacts/` (Git ignored): original hashes, frozen splits, model status, utility models/examples, each `run_*` evaluation, and sanitized replays. Interrupted development runs are preserved with explicit status. No tokens or model weights are copied into the project. Dataset attribution and model/prior-work sources are in `SOURCES.md`. Model weights have separate noncommercial terms; contest or commercial permission is not asserted.

## Repository map

`backend/nextcheck/` contains the API, state machine, data, model, acquisition, replay, storage, and evaluation modules. `frontend/` contains the console. `scripts/` contains reproducible commands. `demo_manifest.json` defines all panels/features/costs, and `score_test_vectors.json` contains clearly labeled mathematical cases. `DEMO_SCRIPT.md` explains the demonstration without invented measurements.
