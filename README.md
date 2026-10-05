# IRONTRACE

IRONTRACE is a local hydraulic inspection replay console by azaan noman. It uses the official TabPFN 3.5 base checkpoint to predict internal pump leakage (no, weak, or severe) while simulating the cost of buying access to additional sensor panels. It does not operate machinery and it does not provide repair or safety decisions.

The public product name is IRONTRACE. Internal Python modules and the compatibility environment variable `NEXTCHECK_ARTIFACTS` still use `nextcheck`/`NEXTCHECK` names.

## Windows quick start

These instructions assume Windows PowerShell and a clone of this repository.

1. Install Git, `uv`, and Node.js. The project requires Python 3.11; `uv` creates the project environment. Node.js 22.12 or newer is required (Node 24 is the validated version). Check the tools:

   ```powershell
   git --version
   uv --version
   node --version
   npm --version
   ```

2. Clone and enter the repository:

   ```powershell
   git clone https://github.com/iecclodd/irontrace.git
   Set-Location irontrace
   ```

3. Install locked dependencies and build the console:

   ```powershell
   .\setup.ps1
   ```

   Setup creates `.venv` with Python 3.11, installs `backend/requirements.lock` with the CUDA 12.8 PyTorch index, installs the backend package, runs `npm.cmd --prefix frontend ci`, and builds the frontend. It can take several minutes and needs internet access.

4. Prepare official UCI 447 data:

   ```powershell
   .\.venv\Scripts\python.exe scripts/prepare_data.py
   ```

   This downloads and validates the official archive, extracts required files, creates row-local features, and writes frozen context/utility/selection/test splits under `artifacts/data`.

5. Prepare the model checkpoint:

   ```powershell
   .\.venv\Scripts\python.exe scripts/prepare_model.py
   ```

   This resolves the official TabPFN 3.5 checkpoint and downloads it through `huggingface_hub` using local Hugging Face authentication. Review and accept the official gated model terms yourself before running it. The command does not accept terms, request tokens, or copy credentials. If access is denied, run `hf auth login`, complete authorization at Hugging Face, and run the command again. Weights remain in the external cache.

6. Run the real model doctor:

   ```powershell
   .\.venv\Scripts\python.exe scripts/doctor.py
   ```

   A successful run performs one held-out development prediction and writes `artifacts/model_status.json`. The first model load can take a few minutes. A failed doctor exits nonzero and prints the blocker.

7. Start IRONTRACE:

   ```powershell
   .\start.ps1
   ```

   Open **http://127.0.0.1:8000**. Wait for the readiness badge to say the model is ready. The server loads one model worker. Stop it with `Ctrl+C`. `start.ps1` uses only this clone's `.venv`; if it says `Run setup.ps1 first`, return to step 3.

## First inspection

The console opens on **Inspection**. Select a budget and cost weight, then choose a policy. **Selected on development data (recommended)** uses the frozen development choice when matching artifacts are available. Other choices include Information, Empirical value, Value without residual, Entropy drop, Raw KL, Random, Static, Prior only, and All panels. Start with budget `6` and cost weight `0.02` for the delivered smoke example.

Click **Start live session**. Initial TS1 summaries are visible. Locked panels show names and simulated prices; hidden readings and the historical label stay unavailable until the corresponding action or terminal report.

To work manually, click **Recommend**, inspect ranked checks, select a candidate, and click **Run recommended check**. This buys one panel, charges its simulated cost, and updates probabilities. A recommendation may be to stop; do not invent a purchase reason. The **Audit** drawer explains candidate score, raw KL, residual, and predicted one-step value. Disagreement is a diagnostic, not proof that a prediction is wrong.

**Autopilot** runs the bounded acquisition loop. **Stop** ends a live session manually. After successful termination, the report can reveal the historical label and the session cannot resume acquisition. Use **Export trace** for the sanitized append-only decision trace.

Use the top selector for **Live session** or a recorded development replay. A replay is marked **RECORDED REPLAY**; its controls and branch actions are disabled. Replay data is sanitized and does not expose raw row indices or hidden readings.

Open **Benchmarks** to inspect persisted evaluation results. The screen reads actual `artifacts/run_*` files and does not generate sample metrics. A fresh clone has no generated artifacts until you run preparation and evaluation. Filter by budget and cost weight. Read scope, status, split counts, model reference, class support, log loss, mean simulated cost, accuracy, and acquisition counts together. The delivered run is an explicitly labeled 8-case smoke evaluation; the full capped 80-utility/80-selection/120-test study has not been run.

## Quick smoke and full study

```powershell
# Small development fit and evaluation
.\.venv\Scripts\python.exe scripts/prepare_utility.py --smoke --context-limit 240 --utility-limit 6 --selection-limit 6
.\.venv\Scripts\python.exe scripts/evaluate.py --smoke --context-limit 240 --utility-limit 6 --selection-limit 6 --test-limit 8 --budgets 0 2 4 6 8 11 --lambdas 0 .005 .01 .02 .05 .1

# Full capped evaluation (longer run)
.\.venv\Scripts\python.exe scripts/prepare_utility.py
.\.venv\Scripts\python.exe scripts/evaluate.py

# Record an illustrative development replay while the server is running
.\.venv\Scripts\python.exe scripts/record_demo.py
```

Smoke and full runs can take minutes or longer depending on GPU, cache state, and model initialization. Restart IRONTRACE after preparing utility artifacts so the worker revalidates model and utility identities. Run `--help` on each script for options. `NEXTCHECK_ARTIFACTS` remains a compatibility override for artifact location.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe scripts/verify_live.py
```

Run `verify_live.py` with the server, data, model, and artifacts ready. Repository evidence records 48 backend tests, a passing frontend build, and eight live acceptance groups against actual TabPFN 3.5. Repeat checks on your machine because device timings and caches vary.

## Troubleshooting

- **A tool is not recognized:** install Git, `uv`, and Node.js 22.12+; open a new PowerShell window and rerun version checks.
- **Torch installation fails:** keep the locked CUDA 12.8 index from `setup.ps1`, confirm network access, and rerun setup.
- **Data preparation fails:** use the official download, or pass raw files with `scripts/prepare_data.py --raw-dir C:\path\to\raw-sensors`.
- **Model access is denied:** review and accept official terms, run `hf auth login`, and rerun `prepare_model.py`.
- **Doctor reports no checkpoint:** fix model access/cache permissions, rerun `prepare_model.py`, then `doctor.py`.
- **Live inference is unavailable:** check PowerShell output, confirm `artifacts/data` and `artifacts/model_status.json`, then press **Recheck**.
- **Benchmarks are empty:** run `prepare_utility.py` and `evaluate.py`; the screen only displays persisted results.
- **A session action fails:** read the red action error, check remaining budget and readiness, and start a new session if artifacts or model identity changed.

## Evidence, licenses, and limits

The source dataset is UCI dataset 447, *Condition monitoring of hydraulic systems*, by Helwig, Pignanelli, and Schütze (2015), under CC BY 4.0. See [SOURCES.md](SOURCES.md). The TabPFN 3.5 checkpoint has separate noncommercial terms; review the official model card and terms yourself. Contest or commercial permission is not asserted. No project-specific software license is included in this release. Dataset and model terms are separate; see [SOURCES.md](SOURCES.md).

The recorded checkpoint is `tabpfn-v3.5-20260909.safetensors`; its hash and provenance are retained in a locally generated `artifacts/model_status.json` and the build report. `artifacts/` is generated locally and is not included in a fresh clone. The build report describes a historical local smoke run; it does not bundle runtime outputs. Smoke results do not establish generalization, calibration, or adaptive-policy advantage. One rig and pragmatic block grouping do not prove independent samples. Temporal stress testing, larger-draw sensitivity, bootstrap sensitivity, cross-machine validation, hardware deployment, and the full capped study remain outside the delivered evidence.

## Repository map

`backend/nextcheck/` contains API, state machine, data, model, acquisition, replay, storage, and evaluation modules. `frontend/` contains the console. `scripts/` contains reproducible commands. `demo_manifest.json` defines panels, features, and costs. Runtime artifacts, weights, credentials, full data, environments, and `node_modules` stay out of Git.
