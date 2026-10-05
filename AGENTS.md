# IRONTRACE agent guidance

IRONTRACE is a local-first hydraulic inspection replay application by azaan noman. The public name is IRONTRACE; the Python package and compatibility environment variable retain `nextcheck` and `NEXTCHECK` names. The stack is Python 3.11, FastAPI, SQLite, NumPy/scikit-learn, real TabPFN 3.5, and React/TypeScript/Vite.

## Run and verify

From the repository root in Windows PowerShell:

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/prepare_model.py
.\.venv\Scripts\python.exe scripts/doctor.py
.\.venv\Scripts\python.exe -m pytest backend/tests -q
npm.cmd --prefix frontend run build
.\start.ps1
```

Open http://127.0.0.1:8000. `start.ps1` must use this checkout's `.venv` and one local model worker. `setup.ps1` installs the locked CUDA 12.8 dependency set and builds the frontend. `prepare_model.py` resolves the official gated checkpoint through local Hugging Face authentication after the user has reviewed and accepted its terms. It does not accept terms or request tokens. If access is denied, tell the operator to use `.\.venv\Scripts\hf.exe auth login`; never print credentials.

The numerical smoke path is:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_utility.py --smoke --context-limit 240 --utility-limit 6 --selection-limit 6
.\.venv\Scripts\python.exe scripts/evaluate.py --smoke --context-limit 240 --utility-limit 6 --selection-limit 6 --test-limit 8 --budgets 0 2 4 6 8 11 --lambdas 0 .005 .01 .02 .05 .1
```

The full capped commands are `scripts/prepare_utility.py` and `scripts/evaluate.py` without `--smoke`. A fresh clone has no generated `artifacts/` directory until preparation runs. The delivered evidence labels the completed run as an 8-case smoke evaluation; do not call the full study complete. Stop the server before doctor, pytest, utility fitting, or evaluation, and keep one GPU process. The validated machine is an NVIDIA RTX 4060 Ti with 8 GB; CPU, macOS, and Linux are unvalidated.

After a complete smoke evaluation, restart `start.ps1` and leave it running. In a second terminal at the repository root, run `.\.venv\Scripts\python.exe scripts/record_demo.py`, then `.\.venv\Scripts\python.exe scripts/verify_live.py`. These scripts call the running API and need the server alive. The live check requires matching prepared utility models, a complete benchmark, and a replay, and should report eight passed groups. Do not report a skipped model integration test as passed real inference.

## Evidence and privacy rules

- Derive every command, UI label, behavior, performance statement, and readiness claim from repository code, artifacts, tests, and build reports. Never invent measurements or readiness claims.
- Never claim that a smoke run proves generalization, calibration, policy advantage, or the stretch savings target. Report scope and class support with every benchmark claim.
- Use only the official TabPFN 3.5 checkpoint. Do not substitute a fixture or another model in production. Fixtures belong only in explicit tests.
- Keep hidden readings and labels out of active session responses. A report may reveal the historical label only after successful session termination. Do not expose credentials, tokens, model weights, raw full data, or private cache contents.
- Preserve the four disjoint block splits. Test labels are available only after decisions. Preserve model, data, utility-model, and artifact identity checks.
- Costs are simulated. The app does not control hardware or provide repair or safety advice.
- Keep runtime artifacts, environments, `node_modules`, full data, weights, and secrets out of Git. Do not add a software license.

## Modular boundaries and concurrency

Keep data/model work in `backend/nextcheck/data`, `backend/nextcheck/model`, `scripts/doctor.py`, and `scripts/prepare_data.py`; acquisition logic in `backend/nextcheck/acquisition`, `backend/nextcheck/agent`, and `backend/nextcheck/replay`; evaluation in `backend/nextcheck/evaluation`, `scripts/prepare_utility.py`, and `scripts/evaluate.py`; and UI work in `frontend`. API, storage, jobs, shared schemas, launcher, and documentation are coordinator-owned. Preserve four disjoint block splits, one model worker, and one GPU process. Parallel writers use isolated worktrees; no child agent spawns grandchildren.

## Ownership and changes

Use an isolated worktree for parallel writers and do not overwrite unrelated user changes. Keep public documentation and internal compatibility details consistent. Before changing a command, locate its implementation and run it when practical. Report files inspected and changed, commands and checks, blockers, and remaining risk. Commit explicit paths only when the task asks for a commit.
