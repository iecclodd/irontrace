# Opening IRONTRACE in Codex or another agent

IRONTRACE is a local Windows repository by azaan noman. Clone it, open the cloned folder in Codex, and let the agent discover the repository guidance before asking it to work.

## Open the cloned folder

```powershell
git clone https://github.com/iecclodd/irontrace.git
Set-Location irontrace
```

In Codex, choose **Open folder** and select the cloned `irontrace` directory. A Codex agent normally discovers the nearest `AGENTS.md` automatically when it starts work in that folder. If another agent, IDE, or automation does not perform automatic discovery, manually attach the repository's `AGENTS.md` to the task and state that it governs project guidance. Keep the agent in the cloned folder so relative commands resolve correctly. The agent can run setup, data preparation, model preparation, doctor, tests, and browser verification autonomously; the operator only handles review of gated model terms and Hugging Face login when required.

Ask the agent to run these steps in order, or run them in PowerShell when operating manually:

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/prepare_model.py
.\.venv\Scripts\python.exe scripts/doctor.py
.\start.ps1
```

Open **http://127.0.0.1:8000**. Model preparation requires the operator to review and accept the [official gated TabPFN terms](https://huggingface.co/Prior-Labs/tabpfn_3_5). If Hugging Face access is denied, use `.\.venv\Scripts\hf.exe auth login`; the repository command does not accept terms or request tokens. The checkpoint is about 836 MiB (876,027,932 bytes). The validated machine has an NVIDIA RTX 4060 Ti with 8 GB and CUDA 12.8; CPU, macOS, and Linux are unvalidated.

## Copy-paste operational prompt

```text
Work in the cloned IRONTRACE repository. Read AGENTS.md and this runbook before acting. Use the public product name IRONTRACE and credit “by azaan noman”; retain internal nextcheck/NEXTCHECK compatibility names where the code requires them.

Derive every command, UI label, behavior, performance statement, and readiness claim from repository evidence. Do not invent measurements or duplicate existing guidance. Use the repository .venv only. Run setup, prepare_data, prepare_model, doctor, and the backend/frontend checks. Stop the server with Ctrl+C before doctor, pytest, utility fitting, or evaluation; keep one GPU process. For complete verification, prepare the documented smoke utility and evaluation artifacts. Restart one server and KEEP IT RUNNING while a second terminal runs record_demo.py followed by verify_live.py; these scripts call its API and must not run against a stopped server. Keep the same server alive for browser checks. For a fresh clone choose Information for the first session. Check Inspection and Benchmarks, including Recommend, Audit, Run recommended check, Autopilot, Stop/report, recorded replay, and persisted benchmark scope. Clicking a candidate opens Audit but does not select a buy target; Run recommended check buys the recommended candidate. Do not expose hidden readings, test labels before termination, credentials, tokens, model weights, or private cache paths.

Keep the official TabPFN 3.5 checkpoint and UCI 447 attribution intact. Treat the delivered evaluation as an 8-case smoke result; the full capped study is not complete. Report files inspected and changed, exact commands, validation results, blockers, and remaining risk. Own only the files assigned to this task and preserve unrelated changes.
```

## Operating constraints

The console is an inspection replay. Costs are simulated and it does not control machinery or make safety or repair decisions. A recorded replay is immutable. Stop produces the terminal report, which can reveal the historical label, and the session cannot resume acquisition. Benchmark charts read persisted, provenance-checked artifacts; an empty benchmark state is valid until evaluation commands produce results. Value policies are unavailable until utility artifacts exist.

The official model terms and dataset license remain separate. The TabPFN checkpoint has separate noncommercial terms; UCI 447 is CC BY 4.0. No project-specific software license is included in this release. Dataset and model terms are separate; see `SOURCES.md`. Runtime artifacts and model weights remain outside Git, and `artifacts/` is generated locally rather than bundled in a clone.

When a task changes code or documentation, inspect Git status first, edit only assigned files, run the narrowest meaningful checks, and preserve existing work. If effective agent model routing is not visible, report it as unverified rather than inferring it from a requested model name.
