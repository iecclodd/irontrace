# Opening IRONTRACE in Codex or another agent

IRONTRACE is a local Windows repository by azaan noman. Clone it, open the cloned folder in Codex, and let the agent discover the repository guidance before asking it to work.

## Open the cloned folder

```powershell
git clone https://github.com/iecclodd/irontrace.git
Set-Location irontrace
```

In Codex, choose **Open folder** and select the cloned `irontrace` directory. A Codex agent normally discovers the nearest `AGENTS.md` automatically when it starts work in that folder. If another agent, IDE, or automation does not perform automatic discovery, manually attach the repository's `AGENTS.md` to the task and state that it governs project guidance. Keep the agent in the cloned folder so relative commands resolve correctly.

Before asking for a live run, complete setup and model preparation in PowerShell:

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/prepare_model.py
.\.venv\Scripts\python.exe scripts/doctor.py
.\start.ps1
```

Open **http://127.0.0.1:8000**. Model preparation requires the operator to review and accept the official gated TabPFN terms first. If Hugging Face access is denied, the operator can run `hf auth login`; the repository command does not accept terms or request tokens.

## Copy-paste operational prompt

```text
Work in the cloned IRONTRACE repository. Read AGENTS.md and the relevant source files before acting. Use the public product name IRONTRACE and credit “by azaan noman”; retain internal nextcheck/NEXTCHECK compatibility names where the code requires them.

Derive every command, UI label, behavior, performance statement, and readiness claim from repository evidence. Do not invent measurements or duplicate existing guidance. For a live verification, use the repository .venv only, run the documented data/model preparation and doctor commands, start one local worker, and open http://127.0.0.1:8000. Check the Inspection and Benchmarks screens, including manual Recommend and Run recommended check, Autopilot, Stop, terminal report, recorded replay, and persisted benchmark scope. Do not expose hidden readings, test labels before termination, credentials, tokens, model weights, or private cache paths.

Keep the official TabPFN 3.5 checkpoint and UCI 447 attribution intact. Treat the delivered evaluation as an 8-case smoke result; the full capped study is not complete. Report files inspected and changed, exact commands, validation results, blockers, and remaining risk. Own only the files assigned to this task and preserve unrelated changes.
```

## Operating constraints

The console is an inspection replay. Costs are simulated and it does not control machinery or make safety or repair decisions. A recorded replay is immutable. A completed report can reveal the historical label and cannot resume acquisition. Benchmark charts read persisted, provenance-checked artifacts; an empty benchmark state is valid until evaluation commands produce results.

The official model terms and dataset license remain separate. The TabPFN checkpoint has separate noncommercial terms; UCI 447 is CC BY 4.0. No project-specific software license is included in this release. Dataset and model terms are separate; see `SOURCES.md`. Runtime artifacts and model weights remain outside Git, and `artifacts/` is generated locally rather than bundled in a clone.

When a task changes code or documentation, inspect Git status first, edit only assigned files, run the narrowest meaningful checks, and preserve existing work. If effective agent model routing is not visible, report it as unverified rather than inferring it from a requested model name.
