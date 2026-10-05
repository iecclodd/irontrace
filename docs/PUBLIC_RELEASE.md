# IRONTRACE publication verification

by azaan noman

Validated 5 October 2026. This records the checks for the IRONTRACE rename and public repository preparation. The original numerical study is described separately in [BUILD_REPORT.md](../BUILD_REPORT.md).

## Changes

- Renamed the visible console, browser title, API title, launcher, frontend package, and current documentation to IRONTRACE. Added the requested author credit to the app header and metadata.
- Replaced workstation-specific launch assumptions with a project-local Python environment, prerequisite checks, and actionable startup failures.
- Added `scripts/prepare_model.py` to download the exact official checkpoint when absent and verify its SHA-256. Existing credentials are used through Hugging Face; model terms are not accepted by the script.
- Expanded the README and `AGENTS.md`, including operational instructions for using an AI coding agent.
- Fixed completed-report text causing horizontal overflow on narrow screens.
- Kept internal `nextcheck` package paths and environment variable names for compatibility. The original build specification is retained as historical requirements.

## Executed checks

| Command or check | Result |
| --- | --- |
| `./setup.ps1` with no project `.venv` | Created Python 3.11 environment; installed all 68 locked packages and editable backend; npm clean install and production build passed |
| `python scripts/prepare_model.py --check-only` | Existing official checkpoint passed exact SHA-256 verification |
| `python scripts/doctor.py` | Real TabPFN 3.5 held-out prediction passed using the new project environment |
| `python -m pytest backend/tests -q` | **52 passed**, including real model integration and four model-preparation boundary tests |
| `npm.cmd --prefix frontend run build` | TypeScript and Vite production build passed after the responsive fix |
| `./start.ps1` | Served the built console at `127.0.0.1:8000` using the project environment |
| `python scripts/verify_live.py` | **8 live API acceptance groups passed** with actual model inference |
| Browser, live API | IRONTRACE title/credit, start, recommend, purchase, stop, terminal report, audit, saved replay, and persisted benchmark results verified |
| Mobile completed replay | 390-pixel viewport and document width; no horizontal overflow |
| Browser application errors | None reported |
| Git whitespace and object checks | `git diff --check` and `git fsck --full --strict` passed |

The model checkpoint was already cached. The new download helper's absent-file path was tested with an explicit test fixture; a second 836 MiB download was not performed. Actual inference always used the real checkpoint. The successful doctor reported SHA-256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`.

`irontrace-console.png`, `inspection-live.png`, `audit-live.png`, `benchmark-results.png`, `recorded-replay.png`, and `mobile-check.png` are captures of the renamed application. `benchmark-empty.png` is retained as historical empty-state evidence from the original NextCheck build. A separate empty-state preview was not rerun for this release because automatic tool policy blocked launching that additional preview process. The main application and live validation completed successfully.

## Publication scope and independent review

The review covered the tracked tree and reachable Git history. No credentials, private keys, model weights, raw dataset archives, prepared dataset arrays, or runtime databases were found in tracked history. Runtime outputs, dependencies, local environments, checkpoints, and secrets remain ignored. Generated `artifacts/` contents referenced in the build report are local reproducible outputs, not bundled public evidence. Screenshots and source documentation are included. No project-specific software license is supplied; third-party model and dataset terms are described in [SOURCES.md](../SOURCES.md).

Two independent agent responsibilities supported this release: a documentation writer (requested Luna Medium) prepared onboarding while the parent changed the launcher and app, and a read-only reviewer (requested Sol High) audited public contents and Git history. Work stayed at one delegation level, with the writer in an isolated worktree. Effective child model/effort routing was not observable and is not claimed as verified.

Windows with an NVIDIA RTX 4060 Ti 8 GB was validated. CPU-only operation, Linux/macOS installation, the full capped study, and larger-data scientific validation were not rerun or claimed. The existing 8-case smoke study does not establish a policy advantage.
