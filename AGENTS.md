# NextCheck project guidance

Local-first hydraulic inspection replay: Python 3.11/FastAPI/SQLite/TabPFN 3.5 and React/TypeScript/Vite. The supplied build specification is product requirements, not evidence that anything has run.

## Boundaries
- `backend/nextcheck/data`, `model`, `scripts/doctor.py`, `scripts/prepare_data.py`: data/model worker.
- `backend/nextcheck/acquisition`, `agent`, `replay`: numerical worker.
- `backend/nextcheck/evaluation`, `scripts/prepare_utility.py`, `scripts/evaluate.py`: evaluation worker.
- `frontend`: interface worker.
- API, storage, jobs, shared schema, integration, launcher, documentation: coordinator.

Use isolated worktrees for parallel writers; no grandchildren. Requested routing is recorded in build evidence; do not assume it was verified. Preserve unrelated edits.

## Rules
No hidden readings or labels in active session responses. Only context rows may feed prediction context or donors. Four disjoint block splits; test labels only after decisions. No substitute models or fabricated model/benchmark results. Fixtures only in explicit tests. Weights, tokens, full data, runtime artifacts, environments and node_modules stay out of Git. Never print credentials. Bind locally; purchases are transactional, versioned and idempotent. No hardware control or repair advice.

## Commands and done
Use shared Python at `../../work/nextcheck-venv/Scripts/python.exe` during development. Final setup/start/test commands belong in README after verified. Backend tests: `python -m pytest backend/tests`. Frontend: `npm run build` in frontend. Completion requires pure/API/browser checks, real 3.5 inference evidence, honest utility and frozen benchmark status. Access failures remain explicit blockers.
