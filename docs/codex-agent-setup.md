# Build orchestration evidence

This was a new repository. The supplied build specification was treated as product requirements, not as evidence that prior experiments had run. Initial inspection found no application files. Git was initialized before implementation; independent writers used separate Git worktrees under `work/trees` and explicit commit integration. The coordinator owns API/storage/jobs, integration, documentation and launch tools.

| Agent | Bounded ownership | Requested model / effort | Why parallel |
|---|---|---|---|
| data_model | UCI data, splits, manifest, actual model adapter, doctor | GPT-6 Sol / High | Long downloads and genuine model validation |
| numerics | Scores, donor sampler, observed-state engine, replay interface | GPT-6 Sol / High | Independent numerical and hidden-data invariants |
| evaluation | Ridge utility, ablation, frozen policies, benchmark | GPT-6 Sol / High | Separate development/test discipline and artifact generation |
| frontend | React console and build | GPT-5.6 Sol / Medium | Independent UI implementation against published contracts |
| integration_review | Read-only API/lifecycle/scientific integration review | Reviewer role, GPT-5.6 Sol / High | Independent race, identity and leakage review |

Runtime metadata exposed agent status but not effective model/effort. **Effective routing is unverified**; requested settings are not reported as verified savings. No grandchildren were spawned. Five child roles plus coordinator stayed below the seven-slot ceiling. High effort was used for numerical/scientific correctness and independent review, rather than mechanical file edits.

The final report records executed checks and unresolved limits. Reviews found artifact identity, interrupted-purchase recovery, optional utility readiness, replay allowlisting, default-policy trace, random baseline overhead, donor seed stability, and cache identity issues; these were fixed with focused regression checks. Earlier interrupted numerical runs are retained in artifacts with reasons; no test outcome was used to choose the fixes. Final measurements use the corrected implementation.

Build commands and artifact/privacy rules are maintained in the root README and AGENTS. Done requires meaningful numerical/API/browser checks and actual checkpoint inference, with smoke/full scope labeled. Do not claim the full capped scientific study was run merely because a small complete smoke grid succeeded.
