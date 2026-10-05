# NextCheck — Codex build specification

**Tagline:** Know what to measure next.

**Product promise:** Given limited observations and a measurement budget, recommend the next sensor panel, revise the prediction after its readings become available, and stop when another panel is not predicted to justify its cost.

**Status of this document:** A proposed hackathon build, not an implemented application or a claim of measured TabPFN performance. Created 4 October 2026. The previous research kit executed mathematical checks and random-forest proxy experiments, not TabPFN 3.5 inference. None of its accuracy numbers may be reused as NextCheck results.

## 1. Build one concrete product

Build a local-first, visually polished inspection-replay application with actual TabPFN 3.5 at its numerical core. The flagship task is classifying a hydraulic rig's **internal pump leakage condition** from selectively revealed historical sensor panels. This is a software-only benchmark and demonstration. Do not connect machinery, control equipment, advise physical repairs, or characterize the application as a validated safety system.

The intended workflow is an inspection team deciding which additional measurements would be useful. The hackathon implementation evaluates that workflow retrospectively: the public dataset already contains complete recordings, and the application simulates buying access to hidden panels. It does not collect new physical measurements.

A session begins with one temperature sensor visible. Pressure, flow, vibration, electrical power, and additional temperature panels are locked. The numerical agent compares candidate panels, chooses a feasible action, receives only the purchased values from the replay service, updates class probabilities, and repeats or stops. A side-by-side benchmark compares the policy with static and simpler acquisition rules.

The project is NOT an open-ended virtual laboratory, a materials-discovery optimizer, a generic chat-with-CSV app, or a multi-agent framework. A language model is optional and can explain verified trace fields; it must never generate probabilities, choose unsupported measurements, or override numerical decisions.

**Scope:** One dataset, one target, five purchasable groups, one coordinating state machine, one primary prediction backend, one inspection screen, one benchmark screen, one audit drawer. CSV import, additional targets, deep lookahead, online fine-tuning, user accounts, payments, and deployment infrastructure are outside the required build.

## 2. What makes the idea worth demonstrating

The product question is not just “What condition is this?” but “Which missing observation is worth obtaining before deciding?”

The research-informed addition is an **acquisition audit and empirical value model**. A candidate measurement can appear useful because its imagined predictions disagree with the current prediction, even if its possible outcomes do not meaningfully distinguish classes. Compute both information and disagreement. Then test whether these diagnostics help estimate actual improvement on separate labeled development examples.

Do not claim that active feature acquisition, mutual information, stopping, or the decomposition below is new. Learning-to-Measure and AFABench are relevant prior work. The hackathon contribution is the application, transparent numerical decision loop, and an honest controlled evaluation of an audit-aware policy using an explicitly identified 3.5 checkpoint. Its improvement is a hypothesis to measure, not a predetermined result. See SOURCES.md.

## 3. Development priorities and gates

### Gate A — prove model and data access

Inspect the repository and preserve existing work. Create an isolated environment. Verify model access and run an actual TabPFN 3.5 fit/predict smoke test. Download and validate the dataset. Persist provenance and the first real probabilities before polishing the interface.

If access or license acceptance needs the user's action, expose the exact blocker. Do not accept terms for the user, bypass access requirements, substitute an older model, or fabricate a successful inference. Continue independent UI/unit-test work using clearly marked mathematical fixtures. Model-dependent readiness remains blocked.

### Gate B — functional MVP

Implement fixed-context predictions, the hidden-data boundary, conditional donor sampling, raw/centered/entropy acquisition scores, cost accounting, a bounded sequential loop, and meaningful stop reasons. Implement no-acquisition, class-prior, static-subset, random, and all-panels references. Connect the live UI to these computations.

At this point, the product is a working **information-guided acquisition prototype**, not yet an empirically calibrated acquisition policy.

### Gate C — research differentiator

Generate out-of-context utility-training examples, fit the small audit-aware value model, compare it with its no-residual ablation, choose defaults using a separate policy-selection split, and run the frozen test benchmark. Only call the product “value-validated” after this is implemented and results exist. If static or prior-only wins, retain that result and expose the selected default truthfully.

### Gate D — demonstration

Prepare a reproducible recorded trace from a development example, a live-run option, complete benchmark artifacts, browser checks, and an honest demo script. Replay and live inference must be visibly different modes.

Do not delay Gates A–C for optional LLM features or animations.

## 4. Stack and repository layout

Use Python 3.11, FastAPI, Pydantic, NumPy, pandas, SciPy, scikit-learn, PyTorch, actual `tabpfn`, and SQLite. Use React + TypeScript + Vite for a local frontend, a lightweight charting library, and ordinary CSS or Tailwind. Lock versions that actually install and pass the smoke test; do not invent package-version pins from memory.

Use one long-lived model worker and a serialized inference queue. Do not load the GPU checkpoint per request or start multiple web workers each owning a copy. Background jobs here are application processes explicitly created by the implementation, not promises that the assistant is doing work later.

Suggested layout:

```text
AGENTS.md
NEXTCHECK_CODEX_BUILD.md
CODEX_START_PROMPT.md
SOURCES.md
DEMO_SCRIPT.md
demo_manifest.json
score_test_vectors.json
backend/
  pyproject.toml
  nextcheck/
    api.py
    schemas.py
    settings.py
    storage.py
    jobs.py
    data/{download,extract,split}.py
    model/{base,tabpfn35}.py
    acquisition/{sampler,scores,utility,policies}.py
    agent/{state,loop}.py
    replay/{store,environment}.py
    evaluation/{benchmark,metrics,provenance}.py
  tests/
frontend/
  package.json
  src/
    pages/{InspectionPage,BenchmarkPage}.tsx
    components/{SensorPanels,PredictionBars,CandidateTable,AuditDrawer,TraceTimeline}.tsx
scripts/
  doctor.py
  prepare_data.py
  prepare_utility.py
  evaluate.py
  launch.py
artifacts/                 # generated outputs, ignored by git except safe small examples
README.md                  # actual setup, executed checks, known limitations
.env.example
.gitignore
```

Adapt paths to an existing project when sensible. Module boundaries matter more than exact filenames. Never overwrite an existing AGENTS.md or unrelated source files blindly.

## 5. Dataset, target, and panel contract

Use UCI **Condition monitoring of hydraulic systems**, dataset 447, DOI `10.24432/C5CW21`. Its documentation describes 2,205 experimentally recorded 60-second cycles, raw sensor matrices, and condition labels in `profile.txt`. It is licensed CC BY 4.0. Attribute its creators and preserve the dataset citation. Its approximately 73 MB download expands substantially; process one sensor file at a time rather than loading every raw matrix simultaneously. The dataset contains no original missing values; our observation masks are simulated. [S3]

Download from the official dataset page's download link. Do not assume the convenience API returns the raw file layout needed for panel extraction. Support an explicit local dataset directory as well. Validate archive paths before extraction, bound download/extraction size, reject path traversal, and never execute archive contents.

### Target

The third column of `profile.txt`, zero-based column 2, is internal pump leakage:

- 0: no leakage.
- 1: weak leakage.
- 2: severe leakage.

Keep this three-class task. Do NOT reinterpret this target as overall machine health: other component faults can coexist. Do not use the other profile columns as predictors. The stable-condition flag and row/cycle identifiers are metadata only, never features. Do not filter using test labels to manufacture a easier benchmark.

### Feature extraction

For each included raw sensor, compute five features within each 60-second cycle: mean, population standard deviation (`ddof=0`), minimum, maximum, and least-squares slope versus time in seconds. Use the sensor's documented sample frequency. Name features `sensor__stat`. These transformations are row-local; no global label-informed feature selection is allowed.

For slope, use `t = arange(n_samples) / sample_rate_hz`; if a sequence has fewer than two samples, reject it with a data-validation error. All scalar outputs must be finite. Validate identical cycle counts and preserve raw row alignment. This yields 70 features from 14 physical sensors.

### Panels and artificial costs

| Panel | Sensors | Sampling rate | Cost units |
|---|---|---:|---:|
| Initially observed temperature | TS1 | 1 Hz | 0 |
| Pressure | PS1, PS2, PS3, PS4, PS5, PS6 | 100 Hz | 4 |
| Flow | FS1, FS2 | 10 Hz | 3 |
| Vibration | VS1 | 1 Hz | 1 |
| Electrical power | EPS1 | 100 Hz | 2 |
| Additional temperatures | TS2, TS3, TS4 | 1 Hz | 1 |

All five purchasable groups cost **11 simulated units** in total. Default session budget: 6 units. Benchmark budgets: 0, 2, 4, 6, 8, 11. Prices are editable scenario assumptions, not sourced industrial prices, dollars, energy measurements, or actual sensor activation expenses. Sensors were recorded together in the original data.

Exclude CE, CP, and SE from the MVP: they are derived/virtual quantities and would require a separately defined dependency and charging model. No free feature may depend on an unpurchased panel. Purchasing a group atomically reveals all five summaries for each of its sensors. Optional raw sparkline data can be returned only for already purchased sensors.

`demo_manifest.json` is the machine-readable contract. Validate every included feature belongs to exactly one panel.

## 6. Split and development-label discipline

Use four disjoint row/entity partitions: prediction context, utility fitting, policy selection, and final test. Default approximate proportions: 50%, 20%, 15%, 15%. Initially cap them at 256 context rows, 96 utility rows, 96 policy-selection rows, and 128 test rows. Preserve and report the unused data; it must not silently enter donors, fitting, normalization, or calibration. These are resource caps, not claims that only 256 labeled examples were used overall.

Use contiguous blocks of 20 original cycles as group IDs, then split **groups** deterministically. This is a pragmatic within-rig dependence control, not an official experimental group annotation or proof of independence. When actual run/batch metadata exists, prefer it and record that choice. A temporal holdout is a secondary stress test; no cross-machine generalization claim is supported by this one-rig dataset.

Fix seed 20261004. Save cycle IDs, block assignments, selected rows, hashes, and exact split code before running final inference. Check class coverage in the context and development partitions; never search across test outcomes for a favorable split. If the predeclared split is unusable, report it and document a development-only revision rather than silently cherry-picking. Test class counts may be reported after evaluation, not used to tune the split.

Only context rows feed the TabPFN context and donor sampler. Only utility rows train the value estimator. Only policy-selection rows tune its regularization, optional risk heuristic, stopping configuration, and default policy. Test labels are accessible only to the evaluator after action selection. All masks generated from one original row remain within that row's partition.

## 7. Real TabPFN 3.5 integration

Use the official package and explicitly request the base 3.5 checkpoint. The reviewed model card documents this API and filename [S1–S2]:

```python
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

model = TabPFNClassifier.create_default_for_version(
    ModelVersion.V3_5,
    device="auto",
    n_estimators=4,
    random_state=20261004,
)
model.fit(X_context, y_context)
probabilities = model.predict_proba(X_query)
```

Inspect the installed API before using optional parameters. Expected base checkpoint: `tabpfn-v3.5-20260909.safetensors`. Record the resolved checkpoint path, its actual SHA-256, library version, device, ensemble settings, training rows, and context hash. A filename alone is not full provenance. Align output columns explicitly through `classes_`; never assume arbitrary output order.

Require successful real inference before reporting model readiness. A fixture backend may support unit tests only. Random forests and older TabPFN versions must never be silent fallback implementations. Base 3.5 is required for the core submission; using another explicitly labeled model would be a separate comparison, not completion of this requirement.

Weights have a separate noncommercial license. Do not distribute weights in the repository. Surface legitimate account/license/token requirements and keep credentials server-side. Do not claim that a prize contest or eventual commercial product is automatically permitted; the user must check applicable terms and event rules. [S2, S7]

### Conditioning contract

MVP: fit once on the full context feature table. At inference, place actual observed values in their correct columns and represent unacquired groups as `NaN`. Hypothetical acquisition fills just the candidate group; all other unacquired groups remain `NaN`. Reuse the same fitted context throughout a session.

Native missing-value handling is not guaranteed Bayesian marginalization. Treat the scores as diagnostics of this composed pipeline. If implementing subset-conditioned models, expose them as a separate development-selected configuration, with separate result labels and no claims that the residuals prove an intrinsic network defect.

No gradient fine-tuning is needed. Do not externally scale or one-hot-encode model inputs just because the donor-distance calculation uses scaling. Verify official caching support before enabling it, and record any `fit_mode` changes. Batch predictions rather than submitting one hypothetical row at a time. [S1]

## 8. Hidden-data boundary

Create two separate interfaces:

```text
ObservedCase: visible group IDs + visible feature values + public schema.
ReplayEnvironment: hidden complete row + hidden label + charged reveal(group).
```

The acquisition policy receives only `ObservedCase`, budget, context-based sampler, predictor, and public panel metadata. It must not receive the full row, label, profile metadata, or a lookup function keyed by cycle ID. API case IDs are opaque handles, not numerical features. Keep the complete-row store in the replay/evaluation layer.

The frontend cannot receive unacquired values and merely blur them with CSS. Omit them from JSON, DOM, tooltips, traces, logs, downloaded active-session exports, and preload routes. Before termination, the true label must also be absent. Completed-case reports may reveal the label for retrospective scoring, but must not permit resuming acquisition on that same revealed session.

A strict test will replace a case's hidden values and hidden label while preserving its visible state. With the same random seed, all pre-acquisition scores and the chosen action must remain unchanged.

## 9. Conditional hypothetical outcomes

Start with a nearest-training-donor sampler, not an invented generative capability:

1. Fit median and interquartile-range scales on context rows for donor distance only. Use a safe positive scale for zero-IQR columns.
2. Compute distances using only already observed features. To avoid a large panel dominating distance solely by column count, average normalized squared distance within each observed panel, then average over panels.
3. Select at most the 32 nearest context rows with deterministic tie-breaking. If there are no observed numerical values, use the full allowed context pool.
4. Sample 16 donor rows with replacement from that neighborhood using a state-seeded RNG. Copy candidate-panel features jointly from each donor row; do not sample its columns independently.
5. Reuse the same donor draws for raw KL, centered information, and entropy-reduction scores. Reuse appropriate shared draws across policy comparisons.

Never condition donors on an unobserved test label or an unacquired test value. Log neighborhood size, observed-only distances, unique donor count, and RNG seed. Repeating with more draws is a sensitivity check; do not call this an exact conditional distribution or a certified uncertainty interval.

## 10. Acquisition scores and their interpretation

Let `q0` be the current class probabilities, `q_m` the probabilities after candidate outcome m, and `qbar` their average. Clip probabilities at 1e-8 and renormalize consistently. Use natural logarithms throughout.

```text
raw_kl       = mean_m KL(q_m || q0)
information  = mean_m KL(q_m || qbar)
residual     = KL(qbar || q0)
entropy_drop = H(q0) - mean_m H(q_m)

raw_kl = information + residual
```

This decomposition must hold numerically under the same finite samples. `information` is a model-implied quantity; it is not established real-world measurement value. `residual` is pipeline disagreement, not an error probability or proof of failure. Low residual does not imply correctness; high residual does not establish which prediction is wrong. Finite sampling can create positive residuals even for coherent underlying predictions.

Implement baseline policies based on raw KL per cost, centered information per cost, and entropy drop per cost. Avoid divide-by-zero: free initial groups are already acquired; any additional zero-cost panels are handled explicitly and may be revealed once. Negative entropy drop remains negative; do not cosmetically clamp it into an apparent gain.

Use `score_test_vectors.json` for deterministic numeric fixtures. Those values are mathematical test cases, not actual model outputs. The UI audit drawer may explain the identity but must not label a constructed counterexample a demonstrated TabPFN defect.

## 11. Empirical value model: required differentiator

On utility-fitting rows that never enter the prediction context, generate four deterministic/seeded partial-observation states per row with different observed-panel sets. Include the initial state and varying later states; exclude terminal all-panels states from action generation. Keep budgets/action feasibility recorded. For each candidate:

- Generate diagnostics using only allowed observations and training donors.
- Separately reveal the actual candidate values in the offline utility evaluator.
- Compute actual held-out log-loss improvement:

```text
loss_before = -log(q0[y])
loss_after  = -log(q_actual_after[y])
delta       = loss_before - loss_after
```

The utility target may use labels **only in this offline development procedure**. The running policy cannot compute delta before choosing an action.

Fit a regularized ridge regression to predict delta from an economical state-action representation: centered information, residual, current entropy, top-probability margin, observed-panel mask/count, candidate ID, and observed-only sampler diagnostics. Avoid redundant inclusion of raw KL when information and residual are already inputs. Cost is subtracted by the decision rule rather than treated as a causal predictor of information benefit.

Fit preprocessing only on utility data. Select regularization using the policy-selection split. The target is expected one-step improvement, not a calibrated guarantee or globally optimal sequence value. Report predicted-versus-realized value plots on test data after policies are frozen.

Ablation: train an otherwise identical value model without residual. Without this comparison, no claim that the audit adds value is supported.

Optional uncertainty heuristic: fit 20 bootstrap ridge models by resampling original row IDs, keeping all of a row's generated states together. Use `mean_delta - k * std_delta` with k selected on policy-selection data. Label this bootstrap sensitivity, not a statistical safety bound. A point-estimate ridge is sufficient for Gate C; do not block the core implementation on bootstrapping.

## 12. Cost-aware loop and stopping

For every affordable, unacquired action compute:

```text
net_value_j = predicted_delta_j - lambda_cost * simulated_cost_j
```

`lambda_cost` has units of log-loss nats per simulated cost unit. Default scenario value: 0.02. Benchmark a small predeclared grid `[0.0, 0.005, 0.01, 0.02, 0.05, 0.1]`; do not pick the best test setting after seeing results. A user-set lambda represents a tradeoff preference, not a scientifically calibrated price.

Select the action with highest positive net value. Otherwise stop. Also stop on exhausted budget, no remaining groups, explicit cancellation, or unrecoverable model failure. Do not turn inference failure into a fabricated completed prediction.

After acquisition, update visible values, charge exactly once, recompute probabilities, and reconsider every remaining feasible action. Repeated purchase of the same group is idempotent. At most five purchases are possible; the loop is bounded.

For the pre-value-model MVP, use centered-information-per-cost with a development-selected stopping threshold. Mark this as a provisional heuristic policy. Do not describe its score as predicted realized loss improvement.

Each terminal trace records a reason: `no_predicted_net_value`, `budget_exhausted`, `no_groups_remaining`, `user_stopped`, or `error`. Stop means “the policy did not choose another panel,” not “the device is safe” or “the class prediction is certainly correct.” Always retain the probability distribution; use “inconclusive” where appropriate rather than inventing certainty.

For each budget/lambda, select the default among eligible policies using policy-selection mean `log_loss + lambda_cost * cost`. Include a class-prior/no-measurement option. If a static policy wins, the interface should say so; users may still inspect the experimental adaptive policy. Guarding against needless measurements is part of the product, not an excuse to hide weak adaptive results.

## 13. Backend/API contract

Use Pydantic models and a state-version field for all mutations. Recommended routes:

```text
GET  /api/health                     -> model readiness, data readiness, blockers
GET  /api/model-info                 -> verified provenance, never credentials
GET  /api/demo-manifest              -> labels, panel names, assumed costs
POST /api/sessions                   -> opaque session with initially visible data
GET  /api/sessions/{id}              -> visible state only
POST /api/sessions/{id}/recommend    -> queued numerical recommendation
POST /api/sessions/{id}/acquire      -> validated, charged reveal of one group
POST /api/sessions/{id}/run          -> bounded automated acquisition job
POST /api/sessions/{id}/stop         -> terminate session
GET  /api/jobs/{id}                  -> state/progress/errors
GET  /api/sessions/{id}/trace         -> sanitized, append-only event trace
GET  /api/sessions/{id}/report        -> completed-session report only
GET  /api/benchmarks                 -> actual persisted evaluation summaries
```

A recommendation includes candidate group ID, affordability, acquisition metrics, predicted utility if available, expected cost, selected action or stop reason, RNG/state hashes, and model provenance reference. Do not return made-up milliseconds or savings. Measure inference time with a monotonic timer and distinguish warm versus cold startup.

Use SQLite transactions and an idempotency key for purchases. Reject stale recommendations after state changes; do not buy a panel based on an old budget or old observed mask. Disable invalid UI controls, but enforce all rules on the server as well. Use local-only binding and restricted local CORS by default. No authentication platform is required for a local demo; public hosting would need a separate security review and authorization.

## 14. Interface design

Make it feel like a working instrument console, not a chatbot homepage. Use readable typography, neutral surfaces, one restrained accent color, and strong hierarchy. Do not use stock robot art, decorative neural networks, or meaningless pulsing loaders.

### Inspection screen

Header: NextCheck / Know what to measure next. Persistent badges: `TabPFN 3.5`, `LIVE INFERENCE` or `RECORDED REPLAY`, and `SIMULATED ACQUISITION COSTS`.

Left: abstract sensor-panel layout, not instructions for a real rig. Locked cards show sensor names and costs, never hidden readings. Purchased cards reveal actual summaries. Budget changes create a new session or explicitly version the unstarted configuration; never rewrite past charges.

Center: three-class probability bars for the pump target. Label these “model probabilities,” not certified confidence. Show cost spent, groups acquired, and measured inference time. Probabilities are allowed to decrease or remain ambiguous.

Right: ranked candidate table and a “Run recommended check” button. Separate expected value from observed improvement. Add an “Autopilot” control that invokes the bounded software replay loop, and a stop button.

Bottom: chronological decision trace: proposed action, computed reason, actual acquired values, updated probabilities, and stop reason. Explain numbers using a deterministic template. An optional single LLM may paraphrase the sanitized trace after the core app works, but cannot access hidden data or invent engineering causal explanations.

### Audit drawer

Show raw KL, centered information, residual, donor count, predicted value, and sensitivity when computed. Explain: “Disagreement is a diagnostic, not proof of an incorrect prediction.” Show whether the information score and empirical value model rank candidates differently. Never force a dramatic example where measured data does not support it.

### Benchmark screen

Plot predictive log loss against mean simulated measurement cost. Provide accuracy, balanced accuracy, class support, Brier score, acquisition count, wall time, and status. Include sample counts and exact split/model/configuration references. Distinguish all-panels from budget-feasible policies. All-panels is a full-information reference, not a guarantee of the best accuracy and not feasible at budgets below 11.

Charts must come from actual result files. Before evaluations exist, show an empty state with instructions, not fabricated sample metrics. Mathematical UI fixtures must have their own unmistakable “fixture” mode and cannot power a purported real-model benchmark.

## 15. Fair benchmark

Use the same context rows, prediction backend, conditioning mode, panel definitions, test cases, and acquisition costs for policy comparisons.

Required policies:

- Class-prior only, with probabilities estimated solely from context labels and no acquisition.
- Initially observed features only, using TabPFN.
- Best development-selected fixed feature subset at each budget. There are only 32 subsets of five groups, including the empty subset; evaluate feasible subsets on policy-selection data, not the test set. Charge their exact panel costs.
- Random feasible acquisition, seeded and averaged across several acquisition RNG seeds.
- Raw-KL, centered-information, and entropy-drop greedy acquisition, with development-selected stopping thresholds where applicable.
- Empirical value policy with and without the residual diagnostic.
- All panels using TabPFN, charged 11 units, shown as a separate reference when unaffordable.

Persist every evaluated policy, not just the winner. Evaluate budgets `[0,2,4,6,8,11]`, and make the smaller smoke benchmark explicit when full runs are incomplete. At least one complete held-out run must use actual 3.5 inference. Multiple seeds are a useful extension, not license to claim independent data replications.

Primary metric: multiclass log loss. Also report accuracy, balanced accuracy, multiclass Brier score (mean sum of squared probability errors), mean cost, number of purchases, prediction calls, hypothetical query rows, and wall time. Report context/utility/selection/test label budgets separately. Do not turn softmax probabilities into a claimed error guarantee. Include a synthetic null task where labels are independent of features; report whether the policy needlessly acquires information instead of forcing a favorable result.

Any statement like “20% cheaper at similar accuracy” needs a prespecified tolerance and an actual comparison at that tolerance. Suggested stretch target: at least 20% lower mean simulated cost with log loss no more than 0.02 nats worse than a selected reference. This is a target, not a measured fact or guaranteed acceptance threshold. Comparing against all-panels alone is insufficient when a strong static subset is cheaper.

Record policy-selection choices before reading final results. Do not tune lambda, choose examples, change features, or select a target using final-test performance. Demo examples may be curated from the development split if labeled illustrative. A benchmark loss is still a valid result; summarize why rather than replacing it with invented wins.

## 16. Runtime and reproducibility

Start with 256 context rows, four estimators, 16 draws, and at most five candidate groups: the first state needs at most 80 hypothetical query rows, plus the current prediction. This is a query-count calculation, not a latency promise. Batch or chunk based on measured memory. Reduce prediction chunk size first if memory is constrained. A smaller declared donor/context configuration is allowed only as an explicitly recorded run configuration used consistently in comparisons; never silently change the model.

Cache diagnostics by checkpoint hash, context hash, actual observed values/mask, panel manifest, sampler configuration/seed, and conditioning mode. Include policy/value-model hashes and costs when caching final recommendations. A cache key based only on row ID or feature count is insufficient and can leak information or return stale decisions.

Save immutable traces from real inference for presentation reliability. Replay files include timestamp, checkpoint hash, input/split hashes, policy configuration, and provenance. Read-only replay cannot pretend to compute new branches after the user changes a budget or action. Such an interaction requires live inference or a genuinely existing matching cached state; otherwise disable it and explain why.

Create `artifacts/run_<id>/` with environment metadata, data hashes, splits, model provenance, utility training metadata, frozen policy selections, per-case outcomes, aggregate metrics, sanitized traces, and `execution_status.json`. Do not include tokens or redistribute weights. Seed stochastic steps; do not claim exact GPU bitwise determinism unless verified.

## 17. Acceptance tests

Implement tests at the appropriate pure-function, API, integration, and browser layers:

1. Score identity on supplied fixtures and random normalized distributions; absolute error below 1e-10 in float64 fixtures.
2. Equal hypothetical predictions produce zero centered information even when raw KL is positive.
3. A coherent-but-wrong mathematical fixture retains nonzero information; documentation does not present consistency as correctness.
4. Probabilities are finite, correctly class-aligned, and sum to one; handle zero or tiny probabilities safely.
5. Hidden-value/hidden-label invariance before acquisition, including serialized responses and logs.
6. Candidate columns are sampled jointly from context rows and donor distances use observed features only.
7. No original row or block crosses partitions; utility masks preserve original-row membership.
8. Exactly 70 extracted features from the 14 named sensors; no profile label, cycle ID, or virtual feature enters X.
9. Atomic group reveals, no duplicates, no overspending, zero budget stops, no affordable action stops, stale requests rejected.
10. Termination after at most five purchases, cancellation handled, model failure distinguished from successful completion.
11. Mathematical policy fixture with known nonpositive net benefits stops. Separately evaluate the actual learned policy on the null task without asserting that it must always pass statistically.
12. Disabling model access produces explicit unavailable status, not another backend labeled TabPFN.
13. A separately marked real-model integration test records the actual checkpoint and successfully predicts a held-out row. Skipped integration is not a pass.
14. Same locked test rows and cost accounting across policy comparisons; all-panels is marked infeasible where necessary.
15. UI browser test: start session, recommend, purchase, verify revealed values and budget, stop, view completed report, inspect benchmark empty/ready state, inspect replay badge, and check console errors.

Frontend screenshots must come from the built app. Do not deliver a screenshot-only mockup as a working product.

## 18. Finish requirements

Provide exact commands that work in the implemented repository for environment setup, data preparation, model doctor/smoke test, unit tests, utility preparation, evaluation, backend startup, and frontend startup. A Python launcher can simplify local use; do not assume Unix `make` exists on Windows. Never claim a proposed CLI already exists before implementing it.

The final implementation report must state: what runs, what was tested, what actual checkpoint was used, whether empirical utility was implemented, which evaluations completed, actual performance including failures, remaining blockers, and exact launch commands. Include an annotated demo script without invented measurements.

**Done means a reproducible numerical application with a visible data boundary and truthful evidence—not a polished interface wrapped around unexecuted model calls.**
