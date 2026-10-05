# NextCheck demonstration

1. Start `./start.ps1` and open http://127.0.0.1:8000. Wait for the verified model-ready badge. The app uses actual base TabPFN 3.5. Point out that costs are simulated and the target is internal pump leakage.
2. Choose budget 6 and cost weight 0.02. Start a session with the development-selected policy. Read the displayed resolved policy and the number of selection examples. A static/prior winner is a legitimate outcome.
3. Only the five TS1 summaries should appear. Locked cards contain sensor names and prices, with no hidden readings in the API response or page.
4. For the adaptive demonstration, start a separate session with Information or Empirical value. Click Recommend. The numerical job evaluates allowed candidate outcomes and returns a ranking. If it recommends stopping, say so; do not invent a reason to purchase.
5. Open Audit. Raw KL is information plus residual. Disagreement is a diagnostic, not proof that a prediction is wrong. Distinguish predicted one-step value from realized improvement, which is only available in offline evaluation.
6. If an action is selected, click Run recommended check. Its actual panel summaries appear; the exact simulated cost is charged once; model probabilities update and may become less decisive. Click Autopilot to run the bounded loop, or Stop to finish manually.
7. The completed report can reveal the historical label. This session cannot resume acquisition after seeing it. A stop reason is a policy outcome, not a safety claim.
8. Open Benchmarks. Explain the actual run scope, class support, context/utility/selection/test counts, and frozen configuration. Compare the selected default, adaptive policies, residual ablation, static subset, prior, random mean and all-panels reference. All-panels costs 11 and is infeasible below that budget. Quote only values currently present in the artifacts; do not claim savings or superiority without the specified supporting comparison.
9. Show the null-task result in the run artifacts. A learned policy may still buy panels when labels are shuffled; this is an observed failure mode, not a result to hide.
10. Select the recorded development replay. The persistent badge must say RECORDED REPLAY. Its branch actions and budget controls are disabled. Returning to Live session creates a new interactive branch with actual inference.

`scripts/record_demo.py` records an illustrative development case and checkpoint/input/split provenance; it never picks an example from final-test outcomes. See `BUILD_REPORT.md` for measured results and executed verification.
