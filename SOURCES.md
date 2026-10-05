# Sources and attribution

Accessed 4 October 2026 (America/Chicago). These sources establish APIs, dataset meaning, and prior work; they are not evidence of IRONTRACE performance.

- **TabPFN official implementation:** https://github.com/PriorLabs/TabPFN — package API and missing-value handling.
- **TabPFN 3.5 model card:** https://huggingface.co/Prior-Labs/tabpfn_3_5 — explicitly selected base checkpoint `tabpfn-v3.5-20260909.safetensors`. Model weights have separate noncommercial terms. Existing local authorized cache was used; weights are not included. Contest/commercial permission is not asserted.
- **UCI dataset 447:** https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems — Helwig, N., Pignanelli, E., and Schütze, A. (2015), *Condition monitoring of hydraulic systems*, DOI https://doi.org/10.24432/C5CW21. CC BY 4.0. Sensor frequencies, cycle labels, and raw files derive from this source. IRONTRACE transforms raw cycles into five row-local summary statistics and simulates partial observations and artificial measurement costs.
- **Learning-To-Measure:** Kobayashi et al., ICML 2026, https://proceedings.mlr.press/v306/kobayashi26a.html — prior work on in-context active feature acquisition.
- **AFABench:** Schütz et al., https://arxiv.org/abs/2508.14734 and https://github.com/Linusaronsson/AFA-Benchmark — prior work on systematic active feature acquisition comparisons.

Acquisition, mutual information, KL decomposition, and stopping are established ideas. IRONTRACE tests an application-specific audit-aware empirical value model; any advantage must be measured on the frozen test split. Pipeline disagreement is not proof of model error, and native NaN handling is not asserted to be exact Bayesian marginalization.
