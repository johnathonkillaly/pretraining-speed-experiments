# Failed, negative, corrected and unrun experiments

This index keeps a failed numerical gate separate from an inconclusive throughput
result, a tooling error, an exploratory success and an experiment that never ran.
Original thresholds remain unchanged. A correction is not permission to erase a
failed version or present a rejected timing as a validated optimization.

| Campaign | Observation | Recorded outcome / authoritative evidence |
|---|---|---|
| MoE | Initial Top-1 smoke attempted VJP through integer gather indices. | Failed before the main fits; discrete Top-K indices were detached while gate probabilities remained differentiable. [Report](experiments/2026-10-04/REPORT.md#failed-experiments). |
| MoE | Initial memory audit retained the loss's forward graph, inflating the gradient-storage delta. | Original `benchmark_primary.jsonl` is superseded for the storage conclusion; v2 releases the graph and adds a gradient-only measure. Use `benchmark_primary_v2.jsonl` and the report. The sweep and corrected primary total 106 cases. |
| MoE graph export | DOT export occurred after evaluation and showed only leaves. | Repaired to export unevaluated graphs outside timing windows. Graph exports are not included in this public archive. |
| MoE | Independent expert modules still returned full-shaped gradients for unused experts; full-tree optimizer state stayed dense. | This defeated the hoped-for ordinary-autodiff memory saving. The fixed-route custom VJP is an exploratory exception with substantial limits, not a full dynamic MoE trainer. |
| MoE learning | Top-1/Top-2 were slower than the comparable active-compute dense pilot and did not establish better convergence. | Nine short fits; sparse routing concentration samples are reported. This does not establish that MoE is universally ineffective. |
| Optimizers | Applying stock MLX Muon to stacked `[experts, rows, cols]` weights orthogonalized experts together. | Invalid substitute for independent per-expert Muon. Changing one expert affected another; the report uses separately validated per-expert batching. [Report](experiments/2026-10-04-optimizers/REPORT.md#muon-has-an-expert-layout-trap). |
| Optimizers | Stock MLX 0.32.2 SGD allocated velocity state even with zero momentum. | Allocation-only stateless control matched three weight updates exactly and omitted that state; no new learning test was run. |
| Recurrent learning | Short R1→R2→R4 depth curriculum did not improve fixed R4/K2 truncation. | Final test loss was higher by 0.0279 nats/token. Both used the same bounded tuning procedure. [Report](experiments/2026-10-04-recurrence/REPORT.md). |
| Recurrence preparation | An initial corpus slice could not supply the requested blocks; original H1 compiled probe failed before a completed update. | Slice selection was corrected before the main freeze; a separate H1 capture amendment preserved the failed source/record. The primary fits completed. [Incidents](experiments/2026-10-04-recurrence/REPORT.md#incidents-limits-and-reproducibility). |
| Windows CUDA v1 | Forced Flash Attention was unavailable in the native-Windows PyTorch environment. | CPU correctness passed, then the campaign stopped before benchmark timing. Frozen v2 audited and forced efficient attention instead. v1 failure is retained in `returned/measurements/v1/results.json`. |
| Windows profiling v3 | First profiler summary counted five GPU annotation ranges as leaf kernels. | v3 kernel totals are superseded. Use v4, which separates compute kernels and memory operations; headline timing remains v2. [Report](experiments/2026-10-05-windows/returned/REPORT.md#optimizer-profile). |
| Windows monitoring | A supplemental v3 transition had a 1.141s sample gap against a one-second requirement. | Disclosed cadence deviation; benchmark windows and corrected v4 profiling had smaller gaps. No full-campaign strict cadence compliance is claimed. |
| MLX library upgrade | 0.32.3 versus 0.32.2 was +1.55% then −0.04% in the paired full-model runs. | Below 1% overall; no material repeatable library-only speed gain established. [Report](experiments/2026-10-05-mlx-loss/REPORT.md). |
| MLX fused loss | Standalone loss/gradient kernels passed, but full BF16 model gradients failed the ≤0.5% relative-L2 limit. | Context 64: 0.8071%; actual context: 0.6537%. Full-training fused timings remain diagnostic, despite the isolated kernel's 5.58× ratio. FP32 diagnostics do not replace the failed BF16 gate. |
| MLX diagnostic launcher | A diagnostic preflight errored before GPU work. | Preserved separately; replacement retained ordinary subprocess behavior and safety checks. [Incident](experiments/2026-10-05-mlx-loss/INCIDENT_DIAGNOSTIC_LAUNCH.md). |
| MLX larger batches | B2/B4 failed short-context loss and gradient matching. | Absolute loss difference ≈0.000357 >0.0002; relative gradient L2 ≈1.18% >0.5%. Their actual 4095-position checks passed, but both contexts were required. Timings remain diagnostic. |
| MLX larger batches | B2/B4 did not reach a 5% gain in both rounds. | B2: +2.02%, −0.15%; B4: +5.50%, −2.85%. Baseline drift exceeded candidate gains. No accepted larger-batch optimization. [Report](experiments/2026-10-05-mlx-batching/REPORT.md). |
| MLX B8 capacity | Conservative forecast ≈94.97GiB exceeded the registered 76GiB forecast threshold within an 80GiB allowance. | Full-context batch 8 **not attempted**. This is not an observed OOM or a measured batch-8 peak. Its short-context numerical gate also failed. |
| Inventory privacy | Initial interactive process listing exposed a background worker credential in private tool output. | Credential/raw listing was not saved to research results or this archive. Later inventories omitted arbitrary arguments. [Incident without credential content](experiments/2026-10-05-mlx-batching/PREFLIGHT_INCIDENT.md). |

## Proposals that did not become results

- Default MoE and ProbMoE were reviewed, not trained as faithful replications.
- BitNet/ternary weights were discussed, then excluded from the training work.
- The recurrent clip-1 versus clip-5 fresh-data learning ablation remains deferred.
- The proposed 298M H1 RTX 4070 Laptop comparison was paused before runner execution. [Paused status](experiments/2026-10-05-h1-laptop/STATUS.md).
- No combined Mac/NVIDIA network-pretraining benchmark or laptop undervolting result exists in this archive.
- The historical RTX 3080 figure mentioned in the recurrence report came from another pipeline/data configuration. It is context, not a controlled hardware comparison.

## Results that did clear their tested gates

Preserving failures does not mean every trial failed. The recurrent truncation
pilots completed intended-gradient, finite-state and checkpoint checks. The
checkpointing-disabled composed-loss H1 recipe passed the tested numerical gates
and improved in both timing rounds. Neither establishes long-run training quality
at production scale. Their scope and remaining confounds stay in the reports.
