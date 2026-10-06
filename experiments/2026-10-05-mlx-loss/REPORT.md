> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# MLX Mac pretraining performance results

Keeping activations rather than recomputing them is the useful measured change. The 298M H1-shaped model reached **3,666 targets/s**, about **13.9% faster** than MLX 0.32.3 with the same loss/optimizer and alternating checkpointing. Peak active MLX memory increased from **11.44 GiB to 15.84 GiB**. Both rounds improved (16.9% and 11.0%). This recipe passed the registered speed/memory gate and the additional compiled full-context gradient check. No production trainer was changed.

## Full training timings

| Configuration | Targets/s | Model/backward/accumulation s/update | Optimizer s/update | Peak active GiB |
|---|---:|---:|---:|---:|
| 0.32.2, composed loss, alternating checkpointing | 3,193 | 9.676 | 0.584 | 11.44 |
| 0.32.3, composed loss, alternating checkpointing | 3,218 | 9.599 | 0.581 | 11.44 |
| 0.32.3, fused loss, alternating checkpointing (diagnostic) | 3,341 | 9.233 | 0.572 | 10.73 |
| 0.32.3, composed loss, no checkpointing | 3,666 | 8.371 | 0.566 | 15.84 |

Each arm used two fresh processes, three warmup and five measured updates per process. The table reports targets divided by mean measured update time over all ten measured updates, not the arithmetic mean of per-update throughputs. Primary order A1,B1,U1,U2,B2,A2; capacity candidates E1,E2 followed. Normal work apps remained open; run order, coarse thermal state and ordinary background load limit attribution of small differences. E was measured after the controls, so its pairing by round does not imply adjacent or randomized matched windows.

| Comparison | Round 1 speed gain | Round 2 speed gain | Median paired gain |
|---|---:|---:|---:|
| B/A | 1.55% | -0.04% | 0.76% |
| U/B | 6.67% | 0.98% | 3.83% |
| U/A | 8.33% | 0.94% | 4.64% |
| E/B | 16.87% | 10.97% | 13.92% |
| E/A | 18.68% | 10.93% | 14.81% |

E uses 4.40 GiB more active memory (+38.5%) than B. Its throughput gain corresponds to about 12.2% less training-step time for a fixed token budget in this benchmark. Relative to the old A environment, E is about 14.8% faster. The library-only upgrade is below 1% overall and the second pair is flat.

## What the new Metal loss did

MLX 0.32.3 includes fused Metal cross-entropy forward/backward kernels ([official release](https://github.com/ml-explore/mlx/releases/tag/v0.32.3), [implementation](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/cross_entropy.cpp)). The isolated compiled loss-plus-logit-gradient benchmark at 4095 × 32768 measured 20.603 ms composed versus 3.691 ms fused: **5.58×** faster. It excludes the vocabulary projection and the rest of the model. At 1024 × 32768 the ratio was 3.01×; at 128 × 1024 it was 1.37×.

In full H1 training, fused versus composed loss improved by 6.67% in one round and 0.98% in the other (median paired 3.83%). Peak active memory fell by roughly 0.72GiB. The modest full-training gain illustrates that a fast loss kernel does not remove the transformer matrix/backward work.

## Numerical results and decisions

All 12 standalone eager/compiled loss cases passed the registered loss and gradient limits, including extreme logits, shared offsets, tied maxima and nonuniform cotangents; small cases also passed an independent NumPy FP64 oracle. Worst composed/fused relative gradient L2 was 1.56e-7.

The original full-model BF16 composed/fused gradient check at context 64 failed: 0.8071% relative L2 versus the registered 0.5% limit, with equal loss. Repeated runs of each implementation were identical. On a full FP32 model, the composed/fused relative L2 was 6.38e-7. At actual 4095-token context, compiled BF16 gradients differed by 0.6537%, again exceeding the unchanged limit. These results are consistent with tiny arithmetic differences amplified by mixed precision. This is an inference; real-data learning parity was not established. The original failed gate remains failed, and fused H1 timings are diagnostic. No fused-loss default is recommended from this campaign.

For the recommended capacity candidate, context 64 checkpointed/noncheckpointed composed-loss gradients and loss matched exactly. At actual 4095-token context, compiled gradients had relative L2 **5.85e-6 (0.000585%)**, well inside the 0.5% limit; loss matched exactly. Every parameter gradient was finite and nonzero. All 64 full-size training updates produced finite loss/norm/state and actual master-weight changes. Initialization and input hashes matched across all eight runs.

The useful next Mac recipe is the tested composed FP32 loss with activation checkpointing disabled for this H1 shape, a 32 GiB process allocation cap and 1 GiB cache cap. It retains BF16 model/backward and FP32 masters/accumulation/state. The normal 0.32.3 loss wrapper selects the fused path; reproducing E therefore requires the explicit composed helper in `scripts/worker.py`, not an automatic package upgrade alone. This is an isolated synthetic recipe recommendation, not a production deployment or learning-quality result.

## Fidelity and limits

298,361,856 parameters; 20 blocks; d1024/h3456; GQA 8/2, head 128; tied vocab 32768; RoPE 500000. Microbatch 1, accumulation 8, 4095 input positions and 32,760 targets/update. Fixed NumPy synthetic sequences, native fresh seed 42 initialization, clipping 1 and explicit LRs .00015/.00006. All arms use the immutable prior Mac Muon/AdamW implementation (independently grouped matrices, zero decay and its existing momentum/bias correction). Those optimizer equations differ from Fedora production; this study repeats the Mac recipe and is not a new CUDA comparison. Timings synchronize microbatches and optimizer updates. Initialization, compilation warmup, correctness scans and file logging are excluded from steady update throughput. No real data, held-out quality test, checkpoint-resume test, Windows workload, Fedora production read, or protected bank was involved.

## Safety and preservation

Mac remained on AC. macOS thermal state ranged nominal/fair (0/1), with no serious/critical state. Numerical temperature sensors were unavailable. Minimum recorded memory-pressure free percentage was 83%; available memory was checked periodically. Maximum recorded worker thermal-monitor interval was 1.587s. Aggregate worker wall time was 660.06s of the 900s cap; benchmark campaign elapsed 1349.84s of 1800s. No research GPU process remains. Original MLX 0.32.2 environment and all existing freezes are intact. Candidate 0.32.3 lives in the separate `.venv-mlx0323` environment.

The v1 gradient-gate failure and a diagnostic launcher preflight error are preserved with their incident notes. The launcher error started no GPU work; its replacement retained normal subprocess behavior and the original safety monitor. No frozen source was edited.

## Records

- `SUMMARY_V2.json`: registered descriptive arithmetic and paired results.
- `FINAL_VERIFICATION.json`: final numerical, safety, budget, process and freeze audit.
- `metrics/timing_rows.json`: all 64 update timing rows without weight/state fragments.
- `metrics/correctness.json`, `gradient_diagnostic.json`, `checkpointing_gate.json`, `full_context_validation.json`: preserved numerical evidence.
- `metrics/*events.jsonl`: exact worker commands and independent monitor records.
- Raw records containing diagnostic weight/state samples are excluded from Git and checksum-backed on SHUTTLE; paths/hashes are in `RAW_RECORD_MANIFEST.json`.
- `PLAN.md`, `EXPLORATORY_PLAN.md`, `AMENDMENT_V2.md` and the freezes record the registered scope and the distinction between rejected and descriptive arms.
