# Pretraining speed experiments: MLX, MoE, Muon and recurrent backpropagation

Experiments by **Johnathon Killaly**, run October 4–5, 2026, published October 6.
AI-assisted implementation, execution and reporting used Codex. This repository
publishes the measurements, methods, negative results and corrections from a
local investigation of faster language-model pretraining. It is an evidence
archive, with historical benchmark sources; it does not ship a pretrained model.

**The clearest measured gains were truncated backpropagation in a small recurrent
model and disabling activation checkpointing in the 298M MLX benchmark. Larger
physical batches did not produce a repeatable gain. The tested MoE implementations
did not establish a pretraining advantage over a comparable dense pilot.**

## Experiments and outcomes

The Mac was an Apple M4 Max with 128GiB unified memory. Mac experiments used MLX
0.32.2 or isolated MLX 0.32.3. The Windows experiment used an RTX 4070 Laptop GPU
with 8GiB VRAM, native Windows, PyTorch 2.7.0+cu128 and eager CUDA.

| Experiment | What ran | Measured outcome | Failure or limitation |
|---|---|---|---|
| [Sparse MoE on MLX](experiments/2026-10-04/REPORT.md) | 106 corrected FFN microbenchmarks; nine equal-token dense/Top-1/Top-2 fits across three seeds | Dense pilot: 39,712 targets/s; Top-1: 33,622; Top-2: 33,017. Ordinary expert gradients and AdamW state stayed dense. | No convincing learning winner; Top-1 showed intermittent routing concentration. Sparse arithmetic alone did not yield a full-training win. |
| [Muon and other optimizers](experiments/2026-10-04-optimizers/REPORT.md) | 37 optimizer state/update microbenchmarks | Muon/Lion used one weight-sized state buffer versus AdamW's two; no-first-moment Adafactor used much less factored state. | Ordinary gradient storage stayed dense. Faster/smaller optimizer updates were not optimizer convergence tests. Stock stacked Muon coupled experts. |
| [Recurrent depth with truncated backpropagation](experiments/2026-10-04-recurrence/REPORT.md) | 15 primary fits, five variants × three seeds, 10.24M targets per fit, plus a bounded LR sweep | Last-two-loop backpropagation over four forward loops: **18.0% higher steady throughput**, 15.2% less training time than full recurrence; test loss lower by 0.0374 nats/token. | Short, undertrained pilot. Clipping activity changed substantially. The short depth curriculum did not improve fixed truncation. |
| [RTX 4070 Laptop CUDA recurrence](experiments/2026-10-05-windows/returned/REPORT.md) | Twelve 10-second synthetic timing windows after correctness gates | TB2: **9.75% median paired throughput gain** and 21.04% median paired backward reduction over full recurrence. | Forced Flash Attention failed before timing; amended efficient attention ran. Synthetic timing does not establish learning quality or a matched Mac/CUDA ratio. |
| [MLX 0.32.3 loss and activation memory](experiments/2026-10-05-mlx-loss/REPORT.md) | Eight short full-model timing runs at 298,361,856 parameters / 4K context; standalone loss checks and full gradients | Composed FP32 loss with checkpointing disabled: **3,666 targets/s**, about **13.9% faster**, using 15.84GiB rather than 11.44GiB. | Library-only upgrade was below 1% overall. Fused loss failed full-model BF16 gradient gates; its faster timings remain diagnostic. |
| [Larger MLX sequence batches / 80GiB allowance](experiments/2026-10-05-mlx-batching/REPORT.md) | Six timing runs; batches 1/2/4 with accumulation 8/4/2, same eight sequences and 32,760 targets per update | Aggregate 3,578 / 3,613 / 3,626 targets/s; peak active memory 15.84 / 26.81 / 49.16GiB. | Neither larger batch achieved a repeated 5% gain. B2/B4 failed the short-context numerical gate. Full-context B8 was skipped by a ~95GiB capacity forecast. |

The first two experiments used small corpus-based learning pilots or optimizer
microbenchmarks; recurrence used the older v7 corpus mix. The H1 loss/batching and
Windows timings used fixed synthetic inputs. Absolute rates across different
geometries, data, optimizers or backends are not an apples-to-apples hardware test.

## The larger-batch result, including both rounds

The 298M batching sweep was intended to test whether spare unified memory could
improve GPU utilization without increasing the effective batch size.

| Sequences processed together | Accumulation passes | Round 1 gain vs batch 1 | Round 2 gain vs batch 1 | Peak active GiB |
|---|---:|---:|---:|---:|
| 1 | 8 | Reference | Reference | 15.84 |
| 2 | 4 | +2.02% | −0.15% | 26.81 |
| 4 | 2 | +5.50% | −2.85% | 49.16 |

![MLX sequence batching throughput and memory](experiments/2026-10-05-mlx-batching/comparison.png)

The baseline moved by 8.67% between rounds, larger than the candidate gains.
Batch 4's favorable first round therefore did not establish a repeatable speedup.
Batches 2 and 4 passed gradient matching at the actual 4095-position context but
failed the separately required 64-position check. These speed records remain
diagnostic. No tolerance was relaxed. Batch 8 was a forecast-based capacity skip,
not an observed OOM. An 80GiB allowance was a capacity budget, not RAM allocated
or filled by every run.

## Negative results and corrections are first-class evidence

Read [FAILURES.md](FAILURES.md) for failed numerical gates, invalid initial timing
accounting, unsupported attention, corrected profiling and unrun proposals. The
original failures and superseded measurements remain in the experiment folders.
Prefer the authoritative versions identified in that index.

An exploratory fixed-route custom VJP reduced physical gradient storage eightfold
by using broadcast zeros for unused experts. It did **not** reduce AdamW state,
implement a dynamic router, or demonstrate learning/scaling benefits. Similarly,
Muon's smaller state is a memory result, not proof of faster full pretraining.

Truncating recurrent backpropagation intentionally changes the gradient while
preserving forward depth. Numerical checks validated that intended operation;
they do not claim that truncated gradients equal full-backprop gradients.

## Reading, checking and reproducing

- Start with the linked reports, then the registered plans/contracts and numerical records.
- Scalar JSON/JSONL/CSV measurements and scientific plots are included. Datasets, tokenizer assets, weights and checkpoint tensor payloads are not distributed.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) describes what the public snapshot can verify and what a new training run would require.
- `SOURCE_MANIFEST.json` records original and published SHA-256 digests for exported artifacts. Public paths/process metadata were sanitized; original private freeze maps do not validate this changed tree.
- `PUBLICATION_SHA256.json` checksums the published tree. Run the CPU-only verifier:

```sh
python3 tools/verify_archive.py
```

The original local frozen experiments and Python environments were preserved.
Publication performed no additional model training, GPU benchmark, system tuning,
undervolting or production change.

## Attribution and citation

The methods reference [MLX](https://github.com/ml-explore/mlx), the
[Huginn recurrent-depth paper](https://arxiv.org/abs/2502.05171),
[Adafactor](https://arxiv.org/abs/1804.04235), and the optimizer/dispatch references
linked in each report. This is an independent local investigation, not an
official benchmark or a faithful Huginn replication. Downloaded third-party
source-reference copies and datasets are omitted; no new license clearance for
those materials is claimed.

Use GitHub's **Cite this repository** entry or [CITATION.cff](CITATION.cff).
Public availability makes these findings readable and linkable; search-engine or
AI-search indexing is controlled by those services and is not guaranteed.
