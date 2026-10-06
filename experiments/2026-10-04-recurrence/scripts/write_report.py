"""Render measured tables and fixed methods after the campaign; no model evaluation."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("dense10", "dense4", "rec_full", "rec_tb2", "rec_curr_tb2")
LABELS = {"dense10": "Dense10", "dense4": "Dense4", "rec_full": "Recurrent full",
          "rec_tb2": "Recurrent last2", "rec_curr_tb2": "Last2 + curriculum"}


def prose_spacing(document):
    """Readable numerical prose while preserving fenced commands and inline identifiers."""
    output, fenced = [], False
    for line in document.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            pieces = re.split(r"(`[^`]*`)", line)
            for index in range(0, len(pieces), 2):
                pieces[index] = re.sub(r"\b([A-Za-z]{3,})(?=\d)", r"\1 ", pieces[index])
                pieces[index] = re.sub(r"\b(on|by|is|at|of|as|in|to|vs)(?=\d)", r"\1 ", pieces[index])
                pieces[index] = re.sub(r"(?<=\d)(?=[A-Za-z]{3,}\b)", " ", pieces[index])
                pieces[index] = re.sub(r"('s)(?=\d)", r"\1 ", pieces[index])
                pieces[index] = re.sub(r"(?<!\d),(?=\d+\.)", ", ", pieces[index])
            line = "".join(pieces)
        output.append(line)
    return "".join(output)


def markdown(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def mean_sd(row, factor=1, places=3):
    return f"{row['mean'] / factor:.{places}f} ± {row['sample_sd'] / factor:.{places}f}"


def main():
    a = json.loads((ROOT / "metrics/analysis.json").read_text()); t = a["summary"]
    assert json.loads((ROOT / "metrics/checkpoint_checks.json").read_text())["passed"]
    assert json.loads((ROOT / "metrics/reload_checks.json").read_text())["passed"]
    assert json.loads((ROOT / "metrics/freeze_audit.json").read_text())["passed"]
    selected = json.loads((ROOT / "metrics/selected_rates.json").read_text())
    c = a["paired_test_comparisons_exploratory"]["rec_tb2_minus_rec_full"]
    speed = 100 * (t["rec_tb2"]["steady_tokens_per_second"]["mean"] / t["rec_full"]["steady_tokens_per_second"]["mean"] - 1)
    saved = 100 * (1 - t["rec_tb2"]["training_seconds_including_compile"]["mean"] / t["rec_full"]["training_seconds_including_compile"]["mean"])
    quality = c["mean_difference_nats"]
    b = a["backward_medians"]
    bw_saved = 100 * (1 - b["rec_tb2"]["backward_ms"] / b["rec_full"]["backward_ms"])
    fwbw_saved = 100 * (1 - b["rec_tb2"]["total_ms"] / b["rec_full"]["total_ms"])
    primary = markdown(["Variant", "Unique params", "Steady k targets/s", "Train sec incl compile", "Process sec", "Peak active GiB", "Final test loss"],
                       [[LABELS[k], f"{t[k]['parameter_count']/1e6:.3f}M",
                         mean_sd(t[k]["steady_tokens_per_second"], 1000, 2),
                         mean_sd(t[k]["training_seconds_including_compile"], 1, 1),
                         mean_sd(t[k]["process_elapsed_seconds"], 1, 1),
                         mean_sd(t[k]["peak_active_bytes"], 2**30, 3),
                         mean_sd(t[k]["final_test_loss"])] for k in KINDS])
    dev = markdown(["Variant", "Final dev loss", "Dev at common training time", "Dev at common elapsed checkpoint time"],
                   [[LABELS[k], mean_sd(t[k]["final_development_loss"]),
                     mean_sd(a["equal_training_time_development"]["aggregate_development_loss"][k]),
                     mean_sd(a["equal_elapsed_time_development"]["aggregate_development_loss"][k])] for k in KINDS])
    pair = markdown(["Candidate − reference", "Final test Δ nats", "Paired seed deltas", "Exploratory bootstrap95%"],
                    [[name.replace("_minus_", " − "), f"{r['mean_difference_nats']:+.4f}",
                      ", ".join(f"{x:+.4f}" for x in r["paired_seed_differences"]),
                      "[" + ", ".join(f"{x:+.4f}" for x in r["exploratory_bootstrap_95_interval"]) + "]"]
                     for name, r in a["paired_test_comparisons_exploratory"].items()])
    micro = markdown(["Variant", "Forward ms", "Backward ms", "Fwd+bwd ms", "Incremental peak GiB", "Gradient logical/retained MiB"],
                     [[LABELS[k], f"{r['forward_ms']:.2f}", f"{r['backward_ms']:.2f}", f"{r['total_ms']:.2f}",
                       f"{r['peak_incremental_bytes']/2**30:.3f}",
                       f"{r['gradient_logical_bytes']/2**20:.2f} / {r['gradient_physical_retained_bytes']/2**20:.2f}"] for k, r in b.items()])
    tuning = markdown(["Variant", "Muon .003 dev", "Muon .01 dev", "Muon .02 dev", "Chosen peak"],
                      [[LABELS[k]] + [f"{next(r['development_loss'] for r in selected[k]['trials'] if r['muon_peak']==rate):.4f}" for rate in (.003, .01, .02)]
                       + [str(selected[k]["muon_peak"])] for k in KINDS])
    phase = markdown(["Variant", "Dev validation sec", "Final test sec", "Checkpoint sec", "Other timed-fit sec"],
                     [[LABELS[k]] + [f"{sum(a['fit_phase_times'][f'{k}_s{s}'][field] for s in (42,43,44))/3:.3f}"
                                     for field in ("development_validation_seconds", "final_test_seconds", "checkpoint_seconds", "remaining_python_data_logging_seconds")]
                      for k in KINDS])
    signatures = markdown(["Variant", "R / K", "First-call ms incl compile", "Steady median ms", "Steady calls"],
                          [[LABELS[k], f"{r['loops']} / {r['backprop_loops']}", f"{r['first_call_ms_including_compile']:.2f}",
                            f"{r['steady_step_ms']['median']:.2f}", r["steady_calls"]]
                           for k in KINDS for r in a["fit_phase_times"][f"{k}_s42"]["signatures"]])
    sources = markdown(["Final-test source", "Docs"] + [LABELS[k] for k in KINDS],
                       [[source, r["documents"]] + [mean_sd(r["loss"][k]) for k in KINDS] for source, r in a["test_source_profile"].items()])
    proxy = markdown(["Variant", "Forward block executions", "Differentiated blocks", "Projection FLOP proxy / target"],
                     [[LABELS[k], r["mean_executed_forward_blocks"], r["mean_differentiated_blocks"],
                       f"{r['total_flops']/1e6:.2f}M"] for k, r in a["projection_flop_proxy_explanatory"]["per_prediction_target"].items()])
    norms = markdown(["Variant", "Mean unclipped gradient norm", "Updates with norm>1"],
                     [[LABELS[k], f"{sum(a['fit_phase_times'][f'{k}_s{s}']['gradient_norm_mean'] for s in (42,43,44))/3:.3f}",
                       f"{100*sum(a['fit_phase_times'][f'{k}_s{s}']['fraction_gradient_norm_above_one'] for s in (42,43,44))/3:.2f}%"] for k in KINDS])
    times = ", ".join(f"seed{s}: {r['seconds']:.1f}s" for s, r in a["equal_training_time_development"]["per_seed"].items())
    interpretation = (ROOT / "INTERPRETATION.md").read_text()
    report = f'''# Recurrent Depth with Truncated Backpropagation

## Executive answer

Completed **15 primary fits**, three seeds per variant, **10.24M prediction
targets per fit**; **153.6M primary targets** in total. All final checkpoints and
frozen inputs passed integrity checks. No production code, data or job changed.
BitNet was excluded as requested.

Backpropagating through the last two of four shared-core loops delivered
**{speed:.1f}% higher steady training throughput** than full recurrence and
**{saved:.1f}% less measured training time** at the same token budget. Its final
test difference was **{quality:+.4f} nats/token** versus full recurrence; positive
means higher loss. This is the direct registered truncation comparison.

The separated eager audit found **{bw_saved:.1f}% less backward time** and
**{fwbw_saved:.1f}% less forward-plus-backward time**. Truncation preserved all
forward loops and the unique gradient tensor sizes. It reduced activation/backward
cost rather than creating sparse recurrent-weight gradients.

Fixed-depth truncation also has the lowest final test loss of all five variants,
and the lowest development loss at the common measured training-time endpoint in
all three seeds. The curriculum did not improve it. **CONTINUE with a registered
follow-up**, including a clipping ablation before extrapolating its quality gain.
This is a short native-pretraining pilot on an old v7 mix, not a capability test
or evidence for changing the existing298M model.

## Registered learning comparison

All cells are mean ± sample SD across seeds42/43/44. Throughput includes the
compiled forward, backward, clipping and optimizer update; first50 calls of each
compiled depth path are excluded only from steady timing. Training time includes
those calls and compilation. Process time comes from controller start/end events
and includes model/data initialization, evaluation and saving. Peak is MLX active
allocation, not total macOS RAM or a cross-backend memory metric.

{primary}

![Final loss and throughput]({ROOT / 'plots/final_comparison.png'})

{dev}

Common training-time endpoints are the shortest completed training duration within
each seed: {times}. Development losses there are linearly interpolated between
logged evaluations. The separate elapsed-checkpoint comparison includes validation
and Python overhead. Neither table is an additional test evaluation or a matched
wall-budget retraining: the cosine schedule is defined by tokens, so the faster
run reaches its cooldown sooner. Evaluation spacing is500updates (1.024M targets),
except additional curriculum transitions. Avoid precise time-to-target claims from
this interpolation.

![Learning curves against tokens and measured time]({ROOT / 'plots/learning_curves.png'})

### Paired final-test differences — exploratory uncertainty analysis

{pair}

Both variants in each comparison see the same512 test documents and paired seeds.
Bootstrap4000 resamples seed and document axes separately, preserving pairing.
These intervals have only three independent seeds; document count does not imply
1536 independent training runs. Domain/source breakdown is descriptive, not a
new selection criterion. Development data selected LRs; final test was evaluated
once per primary model, only after training.

{norms}

The same global clipping rule is applied to every variant. Truncation changes the
gradient, and may change clipping activity as part of that recipe. This pilot does
not separately isolate gradient bias, regularization or clipping effects on loss.

{sources}

## What the backward and memory checks show

{micro}

These are medians of15 synchronized eager windows after one first call plus three
warmups, BF16 batch8×context256. Forward and backward are evaluated separately;
they exclude the optimizer and cannot be substituted for compiled training timing.
Peak is above already-resident model/input memory. Retained gradient memory is
measured after dropping the scalar loss; raw allocation variations are retained.

![Separated backward cost and gradient storage]({ROOT / 'plots/backward_cost.png'})

Full and truncated recurrence share11,700,480 unique parameters, versus16,667,904
for Dense10 and11,700,480 for Dense4. Sharing cuts unique parameters by29.8% versus
Dense10, but fixed four-loop recurrence still executes ten transformer blocks.
The tied32K embedding/head contributes8,388,608 parameters to every model, limiting
the total compression and contributing substantial common vocabulary-head work.
Muon's per-matrix normalization is independent despite batching equal shapes.

{proxy}

This explanatory arithmetic proxy counts two FLOPs per multiply-add in linear
projections and both input/weight products in backward; it excludes attention
score/value products, softmax/CE, norms, nonlinearities, injection, optimizer and
runtime overhead. Curriculum rows average over the registered5%/5%/90% phases.
The common vocabulary head contributes50.33M projection FLOPs/target across its
forward and backward, over half the full-recurrent proxy. Removing half the core's
backward loops therefore does not halve whole-model training computation.

Numerical verification passed: full vs K≥R gradients equal; full/truncated forwards
identical; one-loop recurrence identical to Dense4; toy prefix-loop gradient
contributions zero before the detach; conditional finite differences with prefix
held fixed agree; truncated prelude/embedding/core/coda gradients remain nonzero;
batched/separate Newton–Schulz results and compiled/eager one-update masters agree.
See `metrics/verification.json` for raw values.

All15 checkpoints were checked using CPU-only parsing: 5000 updates/10.24M targets,
finite BF16 weights and FP32 masters/state, exact BF16 rounding of final masters,
matching shapes/optimizer ownership, and SHA-256 for every checkpoint file.
`metrics/checkpoint_checks.json` and `metrics/freeze_audit.json` retain the checks.
Every saved model was also reloaded and its full development loss reproduced to
within2e-6 nats; no final-test data was evaluated again. See `metrics/reload_checks.json`.

## Architecture, data and recipe

Each model uses d256, SwiGLU h864, GQA8/2, head32, context256, vocab32768, RoPE500000,
RMSNorm, tied embedding/head and BF16 compute/gradients. FP32 master weights,
optimizer state and post-backward gradient clipping; no accumulation in the pilot.
Same-seed prototype bank initializes shared embedding/norm and blocks0/1/2/9;
their hashes and the token-order hashes match across all five variants within seed.

Recurrent models have one prelude, a two-block shared core and one coda. At every
loop the input is `0.5*(state+prelude)`, initially state=prelude. RecFull runs R4/K4;
RecLast2 runs R4/K2, detaching prefix state immediately before the third core loop.
Prelude injection preserves its gradient path. The curriculum runs R1/K1 for the
first5% of updates, R2/K2 for the next5%, then R4/K2. All recurrent evaluation uses
R4. This is a minimal architecture inspired by the prelude/core/coda and truncated
gradient ideas in [Huginn](https://arxiv.org/html/2502.05171v1), not a faithful
replication, adaptive halting system or pretrained-model retrofit.

Hybrid Muon on named hidden attention/MLP matrices; AdamW on embedding/head/norms.
Muon momentum 0.95, Nesterov, five BF16 Newton–Schulz iterations, FP32 independent
matrix normalization, shape batches up to5 and LR scale sqrt(max(1,rows/cols)).
AdamW betas (0.9, 0.95), bias correction on. Both branches have zero decay and share
5% warmup/cosine-to0.1 multiplier, with independently specified peak rates.
This is the hybrid recipe family, not the production model's exact LR ratio.

Equal bounded tuning: 600updates/1.2288M targets for each of three Muon peaks per
variant, seed314, AdamW peak0.0003;15 fits/18.432M tuning targets. Choose lowest
final development loss, ties lower Muon peak. All chose0.01. A600-step sweep is
short and does not establish globally optimal architecture-specific rates.

{tuning}

Training contains40,000 distinct257-token windows from16 evenly spaced complete
v7 packed shards, excluding all old MoE-pilot blocks; no cycling within a primary
fit. Within each seed all models use identical ordering. Development reuses the
old128 windows. Final test samples512 new documents, excluding the old256
candidate document indices, with256 non-PAD targets each. Exact train/dev/test
window overlap and old-block reuse are both zero. No new semantic deduplication;
packed-window evaluation and truncated standalone document prefixes have different
distributions. This is the older v7 general mix, not current Fedora v2 data.
Hashes, provenance and source/archive limitations are in `configs/data.json` and
`configs/source_catalog.json`. Upstream remote commits/contribution-specific
license versions were not preserved by the archived corpus; no new clearance is
claimed. Sampled data and all weights remain outside Git on SHUTTLE.

### Timing accounting

{phase}

Times are means across seeds. Other timed-fit time includes batch preparation,
Python/logging and calls outside the timed compiled update. The summary's fit
timer begins before initial development evaluation, after constructing the model;
controller process time also includes construction/startup. Compile is not isolated
by a profiler: first-call latency is retained, and all call times contribute to
learning-per-time curves. Seed42 path timing:

{signatures}

Dense paths ignore R/K values; they execute10 or4 distinct blocks. Curriculum
steady aggregate includes shorter early paths; its final R4 path is separately
logged. Raw times for every update and compiler signature are retained.
Total primary measured training was {a['primary_training_seconds_total']/60:.1f}min;
sum of primary process durations was {a['primary_process_seconds_total']/60:.1f}min.
These exclude tuning, optional probes and experiment preparation.

## Apple versus NVIDIA for this pretraining workload

The new298,361,856-parameter H1-shaped Mac probe reached **3,263 targets/s**, versus
the historical RTX3080 CUDA90-update record of **15,522.91/s**—about **4.76×** higher
pipeline throughput. Dimensions,32,760 targets/update, hybrid optimizer family,
precision and alternating checkpointing match. Implementations, input data and
dates differ; this is not a controlled silicon/framework or power comparison.
Mac peak MLX active allocation was11.44GiB. No NVIDIA production run was queried
or changed. Details and all qualifications: [hardware context]({ROOT / 'HARDWARE_CONTEXT.md'}).

![H1-shaped pipeline comparison]({ROOT / 'plots/hardware_context.png'})

This supports the existing CUDA pipeline for the owner's long298M pretraining
workload that fits the3080's VRAM. The128GB Mac remains useful for local research
and memory capacity. Capacity does not itself improve training arithmetic.
Ampere supports BF16 Tensor Core matrix operations with FP32 accumulation
([NVIDIA guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)); the
benchmark does not attribute the whole gap to that feature. MLX also has a CUDA
backend ([official installation docs](https://ml-explore.github.io/mlx/build/html/install.html)),
so a result about this Apple/MLX pipeline is not a universal MLX limitation.

Local code-path inspection confirms fast causal MLX attention is requested.
Installed MLX0.32.2 enables its fused cross-entropy path only on CUDA; Metal uses
the composed max/gather/logsumexp path with explicitly FP32 logits here for loss
accuracy. This identifies a software difference worth profiling, without measuring
its cost or promising a gain. See `configs/runtime_kernel_audit.json`; no additional
kernel experiment or hot-path change was made.

## Incidents, limits and reproducibility

The initial unfrozen data-preparation attempt selected the final short packed
shard, which could not supply2500 eligible blocks. It failed before dataset output.
Only empty directories belonging to that attempt were removed; complete-shard
selection was corrected before freezing. The corrected slice is the only training
data used. Original corpora/checkpoints/uncommitted work stayed read-only.
An earlier setup import-path collision selected the previous study's `train.py`;
the path ordering was fixed before source freezing and numerical verification.
No substantive fit ran under that import error.

The original optional H1 checkpointed compile failed before its first completed
window. A separately frozen v2 captures mutable model state as both compile input
and output; original source and failure remain. See `AMENDMENT_H1_CAPTURE.md`.

The original durable launchd job used Background/nice10 scheduling and produced
about179ms dense tuning steps, versus about53ms in a direct smoke. Taskpolicy-B
did not remove that limit; restoring nice0 was denied and changed nothing. Only
our controller was paused; its active tuning child completed. Three complete
tuning results were reused after configuration/completion checks. A separately
frozen Interactive/nice0 launcher ran every primary fit. No partial fit was
retried or overwritten. See `AMENDMENT_LAUNCH_PRIORITY.md` and runtime records.

No primary fit failed or was automatically retried. Runs were sequential with
16GiB MLX/1GiB cache limits, competing-process/memory-pressure preflight and a
30min/fit timeout. The original and corrected one-shot launchers are not login
automations. Both own one-shot launchers were unloaded; the corrected controller
exited0. Original repository Git status matches the before snapshot exactly.
Completion cleanup/isolation are in `metrics/launcher_cleanup.json` and
`metrics/isolation_audit.json`.

Source contract commit4189a97; main freeze71dbc47; corrected H1 freeze85b9444;
priority correction d5ed889. Environment: M4 Max128GiB, macOS27.0.1, native Python
3.11.17, MLX/MLX-Metal0.32.2, NumPy2.4.6, Matplotlib3.11.2, Tokenizers0.23.2.
Exact hashes are in three current freeze maps and the older dependency freezes.

From `/RESEARCH_ROOT`, existing source/data:

```sh
.venv/bin/python research/moe-night/results/2026-10-04-recurrence/scripts/audit_freezes.py
.venv/bin/python research/moe-night/results/2026-10-04-recurrence/scripts/verify_checkpoints.py
.venv/bin/python research/moe-night/results/2026-10-04-recurrence/scripts/verify_reload.py
.venv/bin/python research/moe-night/results/2026-10-04-recurrence/scripts/analyze.py
.venv/bin/python research/moe-night/results/2026-10-04-recurrence/scripts/write_report.py
```

Reproducing one primary fit requires an unused tag and no competing local ML job:

```sh
.venv/bin/python research/moe-night/results/2026-10-04-recurrence/scripts/train.py --kind rec_tb2 --seed 42 --steps 5000 --muon-peak 0.01 --tag reproduction_rec_tb2_s42
```

It refuses existing output files. Repeating the complete controller requires a
fresh separately configured/frozen output directory; do not overwrite these data,
metrics or original freezers. Preserve sibling architecture sources and SHUTTLE
mount links. Dependencies are listed in `configs/requirements.txt`; the existing
private `.venv` already has them. The original H1 probe is intentionally preserved
as failed; use separately frozen `benchmark_h1_v2.py` for its corrected reproduction.
Plots are rebuilt from raw JSONL only. `metrics/primary_summary.csv` is a compact
per-seed table; `metrics/analysis.json` retains the descriptive calculations.

## Recommendation

{interpretation}

Interpret the registered final-token and measured-time comparisons together.
This10.24M-target pilot is heavily undertrained and does not establish mature
language modeling, reasoning or gains at d1024/context4095. It also does not compare
every recurrence architecture, truncation depth or optimizer. Any follow-up needs
fresh data, a new contract, multiple seeds, and useful learning per elapsed time;
compression alone is not a pretraining speed result. No production adoption or
larger-scale fit is triggered by this report.
'''
    (ROOT / "REPORT.md").write_text(prose_spacing(report))
    print("Completed measured report and interpretation written.")


if __name__ == "__main__":
    main()
