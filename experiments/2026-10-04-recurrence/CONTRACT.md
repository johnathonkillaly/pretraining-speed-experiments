> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Recurrent-depth truncated-backpropagation pilot

Owner explicitly requested this isolated pretraining experiment. No BitNet work,
production job/configuration changes, or original checkpoint writes are authorized
or needed. Existing frozen studies remain intact.

Primary question: does truncating gradients through a shared transformer core
improve useful learning per second on this M4 Max / MLX 0.32.2?

## Architecture and controls

All pilot models: d=256, SwiGLU h=864, GQA8/2, head_dim32, vocabulary32768,
RoPE500000, sequence256, tied embedding/head. Same-seed embedding, norms,
prelude, core prototypes and coda initialization. One prelude block, two shared
core blocks, one coda block. At each loop, core input is 0.5*(state+prelude).
This parameter-free injection preserves a gradient path to the prelude after
detaching the prefix state. It is a minimal research architecture, not a faithful
Huginn replication. No adaptive halting, learned bridge, or capability claims.

Five controls: dense10, dense4 (matching unique recurrent blocks), recurrent
four loops with full backpropagation, identical recurrence with backpropagation
through the last two loops, and last-two-loop recurrence with a curriculum.
Curriculum: first5% of updates one loop, next5% two, remaining90% four. All
recurrent validation uses four loops. Fixed recurrence executes10 blocks;
dense10 executes10 distinct blocks. Dense4 executes4. Full and truncated
recurrence have identical forwards and parameters at initialization.

## Training and tuning

Compiled MLX, BF16 model weights/compute and FP32 master weights, optimizer
momentum/moments, and post-backward gradient clipping. Gradients originate in
BF16 and are promoted to FP32 for clipping/update; no accumulation in this pilot.
Hybrid Muon on explicitly named hidden attention/FFN matrices; AdamW on tied
embedding/head and norm vectors. Muon momentum.95, Nesterov, five BF16
Newton–Schulz steps, independent normalization/scaling for each 2-D matrix,
batched by shape (up to5 matrices). AdamW betas(.9,.95), bias correction on.
Decay zero on both branches to keep the optimizer semantics explicit.

Same tuning allowance for each variant: seed314,600steps, Muon peaks
.003/.01/.02, independently specified AdamW peak3e-4. Select the peak with
lowest final development loss per variant; ties choose the smaller peak. One
shared 5%-warmup/cosine-to0.1 schedule multiplier for both branches. No tuning
against the newly sampled test documents. Short tuning is a limitation.

Primary: seeds42/43/44,5000steps,batch8 =10,240,000 targets per fit,15 fits.
One fresh40,000-window training slice; no reused block from the old MoE pilot.
Same order within each seed across variants, no cycling. Reuse old128-window
validation only as development data; create512 additional validation documents
excluding old candidates for final-only test evaluation. Original v7 corpus
limitations remain: no new semantic deduplication, not the current Fedora v2 mix.
Never open protected banks. Record exact-window overlap, data/tokenizer hashes,
window provenance and seed. All test documents have256 prediction targets.

Log every update: objective, unclipped gradient norm, LR, loop count, actual
target count, synchronized step time and peak active memory. Dev loss every500
steps and at loop transitions; final test per-document loss once. Report final
seed variability and dev curves versus both tokens and measured training time.
Report compile/startup, validation/checkpoint overhead separately and in total
elapsed time. Throughput summaries exclude first50 updates after a new compiled
loop path; equal-wall learning curves include those costs. All raw times retained.
Save final weights, masters and optimizer state only to our SHUTTLE directory.

## Validation, resource limits, and hardware context

Numerical checks before training: full versus truncation-k=all gradients; identical
full/truncated forwards; one-loop recurrence versus dense4; toy per-loop gradients
zero before the stop boundary; conditional finite difference with a frozen prefix;
nonzero prelude/embedding/core gradients with truncated recurrence; batched Muon
versus separate matrices; compiled/eager one-update agreement within BF16 drift.
Benchmark synchronized forward and backward independently, including gradient
memory, with three warmups and15 repeat windows. No gradient-size sparsity claim:
every unique recurrent weight still needs a gradient.

Optional bounded H1 throughput probe:20layers,d1024,h3456,GQA8/2,head128,
vocab32768,tiedhead,4095 targets/microbatch,microbatch1,accumulation8, BF16
compute/FP32 masters and accumulated gradients, compiled hybrid optimizer,
alternating checkpointing. Three warmup and ten timed updates, synthetic token
inputs, no model-quality evaluation. Compare only as contextual shape-matched
throughput with the documented September CUDA handoff; no fresh NVIDIA run,
same-data equivalence, power normalization or universal hardware ranking.

16GiB MLX allocator limit,1GiB cache, run sequentially, per-fit30min timeout,
no automatic retry. Preflight other substantial local ML jobs and memory pressure
between fits. Stop on nonfinite loss/state, new competing ML work, or resource
failure. Freeze source/config/data hashes before substantive runs. Defects require
a separate pre-run amendment; preserve existing frozen files and failed evidence.

Verdict: a useful candidate must improve the speed/learning tradeoff, not merely
tokens/s. This10M-token pilot is still undertrained and cannot settle capabilities
or larger-scale pretraining. No production adoption or scale-up follows automatically.
