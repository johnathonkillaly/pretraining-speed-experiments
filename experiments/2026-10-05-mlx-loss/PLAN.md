> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# MLX 0.32.3 Metal loss and H1 throughput study

Registered 2026-10-05 following the owner's request to spend time improving MLX
pretraining on this otherwise idle Mac. Work apps may remain active.

## Question and attribution

Does the new fused Metal cross-entropy forward/backward improve practical H1
training speed? Compare exactly three primary arms:

- A: installed MLX/Metal 0.32.2, explicit composed FP32 cross-entropy matching its
  ordinary Metal formula.
- B: isolated MLX/Metal 0.32.3, the same composed FP32 cross-entropy.
- C: isolated MLX/Metal 0.32.3, fused FP32 cross-entropy.

B versus A measures the library upgrade. C versus B isolates loss implementation.
C versus A measures the whole upgrade with fused loss. Preserve FP32 logits and
loss in all primary arms. Do not change precision, model, optimizer, checkpointing,
batching, data, synchronization or memory limits for the attribution experiment.

## Source, environment and inputs

Import immutable `2026-10-04-recurrence/scripts/model.py`, `optimizer.py`, and
`train.py`, and their frozen Llama/RoPE dependencies; never run the archived
controllers or their `main()` functions. The original freezes and output records
are verified before and after work. Copy nothing from protected data/eval banks.
No Fedora or Windows access is part of this campaign. No real-data learning or
quality claim, production checkpoint, operating-system change, or timer change.

Use the same Python 3.11.17 and NumPy 2.4.6 in both environments. Preserve the old
environment. Freeze the new benchmark/controller/analysis sources, exact package
versions and command arguments before GPU execution. Version amendments before
their runs; retain failures. Generate all tokens/logits in memory from fixed seeds.

## Correctness and microbenchmarks

In the new environment, compare fused versus composed loss and the full gradient
for FP32 logits at shapes (128,1024), (1024,32768), and (4095,32768). Fixed inputs
include ordinary logits, extreme +/-100 logits, large shared row offsets,
nonuniform cotangents, and tied maxima. Reject nonfinite results. Require loss
absolute error <=2e-4 and gradient max absolute error <=2e-5, relative L2 <=2e-4.
Compare directly against an independent NumPy float64 reference on small cases.
Use eager and compiled checks; include a real H1-shaped full-model gradient check
at context 64 (same initial state), reporting every parameter's maximum error,
global relative L2, and nonzero/finite coverage. Require relative L2 <=.005 for
BF16 model gradients and loss absolute difference <=2e-4.

Microbenchmarks measure synchronized compiled loss-plus-gradient for FP32 logits
using fixed nonuniform cotangents. Use three warmups and nine measurements per
function, reverse arm order on alternating shapes, and retain all timings/peaks.
These exclude the vocabulary projection and are not whole-training speedups.

Optional diagnostic only after primary correctness: raw fused loss on BF16 logits
versus the composed formula on those identical logits cast to FP32. Record actual
output dtype and full loss/gradient errors. Do not use BF16 loss as a primary arm
or change full-training precision without a separately registered amendment.

## Full H1 benchmark

298,361,856 parameters, dense20, d1024/h3456, GQA8/2 head128, tied vocab32768,
RoPE500000, alternating block checkpointing. BF16 model/backward, FP32 masters,
gradient accumulation/state; immutable independently batched Muon/AdamW recipe
from the prior Mac probe (zero decay and its existing momentum/bias correction).
This repeats the Mac recipe; it does not claim exact production CUDA optimizer
fidelity. Fixed explicit LRs .00015/.00006. Clipping1. Eight synthetic sequences
from NumPy seeds500..507, 4096 IDs each, IDs1..32767, input4095, microbatch1,
accumulation8, 32,760 targets/update. Seed42 initialization for every arm; record
hashes of all initial tensors and inputs and require agreement across arms.

Run A,B,C,C,B,A as separate fresh processes, cooldown between windows. Each run
uses three warmup and five measured updates. Preserve prior microbatch eval and
optimizer synchronization boundaries. Report model/backward/accumulation time,
optimizer time, total targets/s, active/cache peaks, loss and gradient norm. Check
finite gradients/master/state and actual parameter changes outside measurement;
record numerical summaries for matched first updates. No checkpoint saving or
repeated corpus learning. Compilation and initialization times are reported apart.

## Safety, budget and decisions

Limit MLX active allocations to16GiB and cache to1GiB. Require AC power, no
competing ML job, macOS thermal state nominal before each full-model process,
and memory-pressure free percentage >=20. Poll thermal state at least every two
seconds; memory/AC periodically. Stop on serious/critical thermal state, memory
pressure <20%, AC loss, nonfinite result, competing ML job, or user interruption.
Temperature sensors are unavailable without privileged tools; report thermal-state
monitoring as such. Do not kill or reprioritize the owner's work apps.

Cap aggregate worker wall time (including correctness, initialization, compile
and GPU measurements) at900seconds, overall campaign wall time1800seconds,
each H1 process180seconds. Stop and report the completed subset on a failure;
do not silently retry or extend. Sequential workers only. No unattended job.

Primary decision: recommend an isolated upgrade when correctness passes and both
paired rounds show C/A speed improvement >=5%, without >10% higher peak active
memory or any monitor event. Smaller/mixed gains are inconclusive. If a microkernel
improves strongly but whole training changes <5%, report Amdahl's limit and profile
the remaining time rather than promise a substantial pretraining improvement.
No claim that synthetic numerical agreement establishes real-data model quality.
