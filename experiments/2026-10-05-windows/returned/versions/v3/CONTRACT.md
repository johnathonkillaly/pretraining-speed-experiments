> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Registered optimizer profiling and stronger validation addendum, v3

2026-10-05. The twelve v2 timing windows completed before this addendum. Preserve
their results and frozen model/optimizer/config; do not rerun the benchmark suite.
The selected EFFICIENT_ATTENTION backend remains forced; Flash is unavailable.
This addendum imports the hash-verified v2 source without changing its formulas.

The existing exact-forward gates compared scalar objectives. Strengthen the
evidence by comparing every full-geometry BF16 logit for full versus TB2 and
one-loop recurrence versus Dense4 at batch8/context256/vocab32768. Repeat K>=R
gradient equivalence at full geometry using the existing relativeL2<.02 bound.
This is additional verification of the unchanged timed implementation, not a
model change. Add a small CUDA-native checkpoint save/reload, with exact
predictions and serialized optimizer tensors; compare one resumed update within
master relativeL2<.002, also recording exact agreement when available.

Profile the unchanged full recurrent hybrid optimizer on actual precomputed
BF16 gradients from one seed42/input1042 batch at the registered geometry.
Three optimizer warmups, five unprofiled synchronized updates, then five profiled
updates with CPU and CUDA profiler activities where supported. Keep clip1,
Muon .01, Adam .0003, FP32 master/state, BF16 weight copies, and every original
formula/group unchanged. This fixed-gradient probe diagnoses update cost,
not learning. Record unprofiled and instrumented wall times, CUDA leaf kernel
total/top kernels if present, top operators by GPU time and CPU self-time,
and CPU duration of CUDA launch runtime calls. Do not sum nested operator GPU
times or label the entire ~14ms measured update as GPU compute. CPU/GPU totals
overlap; launch API duration excludes unmeasured Python/scheduler overhead.
If no CUDA timeline is available, report that limitation without installation.
No model checkpoint or huge raw profiler trace goes into the return ZIP.

Freeze this script/config/contract and bind the v2 FREEZE hash before execution.
Use the same Monitor and hard600s cap, carrying forward all prior126s (actual
value in config), <=65C starts/cooldowns, >=85C stop, AC requirement, .8s
telemetry and competing-PID rules. Log every active addendum phase. Overall
process timeout300s; no new setup, power/lid/firmware/driver/WSL changes.
Preserve failure evidence and complete results subset if the addendum fails.

Command:
`work/cuda-health-venv/Scripts/python.exe outputs/recurrent-cuda-study/versions/v3/addendum.py --output outputs/recurrent-cuda-study/measurements/v3 --checkpoint-dir work/recurrent-port-checkpoints/v3`
