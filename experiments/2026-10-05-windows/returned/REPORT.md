> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Connected RTX 4070 Laptop CUDA campaign — 2026-10-05

**PASS for the twelve registered timing windows; one supplemental monitor-cadence deviation is disclosed below.** All twelve 10-second

measurement windows completed at the exact batch8/context256 geometry after

CPU/CUDA correctness gates. TB2 delivered a median paired throughput gain of

**9.75%** and a median paired

backward-time reduction of **21.04%**.

This measures eager native-Windows CUDA cost, not learning quality.

The entire campaign, including the stopped Flash attempt, correctness, warmup,

validation and both profiling attempts, used **130.29 / 600 seconds** of

conservatively counted GPU-active phase time. Maximum sampled temperature was

**74°C**, below the 85°C stop. No long training, lid reproduction, WSL install,

firmware/driver change or power-policy change occurred. Research processes exited.

## Measurements

Each row reports medians across seeds42/43/44, one newly initialized short window

per arm/seed, three warmup steps. Same ten-block-bank common-weight SHA and same

synthetic batch SHA were confirmed for all four arms within each seed. Parameter

counts match the source. BF16 weights/gradients and FP32 masters/state were used.

| Arm | Unique parameters | k targets/s | Forward ms | Backward ms | Hybrid ms | Whole step ms | Peak allocated/reserved MiB | Device MiB | Max °C |

| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

| Dense4 | 11,700,480 | 55.92 | 8.59 | 12.13 | 14.64 | 35.83 | 1035.7 / 1446 | 1603 | 69 |
| Dense10 | 16,667,904 | 34.87 | 13.35 | 17.53 | 25.81 | 57.57 | 1238.2 / 1672 | 1829 | 72 |
| Full recurrence | 11,700,480 | 42.78 | 13.44 | 17.88 | 14.51 | 46.58 | 1190.7 / 1618 | 1775 | 73 |
| TB2 recurrence | 11,700,480 | 46.60 | 13.56 | 14.11 | 14.47 | 42.70 | 1087.4 / 1504 | 1661 | 74 |

The timings are synchronized host elapsed times. Forward includes objective/FP32

cross entropy; backward is separate; hybrid includes global clipping, reference

Muon/Adam updates and BF16 weight copies. Whole-step includes zero_grad and scalar

finite/monitor checks. Window throughput includes step-log writes and loop

overhead, so it is not simply2048 divided by the median component step. Empty

CUDA synchronization median host-call cost was 3.10µs,

not subtracted. Component medians need not add exactly. No compilation/data-loader

work is included; tensors were preallocated on GPU. All individual steps, actual

window durations, warmups, clipping flags, losses and telemetry are retained.

Measured-window target accounting totals 5,406,720; warmup/verification

targets are separate, and no real corpus was used.

| Paired seed | Full k targets/s | TB2 k targets/s | TB2 throughput gain | Backward reduction |

| --- | ---: | ---: | ---: | ---: |

| 42 | 42.22 | 46.60 | +10.39% | 21.74% |
| 43 | 42.78 | 46.95 | +9.75% | 21.04% |
| 44 | 43.17 | 46.30 | +7.24% | 21.01% |

These three initialization/window seeds are small hardware repeats, not three

independent learning fits. Full/TB2 order alternated F/T,T/F,F/T; Dense order also

alternated. The repeated fixed batch overfits quickly; its synthetic losses and

clipping rates must not be interpreted as generalization or the proposed clip1/5

learning ablation. TB2 still has all unique parameter gradients and the same

optimizer ownership/state sizes as full recurrence. Its smaller activation peak

and lower backward cost do not imply sparse parameter gradients.

## Correctness and optimizer fidelity

CPU-double gates passed: full/TB2 scalar objective equality, K>=R exact gradients,

one-loop/Dense4 equality, actual intermediate prefix gradients absent after detach,

nonzero retained prelude/embedding/core/coda/norm paths, conditional fixed-prefix

finite differences on actual core/prelude parameters, manual causal attention

agreement, and nontraditional RoPE identities. CUDA BF16 gates passed with finite

state and independently normalized batched versus separate Muon updates.

The original exact-forward gate compared scalar objectives. A committed post-window

addendum strengthened it without changing the timed model: **every logit** at

(8,256,32768) is exactly equal for full/TB2 and one-loop/Dense4. Full-geometry

K>=R gradient tensors are also exactly equal. Both small CPU and CUDA checkpoint

predictions and next resumed optimizer updates agree exactly. CUDA checkpoint

SHA256: `912e66fd966e5c7f54caf08b224fa19472647ce87770911a13d7cfbadeb403c3`; checkpoint retained locally

outside the return package. All checkpoint files are excluded from the ZIP.

Muon uses source-defined projection ownership, independent per-matrix Frobenius

normalization, five NS steps with coefficients3.4445/-4.7750/2.0315, tall-matrix

transpose, groups of at most5, m=.95m+.05g, u=.05g+.95m, and source shape scaling.

Adam owns the one tied embedding and norm vectors, beta(.9,.95), bias correction,

epsilon1e-8; weight decay0. Clip1, Muon LR.01, Adam LR.0003, no tuning. The small

three-step batched/single hybrid master relativeL2 was

0; the tests

also independently checked the first-step Nesterov and Adam formulas.

## Optimizer profile

The hybrid update costs about14.5ms in the recurrent benchmark. A separately

registered fixed-gradient optimizer-only probe used three warmups, five unprofiled

updates and five profiled updates on the unchanged full-geometry optimizer.

Corrected profile: unprofiled median **15.47ms**, profiled

median **20.97ms**. Instrumentation materially raises cost;

headline speed remains the original v2 windows.

Across five profiled updates, compute-kernel time totaled

**47.448ms** (mean

**9.490ms/update**), with a further

1.253ms of device memcpy/memset operations.

The profiler measured **3,975 CUDA launch API calls**

(795/update) and

**23.844ms** of host time inside those calls

(4.769ms/update). GPU and CPU

durations overlap and cannot be added to reconstruct wall time. Launch-call

duration does not include all Python dispatch or scheduler overhead. The measured

~14.5ms wall update is therefore not all GPU compute.

| Operator | Calls across5 updates | Self GPU ms | Self CPU ms |

| --- | ---: | ---: | ---: |

| aten::mul | 1545 | 15.526 | 7.968 |
| aten::add | 470 | 7.341 | 2.083 |
| aten::div | 185 | 6.057 | 0.877 |
| aten::copy_ | 990 | 5.640 | 2.794 |
| aten::baddbmm | 350 | 3.405 | 4.931 |
| aten::bmm | 175 | 1.952 | 2.543 |

Pointwise multiplication/addition/division and copies account for substantial

GPU time as well as many launches. That identifies candidates for a future

formula-preserving fusion/compilation experiment; no optimizer formula or

implementation optimization was made here. These operator self times and

parent/inclusive times in raw JSON must not be mixed or summed twice. Top kernels,

all aggregate compute kernels, memory operations, CPU operator averages and API

call durations are in `measurements/v4/optimizer-profile.json`.

The first profile export (v3) mistakenly counted five GPU annotation ranges as

leaf kernels. It is preserved, explicitly superseded for profiler totals, and not

used above. A frozen v4 amendment reran only the tiny profiling probe, excluding

annotations and separating compute from memory operations; no benchmark-suite

rerun occurred. All its GPU time counts against the original600s cap.

## Safety and backend limits

Every window required AC and a <=65°C start. An .8-second monitor checked power,

temperature, clocks, memory and competing compute PIDs. Benchmark-window telemetry had a maximum

gap of 0.897s; the corrected profile had

0.971s. **One v3 supplemental checkpoint/profile-transition

gap was 1.141s**, exceeding the one-second cadence requirement,

with42°C and43°C samples on either side. This is a disclosed monitor-cadence

deviation. The sampler checked individual query duration and1.5s staleness but

did not reject every >1s inter-record scheduling gap. Strict cadence compliance

therefore does not apply to the whole addendum. No failed sensor readings,

thermal/resource/nonfinite stop or competing research compute was observed.

Before longer GPU work, use an external monitor or a strict inter-sample deadline.

No extra GPU repetition was launched after discovering the deviation. The recorded allocator cap used measured free

VRAM minus1.5GiB, capped at75% physical VRAM. The largest benchmark device-wide

memory sample was1,829MiB. PyTorch allocated/reserved and NVIDIA device-wide

measurements are distinct; the device sample is not a continuous memory peak.

The start temperature was much lower than the earlier generic test; short

windows and <=65°C cooldown checks bounded heat. They do not establish thermal

equilibrium or long unattended training reliability. CPU temperatures were not

available. Some read-only Windows CIM/event diagnostics overlapped benchmark

wall time; desktop scheduling, monitoring, logging and background CPU work can

affect an eager Python implementation. No cross-machine speed ceiling is claimed.

Windows torch2.7.0+cu128 lacks Flash Attention. Frozen v1 passed CPU correctness

then stopped at its forced Flash gate before any benchmark. Its failure is

preserved. Registered v2 audited backend eligibility and selected

**EFFICIENT_ATTENTION**, forced for CUDA correctness and all timings; no math

fallback was needed. GQA repeats KV heads explicitly. Separate BF16 RoPE arithmetic,

RMS normalization casts, CUDA batched addmm and reduction order can round

differently from fused MLX kernels. Adam uses a host integer step and scalar

bias-correction powers before FP32 tensor arithmetic, while MLX uses tensor powers.

Embedding/linear initialization distributions match the reference; backend RNG

draws/tensors do not. Eager CUDA versus compiled MLX, kernel choices, initialization,

instrumentation and workload histories preclude an exact Mac/CUDA ratio.

## Provenance and deliverables

Source archive SHA256:

`688f5517f878e8583dc0584a6a13aec56aa826537582753ff3a90128a2e4e539`.

The supplied commitc022d50ed5268514ef5cfa1348274d3840d0ae4e is sender-attested;

the archive has no Git history. All17 included entries across the three inherited

recurrence freeze maps verified; five missing entries are intentionally omitted

data files. Initial and final full extracted-tree hashes agree. No prior frozen

source or record was changed.

New source/config/contracts are under `versions/v1` through `versions/v4`, each

registered/committed/frozen before its GPU work. v2 measurements are authoritative

for timing; v3 for strengthened verification/checkpoints; v4 for profiling. All

failed/superseded evidence remains. `SUMMARY.json` is CPU-derived, not new testing.

Return package excludes checkpoints, corpora, environments, installers, raw

enormous OS logs and Git internals. The local Git repository retains registration

history; `GIT_HISTORY.txt` records commits for the portable package.

The clip1 versus clip5 fresh-data learning experiment remains deferred. No real

data or final-test set was evaluated. No production change or scale-up follows

from this pilot.
