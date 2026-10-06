> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# H1 laptop fit, speed, and tuning study

Registered 2026-10-05 for the owner's ASUS GU604VI RTX 4070 Laptop GPU.

## Reference and scope

Read-only source snapshot from the archived Fedora CUDA segment-2 worktree,
commit 19bf6427e4060a408a5c4a92d4d5ec5618b5f6a9. Only seven tracked source/config
files were copied. No production data, weights, checkpoints, service state, or
training measurements were accessed or changed. The old Mac/Windows recurrence
freezes remain separate and immutable.

Historical 3080 reference: 15,522.91 targets/s over 90 measured updates at 280 W,
Torch 2.12.1+cu132, compiled model and Muon, FlashAttention, alternating activation
checkpointing, independent Muon matrix batches of five, FP32 update-norm telemetry.
Some historical device-wide samples reached 8,731 MiB, exceeding the laptop's
8,188 MiB. This comparison must distinguish geometry from runtime fidelity.

## Registered sequence

1. Audit stock OS controls, AC power, temperature, memory, and CUDA runtime.
2. Derive an isolated synthetic-input runner from the copied H1 and optimizer
   source. Native fresh initialization only; do not call its data-reader or
   production entry points. Freeze source, config, commands and tolerances on
   Windows before GPU execution. Record model/optimizer ownership counts.
3. Test stock geometry: 298,361,856 parameters, 20 blocks, d1024/h3456,
   GQA 8/2 with head dimension 128, tied 32768 embedding, RoPE 500000,
   4095 input positions from 4096 tokens, microbatch 1, accumulation 8,
   32,760 targets/update. BF16 parameters/gradients and FP32 masters,
   accumulation/state. Use original independent Muon batch 5 + AdamW,
   fixed explicit LRs .00015/.00006, decay .1 and source clipping convention.
4. Prefer the original attention/checkpoint/runtime choices. Where the installed
   Windows runtime cannot execute them, register each change before testing.
   Predeclared fallback order: efficient SDPA with explicit repeated K/V heads,
   full block checkpointing if alternating OOMs, then checkpointed vocabulary
   loss chunks if full-logit CE still OOMs. Preserve model, context, effective
   batch, loss normalization, optimizer equations and all parameter gradients.
   Never silently shrink the model/context or call this an exact 3080 match.
5. Validate attention adaptation, loss-chunk equivalence if used, finite model,
   gradients/master/state, actual weight changes, and independent Muon behavior.
   A full-size update establishes fit; use two warmup and up to five synchronized
   measured updates for speed when the safety/resource budget permits.
6. Only after a successful stock baseline, try one conservative temporary GPU
   voltage/frequency adjustment if an OS control is actually exposed. Record
   exact original values, mutation and rollback first. No startup persistence,
   memory overclock, BIOS changes, CPU unlock, driver replacement or WSL install.
   No concurrent ASUS/G-Helper profile controllers. If controls need unavailable
   elevation or are locked, document that and finish the stock benchmark.
7. Compare stock/candidate on the identical frozen runner and settings, alternating
   order if resources permit. Revert candidates on instability, materially lower
   throughput, or uncertain rollback. A short CUDA check does not certify gaming
   or unattended stability. Leave stock settings if no candidate is validated.

## Resource and reporting contract

Require AC, no competing compute job, lid open, and initial GPU <=65 C. Independent
telemetry at least once per second; stop at >=85 C, AC loss, nonfinite state,
telemetry failure, competing compute, or repeated memory failure. Cap GPU-active
time across correctness, warmup, fit failures and measurements at 600 seconds.
No long unattended test. Log synchronized target throughput, update timing,
allocated/reserved/device memory, power, clocks, temperature and throttle flags.

Failures and amended source are versioned, never overwritten after freezing.
Register any amendment before its execution. Report completed subsets candidly.
Synthetic speed/fit tests do not establish learning quality. A difference from
the historical 3080 includes software, power settings and machine differences.
Archive hashes and verify the returned ZIP and file manifest on the Mac.
