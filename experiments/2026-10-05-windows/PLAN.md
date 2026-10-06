> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Connected RTX 4070 laptop test plan

Prepared 2026-10-05 for the owner's connected `WINDOWS_LAPTOP` laptop.
Reference source: Mac commit b36f523660d7886bc8591f6b5c6d4f4114e2ff78.
This document describes the next campaign; no laptop recurrence test has run yet.

## Existing evidence

The completed Windows chat `Fix laptop lid sleep issue` reports an ASUS GU604VI,
i9-13900H, 32GB RAM, and RTX 4070 Laptop GPU with 8188MiB VRAM. Its isolated
native PyTorch/CUDA environment passed a synthetic BF16 healthcheck, including
finite training state and exact checkpoint reload predictions. Its generic
transformer differs from our recurrent model and cannot establish a matched
Mac/CUDA speed ratio.

That two-minute run ended at 83 degrees C with an 85-degree stop threshold;
temperature had not stabilized. Lid close was already configured to hibernate,
and prior event logs reported failed hibernation resumes and unclean restarts.
These observations require short monitored work before a longer training run.

## Scope

1. Transfer a checksum-verified archive of this research checkout to a new
   laptop directory. Preserve all previously frozen files and existing laptop
   healthcheck outputs. Reuse the working native CUDA environment after an
   audit. Do not install WSL solely for this test.
2. Port the reference recurrent model and hybrid Muon/AdamW optimizer to
   PyTorch in a new study directory. Register and freeze exact source,
   configuration, tolerances, and benchmark commands before CUDA timings.
3. Validate full versus truncated forward equality, K>=R gradient equality,
   one-loop equivalence to Dense4, the detach boundary, retained parameter
   gradients, conditional finite differences, independent per-matrix Muon
   equivalence, finite state, and checkpoint reload agreement.
4. Measure Dense4, Dense10, full recurrence, and last-two-loop recurrence at
   batch 8 / sequence 256. Use seeded synthetic inputs with vocabulary 32768.
   These are hardware measurements and do not measure learning quality.
5. Collect available read-only sleep, driver, power, and event diagnostics.
   Record which administrator-only reports remain unavailable. Do not close
   the lid to reproduce the failure, flash firmware, change drivers or change
   system power policies during this campaign.

## Model and optimizer fidelity

Use the archived `2026-10-04-recurrence/scripts/model.py`, `optimizer.py`, and
their sibling-study imports as the specification. Geometry: d256 / h864,
GQA 8/2 / head dimension 32, RoPE 500000, RMSNorm epsilon 1e-5, tied head,
no bias. Initialize a ten-block bank per seed; Dense4 uses blocks 0,1,2,9.
Recurrence uses prelude 0, shared core 1 and 2, and coda 9; four core loops
inject `0.5*(state+prelude)`. TB2 detaches state before loop three while
preserving the injected prelude gradient path.

Use BF16 model weights/gradients and FP32 master weights/optimizer state.
Muon owns the attention and MLP projection matrices; AdamW owns embedding
and one-dimensional norm parameters, with the tied head stored once. Match
the archived Newton-Schulz coefficients, normalization, Nesterov convention,
shape scaling, Adam bias correction, and no weight decay. Normalize each
matrix independently even when stacking equal shapes. Use clipping 1,
Muon LR .01, Adam LR .0003 for this speed probe, with no tuning.

Keep one shared initialization and input batch within each paired PyTorch
comparison. A shared seed does not create identical MLX/PyTorch weights;
cross-backend results must state this limitation unless identical tensors
are transferred and verified. Record attention implementation, precision,
compilation choices, and any port differences. Begin with eager CUDA; compile
only after correctness and within the remaining resource budget.

## Timing and stop rules

Require AC power, no competing compute job, and initial GPU temperature at
most 65 degrees C. Retain the existing 85-degree threshold. Monitor device
temperature at least once per second; stop on threshold, nonfinite state,
telemetry failure, repeated memory failure, or a new competing compute job.
Allow Windows/display headroom and derive the VRAM limit from measured free
memory, not the GPU label. Log allocated, reserved, and device-wide memory.

Use synchronized warmup and timing; report forward, backward, optimizer,
and whole-step measurements with explicit synchronization overhead. Start
with 10-second measurement windows, alternating full/TB2 order across three
paired windows, with cooldown to at most 65 degrees C before the next window.
Measure Dense4 and Dense10 with similarly bounded windows. A 60-second
sustained TB2 probe is optional only if these windows remain below 80 degrees
C. Cap total GPU-active work, including correctness, warmup, and optional
compilation, at 10 minutes. Stop when that cap is reached and report the
completed subset; do not silently extend it.

Report medians, individual paired measurements, token accounting, temperature
and clock trajectory, power, and peak memory. Separate the new synthetic
benchmarks from prior real-data learning results. Save a small checkpoint and
verify reload, then leave no research GPU process running.

## Deferred learning study

Real-data learning tests need a documented fresh data split and another frozen
contract. The full/TB2 by clipping 1/5 comparison remains proposed, with three
paired seeds and equal development-only tuning allowances. Do not use the
old final test set to select hyperparameters. No long unattended run is part
of this first connected campaign.
