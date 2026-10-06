> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Registered native CUDA hardware campaign, v1

2026-10-05. Owner-authorized bounded synthetic recurrence testing and read-only
diagnostics. Source ZIP hash and inherited integrity maps are recorded at the
study root. The supplied source commit c022d50ed5268514ef5cfa1348274d3840d0ae4e
is sender-attested; the ZIP contains tracked files without Git history.

Freeze model.py, optimizer.py, verify.py, campaign.py, config.json and this
contract before executing correctness or timing. Source defects require a new
version directory and a committed amendment, keeping v1 and failed outputs intact.
No old frozen study, corpus/checkpoint, production, driver, WSL, firmware, lid or
power-policy mutation. The initial archive tree is hashed for an after-run audit.

Exact reference geometry: d256/h864, GQA8/2/head32, vocab32768/context256,
RoPE500000 (nontraditional split halves), RMS epsilon1e-5, no biases, SwiGLU,
tied embedding/head. A CPU FP32 ten-block bank is generated per seed: Dense4
uses 0,1,2,9; Dense10 uses all ten; recurrence uses prelude0/core1,2/coda9.
Recurrence has R4, injection .5*(state+prelude), TB2 state detach immediately
before loop3; prelude injection stays differentiable. Embedding normal std1/sqrt(d),
linear default PyTorch uniform +/-1/sqrt(input_width), norms ones. Check exact
common initialization hashes across arms for each seed. MLX and PyTorch RNGs
are different; no exact cross-backend tensor/learning equivalence is claimed.
Parameter counts must be 11,700,480 for Dense4/recurrent; 16,667,904 for Dense10.

BF16 weights and gradients, FP32 masters/state/clipping. Muon only named hidden
projection matrices, independently normalized by Frobenius norm+1e-7 before BF16,
five NS steps coefficients 3.4445/-4.7750/2.0315, transpose tall matrices,
Nesterov m=.95*m+.05*g and u=.05*g+.95*m, LR .01*sqrt(max(1,rows/cols)).
Group equal shapes, at most five matrices. Adam beta(.9,.95), bias correction,
epsilon1e-8, LR .0003; embedding and 1-D norms only; no decay. Clip1 with
scale=min(1,1/(global_norm+1e-6)). One embedding parameter owns tied logits.

CPU-double semantic tests: exact full/TB2 forward, K>=R gradients, one-loop
Dense4 equality; actual intermediate prefix gradients absent, suffix/prelude
retained, nonzero retained modules; actual fixed-prefix conditional FD on core
and prelude projections eps1e-5, atol2e-7/rtol.002. Manual causal attention agrees
with CPU-double SDPA atol/rtol1e-12. RoPE identity at zero and norm preservation
atol/rtol2e-6. Muon batch/single relativeL2<.02 for tall/wide matrices with
different scales; three-step hybrid master relativeL2<.002; first-step source
Nesterov/Adam equations checked. Small checkpoint exact predictions and next
resumed optimizer update. CUDA BF16 tests: forward exact, K>=R gradients
maximum per-parameter relativeL2<.02; finite BF16 gradients and FP32 state,
detach boundary and batch/single Muon relativeL2<.02. No tolerance retuning.

Force Flash SDPA on CUDA with repeated KV heads, zero dropout and causal mask;
CPU SDPA uses its math implementation. RoPE precomputed FP64 trig tables are
stored FP32 then cast with the model; separate BF16 multiply/add and RMS casts
can round differently from fused MLX kernels. PyTorch batched addmm replaces
MLX addmm preserving the reference polynomial, not guaranteed bit-identical.
Eager only; no compilation, checkpointing, corpus/data loading or tuning.

Seeds42/43/44 and paired orders in config. Each arm begins from the same bank
within seed, with one seeded synthetic (8,257) batch drawn from IDs1..32767.
The same batch is repeated within its short window, not a learning experiment.
2048 targets/update; target0 masking matches source objective. Three warmups
then each 10-second wall window, with all individual steps recorded; stopping
at the first completed step after 10s may slightly overshoot. Full/TB2 order
alternates F/T, T/F, F/T. Dense order alternates D4/D10, D10/D4, D4/D10.
All arms are newly initialized per window. Report every paired seed/window and
medians, temperature/clocks/power and CUDA allocated/reserved/device memory.

Perf-counter timings synchronize CUDA after forward/loss, backward and optimizer.
Forward excludes zero_grad; backward excludes forward finite checks. Optimizer
includes global norm, clipping, all hybrid updates and BF16 weight copies.
Whole-step includes zero_grad and scalar/monitor checks, but not step-log writes;
window elapsed throughput includes log writes and loop overhead. Empty CUDA
synchronize host-call median is reported, not subtracted. Every step checks loss
and global gradient norm; full weights/optimizer/gradient scans run after warmup
and window, outside measured window elapsed but inside GPU-active budget.

Require AC power, initial <=65C, no competing compute; known idle Codex renderer
PID may be baseline desktop overhead. Poll AC, temperature, memory, clocks,
power and processes every .8s; abort on polling >1s, stale readings, >=85C,
AC loss, another compute PID, nonfinite state, CUDA failure or budget expiry.
Use min(measured_free-1.5GiB,75% physicalVRAM) allocator limit, require >2GiB.
Cooldown <=65C before every window; timeout180s. Explicit GPU-phase wall time
including initialization/correctness/warmup/validation is a conservative GPU-active
counter with hard600s cumulative cap. CPU-only numerical work/cooldowns excluded.
No optional60s probe in this first registration; no resource-failure retry or
automatic common-shape adjustment. Campaign timeout1800s. Preserve incomplete
and stopped records, report completed subset. Do not interpret speed, synthetic
loss/clipping or these three measurement seeds as a quality result or a matched
Mac/CUDA ratio. Learning/clipping1v5 study remains deferred.

Command from workspace:
`work/cuda-health-venv/Scripts/python.exe outputs/recurrent-cuda-study/versions/v1/campaign.py --output outputs/recurrent-cuda-study/measurements/v1 --checkpoint-dir work/recurrent-port-checkpoints/v1`

Return a <=64MiB ZIP of new source/contracts/configs/results/report and small
healthcheck metadata, excluding checkpoints, environments, installers, private
data and enormous OS logs. Verify original archive tree unchanged. Stop research
GPU processes on exit. The return destination is the owner's provided LAN server;
transfer credentials are kept out of the report and return archive.
