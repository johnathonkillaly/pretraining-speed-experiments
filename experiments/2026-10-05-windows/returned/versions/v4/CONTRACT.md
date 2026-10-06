> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Registered profiler extraction amendment v4

v3 full-geometry logits/gradients and CUDA checkpoint/continuation checks passed.
Keep them and the twelve v2 windows intact. The v3 profiler extraction wrongly
included five CUDA user annotation ranges named hybrid_optimizer_profile_update
as though they were leaf kernels. Their durations overlap their contained kernels.
Thus its cuda_kernel_total_us and annotation-inclusive top-kernel list are
superseded, not used in the report. Preserve that raw evidence.

Repeat only the same short optimizer-only probe on unchanged hash-verified v2
model/optimizer: three warmups, five unprofiled updates, five profiled updates.
Exclude explicit user annotations and the known annotation name from CUDA leaf
events. Report compute kernels separately from Memcpy/Memset device operations;
keep all aggregate kernel rows for audit. GPU operator table contains only CPU
aten operator events, avoiding mixing operator parents with CUDA kernel aliases.
Keep measured CPU CUDA launch API durations separate and do not add them to
overlapping GPU time or call the entire wall update GPU compute. Same fixed BF16
gradients, source formulas, Flash-unavailable efficient SDPA, precision and shape.
No full benchmark-suite or full verification repetition.

Bind v2 FREEZE hash and freeze this source/config/contract before GPU work.
Carry forward the entire128.27s prior active usage (exact config value) against
the original600s cap, AC required, <=65C start, >=85C stop, .8s telemetry and
competing-process guards. Process timeout300s. No system setting changes.

Command:
work/cuda-health-venv/Scripts/python.exe outputs/recurrent-cuda-study/versions/v4/addendum.py --output outputs/recurrent-cuda-study/measurements/v4 --checkpoint-dir work/recurrent-port-checkpoints/v4
