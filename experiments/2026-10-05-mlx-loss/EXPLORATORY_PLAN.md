> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Capacity-based follow-up, registered before any GPU results

The Mac has 128GiB unified memory and ample current headroom. After all primary
loss-attribution arms succeed, test whether avoiding activation recomputation is
a more useful speed improvement. This is a separate recipe comparison.

Arm D uses MLX0.32.3 and the primary fused FP32 loss, but changes alternating
activation checkpointing to none. Its allocation cap is32GiB rather than16GiB
to accommodate retained activations; the cache cap stays1GiB. Model, inputs,
initialization, precision, optimizer, rates, context, batch, gradient accumulation,
and synchronization stay fixed. Two fresh processes D1,D2 use the same three
warmup/five measured update schedule, only within the existing900-second budget.
No budget extension or silent batch/context shrink on OOM.

Before D timings, compare every full-model BF16 gradient against the checkpointed
fused model at context64: loss absolute error<=2e-4, gradient relative L2<=.005,
finite/nonzero coverage. This gate is included in the initial correctness worker.

Report D versus C as a memory-for-speed tradeoff, using each same-numbered round.
Recommend the exploratory recipe only if both rounds improve>=5%, all numerical
gates pass, peak allocation stays<=32GiB, and no safety-monitor event occurs.
Explicitly report the increased allocation and absence of real-data quality or
checkpoint-resume validation; do not deploy to any production trainer.
