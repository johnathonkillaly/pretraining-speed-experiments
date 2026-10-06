> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Extra validation at the actual context length

2026-10-05, registered before this check runs. The capacity candidate's context64
checkpointing check passed exactly. Before recommending its measured speedup,
compare compiled full-parameter gradients at4095 input positions on the first
actual seeded synthetic sequence used in the timing runs, fresh seed42 H1 weights.

Compare B (composed/alternating checkpointing) versus E (composed/no checkpointing)
at the unchanged loss absolute2e-4 and BF16 gradient relative L2.005 thresholds,
with finite/nonzero coverage for every tensor. Store all tensor error statistics.
These checks evaluate one full-context microbatch; they do not test real-data
quality or checkpoint resume. Optional third full-context fused/alternating
gradient is diagnostic only and cannot repair the original context64 rejection.

Use32GiB allocation/1GiB cache limits and the same independent monitor. Run only
after the eight timing workers finish, within the original900-second aggregate
worker and1800-second campaign limits, including all earlier workers. No retry,
tolerance change or timing-rule change. A failure prevents recipe recommendation.
