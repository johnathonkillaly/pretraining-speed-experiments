> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# H1 checkpointed compilation capture amendment

The optional frozen H1 benchmark failed before its first completed update with
`Attempting to eval an array without a primitive`. The main recurrent training
and backward checks passed and do not use activation checkpointing.

Inspection of MLX's checkpoint wrapper showed that it calls module.update during
recomputation. The H1 microgradient function captured model.state as input but
omitted it from captured outputs. A tiny checkpointed dense model with both input
and output state captured produced valid loss/gradients on two consecutive calls.

Create benchmark_h1_v2.py as a separately frozen copy: capture model.state as both
input/output for microgradients, verify this amendment's hashes in addition to
the original freeze, and use separate v2 result/failure filenames. Preserve the
original source and failure evidence. No model, timing, data, memory-limit or
warmup changes. This is one defect-driven corrected run, not a blind retry.
