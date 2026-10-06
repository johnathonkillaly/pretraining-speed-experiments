> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Full-model numerical gate stopped the first campaign

2026-10-05. The frozen v1 correctness worker passed all12 eager/compiled standalone
loss cases (including FP64 oracle checks) and six microbenchmarks, then failed the
full298M/context64 BF16 gradient gate: loss difference0, global gradient relative
L2=.008071097035077832, above the preregistered.005. The controller stopped after
4.586seconds of aggregate worker time; no primary H1 timing process ran.

The original gate remains failed. No tolerance is relaxed. Original code, freeze,
logs, numerical records and failure remain unchanged. The first worker placed the
full-model detailed record in its AssertionError rather than saving it before the
assertion; the log/failure JSON preserves that record.

New diagnostic, registered before execution: compare repeated composed and fused
gradients within the unchanged BF16 model at context64 to measure repeatability;
compare them with checkpointing disabled; then compare composed/fused gradients
on the identical FP32 model to distinguish mixed-precision amplification. Report
every parameter's error and global relative L2. Checkpointed repeated composed
comparisons, repeated fused comparisons, and composed/fused comparisons use the
same initial weights and inputs. No new pass threshold or primary eligibility is
inferred from these results. Preserve all outcomes. This diagnostic is bounded to
180seconds within the original900-second campaign budget, uses16GiB/1GiB limits,
and runs through the same independent safety monitor.
