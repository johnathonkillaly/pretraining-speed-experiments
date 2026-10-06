> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Pre-run amendment v2

Initial CPU semantics/checkpoint gates passed. Native torch2.7.0+cu128 was
not compiled with Flash Attention; forcing it raised “No available kernel”
during the first small CUDA correctness forward. This was a backend capability
failure, not a thermal, nonfinite, or memory failure. No benchmark ran.

Preserve v1 source/freeze and its CPU verification, failure traceback, telemetry,
and active ledger. Select efficient SDPA only if its capability check accepts
both registered small and full shapes; otherwise use math. The single selected
backend is forced for all CUDA work and reported. FP32 Q/K/V intermediates in a
math fallback are disclosed. Keep all source recipe semantics, tolerances and
resource limits. Include the original attempt's active time in the total cap.
