> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Implementation details registered before GPU execution

October 5, 2026, 9pm America/Denver. The original plan and thresholds are unchanged.

The new worker imports the frozen loss/model/optimizer helpers without invoking
their main functions. It emits descriptive metrics, identities and tensor-level
error summaries only; it saves no weights, input arrays or tensor samples.

One correctness process runs context 64, then a fresh process runs actual context
4095. Each retains the B1 accumulated reference gradient while comparing B2, B4
and B8, without optimizer updates. The full-context check has a 600-second worker
deadline and the small-context check 300 seconds; both count against the original
1800-second aggregate worker budget. Timing workers retain the 300-second deadline.

B8 is attempted only when the conservative full-context peak forecast is at most
76GiB: peak(B4) + 2 * max(0, peak(B4) - peak(B2)). This leaves 4GiB below the active
allocation stop for approximation error. The forecast uses observed correctness
peaks, which include the retained reference gradients and may exceed timing peaks.
A capacity skip is an explicit result, not a failed speed comparison. Other arms
remain bounded by the registered memory, RSS and OS-pressure stops.

Loss/gradient tolerance failures are preserved; finite arms may be timed as
diagnostic, as allowed by PLAN.md. A nonfinite, safety or allocation failure stops
the campaign and reports its completed subset. No optional FP32 diagnostic is
needed to make a throughput decision. B1 timing repeats with the same 80GiB
allowance as all candidates to avoid comparing against the prior daytime run.

The independent monitor reads process executable names and emits only owned
worker commands and safe competitor summaries, never arbitrary command lines.
The aggregate wall clock includes terminated workers and failures. Runs are
sequential with nominal-state cooldowns and no competing training.
