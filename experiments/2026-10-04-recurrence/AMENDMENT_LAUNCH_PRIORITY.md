> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Launcher-priority amendment — 2026-10-04 MDT

Before any primary fit: the one-shot launchd job was created as ProcessType
Background, Nice10, LowPriorityIO. Its first dense10 tuning fit ran at about
179ms/update, much slower than direct smoke/backward timings. Removing the
Darwin background policy with taskpolicy -B succeeded for only our controller
and its child, but did not remove the throughput limitation. Attempting to
restore nice0 returned Permission denied and changed no priority. An initial
explicit /usr/bin/taskpolicy path was absent; that attempt made no mutation.

Pause only our controller with SIGSTOP and allow its current 600-step tuning
child to finish. Unload that one-shot job only after the child completes.
A separate run_suite_v2.py and launchd_v2.plist then continue the registered
protocol at Interactive/nice0 without a background IO clamp. These retain
16GiB/1GiB MLX limits, sequential GPU use, process/memory preflight, 30min/fit,
all source/model/data/optimizer/LR/depth/seed/token/evaluation settings, and
no automatic retry. They are separately frozen before launch.

Completed tuning fits are reused only after checking the exact configuration,
600 training records, final step600 development validation and no test event.
Their timing is not a primary result. An incomplete prior fit is refused, not
rerun. All primary fits must run under the new policy, with no existing primary
result reuse. Original frozen source, launcher and raw evidence are preserved.

This corrects an unintended environment throttle; it is not a learning-rule
change. No external job or process is modified. Runtime scheduling still does
not establish hardware power/thermal normalization.
