> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# MLX batching with spare unified memory

Registered 2026-10-05 before any new GPU run. The owner requested a single
9pm America/Denver run tonight (2026-10-06 03:00 UTC) to test larger sequence
batches with roughly 80GB of memory available. They explicitly authorized
quitting any idle app normally. Active training and unsaved work are preserved.

## Question and fixed recipe

Does processing more sequences together improve steady training targets/second
for the existing 298,361,856-parameter H1 model at 4095 input positions?
Reuse the isolated MLX/Metal 0.32.3 environment and composed FP32 loss from
../2026-10-05-mlx-loss/scripts/worker.py, with all activation checkpointing off.
BF16 model/backward, FP32 masters, accumulation and optimizer state; retain the
same prior Mac Muon/AdamW equations, clipping and explicit learning rates.
This is a Mac recipe benchmark, not a production CUDA comparison.

Keep exactly the same eight synthetic 4096-ID sequences per optimizer update,
NumPy seeds 500..507 and model seed 42. No production weights, real corpus,
protected evaluation banks, Windows or Fedora access. The only algorithmic
variable is physical microbatch size versus accumulation count:

| Arm | Sequences together | Accumulation passes | Sequences per update |
|---|---:|---:|---:|
| B1 | 1 | 8 | 8 |
| B2 | 2 | 4 | 8 |
| B4 | 4 | 2 | 8 |
| B8 | 8 | 1 | 8 |

B8 is conditional on observed capacity, not assumed to fit. Each update still
has 32,760 targets. Average gradients and loss by total targets, apply clipping
once after accumulation and make exactly one identical optimizer update.

## Preparation, freezing and numerical gates

Build new worker/controller sources in this directory. Never edit the earlier
frozen campaign, imported sources or original environment. Commit this plan,
then commit and checksum new sources, environment, arguments and plan before
any GPU execution. Verify all imported freezes before and after the campaign.
Retain dated pre-run amendments and all failed gates without relaxing limits.

Check every initial full model/input hash matches across arms. Compare the
full accumulated gradient on the same eight sequences against B1, first at
context 64 and then at actual 4095 positions. Registered absolute mean loss
difference <=2e-4 and global gradient relative L2 <=0.005; every parameter must
be finite with nonzero gradients. Verify finite optimizer/master/state and
actual parameter changes. If a numerical gate fails, retain that arm's failure;
any subsequent timings are clearly diagnostic, never recommended. An optional
FP32 diagnostic may explain rounding differences but cannot replace the BF16
gate. Synthetic numerical agreement is not proof of learning quality.

## Capacity and monitoring

Allow up to 80GiB of MLX active allocation and retain the same 1GiB buffer cache
allowance for every arm. Record active/cache peaks and process/OS memory; MLX's
limit is a guideline, not an operating-system hard cap. Stop the current worker
if observed active allocation exceeds 80GiB or process RSS exceeds 85GiB.
Require AC, no competing ML/GPU job, nominal thermal state before each worker,
and memory-pressure free percentage >=20. Monitor thermal state at least every
two seconds and AC, memory and competing jobs periodically. Stop on serious or
critical thermal state, memory pressure below 20%, AC loss, nonfinite results,
competing training, allocation errors or user interruption. Normal/fair states
are recorded; temperature sensor values must not be invented. Sequential
workers only, with cooldowns between windows. Do not change wired-memory/OS,
power, fan, undervolt, sleep or login settings.

At the scheduled time, inventory active jobs and running apps before starting.
Quit only apps established to be idle, through normal Quit. Record which quit.
Never force-kill apps or terminate training, services, work processes or save
dialogs. If idle/unsaved state cannot be established, leave that app open. Keep
Codex running. If another GPU training job still owns the Mac, report the blocker
and leave it intact; do not poll indefinitely or start competing training.

## Timing, budget and decision

Run two fresh-process rounds, ascending B1,B2,B4,B8 then descending B8,B4,B2,B1.
Each run has three warmup and five measured optimizer updates. Preserve the
existing per-microbatch materialization and optimizer synchronization boundaries;
do not combine batching with pipeline, loss, precision or evaluation changes.
Report all timings, aggregate targets/mean update duration, model/backward/
accumulation time, optimizer time, compile/warmup costs and memory peaks.

Cap total benchmark worker wall time, including numerical checks, initialization,
compilation and measurements, at 1800 seconds; whole campaign at 5400 seconds;
each timing worker at 300 seconds. If capacity or budget prevents later arms,
report the completed subset and explicit limit. Do not silently extend budgets.

A candidate is useful when its numerical/safety gates pass and it improves
throughput by at least 5% over the same-round B1 in both rounds. Call the gain
strong only if both rounds improve by at least 10%. Report individual rounds and
order/thermal/background-load limits. Select by measured targets/second, not
by allocated RAM. No measured gain is promised in advance.

## Preservation and completion

Keep all raw measurements and failures. Source, contracts, descriptive metrics
and report belong in Git; synthetic data and tensor/weight/state fragments stay
ignored and checksum-backed on SHUTTLE where accessible. Do not delete or change
unrelated files. Write REPORT.md and a compact throughput/memory comparison.
Confirm this campaign's GPU processes have ended. Post one completion report or
an actionable failure in this chat; avoid unchanged status notifications. This
is a one-off scheduled test, not a recurring nightly experiment.
