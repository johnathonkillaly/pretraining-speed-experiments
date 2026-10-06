> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

**CONTINUE — fixed four-loop recurrence with last-two-loop backpropagation is a
promising native-pretraining candidate in this pilot.** Keep the current hybrid
Muon/AdamW recipe and prelude injection as the measured starting point. Leave the
short depth curriculum out of the next recipe: it saved only 0.8% training time
and increased final test loss by 0.0279 nats in all three paired seeds.

Truncation improved final test loss in all three seeds relative to full recurrence
(mean −0.0374 nats) and Dense 10 (−0.0486), while increasing throughput by 18.0% and
26.0%, respectively. It reduced training time by 15.2% versus full recurrence and
20.7% versus Dense 10. Its 11.70M unique parameters are 29.8% fewer than Dense 10's 16.67M.
These speed and parameter savings are distinct: truncation does not shrink the
gradient tensor for any unique recurrent weight.

Dense 4 remains fastest in tokens/s, but has higher final loss. At each seed's
common measured training endpoint (about 184s), interpolated mean development
loss is 7.272 for fixed-depth truncation, versus 7.335 for Dense 4, 7.381 for full
recurrence and 7.445 for Dense 10. The ordering also holds with checkpoint elapsed
time, including validation/Python overhead. These are descriptive points on
fixed-token schedules, rather than models separately fitted to equal wall budgets.

The quality mechanism needs more work. Full recurrence exceeded gradient norm 1
on 30.5% of updates; truncation did so on 2.4%. The common global clipping rule is
unchanged, but truncation changes the gradient and therefore clipping/momentum
behavior. Lower loss could reflect that interaction, truncation's gradient bias,
regularization or several of these. The result supports this recipe's tradeoff;
it does not isolate one mechanism. The source breakdown also has limits: the
largest gains are on general/synthetic prose; the 32-document Stack Exchange
mean is effectively tied with Dense 10, slightly higher for truncation.

The next registered attribution experiment should change only the clipping
threshold, comparing full and truncated recurrence under a higher threshold
with the same fresh-data/token/order controls and numerical safeguards. That
would test whether the quality advantage survives largely removing the different
clipping activity. It is a proposed follow-up, not an additional run launched here.

If that holds, a concrete next geometry is d512/h1728, GQA 8/2/head 64, context 512,
the same 1-prelude/2-core/1-coda layout, R4/K2 and no curriculum: about 30.02M unique
parameters, alongside Dense 4 (~30.02M), Dense 10 (~49.88M) and full recurrence.
Use multiple seeds and a longer fresh token budget, with a new bounded LR sweep.
Doubling width increases transformer work relative to the shared vocabulary head,
so it tests a more relevant cost balance. Its runtime and quality have not been
measured; this is a research next step, not a speed guarantee or mature-training
plan. Substantially longer pretraining and separate capability evaluation remain
necessary before adopting an architecture.

For the owner's long 298M pretraining job, the existing CUDA pipeline remains the
measured throughput choice. The H1-shaped Mac probe provides practical context,
with the code/date/data qualifications above; it does not settle Apple's hardware
ceiling, MLX/CUDA performance, energy efficiency or every larger-memory workload.
No production change or automatic scale-up follows from this report.
