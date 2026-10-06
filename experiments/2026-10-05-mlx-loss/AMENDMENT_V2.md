> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Follow-up after the unchanged numerical rejection

2026-10-05, registered before any full-context H1 timing.

The diagnostic found exact repeated gradients for both loss implementations;
the BF16 composed/fused difference remains .8071% with or without checkpointing.
The same comparison on an FP32 model has relative L2 6.3754e-7. Standalone
loss/gradient checks all passed (worst relative L2 1.5579e-7). The observed full
BF16 difference is consistent with amplification of small arithmetic differences,
not demonstrated nondeterminism. This is an inference, not proof of learning parity.

The original .5% gate and original campaign remain failed. C and D are not
eligible for recommendation. No tolerance is relaxed and no failed check is
relabelled. Frozen v1 sources/records remain unchanged.

This separately registered v2 sequence measures A1,B1,U1,U2,B2,A2,E1,E2:

- A: old0.32.2 composed FP32 loss, alternating checkpointing (original arm).
- B: new0.32.3 composed FP32 loss, alternating checkpointing (original arm).
- U: new0.32.3 fused FP32 loss, alternating checkpointing. PERFORMANCE DIAGNOSTIC
  ONLY: the full-model numerical gate already failed. These synthetic runs may
  measure speed/memory and finite state; they cannot pass the original quality
  gate or justify deployment. Retain first-update gradient/master/state samples
  to describe drift without calling sample agreement a full-tensor guarantee.
- E: new0.32.3 composed FP32 loss, no activation checkpointing,32GiB allocation
  cap. Same capacity tradeoff as the preregistered exploratory plan, now retaining
  the composed loss so the rejected fused backward does not confound it.

Before timings, a new bounded gate directly compares checkpointed/noncheckpointed
composed BF16 gradients over every full-model parameter at context64. Keep the
original .5% gradient and2e-4 loss thresholds. E timing requires this gate to pass.
No new data, model shape, optimizer equation, precision, initialization, batch,
context, rate or synchronization changes. Each timing process retains the v1
three warmup/five measurement schedule and original worker source.

Aggregate worker cap stays900seconds, including4.586318458 seconds from v1 and
3.078352958 seconds from its diagnostic. The failed launcher used no GPU time.
Overall1800-second benchmark campaign deadline starts at the original controller
start, not the v2 restart. Same safety monitor and stop behavior. No extension on
failures. Report incomplete subsets if needed.

E versus B is the useful recipe decision: both paired rounds >=5% faster, its
new checkpointing gate passed, peak allocation<=32GiB, finite state and no monitor
event. B versus A and U versus B remain descriptive runtime comparisons. No
synthetic timing run proves real-data learning quality or checkpoint continuation.
