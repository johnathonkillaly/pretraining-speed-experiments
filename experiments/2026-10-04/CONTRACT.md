> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# MoE Night preregistration — 2026-10-04 MDT

Owner explicitly requested isolated MoE research, including training; this supersedes
the parked-MoE note for this experiment only. No production files or jobs change.

Primary question: distinguish sparse expert arithmetic, logical gradient shapes,
physical allocator storage, and optimizer state. MLX 0.32.2, M4 Max 128 GiB.
Reference: BoothGPT commit 3bdd2fc28761ad1fa4b8de1dcf03d888c6719165,
current H1 20x1024/3456, GQA 8/2, head dimension 128, vocabulary 32768,
RoPE theta 500000, tied embeddings. No pretrained weights used.

1. Microbenchmark before training. 8 experts, forced one-expert and balanced
   routing. Dense SwiGLU, dense execution of all experts, stacked slice dispatch,
   independent modules with full differentiation, independent selected-subtree
   differentiation, unsorted and sorted gather_mm. Float32 for gradient auditing;
   bfloat16 additional main-H1-width timing. Sweep 4/8/16 experts and token counts.
   Evaluate loss first, then gradient graph, including input gradients. Record
   first call and >=3 warmup / >=12 repeat windows, medians and dispersion.
   Physical gradient retention measured through get_active_memory after eval,
   separately from logical nbytes; zero tensors can have compressed storage.
   Routing fixed and excluded from dispatch benchmark; separately timed router.
   Compare sparse against all-expert execution and single FFN; do not conflate them.
2. Numerical equivalence and gradient checks precede training. Grouped dispatch
   has no capacity limit and drops no tokens. Routing uses selected full-softmax
   probabilities for both Top-1 and Top-2 (no selected-only normalization).
   Load balancing: E * sum(mean(probability) * assignment fraction), coefficient
   0.01; assignments detached. Top-1 retains a task gradient through its gate.
3. Training discovery pilot: 8 layers, width 256, intermediate 864, GQA 8/2,
   full original 32768 vocabulary, seq256, batch4; layers 3 and 7 (zero based)
   replaced with 8-expert MoE. Dense/Top-1/Top-2, AdamW (.9,.95), wd .1,
   clip1, LR3e-4, 60-step warmup, cosine to 0.1x. 1200 steps = 1,228,800
   loss tokens/config/seed. Seeds 42,43,44 if runtime permits. Shared tensors
   initialized identically; each expert cloned from corresponding initial dense
   FFN, router independently initialized. Runs sequential, same data order.
   Training data: bounded original v7 packed corpus; held-out original v7
   validation JSONL tokenized by original tokenizer. Never open protected banks.
   Validation at step0 and every200 steps on identical 32768 targets.
   Hash and freeze source/data/config before queued runs; defects require a
   dated pre-run amendment. Timing instrumentation uses synchronized phases;
   report its overhead and do not equate this with optimized production speed.
4. At equal tokens, report held-out loss and across-seed variability, and curves
   versus measured wall time. A small pilot cannot establish capability or
   specialization. Flag load collapse if any expert gets >80% of assignments
   or effective used-expert count falls below2 persistently. Record entropy,
   all expert loads and gradient norms, router gradients, auxiliary loss.
5. Memory limit 16GiB and cache1GiB; experiments should stay far below this.
   Per-process timeout, no indefinite retries, retain failures. If another
   substantial local ML job starts, pause the experiment between runs.
6. Optional Default MoE and ProbMoE: read primary papers; plan instead of
   speculative implementation if faithful math is impractical. No larger run
   without strong forward/backward, memory, and learning evidence.

Verdict: SCALE only if practical throughput, physical-memory behavior, and
equal-token validation give convincing evidence; otherwise CONTINUE, RESEARCH,
or STOP with a concrete next bounded experiment. No positive outcome presumed.
