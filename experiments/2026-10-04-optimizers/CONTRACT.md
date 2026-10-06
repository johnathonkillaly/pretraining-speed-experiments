> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Optimizer follow-up preregistration

Question from the owner: does Muon change the MoE findings, and what about other
optimizers? This is an isolated follow-up. Existing frozen experiments and the
BoothGPT checkout remain unchanged.

Measure optimizer state and update cost, not convergence. Reuse the previous
FFN dimensions: float32 SwiGLU, d=256, h=864, E=8; three matrices per expert.
Compare one dense FFN, eight independent experts, and three stacked expert
tensors. Use fixed seeded gradients, either all experts active or only expert
zero active. Weight decay is explicitly zero to isolate update/state behavior;
learning rates are illustrative and are not comparable learning recipes.

Optimizers: AdamW, SGD without momentum, SGD with momentum, Lion, Adafactor
without first moment, and MLX 0.32.2 Muon. Add a batched per-expert Muon adapter
that orthogonalizes the final two dimensions separately. Stock stacked Muon is
a semantic diagnostic, not a valid per-expert Muon candidate.

Each case: first call, three additional warmups, fifteen synchronized update
windows, median and interquartile range. Time graph construction and evaluation
in eager MLX, excluding gradient computation, initialization and diagnostics.
Record logical state bytes and physical allocator retention after destroying
the optimizer while retaining evaluated weights/gradients. Include parameter
and gradient footprints separately. No compiled/full-training speed claims.

Before timing, verify that batched Muon agrees with independent 2-D Muon,
demonstrate stock stacked Muon's cross-expert coupling, and check whether a
previously active expert moves on the next zero-gradient step. Recheck ordinary
and broadcast-zero custom-VJP gradient memory. These are meaningful numerical
checks of the claims, not optimizer-quality tests.

Freeze source and this contract before executing. Guard memory at 16 GiB,
cache at 1 GiB, and stop if another substantial local ML job is running.
Retain failures; no unbounded retry. Do not run a speculative LR/convergence
bakeoff on the already severely undertrained pilot. A future learning ablation
must tune both optimizer recipes and compare dense/MoE under each at equal
tokens and wall time, with independent Muon and AdamW schedules in a hybrid.
