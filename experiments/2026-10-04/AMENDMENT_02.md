> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Pretraining smoke amendment, 2026-10-04 MDT

Before queued training, Top-1 smoke failed on the first differentiated step:
MLX gather_axis refuses VJP through routing indices produced by argpartition.
Explicitly stop_gradient the discrete indices, as intended by ordinary Top-K
routing; gradients still flow through selected full-softmax probabilities.
No optimizer step or model checkpoint occurred. Failure log and partial smoke
metrics retained. Repeat smoke with seed43. Also align per-window training
provenance with the already-fixed shuffle; data byte hashes stay unchanged.
