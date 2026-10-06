> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Recurrent depth, BitNet, and pretraining speed

Research review, 2026-10-04. This note adds no new training or timing results.
The owner asked about recurrent depth, speed-oriented pretraining tricks, and
Microsoft's ternary weights. Existing experiments and production are unchanged.

## Judgment

Sharing layers reduces unique parameters, gradient buffers and optimizer state.
It can also reduce optimizer work. It does not remove repeated forward/backward
executions: five blocks repeated four times still execute twenty blocks. Full
backpropagation must traverse those executions and accumulate into shared weights.
Activation memory still depends on executed depth unless checkpointing or
truncation changes it. Parameter compression alone is therefore not a proportional
pretraining speedup. Freed memory may enable a different batch/checkpoint policy;
that is a systems experiment with its own speed/quality controls.

The speed-oriented recurrent recipes worth investigating are:

1. Start with fewer recurrences and increase them during training. This spends
   fewer forward passes early and reduces distribution shift. The 2025 retrofit
   work demonstrates recurrence curricula in continued pretraining of mature
   models; it does not establish a gain for a fresh 300M BoothGPT.
2. Sample training depth so the model learns to operate at several loop counts.
   Keep shapes/compiled paths bounded if measuring actual wall time.
3. Backpropagate only through the final k loops. Earlier loops still run forward,
   but their hidden-state history is detached. This saves backward work and
   activation retention at the cost of an approximate/truncated gradient.
   It is different from exact activation checkpointing, which recomputes forwards
   to preserve the full gradient. Huginn's main recipe uses k=8 and injects the
   prelude representation every loop, retaining a gradient path to the prelude.
   A faithful port must preserve this architecture detail.

Inference from the recipes: a shared core with a restrained 1→2→4 loop curriculum
and last-1/2-loop backpropagation is a plausible bounded speed/quality study.
These loop counts are proposed experimental choices, not paper-validated optima.
Compare against dense and full-backprop recurrent controls with the same hybrid
Muon/AdamW recipe, tuning allowance, fresh data and validation set. Report both
equal-token and equal-wall-time learning. No throughput or quality gain is assumed.

## Microsoft ternary weights

The name is **BitNet b1.58**. Quantized BitLinear weights take values in
{-1, 0, +1}; a scale restores magnitude. The theoretical state alphabet needs
log2(3)≈1.585 bits, while practical kernels can use two-bit packing plus scales.

It is quantization-aware training: higher-precision latent weights are quantized
for the forward pass, and a straight-through estimator approximates the gradient
through rounding/clipping. Gradients and optimizer state remain higher precision.
Microsoft's 2B4T release explicitly separates BF16 master weights for training
from packed weights for inference. Its inputs/outputs/norms and other tensors
should not all be budgeted as 1.58-bit values.

This is therefore a credible model-storage/inference-efficiency direction, but
the published inference speedups do not establish a pretraining speedup on MLX
or an RTX 3080. A fake-quantization implementation which rounds weights and then
uses ordinary floating-point matmul adds quantization work without replacing the
matmul kernel. A native low-bit forward kernel alone also does not demonstrate
an efficient full training step; weight/input gradients and optimizer state count.
The official bitnet.cpp repository is an inference implementation.

Precision is not a speed ranking by itself. BF16/FP16 dense kernels can outperform
quantized paths with packing/dequantization overhead, particularly when processing
many training tokens at once. Low-bit inference can benefit from reduced weight
traffic with appropriate kernels. Those regimes need separate measurements.
The existing MLX QuantizedLinear implementation freezes its packed parameters;
converting a model to that class is not full-weight BitNet pretraining.

BitNet has its own optimization recipe. Its papers discuss larger learning rates,
a subsequent cooldown, and changing weight decay on latent weights. Those choices
are not evidence that the same changes improve dense BoothGPT training. Early
short-run loss can also misrepresent its later learning dynamics.

## What BoothGPT has already measured

The local CUDA handoffs document BF16 compute, FP32 masters/accumulation, Flash
Attention, model compilation, alternating activation checkpointing, and a compiled
batched Muon optimizer. The September 17 matched tests found roughly 36% throughput
improvement from compile/checkpoint policy; September 18 added roughly 5% from
batched/compiled optimizer execution. These are historical recorded measurements,
not a fresh inspection of the current Fedora job.

Those records put most update time in model forward/backward. Data loading was
negligible. Recommending Flash Attention, compilation, input prefetching or batched
Muon as newly untried options would ignore existing work. The next systems decision
needs an updated profile; the next architecture decision needs learning per GPU
hour. Recurrent truncation is more directly aimed at reducing training computation
than merely quantizing stored weights, while BitNet remains attractive if a small
deployed model is the objective. Neither justifies modifying the ongoing dense run.

## Primary references and local evidence

- [Geiping et al., recurrent-depth architecture/training](https://arxiv.org/html/2502.05171v1),
  especially training-depth sampling, truncated backpropagation and prelude injection.
- [McLeish et al., recurrence curricula and continued pretraining](https://arxiv.org/html/2511.07384v1).
- [BitNet b1.58 introduction](https://arxiv.org/html/2402.17764v1), including the
  ternary quantizer and inference-kernel measurements.
- [BitNet training paper, sections 2.2 and 2.3](https://jmlr.org/papers/volume26/24-2050/24-2050.pdf),
  including STE, latent master weights and high-precision gradients/optimizer state.
- [BitNet b1.58 2B4T technical report](https://arxiv.org/html/2504.12285v1), including
  separate training/inference artifacts and its two-stage training recipe.
- [Microsoft's inference framework](https://github.com/microsoft/BitNet).
- [MLX quantized layers, pinned 0.32.2 source](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/quantized.py).
- BoothGPT CUDA speed handoff (local resource omitted).
- BoothGPT CUDA optimizer handoff (local resource omitted).
- Existing BoothGPT recurrence plan (local resource omitted).
