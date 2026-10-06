> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Optional routing and sparse-gradient follow-ups

These are implementation plans, not measured training variants. A/B/C and
the allocator investigation have priority. Primary code was inspected read-only;
pinned repository revisions and local content hashes are in
`configs/source_references/`. Reference code was never executed.

## Default MoE

[Paper v3](https://arxiv.org/html/2504.12463v3),
[official implementation](https://github.com/vatsal0/default-moe/blob/962381f9bd8d911e2d1d33b050322aa71359166c/megatron/model/default_vector.py).
The method maintains an EMA vector per expert from already-computed selected
outputs and supplies default outputs for missing experts. The inspected code
masks selected routing scores before multiplying the remaining scores by the
default-vector matrix.

An MLX port would retain an E×D detached buffer, aggregate selected outputs by
expert, update its EMA, and add `probs @ defaults` minus selected default
contributions. No additional expert GEMMs are needed; bookkeeping and an E×D
projection remain. Update order, zero-load experts, training/evaluation behavior,
and whether approximation changes forward values or only router gradients must
match the chosen reference exactly. Tests should compare the router VJP with
a small explicit reference and count executed experts. EMA state belongs in
checkpoints. A vanilla straight-through trick with arbitrary vectors would not
be a validated Default MoE replication. Defer Run D until the basic pilot's
learning and state costs justify extending it.

## ProbMoE

[Paper](https://arxiv.org/html/2606.01509v1),
[official exact-k code](https://github.com/HengHugoZhao/ProbMoE/blob/3f0e538e8cab9b4b91f12479a0c0040c843723d2/train/olmoe/open_instruct/OLMoE/v1/ProbMoE_V1_olmoe_exact.py).
The reference uses cardinality-constrained subset sampling, exact marginals,
and a marginal straight-through mask combined with softmax weights. The sparse
implementation then gathers weights only for sampled experts. This is not
ordinary noisy Top-K or unconstrained independent Bernoulli sampling.

Start an MLX exact-k port at E=8, k=1/2. Implement the log-space probability
dynamic program and conditional sampler from the reference, including its
non-selection-factor convention. Validate normalizers, marginals, and sampled
frequencies against enumeration of all 8-expert subsets. Check nested autodiff
through marginals against finite differences. Preserve fixed-shaped sampled
indices for the existing sorted dispatch. The inspected implementation's
gather means the expert-output sensitivities remain attached to sampled experts;
the marginal Jacobian couples router logits without executing every expert.
Dynamic-k requires padding/masking and separate cardinality accounting. Defer
Run E; a faithful, tested sampler and second-derivative path need their own
experiment rather than an invented overnight approximation.

## Sparse expert gradients and optimizer state

The selected-subtree VJP and real nn.Module probe already provide an exact way
to avoid unused expert gradients for fixed routing: differentiate only parameters
that participated, preserving input VJPs for previous layers. A full transformer
needs dispatch-aware parameter selection at each layer. In a balanced batch,
all experts often participate even though each token uses only one.

[MLX custom_function](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html)
supports custom VJPs, and an exploratory complete-SwiGLU probe now demonstrates
that a fixed-route independent module can return zero cotangents as broadcast
views: all24 logical gradient shapes remain, but physical gradient storage drops
from21,233,664 to2,654,212 bytes. Selected gradients match ordinary autodiff;
generic gradient clipping preserves the compression in this test. AdamW still
allocates its full42,467,340-byte state. Source and raw windows are in
`scripts/custom_vjp_probe.py` and `metrics/custom_vjp_swiglu.json`.

This is a fixed-one-expert proof, not a general dynamic MoE training port.
For stacked inputs, concatenating selected dW with missing-expert zeros would
still materialize the full gradient. A practical interface would carry
expert IDs and local dW buffers outside the ordinary full-parameter gradient
tree and use a dedicated optimizer, or split parameters and differentiate the
selected subtree. No custom sparse-array gradient type was demonstrated here.

Exact AdamW equivalence is subtle: an unused expert's moments still decay,
nonzero historical momentum can move it, and weight decay applies even when
the current data gradient is zero. Simply skipping updates changes the optimizer.
Lazy expert-local updates need global and local step bookkeeping and a defined
weight-decay policy, verified against dense AdamW over changing touched sets.
Once every expert has been visited, optimizer state may still approach the
full footprint, even if per-step gradients are local.

Each linear weight gradient is a product of token activations and output
sensitivities, with rank bounded by the number of tokens assigned to that expert.
Keeping these factors could defer dW materialization. This helps only when
assigned-token count is small relative to matrix dimensions. Elementwise Adam
moments, squaring, and division prevent a straightforward exact low-rank Adam
representation; compressed gradients would require a separate approximation
study. SGD or blockwise updates are easier proof-of-concept targets.
