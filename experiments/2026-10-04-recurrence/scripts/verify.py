"""Numerical tests of stop boundaries, injection, ownership and compilation."""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from model import ROOT, LanguageModel, ModelArgs, objective
from optimizer import zeropower
from train import Trainer


def arrays(tree):
    return {name: np.array(value) for name, value in tree_flatten(tree)}


def gradient_norm(tree):
    return float(mx.sqrt(sum((mx.sum(value.astype(mx.float32) ** 2) for _, value in tree_flatten(tree)), mx.array(0.))))


def main():
    mx.set_memory_limit(16 * 1024 ** 3)
    mx.set_cache_limit(1024 ** 3)
    args = ModelArgs(model_type="llama", hidden_size=32, intermediate_size=64,
                     num_hidden_layers=10, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=8, rms_norm_eps=1e-5, vocab_size=128,
                     rope_theta=500000, max_position_embeddings=16)
    model = LanguageModel("rec_full", 71, args)
    tokens = mx.array(np.random.default_rng(5).integers(1, 128, size=(2, 17)), dtype=mx.int32)
    x, y = tokens[:, :-1], tokens[:, 1:]
    reference = nn.value_and_grad(model, lambda a, b: objective(model, a, b, 4, 4))(x, y)
    same = nn.value_and_grad(model, lambda a, b: objective(model, a, b, 4, 8))(x, y)
    truncated = nn.value_and_grad(model, lambda a, b: objective(model, a, b, 4, 2))(x, y)
    mx.eval(reference, same, truncated)
    for name, value in arrays(reference[1]).items():
        np.testing.assert_array_equal(value, arrays(same[1])[name])
    np.testing.assert_array_equal(np.array(reference[0]), np.array(truncated[0]))
    dense4 = LanguageModel("dense4", 71, args)
    one = model(x, 1, 1)
    dense = dense4(x)
    mx.eval(one, dense)
    np.testing.assert_array_equal(np.array(one), np.array(dense))
    norms = {name: gradient_norm(truncated[1][name]) for name in ("embed_tokens", "prelude", "core", "coda")}
    assert all(value > 0 for value in norms.values()), norms

    # Independent per-loop scalars make the excluded prefix gradients observable.
    def toy(weights, k=2, prefix=None):
        state = mx.array(.3) if prefix is None else prefix
        start = 0 if prefix is None else 2
        for index in range(start, 4):
            if index == 4 - k and index > 0:
                state = mx.stop_gradient(state)
            state = mx.tanh(.5 * (state + .2) * weights[index])
        return state ** 2
    weights = [mx.array(.7 + .1 * index) for index in range(4)]
    grads = mx.grad(toy)(weights)
    mx.eval(grads)
    per_loop = [float(value) for value in grads]
    assert per_loop[0] == per_loop[1] == 0
    assert per_loop[2] != 0 and per_loop[3] != 0
    prefix = mx.array(.3)
    for index in range(2):
        prefix = mx.tanh(.5 * (prefix + .2) * weights[index])
    mx.eval(prefix)
    eps = 1e-3
    changed_plus = list(weights); changed_minus = list(weights)
    changed_plus[2] = weights[2] + eps
    changed_minus[2] = weights[2] - eps
    finite = float((toy(changed_plus, prefix=prefix) - toy(changed_minus, prefix=prefix)) / (2 * eps))
    np.testing.assert_allclose(finite, per_loop[2], atol=2e-6, rtol=2e-3)

    # Same NS operation independently versus batched; allow BF16 matmul drift.
    mx.random.seed(8)
    values = mx.random.normal((5, 64, 32))
    batched = zeropower(values)
    individual = mx.stack([zeropower(values[index:index + 1])[0] for index in range(5)])
    mx.eval(batched, individual)
    muon_difference = float(mx.linalg.norm(batched - individual) / mx.linalg.norm(individual))
    assert muon_difference < .02, muon_difference

    eager, compiled = Trainer("rec_tb2", 91, args), Trainer("rec_tb2", 91, args)
    lr = mx.array(.003), mx.array(.0003)
    eager_values = eager.step_function(4, 2, False)(x, y, *lr)
    compiled_values = compiled.step_function(4, 2, True)(x, y, *lr)
    mx.eval(eager_values, compiled_values, eager.capture, compiled.capture)
    lhs = mx.concatenate([value.reshape(-1) for _, value in tree_flatten(eager.optimizer.state["masters"])])
    rhs = mx.concatenate([value.reshape(-1) for _, value in tree_flatten(compiled.optimizer.state["masters"])])
    difference = float(mx.linalg.norm(lhs - rhs) / mx.linalg.norm(lhs))
    assert difference < 2e-3, difference
    np.testing.assert_allclose(float(eager_values[0]), float(compiled_values[0]), atol=.02, rtol=.002)
    report = {"passed": True, "full_vs_k_at_least_all_gradients_equal": True,
              "full_vs_truncated_forward_equal": True, "one_loop_vs_dense4_equal": True,
              "truncated_module_gradient_norms": norms, "toy_per_loop_gradients": per_loop,
              "conditional_finite_difference": finite, "conditional_analytic_gradient": per_loop[2],
              "batched_vs_individual_muon_relative_difference": muon_difference,
              "compiled_vs_eager_master_relative_difference": difference,
              "compiled_vs_eager_loss": [float(eager_values[0]), float(compiled_values[0])],
              "ownership": compiled.optimizer.ownership()}
    (ROOT / "metrics/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
