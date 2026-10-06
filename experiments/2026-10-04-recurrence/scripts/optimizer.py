"""Explicit hybrid ownership and independently batched 2-D Muon updates."""
from collections import defaultdict

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

SUFFIXES = tuple(f"{role}.weight" for role in (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))


def zeropower(x, steps=5):
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = mx.swapaxes(x, -1, -2)
    x = (x / (mx.sqrt(mx.sum(x * x, axis=(-2, -1), keepdims=True)) + 1e-7)).astype(mx.bfloat16)
    for _ in range(steps):
        gram = x @ mx.swapaxes(x, -1, -2)
        polynomial = mx.addmm(-4.7750 * gram, gram, gram, beta=1., alpha=2.0315)
        x = mx.addmm(3.4445 * x, polynomial, x, beta=1., alpha=1.)
    if transpose:
        x = mx.swapaxes(x, -1, -2)
    return x.astype(mx.float32)


class Hybrid:
    def __init__(self, model, masters):
        self.model = model
        self.names = tuple(sorted(masters))
        self.muon_names = tuple(name for name in self.names if name.endswith(SUFFIXES))
        self.adam_names = tuple(name for name in self.names if name not in self.muon_names)
        for name in self.muon_names:
            assert masters[name].ndim == 2, (name, masters[name].shape)
        for name in self.adam_names:
            assert name == "embed_tokens.weight" or masters[name].ndim == 1, name
        shapes = defaultdict(list)
        for name in self.muon_names:
            shapes[masters[name].shape].append(name)
        self.groups = [names[start:start + 5] for names in shapes.values()
                       for start in range(0, len(names), 5)]
        self.state = {"masters": masters,
                      "muon_m": {name: mx.zeros_like(masters[name]) for name in self.muon_names},
                      "adam_m": {name: mx.zeros_like(masters[name]) for name in self.adam_names},
                      "adam_v": {name: mx.zeros_like(masters[name]) for name in self.adam_names},
                      "step": mx.array(0, dtype=mx.uint32)}
        mx.eval(self.state)

    def apply(self, gradients, muon_lr, adam_lr):
        gradients = {name: value.astype(mx.float32) for name, value in gradients.items()}
        gnorm = mx.sqrt(sum((mx.sum(gradients[name] ** 2) for name in self.names), mx.array(0.)))
        scale = mx.minimum(1., 1. / (gnorm + 1e-6))
        gradients = {name: value * scale for name, value in gradients.items()}
        self.state["step"] = self.state["step"] + 1
        masters = self.state["masters"]
        for names in self.groups:
            updates = []
            for name in names:
                g = gradients[name]
                momentum = .95 * self.state["muon_m"][name] + .05 * g
                self.state["muon_m"][name] = momentum
                updates.append(.05 * g + .95 * momentum)
            update = zeropower(mx.stack(updates))
            rows, cols = masters[names[0]].shape
            rate = muon_lr * max(1, rows / cols) ** .5
            for index, name in enumerate(names):
                masters[name] = masters[name] - rate * update[index]
        step = self.state["step"].astype(mx.float32)
        for name in self.adam_names:
            g = gradients[name]
            m = .9 * self.state["adam_m"][name] + .1 * g
            v = .95 * self.state["adam_v"][name] + .05 * g * g
            self.state["adam_m"][name] = m
            self.state["adam_v"][name] = v
            mhat = m / (1 - .9 ** step)
            vhat = v / (1 - .95 ** step)
            masters[name] = masters[name] - adam_lr * mhat / (mx.sqrt(vhat) + 1e-8)
        self.model.update(tree_unflatten([(name, masters[name].astype(mx.bfloat16)) for name in self.names]))
        return gnorm

    def ownership(self):
        return {"muon_tensors": len(self.muon_names), "adamw_tensors": len(self.adam_names),
                "muon_parameters": sum(self.state["masters"][name].size for name in self.muon_names),
                "adamw_parameters": sum(self.state["masters"][name].size for name in self.adam_names),
                "matrix_groups": self.groups,
                "state_logical_bytes_excluding_masters": sum(value.nbytes for name, value in tree_flatten(self.state) if not name.startswith("masters."))}
