"""Bounded optimizer state/semantics probe. See CONTRACT.md for scope."""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "2026-10-04"
sys.path.insert(0, str(PREVIOUS / "scripts"))
from custom_vjp_probe import Independent

E, D, H = 8, 256, 864
SHAPES = {"gate": (H, D), "up": (H, D), "down": (D, H)}
NAMES = ("adamw", "sgd", "momentum_sgd", "lion", "adafactor", "muon")


class PerExpertMuon(optim.Muon):
    """Same MLX Muon rule, but independent NS iteration over last two axes.

    Only 2-D matrices and 3-D expert stacks are accepted. Weight decay remains
    the stock coupled-L2 rule; all measured cases explicitly use zero decay.
    """

    def apply_single(self, gradient, parameter, state):
        if gradient.ndim == 2:
            return super().apply_single(gradient, parameter, state)
        if gradient.ndim != 3:
            raise ValueError(f"Expected expert matrix stack; got {gradient.shape}")
        if self.weight_decay != 0:
            gradient = gradient + self.weight_decay * parameter
        v = self.momentum * state["v"] + (1 - self.momentum) * gradient
        state["v"] = v
        update = gradient * (1 - self.momentum) + v * self.momentum if self.nesterov else v
        transpose_needed = update.shape[-2] > update.shape[-1]
        x = mx.swapaxes(update, -1, -2) if transpose_needed else update
        x = x / (mx.sqrt(mx.sum(x * x, axis=(-2, -1), keepdims=True)) + 1e-7)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(self.ns_steps):
            gram = x @ mx.swapaxes(x, -1, -2)
            polynomial = mx.addmm(b * gram, gram, gram, beta=1.0, alpha=c)
            x = mx.addmm(a * x, polynomial, x, beta=1.0, alpha=1.0)
        update = mx.swapaxes(x, -1, -2) if transpose_needed else x
        lr = self.learning_rate.astype(gradient.dtype)
        lr = lr * max(1, update.shape[-2] / update.shape[-1]) ** 0.5
        return parameter - lr * update


def make_optimizer(name):
    if name == "adamw":
        return optim.AdamW(.001, betas=(.9, .95), weight_decay=0)
    if name == "sgd":
        return optim.SGD(.001, momentum=0, weight_decay=0)
    if name == "momentum_sgd":
        return optim.SGD(.001, momentum=.95, weight_decay=0)
    if name == "lion":
        return optim.Lion(.001, weight_decay=0)
    if name == "adafactor":
        return optim.Adafactor(.001, beta_1=None, weight_decay=0,
                               scale_parameter=False, relative_step=False)
    if name in ("muon", "per_expert_muon"):
        cls = optim.Muon if name == "muon" else PerExpertMuon
        return cls(.02, momentum=.95, weight_decay=0, ns_steps=5)
    raise ValueError(name)


def total_bytes(tree):
    return sum(array.nbytes for _, array in tree_flatten(tree))


def make_trees(layout, activity):
    mx.random.seed(20261004)
    count = 1 if layout == "dense" else E
    parameters = [{key: mx.random.normal(shape) * .02 for key, shape in SHAPES.items()}
                  for _ in range(count)]
    gradients = [{key: mx.random.normal(shape) * .001 if activity == "all" or i == 0
                  else mx.zeros(shape) for key, shape in SHAPES.items()}
                 for i in range(count)]
    if layout == "dense":
        return parameters[0], gradients[0]
    if layout == "independent":
        return {"experts": parameters}, {"experts": gradients}
    return ({key: mx.stack([item[key] for item in parameters]) for key in SHAPES},
            {key: mx.stack([item[key] for item in gradients]) for key in SHAPES})


def relative_difference(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1e-30))


def numerical_checks():
    # Two real-width matrices; three steps check carried momentum as well as NS.
    mx.random.seed(117)
    p = mx.random.normal((2, H, D)) * .02
    independent = {str(i): p[i] for i in range(2)}
    batched = {"w": p}
    separate_opt, batch_opt = make_optimizer("muon"), make_optimizer("per_expert_muon")
    differences = []
    for _ in range(3):
        g = mx.random.normal(p.shape) * .001
        independent = separate_opt.apply_gradients({str(i): g[i] for i in range(2)}, independent)
        batched = batch_opt.apply_gradients({"w": g}, batched)
        mx.eval(independent, batched, separate_opt.state, batch_opt.state)
        combined = mx.stack([independent[str(i)] for i in range(2)])
        differences.append(relative_difference(batched["w"], combined))
        np.testing.assert_allclose(np.array(batched["w"]), np.array(combined), atol=2e-6, rtol=2e-4)

    # Changing expert one's gradient should not change expert zero's update.
    p = mx.zeros((2, 7, 5))
    g = mx.random.normal(p.shape)
    altered = mx.stack([g[0], mx.random.normal((7, 5)) * 3])
    coupling = {}
    for name in ("muon", "per_expert_muon"):
        left_opt, right_opt = make_optimizer(name), make_optimizer(name)
        left = left_opt.apply_gradients({"w": g}, {"w": p})["w"]
        right = right_opt.apply_gradients({"w": altered}, {"w": p})["w"]
        mx.eval(left, right)
        coupling[name] = {"expert_zero_update_relative_difference": relative_difference(left[0], right[0]),
                          "max_absolute_difference": float(mx.max(mx.abs(left[0] - right[0])))}
    assert coupling["muon"]["expert_zero_update_relative_difference"] > .01
    assert coupling["per_expert_muon"]["max_absolute_difference"] == 0

    # A zero task gradient does not imply that stateful optimizers stop updating.
    previous_activity = {}
    for name in NAMES:
        opt = make_optimizer(name)
        weights = {"w": mx.ones((7, 5)) * .1}
        weights = opt.apply_gradients({"w": g[0]}, weights)
        mx.eval(weights, opt.state)
        before = weights["w"]
        after = opt.apply_gradients({"w": mx.zeros_like(before)}, weights)["w"]
        mx.eval(after, opt.state)
        previous_activity[name] = float(mx.max(mx.abs(after - before)))
    assert previous_activity["sgd"] == 0
    assert previous_activity["adafactor"] == 0
    for name in ("adamw", "momentum_sgd", "lion", "muon"):
        assert previous_activity[name] > 0
    return {"passed": True, "batched_vs_independent_relative_differences": differences,
            "cross_expert_coupling": coupling,
            "zero_gradient_after_active_step_max_weight_change": previous_activity}


def gradient_checks():
    results = []
    for custom in (False, True):
        gc.collect(); mx.clear_cache()
        mx.random.seed(12345)
        model = Independent(custom)
        x = mx.random.normal((512, D))
        mx.eval(model.parameters(), x)
        fn = nn.value_and_grad(model, lambda xx: mx.mean(model(xx) ** 2))
        baseline = mx.get_active_memory()
        value, grads = fn(x)
        mx.eval(value, grads)
        loss = float(value)
        del value
        gc.collect()
        results.append({"custom_broadcast_zeros": custom, "loss": loss,
                        "logical_gradient_bytes": total_bytes(grads),
                        "physical_gradient_retained_bytes": mx.get_active_memory() - baseline})
        del model, x, grads, fn
        gc.collect(); mx.clear_cache()
    return results


def benchmark_case(name, layout, activity):
    gc.collect(); mx.clear_cache()
    params, grads = make_trees(layout, activity)
    mx.eval(params, grads)
    logical_parameter_bytes, logical_gradient_bytes = total_bytes(params), total_bytes(grads)
    opt = make_optimizer(name)
    windows = []
    for index in range(19):
        mx.reset_peak_memory()
        active_before = mx.get_active_memory()
        start = time.perf_counter()
        params = opt.apply_gradients(grads, params)
        mx.eval(params, opt.state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        peak_added = max(0, mx.get_peak_memory() - active_before)
        windows.append({"index": index, "update_ms": elapsed_ms,
                        "incremental_peak_bytes": peak_added})
    state_metadata = [{"path": path, "shape": list(value.shape), "bytes": value.nbytes}
                      for path, value in tree_flatten(opt.state)]
    state_logical = sum(item["bytes"] for item in state_metadata)
    matrix_state_logical = sum(item["bytes"] for item in state_metadata
                               if item["path"] not in ("learning_rate", "step"))
    # No arrays from tree_flatten survive in metadata. Evaluated weights/grads
    # stay alive; only optimizer/state are destroyed for this allocator delta.
    gc.collect()
    with_state = mx.get_active_memory()
    del opt
    gc.collect()
    state_physical = with_state - mx.get_active_memory()
    samples = windows[4:]
    times = [sample["update_ms"] for sample in samples]
    result = {"optimizer": name, "layout": layout, "activity": activity,
              "parameter_logical_bytes": logical_parameter_bytes,
              "gradient_logical_bytes": logical_gradient_bytes,
              "state_logical_bytes": state_logical,
              "matrix_state_logical_bytes": matrix_state_logical,
              "state_physical_retained_bytes": state_physical,
              "state_metadata": state_metadata, "windows": windows,
              "median_update_ms": float(np.median(times)),
              "q25_update_ms": float(np.percentile(times, 25)),
              "q75_update_ms": float(np.percentile(times, 75)),
              "median_incremental_peak_bytes": float(np.median([sample["incremental_peak_bytes"] for sample in samples]))}
    del params, grads
    gc.collect(); mx.clear_cache()
    return result


def preflight():
    records = subprocess.check_output(["ps", "-axo", "pid,rss,command"], text=True)
    substantial = []
    for line in records.splitlines()[1:]:
        fields = line.strip().split(maxsplit=2)
        if len(fields) == 3 and int(fields[1]) > 2 * 1024 * 1024:
            command = fields[2].lower()
            if any(term in command for term in ("python", "mlx", "ollama", "llama-server", "lm studio")):
                substantial.append(line.strip())
    if substantial:
        raise RuntimeError(f"Other substantial ML job detected: {substantial}")
    return {"substantial_ml_jobs": substantial,
            "memory_pressure": subprocess.check_output(["memory_pressure", "-Q"], text=True)}


def main():
    freeze = json.loads((ROOT / "FREEZE.json").read_text())
    for filename, expected in freeze["sha256"].items():
        actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Freeze mismatch: {filename}")
    mx.set_memory_limit(16 * 1024**3)
    mx.set_cache_limit(1024**3)
    start = time.time()
    results = {"preflight": preflight(), "platform": platform.platform(),
               "mlx_version": __import__("importlib.metadata").metadata.version("mlx"),
               "optimizer_source_sha256": hashlib.sha256(Path(inspect.getfile(optim.Muon)).read_bytes()).hexdigest()}
    results["numerical_checks"] = numerical_checks()
    results["gradient_checks"] = gradient_checks()
    results["cases"] = []
    for layout in ("dense", "independent", "stacked"):
        for activity in (("all",) if layout == "dense" else ("all", "one")):
            names = NAMES + (("per_expert_muon",) if layout == "stacked" else ())
            for name in names:
                case = benchmark_case(name, layout, activity)
                results["cases"].append(case)
                print(json.dumps({key: case[key] for key in ("optimizer", "layout", "activity", "median_update_ms", "matrix_state_logical_bytes", "state_physical_retained_bytes")}), flush=True)
                (ROOT / "metrics.json").write_text(json.dumps(results, indent=2))
    results["elapsed_seconds"] = time.time() - start
    results["postflight"] = preflight()
    (ROOT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Completed {len(results['cases'])} cases in {results['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
