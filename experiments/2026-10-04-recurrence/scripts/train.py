"""Compiled mixed-precision learning comparison; no production dependency."""
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from model import KINDS, ROOT, LanguageModel, convert_to_bf16, depths, objective
from optimizer import Hybrid


class Trainer:
    def __init__(self, kind, seed, args=None, checkpoint_mode="none"):
        self.model = LanguageModel(kind, seed, args=args, checkpoint_mode=checkpoint_mode)
        masters = convert_to_bf16(self.model)
        self.optimizer = Hybrid(self.model, masters)
        self.capture = [self.model.state, self.optimizer.state]
        self.step_functions = {}
        self.eval_functions = {}

    def step_function(self, loops, k, compiled=True):
        key = loops, k, compiled
        if key not in self.step_functions:
            value_grad = nn.value_and_grad(self.model, lambda x, y: objective(self.model, x, y, loops, k))
            def step(x, y, muon_lr, adam_lr):
                value, grads = value_grad(x, y)
                flat = dict(tree_flatten(grads))
                gnorm = self.optimizer.apply(flat, muon_lr, adam_lr)
                return value, gnorm
            self.step_functions[key] = mx.compile(step, inputs=self.capture, outputs=self.capture) if compiled else step
        return self.step_functions[key]

    def evaluation_function(self, loops=4):
        if loops not in self.eval_functions:
            def evaluate(x, y):
                logits = self.model(x, loops, loops).astype(mx.float32)
                values = nn.losses.cross_entropy(logits, y, reduction="none")
                return mx.mean(values, axis=-1)
            self.eval_functions[loops] = mx.compile(evaluate, inputs=self.model.state)
        return self.eval_functions[loops]


def validate(trainer, data):
    outputs = []
    fn = trainer.evaluation_function(4)
    for start in range(0, len(data), 8):
        batch = mx.array(data[start:start + 8].astype(np.int32))
        values = fn(batch[:, :-1], batch[:, 1:])
        mx.eval(values)
        outputs.extend(np.array(values).tolist())
    return float(np.mean(outputs)), outputs


def schedule(step, steps):
    warmup = max(1, int(.05 * steps))
    if step <= warmup:
        return step / warmup
    return .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))


def verify_freeze():
    freeze = json.loads((ROOT / "FREEZE.json").read_text())
    for filename, expected in freeze["sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename


def save(trainer, tag, step):
    path = ROOT / "checkpoints" / tag
    path.mkdir(exist_ok=False)
    trainer.model.save_weights(str(path / "model.safetensors"))
    mx.savez(str(path / "training_state.npz"), **dict(tree_flatten(trainer.optimizer.state)))
    (path / "metadata.json").write_text(json.dumps({"step": step, "tokens_seen": step * 2048}, indent=2))


def run(kind, seed, steps, muon_peak, tag, tune=False, smoke=False):
    mx.set_memory_limit(16 * 1024 ** 3)
    mx.set_cache_limit(1024 ** 3)
    if not smoke:
        verify_freeze()
    train_data = np.load(ROOT / "data/train.npy", mmap_mode="r")
    development = np.load(ROOT / "data/development.npy")
    assert steps * 8 <= len(train_data)
    # Reordering is identical for all variants within a seed, without cycling.
    order = np.random.default_rng(seed).permutation(len(train_data))[:steps * 8]
    trainer = Trainer(kind, seed)
    parameter_count = sum(array.size for _, array in tree_flatten(trainer.model.parameters()))
    base = {"kind": kind, "seed": seed, "steps": steps, "tune": tune,
            "muon_peak": muon_peak, "adamw_peak": .0003, "parameter_count": parameter_count,
            "shared_initialization_sha256": trainer.model._shared_initialization_sha256,
            "optimizer_ownership": trainer.optimizer.ownership(),
            "dtype": "BF16 model and gradients; FP32 masters/clipping/optimizer state",
            "batch_size": 8, "sequence_length": 256, "targets_per_update": 2048,
            "compiled": True, "weight_decay": 0,
            "order_sha256": hashlib.sha256(order.tobytes()).hexdigest()}
    start = time.perf_counter()
    train_wall = 0.
    peak = 0
    signatures_seen = {}
    steady_times = []
    with (ROOT / "metrics" / f"{tag}.jsonl").open("x") as file:
        def log(row):
            file.write(json.dumps(row) + "\n")
            file.flush()
        initial, _ = validate(trainer, development[:8] if smoke else development)
        log({"event": "validation", "step": 0, "tokens": 0, "loss": initial,
             "training_seconds": 0., "elapsed_seconds": time.perf_counter() - start})
        print(tag, "INITIAL", initial, flush=True)
        previous_signature = None
        for step in range(1, steps + 1):
            rows = train_data[order[(step - 1) * 8:step * 8]].astype(np.int32)
            batch = mx.array(rows)
            factor = schedule(step, steps)
            rates = mx.array(muon_peak * factor), mx.array(.0003 * factor)
            mx.eval(batch, rates)
            loops, k = depths(kind, step, steps)
            signature = loops, k
            fn = trainer.step_function(loops, k)
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            value, gnorm = fn(batch[:, :-1], batch[:, 1:], *rates)
            mx.eval(value, gnorm, trainer.capture)
            elapsed = time.perf_counter() - t0
            train_wall += elapsed
            peak = max(peak, mx.get_peak_memory())
            loss_value, norm_value = float(value), float(gnorm)
            if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                raise RuntimeError(f"Nonfinite result at {tag}:{step}")
            signatures_seen[signature] = signatures_seen.get(signature, 0) + 1
            warmed_up = signatures_seen[signature] > 50
            if warmed_up:
                steady_times.append(elapsed)
            log({"event": "train", "step": step, "tokens": step * 2048,
                 "loss": loss_value, "gradient_norm": norm_value,
                 "loops": loops, "backprop_loops": k,
                 "muon_lr": muon_peak * factor, "adamw_lr": .0003 * factor,
                 "step_ms": elapsed * 1000, "steady": warmed_up,
                 "training_seconds": train_wall, "elapsed_seconds": time.perf_counter() - start,
                 "peak_active_bytes": mx.get_peak_memory()})
            del value, gnorm, batch, rates
            if step % 500 == 0 or step == steps or (previous_signature is not None and signature != previous_signature):
                val, _ = validate(trainer, development[:8] if smoke else development)
                log({"event": "validation", "step": step, "tokens": step * 2048, "loss": val,
                     "training_seconds": train_wall, "elapsed_seconds": time.perf_counter() - start})
                print(tag, "STEP", step, "DEV", val, "TOK/S", 2048 / elapsed, flush=True)
            previous_signature = signature
        if not tune and not smoke:
            test, documents = validate(trainer, np.load(ROOT / "data/test.npy"))
            log({"event": "final_test", "step": steps, "tokens": steps * 2048,
                 "loss": test, "per_document_loss": documents,
                 "training_seconds": train_wall, "elapsed_seconds": time.perf_counter() - start})
            base["final_test_loss"] = test
            save_start = time.perf_counter()
            save(trainer, tag, steps)
            base["checkpoint_seconds"] = time.perf_counter() - save_start
        base.update({"initial_development_loss": initial, "final_development_loss": val,
                     "training_seconds_including_compile": train_wall,
                     "elapsed_seconds": time.perf_counter() - start,
                     "steady_tokens_per_second": 2048 / float(np.mean(steady_times)) if steady_times else None,
                     "steady_median_step_ms": float(np.median(steady_times)) * 1000 if steady_times else None,
                     "peak_active_bytes": peak,
                     "compiler_signatures": [{"loops": key[0], "k": key[1], "calls": count}
                                             for key, count in signatures_seen.items()]})
        (ROOT / "metrics" / f"{tag}_summary.json").write_text(json.dumps(base, indent=2) + "\n")
    print(tag, "COMPLETE", json.dumps(base), flush=True)
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--muon-peak", type=float, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.kind, args.seed, args.steps, args.muon_peak, args.tag, args.tune, args.smoke)
