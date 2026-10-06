"""Frozen, isolated numerical checks and H1 speed attribution; synthetic only."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

STUDY = Path(__file__).resolve().parents[1]
OLD = STUDY.parent / "2026-10-04-recurrence"
sys.path.insert(0, str(OLD / "scripts"))
from model import ModelArgs
from train import Trainer


def verify_freeze():
    record = json.loads((STUDY / "FREEZE.json").read_text())
    for filename, expected in record["sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def composed(logits, targets):
    z = logits - mx.stop_gradient(mx.max(logits, axis=-1, keepdims=True))
    score = mx.take_along_axis(z, mx.expand_dims(targets, -1), axis=-1).squeeze(-1)
    return mx.logsumexp(z, axis=-1) - score


def fused(logits, targets):
    return mx.fast.cross_entropy(logits, targets)


def loss_function(kind):
    return composed if kind == "composed" else fused


def ce_value_grad(kind, compiled):
    ce = loss_function(kind)
    def objective(logits, targets, cotangents):
        return mx.sum(ce(logits, targets).astype(mx.float32) * cotangents) / logits.shape[0]
    fn = mx.value_and_grad(objective)
    return mx.compile(fn) if compiled else fn


def tensor_errors(reference, candidate):
    a, b = reference.astype(mx.float32), candidate.astype(mx.float32)
    delta = b - a
    values = [mx.max(mx.abs(delta)), mx.sum(delta * delta), mx.sum(a * a),
              mx.all(mx.isfinite(a)), mx.all(mx.isfinite(b))]
    mx.eval(values)
    return {"max_abs": float(values[0]), "squared_difference": float(values[1]),
            "reference_squared_norm": float(values[2]),
            "relative_l2": math.sqrt(float(values[1]) / max(float(values[2]), 1e-30)),
            "finite": bool(values[3]) and bool(values[4])}


def logits_case(rows, vocab, seed, special="ordinary"):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 2, (rows, vocab)).astype(np.float32)
    targets = rng.integers(0, vocab, rows, dtype=np.int32)
    cotangents = rng.uniform(.25, 1.75, rows).astype(np.float32)
    if special == "extreme":
        values[:, 0] = 100
        values[:, 1] = -100
        targets[::2] = 1
    elif special == "offset":
        values += rng.uniform(-1000, 1000, (rows, 1)).astype(np.float32)
    elif special == "ties":
        values[:, :2] = 10
        targets[::2] = 0
    out = mx.array(values), mx.array(targets), mx.array(cotangents)
    mx.eval(out)
    return values, targets, cotangents, out


def numpy_oracle(values, targets, cotangents):
    z = values.astype(np.float64)
    z -= z.max(-1, keepdims=True)
    exp = np.exp(z)
    probabilities = exp / exp.sum(-1, keepdims=True)
    losses = np.log(exp.sum(-1)) - z[np.arange(len(z)), targets]
    grad = probabilities
    grad[np.arange(len(z)), targets] -= 1
    grad *= cotangents[:, None] / len(z)
    return float((losses * cotangents).sum() / len(z)), grad


def args_h1():
    return ModelArgs(model_type="llama", hidden_size=1024, intermediate_size=3456,
                     num_hidden_layers=20, num_attention_heads=8, num_key_value_heads=2,
                     head_dim=128, rms_norm_eps=1e-5, vocab_size=32768,
                     rope_theta=500000, max_position_embeddings=4096, tie_word_embeddings=True)


def model_objective(model, kind, x, y):
    logits = model(x, 1, 1).astype(mx.float32)
    mask = (y != 0).astype(mx.float32)
    return mx.sum(loss_function(kind)(logits, y) * mask) / mx.sum(mask)


def verify():
    report = {"mlx": mx.__version__, "cases": [], "timings": []}
    destination = STUDY / "metrics/correctness.json"
    for rows, vocab in [(128, 1024), (1024, 32768), (4095, 32768)]:
        specials = ("ordinary", "extreme", "offset", "ties") if rows == 128 else ("ordinary",)
        for offset, special in enumerate(specials):
            values, targets, cotangents, inputs = logits_case(rows, vocab, 800 + offset, special)
            for compiled in (False, True):
                la, ga = ce_value_grad("composed", compiled)(*inputs)
                lb, gb = ce_value_grad("fused", compiled)(*inputs)
                mx.eval(la, ga, lb, gb)
                errors = tensor_errors(ga, gb)
                row = {"shape": [rows, vocab], "special": special, "compiled": compiled,
                       "loss_abs": abs(float(la) - float(lb)), "gradient": errors}
                assert errors["finite"] and row["loss_abs"] <= 2e-4, row
                assert errors["max_abs"] <= 2e-5 and errors["relative_l2"] <= 2e-4, row
                if rows == 128:
                    oracle_loss, oracle_grad = numpy_oracle(values, targets, cotangents)
                    row["oracle"] = {"composed_loss_abs": abs(float(la) - oracle_loss),
                                     "fused_loss_abs": abs(float(lb) - oracle_loss),
                                     "fused_gradient_max_abs": float(np.max(np.abs(np.array(gb) - oracle_grad)))}
                    assert row["oracle"]["composed_loss_abs"] <= 2e-4, row
                    assert row["oracle"]["fused_loss_abs"] <= 2e-4, row
                    assert row["oracle"]["fused_gradient_max_abs"] <= 2e-5, row
                report["cases"].append(row)
                save(destination, report)
                del la, lb, ga, gb
            if special == "ordinary":
                order = ("composed", "fused") if rows != 1024 else ("fused", "composed")
                for kind in order:
                    fn = ce_value_grad(kind, True)
                    records = []
                    for iteration in range(12):
                        mx.reset_peak_memory()
                        start = time.perf_counter()
                        loss, grad = fn(*inputs)
                        mx.eval(loss, grad)
                        elapsed = time.perf_counter() - start
                        records.append({"iteration": iteration, "warmup": iteration < 3,
                                        "seconds": elapsed, "peak_active_bytes": mx.get_peak_memory()})
                        del loss, grad
                    report["timings"].append({"shape": [rows, vocab], "kind": kind, "records": records})
                    save(destination, report)
                # Diagnostic: raw fused BF16 loss, never a primary training arm.
                bf16_inputs = (inputs[0].astype(mx.bfloat16), inputs[1], inputs[2])
                baseline = (bf16_inputs[0].astype(mx.float32), inputs[1], inputs[2])
                la, ga = ce_value_grad("composed", False)(*baseline)
                lb, gb = ce_value_grad("fused", False)(*bf16_inputs)
                raw = fused(bf16_inputs[0], bf16_inputs[1])
                mx.eval(la, ga, lb, gb, raw)
                report.setdefault("bf16_diagnostic", []).append({"shape": [rows, vocab],
                    "raw_loss_dtype": str(raw.dtype), "loss_abs": abs(float(la) - float(lb)),
                    "gradient": tensor_errors(ga, gb)})
                save(destination, report)
                del la, ga, lb, gb, raw, bf16_inputs, baseline
            del values, targets, cotangents, inputs
            gc.collect()
            mx.clear_cache()
            print(json.dumps({"event": "ce_case_complete", "shape": [rows, vocab], "special": special}), flush=True)
    trainer = Trainer("dense20", 42, args_h1(), checkpoint_mode="alternating")
    batch = mx.array(np.random.default_rng(999).integers(1, 32768, size=(1, 65), dtype=np.int32))
    gradients = []
    for kind in ("composed", "fused"):
        fn = nn.value_and_grad(trainer.model, lambda x, y: model_objective(trainer.model, kind, x, y))
        loss, grad = fn(batch[:, :-1], batch[:, 1:])
        mx.eval(loss, grad)
        gradients.append((float(loss), dict(tree_flatten(grad))))
    errors = {name: tensor_errors(a, gradients[1][1][name]) for name, a in gradients[0][1].items()}
    relative = math.sqrt(sum(v["squared_difference"] for v in errors.values()) /
                         max(sum(v["reference_squared_norm"] for v in errors.values()), 1e-30))
    model_check = {"loss_abs": abs(gradients[0][0] - gradients[1][0]),
                   "relative_l2": relative, "parameter_count": sum(p.size for _, p in tree_flatten(trainer.model.parameters())),
                   "parameter_gradient_errors": errors}
    assert model_check["parameter_count"] == 298361856
    assert model_check["loss_abs"] <= 2e-4 and relative <= .005, model_check
    assert all(v["finite"] and v["reference_squared_norm"] > 0 for v in errors.values())
    report["model_gradient"] = model_check
    trainer.model._checkpoint_mode = "none"
    fn = nn.value_and_grad(trainer.model, lambda x, y: model_objective(trainer.model, "fused", x, y))
    loss, grad = fn(batch[:, :-1], batch[:, 1:])
    mx.eval(loss, grad)
    errors = {name: tensor_errors(a, dict(tree_flatten(grad))[name]) for name, a in gradients[1][1].items()}
    relative = math.sqrt(sum(v["squared_difference"] for v in errors.values()) /
                         max(sum(v["reference_squared_norm"] for v in errors.values()), 1e-30))
    report["no_checkpoint_gradient"] = {"loss_abs": abs(float(loss) - gradients[1][0]),
                                        "relative_l2": relative, "parameter_gradient_errors": errors}
    assert report["no_checkpoint_gradient"]["loss_abs"] <= 2e-4 and relative <= .005
    assert all(v["finite"] and v["reference_squared_norm"] > 0 for v in errors.values())
    report["passed"] = True
    save(destination, report)
    print(json.dumps({"event": "correctness_complete", "model_relative_l2": relative}), flush=True)


def sample_tree(tree):
    out = {}
    for name, value in sorted(tree_flatten(tree)):
        flat = value.reshape(-1)
        stride = max(1, flat.size // 128)
        out[name] = np.array(flat[::stride][:128].astype(mx.float32)).tolist()
    return out


def finite_tree(tree):
    checks = [mx.all(mx.isfinite(value)) for _, value in tree_flatten(tree)]
    mx.eval(checks)
    return all(bool(value) for value in checks)


def h1(kind, tag, checkpoint_mode):
    begin = time.perf_counter()
    trainer = Trainer("dense20", 42, args_h1(), checkpoint_mode=checkpoint_mode)
    parameters = sum(p.size for _, p in tree_flatten(trainer.model.parameters()))
    assert parameters == 298361856
    digest = hashlib.sha256()
    for name in trainer.optimizer.names:
        value = trainer.optimizer.state["masters"][name]
        digest.update(name.encode())
        digest.update(np.array(value).tobytes())
    initial_samples = sample_tree(trainer.optimizer.state["masters"])
    batches_np = [np.random.default_rng(500 + i).integers(1, 32768, size=(1, 4096)).astype(np.int32)
                  for i in range(8)]
    input_sha = hashlib.sha256(b"".join(value.tobytes() for value in batches_np)).hexdigest()
    batches = [mx.array(value) for value in batches_np]
    value_grad = nn.value_and_grad(trainer.model, lambda x, y: model_objective(trainer.model, kind, x, y))
    def gradients(x, y):
        loss, grad = value_grad(x, y)
        return loss, {name: value.astype(mx.float32) for name, value in tree_flatten(grad)}
    micro = mx.compile(gradients, inputs=trainer.model.state, outputs=trainer.model.state)
    update = mx.compile(trainer.optimizer.apply, inputs=trainer.capture, outputs=trainer.capture)
    muon_lr, adam_lr = mx.array(.00015), mx.array(.00006)
    mx.eval(batches, muon_lr, adam_lr)
    report = {"tag": tag, "mlx": mx.__version__, "loss": kind, "checkpointing": checkpoint_mode, "parameters": parameters,
              "initialization_sha256": digest.hexdigest(), "input_sha256": input_sha,
              "ownership": trainer.optimizer.ownership(), "initialization_seconds": time.perf_counter() - begin,
              "targets_per_update": 32760, "records": []}
    destination = STUDY / "metrics" / (tag + ".json")
    for index in range(8):
        mx.reset_peak_memory()
        start = time.perf_counter()
        accumulated = None
        losses = []
        for batch in batches:
            loss, grad = micro(batch[:, :-1], batch[:, 1:])
            accumulated = ({name: value / 8 for name, value in grad.items()} if accumulated is None else
                           {name: accumulated[name] + value / 8 for name, value in grad.items()})
            mx.eval(loss, accumulated)
            losses.append(float(loss))
            del loss, grad
        backward_end = time.perf_counter()
        # Extra numerical scans are outside timing and only on the first warmup.
        if index == 0:
            assert finite_tree(accumulated)
            report["first_gradient_samples"] = sample_tree(accumulated)
        optimizer_start = time.perf_counter()
        norm = update(accumulated, muon_lr, adam_lr)
        mx.eval(norm, trainer.capture)
        end = time.perf_counter()
        model_seconds = backward_end - start
        optimizer_seconds = end - optimizer_start
        total = model_seconds + optimizer_seconds
        row = {"index": index, "warmup": index < 3,
               "model_backward_accumulation_seconds": model_seconds, "optimizer_seconds": optimizer_seconds,
               "total_seconds": total, "tokens_per_second": 32760 / total,
               "peak_active_bytes": mx.get_peak_memory(), "cache_bytes": mx.get_cache_memory(),
               "mean_loss": float(np.mean(losses)), "gradient_norm": float(norm)}
        assert math.isfinite(row["mean_loss"]) and math.isfinite(row["gradient_norm"]), row
        if index == 0:
            report["first_master_samples"] = sample_tree(trainer.optimizer.state["masters"])
            report["first_state_samples"] = sample_tree(trainer.optimizer.state)
            assert any(a != b for name, original in initial_samples.items() for a, b in
                       zip(original, report["first_master_samples"][name]))
        report["records"].append(row)
        save(destination, report)
        print(json.dumps({"tag": tag, **row}), flush=True)
        del accumulated, norm
    report["finite_final_state"] = finite_tree(trainer.capture)
    assert report["finite_final_state"]
    sample = report["records"][3:]
    report["steady_targets_per_second"] = 32760 / np.mean([r["total_seconds"] for r in sample])
    report["median_targets_per_second"] = float(np.median([r["tokens_per_second"] for r in sample]))
    report["peak_active_bytes"] = max(r["peak_active_bytes"] for r in report["records"])
    report["completed"] = True
    save(destination, report)
    print(json.dumps({"event": "h1_complete", "tag": tag, "targets_per_second": report["steady_targets_per_second"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("verify", "h1"))
    parser.add_argument("--kind", choices=("composed", "fused"), default="composed")
    parser.add_argument("--tag", default="unused")
    parser.add_argument("--checkpoint-mode", choices=("alternating", "none"), default="alternating")
    args = parser.parse_args()
    verify_freeze()
    mx.set_memory_limit((32 if args.checkpoint_mode == "none" else 16) * 1024 ** 3)
    mx.set_cache_limit(1024 ** 3)
    try:
        verify() if args.stage == "verify" else h1(args.kind, args.tag, args.checkpoint_mode)
    except Exception as error:
        save(STUDY / "metrics" / (args.tag + "_failure.json"), {"error": repr(error)})
        raise
