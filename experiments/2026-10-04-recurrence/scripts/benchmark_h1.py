"""Bounded H1-shaped throughput context; does not touch Fedora or checkpoints."""
import json
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from model import ROOT, ModelArgs, objective
from run_suite import preflight
from train import Trainer, verify_freeze


def main():
    verify_freeze()
    mx.set_memory_limit(16 * 1024 ** 3)
    mx.set_cache_limit(1024 ** 3)
    report = {"preflight": preflight(), "dtype": "BF16 model/gradients; FP32 masters/accumulation/state",
              "targets_per_update": 4095 * 8, "microbatch": 1, "accumulation": 8,
              "checkpointing": "alternating", "optimizer": "compiled independently batched Muon/AdamW",
              "input": "fixed seeded uniform synthetic token IDs; no learning interpretation",
              "nvidia_reference": "Historical 2026-09-18 local CUDA handoff; no fresh NVIDIA run"}
    args = ModelArgs(model_type="llama", hidden_size=1024, intermediate_size=3456,
                     num_hidden_layers=20, num_attention_heads=8, num_key_value_heads=2,
                     head_dim=128, rms_norm_eps=1e-5, vocab_size=32768,
                     rope_theta=500000, max_position_embeddings=4096, tie_word_embeddings=True)
    trainer = Trainer("dense20", 42, args, checkpoint_mode="alternating")
    report["parameters"] = sum(value.size for _, value in tree_flatten(trainer.model.parameters()))
    report["ownership"] = trainer.optimizer.ownership()
    value_grad = nn.value_and_grad(trainer.model, lambda x, y: objective(trainer.model, x, y, 1, 1))
    def gradients(x, y):
        loss, grad = value_grad(x, y)
        return loss, {name: value.astype(mx.float32) for name, value in tree_flatten(grad)}
    micro = mx.compile(gradients, inputs=trainer.model.state)
    def update(grad, muon_lr, adam_lr):
        return trainer.optimizer.apply(grad, muon_lr, adam_lr)
    update = mx.compile(update, inputs=trainer.capture, outputs=trainer.capture)
    batches = [mx.array(np.random.default_rng(500 + i).integers(1, 32768, size=(1, 4096)), dtype=mx.int32)
               for i in range(8)]
    muon_lr, adam_lr = mx.array(.00015), mx.array(.00006)
    mx.eval(batches, muon_lr, adam_lr)
    report["windows"] = []
    for index in range(13):
        mx.reset_peak_memory()
        start = time.perf_counter()
        accumulated = None
        losses = []
        for batch in batches:
            loss, grad = micro(batch[:, :-1], batch[:, 1:])
            if accumulated is None:
                accumulated = {name: value / 8 for name, value in grad.items()}
            else:
                accumulated = {name: accumulated[name] + value / 8 for name, value in grad.items()}
            mx.eval(loss, accumulated)
            losses.append(float(loss))
            del loss, grad
        backward_end = time.perf_counter()
        norm = update(accumulated, muon_lr, adam_lr)
        mx.eval(norm, trainer.capture)
        end = time.perf_counter()
        record = {"index": index, "warmup": index < 3,
                  "model_backward_accumulation_seconds": backward_end - start,
                  "optimizer_seconds": end - backward_end, "total_seconds": end - start,
                  "tokens_per_second": 32760 / (end - start),
                  "peak_active_bytes": mx.get_peak_memory(),
                  "mean_loss": float(np.mean(losses)), "gradient_norm": float(norm)}
        assert np.isfinite(record["mean_loss"]) and np.isfinite(record["gradient_norm"])
        report["windows"].append(record)
        (ROOT / "metrics/h1_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        del accumulated, norm
    sample = report["windows"][3:]
    report["mean_tokens_per_second"] = 32760 / float(np.mean([row["total_seconds"] for row in sample]))
    report["median_tokens_per_second"] = float(np.median([row["tokens_per_second"] for row in sample]))
    report["peak_active_bytes"] = max(row["peak_active_bytes"] for row in report["windows"])
    report["postflight"] = preflight()
    (ROOT / "metrics/h1_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print("H1 COMPLETE", report["mean_tokens_per_second"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (ROOT / "metrics/h1_benchmark_failure.json").write_text(json.dumps({"error": repr(error)}, indent=2) + "\n")
        raise
