"""Uncompiled synchronized forward/backward audit of the real pilot modules."""
import gc
import json
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from model import ROOT, LanguageModel, convert_to_bf16, objective
from train import verify_freeze


def main():
    verify_freeze()
    mx.set_memory_limit(16 * 1024 ** 3)
    mx.set_cache_limit(1024 ** 3)
    output = []
    for kind in ("dense10", "dense4", "rec_full", "rec_tb2"):
        gc.collect(); mx.clear_cache()
        model = LanguageModel(kind, 42)
        masters = convert_to_bf16(model)
        # Masters belong to training, not this isolated backward memory window.
        del masters
        batch = mx.array(np.load(ROOT / "data/train.npy")[:8].astype(np.int32))
        mx.eval(batch, model.parameters())
        k = 4 if kind == "rec_full" else 2
        fn = nn.value_and_grad(model, lambda x, y: objective(model, x, y, 4, k))
        repeats = []
        for index in range(19):
            gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
            baseline = mx.get_active_memory()
            t0 = time.perf_counter()
            value, grads = fn(batch[:, :-1], batch[:, 1:])
            mx.eval(value)
            t1 = time.perf_counter()
            mx.eval(grads)
            t2 = time.perf_counter()
            peak = mx.get_peak_memory()
            logical = sum(array.nbytes for _, array in tree_flatten(grads))
            loss = float(value)
            del value
            gc.collect()
            retained = mx.get_active_memory() - baseline
            repeats.append({"forward_ms": (t1 - t0) * 1000, "backward_ms": (t2 - t1) * 1000,
                            "total_ms": (t2 - t0) * 1000, "peak_incremental_bytes": peak - baseline,
                            "gradient_logical_bytes": logical, "gradient_physical_retained_bytes": retained,
                            "loss": loss})
            del grads
        row = {"kind": kind, "batch": 8, "sequence": 256, "dtype": "bfloat16", "repeats": repeats,
               "median": {key: float(np.median([item[key] for item in repeats[4:]])) for key in repeats[0]}}
        output.append(row)
        print(kind, json.dumps(row["median"]), flush=True)
        del model, fn, batch
    (ROOT / "metrics/backward_benchmark.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
