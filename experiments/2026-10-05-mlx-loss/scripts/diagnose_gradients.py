"""Investigate the unchanged failed gate; never relax or relabel it."""
import gc
import json
import math
import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from worker import STUDY, Trainer, args_h1, model_objective, tensor_errors, verify_freeze, save

verify_freeze()
mx.set_memory_limit(16 * 1024**3)
mx.set_cache_limit(1024**3)
trainer = Trainer("dense20", 42, args_h1(), checkpoint_mode="alternating")
batch = mx.array(np.random.default_rng(999).integers(1, 32768, size=(1, 65), dtype=np.int32))
report = {"mlx": mx.__version__, "comparisons": [], "original_gate": "FAILED; unchanged"}
path = STUDY / "metrics/gradient_diagnostic.json"


def gradient(kind):
    def objective(x, y):
        return model_objective(trainer.model, kind, x, y)
    fn = nn.value_and_grad(trainer.model, objective)
    loss, grad = fn(batch[:, :-1], batch[:, 1:])
    mx.eval(loss, grad)
    return float(loss), dict(tree_flatten(grad))


def compare(label, a, b):
    errors = {name: tensor_errors(value, b[1][name]) for name, value in a[1].items()}
    relative = math.sqrt(sum(v["squared_difference"] for v in errors.values()) /
                         max(sum(v["reference_squared_norm"] for v in errors.values()), 1e-30))
    row = {"label": label, "loss_abs": abs(a[0] - b[0]), "relative_l2": relative,
           "finite": all(v["finite"] for v in errors.values()), "parameter_errors": errors}
    report["comparisons"].append(row)
    save(path, report)
    print(json.dumps({k: v for k, v in row.items() if k != "parameter_errors"}), flush=True)


for mode in ("alternating", "none"):
    trainer.model._checkpoint_mode = mode
    first = gradient("composed")
    second = gradient("composed")
    compare("bf16_" + mode + "_composed_repeat", first, second)
    del second
    second = gradient("fused")
    compare("bf16_" + mode + "_composed_vs_fused", first, second)
    del first
    third = gradient("fused")
    compare("bf16_" + mode + "_fused_repeat", second, third)
    del second, third
    gc.collect()
    mx.clear_cache()
trainer.model.update(tree_unflatten(list(trainer.optimizer.state["masters"].items())))
trainer.model._checkpoint_mode = "alternating"
first = gradient("composed")
second = gradient("composed")
compare("fp32_alternating_composed_repeat", first, second)
del second
second = gradient("fused")
compare("fp32_alternating_composed_vs_fused", first, second)
report["completed"] = True
save(path, report)
