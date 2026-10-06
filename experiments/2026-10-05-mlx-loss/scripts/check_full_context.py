"""Compiled gradient check at actual H1 4K context; no optimizer update."""
import gc
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from worker import STUDY, Trainer, args_h1, model_objective, tensor_errors, verify_freeze, save

verify_freeze()
mx.set_memory_limit(32 * 1024**3)
mx.set_cache_limit(1024**3)
trainer = Trainer("dense20", 42, args_h1(), checkpoint_mode="alternating")
batch = mx.array(np.random.default_rng(500).integers(1, 32768, size=(1, 4096)).astype(np.int32))
report = {"context_inputs": 4095, "compiled": True, "parameters": 298361856,
          "original_fused_gate": "FAILED; unchanged", "gradients": []}
path = STUDY / "metrics/full_context_validation.json"


def gradient(kind, mode):
    trainer.model._checkpoint_mode = mode
    def objective(x, y):
        return model_objective(trainer.model, kind, x, y)
    fn = nn.value_and_grad(trainer.model, objective)
    fn = mx.compile(fn, inputs=trainer.model.state, outputs=trainer.model.state)
    mx.reset_peak_memory()
    loss, grad = fn(batch[:, :-1], batch[:, 1:])
    mx.eval(loss, grad)
    report["gradients"].append({"kind": kind, "checkpointing": mode,
                                "loss": float(loss), "peak_active_bytes": mx.get_peak_memory()})
    save(path, report)
    return float(loss), dict(tree_flatten(grad))


def compare(a, b):
    errors = {name: tensor_errors(value, b[1][name]) for name, value in a[1].items()}
    relative = math.sqrt(sum(v["squared_difference"] for v in errors.values()) /
                         max(sum(v["reference_squared_norm"] for v in errors.values()), 1e-30))
    return {"loss_abs": abs(a[0] - b[0]), "relative_l2": relative,
            "finite": all(v["finite"] for v in errors.values()),
            "nonzero": all(v["reference_squared_norm"] > 0 for v in errors.values()),
            "parameter_errors": errors}


baseline = gradient("composed", "alternating")
candidate = gradient("composed", "none")
report["checkpointing"] = compare(baseline, candidate)
save(path, report)
gate = report["checkpointing"]
assert gate["finite"] and gate["nonzero"] and gate["loss_abs"] <= 2e-4 and gate["relative_l2"] <= .005
report["checkpointing_passed"] = True
del candidate
gc.collect()
mx.clear_cache()
diagnostic = gradient("fused", "alternating")
report["fused_diagnostic"] = compare(baseline, diagnostic)
report["completed"] = True
save(path, report)
print({"checkpointing": {k: v for k, v in report["checkpointing"].items() if k != "parameter_errors"},
       "fused_diagnostic": {k: v for k, v in report["fused_diagnostic"].items() if k != "parameter_errors"}}, flush=True)
