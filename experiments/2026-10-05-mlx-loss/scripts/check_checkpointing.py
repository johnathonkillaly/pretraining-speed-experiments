"""New E gate at the unchanged preregistered tolerances."""
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from worker import STUDY, Trainer, args_h1, model_objective, tensor_errors, verify_freeze, save

verify_freeze()
mx.set_memory_limit(16 * 1024**3)
mx.set_cache_limit(1024**3)
trainer = Trainer("dense20", 42, args_h1(), checkpoint_mode="alternating")
batch = mx.array(np.random.default_rng(999).integers(1, 32768, size=(1, 65), dtype=np.int32))
results = []
for mode in ("alternating", "none"):
    trainer.model._checkpoint_mode = mode
    def objective(x, y):
        return model_objective(trainer.model, "composed", x, y)
    loss, gradient = nn.value_and_grad(trainer.model, objective)(batch[:, :-1], batch[:, 1:])
    mx.eval(loss, gradient)
    results.append((float(loss), dict(tree_flatten(gradient))))
errors = {name: tensor_errors(value, results[1][1][name]) for name, value in results[0][1].items()}
relative = math.sqrt(sum(v["squared_difference"] for v in errors.values()) /
                     max(sum(v["reference_squared_norm"] for v in errors.values()), 1e-30))
report = {"loss_abs": abs(results[0][0] - results[1][0]), "relative_l2": relative,
          "finite": all(v["finite"] for v in errors.values()), "parameter_errors": errors}
save(STUDY / "metrics/checkpointing_gate.json", report)
assert report["finite"] and report["loss_abs"] <= 2e-4 and relative <= .005
assert all(v["reference_squared_norm"] > 0 for v in errors.values())
report["passed"] = True
save(STUDY / "metrics/checkpointing_gate.json", report)
print({k: v for k, v in report.items() if k != "parameter_errors"}, flush=True)
