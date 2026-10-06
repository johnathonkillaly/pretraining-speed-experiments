"""One allocation-only SGD control, preregistered in the separate amendment."""
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

import run_optimizer_probe as probe


class StatelessSGD(optim.SGD):
    def init_single(self, parameter, state):
        if self.momentum > 0:
            super().init_single(parameter, state)


def main():
    root = Path(__file__).resolve().parent
    freeze = json.loads((root / "STATELESS_FREEZE.json").read_text())
    for filename, expected in freeze["sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
    mx.set_memory_limit(16 * 1024**3)
    mx.set_cache_limit(1024**3)
    result = {"preflight": probe.preflight()}
    mx.random.seed(33)
    a = {"w": mx.random.normal((31, 7))}
    b = {"w": a["w"]}
    standard, stateless = optim.SGD(.001), StatelessSGD(.001)
    differences = []
    for _ in range(3):
        gradient = {"w": mx.random.normal((31, 7))}
        a = standard.apply_gradients(gradient, a)
        b = stateless.apply_gradients(gradient, b)
        mx.eval(a, b, standard.state, stateless.state)
        difference = float(mx.max(mx.abs(a["w"] - b["w"])))
        differences.append(difference)
        np.testing.assert_array_equal(np.array(a["w"]), np.array(b["w"]))
    result["numerical_checks"] = {"passed": True, "max_differences": differences}
    del a, b, standard, stateless, gradient
    factory = probe.make_optimizer
    probe.make_optimizer = lambda name: StatelessSGD(.001) if name == "stateless_sgd" else factory(name)
    result["cases"] = [probe.benchmark_case("stateless_sgd", layout, activity)
                       for layout in ("dense", "independent", "stacked")
                       for activity in (("all",) if layout == "dense" else ("all", "one"))]
    (root / "stateless_sgd_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        print(json.dumps({key: case[key] for key in ("layout", "activity", "median_update_ms", "state_logical_bytes", "state_physical_retained_bytes")}))


if __name__ == "__main__":
    main()
