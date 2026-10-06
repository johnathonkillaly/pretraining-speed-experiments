"""After all training: reload every model and reproduce its development loss."""
import gc
import json

import mlx.core as mx
import numpy as np

from model import ROOT
from run_suite import preflight
from train import Trainer, validate, verify_freeze


def main():
    assert (ROOT / "metrics/SUITE_COMPLETE.json").exists()
    verify_freeze(); preflight()
    mx.set_memory_limit(16 * 1024**3); mx.set_cache_limit(1024**3)
    development = np.load(ROOT / "data/development.npy")
    rows = []
    for summary in json.loads((ROOT / "metrics/SUITE_COMPLETE.json").read_text())["summaries"]:
        kind, seed = summary["kind"], summary["seed"]
        tag = f"{kind}_s{seed}"
        trainer = Trainer(kind, seed)
        trainer.model.load_weights(str(ROOT / "checkpoints" / tag / "model.safetensors"))
        mx.eval(trainer.model.parameters())
        reproduced, _ = validate(trainer, development)
        expected = summary["final_development_loss"]
        difference = abs(reproduced - expected)
        assert np.isfinite(reproduced) and difference < 2e-6, (tag, reproduced, expected)
        rows.append({"tag": tag, "expected_development_loss": expected,
                     "reloaded_development_loss": reproduced, "absolute_difference": difference,
                     "passed": True})
        print(json.dumps(rows[-1]), flush=True)
        del trainer; gc.collect(); mx.clear_cache()
    (ROOT / "metrics/reload_checks.json").write_text(json.dumps({"passed": True, "test_data_reused": False,
                                                                "records": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
