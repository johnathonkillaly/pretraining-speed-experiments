"""Run the new diagnostic with the frozen independent monitor, not the old suite."""
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parents[1]
for filename in ("FREEZE.json", "DIAGNOSTIC_FREEZE.json", "DIAGNOSTIC_RUNNER_FREEZE.json"):
    for path, expected in json.loads((base / filename).read_text())["sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
source = (base / "scripts/controller.py").read_text()
cut = source.index('\ntry:\n    run(NEW_PYTHON')
prefix = source[:cut].replace("events.jsonl", "diagnostic_events_v2.jsonl")
prefix = prefix.replace("scripts/worker.py", "scripts/diagnose_gradients.py")
namespace = {"__file__": str(base / "scripts/controller.py")}
exec(compile(prefix, str(base / "scripts/controller.py"), "exec"), namespace)
namespace["worker_seconds"] = 4.586318458023015
try:
    namespace["run"](namespace["NEW_PYTHON"], [], "gradient_diagnostic", 180)
finally:
    namespace["events"].close()
