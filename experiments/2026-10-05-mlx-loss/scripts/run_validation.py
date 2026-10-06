"""Use the original monitor and budgets for the final full-context check."""
import hashlib
import json
import time
from pathlib import Path

base = Path(__file__).resolve().parents[1]
for filename in ("FREEZE.json", "V2_FREEZE.json", "FULL_CONTEXT_FREEZE.json"):
    for path, expected in json.loads((base / filename).read_text())["sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
v2_events = [json.loads(x) for x in (base / "metrics/v2_events.jsonl").read_text().splitlines()]
assert v2_events[-1]["event"] == "campaign_complete", "Timings did not finish cleanly"
source = (base / "scripts/controller.py").read_text()
prefix = source[:source.index('\ntry:\n    run(NEW_PYTHON')]
prefix = prefix.replace("events.jsonl", "validation_events.jsonl").replace("scripts/worker.py", "scripts/check_full_context.py")
namespace = {"__file__": str(base / "scripts/controller.py")}
exec(compile(prefix, str(base / "scripts/controller.py"), "exec"), namespace)
namespace["worker_seconds"] = v2_events[-1]["aggregate_worker_seconds"]
original_epoch = (base / "metrics/events.jsonl").stat().st_mtime - 4.586318458023015
namespace["start"] = time.monotonic() - (time.time() - original_epoch)
try:
    namespace["run"](namespace["NEW_PYTHON"], [], "full_context_validation", 180)
    namespace["event"](event="validation_complete", aggregate_worker_seconds=namespace["worker_seconds"])
finally:
    namespace["events"].close()
