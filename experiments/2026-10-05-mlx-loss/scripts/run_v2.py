"""Distinct v2 campaign; no change to rejected v1 sources or thresholds."""
import hashlib
import json
import time
from pathlib import Path

base = Path(__file__).resolve().parents[1]
for name in ("FREEZE.json", "DIAGNOSTIC_FREEZE.json", "DIAGNOSTIC_RUNNER_FREEZE.json", "V2_FREEZE.json"):
    for path, expected in json.loads((base / name).read_text())["sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
source = (base / "scripts/controller.py").read_text()
prefix = source[:source.index('\ntry:\n    run(NEW_PYTHON')].replace("events.jsonl", "v2_events.jsonl")
namespace = {"__file__": str(base / "scripts/controller.py")}
exec(compile(prefix, str(base / "scripts/controller.py"), "exec"), namespace)
namespace["worker_seconds"] = 7.6646714160451666
original_start_epoch = (base / "metrics/events.jsonl").stat().st_mtime - 4.586318458023015
namespace["start"] = time.monotonic() - (time.time() - original_start_epoch)
try:
    # The monitor definition is identical; only this gate's script path differs.
    gate_namespace = {"__file__": str(base / "scripts/controller.py")}
    gate_prefix = prefix.replace("scripts/worker.py", "scripts/check_checkpointing.py")
    gate_prefix = gate_prefix[:gate_prefix.index('events = ')] + gate_prefix[gate_prefix.index('def terminate'):]
    gate_namespace.update({"events": namespace["events"]})
    exec(compile(gate_prefix, str(base / "scripts/controller.py"), "exec"), gate_namespace)
    gate_namespace["start"] = namespace["start"]
    gate_namespace["worker_seconds"] = namespace["worker_seconds"]
    gate_namespace["run"](namespace["NEW_PYTHON"], [], "checkpointing_gate", 180)
    namespace["worker_seconds"] = gate_namespace["worker_seconds"]
    order = [(namespace["OLD_PYTHON"], "composed", "alternating", "A1"),
             (namespace["NEW_PYTHON"], "composed", "alternating", "B1"),
             (namespace["NEW_PYTHON"], "fused", "alternating", "U1"),
             (namespace["NEW_PYTHON"], "fused", "alternating", "U2"),
             (namespace["NEW_PYTHON"], "composed", "alternating", "B2"),
             (namespace["OLD_PYTHON"], "composed", "alternating", "A2"),
             (namespace["NEW_PYTHON"], "composed", "none", "E1"),
             (namespace["NEW_PYTHON"], "composed", "none", "E2")]
    hashes = None
    for python, kind, mode, tag in order:
        namespace["run"](python, ["h1", "--kind", kind, "--checkpoint-mode", mode], tag)
        row = json.loads((base / "metrics" / (tag + ".json")).read_text())
        identity = row["initialization_sha256"], row["input_sha256"]
        hashes = identity if hashes is None else hashes
        if identity != hashes:
            raise RuntimeError("Initialization/input mismatch across arms")
    namespace["event"](event="campaign_complete", aggregate_worker_seconds=namespace["worker_seconds"])
except BaseException as error:
    namespace["event"](event="campaign_stopped", error=repr(error), aggregate_worker_seconds=namespace["worker_seconds"])
    raise
finally:
    namespace["events"].close()
