"""Descriptive v2 timing results; original fused matching gate stays failed."""
import json
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parents[1]
records = {}
for tag in ("A1", "B1", "U1", "U2", "B2", "A2", "E1", "E2"):
    p = base / "metrics" / (tag + ".json")
    if p.exists():
        row = json.loads(p.read_text())
        if row.get("completed"):
            records[tag] = row
result = {"original_fused_matching_gate": "FAILED; unchanged", "arms": {}, "comparisons": {},
          "completed_tags": list(records), "microbenchmarks": []}
correctness = json.loads((base / "metrics/correctness.json").read_text())
for shape in sorted({tuple(r["shape"]) for r in correctness["timings"]}):
    medians = {r["kind"]: float(np.median([x["seconds"] for x in r["records"] if not x["warmup"]]))
               for r in correctness["timings"] if tuple(r["shape"]) == shape}
    result["microbenchmarks"].append({"shape": list(shape), "median_seconds": medians,
                                      "speed_ratio": medians["composed"] / medians["fused"]})
for arm in ("A", "B", "U", "E"):
    runs = [r for tag, r in records.items() if tag.startswith(arm)]
    if runs:
        steps = [s for r in runs for s in r["records"] if not s["warmup"]]
        result["arms"][arm] = {"mlx": runs[0]["mlx"], "loss": runs[0]["loss"],
                               "checkpointing": runs[0]["checkpointing"], "runs": len(runs),
                               "targets_per_second": float(32760 / np.mean([s["total_seconds"] for s in steps])),
                               "model_seconds": float(np.mean([s["model_backward_accumulation_seconds"] for s in steps])),
                               "optimizer_seconds": float(np.mean([s["optimizer_seconds"] for s in steps])),
                               "peak_active_bytes": max(r["peak_active_bytes"] for r in runs),
                               "finite_state": all(r["finite_final_state"] for r in runs)}
for top, bottom in (("B", "A"), ("U", "B"), ("U", "A"), ("E", "B"), ("E", "A")):
    pairs = []
    for i in (1, 2):
        a, b = records.get(top + str(i)), records.get(bottom + str(i))
        if a and b:
            assert a["initialization_sha256"] == b["initialization_sha256"]
            assert a["input_sha256"] == b["input_sha256"]
            pairs.append({"round": i, "speed_ratio": a["steady_targets_per_second"] / b["steady_targets_per_second"],
                          "memory_ratio": a["peak_active_bytes"] / b["peak_active_bytes"]})
    if pairs:
        result["comparisons"][top + "/" + bottom] = {"pairs": pairs,
            "median_speed_gain_pct": float(100 * (np.median([p["speed_ratio"] for p in pairs]) - 1))}
        if top == "U":
            result["comparisons"][top + "/" + bottom]["eligible_for_recommendation"] = False
if "E/B" in result["comparisons"]:
    gate = json.loads((base / "metrics/checkpointing_gate.json").read_text())
    pairs = result["comparisons"]["E/B"]["pairs"]
    result["comparisons"]["E/B"]["registered_recommendation_gate"] = (
        len(pairs) == 2 and gate.get("passed", False) and all(p["speed_ratio"] >= 1.05 for p in pairs)
        and result["arms"]["E"]["peak_active_bytes"] <= 32 * 1024**3
        and result["arms"]["E"]["finite_state"])
events = []
for filename in ("events.jsonl", "diagnostic_events_v2.jsonl", "v2_events.jsonl"):
    p = base / "metrics" / filename
    if p.exists():
        events.extend(json.loads(s) for s in p.read_text().splitlines())
monitors = [r for r in events if r["event"] == "monitor"]
result["safety"] = {"max_thermal_state": max([r["thermal_state"] for r in monitors], default=None),
                     "minimum_memory_free_pct": min([r["memory_free_pct"] for r in monitors], default=None),
                     "all_monitored_on_ac": all(r["ac"] for r in monitors),
                     "no_competing_ml_job": all(not r["competing_ml"] for r in monitors),
                     "worker_seconds": max([r.get("aggregate_worker_seconds", 0) for r in events], default=0)}
(base / "SUMMARY_V2.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
