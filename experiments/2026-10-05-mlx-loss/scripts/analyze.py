"""Registered arithmetic; reads preserved records without launching GPU work."""
import json
from pathlib import Path
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
records = {}
for tag in ("A1", "B1", "C1", "C2", "B2", "A2", "D1", "D2"):
    path = STUDY / "metrics" / (tag + ".json")
    if path.exists():
        value = json.loads(path.read_text())
        if value.get("completed"):
            records[tag] = value

correctness = json.loads((STUDY / "metrics/correctness.json").read_text())
result = {"correctness_passed": correctness.get("passed", False), "completed_tags": list(records),
          "microbenchmarks": [], "arms": {}, "comparisons": {}}
for shape in sorted({tuple(r["shape"]) for r in correctness["timings"]}):
    timings = {r["kind"]: r for r in correctness["timings"] if tuple(r["shape"]) == shape}
    medians = {kind: float(np.median([r["seconds"] for r in row["records"] if not r["warmup"]]))
               for kind, row in timings.items()}
    result["microbenchmarks"].append({"shape": list(shape), "median_seconds": medians,
                                     "fused_speed_ratio": medians["composed"] / medians["fused"]})
for arm in ("A", "B", "C", "D"):
    rows = [value for tag, value in records.items() if tag.startswith(arm)]
    if rows:
        steps = [r for value in rows for r in value["records"] if not r["warmup"]]
        result["arms"][arm] = {"mlx": rows[0]["mlx"], "loss": rows[0]["loss"],
                              "checkpointing": rows[0]["checkpointing"],
                              "targets_per_second": float(32760 / np.mean([r["total_seconds"] for r in steps])),
                              "model_seconds": float(np.mean([r["model_backward_accumulation_seconds"] for r in steps])),
                              "optimizer_seconds": float(np.mean([r["optimizer_seconds"] for r in steps])),
                              "peak_active_bytes": max(row["peak_active_bytes"] for row in rows),
                              "runs": len(rows), "measured_updates": len(steps)}
for numerator, denominator in (("B", "A"), ("C", "B"), ("C", "A"), ("D", "C")):
    pairs = []
    for round_number in (1, 2):
        a, b = records.get(numerator + str(round_number)), records.get(denominator + str(round_number))
        if a and b:
            assert (a["initialization_sha256"], a["input_sha256"]) == (b["initialization_sha256"], b["input_sha256"])
            pairs.append({"round": round_number, "throughput_ratio": a["steady_targets_per_second"] / b["steady_targets_per_second"],
                          "memory_ratio": a["peak_active_bytes"] / b["peak_active_bytes"]})
    if pairs:
        comparison = {"pairs": pairs, "median_speed_gain_pct": float(100 * (np.median([r["throughput_ratio"] for r in pairs]) - 1))}
        if numerator == "C" and denominator == "A":
            comparison["registered_recommendation_gate"] = (len(pairs) == 2 and result["correctness_passed"] and
                all(p["throughput_ratio"] >= 1.05 and p["memory_ratio"] <= 1.10 for p in pairs))
        if numerator == "D":
            comparison["registered_recommendation_gate"] = (len(pairs) == 2 and result["correctness_passed"] and
                all(p["throughput_ratio"] >= 1.05 for p in pairs) and result["arms"]["D"]["peak_active_bytes"] <= 32 * 1024**3)
        result["comparisons"][numerator + "/" + denominator] = comparison
(STUDY / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
