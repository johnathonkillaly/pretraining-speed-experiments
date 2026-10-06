"""Priority-corrected controller; completed tuning reused, no failed-fit retry."""
import datetime
import hashlib
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from model import KINDS, ROOT
from train import verify_freeze


def preflight():
    jobs = []
    for line in subprocess.check_output(["ps", "-axo", "pid,rss,command"], text=True).splitlines()[1:]:
        fields = line.strip().split(maxsplit=2)
        if len(fields) < 3 or str(ROOT) in fields[2]:
            continue
        if int(fields[1]) > 2 * 1024 ** 2 and any(word in fields[2].lower() for word in ("python", "mlx", "ollama", "llama-server")):
            jobs.append({"pid": fields[0], "rss_kib": int(fields[1]), "command": fields[2]})
    if jobs:
        raise RuntimeError(f"Competing local ML jobs: {jobs}")
    return subprocess.check_output(["memory_pressure", "-Q"], text=True)


def main():
    verify_freeze()
    priority_freeze = json.loads((ROOT / "PRIORITY_FREEZE.json").read_text())
    for filename, expected in priority_freeze["sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    event_file = (ROOT / "metrics/suite_events.jsonl").open("a")
    def event(**record):
        record["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event_file.write(json.dumps(record) + "\n"); event_file.flush()
        print(json.dumps(record), flush=True)
    def fit(kind, seed, steps, peak, tag, tune):
        summary_path = ROOT / "metrics" / f"{tag}_summary.json"
        if summary_path.exists():
            assert tune, "No primary results may be reused across launcher policies"
            row = json.loads(summary_path.read_text())
            assert (row["kind"], row["seed"], row["steps"], row["muon_peak"], row["tune"]) == (kind, seed, steps, peak, True)
            history = [json.loads(line) for line in (ROOT / "metrics" / f"{tag}.jsonl").read_text().splitlines()]
            trains = [item for item in history if item["event"] == "train"]
            assert len(trains) == steps and trains[-1]["step"] == steps
            assert history[-1]["event"] == "validation" and history[-1]["step"] == steps
            assert not any(item["event"] == "final_test" for item in history)
            event(event="reuse_completed_tuning", tag=tag, summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest())
            return row
        assert not (ROOT / "metrics" / f"{tag}.jsonl").exists(), f"Partial evidence for {tag}; no retry"
        assert not (ROOT / "logs" / f"{tag}.log").exists(), f"Prior attempt for {tag}; no retry"
        event(event="preflight", tag=tag, memory_pressure=preflight())
        command = [sys.executable, str(ROOT / "scripts/train.py"), "--kind", kind, "--seed", str(seed),
                   "--steps", str(steps), "--muon-peak", str(peak), "--tag", tag]
        if tune:
            command.append("--tune")
        event(event="run_start", tag=tag, command=command)
        with (ROOT / "logs" / f"{tag}.log").open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                event(event="timeout", tag=tag)
                raise RuntimeError(f"Fit timed out: {tag}")
        event(event="run_end", tag=tag, exit_code=code)
        if code != 0:
            raise RuntimeError(f"Fit failed: {tag}")
        return json.loads((ROOT / "metrics" / f"{tag}_summary.json").read_text())
    event(event="suite_priority_corrected_start", pid=os.getpid(), launcher="Interactive, nice0, no background IO clamp")
    selected = {}
    for offset, kind in enumerate(KINDS):
        trials = []
        rates = (.003, .01, .02)
        for peak in rates[offset % 3:] + rates[:offset % 3]:
            tag = f"tune_{kind}_mu{str(peak).replace('.', 'p')}"
            trials.append(fit(kind, 314, 600, peak, tag, True))
        best = min(trials, key=lambda row: (row["final_development_loss"], row["muon_peak"]))
        selected[kind] = {"muon_peak": best["muon_peak"], "adamw_peak": .0003,
                          "selection_metric": best["final_development_loss"],
                          "trials": [{"muon_peak": row["muon_peak"], "development_loss": row["final_development_loss"]} for row in trials]}
        (ROOT / "metrics/selected_rates.json").write_text(json.dumps(selected, indent=2) + "\n")
    summaries = []
    for offset, seed in enumerate((42, 43, 44)):
        for kind in KINDS[offset:] + KINDS[:offset]:
            tag = f"{kind}_s{seed}"
            summaries.append(fit(kind, seed, 5000, selected[kind]["muon_peak"], tag, False))
    (ROOT / "metrics/SUITE_COMPLETE.json").write_text(json.dumps({"summaries": summaries}, indent=2) + "\n")
    event(event="suite_complete", primary_fits=len(summaries))


if __name__ == "__main__":
    main()
