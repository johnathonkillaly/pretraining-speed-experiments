"""Sequential bounded tuning/learning controller. Never retries a failed fit."""
import datetime
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
    event_file = (ROOT / "metrics/suite_events.jsonl").open("x")
    def event(**record):
        record["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event_file.write(json.dumps(record) + "\n"); event_file.flush()
        print(json.dumps(record), flush=True)
    def fit(kind, seed, steps, peak, tag, tune):
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
    event(event="suite_start", pid=os.getpid())
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
