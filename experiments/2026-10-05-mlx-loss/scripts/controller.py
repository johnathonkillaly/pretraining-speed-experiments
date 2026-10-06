"""Sequential bounded workers with independent macOS thermal/AC/memory monitoring."""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parents[3]
THERMAL = STUDY / "bin/thermal"
OLD_PYTHON = PROJECT / ".venv/bin/python"
NEW_PYTHON = PROJECT / ".venv-mlx0323/bin/python"
start = time.monotonic()
worker_seconds = 0.0
events = (STUDY / "metrics/events.jsonl").open("x")
def terminate(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")
signal.signal(signal.SIGTERM, terminate)


def event(**row):
    row["monotonic_seconds"] = time.monotonic() - start
    events.write(json.dumps(row) + "\n")
    events.flush()
    if row.get("event") != "monitor":
        print(json.dumps(row), flush=True)


def thermal_state():
    return int(subprocess.check_output([str(THERMAL)], text=True, timeout=3).strip())


def system_status():
    battery = subprocess.check_output(["pmset", "-g", "batt"], text=True, timeout=3)
    memory = subprocess.check_output(["memory_pressure", "-Q"], text=True, timeout=3)
    match = re.search(r"free percentage:\s*(\d+)%", memory)
    if not match:
        raise RuntimeError("Memory-pressure reading unavailable")
    competitors = []
    output = subprocess.check_output(["ps", "-axo", "pid,rss,command"], text=True, timeout=3)
    for line in output.splitlines()[1:]:
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3 or str(STUDY) in parts[2]:
            continue
        if int(parts[1]) > 2 * 1024 ** 2 and any(s in parts[2].lower() for s in
                ("python", "ollama", "llama-server", "mlx_lm")):
            competitors.append({"pid": int(parts[0]), "rss_kib": int(parts[1]), "command": parts[2]})
    return {"ac": "AC Power" in battery, "memory_free_pct": int(match.group(1)),
            "competing_ml": competitors}


def require_safe(status):
    if not status["ac"] or status["memory_free_pct"] < 20 or status["competing_ml"]:
        raise RuntimeError(f"Safety preflight failed: {status}")


def run(python, arguments, tag, timeout=180):
    global worker_seconds
    status = system_status()
    require_safe(status)
    cool_start = time.monotonic()
    while thermal_state() != 0:
        if time.monotonic() - cool_start > 120:
            raise RuntimeError("Thermal state did not return to nominal")
        time.sleep(2)
    if worker_seconds >= 900 or time.monotonic() - start >= 1800:
        raise RuntimeError("Campaign budget exhausted")
    command = ["nice", "-n", "10", str(python), str(STUDY / "scripts/worker.py"), *arguments, "--tag", tag]
    event(event="worker_start", tag=tag, command=command, preflight=status)
    worker_start = time.monotonic()
    log = (STUDY / "logs" / (tag + ".log")).open("x")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    next_system_check = 0.0
    latest = status
    try:
        while process.poll() is None:
            now = time.monotonic()
            state = thermal_state()
            if now >= next_system_check:
                latest = system_status()
                require_safe(latest)
                next_system_check = now + 10
            event(event="monitor", tag=tag, thermal_state=state, **latest)
            if state >= 2:
                raise RuntimeError(f"Thermal stop at state {state}")
            if now - worker_start > timeout or worker_seconds + now - worker_start > 900 or now - start > 1800:
                raise RuntimeError("Worker/campaign time budget exhausted")
            time.sleep(1.5)
        code = process.wait()
        worker_seconds += time.monotonic() - worker_start
        event(event="worker_end", tag=tag, exit_code=code, aggregate_worker_seconds=worker_seconds)
        if code:
            raise RuntimeError(f"Worker {tag} failed; inspect its preserved log")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        log.close()


try:
    run(NEW_PYTHON, ["verify"], "verify", 180)
    order = [(OLD_PYTHON, "composed", "A1"), (NEW_PYTHON, "composed", "B1"),
             (NEW_PYTHON, "fused", "C1"), (NEW_PYTHON, "fused", "C2"),
             (NEW_PYTHON, "composed", "B2"), (OLD_PYTHON, "composed", "A2")]
    hashes = None
    for python, kind, tag in order:
        run(python, ["h1", "--kind", kind], tag)
        record = json.loads((STUDY / "metrics" / (tag + ".json")).read_text())
        identity = record["initialization_sha256"], record["input_sha256"]
        if hashes is None:
            hashes = identity
        if identity != hashes:
            raise RuntimeError("Initialization or input hashes differ across arms")
    for tag in ("D1", "D2"):
        run(NEW_PYTHON, ["h1", "--kind", "fused", "--checkpoint-mode", "none"], tag)
        record = json.loads((STUDY / "metrics" / (tag + ".json")).read_text())
        if (record["initialization_sha256"], record["input_sha256"]) != hashes:
            raise RuntimeError("Exploratory arm initialization/input mismatch")
    event(event="campaign_complete", aggregate_worker_seconds=worker_seconds)
except BaseException as error:
    event(event="campaign_stopped", error=repr(error), aggregate_worker_seconds=worker_seconds)
    raise
finally:
    events.close()
