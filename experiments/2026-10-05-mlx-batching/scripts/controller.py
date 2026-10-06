"""Sequential monitored workers; no arbitrary process arguments are persisted."""
import datetime
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parents[3]
PYTHON = PROJECT / ".venv-mlx0323/bin/python"
THERMAL = STUDY.parent / "2026-10-05-mlx-loss/bin/thermal"
START = time.monotonic()
worker_seconds = 0.0
events = (STUDY/"metrics/events.jsonl").open("x")


def event(**row):
    row.update(elapsed_seconds=time.monotonic()-START,
               utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    events.write(json.dumps(row)+"\n")
    events.flush()
    if row["event"] != "monitor":
        print(json.dumps(row), flush=True)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Signal {signum}")


signal.signal(signal.SIGTERM, interrupted)


def thermal():
    return int(subprocess.check_output([str(THERMAL)],text=True,timeout=3).strip())


def system_status():
    ac = "AC Power" in subprocess.check_output(["pmset","-g","batt"],text=True,timeout=3)
    memory = subprocess.check_output(["memory_pressure","-Q"],text=True,timeout=3)
    match = re.search(r"free percentage:\s*(\d+)%",memory)
    if not match:
        raise RuntimeError("Memory pressure unavailable")
    competitors = []
    output = subprocess.check_output(["ps","-axo","pid,rss,comm"],text=True,timeout=3)
    for line in output.splitlines()[1:]:
        parts = line.strip().split(None,2)
        if len(parts)!=3: continue
        pid,rss,exe = parts
        is_python = "python" in exe.lower()
        is_inference = any(x in exe.lower() for x in ("ollama","llama-server","mlx_lm"))
        if not (is_python or is_inference): continue
        lookup = subprocess.run(["ps","-p",pid,"-o","command="],capture_output=True,text=True,timeout=3)
        if lookup.returncode: continue
        args = lookup.stdout.strip()
        if str(STUDY) in args: continue
        likely_training = is_python and any(x in args.lower() for x in
            ("train.py","pretrain","torchrun","mlx_lm","assistant_loop.py","sft.fit"))
        if is_inference or likely_training or int(rss)>2*1024**2:
            tokens = shlex.split(args)
            script = next((Path(x).name for x in tokens[1:] if x.endswith('.py')),None)
            competitors.append({"pid":int(pid),"rss_kib":int(rss),"executable":exe,"script_basename":script})
    return {"ac":ac,"memory_free_pct":int(match.group(1)),"competing_ml":competitors}


def require_safe(status):
    if not status["ac"] or status["memory_free_pct"]<20 or status["competing_ml"]:
        raise RuntimeError(f"Safety stop: {status}")


def run(arguments,tag,timeout):
    global worker_seconds
    latest = system_status()
    require_safe(latest)
    cool_start = time.monotonic()
    announced = False
    while thermal()!=0:
        if not announced:
            event(event="cooldown",tag=tag)
            announced=True
        if time.monotonic()-cool_start>120:
            raise RuntimeError("Nominal-state cooldown timeout")
        require_safe(system_status())
        time.sleep(1.5)
    if worker_seconds>=1800 or time.monotonic()-START>=5400:
        raise RuntimeError("Campaign budget exhausted")
    command=["nice","-n","10",str(PYTHON),str(STUDY/"scripts/worker.py"),*arguments,"--tag",tag]
    event(event="worker_start",tag=tag,command=command,preflight=latest)
    before=time.monotonic()
    log=(STUDY/"logs"/(tag+".log")).open("x")
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    next_check=0.0
    stopped=None
    try:
        while process.poll() is None:
            now=time.monotonic()
            state=thermal()
            if now>=next_check:
                latest=system_status()
                require_safe(latest)
                next_check=now+10
            rss_text=subprocess.run(["ps","-p",str(process.pid),"-o","rss="],capture_output=True,text=True,timeout=3).stdout.strip()
            rss=int(rss_text)*1024 if rss_text else 0
            live={}
            live_path=STUDY/"metrics/live.json"
            if live_path.exists(): live=json.loads(live_path.read_text())
            event(event="monitor",tag=tag,thermal_state=state,worker_rss_bytes=rss,live=live,**latest)
            if state>=2: raise RuntimeError(f"Thermal stop: {state}")
            if rss>85*1024**3: raise RuntimeError(f"RSS stop: {rss}")
            if now-before>timeout or worker_seconds+now-before>1800 or now-START>5400:
                raise RuntimeError("Worker/campaign time budget exhausted")
            time.sleep(1.5)
        code=process.wait()
        if code: raise RuntimeError(f"Worker {tag} exited {code}; preserved log {tag}.log")
    except BaseException as error:
        stopped=repr(error)
        raise
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        worker_seconds+=time.monotonic()-before
        event(event="worker_end",tag=tag,exit_code=process.returncode,error=stopped,
              aggregate_worker_seconds=worker_seconds)
        log.close()


try:
    run(["validation","--context","64"],"gate64",300)
    run(["validation","--context","4095"],"gate4095",600)
    full=json.loads((STUDY/"metrics/validation_4095.json").read_text())
    active=[n for n in (1,2,4,8) if full["arms"][str(n)]["status"]=="checked"]
    small=json.loads((STUDY/"metrics/validation_64.json").read_text())
    identities={(full["initialization_sha256"],full["full_input_sha256"]),
                (small["initialization_sha256"],small["full_input_sha256"])}
    if len(identities)!=1: raise RuntimeError("Identity mismatch across validation workers")
    for round_number,order in ((1,active),(2,list(reversed(active)))):
        for size in order:
            tag=f"B{size}_R{round_number}"
            run(["timing","--batch",str(size)],tag,300)
            record=json.loads((STUDY/"metrics"/(tag+".json")).read_text())
            identities.add((record["initialization_sha256"],record["full_input_sha256"]))
            if len(identities)!=1: raise RuntimeError("Identity mismatch across timing arms")
    event(event="campaign_complete",aggregate_worker_seconds=worker_seconds,active_batches=active)
except BaseException as error:
    event(event="campaign_stopped",error=repr(error),aggregate_worker_seconds=worker_seconds)
    raise
finally:
    events.close()
