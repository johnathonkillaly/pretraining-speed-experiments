"""One-shot bounded sequential runner, suitable for launchd; never retries fits."""
import datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def timestamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def substantial_other_jobs():
    lines=subprocess.check_output(['ps','-axo','pid,rss,command'],text=True).splitlines()
    found=[]
    for line in lines[1:]:
        parts=line.strip().split(None,2)
        if len(parts)<3:continue
        pid,rss,command=parts
        if '/moe_night/' in command or 'ps -axo' in command:continue
        if 'python' in command.lower() and int(rss)>2*1024**2 and any(s in command.lower() for s in ('train','sft','mlx','torch')):
            found.append({'pid':int(pid),'rss_kib':int(rss),'command':command})
    return found

def main():
    matrix=json.loads((ROOT/'configs/run_matrix.json').read_text())
    events=(ROOT/'metrics/suite_events.jsonl').open('x')
    def log(**values):
        row={'utc':timestamp(),**values};events.write(json.dumps(row)+'\n');events.flush();print(json.dumps(row),flush=True)
    log(event='start',pid=os.getpid(),matrix=matrix)
    for run in matrix:
        jobs=substantial_other_jobs()
        if jobs:
            log(event='stopped_for_other_training',jobs=jobs);return
        pressure=subprocess.check_output(['memory_pressure','-Q'],text=True)
        log(event='preflight',run=run,memory_pressure=pressure)
        tag=f"{run['kind']}_s{run['seed']}"
        if (ROOT/'metrics'/f'{tag}.jsonl').exists():
            log(event='refuse_existing_run',run=run);return
        command=[sys.executable,str(ROOT/'scripts/train.py'),'--kind',run['kind'],
                 '--seed',str(run['seed']),'--steps',str(run['steps'])]
        with (ROOT/'logs'/f'{tag}.log').open('x') as log_file:
            log(event='run_start',run=run,command=command)
            process=subprocess.Popen(command,stdout=log_file,stderr=subprocess.STDOUT)
            try:
                code=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                log(event='timeout',run=run);continue
        log(event='run_end',run=run,exit_code=code)
    log(event='complete')
    (ROOT/'metrics/SUITE_COMPLETE.json').write_text(json.dumps({'utc':timestamp(),'matrix':matrix},indent=2))

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'logs/supervisor_exception.log').write_text(traceback.format_exc());raise
