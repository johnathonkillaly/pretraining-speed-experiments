"""Reproduce into a new directory; never overwrite original research evidence."""
import argparse,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from zoneinfo import ZoneInfo

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    original=Path(__file__).resolve().parents[1]
    stamp=datetime.datetime.now(ZoneInfo('America/Denver')).strftime('%Y-%m-%d_%H%M%S')
    fresh=original.parent/f'reproduction-{stamp}'
    fresh.mkdir(exist_ok=False)
    shutil.copytree(original/'scripts',fresh/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(original/'configs',fresh/'configs')
    for name in ('CONTRACT.md','AMENDMENT_01.md','AMENDMENT_02.md','FREEZE.json','RESEARCH_NOTES.md'):
        shutil.copyfile(original/name,fresh/name)
    for name in ('logs','metrics','plots'): (fresh/name).mkdir()
    (fresh/'data').symlink_to((original/'data').resolve(),target_is_directory=True)
    checkpoint=(original/'checkpoints').resolve().parent/f'reproduction-checkpoints-{stamp}'
    checkpoint.mkdir(exist_ok=False);(fresh/'checkpoints').symlink_to(checkpoint,target_is_directory=True)
    for relative,digest in json.loads((fresh/'FREEZE.json').read_text())['files'].items():
        assert hashlib.sha256((fresh/relative).read_bytes()).hexdigest()==digest,relative
    print('Fresh output:',fresh,flush=True)
    if args.prepare_only:
        print('Prepared and all frozen hashes verified; no experiments launched.',flush=True)
        return
    commands=[['verify.py'],['benchmark.py','--suite','primary','--tag','_v2'],
              ['benchmark.py','--suite','sweep'],['run_suite.py'],['custom_vjp_probe.py'],['analyze.py']]
    for command in commands:
        with (fresh/'logs'/f'reproduce_{command[0]}.log').open('x') as log:
            subprocess.run([sys.executable,str(fresh/'scripts'/command[0]),*command[1:]],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=18000)
    print('Completed report:',fresh/'REPORT.md',flush=True)

if __name__=='__main__':main()
