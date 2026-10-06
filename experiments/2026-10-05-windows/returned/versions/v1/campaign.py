"""Sequential, frozen, temperature-monitored native CUDA measurements."""
import argparse, contextlib, ctypes, gc, hashlib, json, os, re, statistics, subprocess, threading, time, traceback
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from model import LanguageModel,Geometry,objective,convert_to_bf16
from optimizer import Hybrid
from verify import numerical_cpu,numerical_cuda

VERSION=Path(__file__).resolve().parent

class StopCampaign(RuntimeError): pass

def verify_freeze():
    freeze=json.loads((VERSION/'FREEZE.json').read_text())
    for name,digest in freeze['sha256'].items():
        assert hashlib.sha256((VERSION/name).read_bytes()).hexdigest()==digest,'Frozen file differs: '+name
    return freeze

def ac_status():
    class PowerStatus(ctypes.Structure):
        _fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
    status=PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)): raise StopCampaign('Power telemetry failed')
    return int(status.ACLineStatus)

def query():
    fields='temperature.gpu,power.draw,memory.used,utilization.gpu,clocks.sm'
    p=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
    if p.returncode: raise StopCampaign('NVIDIA telemetry failed: '+p.stderr)
    values=p.stdout.strip().split(', ')
    if len(values)!=5: raise StopCampaign('Unexpected GPU telemetry shape')
    row=dict(zip(['temperature_c','power_w','device_memory_mib','utilization_pct','sm_clock_mhz'],map(float,values)))
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
    if p.returncode: raise StopCampaign('GPU process telemetry failed')
    row['processes']=[]
    for line in p.stdout.strip().splitlines():
        pid,_,name=line.partition(', ')
        if pid.isdigit(): row['processes'].append({'pid':int(pid),'name':name})
    row['ac_line_status']=ac_status()
    return row

class Monitor:
    def __init__(self,out,config):
        self.out=out; self.config=config; self.lock=threading.Lock(); self.abort=None; self.latest=None; self.phase='idle'; self.phase_start=None; self.active_total=0.; self.closed=threading.Event(); self.rows=[]
        initial=query()
        self.allowed_desktop={p['pid'] for p in initial['processes'] if re.fullmatch(r'C:\\Program Files\\WindowsApps\\OpenAI\.Codex_[^\\]+\\app\\ChatGPT\.exe',p['name'],re.IGNORECASE)}
        self.inspect(initial)
        if initial['temperature_c']>config['start_temperature_c']: raise StopCampaign('Starting GPU temperature exceeds 65C')
        if initial['utilization_pct']>5: raise StopCampaign('GPU was not idle at preflight')
        self.record(initial)
        self.thread=threading.Thread(target=self.loop,daemon=True); self.thread.start()
    def record(self,row):
        with self.lock:
            row={**row,'monotonic_s':time.monotonic(),'phase':self.phase,'active_total_s':self.used()}
            self.latest=row; self.rows.append(row)
            with (self.out/'telemetry.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    def used(self): return self.active_total+(time.monotonic()-self.phase_start if self.phase_start is not None else 0)
    def inspect(self,row):
        if row['ac_line_status']!=1: raise StopCampaign('AC power disconnected or unknown')
        if row['temperature_c']>=self.config['stop_temperature_c']: raise StopCampaign('85C hard thermal stop reached')
        for p in row['processes']:
            if p['pid']!=os.getpid() and p['pid'] not in self.allowed_desktop: raise StopCampaign('New competing compute PID: '+str(p['pid']))
        if self.used()>=self.config['gpu_active_cap_s']: raise StopCampaign('600-second GPU-active campaign cap reached')
    def loop(self):
        while not self.closed.is_set():
            tick=time.monotonic()
            try:
                row=query()
                if time.monotonic()-tick>1: raise StopCampaign('Telemetry polling exceeded one second')
                self.inspect(row); self.record(row)
            except Exception as e: self.abort=str(e); self.closed.set(); break
            self.closed.wait(max(0,.8-(time.monotonic()-tick)))
    def check(self):
        if self.abort: raise StopCampaign(self.abort)
        if self.latest is None or time.monotonic()-self.latest['monotonic_s']>1.5: raise StopCampaign('Telemetry became stale')
        if self.used()>=self.config['gpu_active_cap_s']: raise StopCampaign('GPU-active campaign cap reached')
    @contextlib.contextmanager
    def active(self,name):
        self.check(); self.phase=name; self.phase_start=time.monotonic()
        try: yield
        finally:
            elapsed=time.monotonic()-self.phase_start; self.active_total+=elapsed; self.phase_start=None; self.phase='cooldown'
            with (self.out/'gpu-active-ledger.jsonl').open('a') as f: f.write(json.dumps({'phase':name,'elapsed_s':elapsed,'cumulative_gpu_active_s':self.active_total})+'\n')
    def cooldown(self):
        self.phase='cooldown'; start=time.monotonic()
        while True:
            self.check()
            if self.latest['temperature_c']<=self.config['start_temperature_c']: return time.monotonic()-start
            if time.monotonic()-start>180: raise StopCampaign('Cooldown did not reach 65C within 180 seconds')
            self.closed.wait(.5)
    def close(self): self.closed.set(); self.thread.join(timeout=4)

def step(model,opt,x,y,monitor):
    monitor.check(); start=time.perf_counter()
    model.zero_grad(set_to_none=True); zero_done=time.perf_counter()
    loss=objective(model,x,y); torch.cuda.synchronize(); forward_done=time.perf_counter()
    loss_value=float(loss)
    if not __import__('math').isfinite(loss_value): raise StopCampaign('Nonfinite loss')
    backward_start=time.perf_counter(); loss.backward(); torch.cuda.synchronize(); backward_done=time.perf_counter()
    norm=opt.apply(); torch.cuda.synchronize(); optimizer_done=time.perf_counter()
    norm_value=float(norm)
    if not __import__('math').isfinite(norm_value): raise StopCampaign('Nonfinite global gradient norm')
    monitor.check(); done=time.perf_counter()
    return {'loss':loss_value,'gradient_norm':norm_value,'clipped':norm_value>1,'zero_grad_ms':(zero_done-start)*1000,'forward_ms':(forward_done-zero_done)*1000,'backward_ms':(backward_done-backward_start)*1000,'optimizer_ms':(optimizer_done-backward_done)*1000,'whole_step_ms':(done-start)*1000,'monitor_finite_overhead_ms':(backward_start-forward_done+done-optimizer_done)*1000}

def benchmark(kind,seed,out,config,monitor,shared_hashes):
    cooldown_s=monitor.cooldown(); name=f'{kind}_s{seed}'
    with monitor.active(name+'-initialize'):
        model=LanguageModel(kind,seed)
        digest=model.shared_initialization_sha256
        if seed in shared_hashes: assert shared_hashes[seed]==digest,'Pair initialization mismatch'
        shared_hashes[seed]=digest
        expected=16667904 if kind=='dense10' else 11700480
        count=sum(p.numel() for p in model.parameters()); assert count==expected,(kind,count)
        opt=Hybrid(model,convert_to_bf16(model,'cuda'))
        gen=torch.Generator().manual_seed(1000+seed)
        tokens=torch.randint(1,32768,(8,257),generator=gen)
        batch_hash=hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
        tokens=tokens.cuda(); x,y=tokens[:,:-1],tokens[:,1:]
        torch.cuda.synchronize()
        initial={'kind':kind,'seed':seed,'parameters':count,'shared_initialization_sha256':digest,'batch_sha256':batch_hash,'targets_per_update':2048,'ownership':opt.ownership(),'cooldown_seconds':cooldown_s,'start_telemetry':monitor.latest.copy()}
    with monitor.active(name+'-warmup'):
        warm=[step(model,opt,x,y,monitor) for _ in range(config['warmup_steps'])]
        assert opt.finite(),'Nonfinite warmup state'
    torch.cuda.reset_peak_memory_stats()
    measurements=[]; phase_rows_start=len(monitor.rows)
    with monitor.active(name+'-timing'):
        start=time.perf_counter()
        while time.perf_counter()-start<config['measurement_window_s']:
            row=step(model,opt,x,y,monitor); row.update(kind=kind,seed=seed,step=len(measurements)+1,targets=2048)
            measurements.append(row)
            with (out/'steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        elapsed=time.perf_counter()-start
        assert opt.finite(),'Nonfinite final weights/master/momentum/moments'
        assert all(p.grad is not None and p.grad.dtype==torch.bfloat16 and torch.isfinite(p.grad).all().item() for p in model.parameters()),'Invalid BF16 gradients'
        peak_alloc=torch.cuda.max_memory_allocated(); peak_reserved=torch.cuda.max_memory_reserved()
    phase_rows=monitor.rows[phase_rows_start:]
    record={**initial,'warmup':warm,'steps':len(measurements),'actual_window_seconds':elapsed,'targets':2048*len(measurements),'targets_per_second':2048*len(measurements)/elapsed,'median_ms':{key:statistics.median(row[key] for row in measurements) for key in ('zero_grad_ms','forward_ms','backward_ms','optimizer_ms','whole_step_ms','monitor_finite_overhead_ms')},'first_step_ms':measurements[0]['whole_step_ms'],'first_10_median_ms':statistics.median(r['whole_step_ms'] for r in measurements[:10]),'last_10_median_ms':statistics.median(r['whole_step_ms'] for r in measurements[-10:]),'loss_first':measurements[0]['loss'],'loss_last':measurements[-1]['loss'],'clipped_fraction':sum(r['clipped'] for r in measurements)/len(measurements),'peak_allocated_bytes':peak_alloc,'peak_reserved_bytes':peak_reserved,'max_device_memory_mib':max((r['device_memory_mib'] for r in phase_rows),default=0),'max_sampled_temperature_c':max((r['temperature_c'] for r in phase_rows),default=0),'sampled_power_min_max_w':[min(r['power_w'] for r in phase_rows),max(r['power_w'] for r in phase_rows)],'finite_state_and_gradients':True,'end_telemetry':monitor.latest.copy()}
    with (out/'benchmarks.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    print(json.dumps({'phase':'benchmark_completed','kind':kind,'seed':seed,'targets_per_second':record['targets_per_second'],'median_ms':record['median_ms'],'max_temp_c':record['max_sampled_temperature_c'],'gpu_active_total_s':monitor.used()}),flush=True)
    del model,opt,x,y,tokens,warm,measurements
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    return record

def main(out,checkpoint_dir):
    assert not out.exists(),'Output directory exists; preserve prior evidence and use a new output path'
    out.mkdir(parents=True)
    freeze=verify_freeze(); config=json.loads((VERSION/'config.json').read_text())
    result={'status':'RUNNING','version':VERSION.name,'freeze':freeze,'config':config,'started_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'python':__import__('sys').version,'pid':os.getpid(),'attention':'PyTorch Flash SDPA forced on CUDA, explicit KV repetition; eager','benchmarks':[]}
    monitor=None
    try:
        assert torch.__version__=='2.7.0+cu128','Environment changed from registered baseline'
        cpu=numerical_cpu(checkpoint_dir); (out/'correctness-cpu.json').write_text(json.dumps(cpu,indent=2))
        print(json.dumps({'phase':'cpu_correctness','passed':True}),flush=True)
        monitor=Monitor(out,config)
        with monitor.active('cuda-initialize'):
            assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            free,total=torch.cuda.mem_get_info(); budget=min(free-1536*1024**2,int(total*.75))
            assert budget>2*1024**3,'Insufficient measured VRAM headroom'
            torch.cuda.set_per_process_memory_fraction(budget/total)
            result.update(initial_free_bytes=free,total_vram_bytes=total,allocator_budget_bytes=budget,device=torch.cuda.get_device_name(),allowed_desktop_pids=sorted(monitor.allowed_desktop))
            timings=[]
            for _ in range(100):
                t=time.perf_counter(); torch.cuda.synchronize(); timings.append((time.perf_counter()-t)*1e6)
            result['empty_cuda_synchronize_median_host_us']=statistics.median(timings)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with monitor.active('cuda-correctness'):
                cuda=numerical_cuda(); torch.cuda.synchronize()
                (out/'correctness-cuda.json').write_text(json.dumps(cuda,indent=2))
                print(json.dumps({'phase':'cuda_correctness','passed':True}),flush=True)
            gc.collect(); torch.cuda.empty_cache()
            shared={}
            for seed,order in zip(config['seeds'],config['orders']):
                for kind in order: result['benchmarks'].append(benchmark(kind,seed,out,config,monitor,shared))
            result['optional_sustained_probe']='Not registered for this first campaign; short paired windows only'
        result['status']='PASS'
    except Exception as e:
        result.update(status='STOPPED' if isinstance(e,StopCampaign) else 'FAIL',error=str(e),traceback=traceback.format_exc())
        print(json.dumps({'phase':'campaign_stopped','status':result['status'],'error':str(e)}),flush=True)
        raise
    finally:
        if monitor:
            result['gpu_active_seconds']=monitor.used(); monitor.close()
            result['max_sampled_gpu_temperature_c']=max(r['temperature_c'] for r in monitor.rows)
            result['telemetry_samples']=len(monitor.rows)
            result['max_telemetry_sample_gap_s']=max((b['monotonic_s']-a['monotonic_s'] for a,b in zip(monitor.rows,monitor.rows[1:])),default=0)
        result['finished_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        (out/'results.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'phase':'complete','status':result['status'],'gpu_active_seconds':result.get('gpu_active_seconds')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--checkpoint-dir',type=Path,required=True)
    args=p.parse_args(); main(args.output,args.checkpoint_dir)
