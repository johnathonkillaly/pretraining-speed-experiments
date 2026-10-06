"""Registered post-window verification and short optimizer profile; no retiming."""
import argparse, gc, hashlib, json, math, statistics, sys, time, traceback
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

VERSION=Path(__file__).resolve().parent
BASE=VERSION.parent/'v2'
sys.path.insert(0,str(BASE))
from campaign import Monitor
from model import LanguageModel,Geometry,objective,convert_to_bf16
from optimizer import Hybrid
from verify import SMALL,grads

def verify_integrity():
    freeze=json.loads((VERSION/'FREEZE.json').read_text())
    for name,digest in freeze['sha256'].items(): assert hashlib.sha256((VERSION/name).read_bytes()).hexdigest()==digest,name
    original=json.loads((BASE/'FREEZE.json').read_text())
    assert hashlib.sha256((BASE/'FREEZE.json').read_bytes()).hexdigest()==freeze['dependency_freeze_sha256']
    for name,digest in original['sha256'].items(): assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest,name
    return freeze

def full_geometry_checks(monitor):
    model=LanguageModel('rec_full',42); masters=convert_to_bf16(model,'cuda')
    dense=LanguageModel('dense4',42); convert_to_bf16(dense,'cuda')
    tokens=torch.randint(1,32768,(8,257),generator=torch.Generator().manual_seed(1042)).cuda(); x,y=tokens[:,:-1],tokens[:,1:]
    with torch.no_grad():
        full=model(x,4,4); tb2=model(x,4,2)
        assert torch.equal(full,tb2),'Full-geometry logits differ under truncation'
        one=model(x,1,1); d4=dense(x)
        assert torch.equal(one,d4),'Full-geometry one-loop and Dense4 logits differ'
    del full,tb2,one,d4,dense
    # The same full geometry is used here, not just the reduced unit fixture.
    loss1,g1=grads(model,x,y,4); loss2,g2=grads(model,x,y,8)
    relative=max(float((g1[n].float()-g2[n].float()).norm()/(g1[n].float().norm()+1e-12)) for n in g1)
    assert relative<.02,relative
    assert torch.equal(loss1,loss2)
    report={'all_logits_full_vs_tb2_exact':True,'all_logits_one_loop_vs_dense4_exact':True,'shape':[8,256,32768],'k_ge_r_max_parameter_gradient_relative_l2':relative,'k_ge_r_all_gradient_tensors_exact':all(torch.equal(g1[n],g2[n]) for n in g1),'bf16_gradients':all(p.grad.dtype==torch.bfloat16 for p in model.parameters())}
    del model,masters,tokens,x,y,g1,g2,loss1,loss2
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); monitor.check()
    return report

def gpu_checkpoint(checkpoint_dir,monitor):
    model=LanguageModel('rec_tb2',91,SMALL); opt=Hybrid(model,convert_to_bf16(model,'cuda'))
    tokens=torch.randint(1,128,(2,17),generator=torch.Generator().manual_seed(5)).cuda(); x,y=tokens[:,:-1],tokens[:,1:]
    loss=objective(model,x,y); loss.backward(); norm=opt.apply(); assert torch.isfinite(loss) and torch.isfinite(norm) and opt.finite()
    checkpoint_dir.mkdir(parents=True,exist_ok=True); path=checkpoint_dir/'small-cuda-port-checkpoint.pt'
    torch.save({'model':model.state_dict(),'optimizer':opt.state},path)
    saved=torch.load(path,map_location='cuda',weights_only=True)
    clone=LanguageModel('rec_tb2',91,SMALL); oc=Hybrid(clone,convert_to_bf16(clone,'cuda'))
    clone.load_state_dict(saved['model']); oc.state=saved['optimizer']
    with torch.no_grad():
        before=model(x); after=clone(x)
    assert torch.equal(before,after)
    assert all(torch.equal(opt.state[group][n],oc.state[group][n]) for group in ('masters','muon_m','adam_m','adam_v') for n in opt.state[group])
    for m,o in ((model,opt),(clone,oc)):
        m.zero_grad(set_to_none=True); loss=objective(m,x,y); loss.backward(); o.apply(); assert o.finite()
    relative=math.sqrt(sum(float((opt.state['masters'][n]-oc.state['masters'][n]).square().sum()) for n in opt.names)/sum(float(t.square().sum()) for t in opt.state['masters'].values()))
    assert relative<.002,relative
    report={'prediction_exact':True,'serialized_optimizer_tensors_exact':True,'resumed_master_relative_l2':relative,'resumed_master_tensors_exact':all(torch.equal(opt.state['masters'][n],oc.state['masters'][n]) for n in opt.names),'checkpoint_bytes':path.stat().st_size,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    del model,opt,clone,oc,tokens,x,y,saved,before,after,loss
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); monitor.check()
    return report

def profile_optimizer(monitor):
    model=LanguageModel('rec_full',42); opt=Hybrid(model,convert_to_bf16(model,'cuda'))
    tokens=torch.randint(1,32768,(8,257),generator=torch.Generator().manual_seed(1042)).cuda(); x,y=tokens[:,:-1],tokens[:,1:]
    objective(model,x,y).backward()
    for _ in range(3): opt.apply(); torch.cuda.synchronize(); monitor.check()
    assert opt.finite()
    baseline=[]
    for _ in range(5):
        start=time.perf_counter(); opt.apply(); torch.cuda.synchronize(); baseline.append((time.perf_counter()-start)*1000); monitor.check()
    supported=torch.profiler.supported_activities()
    report={'updates':5,'fixed_precomputed_bf16_gradients':True,'unprofiled_warmed_optimizer_ms':baseline,'unprofiled_median_ms':statistics.median(baseline),'supported_activities':[str(x) for x in supported]}
    activities=[torch.profiler.ProfilerActivity.CPU]
    if torch.profiler.ProfilerActivity.CUDA in supported: activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiled=[]
    with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as prof:
        for _ in range(5):
            monitor.check(); start=time.perf_counter()
            with torch.profiler.record_function('hybrid_optimizer_profile_update'):
                opt.apply(); torch.cuda.synchronize()
            profiled.append((time.perf_counter()-start)*1000)
    events=prof.events(); averages=prof.key_averages()
    gpu=[e for e in events if str(e.device_type).endswith('CUDA')]
    kernels=defaultdict(lambda:{'count':0,'time_us':0.})
    for e in gpu:
        kernels[e.name]['count']+=1; kernels[e.name]['time_us']+=e.time_range.elapsed_us()
    launches=[e for e in events if str(e.device_type).endswith('CPU') and (e.name.startswith('cudaLaunch') or e.name.startswith('cuLaunch'))]
    operators=[{'name':e.key,'count':e.count,'self_cpu_us':e.self_cpu_time_total,'cpu_total_us':e.cpu_time_total,'self_device_us':getattr(e,'self_device_time_total',0),'device_total_us':getattr(e,'device_time_total',0)} for e in averages]
    report.update(profiled_optimizer_ms=profiled,profiled_median_ms=statistics.median(profiled),cuda_kernel_events=len(gpu),cuda_kernel_total_us=sum(v['time_us'] for v in kernels.values()) if gpu else None,top_cuda_kernels=[{'name':n,**v} for n,v in sorted(kernels.items(),key=lambda kv:kv[1]['time_us'],reverse=True)[:20]],cuda_launch_api_calls=len(launches),cuda_launch_api_cpu_total_us=sum(e.cpu_time_total for e in launches),top_operators_by_device_time=sorted(operators,key=lambda e:e['device_total_us'],reverse=True)[:25],top_operators_by_self_cpu=sorted(operators,key=lambda e:e['self_cpu_us'],reverse=True)[:25],cautions=['Profiler changes overhead; use original unprofiled windows for headline throughput','CUDA leaf-kernel total excludes double-counting nested operator GPU totals','Host launch API time is only measured runtime-call duration, not all Python overhead','GPU totals and CPU durations overlap and are not additive to wall time','Fixed-gradient optimizer-only probe preserves update formulas and tensor shapes but excludes forward/backward/data loading'])
    report['profiling_status']='GPU kernel timeline available' if gpu else 'No CUDA leaf-kernel timeline; operator event timing only; no inferred GPU compute total'
    assert opt.finite(); monitor.check()
    return report

def main(out,checkpoint_dir):
    assert not out.exists(); out.mkdir(parents=True)
    freeze=verify_integrity(); config=json.loads((VERSION/'config.json').read_text())
    result={'status':'RUNNING','freeze':freeze,'prior_gpu_active_seconds':config['prior_gpu_active_s'],'profiling_is_addendum_not_new_benchmark_suite':True}
    monitor=None
    try:
        monitor=Monitor(out,config); monitor.cooldown()
        with monitor.active('addendum-cuda-initialize'):
            torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            free,total=torch.cuda.mem_get_info(); budget=min(free-1536*1024**2,int(total*.75)); assert budget>2*1024**3
            torch.cuda.set_per_process_memory_fraction(budget/total)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            with monitor.active('full-geometry-verification'):
                result['full_geometry']=full_geometry_checks(monitor)
                (out/'full-geometry-verification.json').write_text(json.dumps(result['full_geometry'],indent=2))
            monitor.cooldown()
            with monitor.active('cuda-checkpoint-reload'):
                result['cuda_checkpoint']=gpu_checkpoint(checkpoint_dir,monitor)
                (out/'cuda-checkpoint-verification.json').write_text(json.dumps(result['cuda_checkpoint'],indent=2))
            monitor.cooldown()
            with monitor.active('optimizer-profile'):
                result['optimizer_profile']=profile_optimizer(monitor)
                (out/'optimizer-profile.json').write_text(json.dumps(result['optimizer_profile'],indent=2))
        result['status']='PASS'
    except Exception as e:
        result.update(status='FAIL',error=str(e),traceback=traceback.format_exc()); raise
    finally:
        if monitor:
            result.update(cumulative_gpu_active_s=monitor.used(),addendum_gpu_active_s=monitor.used()-config['prior_gpu_active_s'],max_gpu_temperature_c=max(r['temperature_c'] for r in monitor.rows),telemetry_samples=len(monitor.rows),max_sample_gap_s=max((b['monotonic_s']-a['monotonic_s'] for a,b in zip(monitor.rows,monitor.rows[1:])),default=0))
            monitor.close()
        (out/'results.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result.get(k) for k in ('status','cumulative_gpu_active_s','addendum_gpu_active_s','max_gpu_temperature_c','error')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--checkpoint-dir',type=Path,required=True)
    args=parser.parse_args(); main(args.output,args.checkpoint_dir)
