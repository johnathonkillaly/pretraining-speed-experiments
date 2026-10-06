"""Synchronized VJP timing and allocator measurements; all raw repeats saved."""
import argparse, gc, json, time, traceback
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from moe import Expert, expert_forward, gathered, grouped

ROOT = Path(__file__).resolve().parents[1]
mx.set_memory_limit(16*1024**3)
mx.set_cache_limit(1024**3)


def arrays(e,d,h,dtype,layout):
    rng = np.random.default_rng(123)
    source = {name:(rng.normal(size=shape)*scale).astype(np.float32)
              for name,shape,scale in [('gate',(e,h,d),d**-.5),
                                      ('up',(e,h,d),d**-.5),
                                      ('down',(e,d,h),h**-.5)]}
    if layout == 'independent':
        modules = [Expert(d,h) for _ in range(e)]
        for i,module in enumerate(modules):
            for name,v in source.items():
                getattr(module,name).weight = mx.array(v[i].copy()).astype(dtype)
        return [{name:getattr(module,name).weight for name in source}
                for module in modules]
    if layout == 'dense':
        return {name:mx.array(v[0].copy()).astype(dtype) for name,v in source.items()}
    return {name:mx.array(v).astype(dtype) for name,v in source.items()}


def case(layout,e,d,h,n,k,pattern,dtype_name,repeats=15):
    dtype = getattr(mx,dtype_name)
    family = ('independent' if layout.startswith('independent') else
              'dense' if layout == 'dense' else 'stacked')
    p = arrays(e,d,h,dtype,family)
    rng = np.random.default_rng(456)
    x = mx.array(rng.normal(size=(n,d)).astype(np.float32)).astype(dtype)
    if pattern == 'one':
        ids_np = np.tile(np.arange(k,dtype=np.int32),(n,1))
    else:
        ids_np = (np.arange(n)[:,None]+np.arange(k)[None,:])%e
        ids_np = ids_np.astype(np.int32)
    ids = mx.array(ids_np)
    touched = sorted(np.unique(ids_np).tolist())
    original = p
    if layout == 'independent_touched':
        p = [p[i] for i in touched]
        remapped = np.vectorize({v:i for i,v in enumerate(touched)}.get)(ids_np)
    flat = tree_flatten(p)
    keys,weights = zip(*flat)
    weights = list(weights)
    def f(*args):
        pp = tree_unflatten(list(zip(keys,args[:-1])))
        xx = args[-1]
        if layout == 'dense':
            y = expert_forward(pp,xx)
        elif layout == 'stacked_slice':
            y = grouped(pp,xx,ids_np,'stacked').sum(axis=1)/k
        elif layout == 'independent':
            y = grouped(pp,xx,ids_np,'independent').sum(axis=1)/k
        elif layout == 'independent_touched':
            y = grouped(pp,xx,remapped,'independent').sum(axis=1)/k
        elif layout.startswith('gather'):
            y = gathered(pp,xx,ids,layout=='gather_sorted').sum(axis=1)/k
        elif layout == 'all_experts':
            all_y = mx.stack([expert_forward({name:w[i] for name,w in pp.items()},xx)
                              for i in range(e)],axis=1)
            y = mx.take_along_axis(all_y,ids[...,None],axis=1).sum(axis=1)/k
        else:
            raise ValueError(layout)
        return mx.mean(y.astype(mx.float32)**2)
    mx.eval(original,x,ids)
    primals = weights+[x]
    mx.clear_cache()
    rows = []
    first = None
    for repeat in range(repeats+4):
        gc.collect()
        baseline = mx.get_active_memory()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        outs,grads = mx.vjp(f,primals,[mx.array(1.0)])
        mx.eval(outs)
        t1 = time.perf_counter()
        forward_retained = mx.get_active_memory()-baseline
        mx.eval(grads)
        t2 = time.perf_counter()
        retained = mx.get_active_memory()-baseline
        peak_delta = mx.get_peak_memory()-baseline
        loss_value = float(outs[0])
        del outs
        gc.collect()
        gradient_only_retained = mx.get_active_memory()-baseline
        row = {'repeat':repeat,'forward_ms':(t1-t0)*1000,
               'backward_ms':(t2-t1)*1000,'total_ms':(t2-t0)*1000,
               'baseline_bytes':baseline,'forward_retained_bytes':forward_retained,
               'gradient_retained_delta_bytes':retained,'peak_delta_bytes':peak_delta,
               'gradient_only_retained_bytes':gradient_only_retained}
        if repeat == 0:
            first = row.copy()
            # Norms force real gradient evaluation; inspect outside timing/memory.
            audit=[]
            expert_norms=np.zeros(e)
            for name,g in zip(keys,grads[:-1]):
                norms = (mx.sum(g.astype(mx.float32)**2,axis=(1,2)) if family=='stacked'
                         else mx.sum(g.astype(mx.float32)**2))
                nv=np.array(norms)
                if family=='stacked':
                    expert_norms+=nv
                elif family=='independent':
                    local=int(name.split('.')[0]); actual=touched[local] if layout=='independent_touched' else local
                    expert_norms[actual]+=float(nv)
                else:
                    expert_norms[0]+=float(nv)
                audit.append({'path':name,'shape':list(g.shape),'dtype':str(g.dtype),
                              'logical_bytes':g.nbytes,'norm':float(mx.sqrt(mx.sum(norms)))})
            gx=grads[-1]
            grad_audit={'objects':len(grads)-1,'weights':audit,
                        'weight_logical_bytes':sum(g.nbytes for g in grads[:-1]),
                        'input_gradient_bytes':gx.nbytes,
                        'expert_gradient_norms':np.sqrt(expert_norms).tolist(),
                        'loss':loss_value}
            del g,gx,norms,nv
        if repeat>=4:
            rows.append(row)
        del grads
        mx.synchronize()
    # Export before evaluation; evaluated arrays alone have no useful graph.
    graph_out,graph_grad = mx.vjp(f,primals,[mx.array(1.0)])
    mx.export_to_dot(str(ROOT/'metrics'/f'graph_{layout}_{pattern}_{d}_{n}_{e}_{k}_{dtype_name}.dot'),*graph_grad)
    del graph_out,graph_grad
    summary={key:{'median':float(np.median([r[key] for r in rows])),
                  'mean':float(np.mean([r[key] for r in rows])),
                  'std':float(np.std([r[key] for r in rows])),
                  'p10':float(np.quantile([r[key] for r in rows],.1)),
                  'p90':float(np.quantile([r[key] for r in rows],.9))}
             for key in rows[0] if key!='repeat'}
    return {'layout':layout,'experts':e,'d':d,'h':h,'tokens':n,'top_k':k,
            'pattern':pattern,'dtype':dtype_name,'touched_experts':touched,
            'expert_counts':np.bincount(ids_np.flatten(),minlength=e).tolist(),
            'first_call':first,'summary':summary,'gradient_audit':grad_audit,'repeats':rows}


def compile_probe():
    p=arrays(8,256,864,mx.float32,'stacked')
    x=mx.random.normal((512,256)); mx.eval(p,x)
    count=[0]
    def loss(pp,xx,ids):
        count[0]+=1
        return mx.mean(gathered(pp,xx,ids,True).astype(mx.float32)**2)
    fn=mx.compile(mx.value_and_grad(loss,argnums=(0,1)))
    times=[]
    for i in range(20):
        ids=((mx.arange(512)+i)%8)[:,None]; mx.eval(ids)
        t=time.perf_counter(); result=fn(p,x,ids); mx.eval(result)
        times.append((time.perf_counter()-t)*1000)
        del result
    return {'trace_count':count[0],'first_ms':times[0],
            'steady_ms':times[4:],'median_ms':float(np.median(times[4:]))}


def routing_probe():
    timings=[]
    for n in (32,512,2048):
        x=mx.random.normal((n,256));w=mx.random.normal((8,256));mx.eval(x,w)
        for k in (1,2):
            for compiled in (False,True):
                def route(xx,ww):
                    logits=xx@ww.T; p=mx.softmax(logits,axis=-1)
                    ids=mx.argpartition(-logits,kth=k-1,axis=-1)[:,:k]
                    return ids,mx.take_along_axis(p,ids,axis=-1),mx.argsort(ids.reshape(-1))
                fn=mx.compile(route) if compiled else route
                ts=[]
                for i in range(24):
                    t=time.perf_counter();y=fn(x,w);mx.eval(y);ts.append((time.perf_counter()-t)*1000);del y
                timings.append({'tokens':n,'d':256,'e':8,'k':k,'compiled':compiled,
                                'first_ms':ts[0],'median_ms':float(np.median(ts[4:])),
                                'std_ms':float(np.std(ts[4:]))})
    return timings


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',default='primary')
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    if args.suite=='primary':
        cases=[(l,8,256,864,512,1,p,'float32')
               for p in ('one','balanced')
               for l in ('dense','all_experts','stacked_slice','independent',
                         'independent_touched','gather_unsorted','gather_sorted')]
    else:
        cases=[(l,e,d,h,n,1,'balanced',dtype)
               for d,h,dtype in ((256,864,'float32'),(1024,3456,'bfloat16'))
               for e in (4,8,16) for n in (32,512,2048)
               for l in ('dense','stacked_slice','independent','gather_sorted')]
        cases += [(l,8,1024,3456,512,k,p,'bfloat16')
                  for k in (1,2) for p in ('one','balanced')
                  for l in ('dense','all_experts','stacked_slice','independent','gather_sorted')]
    output=ROOT/'metrics'/f'benchmark_{args.suite}{args.tag}.jsonl'
    with output.open('x') as f:
        for spec in cases:
            print('CASE',spec,flush=True)
            try:
                row=case(*spec)
                print('RESULT',row['summary']['total_ms']['median'],
                      row['summary']['gradient_retained_delta_bytes']['median'],flush=True)
            except Exception:
                row={'spec':spec,'error':traceback.format_exc()};print(row['error'],flush=True)
            f.write(json.dumps(row)+'\n');f.flush();gc.collect();mx.clear_cache()
    if args.suite=='primary':
        (ROOT/'metrics'/'compile_router.json').write_text(json.dumps(
            {'compile':compile_probe(),'routing':routing_probe()},indent=2))

if __name__=='__main__': main()
