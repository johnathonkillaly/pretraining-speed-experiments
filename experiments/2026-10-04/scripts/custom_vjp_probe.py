"""Exploratory fixed-route custom VJP; full logical zeros, broadcast physical storage.

This is a controlled one-expert SwiGLU proof, not a dynamic-router training port.
"""
import gc,json,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten,tree_map
from moe import Expert,expert_forward

ROOT=Path(__file__).resolve().parents[1]

@mx.custom_function
def selected_ffn(x,*weights):
    return expert_forward(dict(zip(('gate','up','down'),weights[:3])),x)

@selected_ffn.vjp
def selected_vjp(primals,cotangent,output):
    x,*weights=primals
    _,selected=mx.vjp(lambda xx,gate,up,down:expert_forward(
        {'gate':gate,'up':up,'down':down},xx),[x,*weights[:3]],[cotangent])
    zero=mx.array(0.,dtype=weights[0].dtype)
    return tuple(selected+[mx.broadcast_to(zero,w.shape) for w in weights[3:]])

class Independent(nn.Module):
    def __init__(self,custom):
        super().__init__();self.experts=[Expert(256,864) for _ in range(8)];self._custom=custom
    def __call__(self,x):
        if not self._custom:return self.experts[0](x)
        w=[getattr(ep,name).weight for ep in self.experts for name in ('gate','up','down')]
        return selected_ffn(x,*w)

def norm(g):return mx.sqrt(sum((mx.sum(v*v) for _,v in tree_flatten(g)),mx.array(0.)))

def main():
    results=[];reference=None
    for custom in (False,True):
        mx.random.seed(12345);model=Independent(custom);x=mx.random.normal((512,256));mx.eval(model.parameters(),x)
        fn=nn.value_and_grad(model,lambda xx:mx.mean(model(xx)**2))
        repeats=[]
        for i in range(19):
            gc.collect();mx.clear_cache();base=mx.get_active_memory();mx.reset_peak_memory()
            t0=time.perf_counter();value,grads=fn(x);mx.eval(value);t1=time.perf_counter();mx.eval(grads);t2=time.perf_counter()
            objective=float(value);del value;gc.collect();retained=mx.get_active_memory()-base
            if i==0:
                flat=tree_flatten(grads)
                snapshots={name:np.array(g) for name,g in flat}
                if reference is None:reference=snapshots
                else:
                    for name,g in snapshots.items():np.testing.assert_allclose(g,reference[name],rtol=1e-5,atol=1e-7)
                audit={'logical_gradient_bytes':sum(g.nbytes for _,g in flat),
                       'gradient_objects':len(flat),'unselected_norms':[float(norm(ep)) for ep in grads['experts'][1:]],
                       'loss':objective}
                del flat,snapshots
            if i>=4:repeats.append({'forward_ms':(t1-t0)*1000,'backward_ms':(t2-t1)*1000,
                                   'total_ms':(t2-t0)*1000,'retained_gradient_bytes':retained})
            if i==18:
                before_clip=mx.get_active_memory();gnorm=norm(grads)
                clipped=tree_map(lambda g:g*mx.minimum(1.,1./(gnorm+1e-6)),grads);mx.eval(clipped)
                clip_added=mx.get_active_memory()-before_clip
                optimizer=optim.AdamW(learning_rate=.001,weight_decay=.1)
                before_opt=mx.get_active_memory();optimizer.update(model,clipped);mx.eval(model.parameters(),optimizer.state)
                optimizer_added=mx.get_active_memory()-before_opt
                opt_state_bytes=sum(g.nbytes for _,g in tree_flatten(optimizer.state))
                del optimizer,clipped,gnorm
            del grads;gc.collect()
        results.append({'custom':custom,'audit':audit,'repeats':repeats,
                        'median':{k:float(np.median([r[k] for r in repeats])) for k in repeats[0]},
                        'gradient_clip_additional_retained_bytes':clip_added,
                        'optimizer_additional_active_bytes':optimizer_added,
                        'optimizer_state_logical_bytes':opt_state_bytes})
        del model,fn,x;gc.collect();mx.clear_cache()
    (ROOT/'metrics/custom_vjp_swiglu.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
    models=[]
    for custom in (False,True):
        mx.random.seed(777);model=Independent(custom);x=mx.random.normal((12,256));mx.eval(model.parameters(),x);models.append((model,x))
    input_grads=[mx.grad(lambda xx:mx.mean(model(xx)**2))(x) for model,x in models]
    mx.eval(input_grads)
    np.testing.assert_allclose(np.array(input_grads[0]),np.array(input_grads[1]),rtol=1e-5,atol=1e-7)
    (ROOT/'metrics/custom_vjp_validation.json').write_text(json.dumps({
        'input_gradient_max_difference':float(mx.max(mx.abs(input_grads[0]-input_grads[1]))),
        'passed':True},indent=2))

if __name__=='__main__':main()
