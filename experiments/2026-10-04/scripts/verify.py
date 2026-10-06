import gc,json,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
from mlx.utils import tree_flatten
from moe import Expert,init_experts,gathered,grouped,expert_forward,SparseMoE

ROOT=Path(__file__).resolve().parents[1]

def main():
    mx.random.seed(567)
    checks=[]
    for n,k,pattern in ((12,1,'one'),(12,1,'balanced'),(12,2,'balanced'),(9,2,'one')):
        p=init_experts(8,16,32);x=mx.random.normal((n,16))
        ids_np=np.tile(np.arange(k),(n,1)) if pattern=='one' else (np.arange(n)[:,None]+np.arange(k))%8
        ids=mx.array(ids_np.astype(np.int32))
        fn1=lambda pp,xx:mx.mean(grouped(pp,xx,ids_np,'stacked')**2)
        fn2=lambda pp,xx:mx.mean(gathered(pp,xx,ids,True)**2)
        fn3=lambda pp,xx:mx.mean(gathered(pp,xx,ids,False)**2)
        r1=mx.value_and_grad(fn1,argnums=(0,1))(p,x)
        r2=mx.value_and_grad(fn2,argnums=(0,1))(p,x)
        r3=mx.value_and_grad(fn3,argnums=(0,1))(p,x)
        mx.eval(r1,r2,r3)
        diffs=[]
        for reference,candidate in ((r1,r2),(r1,r3)):
            for (a,v),(b,w) in zip(tree_flatten(reference),tree_flatten(candidate)):
                assert a==b
                np.testing.assert_allclose(np.array(v),np.array(w),rtol=2e-4,atol=1e-5)
                diffs.append(float(mx.max(mx.abs(v-w))))
        checks.append({'tokens':n,'k':k,'pattern':pattern,'max_difference':max(diffs)})
    # Central finite difference for input and selected/unselected parameters.
    p=init_experts(8,8,16);x=mx.random.normal((6,8));ids=mx.zeros((6,1),dtype=mx.int32)
    fn=lambda pp,xx:mx.mean(gathered(pp,xx,ids,True)**2)
    value,(g,gx)=mx.value_and_grad(fn,argnums=(0,1))(p,x);mx.eval(value,g,gx)
    eps=.001
    pp=dict(p);w=np.array(p['gate']);w[0,0,0]+=eps;pp['gate']=mx.array(w);plus=float(fn(pp,x))
    w[0,0,0]-=2*eps;pp['gate']=mx.array(w);minus=float(fn(pp,x))
    fd=(plus-minus)/(2*eps);analytic=float(g['gate'][0,0,0]);assert abs(fd-analytic)<2e-4
    assert float(mx.max(mx.abs(g['gate'][1:])))==0
    checks.append({'finite_difference':fd,'analytic':analytic,'unselected_gradient_max':0})
    # Ordinary nn.Module autodiff, not just equivalent functional pytrees.
    class Independent(nn.Module):
        def __init__(self):
            super().__init__();self.experts=[Expert(256,864) for _ in range(8)]
        def __call__(self,x):return self.experts[0](x)
    memory=[]
    for touched_only in (False,True):
        gc.collect();mx.clear_cache();m=Independent();xx=mx.random.normal((512,256));mx.eval(m.parameters(),xx)
        target=m.experts[0] if touched_only else m
        fn=nn.value_and_grad(target,lambda xx:mx.mean(target(xx)**2))
        base=mx.get_active_memory();value,grads=fn(xx);mx.eval(value,grads);loss=float(value);del value;gc.collect()
        gradient_delta=mx.get_active_memory()-base
        grad_bytes=sum(v.nbytes for _,v in tree_flatten(grads))
        unselected_norms=[]
        if not touched_only:
            for ep in grads['experts'][1:]:unselected_norms.append(float(sum((mx.sum(v*v) for _,v in tree_flatten(ep)),mx.array(0.))))
            assert all(v==0 for v in unselected_norms)
        before=np.array(m.experts[7].gate.weight)
        optimizer=opt.AdamW(learning_rate=.001,weight_decay=.1)
        t=time.perf_counter();optimizer.update(target,grads);mx.eval(target.parameters(),optimizer.state);elapsed=(time.perf_counter()-t)*1000
        state_bytes=sum(v.nbytes for _,v in tree_flatten(optimizer.state))
        changed=float(np.max(np.abs(np.array(m.experts[7].gate.weight)-before)))
        memory.append({'touched_only':touched_only,'loss':loss,'gradient_objects':len(tree_flatten(grads)),
                       'gradient_logical_bytes':grad_bytes,'gradient_physical_delta_bytes':gradient_delta,
                       'unselected_squared_norms':unselected_norms,'optimizer_first_step_ms':elapsed,
                       'optimizer_state_logical_bytes':state_bytes,
                       'untouched_expert_max_weight_change':changed})
        del m,target,fn,xx,grads,optimizer;gc.collect();mx.clear_cache()
    # Router task gradients survive detached discrete dispatch.
    router=[]
    for k in (1,2):
        moe=SparseMoE(16,32,8,k,aux_coeff=0)
        vg=nn.value_and_grad(moe,lambda x:mx.mean(moe(x)**2));v,g=vg(mx.random.normal((12,16)));mx.eval(v,g)
        router_norm=float(mx.sqrt(mx.sum(g['router']['weight']**2)));assert router_norm>0
        router.append({'k':k,'task_router_gradient_norm':router_norm})
    result={'equivalence':checks,'independent_nn_and_optimizer':memory,'router':router,'passed':True}
    (ROOT/'metrics'/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
