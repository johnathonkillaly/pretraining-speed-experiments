"""Numerical gates before benchmark work; CPU double tests isolate detach math."""
import hashlib, math
from pathlib import Path
import torch
from torch.nn import functional as F
from model import LanguageModel,Geometry,objective,convert_to_bf16
from optimizer import Hybrid,zeropower

SMALL=Geometry(d=32,h=64,heads=4,kv_heads=2,head_dim=8,vocab=128,context=16)

def grads(model,x,y,k,trace=None):
    model.zero_grad(set_to_none=True)
    loss=objective(model,x,y,4,k,trace=trace)
    loss.backward()
    return loss.detach(),{n:p.grad.detach().clone() for n,p in model.named_parameters()}

def numerical_cpu(checkpoint_dir):
    report={}
    torch.set_num_threads(8)
    model=LanguageModel('rec_full',71,SMALL).double()
    gen=torch.Generator().manual_seed(5)
    tokens=torch.randint(1,128,(2,17),generator=gen); x,y=tokens[:,:-1],tokens[:,1:]
    reference,g1=grads(model,x,y,4); same,g2=grads(model,x,y,8)
    assert torch.equal(reference,same)
    assert all(torch.equal(g1[n],g2[n]) for n in g1)
    trace={}; truncated,gt=grads(model,x,y,2,trace)
    assert torch.equal(reference,truncated)
    with torch.no_grad():
        one=model(x,1,1); dense=LanguageModel('dense4',71,SMALL).double()(x)
    assert torch.equal(one,dense)
    report.update(full_vs_truncated_forward_exact=True,k_ge_r_gradient_exact=True,one_loop_vs_dense4_exact=True)
    norms={prefix:math.sqrt(sum(float(g.square().sum()) for n,g in gt.items() if n.startswith(prefix))) for prefix in ('embed_tokens','prelude','core','coda','norm')}
    assert all(v>0 for v in norms.values()),norms
    assert trace['loop0'].grad is None and trace['loop1'].grad is None
    assert trace['loop2'].grad.norm()>0 and trace['loop3'].grad.norm()>0 and trace['prelude'].grad.norm()>0
    report.update(retained_parameter_gradient_norms=norms,actual_loop_prefix_gradients_absent=True,retained_suffix_and_prelude_activation_gradients_nonzero=True)
    # Freeze the actual prefix state; differentiate only suffix operations and
    # the prelude injection, matching the conditional derivative TB2 computes.
    fd=[]
    for prefix in ('core.0.self_attn.q_proj','prelude.self_attn.q_proj'):
        name=prefix+'.weight'; p=dict(model.named_parameters())[name]
        flat_index=int(gt[name].abs().argmax()); index=(flat_index//p.shape[1],flat_index%p.shape[1])
        value=float(p[index].detach()); eps=1e-5
        with torch.no_grad():
            p[index]=value+eps; plus=float(objective(model,x,y,4,2,frozen_prefix=trace['prefix']))
            p[index]=value-eps; minus=float(objective(model,x,y,4,2,frozen_prefix=trace['prefix']))
            p[index]=value
        estimate=(plus-minus)/(2*eps); analytic=float(gt[name][index])
        assert abs(estimate-analytic)<=2e-7+.002*abs(analytic),(name,estimate,analytic)
        fd.append({'name':name,'index':list(index),'analytic':analytic,'finite_difference':estimate,'epsilon':eps})
    report['actual_conditional_finite_differences']=fd
    # Independent attention equation and half-pair RoPE identity.
    q=torch.randn(2,4,16,8,dtype=torch.float64); k=torch.randn_like(q); v=torch.randn_like(q)
    mask=torch.tril(torch.ones(16,16,dtype=torch.bool))
    scores=(q@k.transpose(-1,-2))*8**-.5
    manual=scores.masked_fill(~mask,float('-inf')).softmax(-1)@v
    sdpa=F.scaled_dot_product_attention(q,k,v,is_causal=True)
    torch.testing.assert_close(sdpa,manual,atol=1e-12,rtol=1e-12)
    rope=model.core[0].self_attn.rope(q)
    torch.testing.assert_close(rope[:,:,0],q[:,:,0],atol=0,rtol=0)
    torch.testing.assert_close(rope.square().sum(-1),q.square().sum(-1),atol=2e-6,rtol=2e-6)
    report.update(causal_sdpa_vs_manual_double=True,nontraditional_rope_position_zero_identity_and_norm=True)
    torch.manual_seed(8)
    ns=[]
    for rows,cols in ((64,32),(32,64)):
        values=torch.randn(5,rows,cols)*torch.tensor([.01,.1,1,10,100])[:,None,None]
        batched=zeropower(values); separate=torch.stack([zeropower(v) for v in values])
        relative=float((batched-separate).norm()/separate.norm())
        assert relative<.02,relative
        ns.append({'shape':[5,rows,cols],'relative_l2':relative})
    report['batched_vs_individual_muon']=ns
    a=LanguageModel('rec_tb2',91,SMALL); b=LanguageModel('rec_tb2',91,SMALL)
    oa=Hybrid(a,convert_to_bf16(a,'cpu')); ob=Hybrid(b,convert_to_bf16(b,'cpu'))
    ob.groups=[[n] for n in ob.muon_names]
    for step in range(3):
        gradients={n:torch.randn_like(v)*.1 for n,v in oa.state['masters'].items()}
        if step==0:
            norm=torch.stack([g.square().sum() for g in gradients.values()]).sum().sqrt()
            scale=torch.minimum(torch.ones_like(norm),1/(norm+1e-6)); name=oa.muon_names[0]
            g=gradients[name]*scale; rows,cols=g.shape
            expected=oa.state['masters'][name]-.01*max(1,rows/cols)**.5*zeropower(.05*g+.95*(.05*g))
            adam_name='embed_tokens.weight'; ag=gradients[adam_name]*scale
            am=.1*ag; av=.05*ag*ag
            adam_expected=oa.state['masters'][adam_name]-.0003*(am/(1-.9))/((av/(1-.95)).sqrt()+1e-8)
        norm_a=oa.apply(gradients=gradients); norm_b=ob.apply(gradients=gradients)
        assert torch.equal(norm_a,norm_b)
        if step==0:
            torch.testing.assert_close(oa.state['masters'][name],expected,rtol=2e-3,atol=2e-4)
            torch.testing.assert_close(oa.state['masters'][adam_name],adam_expected,rtol=1e-6,atol=1e-7)
    master_relative=math.sqrt(sum(float((oa.state['masters'][n]-ob.state['masters'][n]).square().sum()) for n in oa.names)/sum(float(v.square().sum()) for v in oa.state['masters'].values()))
    assert master_relative<.002,master_relative
    assert oa.finite() and ob.finite()
    assert all(p.dtype==torch.bfloat16 for p in a.parameters())
    assert all(v.dtype==torch.float32 for group in ('masters','muon_m','adam_m','adam_v') for v in oa.state[group].values())
    report.update(hybrid_batched_vs_individual_three_step_master_relative_l2=master_relative,source_nesterov_and_adam_first_step_formula=True,ownership=oa.ownership(),bf16_model_fp32_masters_state=True)
    checkpoint_dir.mkdir(parents=True,exist_ok=True); path=checkpoint_dir/'small-port-checkpoint.pt'
    torch.save({'model':a.state_dict(),'optimizer':oa.state,'geometry':SMALL.__dict__},path)
    saved=torch.load(path,weights_only=True)
    c=LanguageModel('rec_tb2',91,SMALL); oc=Hybrid(c,convert_to_bf16(c,'cpu'))
    c.load_state_dict(saved['model']); oc.state=saved['optimizer']
    with torch.no_grad():
        before=a(x); after=c(x)
    assert torch.equal(before,after)
    for m,o in ((a,oa),(c,oc)):
        m.zero_grad(set_to_none=True); loss=objective(m,x,y); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        o.apply(); assert o.finite()
    assert all(torch.equal(oa.state['masters'][n],oc.state['masters'][n]) for n in oa.names)
    report.update(checkpoint_prediction_exact=True,checkpoint_resumed_update_exact=True,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_bytes=path.stat().st_size,passed=True)
    return report

def numerical_cuda():
    model=LanguageModel('rec_full',71,SMALL); optimizer=Hybrid(model,convert_to_bf16(model,'cuda'))
    gen=torch.Generator().manual_seed(5); tokens=torch.randint(1,128,(2,17),generator=gen).cuda(); x,y=tokens[:,:-1],tokens[:,1:]
    full,g1=grads(model,x,y,4); same,g2=grads(model,x,y,8); trace={}; truncated,gt=grads(model,x,y,2,trace)
    assert torch.equal(full,same) and torch.equal(full,truncated)
    rel=max(float((g1[n].float()-g2[n].float()).norm()/(g1[n].float().norm()+1e-12)) for n in g1)
    assert rel<.02,rel
    assert all(torch.isfinite(g).all() for g in gt.values())
    assert trace['loop0'].grad is None and trace['loop1'].grad is None and trace['loop2'].grad.norm()>0
    norm=optimizer.apply(); assert torch.isfinite(norm) and optimizer.finite()
    assert all(p.grad.dtype==torch.bfloat16 for p in model.parameters())
    values=torch.randn(5,64,32,device='cuda')*torch.tensor([.01,.1,1,10,100],device='cuda')[:,None,None]
    batched=zeropower(values); individual=torch.stack([zeropower(v) for v in values])
    ns=float((batched-individual).norm()/individual.norm()); assert ns<.02,ns
    return {'passed':True,'bf16_forward_full_vs_tb2_exact':True,'k_ge_r_max_parameter_gradient_relative_l2':rel,'bf16_gradients':True,'detach_boundary':True,'batched_vs_individual_muon_relative_l2':ns,'finite_hybrid_step':True,'loss':float(full),'gradient_norm':float(norm)}
