"""Source-defined hybrid update; no torch.optim Muon substitution."""
from collections import defaultdict
import torch

SUFFIXES=tuple(f'{role}.weight' for role in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj'))

def zeropower(x,steps=5):
    single=x.ndim==2
    if single: x=x.unsqueeze(0)
    transpose=x.shape[-2]>x.shape[-1]
    if transpose: x=x.transpose(-1,-2)
    x=(x/(x.square().sum((-2,-1),keepdim=True).sqrt()+1e-7)).bfloat16()
    for _ in range(steps):
        gram=x@x.transpose(-1,-2)
        polynomial=torch.baddbmm(-4.7750*gram,gram,gram,beta=1,alpha=2.0315)
        x=torch.baddbmm(3.4445*x,polynomial,x,beta=1,alpha=1)
    if transpose: x=x.transpose(-1,-2)
    return x.float()[0] if single else x.float()

class Hybrid:
    def __init__(self,model,masters):
        self.model=model; self.params=dict(model.named_parameters()); self.names=tuple(sorted(masters))
        self.muon_names=tuple(n for n in self.names if n.endswith(SUFFIXES))
        self.adam_names=tuple(n for n in self.names if n not in self.muon_names)
        assert set(self.params)==set(masters)
        for n in self.muon_names: assert masters[n].ndim==2
        for n in self.adam_names: assert n=='embed_tokens.weight' or masters[n].ndim==1
        shapes=defaultdict(list)
        for n in self.muon_names: shapes[tuple(masters[n].shape)].append(n)
        self.groups=[ns[i:i+5] for ns in shapes.values() for i in range(0,len(ns),5)]
        self.state={'masters':masters,'muon_m':{n:torch.zeros_like(masters[n]) for n in self.muon_names},'adam_m':{n:torch.zeros_like(masters[n]) for n in self.adam_names},'adam_v':{n:torch.zeros_like(masters[n]) for n in self.adam_names},'step':0}
    @torch.no_grad()
    def apply(self,muon_lr=.01,adam_lr=.0003,gradients=None):
        gradients={n:self.params[n].grad.float() for n in self.names} if gradients is None else {n:gradients[n].float() for n in self.names}
        norm=torch.stack([gradients[n].square().sum() for n in self.names]).sum().sqrt()
        scale=torch.minimum(torch.ones_like(norm),1/(norm+1e-6))
        gradients={n:g*scale for n,g in gradients.items()}
        self.state['step']+=1; step=self.state['step']; masters=self.state['masters']
        for names in self.groups:
            updates=[]
            for n in names:
                g=gradients[n]; momentum=.95*self.state['muon_m'][n]+.05*g
                self.state['muon_m'][n]=momentum
                updates.append(.05*g+.95*momentum)
            update=zeropower(torch.stack(updates)); rows,cols=masters[names[0]].shape
            rate=muon_lr*max(1,rows/cols)**.5
            for i,n in enumerate(names): masters[n]=masters[n]-rate*update[i]
        for n in self.adam_names:
            g=gradients[n]; m=.9*self.state['adam_m'][n]+.1*g; v=.95*self.state['adam_v'][n]+.05*g*g
            self.state['adam_m'][n]=m; self.state['adam_v'][n]=v
            masters[n]=masters[n]-adam_lr*(m/(1-.9**step))/((v/(1-.95**step)).sqrt()+1e-8)
        for n in self.names: self.params[n].copy_(masters[n].bfloat16())
        return norm
    def ownership(self):
        return {'muon_tensors':len(self.muon_names),'adamw_tensors':len(self.adam_names),'muon_parameters':sum(self.state['masters'][n].numel() for n in self.muon_names),'adamw_parameters':sum(self.state['masters'][n].numel() for n in self.adam_names),'matrix_groups':self.groups,'tied_embedding_head_occurrences':sum(n=='embed_tokens.weight' for n in self.names),'weight_decay':0}
    def finite(self):
        return all(torch.isfinite(v).all().item() for group in ('masters','muon_m','adam_m','adam_v') for v in self.state[group].values()) and all(torch.isfinite(p).all().item() for p in self.params.values())
