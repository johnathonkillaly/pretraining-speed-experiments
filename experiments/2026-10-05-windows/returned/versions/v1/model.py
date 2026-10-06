"""Explicit PyTorch port of the frozen four-loop recurrent Llama pilot."""
import hashlib, math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class Geometry:
    d: int=256
    h: int=864
    heads: int=8
    kv_heads: int=2
    head_dim: int=32
    vocab: int=32768
    context: int=256
    eps: float=1e-5
    rope_theta: float=500000

class RMSNorm(nn.Module):
    def __init__(self,g):
        super().__init__(); self.weight=nn.Parameter(torch.ones(g.d)); self.eps=g.eps
    def forward(self,x):
        acc=x if x.dtype==torch.float64 else x.float()
        normed=acc*torch.rsqrt(acc.square().mean(-1,keepdim=True)+self.eps)
        return normed.to(x.dtype)*self.weight

class Attention(nn.Module):
    def __init__(self,g):
        super().__init__(); self.g=g
        self.q_proj=nn.Linear(g.d,g.heads*g.head_dim,bias=False)
        self.k_proj=nn.Linear(g.d,g.kv_heads*g.head_dim,bias=False)
        self.v_proj=nn.Linear(g.d,g.kv_heads*g.head_dim,bias=False)
        self.o_proj=nn.Linear(g.heads*g.head_dim,g.d,bias=False)
        angles=torch.arange(g.context,dtype=torch.float64)[:,None]*g.rope_theta**(-torch.arange(0,g.head_dim,2,dtype=torch.float64)/g.head_dim)
        # Nontraditional RoPE pairs the two halves of each head. Tables are
        # computed once, then cast with the model to its compute dtype.
        self.register_buffer('cos',angles.cos().float(),persistent=False)
        self.register_buffer('sin',angles.sin().float(),persistent=False)
    def rope(self,x):
        a,b=x.chunk(2,dim=-1); c=self.cos[:x.shape[-2]]; s=self.sin[:x.shape[-2]]
        return torch.cat((a*c-b*s,b*c+a*s),dim=-1)
    def forward(self,x):
        b,t,_=x.shape; g=self.g
        q=self.rope(self.q_proj(x).view(b,t,g.heads,g.head_dim).transpose(1,2))
        k=self.rope(self.k_proj(x).view(b,t,g.kv_heads,g.head_dim).transpose(1,2))
        v=self.v_proj(x).view(b,t,g.kv_heads,g.head_dim).transpose(1,2)
        # Explicit KV repetition permits CUDA Flash SDPA in PyTorch 2.7.
        k=k.repeat_interleave(g.heads//g.kv_heads,dim=1)
        v=v.repeat_interleave(g.heads//g.kv_heads,dim=1)
        z=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0,scale=g.head_dim**-.5)
        return self.o_proj(z.transpose(1,2).contiguous().view(b,t,g.heads*g.head_dim))

class MLP(nn.Module):
    def __init__(self,g):
        super().__init__()
        self.gate_proj=nn.Linear(g.d,g.h,bias=False)
        self.up_proj=nn.Linear(g.d,g.h,bias=False)
        self.down_proj=nn.Linear(g.h,g.d,bias=False)
    def forward(self,x): return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))

class Block(nn.Module):
    def __init__(self,g):
        super().__init__(); self.self_attn=Attention(g); self.mlp=MLP(g)
        self.input_layernorm=RMSNorm(g); self.post_attention_layernorm=RMSNorm(g)
    def forward(self,x):
        x=x+self.self_attn(self.input_layernorm(x))
        return x+self.mlp(self.post_attention_layernorm(x))

class LanguageModel(nn.Module):
    def __init__(self,kind,seed,g=None):
        super().__init__(); self.kind=kind; self.g=g or Geometry()
        assert kind in ('dense4','dense10','rec_full','rec_tb2')
        torch.manual_seed(seed)
        self.embed_tokens=nn.Embedding(self.g.vocab,self.g.d)
        nn.init.normal_(self.embed_tokens.weight,std=1/math.sqrt(self.g.d))
        bank=[Block(self.g) for _ in range(10)]
        self.norm=RMSNorm(self.g)
        digest=hashlib.sha256()
        for value in [self.embed_tokens.weight,self.norm.weight]+[p for i in (0,1,2,9) for p in bank[i].parameters()]:
            digest.update(value.detach().numpy().tobytes())
        self.shared_initialization_sha256=digest.hexdigest()
        if kind.startswith('dense'):
            self.blocks=nn.ModuleList([bank[i] for i in ((0,1,2,9) if kind=='dense4' else range(10))])
        else:
            self.prelude=bank[0]; self.core=nn.ModuleList(bank[1:3]); self.coda=bank[9]
    def forward(self,tokens,loops=4,k=None,trace=None,frozen_prefix=None):
        x=self.embed_tokens(tokens)
        if self.kind.startswith('dense'):
            for block in self.blocks: x=block(x)
        else:
            e=self.prelude(x); x=e if frozen_prefix is None else frozen_prefix
            if trace is not None:
                if e.requires_grad: e.retain_grad()
                trace['prelude']=e
            start=0 if frozen_prefix is None else 2
            k=(2 if self.kind=='rec_tb2' else loops) if k is None else k
            for i in range(start,loops):
                if i==loops-k and i>0: x=x.detach()
                if trace is not None and i==2: trace['prefix']=x.detach().clone()
                x=.5*(x+e)
                for block in self.core: x=block(x)
                if trace is not None:
                    if x.requires_grad: x.retain_grad()
                    trace[f'loop{i}']=x
            x=self.coda(x)
        return F.linear(self.norm(x),self.embed_tokens.weight)

def objective(model,x,y,loops=4,k=None,**kwargs):
    logits=model(x,loops,k,**kwargs)
    logits=logits if logits.dtype==torch.float64 else logits.float()
    losses=F.cross_entropy(logits.reshape(-1,model.g.vocab),y.reshape(-1),reduction='none')
    mask=y.reshape(-1)!=0
    return (losses*mask).sum()/mask.sum()

def convert_to_bf16(model,device):
    masters={n:p.detach().to(device=device,dtype=torch.float32).clone() for n,p in model.named_parameters()}
    model.to(device=device,dtype=torch.bfloat16)
    return masters
