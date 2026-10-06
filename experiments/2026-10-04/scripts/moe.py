"""Isolated sparse SwiGLU experiments; weights follow BoothGPT's out,in layout."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class Expert(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


def expert_forward(p, x):
    return (nn.silu(x @ p['gate'].T) * (x @ p['up'].T)) @ p['down'].T


def init_experts(e, d, h, dtype=mx.float32):
    return {name: (mx.random.normal(shape) * scale).astype(dtype)
            for name, shape, scale in [
                ('gate', (e,h,d), d**-.5), ('up', (e,h,d), d**-.5),
                ('down', (e,d,h), h**-.5)]}


def gathered(p, x, ids, sorted_indices=True):
    """No expert-weight take: gather_mm reads selected matrices directly."""
    n, k = ids.shape
    flat_ids = ids.reshape(-1)
    rows = mx.broadcast_to(mx.arange(n)[:,None], (n,k)).reshape(-1)
    if sorted_indices:
        order = mx.argsort(flat_ids)
        inverse = mx.argsort(order)
        flat_ids, rows = flat_ids[order], rows[order]
    xx = x[rows][:,None,:]
    def mm(z, w):
        return mx.gather_mm(z, w.swapaxes(-1,-2), rhs_indices=flat_ids,
                            sorted_indices=sorted_indices)
    y = mm(nn.silu(mm(xx,p['gate'])) * mm(xx,p['up']), p['down'])[:,0,:]
    if sorted_indices:
        y = y[inverse]
    return y.reshape(n,k,-1)


def grouped(p, x, ids_np, layout):
    """Static host groups for controlled benchmarks, never route all-token GEMMs."""
    n,k = ids_np.shape
    pieces, positions = [], []
    for e in np.unique(ids_np):
        token, slot = np.where(ids_np == e)
        index = mx.array(token)
        ep = ({name:w[int(e)] for name,w in p.items()} if layout == 'stacked'
              else p[int(e)])
        pieces.append(expert_forward(ep, x[index]))
        positions.append(mx.array(token*k+slot))
    y = mx.concatenate(pieces,axis=0)
    pos = mx.concatenate(positions)
    return y[mx.argsort(pos)].reshape(n,k,-1)


class SparseMoE(nn.Module):
    def __init__(self, d, h, experts=8, top_k=1, aux_coeff=.01, layout='stacked'):
        super().__init__()
        self.router = nn.Linear(d, experts, bias=False)
        self.expert_count, self.top_k = experts, top_k
        self.aux_coeff, self.layout = aux_coeff, layout
        if layout == 'stacked':
            self.weights = init_experts(experts,d,h)
        elif layout == 'independent':
            self.experts = [Expert(d,h) for _ in range(experts)]
        else:
            raise ValueError(layout)
        self._stats = None
        self._aux = None

    def __call__(self, x):
        shape = x.shape
        xx = x.reshape(-1,shape[-1])
        logits = self.router(xx).astype(mx.float32)
        probs = mx.softmax(logits,axis=-1)
        ids = mx.stop_gradient(mx.argpartition(-logits,kth=self.top_k-1,axis=-1)[:,:self.top_k])
        gates = mx.take_along_axis(probs,ids,axis=-1).astype(xx.dtype)
        if self.layout == 'stacked':
            outputs = gathered(self.weights,xx,ids,True)
        else:
            # Dynamic host materialization is explicit, unsuitable for compile.
            outputs = grouped([{k:v['weight'] for k,v in ep.parameters().items()}
                               for ep in self.experts], xx, np.array(ids), 'independent')
        loads = mx.mean((ids[...,None] == mx.arange(self.expert_count)).astype(mx.float32),axis=(0,1))
        entropy = -mx.mean(mx.sum(probs * mx.log(mx.maximum(probs,1e-9)),axis=-1))
        aux = self.expert_count * mx.sum(mx.mean(probs,axis=0) * mx.stop_gradient(loads))
        self._stats = (entropy,loads,aux)
        self._aux = self.aux_coeff * aux
        return mx.sum(outputs * gates[...,None],axis=1).reshape(shape)
