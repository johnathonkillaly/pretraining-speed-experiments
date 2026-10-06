import argparse,gc,hashlib,json,math,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten,tree_map
from arch.llama import Model,ModelArgs
from moe import SparseMoE

ROOT=Path(__file__).resolve().parents[1]
mx.set_memory_limit(16*1024**3);mx.set_cache_limit(1024**3)

def build(kind,seed):
    mx.random.seed(seed)
    args=ModelArgs(model_type='llama',hidden_size=256,intermediate_size=864,
                   num_hidden_layers=8,num_attention_heads=8,num_key_value_heads=2,
                   head_dim=32,rms_norm_eps=1e-5,vocab_size=32768,
                   rope_theta=500000,max_position_embeddings=256,tie_word_embeddings=True)
    model=Model(args)
    mx.eval(model.parameters())
    shared_hash=hashlib.sha256()
    for name,w in tree_flatten(model.parameters()):
        shared_hash.update(name.encode());shared_hash.update(np.array(w).tobytes())
    if kind!='dense':
        for i in range(3,len(model.layers),4):
            old=model.layers[i].mlp
            moe=SparseMoE(256,864,8,1 if kind=='top1' else 2,.01)
            moe.weights={'gate':mx.repeat(old.gate_proj.weight[None],8,axis=0),
                         'up':mx.repeat(old.up_proj.weight[None],8,axis=0),
                         'down':mx.repeat(old.down_proj.weight[None],8,axis=0)}
            model.layers[i].mlp=moe
    mx.eval(model.parameters())
    return model,args,shared_hash.hexdigest()

def loss(model,x,y):
    logits=model(x).astype(mx.float32)
    mask=(y!=0).astype(mx.float32)
    ce=(nn.losses.cross_entropy(logits,y,reduction='none')*mask).sum()/mask.sum()
    moes=[layer.mlp for layer in model.layers if isinstance(layer.mlp,SparseMoE)]
    aux=sum((m._aux for m in moes),mx.array(0.))
    return ce+aux,ce,tuple(m._stats for m in moes)

def norm(g):return mx.sqrt(sum((mx.sum(v.astype(mx.float32)**2) for _,v in tree_flatten(g)),mx.array(0.)))

def health(stats,grads=None):
    result=[]
    for li,(entropy,loads,aux) in zip(range(3,8,4),stats):
        loads=np.array(loads);ent=float(entropy)
        record={'layer':li,'probability_entropy':ent,'utilization':loads.tolist(),
                'assignment_entropy':float(-np.sum(loads*np.log(np.maximum(loads,1e-12)))),
                'max_load':float(loads.max()),'min_load':float(loads.min()),
                'coefficient_of_variation':float(loads.std()/loads.mean()),
                'balance_loss':float(aux),'dropped_tokens':0,'overflow_tokens':0,
                'collapse':bool(loads.max()>.8 or math.exp(-np.sum(loads*np.log(np.maximum(loads,1e-12))))<2)}
        if grads is not None:
            pg=grads['model']['layers'][li]['mlp']
            record['router_gradient_norm']=float(norm(pg['router']))
            norms=sum((mx.sum(w.astype(mx.float32)**2,axis=(1,2)) for w in pg['weights'].values()),mx.zeros((8,)))
            record['expert_gradient_norms']=np.sqrt(np.array(norms)).tolist()
        result.append(record)
    return result

def validate(model,data):
    losses=[];stats=[]
    model.eval()
    for start in range(0,len(data),4):
        batch=mx.array(data[start:start+4].astype(np.int32))
        values=loss(model,batch[:,:-1],batch[:,1:]);mx.eval(values)
        losses.append(float(values[1]));stats.append(health(values[2]))
    model.train()
    return float(np.mean(losses)),stats

def save(model,opt,tag,step):
    path=ROOT/'checkpoints'/tag
    path.mkdir(exist_ok=True)
    model.save_weights(str(path/f'step_{step}.safetensors'))
    mx.savez(str(path/f'optimizer_{step}.npz'),**dict(tree_flatten(opt.state)))
    (path/f'step_{step}.json').write_text(json.dumps({'step':step,'tokens_seen':step*1024,
                                                   'optimizer_steps':step}))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['dense','top1','top2'],required=True)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    freeze=ROOT/'FREEZE.json'
    if not args.smoke:
        for relative,digest in json.loads(freeze.read_text())['files'].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    tag=f'{args.kind}_s{args.seed}'+('_smoke' if args.smoke else '')
    train=np.load(ROOT/'data'/'train.npy');val=np.load(ROOT/'data'/'validation.npy')
    model,model_args,shared_hash=build(args.kind,args.seed)
    params=sum(p.size for _,p in tree_flatten(model.parameters()))
    moe_params=sum(p.size for name,p in tree_flatten(model.parameters()) if '.mlp.weights.' in name)
    router_params=sum(p.size for name,p in tree_flatten(model.parameters()) if '.mlp.router.' in name)
    active=params-moe_params+moe_params*(0 if args.kind=='dense' else 1 if args.kind=='top1' else 2)/8
    config={'kind':args.kind,'seed':args.seed,'steps':args.steps,'dtype':'float32',
            'model':model_args.__dict__,'total_parameters':params,
            'active_parameters_per_token_estimate':int(active),'moe_parameters':moe_params,
            'router_parameters':router_params,'experts':0 if args.kind=='dense' else 8,
            'experts_per_token':0 if args.kind=='dense' else 1 if args.kind=='top1' else 2,
            'moe_layers':[] if args.kind=='dense' else [3,7],
            'batch_size':4,'sequence_length':256,'tokens_per_step':1024,
            'shared_initialization_sha256':shared_hash,'aux_coefficient':.01,
            'optimizer':'AdamW','betas':[.9,.95],'weight_decay':.1,'lr':3e-4,
            'warmup_steps':60,'min_lr_ratio':.1,'compile':False,
            'timing':'eager synchronized forward/loss, backward, gradient clip/optimizer; stats excluded'}
    (ROOT/'configs'/f'{tag}.json').write_text(json.dumps(config,indent=2))
    opt=optim.AdamW(learning_rate=3e-4,betas=[.9,.95],weight_decay=.1)
    fn=nn.value_and_grad(model,lambda x,y:loss(model,x,y))
    start_time=time.perf_counter();step_wall=0.;peak=0
    with (ROOT/'metrics'/f'{tag}.jsonl').open('x') as file:
        initial,vhealth=validate(model,val if not args.smoke else val[:4])
        file.write(json.dumps({'event':'validation','step':0,'tokens_seen':0,'validation_loss':initial,
                               'validation_perplexity':math.exp(initial),'wall_seconds':time.perf_counter()-start_time})+'\n');file.flush()
        print(tag,'INITIAL',initial,flush=True)
        for step in range(1,args.steps+1):
            batch=mx.array(train[(step-1)*4:step*4].astype(np.int32));mx.eval(batch)
            lr=3e-4*(step/60 if step<=60 else .1+.9*.5*(1+math.cos(math.pi*(step-60)/(1200-60))))
            opt.learning_rate=lr
            mx.reset_peak_memory()
            t0=time.perf_counter();values,grads=fn(batch[:,:-1],batch[:,1:]);mx.eval(values);t1=time.perf_counter()
            mx.eval(grads);t2=time.perf_counter()
            gnorm=norm(grads);grads=tree_map(lambda g:g*mx.minimum(1.,1./(gnorm+1e-6)),grads)
            opt.update(model,grads);mx.eval(model.parameters(),opt.state);t3=time.perf_counter()
            times={'forward_ms':(t1-t0)*1000,'backward_ms':(t2-t1)*1000,
                   'optimizer_ms':(t3-t2)*1000,'total_ms':(t3-t0)*1000}
            step_wall+=t3-t0;peak=max(peak,mx.get_peak_memory())
            row={'event':'train','step':step,'tokens_seen':step*1024,'train_loss':float(values[1]),
                 'objective':float(values[0]),'lr':lr,'gradient_norm':float(gnorm),
                 'tokens_per_second':1024/(t3-t0),'peak_mlx_active_bytes':mx.get_peak_memory(),
                 'active_bytes':mx.get_active_memory(),'wall_seconds':time.perf_counter()-start_time,
                 'training_phase_wall_seconds':step_wall,'warmup':step<=60,**times}
            if step%25==0 or step==1: row['router_health']=health(values[2],grads)
            file.write(json.dumps(row)+'\n');file.flush()
            del values,grads,batch,gnorm
            if step%200==0 or step==args.steps:
                vl,vhealth=validate(model,val if not args.smoke else val[:4])
                file.write(json.dumps({'event':'validation','step':step,'tokens_seen':step*1024,
                                       'validation_loss':vl,'validation_perplexity':math.exp(vl),
                                       'router_health_batches':vhealth,'wall_seconds':time.perf_counter()-start_time})+'\n');file.flush()
                print(tag,'STEP',step,'VAL',vl,'TOK/S',row['tokens_per_second'],flush=True)
            if step in (200,args.steps) and not args.smoke: save(model,opt,tag,step)
        config['peak_mlx_active_bytes']=peak
        config['elapsed_seconds']=time.perf_counter()-start_time
        (ROOT/'metrics'/f'{tag}_summary.json').write_text(json.dumps(config,indent=2))

if __name__=='__main__':main()
