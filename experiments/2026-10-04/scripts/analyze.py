"""Rebuild all plots and the morning report from saved measurements."""
import csv,json,math,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
COLORS={'dense':'#2563eb','top1':'#d97706','top2':'#059669'}

def read(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def median(rows,key):return float(np.median([r[key] for r in rows]))
def ms(row,key):return row['summary'][key]['median']
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    assert (ROOT/'metrics/SUITE_COMPLETE.json').exists(),'Suite has not finished.'
    matrix=json.loads((ROOT/'configs/run_matrix.json').read_text())
    runs=[];summaries=[];router_history=[]
    for config in matrix:
        tag=f"{config['kind']}_s{config['seed']}"
        rows=read(ROOT/'metrics'/f'{tag}.jsonl')
        train=[r for r in rows if r['event']=='train'];val=[r for r in rows if r['event']=='validation']
        assert len(train)==1200 and val[-1]['step']==1200,tag
        assert all(math.isfinite(r['train_loss']) for r in train),tag
        conf=json.loads((ROOT/'configs'/f'{tag}.json').read_text())
        steady=[r for r in train if r['step']>60]
        for r in train:
            for h in r.get('router_health',[]):
                assignments=1024*(1 if config['kind']=='top1' else 2)
                counts=[int(round(v*assignments)) for v in h['utilization']]
                assert sum(counts)==assignments
                router_history.append({'config':config['kind'],'seed':config['seed'],
                                       'step':r['step'],'tokens_seen':r['tokens_seen'],
                                       'expert_token_assignment_counts':counts,
                                       'maximum_expert_count':max(counts),'minimum_expert_count':min(counts),**h})
        last_health=val[-1].get('router_health_batches',[])
        health=[h for batch in last_health for h in batch]
        loads=np.mean([h['utilization'] for h in health],axis=0) if health else np.zeros(8)
        summary={'config':config['kind'],'seed':config['seed'],'total_params':conf['total_parameters'],
                 'active_params':conf['active_parameters_per_token_estimate'],'tokens_seen':1200*1024,
                 'final_train_loss':train[-1]['train_loss'],'final_validation_loss':val[-1]['validation_loss'],
                 'final_validation_perplexity':val[-1]['validation_perplexity'],
                 'steady_tokens_per_second':1024*len(steady)/sum(r['total_ms']/1000 for r in steady),
                 'forward_ms':median(steady,'forward_ms'),'backward_ms':median(steady,'backward_ms'),
                 'optimizer_ms':median(steady,'optimizer_ms'),'step_ms':median(steady,'total_ms'),
                 'step_std_ms':float(np.std([r['total_ms'] for r in steady])),
                 'peak_mlx_active_gib':max(r['peak_mlx_active_bytes'] for r in train)/1024**3,
                 'wall_seconds_at_final_validation':val[-1]['wall_seconds'],
                 'first_step_ms':train[0]['total_ms'],
                 'final_router_probability_entropy':float(np.mean([h['probability_entropy'] for h in health])) if health else None,
                 'final_mean_expert_loads':loads.tolist(),
                 'maximum_final_batch_expert_load':max([h['max_load'] for h in health],default=0),
                 'training_collapse_samples':sum(h['collapse'] for r in train for h in r.get('router_health',[])),
                 'recorded_router_samples':sum(len(r.get('router_health',[])) for r in train),
                 'shared_initialization_sha256':conf['shared_initialization_sha256']}
        summaries.append(summary);runs.append((config,train,val))
    for seed in (42,43,44):
        assert len({r['shared_initialization_sha256'] for r in summaries if r['seed']==seed})==1
    (ROOT/'metrics/training_summary.json').write_text(json.dumps(summaries,indent=2))
    (ROOT/'metrics/router_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in router_history))
    with (ROOT/'metrics/training_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(14,4.3))
    for kind in COLORS:
        subset=[r for c,r,v in runs if c['kind']==kind]
        loss=np.mean([[q['train_loss'] for q in rows] for rows in subset],axis=0)
        smoothed=np.convolve(loss,np.ones(50)/50,mode='valid')
        ax[0].plot(np.arange(50,1201)*1024/1e6,smoothed,label=kind,color=COLORS[kind])
        vals=[v for c,t,v in runs if c['kind']==kind]
        vv=np.array([[q['validation_loss'] for q in rows] for rows in vals]);xx=np.array([q['tokens_seen'] for q in vals[0]])/1e6
        ax[1].plot(xx,vv.mean(axis=0),marker='o',label=kind,color=COLORS[kind]);ax[1].fill_between(xx,vv.mean(axis=0)-vv.std(axis=0),vv.mean(axis=0)+vv.std(axis=0),alpha=.15,color=COLORS[kind])
        for i,(c,t,v) in enumerate([r for r in runs if r[0]['kind']==kind]):
            ax[2].plot([q['wall_seconds'] for q in v],[q['validation_loss'] for q in v],color=COLORS[kind],alpha=.55,label=kind if i==0 else None)
    for a in ax:a.grid(alpha=.2);a.legend()
    ax[0].set(title='Training loss, mean of 3 seeds',xlabel='Million target positions',ylabel='Cross-entropy, 50-step average')
    ax[1].set(title='Held-out loss at equal tokens',xlabel='Million target positions',ylabel='Cross-entropy; shading = seed SD')
    ax[2].set(title='Held-out loss vs actual run time',xlabel='Wall seconds incl. validation/checkpoint IO',ylabel='Cross-entropy')
    fig.tight_layout();fig.savefig(ROOT/'plots/training_curves.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for kind in ('top1','top2'):
        for c,t,v in runs:
            if c['kind']!=kind:continue
            entries=[(r['tokens_seen'],h) for r in t for h in r.get('router_health',[])]
            ax[0].plot([x/1e6 for x,h in entries],[h['probability_entropy'] for x,h in entries],'.',alpha=.25,color=COLORS[kind])
        loads=np.mean([r['final_mean_expert_loads'] for r in summaries if r['config']==kind],axis=0)
        ax[1].bar(np.arange(8)+(-.18 if kind=='top1' else .18),loads,width=.36,label=kind,color=COLORS[kind])
    ax[0].axhline(math.log(8),ls='--',color='gray');ax[0].set(xlabel='Million target positions',ylabel='Router probability entropy, nats',title='Router entropy across layers and seeds')
    ax[1].axhline(.125,ls='--',color='gray');ax[1].set(xlabel='Expert ID',ylabel='Assignment fraction',title='Final held-out utilization, mean of layers/seeds');ax[1].legend()
    fig.tight_layout();fig.savefig(ROOT/'plots/router_health.png');plt.close(fig)
    primary=read(ROOT/'metrics/benchmark_primary_v2.jsonl');sweep=read(ROOT/'metrics/benchmark_sweep.jsonl')
    assert all('error' not in r for r in primary+sweep)
    benchmark_summary=[]
    for row in primary+sweep:
        dense=row['layout']=='dense';total=3*row['d']*row['h']*(1 if dense else row['experts'])
        summary={'suite':'primary' if row in primary else 'sweep','layout':row['layout'],
                 'd':row['d'],'h':row['h'],'dtype':row['dtype'],'total_parameters':total,
                 'active_parameters_per_token':3*row['d']*row['h']*(1 if dense else row['top_k']),
                 'moe_parameters':0 if dense else total,'experts':0 if dense else row['experts'],
                 'top_k':0 if dense else row['top_k'],'tokens':row['tokens'],'pattern':row['pattern'],
                 'touched_experts':len(row['touched_experts']) if not dense else 1,
                 'first_forward_ms':row['first_call']['forward_ms'],
                 'first_backward_ms':row['first_call']['backward_ms'],
                 'logical_weight_gradient_bytes':row['gradient_audit']['weight_logical_bytes']}
        for key in ('forward_ms','backward_ms','total_ms','peak_delta_bytes','gradient_only_retained_bytes'):
            for stat in ('median','mean','std'):summary[f'{key}_{stat}']=row['summary'][key][stat]
        benchmark_summary.append(summary)
    with (ROOT/'metrics/benchmark_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(benchmark_summary[0]));writer.writeheader();writer.writerows(benchmark_summary)
    one=[r for r in primary if r['pattern']=='one']
    fig,ax=plt.subplots(1,2,figsize=(12,4.7));labels=[r['layout'] for r in one]
    ax[0].barh(labels,[ms(r,'forward_ms') for r in one],label='forward + VJP graph construction',color='#2563eb')
    ax[0].barh(labels,[ms(r,'backward_ms') for r in one],left=[ms(r,'forward_ms') for r in one],label='backward execution',color='#f59e0b');ax[0].set(xlabel='Median ms',title='One selected expert; d256 h864, 512 tokens');ax[0].legend(fontsize=8)
    ax[1].barh(labels,[ms(r,'gradient_only_retained_bytes')/1024**2 for r in one],color='#7c3aed');ax[1].set(xlabel='Retained MiB, all gradients including input',title='Physical allocation after releasing loss graph')
    fig.tight_layout();fig.savefig(ROOT/'plots/sparse_backward.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for a,d,title in zip(ax,(256,1024),('Float32, d256 h864','Bfloat16, H1 d1024 h3456')):
        for e in (4,8,16):
            points=[next(r for r in sweep if r['d']==d and r['experts']==e and r['tokens']==n and r['layout']=='gather_sorted' and r['top_k']==1 and r['pattern']=='balanced') for n in (32,512,2048)]
            dense=[next(r for r in sweep if r['d']==d and r['experts']==e and r['tokens']==n and r['layout']=='dense' and r['top_k']==1 and r['pattern']=='balanced') for n in (32,512,2048)]
            a.plot((32,512,2048),[ms(p,'total_ms')/ms(q,'total_ms') for p,q in zip(points,dense)],marker='o',label=f'{e} experts')
        a.axhline(1,color='gray',ls='--');a.set_xscale('log',base=2);a.set(title=title,xlabel='Tokens in FFN batch',ylabel='Sorted sparse / single dense step time');a.legend();a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'plots/expert_sweep.png');plt.close(fig)
    compiled=json.loads((ROOT/'metrics/compile_router.json').read_text());verify=json.loads((ROOT/'metrics/verification.json').read_text())
    custom_path=ROOT/'metrics/custom_vjp_swiglu.json'
    custom_results=json.loads(custom_path.read_text()) if custom_path.exists() else []
    custom_table=table(['VJP','Physical retained grad MiB','Logical grad MiB','Grad objects','AdamW state MiB'],
        [[('Custom broadcast-zero' if r['custom'] else 'Ordinary'),f"{r['median']['retained_gradient_bytes']/1024**2:.6f}",
          f"{r['audit']['logical_gradient_bytes']/1024**2:.3f}",r['audit']['gradient_objects'],
          f"{r['optimizer_state_logical_bytes']/1024**2:.3f}"] for r in custom_results])
    groups={k:[r for r in summaries if r['config']==k] for k in COLORS}
    means={k:float(np.mean([r['final_validation_loss'] for r in groups[k]])) for k in groups}
    best=min(means,key=means.get)
    common_seconds=min(r['wall_seconds_at_final_validation'] for r in summaries)
    wall_means={kind:float(np.mean([np.interp(common_seconds,[v['wall_seconds'] for v in vals],[v['validation_loss'] for v in vals]) for c,t,vals in runs if c['kind']==kind])) for kind in COLORS}
    best_wall=min(wall_means,key=wall_means.get)
    train_table=[];timing_table=[]
    for kind,rows in groups.items():
        train_table.append([kind,f"{rows[0]['total_params']/1e6:.3f}M",f"{rows[0]['active_params']/1e6:.3f}M",f"{np.mean([r['steady_tokens_per_second'] for r in rows]):,.0f}",f"{max(r['peak_mlx_active_gib'] for r in rows):.3f} GiB",f"{means[kind]:.6f} ± {np.std([r['final_validation_loss'] for r in rows],ddof=1):.6f}",'n/a' if kind=='dense' else f"{sum(r['training_collapse_samples'] for r in rows)}/{sum(r['recorded_router_samples'] for r in rows)} collapse samples"])
        timing_table.append([kind]+[f"{np.mean([r[key] for r in rows]):.2f}" for key in ('forward_ms','backward_ms','optimizer_ms','step_ms','wall_seconds_at_final_validation')])
    micro_table=[[r['layout'],f"{ms(r,'forward_ms'):.3f}",f"{ms(r,'backward_ms'):.3f}",f"{ms(r,'total_ms'):.3f}",str(r['gradient_audit']['objects']),f"{r['gradient_audit']['weight_logical_bytes']/1024**2:.3f}",f"{ms(r,'gradient_only_retained_bytes')/1024**2:.3f}"] for r in one]
    h1=[r for r in sweep if r['d']==1024 and r['experts']==8 and r['tokens']==512 and r['pattern']=='balanced' and r['top_k']==1][-5:]
    h1table=[[r['layout'],f"{ms(r,'forward_ms'):.2f}",f"{ms(r,'backward_ms'):.2f}",f"{ms(r,'total_ms'):.2f}",f"{ms(r,'peak_delta_bytes')/1024**2:.1f}"] for r in h1]
    sweep_table=[]
    for e in (4,8,16):
        sweep_table.append([str(e)]+[f"{ms(next(r for r in sweep if r['d']==1024 and r['experts']==e and r['tokens']==n and r['layout']=='gather_sorted' and r['top_k']==1 and r['pattern']=='balanced'),'total_ms'):.2f}" for n in (32,512,2048)])
    paired={k:[next(r for r in groups[k] if r['seed']==s)['final_validation_loss']-next(r for r in groups['dense'] if r['seed']==s)['final_validation_loss'] for s in (42,43,44)] for k in ('top1','top2')}
    touched={kind:[sum(c>0 for c in r['expert_token_assignment_counts']) for r in router_history if r['config']==kind] for kind in ('top1','top2')}
    all_touched={kind:100*np.mean(np.array(values)==8) for kind,values in touched.items()}
    collapsed=[r for r in router_history if r['collapse']]
    (ROOT/'metrics/comparison.json').write_text(json.dumps({'mean_final_validation':means,'paired_moe_minus_dense':paired,'equal_wall_time_seconds':common_seconds,'equal_wall_loss_linear_interpolation':wall_means,'best_per_token_point_estimate':best,'best_equal_wall_point_estimate':best_wall},indent=2))
    train_data=np.load(ROOT/'data/train.npy');val_data=np.load(ROOT/'data/validation.npy')
    actual_train=int(np.count_nonzero(train_data[:,1:]));actual_val=int(np.count_nonzero(val_data[:,1:]))
    report=f'''# MoE Night Results

## Executive answer

**D. RESEARCH — sparse arithmetic works, but ordinary MLX autodiff and AdamW keep dense expert gradients/state. Do not scale this into the main model yet.**

1. **Forward compute:** yes, relative to executing all experts. At H1 FFN width, 512 tokens, eight experts, sorted Top-1 is about 4 ms versus 35 ms for dense execution of all eight. It is slightly slower than one ordinary dense FFN at comparable active compute.
2. **Backward compute:** selected expert GEMM arithmetic is sparse with independent dispatch or sorted gather_mm. Full-shaped gradient writes remain. Stacked Python slices introduce large extra backward allocation/accumulation costs.
3. **Memory:** activation-related intermediates decrease relative to all-expert execution; ordinary parameter gradients do **not** become sparse. Direct nn.Module measurements retain 20.25 MiB of expert gradients versus 2.531 MiB when only one expert subtree is differentiated. An exploratory custom VJP also reaches 2.531 MiB by returning broadcast zeros for unused independent weights. AdamW state remains full-sized. MLX allocator peaks below are not whole-machine unified-memory peaks.
4. **Layout:** independent modules improve naive dispatch but do not automatically omit unused gradients. The differentiated parameter tree matters. Sorted stacked dispatch is the best general path measured here.
5. **Best per token:** `{best}` has the lowest three-seed mean held-out loss ({means[best]:.6f}), but the differences are small and the pilot is severely undertrained. No convincing convergence winner.
6. **Best per wall time:** `{best_wall}` at the common {common_seconds:.1f}-second endpoint, using linear interpolation between saved validation evaluations. This is an exploratory comparison, not a measured wall-clock-hour result. No hour-long quality comparison was run.
7. **Scale recommendation:** no larger training launch. First investigate expert-local gradients/state and run a longer bounded learning experiment.

{table(['Config','Total Params','Active Params/token est.','Steady Tok/s','Peak MLX active','Final Val Loss (mean ± seed SD)','Router Health'],train_table)}

All nine runs completed: 1,200 updates × 1,024 scheduled target positions = **1,228,800 per run**, three seeds (42–44), **11,059,200 total training positions**. Actual non-PAD loss targets: {actual_train:,} per run; validation {actual_val:,}. The models contain 8 layers, d256/h864, GQA 8/2, full 32,768-token vocabulary, tied embeddings, sequence256, batch4. Two MoE layers replace FFNs at zero-based layers3 and7. This is an implementation/learning pilot, not a capability evaluation.

## Most surprising finding

Independent expert modules alone do not solve gradient storage. A real nn.value_and_grad call differentiating an eight-expert module returned **24 full gradient tensors**, with seven experts' norms exactly zero, and allocator storage matched their full logical size. Differentiating just its selected expert returned **three** tensors and reduced gradient storage eightfold. AdamW additionally changed an unused expert through weight decay; skipping those updates would alter optimizer semantics.

**Exploratory custom-VJP result:** a complete SwiGLU with fixed routing to expert0 can return all24 logically full-shaped gradient arrays while storing unused zeros as broadcast views of one scalar. Physical retained gradient bytes fell from21,233,664 to2,654,212, and every gradient matched ordinary autodiff. Generic global gradient clipping preserved this compression in the probe. This is not CSR sparsity, a dynamic-router training implementation, or an optimizer-state saving.

{custom_table}

This post-pilot probe is clearly separate from the frozen A/B/C fits. Its script and15-repeat raw windows are saved in `scripts/custom_vjp_probe.py` and `metrics/custom_vjp_swiglu.json`. AdamW still allocates40.5MiB of moments in either path. Custom VJP is therefore a concrete next implementation route, not merely a hypothetical API suggestion.

## Sparse backward investigation

Controlled float32 audit: E8, d256/h864, 512 tokens, all routed to expert0, no router. Three warmups after the first call and 15 measured repeats. Timings include host graph construction in forward; backward is incremental evaluation after loss evaluation on the same VJP graph. Input gradients are included. Physical retained storage releases the loss graph before measurement; all gradient arrays stay alive.

{table(['Layout','Fwd ms','Bwd ms','Total ms','Weight grad objects','Logical weight grad MiB','Physical grads incl. input MiB'],micro_table)}

Stacked shapes: gate/up `[8,864,256]`, down `[8,256,864]`; independent weights: 24 arrays of `[864,256]` or `[256,864]`. Unselected expert gradients are exactly zero in the forced-routing audit. Selected-subtree results exclude unselected experts from the differentiation inputs. Balanced routing touches every expert across a batch, so even exact expert-local accumulation requires all experts' gradients in that case. Per-token sparsity is not per-batch sparsity.

The installed-version [gather_mm VJP source](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/primitives.cpp#L113) uses segmented matrix multiplication for sorted expert IDs, but returns the full expert batch shape. Its [Metal allocator](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/matmul.cpp#L3037) allocates out.nbytes. Empty segments skip expert multiply iterations while still writing zero output tiles. Unsorted dispatch instead forms per-assignment gradient products before scatter-add; the 512-token float32 microbenchmark reached hundreds of MiB of intermediates. Source, unevaluated DOT graphs, exact norms, physical allocator deltas, and numerical equivalence support the conclusion together; timing alone is insufficient.

Equivalence tests compared forward, all parameter gradients, and input gradients against explicit expert loops for Top-1/Top-2 and imbalanced/balanced routing. Maximum float32 discrepancy was {max(r.get('max_difference',0) for r in verify['equivalence']):.2g}; central finite differences agreed. Top-1 and Top-2 router task gradients are nonzero after detaching discrete indices. See `metrics/verification.json`.

![Sparse backward and gradient retention](plots/sparse_backward.png)

## Training curves

![Training curves](plots/training_curves.png)

Initialization is shared tensor-for-tensor within each seed (hashes verified); every MoE expert starts as a clone of its corresponding dense FFN. Router weights are independently initialized. Both Top-K modes use selected probabilities from the **full softmax**, without selected-only renormalization. Thus initial residual magnitudes differ from dense and from normalized Mixtral-style gating. This is an active-compute comparison with larger MoE capacity, not equal-total-parameter fairness. Cloned experts can delay specialization.

All runs use AdamW (.9,.95), weight decay.1, gradient clipping1, LR3e-4, 60-step warmup, cosine to 0.1×, float32. Run order is counterbalanced across seeds. Data and source were hashed and frozen before fits. Training samples 300 distinct blocks from each of 16 evenly spaced original v7 packed shards; validation is 128 eligible held-out documents from a deterministic uniform candidate sample. Validation contains 107 FineWeb-Edu and21 FineMath documents; it does not assess every training domain. This old v7 corpus is not the current Fedora v2 mixture. Original split is preserved; exact window overlap is zero, semantic deduplication was not redone.

Paired final MoE-minus-dense loss deltas by seed42/43/44: Top-1 {', '.join(f'{v:+.6f}' for v in paired['top1'])}; Top-2 {', '.join(f'{v:+.6f}' for v in paired['top2'])}. Treat these tiny differences skeptically: three seeds, 32K held-out targets, and only ~0.08 tokens per active parameter. No test of assistant capability or expert specialization was performed. An hour of learning quality cannot be inferred from these short runs.

{table(['Config','Forward ms','Backward ms','Clip/optimizer ms','Step ms','Final validation wall sec'],timing_table)}

Timings are means of per-run steady medians after update60. Throughput divides total scheduled steady tokens by summed phase time. Actual wall curves include validation and checkpoint IO. Synchronization, eager graph construction, monitoring and a full vocabulary head limit extrapolation to compiled production throughput. First calls and repeated timing windows are saved separately; training warmup includes the LR warmup, and should not be read as compilation time.

## Router behavior

![Router utilization](plots/router_health.png)

Router records include probability entropy, assignment entropy, every expert's utilization, coefficient of variation, max/min load, auxiliary loss, router/expert gradient norms, collapse flags and zero dropped/overflow tokens. Assignment fractions count Top-2 assignments and sum to1. Counts are fractions times batch tokens × k; all assignments are executed without capacity truncation. Gradient norms are after global clipping. Load balancing coefficient is.01, with detached assignment fractions. Healthy utilization does not demonstrate semantic specialization. Per-layer, per-seed, per-batch records are retained, rather than only the aggregate bar plot.

Top-1 triggered **{len(collapsed)}/294** sampled layer/batch collapse flags. The worst assigned **{100*max(r['max_load'] for r in collapsed):.1f}%** to one expert; the last recorded flag occurred at update{max(r['step'] for r in collapsed)}. At final held-out evaluation, the largest individual expert batch share was {100*max(r['maximum_final_batch_expert_load'] for r in groups['top1']):.1f}%. Top-2 had no collapse flags in these samples. The flags show intermittent concentration, not demonstrated permanent global collapse; aggregation can hide it, and sparse sampling cannot exclude events between logs.

All eight experts were touched in **{all_touched['top1']:.1f}% of Top-1** and **{all_touched['top2']:.1f}% of Top-2** sampled layer/batches. Median touched count was8 for both. The custom-VJP eightfold storage reduction applies to the forced-one-expert case; it cannot eliminate gradients for experts that actually received other tokens in the same batch. Exact assignment counts are saved in `metrics/router_history.jsonl`.

## Hardware bottlenecks

H1-width FFN (d1024/h3456), bfloat16, E8, 512 tokens, balanced Top-1, steady synchronized VJP timing:

{table(['Execution','Forward ms','Backward ms','Total ms','Incremental peak MiB'],h1table)}

Sorted dispatch routes and gathers token rows on GPU, groups selected matrices, uses segmented backward, and restores token order. Routing itself costs about0.18–0.23ms at d256 with512–2048 tokens in the isolated measurements. Naive Python grouped dispatch has multiple launch/graph overheads; slicing stacked weights additionally incurs full-shaped gradient construction and accumulation. The unsorted gather path is not a suitable training baseline. No unsorted H1-width large-token test was attempted because its predicted intermediates were needlessly large.

Sorted H1-width Top-1 step milliseconds versus expert count and total token batch:

{table(['Experts','32 tokens','512 tokens','2048 tokens'],sweep_table)}

At32 tokens, the ordinary dense FFN is ~1.5ms and expert fragmentation/state writes dominate. At512 and2048 tokens sorted Top-1 is close to the single-dense active-compute baseline, but never a clear speed win in this sweep. There is no universal width/expert crossover established by three widths/count windows. Top-2 increases actual routed tokens and transient work; use its learning benefit, not capacity alone, to justify it.

The compiled sorted-dispatch probe changed routing IDs on every call and traced **{compiled['compile']['trace_count']}** time(s) at fixed shapes: first call {compiled['compile']['first_ms']:.2f}ms, steady median {compiled['compile']['median_ms']:.2f}ms. Static-shaped GPU indices compile efficiently here. Python host routing with variable token groups was not compiled; no general claim about arbitrary dynamic expert modules follows.

![Expert-count/token sweep](plots/expert_sweep.png)

## Failed experiments

- Initial Top-1 training smoke refused VJP through integer gather indices. Fixed before fits by stop_gradient on the discrete Top-K selection; gate probabilities still differentiate. Failure log and partial smoke metrics preserved.
- Initial memory audit retained the loss's forward graph, inflating the gradient delta; v2 releases it and adds a gradient-only measure. Original v1 output is retained, not used for the storage conclusion.
- Original DOT export occurred after evaluation and showed only leaves; repaired to export unevaluated graphs outside timed windows.
- All92 sweep cases and14 corrected primary cases completed without OOM. No queued fit crashed or timed out. Optional Default MoE and ProbMoE were researched, not run; faithful ports need separate tests, especially EMA semantics and nested marginal derivatives. Plans and primary links: `RESEARCH_NOTES.md`.

## Recommended next experiment

Keep the next model at **8 layers, width256, FFN864, eight experts every fourth FFN, Top-1**, total24.306M / active15.016M parameters. Observed MLX active peak is about1.18GiB; allow roughly3–4GiB process/cache headroom and retain the16GiB guard. Expected whole-machine memory includes macOS and other jobs, which this allocator metric excludes.

Generalize the measured broadcast-zero custom VJP from fixed expert0 to changing touched expert sets, preserve input/router VJPs, and verify multi-step AdamW equivalence. Compare it with expert-local differentiation/state on controlled touched sets, then train the same architecture for **10–30M fresh tokens per configuration**, at least three seeds, with a domain-stratified held-out set and equal residual/gating initialization controls. Include sorted stacked versus independent selected-subtree paths when only a subset is touched. Record how frequently all experts are touched at realistic batch sizes; it limits memory savings. Do not launch the20-layer MoE or replace the current H1 production model based on this pilot.

Custom VJPs can customize arithmetic and, with broadcast-zero views for independent weights, physically compress unused gradients without changing logical shapes. No general sparse gradient data type was implemented. Selected-subtree VJP is another measured proof of concept. Expert-ID/local-gradient buffers plus a dedicated optimizer look feasible; exact AdamW equivalence requires handling momentum decay and weight decay on untouched experts. Low-rank dW factors are possible for small token groups, but elementwise Adam state breaks a simple exact low-rank update. Details and limits are in `RESEARCH_NOTES.md`.

## Exact commands to reproduce

Run from `/RESEARCH_ROOT`. Dependencies are pinned in `configs/requirements.lock`. Existing outputs are protected by exclusive creation. Use the reproduction script to create a fresh output directory with the exact frozen source/data:

```sh
.venv/bin/python research/moe-night/results/2026-10-04/scripts/reproduce.py
```

The reproduction script prints its new directory and executes verification, primary/sweep benchmarks, the nine bounded fits, and this analysis. Dataset/weights remain on SHUTTLE; its bounded data slice is reused after hash verification. Logs, JSONL, configs, CSV summaries, graphs, plot source and checkpoint links remain under the new result directory. To rebuild only current plots/report:

```sh
.venv/bin/python research/moe-night/results/2026-10-04/scripts/analyze.py
```

macOS27.0.1 (26A434); Apple M4 Max128GiB; MLX0.32.2; Python3.11.17. Original BoothGPT commit3bdd2fc28761ad1fa4b8de1dcf03d888c6719165; isolated experiment source commit2f4fe22. Original research/checkpoints/jobs were not modified. Raw environment, initial process snapshot, source status, freeze hashes, source references, and per-run memory preflights are saved. Memory-pressure readings stayed healthy at each run boundary; no existing processes were killed.

**D. RESEARCH — investigate sparse gradient/state interfaces before scaling.**
'''
    replacements={'expert0':'expert 0','all24':'all 24','from21,233,664':'from 21,233,664',
        'to2,654,212':'to 2,654,212','and15-repeat':'and 15-repeat','allocates40.5MiB':'allocates 40.5 MiB',
        'sequence256':'sequence 256','batch4':'batch 4','layers3 and7':'layers 3 and 7',
        'weight decay.1':'weight decay 0.1','gradient clipping1':'gradient clipping 1','LR3e-4':'LR 3e-4',
        'and21 FineMath':'and 21 FineMath','seed42/43/44':'seeds 42/43/44','update60':'update 60',
        'sum to1':'sum to 1','coefficient is.01':'coefficient is 0.01','at d256 with512':'at d256 with 512',
        'At32 tokens':'At 32 tokens','At512 and2048':'At 512 and 2048','All92':'All 92','and14':'and 14',
        'layer MoE':'layer MoE','total24.306M':'total 24.306M','active15.016M':'active 15.016M',
        'about1.18GiB':'about 1.18 GiB','roughly3–4GiB':'roughly 3–4 GiB','the16GiB':'the 16 GiB',
        'the20-layer':'the 20-layer','model based':'model based','macOS27.0.1':'macOS 27.0.1',
        'Max128GiB':'Max 128 GiB','MLX0.32.2':'MLX 0.32.2','Python3.11.17':'Python 3.11.17',
        'commit3bdd':'commit 3bdd','commit2f4':'commit 2f4','count was8':'count was 8','update475':'update 475'}
    for old,new in replacements.items():report=report.replace(old,new)
    (ROOT/'REPORT.md').write_text(report)
    print(json.dumps({'mean_validation':means,'paired':paired,'equal_wall':wall_means,'training_runs':len(summaries)},indent=2))

if __name__=='__main__':main()
