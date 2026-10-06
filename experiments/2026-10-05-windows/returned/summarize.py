"""CPU-only report assembly from preserved measurements; no GPU execution."""

import hashlib,json,statistics,subprocess

from pathlib import Path



ROOT=Path(__file__).resolve().parent

v2=json.loads((ROOT/'measurements/v2/results.json').read_text())

v3=json.loads((ROOT/'measurements/v3/results.json').read_text())

v4=json.loads((ROOT/'measurements/v4/results.json').read_text())

prof=v4['optimizer_profile']; hardware=json.loads((ROOT/'diagnostics/hardware-current.json').read_text())

groups={k:[r for r in v2['benchmarks'] if r['kind']==k] for k in ('dense4','dense10','rec_full','rec_tb2')}

summary={k:{'parameters':rows[0]['parameters'],'targets_s_median':statistics.median(r['targets_per_second'] for r in rows),'targets_s_min':min(r['targets_per_second'] for r in rows),'targets_s_max':max(r['targets_per_second'] for r in rows),'median_ms':{name:statistics.median(r['median_ms'][name] for r in rows) for name in ('forward_ms','backward_ms','optimizer_ms','whole_step_ms')},'peak_allocated_mib':max(r['peak_allocated_bytes']/1024**2 for r in rows),'peak_reserved_mib':max(r['peak_reserved_bytes']/1024**2 for r in rows),'device_memory_mib':max(r['max_device_memory_mib'] for r in rows),'max_temperature_c':max(r['max_sampled_temperature_c'] for r in rows)} for k,rows in groups.items()}

paired=[]

for seed in (42,43,44):

    rows={r['kind']:r for r in v2['benchmarks'] if r['seed']==seed}

    assert len({r['shared_initialization_sha256'] for r in rows.values()})==1

    assert len({r['batch_sha256'] for r in rows.values()})==1

    f,t=rows['rec_full'],rows['rec_tb2']

    paired.append({'seed':seed,'full_targets_s':f['targets_per_second'],'tb2_targets_s':t['targets_per_second'],'tb2_speed_gain_percent':100*(t['targets_per_second']/f['targets_per_second']-1),'tb2_backward_time_reduction_percent':100*(1-t['median_ms']['backward_ms']/f['median_ms']['backward_ms']),'shared_initialization_sha256':f['shared_initialization_sha256'],'batch_sha256':f['batch_sha256'],'parameters_equal':f['parameters']==t['parameters'],'optimizer_ownership_equal':f['ownership']==t['ownership']})

assert all(p['parameters_equal'] and p['optimizer_ownership_equal'] for p in paired)

usage=v4['cumulative_gpu_active_s']

profile_summary={'updates':5,'unprofiled_median_ms':prof['unprofiled_median_ms'],'profiled_median_ms':prof['profiled_median_ms'],'compute_kernel_total_ms':prof['cuda_kernel_total_us']/1000,'compute_kernel_mean_ms_per_update':prof['cuda_kernel_total_us']/5000,'memory_operations_total_ms':prof['cuda_memory_operation_total_us']/1000,'cuda_launch_api_calls':prof['cuda_launch_api_calls'],'cuda_launch_cpu_total_ms':prof['cuda_launch_api_cpu_total_us']/1000,'cuda_launch_cpu_mean_ms_per_update':prof['cuda_launch_api_cpu_total_us']/5000,'excluded_gpu_annotations':prof['excluded_annotation_count']}

derived={'status':'PASS_WITH_SUPPLEMENTAL_MONITOR_CADENCE_DEVIATION','benchmark_summary':summary,'paired_recurrence':paired,'median_paired_tb2_speed_gain_percent':statistics.median(r['tb2_speed_gain_percent'] for r in paired),'median_paired_backward_reduction_percent':statistics.median(r['tb2_backward_time_reduction_percent'] for r in paired),'cumulative_gpu_active_seconds':usage,'optimizer_profile':profile_summary,'full_geometry_verification':v3['full_geometry'],'cuda_checkpoint_verification':v3['cuda_checkpoint'],'numerical_backend':'EFFICIENT_ATTENTION; Flash unavailable','scope':'Synthetic hardware timing; no learning-quality or exact Mac/CUDA ratio claim'}

derived['monitor_cadence']={'benchmark_max_gap_s':v2['max_telemetry_sample_gap_s'],'corrected_profile_max_gap_s':v4['max_sample_gap_s'],'supplemental_v3_max_gap_s':v3['max_sample_gap_s'],'supplemental_v3_deviation':True}

(ROOT/'SUMMARY.json').write_text(json.dumps(derived,indent=2))

labels={'dense4':'Dense4','dense10':'Dense10','rec_full':'Full recurrence','rec_tb2':'TB2 recurrence'}

rows=[]

for kind,r in summary.items():

    ms=r['median_ms']; rows.append(f"| {labels[kind]} | {r['parameters']:,} | {r['targets_s_median']/1000:.2f} | {ms['forward_ms']:.2f} | {ms['backward_ms']:.2f} | {ms['optimizer_ms']:.2f} | {ms['whole_step_ms']:.2f} | {r['peak_allocated_mib']:.1f} / {r['peak_reserved_mib']:.0f} | {r['device_memory_mib']:.0f} | {r['max_temperature_c']:.0f} |")

pair_rows='\n'.join(f"| {r['seed']} | {r['full_targets_s']/1000:.2f} | {r['tb2_targets_s']/1000:.2f} | +{r['tb2_speed_gain_percent']:.2f}% | {r['tb2_backward_time_reduction_percent']:.2f}% |" for r in paired)

profile_ops=[r for r in prof['top_operators_by_device_time'] if r['name'] in ('aten::mul','aten::add','aten::div','aten::copy_','aten::baddbmm','aten::bmm')]

operator_rows='\n'.join(f"| {r['name']} | {r['count']} | {r['self_device_us']/1000:.3f} | {r['self_cpu_us']/1000:.3f} |" for r in profile_ops)

gap=max(v2['max_telemetry_sample_gap_s'],v3['max_sample_gap_s'],v4['max_sample_gap_s'])

total_targets=sum(r['targets'] for r in v2['benchmarks'])

cpu=json.loads((ROOT/'measurements/v2/correctness-cpu.json').read_text())

diagnostics=json.loads((ROOT/'diagnostics/sleep-event-summary.json').read_text())

history=subprocess.check_output(['git','-C',str(ROOT),'log','--format=%h %s'],text=True).strip()

text=f'''# Connected RTX 4070 Laptop CUDA campaign — 2026-10-05



**PASS for the twelve registered timing windows; one supplemental monitor-cadence deviation is disclosed below.** All twelve 10-second

measurement windows completed at the exact batch8/context256 geometry after

CPU/CUDA correctness gates. TB2 delivered a median paired throughput gain of

**{derived['median_paired_tb2_speed_gain_percent']:.2f}%** and a median paired

backward-time reduction of **{derived['median_paired_backward_reduction_percent']:.2f}%**.

This measures eager native-Windows CUDA cost, not learning quality.



The entire campaign, including the stopped Flash attempt, correctness, warmup,

validation and both profiling attempts, used **{usage:.2f} / 600 seconds** of

conservatively counted GPU-active phase time. Maximum sampled temperature was

**74°C**, below the 85°C stop. No long training, lid reproduction, WSL install,

firmware/driver change or power-policy change occurred. Research processes exited.



## Measurements



Each row reports medians across seeds42/43/44, one newly initialized short window

per arm/seed, three warmup steps. Same ten-block-bank common-weight SHA and same

synthetic batch SHA were confirmed for all four arms within each seed. Parameter

counts match the source. BF16 weights/gradients and FP32 masters/state were used.



| Arm | Unique parameters | k targets/s | Forward ms | Backward ms | Hybrid ms | Whole step ms | Peak allocated/reserved MiB | Device MiB | Max °C |

| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

{chr(10).join(rows)}



The timings are synchronized host elapsed times. Forward includes objective/FP32

cross entropy; backward is separate; hybrid includes global clipping, reference

Muon/Adam updates and BF16 weight copies. Whole-step includes zero_grad and scalar

finite/monitor checks. Window throughput includes step-log writes and loop

overhead, so it is not simply2048 divided by the median component step. Empty

CUDA synchronization median host-call cost was {v2['empty_cuda_synchronize_median_host_us']:.2f}µs,

not subtracted. Component medians need not add exactly. No compilation/data-loader

work is included; tensors were preallocated on GPU. All individual steps, actual

window durations, warmups, clipping flags, losses and telemetry are retained.

Measured-window target accounting totals {total_targets:,}; warmup/verification

targets are separate, and no real corpus was used.



| Paired seed | Full k targets/s | TB2 k targets/s | TB2 throughput gain | Backward reduction |

| --- | ---: | ---: | ---: | ---: |

{pair_rows}



These three initialization/window seeds are small hardware repeats, not three

independent learning fits. Full/TB2 order alternated F/T,T/F,F/T; Dense order also

alternated. The repeated fixed batch overfits quickly; its synthetic losses and

clipping rates must not be interpreted as generalization or the proposed clip1/5

learning ablation. TB2 still has all unique parameter gradients and the same

optimizer ownership/state sizes as full recurrence. Its smaller activation peak

and lower backward cost do not imply sparse parameter gradients.



## Correctness and optimizer fidelity



CPU-double gates passed: full/TB2 scalar objective equality, K>=R exact gradients,

one-loop/Dense4 equality, actual intermediate prefix gradients absent after detach,

nonzero retained prelude/embedding/core/coda/norm paths, conditional fixed-prefix

finite differences on actual core/prelude parameters, manual causal attention

agreement, and nontraditional RoPE identities. CUDA BF16 gates passed with finite

state and independently normalized batched versus separate Muon updates.



The original exact-forward gate compared scalar objectives. A committed post-window

addendum strengthened it without changing the timed model: **every logit** at

(8,256,32768) is exactly equal for full/TB2 and one-loop/Dense4. Full-geometry

K>=R gradient tensors are also exactly equal. Both small CPU and CUDA checkpoint

predictions and next resumed optimizer updates agree exactly. CUDA checkpoint

SHA256: `{v3['cuda_checkpoint']['checkpoint_sha256']}`; checkpoint retained locally

outside the return package. All checkpoint files are excluded from the ZIP.



Muon uses source-defined projection ownership, independent per-matrix Frobenius

normalization, five NS steps with coefficients3.4445/-4.7750/2.0315, tall-matrix

transpose, groups of at most5, m=.95m+.05g, u=.05g+.95m, and source shape scaling.

Adam owns the one tied embedding and norm vectors, beta(.9,.95), bias correction,

epsilon1e-8; weight decay0. Clip1, Muon LR.01, Adam LR.0003, no tuning. The small

three-step batched/single hybrid master relativeL2 was

{cpu['hybrid_batched_vs_individual_three_step_master_relative_l2']:.8g}; the tests

also independently checked the first-step Nesterov and Adam formulas.



## Optimizer profile



The hybrid update costs about14.5ms in the recurrent benchmark. A separately

registered fixed-gradient optimizer-only probe used three warmups, five unprofiled

updates and five profiled updates on the unchanged full-geometry optimizer.

Corrected profile: unprofiled median **{prof['unprofiled_median_ms']:.2f}ms**, profiled

median **{prof['profiled_median_ms']:.2f}ms**. Instrumentation materially raises cost;

headline speed remains the original v2 windows.



Across five profiled updates, compute-kernel time totaled

**{profile_summary['compute_kernel_total_ms']:.3f}ms** (mean

**{profile_summary['compute_kernel_mean_ms_per_update']:.3f}ms/update**), with a further

{profile_summary['memory_operations_total_ms']:.3f}ms of device memcpy/memset operations.

The profiler measured **{prof['cuda_launch_api_calls']:,} CUDA launch API calls**

({prof['cuda_launch_api_calls']/5:.0f}/update) and

**{profile_summary['cuda_launch_cpu_total_ms']:.3f}ms** of host time inside those calls

({profile_summary['cuda_launch_cpu_mean_ms_per_update']:.3f}ms/update). GPU and CPU

durations overlap and cannot be added to reconstruct wall time. Launch-call

duration does not include all Python dispatch or scheduler overhead. The measured

~14.5ms wall update is therefore not all GPU compute.



| Operator | Calls across5 updates | Self GPU ms | Self CPU ms |

| --- | ---: | ---: | ---: |

{operator_rows}



Pointwise multiplication/addition/division and copies account for substantial

GPU time as well as many launches. That identifies candidates for a future

formula-preserving fusion/compilation experiment; no optimizer formula or

implementation optimization was made here. These operator self times and

parent/inclusive times in raw JSON must not be mixed or summed twice. Top kernels,

all aggregate compute kernels, memory operations, CPU operator averages and API

call durations are in `measurements/v4/optimizer-profile.json`.



The first profile export (v3) mistakenly counted five GPU annotation ranges as

leaf kernels. It is preserved, explicitly superseded for profiler totals, and not

used above. A frozen v4 amendment reran only the tiny profiling probe, excluding

annotations and separating compute from memory operations; no benchmark-suite

rerun occurred. All its GPU time counts against the original600s cap.



## Safety and backend limits



Every window required AC and a <=65°C start. An .8-second monitor checked power,

temperature, clocks, memory and competing compute PIDs. Benchmark-window telemetry had a maximum

gap of {v2['max_telemetry_sample_gap_s']:.3f}s; the corrected profile had

{v4['max_sample_gap_s']:.3f}s. **One v3 supplemental checkpoint/profile-transition

gap was {v3['max_sample_gap_s']:.3f}s**, exceeding the one-second cadence requirement,

with42°C and43°C samples on either side. This is a disclosed monitor-cadence

deviation. The sampler checked individual query duration and1.5s staleness but

did not reject every >1s inter-record scheduling gap. Strict cadence compliance

therefore does not apply to the whole addendum. No failed sensor readings,

thermal/resource/nonfinite stop or competing research compute was observed.

Before longer GPU work, use an external monitor or a strict inter-sample deadline.

No extra GPU repetition was launched after discovering the deviation. The recorded allocator cap used measured free

VRAM minus1.5GiB, capped at75% physical VRAM. The largest benchmark device-wide

memory sample was1,829MiB. PyTorch allocated/reserved and NVIDIA device-wide

measurements are distinct; the device sample is not a continuous memory peak.



The start temperature was much lower than the earlier generic test; short

windows and <=65°C cooldown checks bounded heat. They do not establish thermal

equilibrium or long unattended training reliability. CPU temperatures were not

available. Some read-only Windows CIM/event diagnostics overlapped benchmark

wall time; desktop scheduling, monitoring, logging and background CPU work can

affect an eager Python implementation. No cross-machine speed ceiling is claimed.



Windows torch2.7.0+cu128 lacks Flash Attention. Frozen v1 passed CPU correctness

then stopped at its forced Flash gate before any benchmark. Its failure is

preserved. Registered v2 audited backend eligibility and selected

**EFFICIENT_ATTENTION**, forced for CUDA correctness and all timings; no math

fallback was needed. GQA repeats KV heads explicitly. Separate BF16 RoPE arithmetic,

RMS normalization casts, CUDA batched addmm and reduction order can round

differently from fused MLX kernels. Adam uses a host integer step and scalar

bias-correction powers before FP32 tensor arithmetic, while MLX uses tensor powers.

Embedding/linear initialization distributions match the reference; backend RNG

draws/tensors do not. Eager CUDA versus compiled MLX, kernel choices, initialization,

instrumentation and workload histories preclude an exact Mac/CUDA ratio.



## Read-only laptop diagnostics



Actual saved JSON is valid. The command that displayed null BIOS/OS/Admin values

applied Select-Object to a dictionary incorrectly; the underlying structured

record was intact and was parsed again successfully.



Current OS: {hardware['OS']['Caption']} build{hardware['OS']['BuildNumber']}, last boot

{hardware['OS']['LastBootUpTime']}. BIOS remains {hardware['BIOS']['SMBIOSBIOSVersion']}.

NVIDIA driver616.92; Intel Iris Xe driver31.0.101.4502 remains recorded from2023.

Windows reports the SSD healthy. AC was connected and battery100% during preflight.

The session was not elevated. SleepStudy, power requests and detailed storage

counters remain administrator-only limitations. No WHEA records were returned

for30 days; that does not certify RAM/hardware health.



Current lid-close AC/DC policies remain Hibernate; S0 Modern Standby/Hibernate/

Fast Startup are available and S3 unavailable. The recent event summary still

contains {len(diagnostics['UncleanRestarts'])} unclean restarts and

{len(diagnostics['FailedHibernateResumes'])} failed hibernate resumes with0xC000007B.

The lid issue remains unresolved; no lid-close or hibernate reproduction was

attempted. Firmware/driver/system-file/resume diagnostics remain separate

maintenance work. Existing laptop healthcheck reports were preserved.



## Provenance and deliverables



Source archive SHA256:

`688f5517f878e8583dc0584a6a13aec56aa826537582753ff3a90128a2e4e539`.

The supplied commitc022d50ed5268514ef5cfa1348274d3840d0ae4e is sender-attested;

the archive has no Git history. All17 included entries across the three inherited

recurrence freeze maps verified; five missing entries are intentionally omitted

data files. Initial and final full extracted-tree hashes agree. No prior frozen

source or record was changed.



New source/config/contracts are under `versions/v1` through `versions/v4`, each

registered/committed/frozen before its GPU work. v2 measurements are authoritative

for timing; v3 for strengthened verification/checkpoints; v4 for profiling. All

failed/superseded evidence remains. `SUMMARY.json` is CPU-derived, not new testing.

Return package excludes checkpoints, corpora, environments, installers, raw

enormous OS logs and Git internals. The local Git repository retains registration

history; `GIT_HISTORY.txt` records commits for the portable package.



The clip1 versus clip5 fresh-data learning experiment remains deferred. No real

data or final-test set was evaluated. No production change or scale-up follows

from this pilot.

'''

(ROOT/'REPORT.md').write_text(text,encoding='utf-8')

(ROOT/'GIT_HISTORY.txt').write_text(history+'\n')

print(json.dumps({'status':'PASS','paired_speed_gain_percent':derived['median_paired_tb2_speed_gain_percent'],'backward_reduction_percent':derived['median_paired_backward_reduction_percent'],'gpu_active_seconds':usage,'profile':profile_summary},indent=2))
