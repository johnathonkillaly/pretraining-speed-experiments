> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# Apple / NVIDIA pretraining context

The new H1-shaped MLX probe sustained **3,263 prediction targets/s** on this
M4 Max. The September 18 CUDA study recorded **15,522.91 targets/s** across
90 measured updates on the owner's RTX 3080 at 280 W: about **4.76×** the Mac
pipeline throughput. Its later production sample was 15,667 targets/s. The
longer 90-update record is the reference used in this follow-up.

| Property | New Mac probe | Recorded CUDA study |
|---|---|---|
| Parameters | 298,361,856 | 298,361,856 |
| Shape | 20 layers, d1024, h3456, GQA 8/2, tied 32K head | Same |
| Update tokens | 8 × 4,095 = 32,760 | Same |
| Precision | BF16 model/backward, FP32 masters/accumulation/state | Same family |
| Optimizer | Compiled Muon/AdamW, independent matrices grouped up to 5 | Same family |
| Checkpointing | Alternating activation checkpointing | Alternating |
| Inputs | Fixed seeded uniform synthetic IDs | Actual pretraining data |
| Timing windows | 3 initial + 10 measured | 10 initial + 90 measured |
| Throughput | 3,263/s mean, 3,247/s median | 15,522.91/s long-run record |
| Allocator peak | 11.44 GiB MLX active | Different CUDA reserved/external metrics |

This answers a practical pipeline question, not a universal hardware ranking.
The code, kernels, cross-entropy path, checkpoint implementation, data and
measurement dates differ. There is no fresh NVIDIA run, common power budget,
thermal normalization or isolation of framework from silicon. The Mac timing
includes synchronized gradient accumulation, clipping and optimizer updates;
it excludes dataset I/O, validation and checkpoint writes. All 13 windows are
retained in `metrics/h1_benchmark_v2.json`. Loss decreases on repeated random
IDs are not model-quality evidence. Production was never contacted or changed.

The original optional probe failed before a completed window because the
checkpointed gradient function captured mutable model state only as compiled
input. `AMENDMENT_H1_CAPTURE.md` and a separately frozen v2 function capture it
as both input and output. The successful result comes from that corrected probe;
the original failure and source remain intact.

For the owner's current 298M pretraining workload, which already fits the
3080's VRAM, this evidence supports using the existing CUDA pipeline for the
long training run. The Mac is useful for local experiments and models whose
working set needs its much larger unified-memory capacity. More capacity does
not by itself shorten matrix multiplication or backward passes.

The local code-path audit requests MLX's fast causal attention. In installed
MLX 0.32.2, `nn.losses.cross_entropy` enables its fused fast path only on CUDA;
Metal uses the composed max/gather/logsumexp path. This experiment promotes
logits to FP32 for Metal loss accuracy, as the installed library documents.
The audit is in `configs/runtime_kernel_audit.json`. It identifies a software
difference worth profiling, not its fraction of the gap or a demonstrated
optimization opportunity; no additional kernel benchmark or change was made.

NVIDIA's Ampere architecture supplies BF16 Tensor Core matrix operations with
FP32 accumulation; that is relevant to this BF16 training workload. Attributing
the entire measured difference to that feature would be an inference unsupported
by this benchmark. Kernel choices and graph execution also matter.
See [NVIDIA's architecture guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).

MLX and Apple hardware are separate questions: MLX also offers a CUDA backend
on supported Linux/NVIDIA systems. This test does not compare MLX/CUDA with
PyTorch/CUDA, and gives no reason to migrate the running production stack.
See [MLX installation documentation](https://ml-explore.github.io/mlx/build/html/install.html).

The historical source is the unchanged local
CUDA optimizer report (local resource omitted).
The numerical comparison is reproducible from that record and the new raw JSON;
it should retain the above qualifications whenever reused.
