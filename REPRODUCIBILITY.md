# Evidence, export integrity and reproduction limits

## What this repository contains

The public snapshot contains selected reports, registered contracts/plans and
amendments, numerical failures, scalar training/timing records, plots and
historical experiment Python sources from the local `moe_night` project.
Windows versions v1–v4 preserve the stopped backend attempt and corrected profile.
No accelerator work occurred during publication.

`SOURCE_MANIFEST.json` identifies the local source commit and, for every exported
artifact, its original relative path, original SHA-256 and published SHA-256.
An unchanged file has the same two hashes. Sanitized files have different hashes.
This is a publication provenance record; it is not independent proof of when the
original measurements or preregistrations occurred.

`PUBLICATION_SHA256.json` records checksums for the complete published tree except
itself and Git internals. `tools/verify_archive.py` verifies them, parses all
JSON/JSONL, checks exported-file hashes and recomputes the latest batching summary
from the six saved runs. This is CPU-only evidence verification, not a new
numerical-gradient or training run.

## What changed for public release

- Machine-specific home, volume and temporary paths became `/RESEARCH_ROOT` or opaque `/EXTERNAL_RESOURCE/...` placeholders. Report image links now point to published plots.
- Arbitrary command lines, process lists/PIDs, host identifiers and unrelated laptop-health diagnostics were removed. An environment record containing a broad process inventory was replaced by selected public package/device metadata.
- Text was normalized to UTF-8 and LF line endings. Reports carry an archival-copy notice; the Windows report omits its unrelated OS-maintenance section.
- Selected third-party downloaded source references, private Git history, operational launch configuration and production trainer configuration were not exported.
- Corpora, tokenizer assets, checkpoints, weights, input arrays, tensor samples, stdout/error logs, graph exports, environments, installers and ZIP transfers were not exported. Some report references to those records concern originals retained locally, rather than files distributed here.

Losses, gradient-error statistics, timing measurements, mathematical formulas,
experiment decisions and failed tolerances were not revised to improve results.
The local original frozen files and environments remain separate and intact.

## Historical sources are not a turnkey public runner

Sources under `experiments/` are the measured implementation records, with path
sanitization where needed. They may refer to private freeze maps, unavailable data,
original directory layouts or missing operational files. **Do not expect their
original freeze checks or reproduction commands to succeed in this public tree.**
The original hashes refer to original bytes. They must not be presented as valid
checksums of the sanitized publication. Do not remove a freeze check and then
claim that a new run reproduced the frozen source.

For a new run, create a fresh output directory and environment; explicitly adapt
the paths/data; register that new scope; freeze the new source and inputs; run
the numerical checks and budgets for that new recipe. Compare the new record to
this archive with those differences disclosed. Never point historical workers
at existing evidence outputs: some workers write their `metrics/` files.

## Geometry and recipe details to preserve

The 298M H1 MLX loss/batching study used 20 layers, width 1024, SwiGLU intermediate
3456, GQA 8/2, head dimension 128, tied 32768 vocabulary and RoPE theta 500000.
There are 4095 input positions from 4096-ID sequences. Model/backward are BF16;
master weights, accumulation and optimizer state are FP32. Loss is explicitly
composed in FP32 for the accepted memory recipe. The prior Mac Muon/AdamW equations
were preserved and differ from production CUDA. Do not substitute an optimizer
with a matching name and assume matching update equations.

The larger-batch sweep kept the same eight NumPy-seeded synthetic sequences,
32,760 targets per optimizer update, clipping once after accumulation, no
activation checkpointing and two fresh-process rounds. Numerical limits were
absolute mean-loss difference ≤2e-4 and full accumulated-gradient relative L2
≤0.005, at both context 64 and actual context 4095. B2/B4 fail the former.

The corpus-based recurrence pilot used d256/h864, context 256, GQA 8/2, tied 32768
vocabulary, a prelude, two-block shared core and coda. R4/K2 preserves all four
forward loops and detaches prefix state before the third loop while retaining
the prelude injection gradient path. All recurrent evaluation used R4. Same-seed
initial common-weight and token-order hashes were checked across variants. The
gradient is intentionally different from full backpropagation.

The old corpus archive did not preserve all upstream commit/license versions.
Original sampled data and tokenizer are not publicly distributed here. Therefore
the exact corpus-based fits cannot be rerun from this repository alone. Published
hashes and scalar loss curves are evidence of those local runs, not a released
dataset or a licensing clearance.

Native Windows CUDA timings used efficient attention with explicit GQA head
repetition, eager PyTorch and a source-defined hybrid optimizer. Compiled MLX,
different kernels/rounding/RNG, measurement instrumentation and background load
prevent interpreting absolute cross-machine rates as hardware-only differences.

## What was not measured

Short synthetic windows do not establish convergence, quality parity, thermal
equilibrium or unattended-training reliability. Three small recurrent learning
seeds do not establish a production-scale capability gain. Learning-curve
interpolations are descriptive, not new matched-wall-budget training. Mac MLX
active allocator peaks, process RSS, PyTorch allocated/reserved VRAM and NVIDIA
device-wide memory are different metrics. They must not be equated.

More unified-memory capacity did not add compute or bandwidth in this sweep.
The data support a limited negative batching result, not a universal best batch
size or a prediction that all larger-RAM machines train at the same speed.
