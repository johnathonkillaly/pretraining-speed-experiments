> Public archival copy. Paths and operational metadata were sanitized; original decisions and numerical failures are retained. See the root REPRODUCIBILITY.md for export and reproduction limits.

# CUDA backend context

Checked 2026-10-05 while the connected laptop campaign was running.

The inspected laptop environment is native Windows, Python 3.13.15, and
PyTorch 2.7.0+cu128 (CUDA runtime 12.8). The first campaign's forced Flash
Attention gate failed because this particular wheel lacks that kernel. The
laptop retained the failed version and registered an available-kernel probe
before timing. The final measurement records, rather than a platform-wide
assumption, determine which attention implementation actually ran.

PyTorch's [Windows Flash Attention tracking issue](https://github.com/pytorch/pytorch/issues/108175)
was still open when checked. A maintainer's
[related regression report](https://github.com/pytorch/pytorch/issues/125070)
describes memory-efficient or math attention as alternatives for affected
Windows builds. This supports probing those alternatives; it does not imply
that every Windows package or third-party kernel has the same limitation.

The reused environment is an initial baseline. The official
[PyTorch 2.10 release post](https://pytorch.org/blog/pytorch-2-10-release-blog/)
already describes later compiler improvements, including fusion of independent
operations to reduce kernel launch overhead. No newer-wheel speedup was measured
in this campaign. Neither eager Windows timings nor compiled MLX timings alone
identify a hardware performance ceiling.

A follow-up comparison could use a current, verified Linux/WSL CUDA stack with
supported fast attention and compilation, then match initialization tensors,
inputs, objective, optimizer and timing policy. NVIDIA documents CUDA on WSL in
its [official guide](https://docs.nvidia.com/cuda/wsl-user-guide/). The present
campaign does not install WSL or change the laptop's driver or power policies.
