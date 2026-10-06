"""RoPE implementations used by the local Llama architecture."""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class Llama3RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scaling_config: dict | None = None,
    ):
        super().__init__()
        if scaling_config is None:
            raise ValueError("Llama3RoPE requires a scaling configuration")
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional

        factor = scaling_config["factor"]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get(
            "original_max_position_embeddings",
            8192,
        )
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, dims, 2) / dims)
        wavelens = 2 * mx.pi * freqs
        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (
            wavelens < low_freq_wavelen
        )
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / (
            (1 - smooth_factors) / factor + smooth_factors
        )
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class YarnRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32,
        beta_slow: float = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ):
        super().__init__()

        def correction_dim(num_rotations: float) -> float:
            return (
                dims
                * math.log(
                    original_max_position_embeddings
                    / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dims - 1)

        def yarn_mscale(scale: float, multiplier: float) -> float:
            return 1.0 if scale <= 1 else 0.1 * multiplier * math.log(scale) + 1.0

        self.mscale = yarn_mscale(scaling_factor, mscale) / yarn_mscale(
            scaling_factor,
            mscale_all_dim,
        )
        freq_extra = base ** (
            mx.arange(0, dims, 2, dtype=mx.float32) / dims
        )
        freq_inter = scaling_factor * base ** (
            mx.arange(0, dims, 2, dtype=mx.float32) / dims
        )
        if low == high:
            high += 0.001
        ramp = (
            mx.arange(dims // 2, dtype=mx.float32) - low
        ) / (high - low)
        freq_mask = 1.0 - mx.clip(ramp, 0, 1)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        if self.mscale != 1.0:
            x[..., : self.dims] = self.mscale * x[..., : self.dims]
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: Optional[dict] = None,
    max_position_embeddings: Optional[int] = None,
) -> nn.Module:
    rope_type = (
        scaling_config.get("type")
        or scaling_config.get("rope_type", "default")
        if scaling_config is not None
        else "default"
    )
    if rope_type in {"default", "linear"}:
        scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)
    if rope_type == "llama3":
        return Llama3RoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings or 2048,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )
    if rope_type == "yarn":
        rope_kwargs = {
            key: scaling_config[key]
            for key in (
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
            if key in scaling_config
        }
        return YarnRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings or 2048,
            traditional=traditional,
            base=base,
            scaling_factor=scaling_config["factor"],
            **rope_kwargs,
        )
    raise ValueError(f"Unsupported RoPE type {rope_type}")
