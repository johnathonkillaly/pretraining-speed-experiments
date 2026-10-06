"""Local Llama architecture compatible with BoothGPT MLX checkpoints."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint
from mlx.utils import tree_map

from .rope_utils import initialize_rope


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params: dict[str, Any]):
        return cls(
            **{
                key: value
                for key, value in params.items()
                if key in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


def create_causal_mask(
    length: int,
    offset: int = 0,
    window_size: Optional[int] = None,
) -> mx.array:
    right = mx.arange(offset + length)[None]
    left = (
        mx.arange(offset, offset + length)
        if offset
        else mx.arange(offset + length)
    )[:, None]
    mask = left >= right
    if window_size is not None:
        mask = mask & (left <= right + window_size)
    return mask


def create_attention_mask(
    hidden: mx.array,
    cache: Optional[Any] = None,
    return_array: bool = False,
):
    length = hidden.shape[1]
    if length <= 1:
        return None
    offset = 0
    window_size = None
    if cache is not None and cache[0] is not None:
        first_cache = cache[0]
        offset = first_cache.offset
        if hasattr(first_cache, "max_size"):
            window_size = first_cache.max_size
            offset = min(window_size, offset)
            return_array = return_array or offset + length > window_size
    if return_array:
        return create_causal_mask(length, offset, window_size)
    return "causal"


def _is_quantized_cache(cache: Any) -> bool:
    return cache is not None and all(
        hasattr(cache, name) for name in ("group_size", "bits")
    )


def scaled_dot_product_attention(
    queries: mx.array,
    keys: Any,
    values: Any,
    cache: Any,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    if not _is_quantized_cache(cache):
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
        )

    batch, query_heads, length, dims = queries.shape
    kv_heads = keys[0].shape[-3]
    repeats = query_heads // kv_heads
    queries *= scale
    if repeats > 1:
        queries = mx.reshape(
            queries,
            (batch, kv_heads, repeats, length, dims),
        )
        keys = tree_map(lambda value: mx.expand_dims(value, axis=-3), keys)
        values = tree_map(lambda value: mx.expand_dims(value, axis=-3), values)
    scores = mx.quantized_matmul(
        queries,
        *keys,
        transpose=True,
        group_size=cache.group_size,
        bits=cache.bits,
    )
    if mask is not None:
        scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    output = mx.quantized_matmul(
        scores,
        *values,
        transpose=False,
        group_size=cache.group_size,
        bits=cache.bits,
    )
    if repeats > 1:
        output = mx.reshape(output, (batch, query_heads, length, dims))
    return output


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dimensions = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or dimensions // self.n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            dimensions,
            self.n_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            dimensions,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            dimensions,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            dimensions,
            bias=args.attention_bias,
        )
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.reshape(
            batch,
            length,
            self.n_heads,
            -1,
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch,
            length,
            self.n_kv_heads,
            -1,
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch,
            length,
            self.n_kv_heads,
            -1,
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=args.mlp_bias,
        )
        self.down_proj = nn.Linear(
            args.intermediate_size,
            args.hidden_size,
            bias=args.mlp_bias,
        )
        self.up_proj = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=args.mlp_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.activation_checkpointing = args.activation_checkpointing

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden = self.embed_tokens(inputs)
        if mask is None:
            mask = create_attention_mask(hidden, cache)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            layer_fn = (
                checkpoint(layer)
                if self.activation_checkpointing and layer_cache is None
                else layer
            )
            hidden = layer_fn(hidden, mask, cache=layer_cache)
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size,
                args.vocab_size,
                bias=False,
            )

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        output = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(output)
        return self.lm_head(output)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        weights = {
            key: value
            for key, value in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in key
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers
