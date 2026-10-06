"""Small, explicitly specified recurrent core with truncated depth gradients."""
import hashlib
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import checkpoint
from mlx.utils import tree_flatten, tree_map

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "2026-10-04"
sys.path.append(str(OLD / "scripts"))
from arch.llama import ModelArgs, TransformerBlock, create_attention_mask

KINDS = ("dense10", "dense4", "rec_full", "rec_tb2", "rec_curr_tb2")


def pilot_args():
    return ModelArgs(model_type="llama", hidden_size=256, intermediate_size=864,
                     num_hidden_layers=10, num_attention_heads=8, num_key_value_heads=2,
                     head_dim=32, rms_norm_eps=1e-5, vocab_size=32768,
                     rope_theta=500000, max_position_embeddings=256,
                     tie_word_embeddings=True)


class LanguageModel(nn.Module):
    def __init__(self, kind, seed, args=None, checkpoint_mode="none"):
        super().__init__()
        args = args or pilot_args()
        mx.random.seed(seed)
        self._kind = kind
        self._checkpoint_mode = checkpoint_mode
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        bank = [TransformerBlock(args) for _ in range(max(10, args.num_hidden_layers))]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Evaluate the whole prototype bank so common initialization is auditable.
        mx.eval([block.parameters() for block in bank], self.parameters())
        common = [self.embed_tokens.weight, self.norm.weight]
        for index in (0, 1, 2, 9):
            common.extend(value for _, value in tree_flatten(bank[index].parameters()))
        digest = hashlib.sha256()
        for value in common:
            digest.update(np.array(value).tobytes())
        self._shared_initialization_sha256 = digest.hexdigest()
        if kind.startswith("dense"):
            indices = (0, 1, 2, 9) if kind == "dense4" else range(args.num_hidden_layers)
            self.blocks = [bank[index] for index in indices]
        else:
            self.prelude = bank[0]
            self.core = [bank[1], bank[2]]
            self.coda = bank[9]

    def __call__(self, tokens, loops=4, backprop_loops=4):
        hidden = self.embed_tokens(tokens)
        mask = create_attention_mask(hidden)
        if self._kind.startswith("dense"):
            for index, block in enumerate(self.blocks):
                fn = checkpoint(block) if self._checkpoint_mode == "alternating" and index % 2 == 0 else block
                hidden = fn(hidden, mask)
        else:
            embedding = self.prelude(hidden, mask)
            hidden = embedding
            for index in range(loops):
                if index == loops - backprop_loops and index > 0:
                    hidden = mx.stop_gradient(hidden)
                hidden = .5 * (hidden + embedding)
                for block in self.core:
                    hidden = block(hidden, mask)
            hidden = self.coda(hidden, mask)
        return self.embed_tokens.as_linear(self.norm(hidden))


def depths(kind, step, steps):
    if kind == "rec_curr_tb2":
        loops = 1 if step <= max(1, int(.05 * steps)) else 2 if step <= max(2, int(.10 * steps)) else 4
        return loops, min(loops, 2)
    return 4, 4 if kind == "rec_full" else 2


def objective(model, inputs, targets, loops=4, backprop_loops=4):
    logits = model(inputs, loops, backprop_loops).astype(mx.float32)
    mask = (targets != 0).astype(mx.float32)
    return (nn.losses.cross_entropy(logits, targets, reduction="none") * mask).sum() / mask.sum()


def convert_to_bf16(model):
    masters = {name: value.astype(mx.float32) for name, value in tree_flatten(model.trainable_parameters())}
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    mx.eval(model.parameters(), masters)
    return masters
