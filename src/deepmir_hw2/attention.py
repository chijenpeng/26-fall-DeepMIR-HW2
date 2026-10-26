# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Modified for this project: query blocking, recomputation and lazy relative bias.
"""Exact full-context T5 attention with bounded query-block intermediates.

Projection/cache handling follows Transformers 4.57.1's Apache-2.0 T5Attention.
Every query still attends to every permitted key; blocking does not shorten music.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.t5.modeling_t5 import T5Attention


@dataclass(frozen=True)
class RelativeBias:
    weight: torch.Tensor | None
    indices: torch.Tensor | None


def attend(query, key, value, weight, mask, buckets, *, dropout):
    """Materialize relative bias only for this query block, with T5's scale=1."""
    if weight is not None:
        additive = F.embedding(buckets.long(), weight).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            additive = additive + mask
    else:
        additive = mask
    if additive is not None:
        # Embedding/permute leaves a head-contiguous bias. CUDA's fused SDPA
        # needs its key dimension contiguous; otherwise it falls back to dense math.
        additive = additive.to(query.dtype).contiguous()
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=additive, dropout_p=dropout, scale=1.0
    )


class ChunkedT5Attention(T5Attention):
    """T5-compatible weights/cache with shared compact relative-position bucket IDs."""

    def __init__(self, config, has_relative_attention_bias=False, layer_idx=None):
        super().__init__(config, has_relative_attention_bias, layer_idx)
        self.chunk_size = config.query_block_size
        if self.chunk_size <= 0:
            raise ValueError("query_block_size must be positive")

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_values=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        if output_attentions or self.pruned_heads:
            raise ValueError(
                "Chunked T5 does not materialize attention maps or support head pruning"
            )
        batch, length = hidden_states.shape[:2]
        query = self.q(hidden_states).view(batch, -1, self.n_heads, self.key_value_proj_dim)
        query = query.transpose(1, 2)
        cross = key_value_states is not None
        updated = False
        if isinstance(past_key_values, EncoderDecoderCache):
            updated = past_key_values.is_updated.get(self.layer_idx)
            cache = (
                past_key_values.cross_attention_cache
                if cross
                else past_key_values.self_attention_cache
            )
        else:
            cache = past_key_values
        states = key_value_states if cross else hidden_states
        if cross and past_key_values is not None and updated:
            key = cache.layers[self.layer_idx].keys
            value = cache.layers[self.layer_idx].values
        else:
            key = self.k(states).view(batch, -1, self.n_heads, self.key_value_proj_dim)
            value = self.v(states).view(batch, -1, self.n_heads, self.key_value_proj_dim)
            key, value = key.transpose(1, 2), value.transpose(1, 2)
            if cache is not None:
                key, value = cache.update(
                    key,
                    value,
                    self.layer_idx,
                    {"cache_position": None if cross else cache_position},
                )
                if cross and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True
        positions = (
            cache_position
            if cache_position is not None
            else torch.arange(length, device=query.device)
        )
        if position_bias is None:
            weight, indices = None, None
            if self.has_relative_attention_bias:
                weight = self.relative_attention_bias.weight
                keys = torch.arange(key.shape[-2], device=query.device)[None, :]
                dtype = torch.uint8 if self.relative_attention_num_buckets <= 256 else torch.int32
                chunks = []
                for start in range(0, length, self.chunk_size):
                    relative = keys - positions[start : start + self.chunk_size, None]
                    chunks.append(
                        self._relative_position_bucket(
                            relative,
                            bidirectional=not self.is_decoder,
                            num_buckets=self.relative_attention_num_buckets,
                            max_distance=self.relative_attention_max_distance,
                        ).to(dtype)
                    )
                # Share compact integer bucket IDs between layers, rather than
                # repeating the position/logarithm calculation in every layer.
                indices = torch.cat(chunks)
            position_bias = RelativeBias(weight, indices)
        outputs = []
        for start in range(0, length, self.chunk_size):
            end = min(start + self.chunk_size, length)
            block_mask = None
            if mask is not None:
                block_mask = mask[..., : key.shape[-2]]
                if block_mask.shape[-2] != 1:
                    block_mask = block_mask[..., start:end, :]
            arguments = (
                query[:, :, start:end],
                key,
                value,
                position_bias.weight,
                block_mask,
                None if position_bias.indices is None else position_bias.indices[start:end],
            )
            options = {"dropout": self.dropout if self.training else 0.0}
            # Also bound saved intermediates during backward, independently of the
            # outer Transformer-block checkpoint and without changing RNG semantics.
            if self.training and torch.is_grad_enabled():
                result = checkpoint(attend, *arguments, use_reentrant=False, **options)
            else:
                result = attend(*arguments, **options)
            outputs.append(result)
        output = torch.cat(outputs, dim=2)
        if layer_head_mask is not None:
            output = output * layer_head_mask
        output = output.transpose(1, 2).contiguous().view(batch, length, self.inner_dim)
        return self.o(output), position_bias


def use_chunked_t5_attention(model):
    """Replace attention only, retaining the model's initialized parameter values."""
    for stack in (model.encoder, model.decoder):
        for block in stack.block:
            layers = [(block.layer[0], "SelfAttention")]
            if stack.is_decoder:
                layers.append((block.layer[1], "EncDecAttention"))
            for layer, name in layers:
                original = getattr(layer, name)
                replacement = ChunkedT5Attention(
                    stack.config, original.has_relative_attention_bias, original.layer_idx
                )
                replacement.load_state_dict(original.state_dict())
                setattr(layer, name, replacement)
    return model
