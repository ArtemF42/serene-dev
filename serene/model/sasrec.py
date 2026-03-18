import logging
from typing import Literal

import torch
import torch.nn as nn

from rotary_embedding_torch import RotaryEmbedding

from .attention import CausalSelfAttention
from .feed_forward_network import FeedForwardNetwork


class SASRecBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        intermediate_dim: int,
        dropout_p: float,
        activation: Literal["relu", "gelu", "silu", "swiglu"] = "swiglu",
    ) -> None:
        super().__init__()

        self.pre_attn_rms_norm = nn.RMSNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, dropout_p)

        self.pre_ffn_rms_norm = nn.RMSNorm(embedding_dim)
        self.ffn = FeedForwardNetwork(embedding_dim, intermediate_dim, dropout_p, activation)

    def forward(self, x: torch.Tensor, rotary_embedding: RotaryEmbedding | None = None) -> torch.Tensor:
        x = x + self.attn(self.pre_attn_rms_norm(x), rotary_embedding)
        x = x + self.ffn(self.pre_ffn_rms_norm(x))

        return x


class SASRecModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout_p: float,
        padding_idx: int = 0,
        intermediate_dim: int | None = None,
        activation: Literal["relu", "gelu", "silu", "swiglu"] = "swiglu",
    ) -> None:
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.padding_idx = padding_idx

        if intermediate_dim is None:
            intermediate_dim = embedding_dim * 4
            logging.info(f"`intermediate_dim` was not provided, set to {intermediate_dim}.")

        self.intermediate_dim = intermediate_dim
        self.activation = activation

        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx)

        self.embedding_dropout = nn.Dropout(dropout_p)
        self.rotary_embedding = RotaryEmbedding(embedding_dim // num_heads)

        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            block = SASRecBlock(
                embedding_dim,
                num_heads,
                intermediate_dim,
                dropout_p,
                activation,
            )
            self.blocks.append(block)

        self.out_rms_norm = nn.RMSNorm(embedding_dim)

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        inputs_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs is not None:
            if inputs_embeddings is not None:
                raise ValueError("cannot specify both `inputs` and `inputs_embeddings` at the same time.")

            inputs_embeddings = self.item_embedding(inputs)

        if inputs_embeddings is None:
            raise ValueError("either `inputs` or `inputs_embeddings` must be specified.")

        x = self.embedding_dropout(inputs_embeddings)

        for block in self.blocks:
            x = block(x, self.rotary_embedding)

        return self.out_rms_norm(x)
