import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from rotary_embedding_torch import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Optimized implementation of causal self-attention."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout_p: float) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("`embedding_dim` must be divisible by `num_heads`.")

        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, rotary_embedding: RotaryEmbedding | None = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape (B, L, E).

        Returns:
            torch.Tensor: attention output of shape (B, L, E).
        """
        q, k, v = map(self._split_heads, self.qkv_proj(x).chunk(3, dim=-1))

        if rotary_embedding is not None:
            q = rotary_embedding.rotate_queries_or_keys(q)
            k = rotary_embedding.rotate_queries_or_keys(k)

        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
            attn_out = nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )  # fmt: skip

        return self.dropout(self.out_proj(self._merge_heads(attn_out)))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape (B, L, E).

        Returns:
            torch.Tensor: tensor of shape (B, H, L, E // H).
        """
        return x.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape (B, H, L, E // H).

        Returns:
            torch.Tensor: tensor of shape (B, L, E).
        """
        return x.transpose(1, 2).flatten(-2)
