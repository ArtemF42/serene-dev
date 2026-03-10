import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn


def compute_hits(recs: torch.Tensor, actuals: torch.Tensor) -> torch.Tensor:
    """
    Args:
        recs (torch.Tensor): tensor of shape (B, K).
        actuals (torch.Tensor): tensor of shape (B,).

    Returns:
        torch.Tensor: tensor of shape (B, K).
    """
    return recs.eq(actuals.unsqueeze(-1)).float()


class BaseMetric(nn.Module, ABC):
    def __init__(self, top_k: int | Sequence[int]) -> None:
        super().__init__()

        top_k = tuple(top_k) if isinstance(top_k, Sequence) else (top_k,)
        max_k = max(top_k)

        if not all(k > 0 for k in top_k):
            raise ValueError("all K values must be greater than 0.")

        self.top_k = top_k
        self.max_k = max_k

        self.register_buffer("k_idx", torch.tensor(top_k) - 1)

    def forward(
        self,
        recs: torch.Tensor | None = None,
        actuals: torch.Tensor | None = None,
        hits: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if hits is None:
            if recs is None or actuals is None:
                raise ValueError("either `hits`, or both `recs` and `actuals` must be provided.")

            hits = compute_hits(recs, actuals)
        else:
            if recs is not None or actuals is not None:
                logging.warning("`recs` and `actuals` are ignored when `hits` is provided.")

        values = self._compute(hits[:, : self.max_k])[:, self.k_idx].mean(dim=0)
        return {f"{self.name}@{k}": value.item() for k, value in zip(self.top_k, values)}

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _compute(self, hits: torch.Tensor) -> torch.Tensor: ...


class HitRate(BaseMetric):
    """Implementation of Hit Rate."""

    def _compute(self, hits: torch.Tensor) -> torch.Tensor:
        return torch.cummax(hits, 1).values


class MRR(BaseMetric):
    """Implementation of Mean Reciprocal Rank (MRR)."""

    def __init__(self, top_k: int | Sequence[int]) -> None:
        super().__init__(top_k=top_k)

        self.register_buffer("reciprocal_ranks", 1 / (torch.arange(self.max_k) + 1))

    def _compute(self, hits: torch.Tensor) -> torch.Tensor:
        return torch.cummax(hits * self.reciprocal_ranks, 1).values


class NDCG(BaseMetric):
    """Implementation of Normalized Discounted Cumulative Gain (NDCG).

    Note:
        Formally computes DCG, which is equivalent to NDCG since the ground truth contains exactly one item.
    """

    def __init__(self, top_k: int | Sequence[int]) -> None:
        super().__init__(top_k=top_k)

        self.register_buffer("discount_factors", 1 / torch.log2(torch.arange(self.max_k) + 2))

    def _compute(self, hits: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(hits * self.discount_factors, 1)
