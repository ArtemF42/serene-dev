from typing import Self

import numpy as np
import polars as pl
import torch


class _AliasTable:
    def __init__(self, weights: np.ndarray) -> None:
        assert weights.ndim == 1

        weights = weights.astype(np.float64)

        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()

        n = weights.size

        probs = weights * n
        alias = np.zeros(n, dtype=np.int64)

        under, over = [], []

        for i, p in enumerate(probs):
            (under if p < 1.0 else over).append(i)

        while under and over:
            under_i, over_i = under.pop(), over.pop()

            probs[over_i] -= 1.0 - probs[under_i]
            alias[under_i] = over_i

            if probs[over_i] < 1.0:
                under.append(over_i)
            else:
                over.append(over_i)

        for i in (*under, *over):
            probs[i] = 1.0
            alias[i] = i

        self.n = n
        self.probs = probs
        self.alias = alias

    def __call__(self, size: int = 1) -> np.ndarray:
        i = np.random.randint(0, self.n, size=size)
        u = np.random.uniform(size=size)
        return np.where(u < self.probs[i], i, self.alias[i])


class RandomSampler:
    def __init__(
        self,
        items: np.ndarray,
        freqs: np.ndarray,
        alpha: float = 0.0,
        n_samples: int = 1,
    ) -> None:
        assert items.ndim == freqs.ndim == 1
        assert items.size == freqs.size
        assert 0.0 <= alpha <= 1.0
        assert n_samples >= 1

        self.items = items
        self.freqs = freqs**alpha
        self.alpha = alpha
        self.n_samples = n_samples

        self._alias_table = _AliasTable(self.freqs)

    def __call__(self) -> torch.Tensor:
        return torch.from_numpy(self.items[self._alias_table(size=self.n_samples)])

    @classmethod
    def from_polars(
        cls,
        data: pl.DataFrame,
        item_key: str = "item_id",
        alpha: float = 0.0,
        n_samples: int = 1,
    ) -> Self:
        """Initialize sampler from `polars.DataFrame`.

        Args:
            data (pl.DataFrame): interactions used to build sampler.
            item_key (str, optional): name of item column in data.
                Defaults to "item_id".
            alpha (float, optional): smoothing factor for item frequency.
                Defaults to 0.0.
            n_samples (int, optional): number of negative samples to draw.
                Defaults to 1.

        Returns:
            RandomSampler: random sampler.
        """
        items, freqs = data.group_by(item_key).len().to_numpy().T
        return cls(items, freqs, alpha, n_samples)
