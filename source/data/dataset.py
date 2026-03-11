from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs

import torch
from torch.utils.data import Dataset


@dataclass
class SequentialDataset(Dataset):
    events: pl.DataFrame
    max_length: int
    min_length: int = 1
    user_key: str = "user_id"
    item_key: str = "item_id"
    time_key: str = "timestamp"
    feature_keys: Sequence[str] = ()
    random_slice: bool = False

    def __post_init__(self) -> None:
        events = (
            self.events.select(self.user_key, self.item_key, self.time_key, *self.feature_keys)
            .with_columns(
                cs.integer().cast(pl.Int64),
                cs.float().cast(pl.Float32),
            )
            .filter(pl.len().over(self.user_key) >= self.min_length)
            .sort(self.user_key, self.time_key)
        )
        lengths = events.group_by(self.user_key, maintain_order=True).len(name="length")

        self.events: dict[str, np.ndarray] = {
            key: events.get_column(key).to_numpy(writable=True)
            for key in (self.item_key, *self.feature_keys)
        }  # fmt: skip
        self.users: list[Any] = lengths[self.user_key].to_list()

        lengths = lengths["length"].to_numpy()

        self._offsets: np.ndarray = np.zeros(len(self.users) + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum(lengths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start, end = self._offsets[idx], self._offsets[idx + 1]
        length = end - start

        if length <= self.max_length:
            _slice = slice(None)
        elif self.random_slice:
            shift = np.random.randint(0, length - self.max_length + 1)
            _slice = slice(shift, shift + self.max_length)
        else:
            _slice = slice(-self.max_length, None)

        example: dict[str, Any] = {self.user_key: self.users[idx], "history": self.events[self.item_key][start:end]}

        for key in (self.item_key, *self.feature_keys):
            example[f"inputs.{key}"] = torch.from_numpy(self.events[key][start:end][_slice])

        return example

    def __len__(self) -> int:
        return len(self.users)
