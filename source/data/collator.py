from collections.abc import Sequence
from functools import partial
from typing import Any, Literal

import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorForCausalModeling:
    def __init__(
        self,
        stage: Literal["train", "eval", "predict"],
        padding_idx: int = 0,
        item_key: str = "item_id",
        feature_keys: Sequence[str] = (),
    ) -> None:
        self.stage = stage

        self.padding_idx = padding_idx

        self.item_key = item_key
        self.feature_keys = feature_keys

        self._pad_sequence = partial(pad_sequence, batch_first=True, padding_value=padding_idx, padding_side="right")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = {key: [example[key] for example in examples] for key in examples[0].keys()}

        if self.stage == "train" or self.stage == "eval":
            batch = self._process_train_eval_batch(batch)

        batch["__last__"] = torch.tensor([len(i) - 1 for i in batch[f"inputs.{self.item_key}"]], dtype=torch.long)

        for key in (self.item_key, *self.feature_keys):
            batch[f"inputs.{key}"] = self._pad_sequence(batch[f"inputs.{key}"])

            if self.stage == "train":
                batch[f"labels.{key}"] = self._pad_sequence(
                    batch[f"labels.{key}"],
                    padding_value=-100 if key == self.item_key else self.padding_idx,
                )
            elif self.stage == "eval":
                batch[f"labels.{key}"] = torch.stack(batch[f"labels.{key}"])

        return batch

    def _process_train_eval_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch["history"] = [history[:-1] for history in batch["history"]]

        for key in (self.item_key, *self.feature_keys):
            sequences = batch[f"inputs.{key}"]

            batch[f"inputs.{key}"] = [seq[:-1] for seq in sequences]

            if self.stage == "train":
                batch[f"labels.{key}"] = [seq[1:] for seq in sequences]
            else:
                batch[f"labels.{key}"] = [seq[-1] for seq in sequences]

        return batch
