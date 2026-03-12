import logging

import polars as pl


def apply_min_count_filter(events: pl.DataFrame, min_count: int, key: str) -> pl.DataFrame:
    return events.filter(pl.len().over(key) >= min_count)


def apply_n_core_filter(
    events: pl.DataFrame,
    min_count: int | None = None,
    user_min_count: int | None = None,
    item_min_count: int | None = None,
    user_key: str = "user_id",
    item_key: str = "item_id",
) -> pl.DataFrame:
    if min_count is None:
        if user_min_count is None or item_min_count is None:
            raise ValueError("if `min_count` is not specified, both `user_min_count` and `item_min_count` must be provided.")  # fmt: skip
    else:
        if user_min_count is not None or item_min_count is not None:
            logging.warning("`user_min_count` and `item_min_count` are overridden by `min_count`.")

        user_min_count = item_min_count = min_count

    height = -1

    while events.height != height:
        height = events.height

        events = apply_min_count_filter(events, user_min_count, user_key)
        events = apply_min_count_filter(events, item_min_count, item_key)

    return events


def remove_cold_items(events: pl.DataFrame, item_mapping: dict, item_key: str = "item_id") -> pl.DataFrame:
    return events.filter(pl.col(item_key).is_in(list(item_mapping)))


def subsample_users(
    events: pl.DataFrame,
    n_users: int | None = None,
    frac: float | None = None,
    seed: int = 42,
    user_key: str = "user_id",
) -> pl.DataFrame:
    if n_users is None and frac is None:
        raise ValueError("either `n_users` or `frac` must be specified.")

    users = (
        events.get_column(user_key)
        .unique(maintain_order=True)
        .sample(n=n_users, fraction=frac, seed=seed)
        .implode()
    )  # fmt: skip

    return events.filter(pl.col(user_key).is_in(users))
