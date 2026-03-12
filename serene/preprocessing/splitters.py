from datetime import datetime

import polars as pl


def apply_global_time_splitter(
    events: pl.DataFrame,
    time_threshold: int | datetime | str | float,
    time_format: str | None = None,
    user_key: str = "user_id",
    time_key: str = "timestamp",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if isinstance(time_threshold, str):
        if time_format is None:
            raise ValueError("string `time_threshold` requires a `time_format`.")

        time_threshold = datetime.strptime(time_threshold, time_format)
    elif isinstance(time_threshold, float):
        if not 0 <= time_threshold <= 1:
            raise ValueError("float `time_threshold` must be between 0.0 and 1.0.")

        time_threshold = events.get_column(time_key).quantile(time_threshold)

    condition = pl.col(time_key) <= time_threshold

    return events.filter(condition), events.filter(condition.not_().any().over(user_key))


def apply_random_user_splitter(
    events: pl.DataFrame,
    frac: float = 0.9,
    seed: int = 42,
    user_key: str = "user_id",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    users = events.get_column(user_key).unique(maintain_order=True).sample(fraction=frac, seed=seed).implode()

    condition = pl.col(user_key).is_in(users)

    return events.filter(condition), events.filter(~condition)


def apply_last_n_splitter(
    events: pl.DataFrame,
    n: int = 1,
    user_key: str = "user_id",
    time_key: str = "timestamp",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    events = events.with_columns(
        pl.col(time_key)
        .rank(method="ordinal", descending=True)
        .over(user_key)
        .alias("__rank__")
    )  # fmt: skip

    condition = pl.col("__rank__") > n

    return events.filter(condition).drop("__rank__"), events.filter(~condition).drop("__rank__")
