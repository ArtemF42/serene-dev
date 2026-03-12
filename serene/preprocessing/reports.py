import polars as pl


def report_stats(events: pl.DataFrame, user_key: str = "user_id", item_key: str = "item_id") -> pl.DataFrame:
    return events.select(
        pl.len().alias("Num. Events"),
        pl.col(user_key).n_unique().alias("Num. Users"),
        pl.col(item_key).n_unique().alias("Num. Items"),
        (pl.len() / pl.col(user_key).n_unique()).alias("Avg. Length"),
    )


def report_distribution(events: pl.DataFrame, key: str) -> pl.DataFrame:
    return events.group_by(key).len(name="count").get_column("count").describe()
