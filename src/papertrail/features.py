"""Per-transaction tabular features (XGBoost v1) and the one history feature the rule baseline needs.

`build_features` uses only the row itself. `add_out_count_24h` looks strictly backwards in time, so it
never sees the current row or any later row. Richer as-of and graph features come in Phase 4.
"""
import polars as pl

# Laundering amounts peak around $3K-$30K (reports/eda_summary.md, finding F5).
AMOUNT_BAND = (3_000.0, 30_000.0)


def add_out_count_24h(df: pl.DataFrame, ts_col: str = "ts") -> pl.DataFrame:
    """Number of transactions the sender (`src`) made in the 24h before this one, excluding this one.

    The window is [t - 24h, t): the current row and rows at the same timestamp are excluded.
    `df` must be sorted by `ts_col`.
    """
    if not df[ts_col].is_sorted():
        raise ValueError(f"df must be sorted by {ts_col}")
    return df.with_columns(
        pl.col("txn_id").is_not_null().cast(pl.UInt32)
        .rolling_sum_by(ts_col, "24h", closed="left")
        .over("src")
        .fill_null(0)
        .alias("out_count_24h")
    )


def fit_categories(train: pl.DataFrame) -> dict[str, list[str]]:
    """Category lists learned from the training split only (unseen values later map to -1 / all-zero)."""
    return {
        "pay_format": sorted(train["pay_format"].unique().to_list()),
        "cur_paid": sorted(train["cur_paid"].unique().to_list()),
    }


def build_features(df: pl.DataFrame, categories: dict[str, list[str]]) -> pl.DataFrame:
    """Feature table for XGBoost v1. Every feature is numeric; no history, no date, no weekday."""
    lo, hi = AMOUNT_BAND
    cur_map = {c: i for i, c in enumerate(categories["cur_paid"])}
    fmt = [
        (pl.col("pay_format") == f).cast(pl.Int8).alias("fmt_" + f.replace(" ", "_"))
        for f in categories["pay_format"]
    ]
    return df.select(
        pl.col("amt_paid").log1p().alias("log_amt"),
        pl.col("ts").dt.hour().cast(pl.Int8).alias("hour"),
        (pl.col("pay_format") == "ACH").cast(pl.Int8).alias("is_ach"),
        (pl.col("cur_paid") != pl.col("cur_recv")).cast(pl.Int8).alias("is_cross_currency"),
        (pl.col("from_bank") == pl.col("to_bank")).cast(pl.Int8).alias("is_same_bank"),
        (pl.col("src") == pl.col("dst")).cast(pl.Int8).alias("is_self_transfer"),
        ((pl.col("amt_paid") % 100) == 0).cast(pl.Int8).alias("is_round_amount"),
        pl.col("amt_paid").is_between(lo, hi).cast(pl.Int8).alias("in_amount_band"),
        pl.col("cur_paid").replace_strict(cur_map, default=-1, return_dtype=pl.Int16).alias("cur_code"),
        *fmt,
    )
