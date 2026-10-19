"""Rule-based baseline: what a bank's transaction-monitoring rules look like today.

Thresholds are learned on the training split only. A transaction's score is the number of rules it
triggers; self-transfers are suppressed (they are almost never laundering). Ties are broken by amount,
the way an analyst would review the biggest alerts first. Expects an `out_count_24h` column
(see `features.add_out_count_24h`).
"""
import numpy as np
import polars as pl

from papertrail.features import AMOUNT_BAND


def fit_rule_params(train: pl.DataFrame, q: float = 0.99) -> dict[str, float]:
    return {
        "amt_high": float(train["amt_paid"].quantile(q)),
        "burst": float(train["out_count_24h"].quantile(q)),
    }


def rule_flags(df: pl.DataFrame, params: dict[str, float]) -> pl.DataFrame:
    lo, hi = AMOUNT_BAND
    return df.select(
        (pl.col("amt_paid") > params["amt_high"]).alias("r_high_amount"),
        ((pl.col("pay_format") == "ACH") & pl.col("amt_paid").is_between(lo, hi)).alias("r_ach_band"),
        (pl.col("out_count_24h") > params["burst"]).alias("r_burst"),
        (pl.col("src") == pl.col("dst")).alias("is_self_transfer"),
    )


def rule_score(df: pl.DataFrame, params: dict[str, float]) -> np.ndarray:
    """Score in [0, 3.3): rules triggered (0 for self-transfers) plus a small amount-based tie-breaker."""
    f = rule_flags(df, params)
    n_rules = (
        f["r_high_amount"].cast(pl.Int8) + f["r_ach_band"].cast(pl.Int8) + f["r_burst"].cast(pl.Int8)
    ).to_numpy()
    not_self = ~f["is_self_transfer"].to_numpy()
    tiebreak = df["amt_paid"].log1p().to_numpy() / 100.0  # log1p(amount) stays well below 100
    return n_rules * not_self + tiebreak
