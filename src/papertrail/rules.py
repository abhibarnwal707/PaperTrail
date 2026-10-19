"""Rule-based baseline: what a bank's transaction-monitoring rules look like today.

Thresholds are learned on the training split only. A transaction's score is the number of rules it
triggers; self-transfers are suppressed (they are almost never laundering). Ties are broken by amount,
the way an analyst would review the biggest alerts first. Expects an `out_count_24h` column
(see `features.add_out_count_24h`).
"""
import numpy as np
import polars as pl

from papertrail.features import AMOUNT_BAND

BURST_GRID = (5, 10, 20, 50, 100, 200, 500, 1000, 5000)


def fit_rule_params(train: pl.DataFrame, q: float = 0.99, burst: float | None = None) -> dict[str, float]:
    """Rule thresholds from the training split. `burst` overrides the send-count threshold (rules_v2)."""
    return {
        "amt_high": float(train["amt_paid"].quantile(q)),
        "burst": float(train["out_count_24h"].quantile(q)) if burst is None else float(burst),
    }


def fit_burst_threshold(
    train: pl.DataFrame, grid=BURST_GRID, min_flagged: int = 1000
) -> tuple[float, pl.DataFrame]:
    """Pick the send-count threshold N (rule: out_count_24h > N) from training data only.

    The v1 threshold (99th percentile over all rows) was set by a handful of hub accounts and never fired.
    Here we try a grid of N values and keep the one with the highest training precision lift
    (precision / base rate) among those that flag at least `min_flagged` rows. Returns (N, table of candidates).
    """
    y = train["label"].to_numpy()
    n24 = train["out_count_24h"].to_numpy()
    base = float(y.mean())
    rows = []
    for n in grid:
        fired = n24 > n
        flagged = int(fired.sum())
        tp = int((fired & (y == 1)).sum())
        precision = tp / flagged if flagged else 0.0
        rows.append({"burst": float(n), "flagged": flagged, "tp": tp,
                     "precision": precision, "lift": precision / base if base else 0.0})
    table = pl.DataFrame(rows)
    ok = [r for r in rows if r["flagged"] >= min_flagged]
    if not ok:
        raise ValueError(f"no candidate threshold flags at least {min_flagged} training rows")
    best = max(ok, key=lambda r: r["lift"])  # ties keep the smallest N
    return best["burst"], table


def fit_rule_params_v2(train: pl.DataFrame, q: float = 0.99) -> tuple[dict[str, float], pl.DataFrame]:
    """rules_v2: same rules, with a send-count threshold that actually fires. Returns (params, candidates)."""
    burst, table = fit_burst_threshold(train)
    return fit_rule_params(train, q=q, burst=burst), table


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
