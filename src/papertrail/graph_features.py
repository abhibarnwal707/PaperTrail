"""Leakage-safe account-history and graph features (Phase 4).

Golden rule: a transaction's features may only use transactions that happened strictly before it.
Every feature here is computed "as of" the transaction's timestamp, or from a snapshot of the graph
taken at the start of the previous day. That keeps train, validation and test features distributed
identically, unlike a snapshot built from the whole training window (which would let a training row
see its own and later edges).

Account view: each transaction creates two events, one for the sender (direction "out") and one for the
receiver (direction "in"). Features describe the account's past events, and each transaction gets the
sender's (`s_*`) and the receiver's (`d_*`) values.
"""
import numpy as np
import polars as pl

WINDOWS = ("24h", "7d")

ACCOUNT_FEATURES = [
    "out_cnt_24h", "out_cnt_7d", "in_cnt_24h", "in_cnt_7d",
    "out_sum_24h", "out_sum_7d", "in_sum_24h", "in_sum_7d",
    "out_mean_7d", "in_mean_7d",
    "uniq_out", "uniq_in",          # distinct counterparties paid / received from so far (all time)
    "tot_out", "tot_in",            # all-time transaction counts so far
    "gap_s", "age_s",               # seconds since the account's previous event / first event
]
HISTORY_FEATURES = (
    [f"s_{c}" for c in ACCOUNT_FEATURES]
    + [f"d_{c}" for c in ACCOUNT_FEATURES]
    + ["has_reverse_before", "s_pagerank", "d_pagerank"]
)


# "Stationary" set: windows of at most 24 h plus the gap since the last event. Windows longer than the history we
# have (7 d, all-time totals, account age, distinct counterparties so far, PageRank) change meaning as the calendar
# moves: on day 3 a "7-day" window covers 2 days, on day 9 it covers 7. These short windows mean the same thing on
# every day from day 2 on, so train, validation and test see the same feature distribution.
RECENT_WINDOWS = ("1h", "6h", "24h")
RECENT_ACCOUNT_FEATURES = (
    [f"{k}_{w}" for w in RECENT_WINDOWS for k in ("out_cnt", "in_cnt", "out_sum", "in_sum")]
    + ["out_mean_24h", "in_mean_24h", "gap_s"]
)
RECENT_FEATURES = [f"s_{c}" for c in RECENT_ACCOUNT_FEATURES] + [f"d_{c}" for c in RECENT_ACCOUNT_FEATURES]


def _check_sorted(df: pl.DataFrame, ts_col: str = "ts") -> None:
    if not df[ts_col].is_sorted():
        raise ValueError(f"df must be sorted by {ts_col}")


def _event_table(df: pl.DataFrame) -> pl.DataFrame:
    out = df.select(
        "txn_id", "ts",
        pl.col("src").alias("acct"), pl.col("dst").alias("cp"),
        pl.lit(0, dtype=pl.Int8).alias("dir"), pl.col("amt_paid").alias("amt"),
    )
    inn = df.select(
        "txn_id", "ts",
        pl.col("dst").alias("acct"), pl.col("src").alias("cp"),
        pl.lit(1, dtype=pl.Int8).alias("dir"), pl.col("amt_paid").alias("amt"),
    )
    return pl.concat([out, inn]).sort(["ts", "txn_id", "dir"])


def account_features(df: pl.DataFrame) -> pl.DataFrame:
    """Sender (`s_*`) and receiver (`d_*`) history features for every transaction, keyed by `txn_id`.

    Windows are [t - w, t): the current transaction and anything at the same timestamp are excluded.
    `df` must have columns txn_id, ts, src, dst, amt_paid and be sorted by ts.
    """
    _check_sorted(df)
    ev = _event_table(df)
    is_out = pl.col("dir") == 0

    ev = ev.with_columns(
        is_out.cast(pl.UInt32).alias("c_out"),
        (~is_out).cast(pl.UInt32).alias("c_in"),
        pl.when(is_out).then(pl.col("amt")).otherwise(0.0).alias("a_out"),
        pl.when(~is_out).then(pl.col("amt")).otherwise(0.0).alias("a_in"),
        pl.struct(["acct", "cp", "dir"]).is_first_distinct().alias("_first"),
    )
    rolling = [
        pl.col(col).rolling_sum_by("ts", w, closed="left").over("acct").fill_null(0).alias(f"{name}_{w}")
        for w in WINDOWS
        for col, name in (("c_out", "out_cnt"), ("c_in", "in_cnt"), ("a_out", "out_sum"), ("a_in", "in_sum"))
    ]
    new_out = (pl.col("_first") & is_out).cast(pl.UInt32)
    new_in = (pl.col("_first") & ~is_out).cast(pl.UInt32)
    ev = ev.with_columns(
        *rolling,
        (new_out.cum_sum().over("acct") - new_out).alias("uniq_out"),
        (new_in.cum_sum().over("acct") - new_in).alias("uniq_in"),
        (pl.col("c_out").cum_sum().over("acct") - pl.col("c_out")).alias("tot_out"),
        (pl.col("c_in").cum_sum().over("acct") - pl.col("c_in")).alias("tot_in"),
        (pl.col("ts") - pl.col("ts").shift(1).over("acct")).dt.total_seconds().alias("gap_s"),
        (pl.col("ts") - pl.col("ts").first().over("acct")).dt.total_seconds().alias("age_s"),
    ).with_columns(
        pl.when(pl.col("out_cnt_7d") > 0).then(pl.col("out_sum_7d") / pl.col("out_cnt_7d")).alias("out_mean_7d"),
        pl.when(pl.col("in_cnt_7d") > 0).then(pl.col("in_sum_7d") / pl.col("in_cnt_7d")).alias("in_mean_7d"),
    )

    def side(direction: int, prefix: str) -> pl.DataFrame:
        return ev.filter(pl.col("dir") == direction).select(
            "txn_id", *[pl.col(c).alias(prefix + c) for c in ACCOUNT_FEATURES]
        )

    return (df.select("txn_id")
              .join(side(0, "s_"), on="txn_id", how="left")
              .join(side(1, "d_"), on="txn_id", how="left"))


def recent_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calendar-stable sender (`s_*`) and receiver (`d_*`) features, keyed by `txn_id`.

    Counts and sums of the account's earlier outgoing / incoming events in [t - w, t) for w in 1h, 6h, 24h, the 24h
    mean amounts, and `gap_s` (seconds since the account's previous event, null if it has none). Nothing here
    depends on how long the dataset has been running, apart from the first 24 h where windows are not yet full.
    """
    _check_sorted(df)
    ev = _event_table(df)
    is_out = pl.col("dir") == 0
    ev = ev.with_columns(
        is_out.cast(pl.UInt32).alias("c_out"),
        (~is_out).cast(pl.UInt32).alias("c_in"),
        pl.when(is_out).then(pl.col("amt")).otherwise(0.0).alias("a_out"),
        pl.when(~is_out).then(pl.col("amt")).otherwise(0.0).alias("a_in"),
    )
    ev = ev.with_columns(
        *[pl.col(col).rolling_sum_by("ts", w, closed="left").over("acct").fill_null(0).alias(f"{name}_{w}")
          for w in RECENT_WINDOWS
          for col, name in (("c_out", "out_cnt"), ("c_in", "in_cnt"), ("a_out", "out_sum"), ("a_in", "in_sum"))],
        (pl.col("ts") - pl.col("ts").shift(1).over("acct")).dt.total_seconds().alias("gap_s"),
    ).with_columns(
        pl.when(pl.col("out_cnt_24h") > 0).then(pl.col("out_sum_24h") / pl.col("out_cnt_24h")).alias("out_mean_24h"),
        pl.when(pl.col("in_cnt_24h") > 0).then(pl.col("in_sum_24h") / pl.col("in_cnt_24h")).alias("in_mean_24h"),
    )

    def side(direction: int, prefix: str) -> pl.DataFrame:
        return ev.filter(pl.col("dir") == direction).select(
            "txn_id", *[pl.col(c).alias(prefix + c) for c in RECENT_ACCOUNT_FEATURES]
        )

    return (df.select("txn_id")
              .join(side(0, "s_"), on="txn_id", how="left")
              .join(side(1, "d_"), on="txn_id", how="left"))


def add_recent_features(df: pl.DataFrame) -> pl.DataFrame:
    """`df` with all RECENT_FEATURES present (same rows, same order). `df` must be sorted by ts.

    Some recent features (the 24 h counts/sums and `gap_s`) have the same definition as columns of
    `add_history_features`; those are not added a second time.
    """
    feats = recent_features(df)
    return df.join(feats.select("txn_id", *[c for c in RECENT_FEATURES if c not in df.columns]), on="txn_id", how="left")


def reciprocal_flag(df: pl.DataFrame) -> pl.DataFrame:
    """`has_reverse_before` = 1 if a transfer in the opposite direction (dst -> src) happened strictly earlier."""
    _check_sorted(df)
    first_ts = (df.group_by(["src", "dst"]).agg(pl.col("ts").min().alias("_rev_first"))
                  .rename({"src": "dst", "dst": "src"}))  # keyed so that (src, dst) finds the reverse pair
    return (df.select("txn_id", "src", "dst", "ts")
              .join(first_ts, on=["src", "dst"], how="left")
              .select("txn_id",
                      (pl.col("_rev_first") < pl.col("ts")).fill_null(False).cast(pl.Int8)
                      .alias("has_reverse_before")))


def pagerank(src_idx, dst_idx, n: int, damping: float = 0.85, iters: int = 50, tol: float = 1e-10) -> np.ndarray:
    """PageRank by power iteration on a sparse matrix. Repeated edges count as weights; sums to 1."""
    import scipy.sparse as sp

    a = sp.csr_matrix((np.ones(len(src_idx)), (src_idx, dst_idx)), shape=(n, n))
    out = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1.0, out, out=np.zeros_like(out), where=out > 0)
    pt = (sp.diags(inv) @ a).T.tocsr()
    dangling = out == 0
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        new = damping * (pt @ r + r[dangling].sum() / n) + (1.0 - damping) / n
        done = np.abs(new - r).sum() < tol
        r = new
        if done:
            break
    return r


def daily_pagerank(df: pl.DataFrame) -> pl.DataFrame:
    """Sender/receiver PageRank from a snapshot of all edges before the start of the transaction's day.

    PageRank is expensive and not additive over time, so it is recomputed once per day. A transaction on
    day d sees the graph of days < d, never its own day. Day-1 rows get null. Scores are scaled by the number
    of accounts, so 1.0 means "average".
    """
    _check_sorted(df)
    ids = (pl.concat([df["src"], df["dst"]]).unique().to_frame("acct").with_row_index("id"))
    n = ids.height
    s_id = df.select("src").join(ids.rename({"acct": "src", "id": "sid"}), on="src", how="left")["sid"].to_numpy()
    d_id = df.select("dst").join(ids.rename({"acct": "dst", "id": "did"}), on="dst", how="left")["did"].to_numpy()
    day = df.select(pl.col("ts").dt.truncate("1d").rank("dense").alias("d"))["d"].to_numpy()

    s_pr = np.full(df.height, np.nan)
    d_pr = np.full(df.height, np.nan)
    for d in range(2, int(day.max()) + 1):
        before = day < d
        r = pagerank(s_id[before], d_id[before], n) * n
        now = day == d
        s_pr[now] = r[s_id[now]]
        d_pr[now] = r[d_id[now]]
    return pl.DataFrame({
        "txn_id": df["txn_id"],
        "s_pagerank": pl.Series(s_pr).fill_nan(None),
        "d_pagerank": pl.Series(d_pr).fill_nan(None),
    })


def add_history_features(df: pl.DataFrame, with_pagerank: bool = True) -> pl.DataFrame:
    """`df` with all HISTORY_FEATURES appended (same rows, same order). `df` must be sorted by ts."""
    feats = account_features(df).join(reciprocal_flag(df), on="txn_id", how="left")
    if with_pagerank:
        feats = feats.join(daily_pagerank(df), on="txn_id", how="left")
    return df.join(feats, on="txn_id", how="left")
