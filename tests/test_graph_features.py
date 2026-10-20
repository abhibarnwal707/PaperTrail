from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from papertrail.graph_features import (
    HISTORY_FEATURES,
    RECENT_FEATURES,
    account_features,
    add_history_features,
    add_recent_features,
    daily_pagerank,
    pagerank,
    recent_features,
    reciprocal_flag,
)

T0 = datetime(2022, 9, 1)
SCHEMA = ["txn_id", "ts", "src", "dst", "amt_paid"]

# (hours since T0, src, dst, amount)
TOY = [
    (0, "a", "b", 100.0),
    (1, "b", "a", 50.0),
    (2, "a", "c", 10.0),
    (30, "a", "b", 20.0),
    (31, "c", "a", 5.0),
    (200, "a", "b", 1.0),
]


def _frame(rows, first_id=0) -> pl.DataFrame:
    return pl.DataFrame(
        [(first_id + i, T0 + timedelta(hours=h), s, d, a) for i, (h, s, d, a) in enumerate(rows)],
        schema=SCHEMA, orient="row",
    )


def _row(f: pl.DataFrame, txn_id: int) -> dict:
    return f.filter(pl.col("txn_id") == txn_id).to_dicts()[0]


def test_account_features_hand_computed():
    f = account_features(_frame(TOY))
    # txn 0: nothing has happened yet
    r = _row(f, 0)
    assert r["s_out_cnt_7d"] == 0 and r["s_tot_out"] == 0 and r["s_age_s"] == 0
    assert r["s_gap_s"] is None and r["s_out_mean_7d"] is None
    # txn 1: sender b only ever received 100 from a, one hour ago
    r = _row(f, 1)
    assert (r["s_in_cnt_24h"], r["s_in_sum_24h"], r["s_in_mean_7d"]) == (1, 100.0, 100.0)
    assert (r["s_out_cnt_24h"], r["s_uniq_in"], r["s_gap_s"], r["s_age_s"]) == (0, 1, 3600, 3600)
    # txn 2: sender a had paid 100 (out) and received 50 (in)
    r = _row(f, 2)
    assert (r["s_out_cnt_24h"], r["s_out_sum_24h"], r["s_in_cnt_24h"], r["s_in_sum_24h"]) == (1, 100.0, 1, 50.0)
    assert (r["s_uniq_out"], r["s_tot_out"], r["s_tot_in"], r["s_gap_s"], r["s_age_s"]) == (1, 1, 1, 3600, 7200)
    # txn 3 (30h): a's earlier events are all older than 24h but inside 7d
    r = _row(f, 3)
    assert (r["s_out_cnt_24h"], r["s_out_cnt_7d"], r["s_out_sum_7d"], r["s_out_mean_7d"]) == (0, 2, 110.0, 55.0)
    assert (r["s_uniq_out"], r["s_gap_s"], r["s_age_s"]) == (2, 28 * 3600, 30 * 3600)
    # txn 5 (200h): everything is older than 7d but still counted in all-time totals
    r = _row(f, 5)
    assert (r["s_out_cnt_7d"], r["s_in_cnt_7d"]) == (0, 0)
    assert (r["s_tot_out"], r["s_tot_in"], r["s_uniq_out"], r["s_uniq_in"]) == (3, 2, 2, 2)
    # receiver side: txn 4's receiver is a, who paid 20 in the last 24h and 130 in 7d
    r = _row(f, 4)
    assert (r["d_out_cnt_24h"], r["d_out_sum_24h"], r["d_out_cnt_7d"], r["d_out_sum_7d"]) == (1, 20.0, 3, 130.0)
    assert r["d_out_mean_7d"] == pytest.approx(130.0 / 3)


def test_account_features_do_not_use_the_future():
    base = _frame(TOY)
    later = _frame([(300 + i, "a", "b", 7.0) for i in range(5)] + [(400, "c", "a", 3.0)], first_id=100)
    extended = pl.concat([base, later])
    f_base = account_features(base)
    f_ext = account_features(extended).head(base.height)
    assert_frame_equal(f_base, f_ext)


def test_account_features_require_sorted_input():
    with pytest.raises(ValueError):
        account_features(_frame(TOY).reverse())


def test_reciprocal_flag():
    r = reciprocal_flag(_frame(TOY))["has_reverse_before"].to_list()
    # txn0 a->b: no b->a yet. txn1 b->a: a->b at 0h. txn2 a->c: no c->a yet.
    # txn3 a->b: b->a at 1h. txn4 c->a: a->c at 2h. txn5 a->b: yes.
    assert r == [0, 1, 0, 1, 1, 1]


def test_reciprocal_flag_ignores_same_timestamp_and_future():
    df = _frame([(0, "a", "b", 1.0), (0, "b", "a", 1.0), (5, "b", "a", 1.0)])
    assert reciprocal_flag(df)["has_reverse_before"].to_list() == [0, 0, 1]


def test_pagerank_cycle_is_uniform_and_star_favours_the_sink():
    cyc = pagerank(np.array([0, 1, 2]), np.array([1, 2, 0]), 3)
    assert cyc == pytest.approx([1 / 3] * 3)
    star = pagerank(np.array([0, 1, 2]), np.array([3, 3, 3]), 4)  # three nodes pay node 3
    assert star.sum() == pytest.approx(1.0)
    assert star.argmax() == 3 and star[3] > star[0]


def test_daily_pagerank_only_sees_earlier_days():
    df = _frame(TOY)  # days: Sep 1 (txn 0-2), Sep 2 (txn 3-4), Sep 9 (txn 5)
    pr = daily_pagerank(df)
    assert pr["s_pagerank"][:3].null_count() == 3          # day 1 has no history
    assert pr["s_pagerank"][3:].null_count() == 0
    # Adding more transactions on a LATER day must not change earlier rows.
    later = _frame([(24 * 20, "c", "b", 5.0), (24 * 20 + 1, "b", "c", 5.0)], first_id=50)
    pr2 = daily_pagerank(pl.concat([df, later])).head(df.height)
    assert_frame_equal(pr, pr2)


def test_daily_pagerank_same_day_edges_are_not_used():
    # Two days. Day-2 rows must be scored from day-1 edges only, so changing a day-2 edge changes nothing.
    d1 = [(0, "a", "b", 1.0), (1, "b", "c", 1.0)]
    a = _frame(d1 + [(25, "a", "c", 1.0), (26, "c", "a", 1.0)])
    b = _frame(d1 + [(25, "a", "c", 1.0), (26, "a", "b", 1.0)])
    pa, pb = daily_pagerank(a), daily_pagerank(b)
    assert pa["s_pagerank"][2] == pytest.approx(pb["s_pagerank"][2])
    assert pa["s_pagerank"][2] is not None


def test_add_history_features_keeps_rows_and_adds_all_columns():
    df = _frame(TOY).with_columns(label=pl.lit(0))
    out = add_history_features(df)
    assert out.height == df.height
    assert out["txn_id"].to_list() == df["txn_id"].to_list()
    assert all(c in out.columns for c in HISTORY_FEATURES)
    assert len(HISTORY_FEATURES) == len(set(HISTORY_FEATURES)) == 35


def test_recent_features_hand_computed():
    f = recent_features(_frame(TOY))
    # txn 2 (2h), sender a: paid 100 at 0h (2h ago), received 50 at 1h (exactly 1h ago, inside [t-1h, t))
    r = _row(f, 2)
    assert (r["s_out_cnt_1h"], r["s_in_cnt_1h"], r["s_in_sum_1h"]) == (0, 1, 50.0)
    assert (r["s_out_cnt_6h"], r["s_out_sum_6h"], r["s_in_cnt_6h"]) == (1, 100.0, 1)
    assert (r["s_out_mean_24h"], r["s_in_mean_24h"], r["s_gap_s"]) == (100.0, 50.0, 3600)
    # txn 3 (30h): everything a did is 28h or more ago, so all windows are empty and the gap is 28h
    r = _row(f, 3)
    assert (r["s_out_cnt_24h"], r["s_in_cnt_24h"], r["s_out_mean_24h"], r["s_gap_s"]) == (0, 0, None, 28 * 3600)
    # txn 4 (31h): sender c last acted 29h ago; receiver a paid 20 an hour ago
    r = _row(f, 4)
    assert (r["s_out_cnt_24h"], r["s_in_cnt_24h"]) == (0, 0)
    assert (r["d_out_cnt_1h"], r["d_out_sum_1h"], r["d_out_cnt_24h"], r["d_out_mean_24h"], r["d_gap_s"]) == (1, 20.0, 1, 20.0, 3600)
    # first event of an account: nothing before it
    r = _row(f, 0)
    assert r["s_gap_s"] is None and r["s_out_cnt_24h"] == 0


def test_recent_features_do_not_use_the_future():
    base = _frame(TOY)
    later = _frame([(300 + i, "a", "b", 7.0) for i in range(5)] + [(400, "c", "a", 3.0)], first_id=100)
    assert_frame_equal(recent_features(base), recent_features(pl.concat([base, later])).head(base.height))


def test_recent_features_are_calendar_stable():
    # The same burst of activity a week later gives identical features: nothing counts "time since the data began".
    burst = [(0, "a", "b", 10.0), (1, "a", "c", 20.0), (2, "a", "d", 30.0)]
    early = recent_features(_frame(burst))
    late = recent_features(_frame([(h + 7 * 24, s, d, a) for h, s, d, a in burst]))
    assert_frame_equal(early, late)


def test_recent_features_require_sorted_input_and_add_all_columns():
    with pytest.raises(ValueError):
        recent_features(_frame(TOY).reverse())
    out = add_recent_features(_frame(TOY))
    assert out.height == len(TOY) and all(c in out.columns for c in RECENT_FEATURES)
    assert len(RECENT_FEATURES) == len(set(RECENT_FEATURES)) == 30


def test_recent_features_match_history_features_where_they_overlap():
    df = _frame(TOY)
    overlap = [c for c in RECENT_FEATURES if c in HISTORY_FEATURES]
    assert len(overlap) == 10          # 24 h counts and sums (4) plus gap_s, on both sides
    both = account_features(df).join(recent_features(df), on="txn_id", suffix="_recent")
    for c in overlap:
        assert both[c].to_list() == both[c + "_recent"].to_list(), c


def test_add_recent_features_does_not_duplicate_columns():
    out = add_recent_features(add_history_features(_frame(TOY).with_columns(label=pl.lit(0))))
    assert len(out.columns) == len(set(out.columns))
    assert all(c in out.columns for c in RECENT_FEATURES + HISTORY_FEATURES)
