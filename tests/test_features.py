from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from papertrail.features import add_out_count_24h, build_features, fit_categories
from papertrail.rules import fit_rule_params, rule_flags, rule_score

T0 = datetime(2022, 9, 1)


def _txns() -> pl.DataFrame:
    rows = [
        # ts offset (h), src, dst, from_bank, to_bank, amt_paid, cur_paid, cur_recv, pay_format
        (0, "1_a", "2_b", 1, 2, 5000.0, "Euro", "Euro", "ACH"),
        (1, "1_a", "2_c", 1, 2, 250.5, "Euro", "Euro", "Cheque"),
        (1, "3_d", "3_d", 3, 3, 100.0, "US Dollar", "Euro", "Reinvestment"),
        (25, "1_a", "2_b", 1, 2, 40000.0, "Euro", "Euro", "ACH"),
        (26, "1_a", "9_z", 1, 9, 7.0, "Yen", "Yen", "Wire"),
    ]
    return pl.DataFrame(
        [(i, T0 + timedelta(hours=r[0]), *r[1:]) for i, r in enumerate(rows)],
        schema=["txn_id", "ts", "src", "dst", "from_bank", "to_bank",
                "amt_paid", "cur_paid", "cur_recv", "pay_format"],
        orient="row",
    )


def test_out_count_24h_hand_computed():
    out = add_out_count_24h(_txns())["out_count_24h"].to_list()
    # txn0: nothing before. txn1: txn0 (1h ago). txn2: another sender. txn3 (25h): txn0 is 25h old
    # and out, txn1 is 24h old and in. txn4 (26h): txn3 only (txn1 is 25h old).
    assert out == [0, 1, 0, 1, 1]


def test_out_count_24h_never_uses_the_future():
    df = _txns()
    base = add_out_count_24h(df)["out_count_24h"].to_list()
    # Appending later transactions must not change any earlier row's value.
    later = pl.DataFrame(
        [(10 + i, T0 + timedelta(hours=100 + i), "1_a", "2_b", 1, 2, 1.0, "Euro", "Euro", "ACH")
         for i in range(5)],
        schema=df.schema, orient="row",
    )
    extended = add_out_count_24h(pl.concat([df, later]))["out_count_24h"].to_list()
    assert extended[: df.height] == base


def test_out_count_24h_requires_sorted_input():
    with pytest.raises(ValueError):
        add_out_count_24h(_txns().reverse())


def test_build_features_values():
    df = _txns()
    cats = fit_categories(df)
    f = build_features(df, cats)
    assert f["log_amt"][0] == pytest.approx(np.log1p(5000.0))
    assert f["is_ach"].to_list() == [1, 0, 0, 1, 0]
    assert f["is_cross_currency"].to_list() == [0, 0, 1, 0, 0]
    assert f["is_same_bank"].to_list() == [0, 0, 1, 0, 0]
    assert f["is_self_transfer"].to_list() == [0, 0, 1, 0, 0]
    assert f["is_round_amount"].to_list() == [1, 0, 1, 1, 0]
    assert f["in_amount_band"].to_list() == [1, 0, 0, 0, 0]  # 40,000 is above the band
    assert set(f["hour"].to_list()) == {0, 1, 2}
    assert "fmt_ACH" in f.columns and "fmt_Reinvestment" in f.columns
    assert f.height == df.height


def test_build_features_unseen_category_is_safe():
    train, other = _txns().head(2), _txns().tail(1)  # "Yen" and "Wire" not in train
    cats = fit_categories(train)
    f = build_features(other, cats)
    assert f["cur_code"][0] == -1
    assert all(f[c][0] == 0 for c in f.columns if c.startswith("fmt_"))


def test_rules_fire_and_suppress_self_transfers():
    df = add_out_count_24h(_txns())
    params = {"amt_high": 30_000.0, "burst": 0.0}
    flags = rule_flags(df, params)
    assert flags["r_high_amount"].to_list() == [False, False, False, True, False]
    assert flags["r_ach_band"].to_list() == [True, False, False, False, False]
    assert flags["r_burst"].to_list() == [False, True, False, True, True]
    score = rule_score(df, params)
    assert int(score[0]) == 1 and int(score[1]) == 1 and int(score[3]) == 2
    assert score[2] < 1  # self-transfer: rules suppressed, only the tie-breaker remains
    assert score.max() < 3.3


def test_fit_rule_params_uses_given_frame_only():
    df = add_out_count_24h(_txns())
    p = fit_rule_params(df, q=1.0)
    assert p["amt_high"] == 40000.0


def test_fit_burst_threshold_picks_best_lift_with_enough_support():
    from papertrail.rules import fit_burst_threshold

    # 2000 quiet rows, 1500 rows with 30 sends (10% laundering), 1200 rows with 600 sends (1% laundering)
    label = [0] * 2000 + [1] * 150 + [0] * 1350 + [1] * 12 + [0] * 1188
    n24 = [0] * 2000 + [30] * 1500 + [600] * 1200
    train = pl.DataFrame({"label": label, "out_count_24h": n24})
    n, table = fit_burst_threshold(train, grid=(5, 10, 50, 100, 1000), min_flagged=1000)
    assert n == 5                                  # flags 2700 rows at the highest lift; ties keep the smallest N
    row = table.filter(pl.col("burst") == 5).to_dicts()[0]
    assert row["flagged"] == 2700 and row["tp"] == 162
    assert table.filter(pl.col("burst") == 1000)["flagged"][0] == 0   # a dead threshold, never chosen


def test_fit_burst_threshold_needs_support():
    from papertrail.rules import fit_burst_threshold

    train = pl.DataFrame({"label": [0, 1] * 10, "out_count_24h": list(range(20))})
    with pytest.raises(ValueError):
        fit_burst_threshold(train, min_flagged=1000)


def test_fit_rule_params_burst_override():
    df = add_out_count_24h(_txns())
    assert fit_rule_params(df, burst=42)["burst"] == 42.0
