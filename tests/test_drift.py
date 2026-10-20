import numpy as np
import polars as pl
import pytest

from papertrail.drift import (
    band,
    daily_psi,
    drift_table,
    is_calendar_dependent,
    psi,
    retrain_decision,
)

rng = np.random.default_rng(0)


def test_psi_is_near_zero_for_the_same_distribution():
    a, b = rng.normal(size=50_000), rng.normal(size=50_000)
    assert psi(a, b) < 0.01


def test_psi_grows_with_the_shift():
    a = rng.normal(size=50_000)
    small, large = psi(a, rng.normal(0.2, 1, 50_000)), psi(a, rng.normal(1.0, 1, 50_000))
    assert 0.01 < small < 0.1
    assert large > 0.25


def test_psi_counts_missing_values_as_their_own_bin():
    a = rng.normal(size=10_000)
    b = a.copy()
    b[: len(b) // 2] = np.nan                       # half the current values go missing
    assert psi(a, b) > 0.25


def test_psi_handles_constant_and_binary_features():
    assert psi(np.zeros(1000), np.zeros(1000)) < 1e-6
    flags = np.r_[np.zeros(900), np.ones(100)]
    assert psi(flags, flags) < 1e-6
    assert psi(flags, np.r_[np.zeros(500), np.ones(500)]) > 0.25


def test_psi_rejects_empty_input():
    with pytest.raises(ValueError):
        psi([], [1.0])


def test_band_edges():
    assert (band(0.05), band(0.1), band(0.2), band(0.25)) == ("stable", "moderate", "moderate", "major")


def test_calendar_dependent_features_follow_the_eda_finding():
    for f in ("s_out_cnt_7d", "d_in_sum_7d", "s_tot_out", "d_uniq_in", "s_pagerank", "s_age_s", "has_reverse_before"):
        assert is_calendar_dependent(f), f
    for f in ("log_amt", "is_ach", "s_out_cnt_24h", "d_in_sum_24h", "s_gap_s", "hour"):
        assert not is_calendar_dependent(f), f


def _frames():
    n = 20_000
    ref = pl.DataFrame({"is_ach": rng.integers(0, 2, n), "s_out_cnt_24h": rng.poisson(3, n), "s_tot_out": rng.poisson(5, n),
                        "log_amt": rng.normal(8, 1, n)})
    cur = pl.DataFrame({"is_ach": rng.integers(0, 2, n), "s_out_cnt_24h": rng.poisson(3, n), "s_tot_out": rng.poisson(40, n),
                        "log_amt": rng.normal(9.5, 1, n)})
    return ref, cur


def test_drift_table_sorts_by_importance_and_marks_calendar_features():
    ref, cur = _frames()
    t = drift_table(ref, cur, ref.columns, importance={"log_amt": 0.5, "s_tot_out": 0.3, "is_ach": 0.2})
    assert t["feature"].to_list()[:3] == ["log_amt", "s_tot_out", "is_ach"]
    row = {r["feature"]: r for r in t.to_dicts()}
    assert row["s_tot_out"]["calendar_dependent"] and row["s_tot_out"]["band"] == "major"
    assert row["log_amt"]["band"] == "major" and row["is_ach"]["band"] == "stable"


def test_retrain_rule_ignores_calendar_drift():
    ref, cur = _frames()
    t = drift_table(ref, cur, ref.columns, importance={"log_amt": 0.5, "s_tot_out": 0.3, "is_ach": 0.2, "s_out_cnt_24h": 0.1})
    d = retrain_decision(t, count_limit=1)
    assert d["retrain"] and d["drifted"] == ["log_amt"] and d["expected_calendar_drift"] == ["s_tot_out"]
    assert not retrain_decision(t, count_limit=2)["retrain"]


def test_daily_psi_has_one_row_per_day_and_feature():
    ref, cur = _frames()
    cur = cur.with_columns(pl.Series("day", ["d1", "d2"] * (cur.height // 2)))
    out = daily_psi(ref, cur, ["is_ach", "log_amt"])
    assert out.height == 4 and set(out["day"]) == {"d1", "d2"}


def test_drift_page_renders_the_verdict_tables_and_links():
    from papertrail.drift_page import render

    ref, cur = _frames()
    t = drift_table(ref, cur, ref.columns, importance={"log_amt": 0.5, "s_tot_out": 0.3})
    daily = daily_psi(ref, cur.with_columns(pl.Series("day", ["09-09", "09-10"] * (cur.height // 2))), ["log_amt", "s_tot_out"])
    summary = {"generated": "2026-10-20", "reference": "train", "current": "test", "decision": retrain_decision(t, count_limit=1),
               "score_psi": {"xgb": 0.01, "gnn": 0.02, "blend": 0.015},
               "alert_rate": [{"window": "validation", "xgb": 0.001, "gnn": 0.002, "blend": 0.0015}],
               "perf_by_day": [{"day": "09-09", "n": 400000, "positives": 500, "alert_rate": 0.001, "pr_auc": 0.5}],
               "links": {"Full feature report": "feature_drift.html"}}
    page = render(summary, t, daily, labels={"log_amt": "Payment amount <log>"})
    assert page.startswith("<!doctype html>") and "Retrain recommended" in page
    assert "Payment amount &lt;log&gt;" in page and "feature_drift.html" in page and "09-10" in page
