import numpy as np
import polars as pl
import pytest

from app import logic

NAMES = ["log_amt", "hour", "is_ach", "is_cross_currency", "is_same_bank", "is_self_transfer", "is_round_amount", "in_amount_band",
         "cur_code", "fmt_ACH", "fmt_Wire", "s_out_cnt_24h", "s_out_cnt_7d", "s_tot_out", "s_uniq_out", "s_out_mean_7d",
         "s_out_sum_24h", "s_out_sum_7d", "d_out_cnt_24h"]
UNITS = {"log_amt": "amount", "hour": "hour", "s_out_sum_24h": "amount", "s_out_sum_7d": "amount", "s_out_mean_7d": "amount"}


def _bundle(boosters=None) -> logic.Bundle:
    labels = {n: {"label": f"label of {n}", "unit": UNITS.get(n, "flag" if n.startswith(("is_", "fmt_")) else "count")} for n in NAMES}
    profiles = {
        "profiles": {
            "new_account": {"description": "new", "n_rows": 10, "laundering_rate": 0.0,
                            "features": {"s_out_cnt_24h": 0.0, "s_out_cnt_7d": 0.0, "s_tot_out": 0.0, "s_uniq_out": 0.0, "s_out_mean_7d": None}},
            "typical_customer": {"description": "typical", "n_rows": 10, "laundering_rate": 0.001,
                                 "features": {"s_out_cnt_24h": 1.0, "s_out_cnt_7d": 3.0, "s_tot_out": 4.0, "s_uniq_out": 2.0, "s_out_mean_7d": 200.0,
                                              "s_out_sum_7d": 600.0}},
        },
        "receiver_default": {"d_out_cnt_24h": 2.0}, "sliders": {},
    }
    cats = {"pay_format": ["ACH", "Wire"], "currencies": {"Euro": 0, "US Dollar": 1}, "amount_band": [3000, 30000], "feature_names": NAMES}
    txns = pl.DataFrame({"txn_id": [1, 2, 3], "xgb_z": [0.0, 2.0, 4.0], "xgb_pct": [0.5, 0.9, 0.999], "src": ["a", "b", "a"], "dst": ["b", "c", "c"],
                         "ts": [1, 2, 3]})
    alerts = pl.DataFrame({"account": ["a", "b"], "xgb_risk": [4.0, 2.0], "xgb_rank": [1, 2], "n_flag_xgb": [2, 0], "n_txn": [5, 3], "why": ["w1", "w2"],
                           "n_laundering": [1, 0], "typologies_truth": ["FAN-OUT", ""], "tag": ["top risk", "false positive"]})
    shap = pl.DataFrame({"account": ["a", "a", "a"], "txn_id": [3, 3, 3], "feature": ["log_amt", "s_out_cnt_24h", "is_ach"],
                         "value": [np.log1p(5000.0), 14.0, 1.0], "contribution": [0.2, 1.5, -0.7]})
    meta = {"feature_names": NAMES, "threshold": 0.5, "files": []}
    return logic.Bundle(root=None, alerts=alerts, edges=pl.DataFrame({"center": [], "txn_id": [], "hop": []}), history=pl.DataFrame(),
                        shap=shap, txns=txns, truth=pl.DataFrame(), metrics={}, labels=labels, profiles=profiles, categories=cats, meta=meta,
                        _boosters=boosters or [])


def test_human_formats():
    assert logic.human(7200, "seconds") == "2.0 hours" and logic.human(90, "seconds") == "2 minutes"
    assert logic.human(1.0, "flag") == "yes" and logic.human(0.0, "flag") == "no"
    assert logic.human(None, "count") == "not available" and logic.human(float("nan"), "count") == "not available"
    assert logic.human(12345.6, "amount") == "12,346" and logic.human(9, "hour") == "09:00"


def test_risk_percent_text():
    assert logic.risk_percent_text(0.99) == "top 1%" and logic.risk_percent_text(0.9999) == "top 0.01%"
    assert logic.risk_percent_text(1.0) == "top 0.01%" and logic.risk_percent_text(None) == "n/a"


def test_risk_percentile_interpolates_and_is_monotone():
    b = _bundle()
    assert logic.risk_percentile(b, "xgb", 2.0) == pytest.approx(0.9)
    assert logic.risk_percentile(b, "xgb", 3.0) == pytest.approx(0.9495)
    assert logic.risk_percentile(b, "xgb", 99.0) == pytest.approx(0.999)       # beyond the stored range: clipped, not extrapolated


def test_describe_shows_the_amount_not_its_log():
    b = _bundle()
    assert logic.describe(b.labels, "log_amt", float(np.log1p(5000.0))) == ("Payment amount", "5,000")
    assert logic.describe(b.labels, "s_out_cnt_24h", 14.0) == ("label of s_out_cnt_24h", "14")


def test_alert_table_orders_by_model_rank_and_hides_the_answers_by_default():
    b = _bundle()
    t = logic.alert_table(b, "xgb")
    assert t["Account"].to_list() == ["a", "b"] and "True pattern" not in t.columns
    t2 = logic.alert_table(b, "xgb", show_truth=True)
    assert t2["True pattern"].to_list() == ["FAN-OUT", ""] and "Actually laundering (transactions)" in t2.columns


def test_account_reasons_sorted_by_strength_with_plain_labels():
    r = logic.account_reasons(_bundle(), "a")
    assert r["feature"].to_list() == ["s_out_cnt_24h", "is_ach", "log_amt"]
    assert r["direction"].to_list() == ["raises", "lowers", "raises"]
    assert r["value_text"][2] == "5,000"


def test_ring_layout_places_every_endpoint_on_a_ring():
    edges = pl.DataFrame({"src": ["c", "x", "y", "x"], "dst": ["x", "c", "x", "z"]})        # c<->x direct, y->x and x->z are second hop
    pos, rows = logic.ring_layout(edges, "c")
    assert pos["c"] == (0.0, 0.0, 0) and len(rows) == 4
    assert pos["x"][2] == 1 and np.hypot(pos["x"][0], pos["x"][1]) == pytest.approx(1.0)
    for n in ("y", "z"):
        assert pos[n][2] == 2 and np.hypot(pos[n][0], pos[n][1]) == pytest.approx(2.0)
    assert set(pos) == {"c", "x", "y", "z"}


def test_whatif_features_from_the_form():
    b = _bundle()
    f = logic.whatif_features(b, amount=5000.0, pay_format="ACH", pay_currency="Euro", different_receive_currency=True, same_bank=False, hour=14,
                              profile="typical_customer", out_cnt_24h=6, uniq_out=9)
    assert f["log_amt"] == pytest.approx(np.log1p(5000.0)) and f["is_ach"] == 1.0 and f["fmt_ACH"] == 1.0 and f["fmt_Wire"] == 0.0
    assert f["is_cross_currency"] == 1.0 and f["is_same_bank"] == 0.0 and f["is_self_transfer"] == 0.0
    assert f["is_round_amount"] == 1.0 and f["in_amount_band"] == 1.0 and f["cur_code"] == 0.0 and f["hour"] == 14.0
    assert f["d_out_cnt_24h"] == 2.0                                                    # receiver default
    # sliders override and keep the related counts consistent: 7-day >= 24-hour, total >= distinct recipients
    assert f["s_out_cnt_24h"] == 6.0 and f["s_out_cnt_7d"] == 6.0 and f["s_uniq_out"] == 9.0 and f["s_tot_out"] == 9.0
    assert f["s_out_sum_24h"] == pytest.approx(6 * 200.0)
    assert set(f) == set(NAMES)


def test_whatif_amount_flags():
    b = _bundle()
    f = logic.whatif_features(b, amount=250.5, pay_format="Wire", pay_currency="US Dollar", different_receive_currency=False, same_bank=True, hour=3,
                              profile="typical_customer", out_cnt_24h=0, uniq_out=0)
    assert f["is_round_amount"] == 0.0 and f["in_amount_band"] == 0.0 and f["is_ach"] == 0.0 and f["fmt_Wire"] == 1.0 and f["cur_code"] == 1.0


def test_new_account_profile_ignores_the_sliders():
    f = logic.whatif_features(_bundle(), amount=5000.0, pay_format="ACH", pay_currency="Euro", different_receive_currency=False, same_bank=False,
                              hour=10, profile="new_account", out_cnt_24h=25, uniq_out=30)
    assert f["s_out_cnt_24h"] == 0.0 and f["s_uniq_out"] == 0.0 and f["s_tot_out"] == 0.0


def test_score_features_with_a_real_booster():
    try:
        import xgboost as xgb
    except Exception:                                       # noqa: BLE001  (e.g. no OpenMP on a laptop)
        pytest.skip("xgboost is not usable here")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, len(NAMES))).astype(np.float32)
    y = (x[:, NAMES.index("s_out_cnt_24h")] + 0.3 * rng.normal(size=600) > 0.5).astype(int)
    boosters = [xgb.train({"objective": "binary:logistic", "max_depth": 3, "seed": s}, xgb.DMatrix(x, label=y, feature_names=NAMES), 25) for s in (0, 1)]
    b = _bundle(boosters)
    hi = logic.score_features(b, {**dict.fromkeys(NAMES, 0.0), "s_out_cnt_24h": 3.0})
    lo = logic.score_features(b, {**dict.fromkeys(NAMES, 0.0), "s_out_cnt_24h": -3.0})
    assert 0.0 <= lo["probability"] < hi["probability"] <= 1.0 and hi["alert"] and not lo["alert"]
    assert hi["reasons"]["feature"][0] == "s_out_cnt_24h" and hi["reasons"]["direction"][0] == "raises"
    assert hi["reasons"]["contribution"].abs().is_sorted(descending=True)


def test_queue_findings_count_flags_and_disagreements():
    b = _bundle()
    b.alerts = pl.DataFrame({"account": ["a", "b", "c", "d"], "blend_rank": [1, 2, 3, 4], "n_flag_blend": [3, 1, 0, 2], "n_flag_xgb": [2, 0, 0, 1],
                             "n_flag_gnn": [1, 1, 0, 0]})
    f = logic.queue_findings(b, "blend")
    assert f["accounts"] == 4 and f["flagged"] == {"blend": 3, "xgb": 2, "gnn": 2}
    assert f["every"] == 1 and f["gnn_only"] == 1 and f["xgb_only"] == 1 and f["top"] == "a"


def test_model_findings_pick_the_stronger_single_model():
    b = _bundle()
    b.metrics = {"models": {"xgb": {"recall_at_1pct_fpr": 0.66, "pr_auc": 0.44}, "gnn": {"recall_at_1pct_fpr": 0.68, "pr_auc": 0.28},
                            "blend": {"recall_at_1pct_fpr": 0.73, "pr_auc": 0.50}}}
    f = logic.model_findings(b)
    assert f["best_single"] == "gnn" and f["best_single_recall"] == 0.68 and f["best_single_pr"] == 0.44 and f["blend_recall"] == 0.73


def test_why_text_uses_words_values_and_signed_pushes():
    assert logic.why_text(_bundle(), "a") == "label of s_out_cnt_24h: 14 (+1.50); label of is_ach: yes (-0.70)"


def test_human_never_prints_scientific_notation():
    assert logic.human(1629000.0, "amount") == "1,629,000" and logic.human(1629000.0, "count") == "1,629,000" and logic.human(0.5, "score") == "0.5"
