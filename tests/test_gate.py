import polars as pl
import pytest

from papertrail.lifecycle.config import GateTolerances
from papertrail.lifecycle.gate import check_point_in_time, evaluate_gate

TOL = GateTolerances(pr_auc=0.01, recall_at_1pct_fpr=0.02)
CUR = {"pr_auc": 0.50, "recall_at_1pct_fpr": 0.72}


def test_promotes_when_equal_or_better():
    d = evaluate_gate({"pr_auc": 0.51, "recall_at_1pct_fpr": 0.73}, CUR, TOL, True)
    assert d.promote and d.reasons == []


def test_within_tolerance_still_promotes():
    assert evaluate_gate({"pr_auc": 0.495, "recall_at_1pct_fpr": 0.71}, CUR, TOL, True).promote


@pytest.mark.parametrize("cand,metric", [({"pr_auc": 0.45, "recall_at_1pct_fpr": 0.72}, "pr_auc"),
                                         ({"pr_auc": 0.50, "recall_at_1pct_fpr": 0.60}, "recall_at_1pct_fpr")])
def test_rejects_regression_beyond_tolerance(cand, metric):
    d = evaluate_gate(cand, CUR, TOL, True)
    assert not d.promote and any(r.startswith(metric) for r in d.reasons)


def test_rejects_on_leakage_even_if_metrics_better():
    d = evaluate_gate({"pr_auc": 0.9, "recall_at_1pct_fpr": 0.9}, CUR, TOL, False, "feature x leaks")
    assert not d.promote and d.reasons == ["point_in_time_leakage: feature x leaks"]


def test_first_model_only_needs_leakage_check():
    assert evaluate_gate({"pr_auc": 0.0, "recall_at_1pct_fpr": 0.0}, None, TOL, True).promote
    assert not evaluate_gate({"pr_auc": 1.0, "recall_at_1pct_fpr": 1.0}, None, TOL, False).promote


def test_boundary_is_inclusive():
    assert evaluate_gate({"pr_auc": 0.49, "recall_at_1pct_fpr": 0.70}, CUR, GateTolerances(0.0100001, 0.0200001), True).promote


def _toy(n=300):
    import numpy as np
    rng = np.random.default_rng(1)
    ts = pl.datetime_range(pl.datetime(2022, 9, 1), pl.datetime(2022, 9, 8), "30m", eager=True)[:n]
    acc = [f"1_{i}" for i in range(12)]
    return pl.DataFrame({"txn_id": range(n), "ts": ts, "src": rng.choice(acc, n), "dst": rng.choice(acc, n),
                         "amt_paid": rng.uniform(10, 5000, n), "from_bank": 1, "to_bank": 1})


def test_point_in_time_passes_for_production_builder():
    from papertrail.graph_features import HISTORY_FEATURES
    ok, detail = check_point_in_time(_toy(), HISTORY_FEATURES)
    assert ok, detail


def test_point_in_time_catches_a_leaky_builder():
    """A 'feature' that counts ALL of the sender's transactions (including later ones) must be flagged."""
    def leaky(df):
        return df.with_columns(pl.len().over("src").cast(pl.Float64).alias("s_leak"))
    ok, detail = check_point_in_time(_toy(), ["s_leak"], builder=leaky)
    assert not ok and "s_leak" in detail
