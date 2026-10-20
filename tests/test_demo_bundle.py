"""Integrity checks for the Phase 6 demo bundle in data/demo/. Skipped until the bundle has been downloaded and unzipped there."""
import json
import os
from pathlib import Path

import polars as pl
import pytest

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"
pytestmark = pytest.mark.skipif(not (DEMO / "alerts.parquet").exists(), reason="demo bundle not installed in data/demo/")

FILES = ["alerts.parquet", "edges.parquet", "history.parquet", "shap.parquet", "txns.parquet", "truth.parquet", "metrics.json",
         "blend.json", "feature_labels.json", "profiles.json", "categories.json", "model/meta.json"]


def test_all_files_present_and_total_size_under_50mb():
    for f in FILES:
        assert (DEMO / f).exists(), f
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(DEMO) for f in fs)
    assert total < 50e6


def test_alerts_are_ranked_and_explained():
    a = pl.read_parquet(DEMO / "alerts.parquet")
    assert 100 <= a.height <= 200 and a["account"].n_unique() == a.height
    for m in ("blend", "xgb", "gnn"):
        assert sorted(a[f"{m}_rank"].to_list()) == list(range(1, a.height + 1))
    assert a["why"].null_count() == 0 and (a["why"].str.len_chars() > 0).all()


def test_evidence_tables_are_consistent_with_each_other():
    a, e, h = (pl.read_parquet(DEMO / f"{n}.parquet") for n in ("alerts", "edges", "history"))
    s, t = pl.read_parquet(DEMO / "shap.parquet"), pl.read_parquet(DEMO / "txns.parquet")
    accounts = set(a["account"])
    assert set(e["center"]) <= accounts and set(h["account"]) <= accounts and set(s["account"]) <= accounts
    assert e.group_by("center").len()["len"].max() <= 150 and h.group_by("account").len()["len"].max() <= 50
    assert s.group_by("account").len()["len"].max() <= 10
    known = set(t["txn_id"])
    assert set(e["txn_id"]) <= known and set(h["txn_id"]) <= known and set(s["txn_id"]) <= known


def test_model_meta_matches_labels_profiles_and_booster_files():
    meta = json.loads((DEMO / "model" / "meta.json").read_text())
    labels = json.loads((DEMO / "feature_labels.json").read_text())
    assert set(meta["feature_names"]) <= set(labels)
    for fname in meta["files"]:
        assert (DEMO / "model" / fname).exists()
    prof = json.loads((DEMO / "profiles.json").read_text())
    assert prof["profiles"], "no sender profiles"
    for p in prof["profiles"].values():
        assert set(p["features"]) <= set(meta["feature_names"])
    assert 0.0 < meta["threshold"] < 1.0


def test_metrics_cover_all_three_models_and_the_blend_wins():
    m = json.loads((DEMO / "metrics.json").read_text())
    for name in ("xgb", "gnn", "blend"):
        assert 0.0 < m["models"][name]["pr_auc"] < 1.0
    assert m["models"]["blend"]["pr_auc"] >= max(m["models"]["xgb"]["pr_auc"], m["models"]["gnn"]["pr_auc"])
    assert len(m["pr_curves"]["blend"]["precision"]) == len(m["pr_curves"]["blend"]["recall"])


def test_api_can_load_the_bundled_model_and_score():
    from api.main import Scorer

    try:
        scorer = Scorer.load(DEMO / "model")
    except Exception as e:  # noqa: BLE001  (xgboost unusable on this machine)
        pytest.skip(f"xgboost not usable: {e!r}"[:120])
    s = pl.read_parquet(DEMO / "shap.parquet")
    feats = dict(zip(s.filter(pl.col("account") == s["account"][0])["feature"], s.filter(pl.col("account") == s["account"][0])["value"]))
    p = float(scorer.score([feats])[0])
    assert 0.0 <= p <= 1.0
