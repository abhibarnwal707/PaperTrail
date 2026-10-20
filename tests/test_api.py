import json

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from api.main import Scorer, create_app


class FakeBooster:
    """score = sigmoid(a - b): deterministic, no xgboost needed."""

    def predict(self, dmatrix):
        x = dmatrix
        return 1 / (1 + np.exp(-(x[:, 0] - x[:, 1])))


class FakeScorer(Scorer):
    def __init__(self, threshold=0.7):
        super().__init__(FakeBooster(), ["a", "b"], threshold, name="fake")

    def score(self, rows):
        x = np.array([[np.nan if r.get(f) is None else r[f] for f in self.feature_names] for r in rows], dtype=np.float32)
        return self.booster.predict(x)


@pytest.fixture
def demo(tmp_path):
    pl.DataFrame({
        "account": ["A1"] * 4 + ["B2"],
        "feature": ["f1", "f2", "f3", "f4", "f1"],
        "value": [1.0, 2.0, None, 4.0, 5.0],
        "contribution": [0.1, -0.9, 0.5, 0.2, 0.3],
    }).write_parquet(tmp_path / "shap.parquet")
    return tmp_path


@pytest.fixture
def client(demo):
    return TestClient(create_app(scorer=FakeScorer(), demo_dir=demo))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_score_valid_payload_and_threshold(client):
    hi = client.post("/score", json={"features": {"a": 3.0, "b": 0.0}}).json()      # sigmoid(3) = 0.95
    lo = client.post("/score", json={"features": {"a": 0.0, "b": 3.0}}).json()      # sigmoid(-3) = 0.047
    assert hi["score"] == pytest.approx(0.9526, abs=1e-3) and hi["flagged"] is True
    assert lo["score"] < 0.1 and lo["flagged"] is False
    assert hi["threshold"] == 0.7 and hi["model"] == "fake"


def test_missing_features_are_accepted_as_missing(client):
    r = client.post("/score", json={"features": {"a": 3.0}})
    assert r.status_code == 200
    r = client.post("/score", json={"features": {"a": 3.0, "b": None}})
    assert r.status_code == 200


@pytest.mark.parametrize("payload", [
    {},                                                  # no "features" key
    {"features": "not a dict"},
    {"features": {"a": "abc"}},                          # non-numeric value
    {"features": {"a": 1.0, "zzz": 2.0}},                # feature the model does not know
])
def test_bad_payload_returns_422(client, payload):
    assert client.post("/score", json=payload).status_code == 422


def test_score_without_a_model_returns_503(demo):
    c = TestClient(create_app(scorer=None, demo_dir=demo))     # no lifespan started: nothing loaded
    assert c.get("/health").json()["model_loaded"] is False
    assert c.post("/score", json={"features": {"a": 1.0}}).status_code == 503


def test_explain_returns_drivers_sorted_by_absolute_contribution(client):
    r = client.get("/explain/A1").json()
    assert r["account"] == "A1"
    assert [d["feature"] for d in r["drivers"]] == ["f2", "f3", "f4", "f1"]
    assert r["drivers"][2]["value"] == 4.0 and r["drivers"][1]["value"] is None
    assert len(client.get("/explain/A1?top=2").json()["drivers"]) == 2


def test_explain_unknown_account_and_missing_bundle(client, tmp_path):
    assert client.get("/explain/NOPE").status_code == 404
    empty = TestClient(create_app(scorer=FakeScorer(), demo_dir=tmp_path / "nothing"))
    assert empty.get("/explain/A1").status_code == 404


def test_latency_header_is_present(client):
    r = client.get("/health")
    assert float(r.headers["X-Process-Time-ms"]) >= 0


def test_real_xgboost_model_roundtrip(tmp_path):
    try:
        import xgboost as xgb
    except Exception:                                       # noqa: BLE001  (e.g. missing OpenMP on a laptop)
        pytest.skip("xgboost is not usable here")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3)).astype(np.float32)
    y = (x[:, 0] + rng.normal(scale=0.3, size=500) > 0).astype(int)
    names = ["f_a", "f_b", "f_c"]
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 2}, xgb.DMatrix(x, label=y, feature_names=names), 20)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    booster.save_model(str(model_dir / "xgb.json"))
    (model_dir / "meta.json").write_text(json.dumps({"feature_names": names, "threshold": 0.5, "model": "tiny"}))
    scorer = Scorer.load(model_dir)
    c = TestClient(create_app(scorer=scorer, demo_dir=tmp_path))
    row = {"f_a": 2.0, "f_b": 0.0, "f_c": 0.0}
    got = c.post("/score", json={"features": row}).json()
    expect = float(booster.predict(xgb.DMatrix(np.array([[2.0, 0.0, 0.0]], dtype=np.float32), feature_names=names))[0])
    assert got["score"] == pytest.approx(expect, abs=1e-6) and got["flagged"] is (expect >= 0.5)
    assert c.post("/score", json={"features": {"f_a": 2.0}}).status_code == 200      # missing -> NaN is fine for XGBoost


def test_real_xgboost_ensemble_averages_probabilities(tmp_path):
    try:
        import xgboost as xgb
    except Exception:                                       # noqa: BLE001
        pytest.skip("xgboost is not usable here")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    names = ["a", "b"]
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    boosters = []
    for i in range(2):
        b = xgb.train({"objective": "binary:logistic", "max_depth": 2, "seed": i, "subsample": 0.7},
                      xgb.DMatrix(X, label=y, feature_names=names), 10)
        b.save_model(str(model_dir / f"m{i}.json"))
        boosters.append(b)
    (model_dir / "meta.json").write_text(json.dumps(
        {"feature_names": names, "threshold": 0.5, "model": "ens", "files": ["m0.json", "m1.json"]}))
    scorer = Scorer.load(model_dir)
    assert len(scorer.boosters) == 2
    d = xgb.DMatrix(np.array([[1.5, 0.0]], dtype=np.float32), feature_names=names)
    expect = float(np.mean([b.predict(d)[0] for b in boosters]))
    got = TestClient(create_app(scorer=scorer, demo_dir=tmp_path)).post("/score", json={"features": {"a": 1.5, "b": 0.0}}).json()
    assert got["score"] == pytest.approx(expect, abs=1e-6)
