import json

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("xgboost")

from papertrail.lifecycle import data as D
from papertrail.lifecycle.config import LifecycleConfig
from papertrail.lifecycle.registry import PREVIOUS, PRODUCTION, Registry


@pytest.fixture
def reg(tmp_path):
    return Registry(LifecycleConfig(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", model_name="t-model", experiment="t-exp"))


def _model_dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"feature_names": ["a"], "threshold": 0.5, "model": name, "files": ["xgb.json"]}))
    (d / "xgb.json").write_text("{}")
    return d


def test_register_promote_rollback_cycle(reg, tmp_path):
    tags = {"data_hash": "h", "git_sha": "g", "feature_list_hash": "f"}
    v1 = reg.register(_model_dir(tmp_path, "m1"), {"pr_auc": 0.5}, tags)
    assert reg.version_of(PRODUCTION) is None
    reg.promote(v1)
    assert reg.version_of(PRODUCTION) == v1 and reg.version_of(PREVIOUS) is None
    v2 = reg.register(_model_dir(tmp_path, "m2"), {"pr_auc": 0.6}, tags)
    reg.promote(v2)
    assert (reg.version_of(PRODUCTION), reg.version_of(PREVIOUS)) == (v2, v1)
    assert reg.rollback() == (v2, v1)
    assert reg.version_of(PRODUCTION) == v1 and reg.version_of(PREVIOUS) is None
    assert "retired" in reg.info(v2)["tags"]
    with pytest.raises(RuntimeError):
        reg.rollback()                                           # nothing left to restore


def test_version_carries_lineage_tags_and_reject_records_reasons(reg, tmp_path):
    v = reg.register(_model_dir(tmp_path, "m"), {"pr_auc": 0.1}, {"data_hash": "abc", "git_sha": "def", "feature_list_hash": "ghi"})
    reg.reject(v, ["pr_auc: worse"])
    t = reg.info(v)["tags"]
    assert (t["data_hash"], t["git_sha"], t["feature_list_hash"]) == ("abc", "def", "ghi")
    assert t["rejected_reasons"] == "pr_auc: worse" and reg.version_of(PRODUCTION) is None


def test_materialise_round_trips_the_model_dir(reg, tmp_path):
    v = reg.register(_model_dir(tmp_path, "m"), {"pr_auc": 0.1}, {})
    out = reg.materialise(v, tmp_path / "out")
    assert json.loads((out / "meta.json").read_text())["model"] == "m"


def test_end_to_end_promote_script_on_demo_slice(tmp_path):
    from papertrail.lifecycle import promotion as promote
    cfg = LifecycleConfig(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", model_name="t2", experiment="t2")
    first = promote.run(0, cfg=cfg)
    assert first["promoted"] and first["current_metrics"] is None
    second = promote.run(1, cfg=cfg)
    assert second["current_metrics"] is not None and second["decision"]["checks"][2]["passed"]
    assert D.git_sha()
