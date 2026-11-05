"""The whole drift -> retrain -> gate loop on the demo slice (a few seconds): no drift, real drift, and drift with a broken retrain."""
import pytest

pytest.importorskip("mlflow")
pytest.importorskip("xgboost")

from papertrail.lifecycle import data as D
from papertrail.lifecycle import promotion
from papertrail.lifecycle.config import LifecycleConfig
from papertrail.lifecycle.loop import (
    check_and_retrain,
    simulate_drift,
    stationary_control,
)
from papertrail.lifecycle.registry import PRODUCTION, Registry


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    cfg = LifecycleConfig(tracking_uri=f"sqlite:///{tmp_path_factory.mktemp('mlf') / 'mlflow.db'}", model_name="loop", experiment="loop")
    control = stationary_control(D.load_slice())
    promotion.run(0, txns=control, cfg=cfg, label="bootstrap")
    return cfg, control


def test_control_does_not_fire_the_rule(world):
    cfg, control = world
    before = Registry(cfg).version_of(PRODUCTION)
    out = check_and_retrain(control, cfg)
    assert not out["drift"]["retrain"] and not out["retrained"]
    assert Registry(cfg).version_of(PRODUCTION) == before


def test_simulated_drift_fires_retrains_and_gate_promotes(world):
    cfg, control = world
    out = check_and_retrain(simulate_drift(control), cfg)
    assert out["drift"]["retrain"] and len(out["drift"]["drifted"]) >= 3
    assert out["retrained"] and out["gate"]["promoted"]
    assert out["gate"]["candidate_metrics"]["pr_auc"] > out["gate"]["current_metrics"]["pr_auc"]


def test_drift_with_broken_training_labels_is_rejected_by_the_gate(world):
    cfg, control = world
    reg = Registry(cfg)
    before = reg.version_of(PRODUCTION)
    out = check_and_retrain(simulate_drift(control, poison_labels=True), cfg)
    assert out["drift"]["retrain"] and out["retrained"] and not out["gate"]["promoted"]
    assert any("pr_auc" in r for r in out["gate"]["decision"]["reasons"])
    assert reg.version_of(PRODUCTION) == before                   # production untouched


def test_every_decision_is_logged_to_mlflow(world):
    cfg, _ = world
    reg = Registry(cfg)
    names = {r.data.tags["mlflow.runName"] for r in reg.client.search_runs([reg.client.get_experiment_by_name(cfg.experiment).experiment_id])}
    assert {"drift_check", "gate_decision"} <= names
