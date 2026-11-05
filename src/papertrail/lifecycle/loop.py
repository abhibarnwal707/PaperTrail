"""Drift-triggered retraining: check drift -> (rule fires) -> retrain -> evaluation gate -> promote or reject. Every step is logged to MLflow.

Reference window = the first 40% of the data in time, current window = the last 40% (the middle 20% is skipped so the windows do not
touch). Importance for the rule comes from the production model's total gain. `simulate_drift` shifts raw payments after a cut-off so the
whole loop can be demonstrated; it changes inputs (amounts, hours, payment formats) and lets the real feature code propagate the shift.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from papertrail.drift import drift_table, retrain_decision
from papertrail.lifecycle import data as D
from papertrail.lifecycle import promotion
from papertrail.lifecycle.config import LifecycleConfig
from papertrail.lifecycle.registry import PRODUCTION, Registry


def simulate_drift(txns: pl.DataFrame, after_frac: float = 0.4, seed: int = 0, poison_labels: bool = False) -> pl.DataFrame:
    """Shift the payments after the `after_frac` quantile in time: amounts x4, 5 hours later, 40% of payments moved to Wire.
    `poison_labels` randomises the labels in the TRAINING window (first 60% in time), a broken labelling pipeline: the retrain learns noise while
    the evaluation window keeps true labels, so the gate must reject it. (Corrupting the evaluation labels too would fool any gate.)"""
    txns = txns.sort(["ts", "txn_id"])
    rng = np.random.default_rng(seed)
    cut = int(txns.height * after_frac)
    late = np.arange(txns.height) >= cut
    wire = late & (rng.random(txns.height) < 0.4)
    out = txns.with_columns(
        pl.when(pl.Series(late)).then(pl.col("amt_paid") * 4).otherwise(pl.col("amt_paid")).alias("amt_paid"),
        pl.when(pl.Series(late)).then(pl.col("ts") + pl.duration(hours=5)).otherwise(pl.col("ts")).alias("ts"),
        pl.when(pl.Series(wire)).then(pl.lit("Wire")).otherwise(pl.col("pay_format")).alias("pay_format"),
    )
    if poison_labels:
        shuffled = rng.permutation(out["label"].to_numpy())
        in_train = np.arange(out.height) < int(out.height * 0.6)
        out = out.with_columns(pl.when(pl.Series(in_train)).then(pl.Series(shuffled)).otherwise(pl.col("label")).alias("label"))
    return out.sort(["ts", "txn_id"])


def stationary_control(txns: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """A no-drift control: payments shuffled in time and timestamps spaced evenly over the same span, so behaviour AND payment density are the
    same at every time. (The raw demo slice is a sample built around alerted accounts and is NOT stationary: the rule already fires on it.)"""
    txns = txns.sort(["ts", "txn_id"])
    rng = np.random.default_rng(seed)
    as_us = txns["ts"].cast(pl.Int64)                      # microseconds since the epoch, timezone-free
    even = pl.Series("ts", np.linspace(as_us.min(), as_us.max(), txns.height)).cast(pl.Int64).cast(pl.Datetime("us"))
    shuffled = txns.drop("ts").select(pl.all().gather(rng.permutation(txns.height)))
    return shuffled.with_columns(even.alias("ts")).sort(["ts", "txn_id"])


def production_importance(reg: Registry, work: Path) -> dict[str, float]:
    import xgboost as xgb

    v = reg.version_of(PRODUCTION)
    if v is None:
        raise RuntimeError("no production model; run scripts/promote.py first")
    d = reg.materialise(v, work / "prod")
    gains: dict[str, float] = {}
    import json
    for f in json.loads((d / "meta.json").read_text()).get("files", ["xgb.json"]):
        b = xgb.Booster()
        b.load_model(str(d / f))
        for k, g in b.get_score(importance_type="total_gain").items():
            gains[k] = gains.get(k, 0.0) + g
    total = sum(gains.values()) or 1.0
    return {k: g / total for k, g in gains.items()}


def check_and_retrain(txns: pl.DataFrame, cfg: LifecycleConfig | None = None, retrain_txns: pl.DataFrame | None = None,
                      trigger: str = "drift") -> dict:
    """Evaluate the retrain rule on `txns`; when it fires, retrain on `retrain_txns` (default: `txns`) and run the gate.
    Returns {"drift": decision dict, "retrained": bool, "gate": result or None}; everything is logged to MLflow."""
    cfg = cfg or LifecycleConfig()
    reg = Registry(cfg)
    work = Path(tempfile.mkdtemp())
    frame = D.feature_frame(txns)
    n = frame.height
    ref, cur = frame[: int(n * 0.4)], frame[int(n * 0.6):]
    table = drift_table(ref, cur, D.feature_names(), production_importance(reg, work))
    decision = retrain_decision(table)
    reg.log_decision("drift_check", {"rule_fired": decision["retrain"], "rule": decision["rule"], "drifted": ",".join(decision["drifted"])},
                     {"n_drifted_stable_features": float(len(decision["drifted"])), "max_psi": float(table["psi"].max())}, {**decision, "psi": table.select("feature", "psi", "calendar_dependent", "importance").to_dicts()})
    out = {"drift": decision, "retrained": False, "gate": None, "psi_table": table}
    if decision["retrain"]:
        out["retrained"] = True
        out["gate"] = promotion.run(seed=1, txns=retrain_txns if retrain_txns is not None else txns, cfg=cfg, label=trigger)
    return out
