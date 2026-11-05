"""Evaluate the drift retraining rule; if it fires, retrain and run the evaluation gate (see docs/MODEL_LIFECYCLE.md).

    python scripts/check_drift_and_retrain.py                      # on the real demo slice
    python scripts/check_drift_and_retrain.py --simulate-drift     # inject shifted payments: rule fires -> retrain -> gate
    python scripts/check_drift_and_retrain.py --simulate-drift --poison-labels   # ... and a broken retrain the gate must reject

If no production model exists yet, one is trained and promoted first (the stationary-control version of the demo slice). Exit code 0 always; the decision is
printed and logged to MLflow (runs `drift_check` and `gate_decision`).
"""
import argparse

from papertrail.lifecycle import data as D
from papertrail.lifecycle import promotion
from papertrail.lifecycle.config import LifecycleConfig
from papertrail.lifecycle.loop import (
    check_and_retrain,
    simulate_drift,
    stationary_control,
)
from papertrail.lifecycle.registry import PRODUCTION, Registry

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate-drift", action="store_true")
    ap.add_argument("--poison-labels", action="store_true", help="with --simulate-drift: also corrupt the labels of shifted rows")
    a = ap.parse_args()
    cfg = LifecycleConfig()
    base = stationary_control(D.load_slice())      # the raw slice is not stationary (the rule fires on it); see docs/MODEL_LIFECYCLE.md
    if Registry(cfg).version_of(PRODUCTION) is None:
        print("no production model: bootstrapping one from the unshifted slice")
        promotion.run(0, txns=base, cfg=cfg, label="bootstrap")
    txns = simulate_drift(base, poison_labels=a.poison_labels) if a.simulate_drift else base
    out = check_and_retrain(txns, cfg)
    d = out["drift"]
    print(f"\nrule: {d['rule']}\nfired: {d['retrain']}  drifted: {d['drifted']}\n(expected calendar drift, ignored by the rule: {len(d['expected_calendar_drift'])} features)")
    print("retrained + gated:", out["gate"]["promoted"] if out["gate"] else "no retrain needed")
