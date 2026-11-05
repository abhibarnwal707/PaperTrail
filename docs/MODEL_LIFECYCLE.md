# Model lifecycle

How a PaperTrail model gets registered, promoted, monitored, retrained, rolled back and retired. Code is in `src/papertrail/lifecycle/`;
every command below can be re-run locally (CPU only, a few seconds).

> **Scope of the numbers.** The full 5.08M-row pipeline trains on Kaggle. The lifecycle machinery here runs on the committed demo slice
> (7,981 payments, about 12% laundering, built around alerted accounts). Metrics printed by these scripts show that the *mechanism* works;
> they are not model-quality results and must not be compared with the headline PR-AUC in the README.

## Flow

```
train candidate ─► register (MLflow, alias "candidate") ─► evaluation gate ─┬─► promote: alias "production" (old one becomes "previous")
                                                                           └─► reject: reasons stored as a tag on the version
drift check (PSI rule) ── fires ──► retrain ──► same gate          rollback: "previous" ──► "production"
```

Every step writes an MLflow run (`drift_check`, `gate_decision`, `rollback`) so the history can be audited.

## Registry

`lifecycle/registry.py`, MLflow model registry (local sqlite by default, `MLFLOW_TRACKING_URI` for DagsHub). MLflow 3 deprecates stages, so
stages are aliases: `candidate`, `production`, `previous`. Each version stores: data fingerprint, git SHA, feature-list fingerprint,
feature count, evaluation window and its metrics. `python scripts/promote.py` prints the version's model card
(`lifecycle/model_card.py`, rendered only from registry metadata).

## Promotion criteria (the gate, `lifecycle/gate.py`)

A candidate is promoted only if all of these hold, evaluated on the same held-out data for candidate and production:

1. PR-AUC is not worse than production by more than `GATE_TOL_PR_AUC` (default 0.01, absolute).
2. Recall at 1% FPR is not worse by more than `GATE_TOL_RECALL_1FPR` (default 0.02).
3. The point-in-time check passes: history features rebuilt without the future rows must equal those built with them.

With no production model only check 3 applies. Otherwise the candidate is rejected, the reasons go on the version and into MLflow.
The tolerances are policy choices, not tuned values.

**Limit of the gate:** it trusts the evaluation labels. If the evaluation window's labels are corrupted too, a broken model can pass.
The simulated "broken labels" test corrupts only the training window for that reason.

## Rollback

`python scripts/rollback.py` restores the `previous` version as `production` and tags the retired one. With no previous version it fails loudly.
Serving: `python scripts/export_production.py data/local/production_model`, then `PAPERTRAIL_MODEL_DIR=data/local/production_model uvicorn api.main:app`.
If that directory cannot be loaded the API serves the packaged `data/demo/model`, so the deployed demo is unaffected.

## Retrain triggers (`drift.py`, `lifecycle/loop.py`)

Retrain when PSI > 0.2 on 3 or more of the 15 most important calendar-stable features (importance = production model's total gain).
Calendar-dependent features (7-day windows, all-time totals, PageRank, account age) drift by construction and never trigger it.
`python scripts/check_drift_and_retrain.py` evaluates the rule, and when it fires retrains and runs the gate.

Simulated drift test (`tests/test_drift_loop.py`, or by hand), on the stationary control of the demo slice:

| scenario | command | rule | outcome |
|---|---|---|---|
| no drift | `python scripts/check_drift_and_retrain.py` | not fired (2 stable features over 0.2; 3 are needed) | nothing retrained |
| drift injected (amounts x4, +5 h, 40% moved to Wire) | `... --simulate-drift` | fired (`s_gap_s`, `d_gap_s`, `log_amt`) | retrained, gate **promoted** |
| drift + corrupted training labels | `... --simulate-drift --poison-labels` | fired | retrained, gate **rejected**, production unchanged |

Slice-scale PR-AUCs for these runs are in the script output and the MLflow runs; they will vary a little with library versions.

Known caveat: the no-drift control sits at 2 of the 3 features needed to fire, so the rule is not far from firing on pure noise at this
sample size. At 5M rows the PSI estimates are far less noisy; that has not been tested here.

## Retirement criteria (policy, not measured)

A version is retired (alias dropped, tagged `retired`) when it is replaced by a promoted candidate and a rollback has not been needed within
an agreed observation period, or immediately after a rollback away from it. The observation period is not set because no production
traffic exists to size it. **TODO** once real usage data exists.

## Scheduled check

`.github/workflows/drift-check.yml` runs the check weekly and on demand on the demo slice. It has not been run on GitHub yet.
