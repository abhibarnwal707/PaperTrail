# Results summary

One line per measured result, with the command that reproduces it. Run from the repo root (`pip install -e ".[dev,api,app,agent]"`; on macOS set
`DYLD_FALLBACK_LIBRARY_PATH` to scikit-learn's `.dylibs` for XGBoost). Lifecycle numbers use a throw-away store: `export MLFLOW_TRACKING_URI=sqlite:////tmp/lc.db`.

**Scope warning.** Lifecycle results (sections A and B) run on the committed ~8k-row demo slice, which is about 12% laundering. They show that the
mechanism works. They are not model-quality numbers and must not be compared with the headline model in section D. Numbers vary a little with
library versions.

## A. Registry, gate, rollback

| Result | Command |
|---|---|
| Gate rejects a candidate whose PR-AUC and recall@1%FPR are worse than production beyond tolerance, promotes one within tolerance, and rejects any candidate that fails the leakage check | `pytest tests/test_gate.py` (9 tests) |
| Register -> promote -> promote -> rollback restores the previous version; versions carry data hash, git SHA, feature-list hash | `pytest tests/test_registry.py` |
| The point-in-time check found a leak in the legacy PageRank feature (snapshots sized by future accounts); the opt-in `as_of_nodes=True` build passes it | `pytest tests/test_gate.py::test_legacy_pagerank_leaks_future_account_count_and_strict_does_not` |
| API serves the packaged model when the configured model dir is broken | `pytest tests/test_api.py::test_broken_model_dir_falls_back_to_the_packaged_model` |

Not measured: whether the legacy PageRank leak changed the headline metrics (needs a Kaggle rerun).

## B. Drift-triggered retraining (stationary control of the demo slice)

| Result | Command |
|---|---|
| No injected drift: rule does not fire (2 stable features over PSI 0.2, 3 needed); nothing retrained | `python scripts/check_drift_and_retrain.py` |
| Injected drift (amounts x4, +5 h, 40% moved to Wire): rule fires on `s_gap_s`, `d_gap_s`, `log_amt`; retrain promoted (slice PR-AUC 0.880 vs 0.672 for the old model) | `python scripts/check_drift_and_retrain.py --simulate-drift` |
| Same drift with corrupted training labels: rule fires, retrain rejected by the gate (PR-AUC 0.069 vs 0.880), production unchanged | `python scripts/check_drift_and_retrain.py --simulate-drift --poison-labels` |
| All three scenarios and the MLflow audit trail asserted in tests | `pytest tests/test_drift_loop.py` |

Caveats: the raw slice is not stationary (the rule fires on it without injected drift), so the control reshuffles it; the control sits 1 feature
below the firing point; a gate cannot catch corrupted evaluation labels.

## C. Agent tracing, replay and evaluation

| Result | Command |
|---|---|
| A recorded run replays to the same result without data or an LLM, for a clean run, a retried run, a fallback run and an LLM outage | `pytest tests/test_tracing.py` |
| Guardrail rejections of uncited numbers are traced; tracing is off by default | `pytest tests/test_tracing.py` |
| 30-case evaluation of the LLM agent: 100% of summaries pass the guardrail, 99% of 154 first-draft claims match the cited tool output, 2 of 30 drafts rejected and retried, 0 fallbacks, 25% typology accuracy on 24 laundering accounts, 0 of 6 clean accounts escalated, mean 9.2 s and 4,820 tokens per case | `python scripts/agent_eval.py` (needs network and a key; results cached in `reports/agent_eval.md`) |
| Agent vs deterministic no-LLM baseline on the same 30 cases: typology 6/24 vs 5/24, same action on 28/30, 0 vs 0 false escalations | `python scripts/agent_baseline.py` (offline, writes `reports/agent_vs_baseline.md`) |
| The baseline's template summaries pass the guardrail on every demo account | `pytest tests/test_baseline.py` |

Not measured: dollar cost (the runs used free-tier models; tokens are reported, and `--usd-per-mtok` converts them at a price you supply); live-LLM traces
(none recorded here); Langfuse/OpenTelemetry export (not implemented).

## D. Pre-existing headline results (not re-run in this upgrade)

From `reports/final_comparison.csv` and `reports/blend_summary.json`, produced by the Kaggle notebooks: blend of GNN and XGBoost, test PR-AUC 0.497,
recall at 1% FPR 0.724, 3-seed mean. Reproduce by running the notebooks in `notebooks/` on Kaggle.

## E. Test suite

`pytest` -> 191 passed; `ruff check` clean.
