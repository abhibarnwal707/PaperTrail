# Upgrade notes: ModelOps lifecycle and agent observability

Working notes for `../PAPERTRAIL_UPGRADE_PLAN.md`. Written after reading the repo, before changing code.

## What exists (verified by reading the code, 151 tests pass)

| Area | Where | Relevant facts |
|---|---|---|
| Metrics | `src/papertrail/metrics.py` | `evaluate()` returns `pr_auc`, `f1`, `recall_at_1pct_fpr`, `precision_at_1000`; `best_f1_threshold` is validation-only |
| Features | `features.py`, `graph_features.py` | per-row tabular + as-of history features; `HISTORY_FEATURES`; `add_history_features(df)` needs `ts`-sorted input |
| Drift | `drift.py` | `psi`, `drift_table`, `retrain_decision` (PSI > 0.2 on 3+ of top 15 calendar-stable features) |
| Training | `notebooks/04b-xgboost-final.ipynb` (Kaggle) | 3-seed XGBoost, logged to DagsHub MLflow experiment `papertrail`; **no registry use, no model-version metadata** |
| Serving | `api/main.py` | `Scorer.load(dir)` reads `meta.json` + booster files from `data/demo/model`; deliberately imports nothing from `papertrail` |
| Agent | `src/papertrail/agent/` | LangGraph `gather -> draft -> verify -> retry/fallback`; guardrail in `verify.py`; LLM calls in `llm.py` |
| Agent eval | `scripts/agent_eval.py`, `reports/agent_eval.md` | 30 cases already evaluated (this overlaps plan section 4) |
| Local data | `data/demo/` | ~8k-row slice with scores, **not** the 5M-row feature matrix |

## Constraints that shape the design

1. The full 5.08M-row data and trained feature matrix are on Kaggle, not on this laptop. Any retrain run here uses the
   **demo slice** (`data/demo/txns.parquet`, features rebuilt with `features.py` + `graph_features.py`). Those numbers
   are about the *mechanism* (registry, gate, rollback, loop), not about model quality. Docs must say so; none of them
   may be quoted as the headline PR-AUC (0.50).
2. `api/main.py` must stay import-free of `papertrail` (small Docker image). Registry loading therefore only
   *materialises* a registered version into a directory of the same shape (`meta.json` + boosters) and the API reads
   that directory, falling back to `data/demo/model`. Default behaviour is unchanged unless a flag is set.
3. `mlflow` is already a base dependency. Local runs use a file/sqlite store under `data/local/mlruns` (git-ignored);
   Kaggle/DagsHub use `MLFLOW_TRACKING_URI`.
4. Python 3.15 locally, 3.11 in CI: no new heavy dependencies; tracing falls back to local JSONL with no extra install.

## Plan

New package `src/papertrail/lifecycle/` (pure logic separated from MLflow I/O so it is unit-testable):

- `gate.py`: `evaluate_gate(candidate, current, tolerances, leakage_ok)` returns a `GateDecision` with per-check results and
  reasons. `check_point_in_time(...)`: recompute history features on a prefix and on the full frame and require identical values for
  earlier rows (a perturbation test: changing a future row must not change an earlier row's features).
- `registry.py`: register a version with tags (data hash, git SHA, feature-list hash, metrics), set stage via aliases
  (`production`, `previous`), `rollback()`, `materialise(version, dir)`.
- `model_card.py`: renders a one-page markdown card per version from registry metadata.
- `scripts/promote.py`, `scripts/rollback.py`, `scripts/check_drift_and_retrain.py` (section 2), `scripts/replay.py` (section 3).
- Config in one place: `src/papertrail/lifecycle/config.py` (tolerances, tracking URI, flags from env), `.env.example` updated.

Order: (1) gate + tests, (2) registry + promote/rollback + tests, (3) API registry loading behind `PAPERTRAIL_REGISTRY_MODEL`,
(4) model card, then sections 2, 3, 4, `docs/MODEL_LIFECYCLE.md`, README, `RESULTS_SUMMARY.md`.

## Open questions / TODO

- Retirement criteria need a number I cannot derive from data here; they will be stated as policy, marked as not measured.
- Langfuse needs keys I do not have; JSONL is the default sink, Langfuse is wired behind a flag and untested live.
- Section 4 largely exists (`reports/agent_eval.md`); the missing pieces are guardrail unit tests (check `tests/test_agent.py` first),
  a no-LLM baseline comparison, and cost estimates.

## Progress log

### Section 1 (registry, gate, rollback): done except README/MODEL_LIFECYCLE write-up
- `lifecycle/{config,gate,registry,model_card,data}.py`, `scripts/{promote,rollback,export_production}.py`, tests in `tests/test_gate.py`, `tests/test_registry.py`.
- Aliases (`candidate`/`production`/`previous`) instead of MLflow stages (deprecated in MLflow 3).
- API loads `$PAPERTRAIL_MODEL_DIR` (written by `export_production.py`) and falls back to `data/demo/model`; `mlflow` stays out of the API image.
- The packaged demo model is NOT registered: its features came from the full Kaggle pipeline and cannot be rebuilt faithfully from the slice,
  so the registry's first production model is the slice-trained one and the packaged model remains the API fallback.

### Finding: the point-in-time check caught a real (small) leak in `daily_pagerank`
Each daily PageRank snapshot was sized by ALL accounts in the frame, including accounts that first appear later, so `s_pagerank`/`d_pagerank`
depended weakly on the future account count. Edges used were always earlier ones. Fix is opt-in (`as_of_nodes=True`) so the published results
(trained with the legacy behaviour) stay reproducible; the lifecycle pipeline uses the strict version. Whether the legacy leak changes the headline
metrics is NOT measured: it would need a Kaggle rerun (TODO, do not claim either way).
