# Evaluation protocol

Every model (rules, XGBoost v1/v2/v3, GNN) is evaluated under these rules, so the comparison table is apples-to-apples. The code is in `src/papertrail/split.py` and `src/papertrail/metrics.py`.

## Rules
1. **Fixed split.** Use `temporal_split` with the default boundaries (train Sep 1–6, validation Sep 7–8, test Sep 9–10; Sep 11–18 dropped). See `reports/split.md`.
2. **Choose everything on validation.** Hyperparameters, early stopping and the decision threshold (`best_f1_threshold`) are chosen using validation data only.
3. **Touch test once per model.** Apply the validation-chosen threshold unchanged to the test split and record the result. Do not re-tune after seeing test numbers. If a model is changed, it counts as a new model and its test score is reported separately.
4. **Never report accuracy.** With 0.1% positives, predicting "normal" for everything scores about 99.9%.
5. **Same metric names.** Every run logs exactly the metrics in `papertrail.metrics.METRIC_NAMES`, through `evaluate(y, p, thr)`:

| MLflow metric name | Meaning |
|---|---|
| `pr_auc` | Area under the precision-recall curve. Threshold-free and the primary ranking metric. |
| `f1` | Minority-class (laundering) F1 at the validation-chosen threshold. |
| `recall_at_1pct_fpr` | Share of laundering caught while flagging at most 1% of normal transactions. |
| `precision_at_1000` | Share of true laundering among the 1,000 highest-scored transactions, matching an investigator's daily capacity. |

6. **Features use only the past.** A transaction's features may only use transactions that happened before it. Graph features for validation and test use only edges before that split's start (see playbook Phase 4).
7. **Log to MLflow (DagsHub).** Log parameters, the four metrics for validation and test (with a `val_` / `test_` prefix), the threshold and the PR curve image. Use run names such as `xgb_v1_tabular`.

## Reporting
- **Primary table:** all test transactions, the five models side by side.
- **Secondary view:** ACH-only transactions, because 86.6% of positives are ACH.
- Report the number of positives behind each number (test has 956). Differences of a few percent between models are within noise.
- Rules, XGBoost and GNN are compared on the same test rows, and the base rate (0.111% on test) is stated next to PR-AUC.

## Known limitations
- Laundering attempts can straddle splits (see `reports/split.md`).
- The data is synthetic, so conclusions describe the generator, not real banks.
