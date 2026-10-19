# Kaggle notebook 02_baseline (CPU). Each "# %%" block is one notebook cell.
# Lines starting with "# !" are notebook shell commands: delete the "# " when you paste them into Kaggle.
#
# Before you start:
#   1. Push the latest code to GitHub (features.py, rules.py, split.py, metrics.py must be there).
#   2. New Kaggle notebook "02_baseline", CPU, Internet ON.
#   3. Add Input -> your private dataset "papertrail-processed".
#   4. Add-ons -> Secrets -> attach DAGSHUB_TOKEN.

# %% Cell 1: setup (the username is already filled in)
# !pip install -q git+https://github.com/abhibarnwal707/PaperTrail.git
# (If you push new code and re-run in the SAME session, use:
#  !pip install -q --force-reinstall --no-deps git+https://github.com/abhibarnwal707/PaperTrail.git)
import os
from kaggle_secrets import UserSecretsClient
os.environ["DAGSHUB_USER_TOKEN"] = UserSecretsClient().get_secret("DAGSHUB_TOKEN")
import dagshub, mlflow
dagshub.init(repo_owner="abhibarnwal707", repo_name="papertrail", mlflow=True)

# %% Cell 2: load the processed data from the Kaggle Dataset
import glob
import numpy as np
import polars as pl

path = glob.glob("/kaggle/input/**/transactions.parquet", recursive=True)[0]
df = pl.read_parquet(path).sort(["ts", "txn_id"])
print(path, df.shape)

# %% Cell 3: history feature (needed by one rule), then the fixed split from Phase 2
from papertrail.features import add_out_count_24h
from papertrail.split import assert_valid_split, split_summary, temporal_split

df = add_out_count_24h(df)          # looks strictly backwards, so it is safe to compute before splitting
train, val, test = temporal_split(df)
summary = split_summary(df)
pl.Config.set_tbl_cols(-1)
print(summary)
assert_valid_split(summary)         # every split has positives, no date overlap

# %% Cell 4: sanity check that the 3K-30K amount band covers the training laundering amounts
pos_amt = train.filter(pl.col("label") == 1)["amt_paid"]
print({q: round(pos_amt.quantile(q), 2) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)})

# %% Cell 5: shared evaluation helper (protocol: threshold on validation, test scored once per model)
from papertrail.metrics import best_f1_threshold, evaluate

y_val = val["label"].to_numpy()
y_test = test["label"].to_numpy()
ach_test = (test["pay_format"] == "ACH").to_numpy()


def run_eval(p_val, p_test):
    thr = best_f1_threshold(y_val, p_val)
    res = {"threshold": float(thr)}
    for k, v in evaluate(y_val, p_val, thr).items():
        res[f"val_{k}"] = v
    for k, v in evaluate(y_test, p_test, thr).items():
        res[f"test_{k}"] = v
    # secondary view from the protocol: ACH-only transactions (86.6% of all positives)
    for k, v in evaluate(y_test[ach_test], p_test[ach_test], thr).items():
        res[f"test_ach_{k}"] = v
    return res


# %% Cell 6: Step 3.1 rule-based baseline
from papertrail.rules import fit_rule_params, rule_flags, rule_score

rule_params = fit_rule_params(train)   # thresholds come from the training split only
print(rule_params)

# each rule on its own, on validation (for insight only, nothing is tuned on it)
flags = rule_flags(val, rule_params)
for c in ["r_high_amount", "r_ach_band", "r_burst"]:
    fired = flags[c].to_numpy()
    tp = int((fired & (y_val == 1)).sum())
    print(f"{c:15s} flagged={int(fired.sum()):8d}  precision={tp / max(int(fired.sum()), 1):.4f}  "
          f"recall={tp / int(y_val.sum()):.3f}")

s_val, s_test = rule_score(val, rule_params), rule_score(test, rule_params)
with mlflow.start_run(run_name="rules_baseline"):
    mlflow.log_params({f"rule_{k}": v for k, v in rule_params.items()})
    rules_res = run_eval(s_val, s_test)
    mlflow.log_metrics(rules_res)
print({k: round(v, 4) for k, v in rules_res.items() if k.startswith("test_") and "ach" not in k})

# %% Cell 7: Step 3.2 + 3.3 XGBoost v1 on tabular features
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import precision_recall_curve

from papertrail.features import build_features, fit_categories

cats = fit_categories(train)


def to_matrix(part):
    return build_features(part, cats).cast(pl.Float32).to_numpy()


X_train, X_val, X_test = to_matrix(train), to_matrix(val), to_matrix(test)
y_train = train["label"].to_numpy()
feat_names = build_features(train.head(5), cats).columns
spw = float((y_train == 0).sum() / y_train.sum())
print(X_train.shape, "scale_pos_weight =", round(spw, 1))

xgb_params = dict(
    tree_method="hist", n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    eval_metric="aucpr", early_stopping_rounds=50, n_jobs=-1, random_state=0,
)
model = xgb.XGBClassifier(**xgb_params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)   # early stopping on validation PR-AUC

p_val = model.predict_proba(X_val)[:, 1]
p_test = model.predict_proba(X_test)[:, 1]

with mlflow.start_run(run_name="xgb_v1_tabular"):
    mlflow.log_params({k: v for k, v in xgb_params.items() if k != "eval_metric"})
    mlflow.log_param("n_features", len(feat_names))
    mlflow.log_param("best_iteration", int(model.best_iteration))
    xgb_res = run_eval(p_val, p_test)
    mlflow.log_metrics(xgb_res)

    # PR curves (validation and test) as an image
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y, p in (("validation", y_val, p_val), ("test", y_test, p_test)):
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=name)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("xgb_v1_tabular")
    ax.legend()
    mlflow.log_figure(fig, "pr_curve.png")
    plt.show()

    # model + feature importance as artifacts (registry comes later, in Step 5.4)
    model.save_model("/kaggle/working/xgb_v1.json")
    mlflow.log_artifact("/kaggle/working/xgb_v1.json")
    imp = pl.DataFrame({"feature": feat_names, "gain": model.feature_importances_.tolist()}).sort(
        "gain", descending=True)
    imp.write_csv("/kaggle/working/xgb_v1_importance.csv")
    mlflow.log_artifact("/kaggle/working/xgb_v1_importance.csv")
print(imp)

# %% Cell 8: comparison table (test split) and files to download
rows = []
for name, res in (("Rules", rules_res), ("XGB tabular", xgb_res)):
    rows.append({
        "Model": name,
        "PR-AUC": res["test_pr_auc"],
        "F1 (minority)": res["test_f1"],
        "Recall@1%FPR": res["test_recall_at_1pct_fpr"],
        "Precision@1000": res["test_precision_at_1000"],
        "ACH-only PR-AUC": res["test_ach_pr_auc"],
        "ACH-only F1": res["test_ach_f1"],
    })
table = pl.DataFrame(rows)
print(f"test positives: {int(y_test.sum())} of {len(y_test)} (base rate {y_test.mean():.4%})")
print(table)

import json
table.write_csv("/kaggle/working/baseline_table.csv")
with open("/kaggle/working/baseline_metrics.json", "w") as fh:
    json.dump({"rules_baseline": rules_res, "xgb_v1_tabular": xgb_res, "rule_params": rule_params,
               "scale_pos_weight": spw, "best_iteration": int(model.best_iteration)}, fh, indent=2)

print("commit message idea:",
      f"XGB tabular baseline: PR-AUC {xgb_res['test_pr_auc']:.3f}, F1 {xgb_res['test_f1']:.2f}")
# Then download baseline_table.csv, baseline_metrics.json and xgb_v1_importance.csv from the Output tab
# into PaperTrail/reports/ and commit them. Finally "Save Version" -> "Save & Run All (Commit)".
