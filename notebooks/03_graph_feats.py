# Kaggle notebook 03_graph_feats (CPU, high-RAM if offered). Each "# %%" block is one notebook cell.
# Lines starting with "# !" are notebook shell commands: delete the "# " when you paste them into Kaggle.
#
# What it does: (1) rules_v2, the baseline with a working send-count rule; (2) leakage-safe history/graph
# features for every transaction; (3) XGBoost v2 on tabular + history features, with SHAP; (4) comparison table.
#
# Before you start:
#   1. Push the latest code to GitHub (graph_features.py, rules.py changes). Then in Kaggle: new notebook
#      "03_graph_feats", CPU, Internet ON, Add Input -> papertrail-processed, attach DAGSHUB_TOKEN secret.
#   2. Do NOT run Cell 6 on the full data until the dry run (Cell 5) works.

# %% Cell 1: setup
# !pip install -q git+https://github.com/abhibarnwal707/PaperTrail.git
import os
from kaggle_secrets import UserSecretsClient
os.environ["DAGSHUB_USER_TOKEN"] = UserSecretsClient().get_secret("DAGSHUB_TOKEN")
import dagshub, mlflow
dagshub.init(repo_owner="abhibarnwal707", repo_name="papertrail", mlflow=True)

# %% Cell 2: load
import glob
import time
import numpy as np
import polars as pl

path = glob.glob("/kaggle/input/**/transactions.parquet", recursive=True)[0]
df = pl.read_parquet(path).sort(["ts", "txn_id"])
print(path, df.shape)

# %% Cell 3: split + shared evaluation helper (same protocol as 02_baseline)
from papertrail.features import add_out_count_24h
from papertrail.metrics import best_f1_threshold, evaluate
from papertrail.split import assert_valid_split, split_summary, temporal_split

df = add_out_count_24h(df)
summary = split_summary(df)
pl.Config.set_tbl_cols(-1)
print(summary)
assert_valid_split(summary)


def make_eval(val, test):
    y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()
    ach_test = (test["pay_format"] == "ACH").to_numpy()

    def run_eval(p_val, p_test):
        thr = best_f1_threshold(y_val, p_val)
        res = {"threshold": float(thr)}
        for k, v in evaluate(y_val, p_val, thr).items():
            res[f"val_{k}"] = v
        for k, v in evaluate(y_test, p_test, thr).items():
            res[f"test_{k}"] = v
        for k, v in evaluate(y_test[ach_test], p_test[ach_test], thr).items():
            res[f"test_ach_{k}"] = v
        return res

    return y_val, y_test, run_eval


train, val, test = temporal_split(df)
y_val, y_test, run_eval = make_eval(val, test)

# %% Cell 4: rules_v2 (the send-count threshold is now chosen on TRAINING data and actually fires)
from papertrail.rules import fit_rule_params_v2, rule_flags, rule_score

params_v2, burst_table = fit_rule_params_v2(train)
print(burst_table)
print(params_v2)

flags = rule_flags(val, params_v2)
for c in ["r_high_amount", "r_ach_band", "r_burst"]:
    fired = flags[c].to_numpy()
    tp = int((fired & (y_val == 1)).sum())
    print(f"{c:15s} flagged={int(fired.sum()):8d}  precision={tp / max(int(fired.sum()), 1):.4f}  "
          f"recall={tp / int(y_val.sum()):.3f}")

with mlflow.start_run(run_name="rules_v2"):
    mlflow.log_params({f"rule_{k}": v for k, v in params_v2.items()})
    rules2_res = run_eval(rule_score(val, params_v2), rule_score(test, params_v2))
    mlflow.log_metrics(rules2_res)
print({k: round(v, 4) for k, v in rules2_res.items() if k.startswith("test_") and "ach" not in k})

# %% Cell 5: DRY RUN of the history features on the first 500K rows (checks speed and errors)
from papertrail.graph_features import HISTORY_FEATURES, add_history_features

small = df.head(500_000)
t0 = time.time()
small_f = add_history_features(small)
print(f"500K rows in {time.time() - t0:.0f}s ->", small_f.shape)
print(small_f.select(HISTORY_FEATURES).describe())
# Full data is about 10x this (and the event table doubles it). If this took more than ~5 min, ask for help first.

# %% Cell 6: history + graph features for ALL rows (past-only), then save them
t0 = time.time()
dfx = add_history_features(df)
print(f"all rows in {time.time() - t0:.0f}s ->", dfx.shape)
dfx.select(["txn_id"] + HISTORY_FEATURES).write_parquet("/kaggle/working/features_v2.parquet")

# %% Cell 7: XGBoost v2 = v1 tabular features + history/graph features
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import precision_recall_curve

from papertrail.features import build_features, fit_categories

train, val, test = temporal_split(dfx)     # same split, now with the extra columns
cats = fit_categories(train)


def to_matrix(part):
    base = build_features(part, cats)
    return pl.concat([base, part.select(HISTORY_FEATURES)], how="horizontal").cast(pl.Float32)


Xtr_df, Xva_df, Xte_df = to_matrix(train), to_matrix(val), to_matrix(test)
feat_names = Xtr_df.columns
X_train, X_val, X_test = Xtr_df.to_numpy(), Xva_df.to_numpy(), Xte_df.to_numpy()   # nulls become NaN
y_train = train["label"].to_numpy()
spw = float((y_train == 0).sum() / y_train.sum())
print(X_train.shape, "scale_pos_weight =", round(spw, 1))

xgb_params = dict(
    tree_method="hist", n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    eval_metric="aucpr", early_stopping_rounds=50, n_jobs=-1, random_state=0,
)
model = xgb.XGBClassifier(**xgb_params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

p_val = model.predict_proba(X_val)[:, 1]
p_test = model.predict_proba(X_test)[:, 1]

with mlflow.start_run(run_name="xgb_v2_graph"):
    mlflow.log_params({k: v for k, v in xgb_params.items() if k != "eval_metric"})
    mlflow.log_param("n_features", len(feat_names))
    mlflow.log_param("best_iteration", int(model.best_iteration))
    xgb2_res = run_eval(p_val, p_test)
    mlflow.log_metrics(xgb2_res)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y, p in (("validation", y_val, p_val), ("test", y_test, p_test)):
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=name)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("xgb_v2_graph")
    ax.legend()
    mlflow.log_figure(fig, "pr_curve.png")
    plt.show()

    model.save_model("/kaggle/working/xgb_v2.json")
    mlflow.log_artifact("/kaggle/working/xgb_v2.json")
    imp = pl.DataFrame({"feature": feat_names, "gain": model.feature_importances_.tolist()}).sort(
        "gain", descending=True)
    imp.write_csv("/kaggle/working/xgb_v2_importance.csv")
    mlflow.log_artifact("/kaggle/working/xgb_v2_importance.csv")
pl.Config.set_tbl_rows(30)
print(imp)

# %% Cell 8: SHAP on a validation sample (a failure here must not lose the run above)
try:
    import shap

    rng = np.random.default_rng(0)
    pick = rng.choice(len(X_val), size=20_000, replace=False)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_val[pick])
    plt.figure()
    shap.summary_plot(sv, X_val[pick], feature_names=feat_names, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig("/kaggle/working/shap_v2.png", dpi=150)
    plt.show()
    with mlflow.start_run(run_name="xgb_v2_graph_shap"):
        mlflow.log_artifact("/kaggle/working/shap_v2.png")
except Exception as e:  # noqa: BLE001
    print("SHAP failed, continuing without it:", repr(e))

# %% Cell 9: comparison table (test split) + files to download
import json
import urllib.request

try:  # earlier results from the repo, so the table has all four rows
    url = "https://raw.githubusercontent.com/abhibarnwal707/PaperTrail/main/reports/baseline_metrics.json"
    old = json.load(urllib.request.urlopen(url))
except Exception as e:  # noqa: BLE001
    print("could not fetch baseline_metrics.json from GitHub:", repr(e))
    old = {}

runs = []
if "rules_baseline" in old:
    runs.append(("Rules (v1)", old["rules_baseline"]))
runs.append(("Rules v2", rules2_res))
if "xgb_v1_tabular" in old:
    runs.append(("XGB tabular", old["xgb_v1_tabular"]))
runs.append(("XGB + graph (v2)", xgb2_res))

table = pl.DataFrame([{
    "Model": n,
    "PR-AUC": r["test_pr_auc"], "F1 (minority)": r["test_f1"],
    "Recall@1%FPR": r["test_recall_at_1pct_fpr"], "Precision@1000": r["test_precision_at_1000"],
    "ACH-only PR-AUC": r["test_ach_pr_auc"], "ACH-only F1": r["test_ach_f1"],
} for n, r in runs])
print(f"test positives: {int(y_test.sum())} of {len(y_test)} (base rate {y_test.mean():.4%})")
print(table)

table.write_csv("/kaggle/working/comparison_table.csv")
with open("/kaggle/working/phase4_metrics.json", "w") as fh:
    json.dump({"rules_v2": rules2_res, "xgb_v2_graph": xgb2_res, "rule_params_v2": params_v2,
               "burst_candidates": burst_table.to_dicts(), "scale_pos_weight": spw,
               "best_iteration": int(model.best_iteration)}, fh, indent=2)
print("commit message idea:",
      f"Add as-of + PageRank features, XGB v2: PR-AUC {old.get('xgb_v1_tabular', {}).get('test_pr_auc', float('nan')):.3f}"
      f" -> {xgb2_res['test_pr_auc']:.3f}")
# Download comparison_table.csv, phase4_metrics.json, xgb_v2_importance.csv and shap_v2.png from the Output tab
# into PaperTrail/reports/ (features_v2.parquet is large: keep it in Kaggle, do not commit it).
