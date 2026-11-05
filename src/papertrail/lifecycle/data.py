"""Dataset and (re)training on the demo slice, CPU-only.

The full 5.08M-row pipeline runs on Kaggle. Here the same feature code (`features.py` + `graph_features.py`) is applied to the committed
demo slice so the lifecycle (register -> gate -> promote -> rollback -> drift-triggered retrain) can run on a laptop and in CI.
Numbers from this module describe that mechanism on a small sample; they are NOT the headline model metrics.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

from papertrail.features import build_features
from papertrail.graph_features import add_history_features
from papertrail.metrics import best_f1_threshold, evaluate

DEMO = Path("data/demo")
RAW_COLS = ["txn_id", "ts", "src", "dst", "amt_paid", "cur_paid", "cur_recv", "pay_format", "from_bank", "to_bank", "label"]
XGB_PARAMS = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
              "tree_method": "hist", "eval_metric": "aucpr", "n_jobs": 2}


def feature_names(demo: Path = DEMO) -> list[str]:
    return json.loads((demo / "categories.json").read_text())["feature_names"]


def load_slice(demo: Path = DEMO) -> pl.DataFrame:
    return pl.read_parquet(demo / "txns.parquet", columns=RAW_COLS).sort(["ts", "txn_id"])


def feature_frame(txns: pl.DataFrame, demo: Path = DEMO) -> pl.DataFrame:
    """txn_id, ts, label and every model feature, built as-of each transaction (history features look strictly backwards)."""
    cats = json.loads((demo / "categories.json").read_text())
    categories = {"pay_format": cats["pay_format"], "cur_paid": sorted(cats["currencies"], key=cats["currencies"].get)}
    txns = txns.sort(["ts", "txn_id"])
    hist = add_history_features(txns, with_pagerank=True, as_of_nodes=True)
    tab = build_features(txns, categories)
    out = pl.concat([txns.select("txn_id", "ts", "label"), tab, hist.select([c for c in cats["history_features"]])], how="horizontal")
    return out.select("txn_id", "ts", "label", *cats["feature_names"])


def temporal_thirds(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """60/20/20 by row count in time order (the repo's split rule, applied to the slice)."""
    n = frame.height
    a, b = int(n * 0.6), int(n * 0.8)
    return frame[:a], frame[a:b], frame[b:]


def data_hash(txns: pl.DataFrame) -> str:
    """Stable fingerprint of the raw rows a model was trained on."""
    return hashlib.sha256(txns.sort("txn_id").write_csv().encode()).hexdigest()[:16]


def feature_list_hash(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names).encode()).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def train_model(frame: pl.DataFrame, out_dir: Path, seed: int = 0, name: str = "xgb_slice") -> dict:
    """Fit on the first 60%, choose the threshold on the next 20%, write `xgb.json` + `meta.json` (the API's model-dir shape).
    Returns the validation-based threshold and the held-out metrics for the last 20%."""
    import xgboost as xgb

    names = feature_names()
    train, val, test = temporal_thirds(frame)
    model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    model.fit(train.select(names).to_numpy().astype(np.float32), train["label"].to_numpy(),
              eval_set=[(val.select(names).to_numpy().astype(np.float32), val["label"].to_numpy())], verbose=False)
    thr = best_f1_threshold(val["label"].to_numpy(), model.predict_proba(val.select(names).to_numpy().astype(np.float32))[:, 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(out_dir / "xgb.json"))
    (out_dir / "meta.json").write_text(json.dumps({"feature_names": names, "threshold": thr, "model": name, "files": ["xgb.json"]}, indent=1))
    return {"threshold": thr, "metrics": evaluate_dir(out_dir, test)}


def evaluate_dir(model_dir: Path, frame: pl.DataFrame) -> dict[str, float]:
    """Score `frame` with the model in `model_dir` and return the standard metrics (threshold taken from its meta.json)."""
    import xgboost as xgb

    meta = json.loads((model_dir / "meta.json").read_text())
    p = np.zeros(frame.height)
    files = meta.get("files", ["xgb.json"])
    for f in files:
        b = xgb.Booster()
        b.load_model(str(model_dir / f))
        d = xgb.DMatrix(frame.select(meta["feature_names"]).to_numpy().astype(np.float32), feature_names=meta["feature_names"])
        p += b.predict(d) / len(files)
    return evaluate(frame["label"].to_numpy(), p, meta["threshold"])
