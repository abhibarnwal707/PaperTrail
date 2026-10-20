"""Pure logic for the Streamlit app (no streamlit import, so it is unit-tested): read the demo bundle, turn the friendly
"try a transaction" form into model features, explain a score in plain language, and lay out an account's network.

The bundle is produced by notebooks/06-demo-slice.ipynb (see the playbook, Phase 6).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

MODELS = {"blend": "Blend (GNN + XGBoost)", "xgb": "XGBoost", "gnn": "GNN (graph neural network)"}
MODEL_BLURB = {
    "blend": "Combines the two models below. The most accurate option.",
    "xgb": "Looks at each account's recent behaviour (how often it sends, to how many people, how recently).",
    "gnn": "Looks at the web of payments around the accounts, not at their individual statistics.",
}


@dataclass
class Bundle:
    root: Path
    alerts: pl.DataFrame
    edges: pl.DataFrame
    history: pl.DataFrame
    shap: pl.DataFrame
    txns: pl.DataFrame
    truth: pl.DataFrame
    metrics: dict
    labels: dict
    profiles: dict
    categories: dict
    meta: dict
    _boosters: list = field(default_factory=list, repr=False)

    @property
    def boosters(self) -> list:
        """The XGBoost models, loaded on first use. Empty if xgboost cannot be loaded here."""
        if not self._boosters:
            try:
                import xgboost as xgb

                for fname in self.meta["files"]:
                    b = xgb.Booster()
                    b.load_model(str(self.root / "model" / fname))
                    self._boosters.append(b)
            except Exception:  # noqa: BLE001
                self._boosters = []
        return self._boosters


def bundle_present(root: Path) -> bool:
    return (Path(root) / "alerts.parquet").exists() and (Path(root) / "model" / "meta.json").exists()


def load_bundle(root: Path | str) -> Bundle:
    root = Path(root)
    pq = {n: pl.read_parquet(root / f"{n}.parquet") for n in ("alerts", "edges", "history", "shap", "txns", "truth")}
    js = {n: json.loads((root / f"{n}.json").read_text()) for n in ("metrics", "profiles", "categories")}
    return Bundle(root=root, **pq, metrics=js["metrics"], labels=json.loads((root / "feature_labels.json").read_text()),
                  profiles=js["profiles"], categories=js["categories"], meta=json.loads((root / "model" / "meta.json").read_text()))


# ---------------------------------------------------------------- plain-language formatting

def human(value, unit: str) -> str:
    """A feature value in words a non-technical reader understands."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not available"
    if unit == "seconds":
        return f"{value / 3600:.1f} hours" if value >= 3600 else f"{value / 60:.0f} minutes"
    if unit == "flag":
        return "yes" if value >= 0.5 else "no"
    if unit == "amount":
        return f"{value:,.0f}"
    if unit == "hour":
        return f"{int(value):02d}:00"
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.4g}"


def describe(labels: dict, feature: str, value) -> tuple[str, str]:
    """(plain label, plain value) for one feature. The model sees log amount; people should see the amount."""
    info = labels.get(feature, {"label": feature, "unit": ""})
    if feature == "log_amt":
        return "Payment amount", "not available" if value is None else f"{np.expm1(value):,.0f}"
    return info["label"], human(value, info["unit"])


def risk_percent_text(pct: float | None) -> str:
    """'Top 0.02% riskiest' from a percentile in [0, 1]."""
    if pct is None or np.isnan(pct):
        return "n/a"
    top = max((1.0 - pct) * 100, 0.0)
    return f"top {top:.2g}%" if top >= 0.01 else "top 0.01%"


def risk_percentile(b: Bundle, model: str, z: float) -> float:
    """Percentile of a model score within the test window, interpolated from the (score, percentile) pairs stored with the bundle."""
    t = b.txns.filter(pl.col(f"{model}_z").is_not_null()).select(f"{model}_z", f"{model}_pct").sort(f"{model}_z")
    if t.height == 0:
        return float("nan")
    return float(np.interp(z, t[f"{model}_z"].to_numpy(), t[f"{model}_pct"].to_numpy()))


# ---------------------------------------------------------------- alert queue and account pages

def alert_table(b: Bundle, model: str, show_truth: bool = False) -> pl.DataFrame:
    a = b.alerts.sort(f"{model}_rank")
    pct = [risk_percent_text(risk_percentile(b, model, z)) for z in a[f"{model}_risk"].to_list()]
    why = [why_text(b, acct) for acct in a["account"].to_list()]
    out = pl.DataFrame({
        "Rank": a[f"{model}_rank"], "Account": a["account"], "How risky": pct,
        "Flagged transactions": a[f"n_flag_{model}"], "Transactions in window": a["n_txn"], "Why it stands out": why,
    })
    if show_truth:
        out = out.with_columns(a["n_laundering"].alias("Actually laundering (transactions)"), a["typologies_truth"].alias("True pattern"),
                               a["tag"].alias("Why it was picked for the demo"))
    return out


def queue_findings(b: Bundle, model: str) -> dict:
    """The facts the queue's title row states: how many accounts each model flags, and where the models disagree."""
    a = b.alerts
    flagged = {m: int((a[f"n_flag_{m}"] > 0).sum()) for m in MODELS}
    every = int(((a["n_flag_blend"] > 0) & (a["n_flag_xgb"] > 0) & (a["n_flag_gnn"] > 0)).sum())
    return {"accounts": a.height, "flagged": flagged, "every": every,
            "gnn_only": int(((a["n_flag_gnn"] > 0) & (a["n_flag_xgb"] == 0)).sum()),
            "xgb_only": int(((a["n_flag_xgb"] > 0) & (a["n_flag_gnn"] == 0)).sum()),
            "top": a.sort(f"{model}_rank")["account"][0]}


def model_findings(b: Bundle) -> dict:
    """The facts the model page's title row states, taken from metrics.json."""
    m = b.metrics["models"]
    best = max(("xgb", "gnn"), key=lambda k: m[k]["recall_at_1pct_fpr"])
    return {"blend_recall": m["blend"]["recall_at_1pct_fpr"], "best_single": best, "best_single_recall": m[best]["recall_at_1pct_fpr"],
            "blend_pr": m["blend"]["pr_auc"], "best_single_pr": max(m["xgb"]["pr_auc"], m["gnn"]["pr_auc"])}


def account_reasons(b: Bundle, account: str, top: int = 8) -> pl.DataFrame:
    """The account's biggest drivers with plain labels (from the XGBoost models' attributions)."""
    s = b.shap.filter(pl.col("account") == account)
    rows = []
    for r in s.to_dicts():
        label, value_text = describe(b.labels, r["feature"], r["value"])
        rows.append({"feature": r["feature"], "label": label, "value_text": value_text,
                     "contribution": r["contribution"], "direction": "raises" if r["contribution"] > 0 else "lowers"})
    out = pl.DataFrame(rows, schema={"feature": pl.String, "label": pl.String, "value_text": pl.String, "contribution": pl.Float64, "direction": pl.String})
    return out.with_columns(pl.col("contribution").abs().alias("_a")).sort("_a", descending=True).drop("_a").head(top)


def why_text(b: Bundle, account: str, n: int = 2) -> str:
    """One line for the queue: the account's strongest drivers in words, each with its signed push on the score."""
    r = account_reasons(b, account, top=n)
    return "; ".join(f"{x['label']}: {x['value_text']} ({x['contribution']:+.2f})" for x in r.to_dicts())


def account_history(b: Bundle, account: str) -> pl.DataFrame:
    h = b.history.filter(pl.col("account") == account).join(b.txns, on="txn_id", how="left").sort("rank")
    return h.with_columns(
        pl.when(pl.col("src") == account).then(pl.lit("sent")).otherwise(pl.lit("received")).alias("direction"),
        pl.when(pl.col("src") == account).then(pl.col("dst")).otherwise(pl.col("src")).alias("counterparty"),
    )


def account_edges(b: Bundle, account: str) -> pl.DataFrame:
    return b.edges.filter(pl.col("center") == account).join(b.txns, on="txn_id", how="left")


def ring_layout(edges: pl.DataFrame, center: str) -> tuple[dict, list[dict]]:
    """Deterministic layout: the account in the middle, its direct counterparties on an inner ring, the rest on an outer ring
    placed next to the inner node they connect to. Returns ({node: (x, y, hop)}, [edge dict])."""
    rows = edges.to_dicts()
    direct: dict[str, int] = {}
    for r in rows:
        if r["src"] == center and r["dst"] != center:
            direct[r["dst"]] = direct.get(r["dst"], 0) + 1
        elif r["dst"] == center and r["src"] != center:
            direct[r["src"]] = direct.get(r["src"], 0) + 1
    inner = sorted(direct, key=lambda n: (-direct[n], n))
    pos: dict[str, tuple[float, float, int]] = {center: (0.0, 0.0, 0)}
    angle = {}
    for i, n in enumerate(inner):
        angle[n] = 2 * np.pi * i / max(len(inner), 1)
        pos[n] = (float(np.cos(angle[n])), float(np.sin(angle[n])), 1)
    outer: dict[str, list[str]] = {}
    for r in rows:
        for a, c in ((r["src"], r["dst"]), (r["dst"], r["src"])):
            if a in direct and c not in pos and c != center:
                outer.setdefault(a, []).append(c)
    placed = set()
    for parent, kids in outer.items():
        kids = [k for k in sorted(set(kids)) if k not in placed]
        for j, k in enumerate(kids):
            spread = (j - (len(kids) - 1) / 2) * 0.12
            pos[k] = (float(2.0 * np.cos(angle[parent] + spread)), float(2.0 * np.sin(angle[parent] + spread)), 2)
            placed.add(k)
    stray = sorted({n for r in rows for n in (r["src"], r["dst"])} - set(pos))
    for i, n in enumerate(stray):                                   # endpoints with no link to the centre's neighbours
        pos[n] = (float(2.4 * np.cos(2 * np.pi * i / max(len(stray), 1))), float(2.4 * np.sin(2 * np.pi * i / max(len(stray), 1))), 2)
    return pos, rows


# ---------------------------------------------------------------- the "try a transaction" form

def whatif_features(b: Bundle, *, amount: float, pay_format: str, pay_currency: str, different_receive_currency: bool,
                    same_bank: bool, hour: int, profile: str, out_cnt_24h: float, uniq_out: float) -> dict:
    """Turn the form into the model's feature dictionary. History features come from the chosen sender profile; the two sliders
    override the sender's 24-hour send count and number of different recipients, and related counts are kept consistent."""
    cat = b.categories
    f: dict = {n: None for n in cat["feature_names"]}
    lo, hi = cat["amount_band"]
    f.update({"log_amt": float(np.log1p(amount)), "hour": float(hour), "is_ach": float(pay_format == "ACH"),
              "is_cross_currency": float(different_receive_currency), "is_same_bank": float(same_bank), "is_self_transfer": 0.0,
              "is_round_amount": float(amount % 100 == 0), "in_amount_band": float(lo <= amount <= hi),
              "cur_code": float(cat["currencies"].get(pay_currency, -1))})
    for fmt in cat["pay_format"]:
        key = "fmt_" + fmt.replace(" ", "_")
        if key in f:
            f[key] = float(fmt == pay_format)
    prof = b.profiles["profiles"][profile]["features"]
    for k, v in {**b.profiles["receiver_default"], **prof}.items():
        if k in f:
            f[k] = v
    if profile == "new_account":                                    # no history at all: sliders do not apply
        out_cnt_24h, uniq_out = 0.0, 0.0
    n24 = float(out_cnt_24h)
    f["s_out_cnt_24h"] = n24
    f["s_out_cnt_7d"] = max(f.get("s_out_cnt_7d") or 0.0, n24)
    f["s_tot_out"] = max(f.get("s_tot_out") or 0.0, f["s_out_cnt_7d"], float(uniq_out))
    f["s_uniq_out"] = float(uniq_out)
    typical = f.get("s_out_mean_7d") or amount
    f["s_out_sum_24h"] = n24 * typical
    f["s_out_sum_7d"] = max(f.get("s_out_sum_7d") or 0.0, f["s_out_cnt_7d"] * typical)
    return f


def score_features(b: Bundle, feats: dict) -> dict:
    """Probability, alert decision and per-feature contributions (log-odds, averaged over the XGBoost models)."""
    import xgboost as xgb

    names = b.meta["feature_names"]
    x = np.array([[np.nan if feats.get(n) is None else feats[n] for n in names]], dtype=np.float32)
    d = xgb.DMatrix(x, feature_names=names)
    p = float(np.mean([m.predict(d)[0] for m in b.boosters]))
    contrib = np.mean([m.predict(d, pred_contribs=True)[0] for m in b.boosters], axis=0)
    rows = []
    for n, c in zip(names, contrib[:-1]):
        label, value_text = describe(b.labels, n, feats.get(n))
        rows.append({"feature": n, "label": label, "value_text": value_text,
                     "contribution": float(c), "direction": "raises" if c > 0 else "lowers"})
    reasons = pl.DataFrame(rows).with_columns(pl.col("contribution").abs().alias("_a")).sort("_a", descending=True).drop("_a")
    return {"probability": p, "threshold": float(b.meta["threshold"]), "alert": p >= float(b.meta["threshold"]), "reasons": reasons}
