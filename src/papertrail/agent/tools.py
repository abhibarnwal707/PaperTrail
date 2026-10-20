"""The investigator's tools. Each reads the demo bundle (data/demo/) and returns plain JSON tagged with a `source_id`,
so every claim in a case summary can be traced to the tool output it came from.

Rules that keep the agent honest:
- the ground truth never leaves this module: `label`, `typology_truth`, `n_laundering`, `tag` and `typologies_truth`
  are in the bundle for the app's hidden answer key and evaluation only;
- numbers are rounded once, here (money to 2 decimals, scores to 2), so the verifier can check that a claim repeats
  a number that is really in the cited output.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

MODELS = ("blend", "xgb", "gnn")
HIDDEN = {"label", "typology_truth", "n_laundering", "tag", "typologies_truth"}   # never shown to the model


def _r(x, d: int = 2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


class Toolbox:
    """The four tools over one demo bundle. Construct once; the tools are cheap after that."""

    def __init__(self, root: Path | str):
        root = Path(root)
        self.alerts = pl.read_parquet(root / "alerts.parquet")
        self.edges = pl.read_parquet(root / "edges.parquet")
        self.history = pl.read_parquet(root / "history.parquet")
        self.shap = pl.read_parquet(root / "shap.parquet")
        self.txns = pl.read_parquet(root / "txns.parquet").drop([c for c in HIDDEN if c in pl.read_parquet_schema(root / "txns.parquet")])
        self.labels = json.loads((root / "feature_labels.json").read_text())
        self.thresholds = json.loads((root / "blend.json").read_text())["thresholds_z"]
        self.accounts = set(self.alerts["account"].to_list())

    def _check(self, account: str) -> None:
        if account not in self.accounts:
            raise KeyError(f"unknown account {account!r}: the tools only cover the {len(self.accounts)} accounts in the alert queue")

    # ------------------------------------------------------------------ get_alert
    def get_alert(self, account: str) -> dict:
        """Where the account sits in the queue under each model, and how many of its payments each model flags."""
        self._check(account)
        r = self.alerts.filter(pl.col("account") == account).to_dicts()[0]
        return {"source_id": f"alert:{account}", "account": account, "accounts_in_queue": self.alerts.height,
                "payments_in_window": r["n_txn"],
                "models": {m: {"rank": r[f"{m}_rank"], "risk_score": _r(r[f"{m}_risk"]), "alert_line": _r(self.thresholds[m]),
                               "flagged_payments": r[f"n_flag_{m}"]} for m in MODELS},
                "note": "risk_score is the account's riskiest payment on a standardised scale; a payment is flagged when it is at or above the alert_line"}

    # ------------------------------------------------------------------ get_subgraph
    def get_subgraph(self, account: str) -> dict:
        """Shape of the payment network around the account (summary statistics, not raw edges)."""
        self._check(account)
        e = self.edges.filter(pl.col("center") == account).join(self.txns, on="txn_id", how="inner")
        out, inn = e.filter(pl.col("src") == account), e.filter(pl.col("dst") == account)
        receivers, senders = set(out["dst"].to_list()) - {account}, set(inn["src"].to_list()) - {account}
        pairs = {(s, d) for s, d in zip(e["src"].to_list(), e["dst"].to_list())}
        two_cycles = sorted(x for x in receivers if (x, account) in pairs)
        three_cycles = sorted({(x, y) for x in receivers for (a, y) in pairs if a == x and y not in (account, x) and (y, account) in pairs})
        # scatter-gather: several of the account's receivers pay the same next account; gather-scatter mirror image upstream
        downstream = Counter(d for (s, d) in pairs if s in receivers and d != account and d not in receivers)
        upstream = Counter(s for (s, d) in pairs if d in senders and s != account and s not in senders)
        fl = e.filter(pl.col("blend_flag"))
        return {"source_id": f"subgraph:{account}", "account": account,
                "payments_shown": e.height, "direct_payments_sent": out.height, "direct_payments_received": inn.height,
                "fan_out_distinct_receivers": len(receivers), "fan_in_distinct_senders": len(senders),
                "counterparties_both_directions": len(receivers & senders),
                "two_step_cycles": len(two_cycles), "three_step_cycles": len(three_cycles),
                "max_receivers_paying_one_account": max(downstream.values(), default=0),
                "max_senders_paid_by_one_account": max(upstream.values(), default=0),
                "flagged_payments_in_neighbourhood": fl.height,
                "flagged_payments_touching_account": fl.filter((pl.col("src") == account) | (pl.col("dst") == account)).height,
                "banks_involved": len(set(e["from_bank"].to_list()) | set(e["to_bank"].to_list())),
                "payment_formats": dict(Counter(e["pay_format"].to_list()).most_common()),
                "currencies": sorted(set(e["cur_paid"].to_list()) | set(e["cur_recv"].to_list())),
                "note": "two-hop neighbourhood, capped at the riskiest payments, so counts are lower bounds"}

    # ------------------------------------------------------------------ get_shap
    def get_shap(self, account: str) -> dict:
        """What pushed the XGBoost score of the account's riskiest payment up or down."""
        self._check(account)
        s = self.shap.filter(pl.col("account") == account)
        drivers = []
        for r in s.to_dicts():
            info = self.labels.get(r["feature"], {"label": r["feature"], "unit": ""})
            drivers.append({"factor": info["label"] if r["feature"] != "log_amt" else "Payment amount",
                            "value": _value(r["feature"], r["value"], info["unit"]), "push": _r(r["contribution"]),
                            "direction": "raises risk" if r["contribution"] > 0 else "lowers risk"})
        drivers.sort(key=lambda d: -abs(d["push"] or 0))
        return {"source_id": f"shap:{account}", "account": account, "payment_id": int(s["txn_id"][0]) if s.height else None,
                "explains": "the account's riskiest payment (its highest blend score), not necessarily its most recent one",
                "drivers": drivers, "note": "push is the change in the XGBoost log-odds; positive raises risk"}

    # ------------------------------------------------------------------ get_txn_history
    def get_txn_history(self, account: str, n: int = 20) -> dict:
        """The account's most recent payments, newest first."""
        self._check(account)
        h = (self.history.filter(pl.col("account") == account).join(self.txns, on="txn_id", how="inner").sort("rank").head(n))
        rows = []
        for r in h.to_dicts():
            sent = r["src"] == account
            rows.append({"time": str(r["ts"])[:16], "direction": "sent" if sent else "received", "counterparty": r["dst"] if sent else r["src"],
                         "amount": _r(r["amt_paid"]), "currency": r["cur_paid"], "format": r["pay_format"],
                         "flagged_by_blend": bool(r["blend_flag"]), "blend_score": _r(r["blend_z"])})
        amounts = [x["amount"] for x in rows if x["amount"] is not None]
        return {"source_id": f"history:{account}", "account": account, "payments": rows, "count": len(rows),
                "sent": sum(x["direction"] == "sent" for x in rows), "received": sum(x["direction"] == "received" for x in rows),
                "flagged": sum(x["flagged_by_blend"] for x in rows),
                "largest_amount": max(amounts, default=None), "smallest_amount": min(amounts, default=None)}

    def gather(self, account: str) -> dict[str, dict]:
        """All four tools, keyed by source_id."""
        outs = [self.get_alert(account), self.get_subgraph(account), self.get_shap(account), self.get_txn_history(account)]
        return {o["source_id"]: o for o in outs}


def _value(feature: str, value, unit: str):
    """A feature value in the unit a person reads: the amount instead of its log, hours instead of seconds, yes/no for flags."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not available"
    if feature == "log_amt":
        return _r(np.expm1(value))
    if unit == "seconds":
        return f"{value / 3600:.1f} hours"
    if unit == "flag":
        return "yes" if value >= 0.5 else "no"
    return _r(value)
