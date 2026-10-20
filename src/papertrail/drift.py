"""Drift monitoring: Population Stability Index per feature and per score, and the retraining rule built on it.

PSI compares a reference sample (what the model was fitted or tuned on) with a current sample. The bins are the
reference's quantiles, missing values get a bin of their own, and empty bins are floored at `eps` so the log is defined.
Usual reading: < 0.1 stable, 0.1-0.25 moderate shift, >= 0.25 major shift.

The data only starts on Sep 1, so some features change meaning as the calendar advances (reports/eda_summary.md 5d):
7-day windows are never full in train, all-time totals and "distinct so far" counts only grow, PageRank and the
reverse-edge flag see a bigger graph, account age is a calendar artefact. Their drift is expected by construction and
is reported but kept out of the retraining rule; only the calendar-stable features (payment details, <= 24 h windows,
the gap since the last event) can trigger it.
"""
from __future__ import annotations

import numpy as np
import polars as pl

STABLE, MODERATE = 0.1, 0.25                       # PSI bands: < 0.1 stable, 0.1-0.25 moderate, >= 0.25 major
RETRAIN_PSI, RETRAIN_COUNT, RETRAIN_TOP_K = 0.2, 3, 15


def is_calendar_dependent(feature: str) -> bool:
    """True when the feature's distribution moves with how long the data has run, not with behaviour."""
    return feature.endswith(("_7d", "_pagerank", "_age_s")) or "_tot_" in feature or "_uniq_" in feature or feature == "has_reverse_before"


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of `current` against `reference` (1-D numeric; NaN/None are their own bin)."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    if ref.size == 0 or cur.size == 0:
        raise ValueError("reference and current must be non-empty")
    ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1)[1:-1])) if ref_ok.size else np.array([])

    def shares(x_ok: np.ndarray, n_total: int) -> np.ndarray:
        counts = np.bincount(np.searchsorted(edges, x_ok, side="right"), minlength=edges.size + 1)
        return np.append(counts, n_total - x_ok.size) / n_total            # last bin: missing values

    r, c = shares(ref_ok, ref.size), shares(cur_ok, cur.size)
    r, c = np.maximum(r, eps), np.maximum(c, eps)
    return float(np.sum((c - r) * np.log(c / r)))


def band(value: float) -> str:
    return "stable" if value < STABLE else "moderate" if value < MODERATE else "major"


def drift_table(reference: pl.DataFrame, current: pl.DataFrame, features: list[str], importance: dict | None = None) -> pl.DataFrame:
    """One row per feature: PSI, its band, whether it is calendar-dependent, means and missing shares on both sides,
    and the model importance (share of total gain) when given. Sorted by importance, then PSI."""
    importance = importance or {}
    rows = []
    for f in features:
        r, c = reference[f].cast(pl.Float64).to_numpy(), current[f].cast(pl.Float64).to_numpy()
        p = psi(r, c)
        rows.append({"feature": f, "psi": p, "band": band(p), "calendar_dependent": is_calendar_dependent(f),
                     "importance": float(importance.get(f, 0.0)),
                     "ref_mean": float(np.nanmean(r)) if np.isfinite(r).any() else None, "cur_mean": float(np.nanmean(c)) if np.isfinite(c).any() else None,
                     "ref_missing": float(np.isnan(r).mean()), "cur_missing": float(np.isnan(c).mean())})
    return pl.DataFrame(rows).sort(["importance", "psi"], descending=True)


def daily_psi(reference: pl.DataFrame, frame: pl.DataFrame, features: list[str], day_col: str = "day") -> pl.DataFrame:
    """PSI of each day in `frame` against the reference, for each feature (long format: day, feature, psi)."""
    out = []
    for (day,), part in frame.partition_by(day_col, as_dict=True, maintain_order=True).items():
        for f in features:
            out.append({"day": day, "feature": f, "psi": psi(reference[f].cast(pl.Float64).to_numpy(), part[f].cast(pl.Float64).to_numpy())})
    return pl.DataFrame(out).sort(["feature", "day"])


def retrain_decision(table: pl.DataFrame, top_k: int = RETRAIN_TOP_K, psi_limit: float = RETRAIN_PSI, count_limit: int = RETRAIN_COUNT) -> dict:
    """The retraining rule: retrain when at least `count_limit` of the `top_k` most important calendar-stable
    features have PSI > `psi_limit`. Calendar-dependent features are listed but never trigger it."""
    stable = table.filter(~pl.col("calendar_dependent")).sort("importance", descending=True).head(top_k)
    drifted = stable.filter(pl.col("psi") > psi_limit)["feature"].to_list()
    expected = table.filter(pl.col("calendar_dependent") & (pl.col("psi") > psi_limit))["feature"].to_list()
    return {"retrain": len(drifted) >= count_limit, "drifted": drifted, "expected_calendar_drift": expected,
            "rule": f"retrain if PSI > {psi_limit} on {count_limit}+ of the top {top_k} calendar-stable features"}
