"""The evaluation gate: decides whether a candidate model may replace the current production model.

Pure logic (no MLflow), so every rule is unit-tested. A candidate is promoted only if all three checks pass:
  1. PR-AUC is not worse than the current model by more than `tolerances.pr_auc`;
  2. recall at 1% FPR is not worse by more than `tolerances.recall_at_1pct_fpr`;
  3. the point-in-time leakage check passed (no feature uses information from after the prediction time).
With no current model (first registration) the metric checks are skipped and only the leakage check applies.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from papertrail.graph_features import add_history_features
from papertrail.lifecycle.config import GateTolerances

GATED_METRICS = ("pr_auc", "recall_at_1pct_fpr")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    checks: tuple[Check, ...]

    @property
    def reasons(self) -> list[str]:
        """Why the candidate was rejected (empty when promoted)."""
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {"promote": self.promote, "checks": [asdict(c) for c in self.checks], "reasons": self.reasons}


def evaluate_gate(candidate: dict[str, float], current: dict[str, float] | None, tolerances: GateTolerances, leakage_ok: bool,
                  leakage_detail: str = "") -> GateDecision:
    """Apply the three checks. `candidate` / `current` map metric names to values on the SAME evaluation data."""
    checks: list[Check] = []
    for metric in GATED_METRICS:
        tol = getattr(tolerances, metric)
        if current is None:
            checks.append(Check(metric, True, "no current production model, nothing to compare against"))
            continue
        cand, cur = candidate[metric], current[metric]
        ok = cand >= cur - tol
        checks.append(Check(metric, ok, f"candidate {cand:.4f} vs current {cur:.4f} (tolerance {tol}, delta {cand - cur:+.4f})"))
    checks.append(Check("point_in_time_leakage", leakage_ok, leakage_detail or ("passed" if leakage_ok else "failed")))
    return GateDecision(all(c.passed for c in checks), tuple(checks))


def check_point_in_time(txns: pl.DataFrame, features: list[str], probe_rows: int = 200, seed: int = 0,
                        builder: Callable[[pl.DataFrame], pl.DataFrame] = add_history_features) -> tuple[bool, str]:
    """Perturbation test for temporal leakage in the history features.

    Rebuild the features on the data truncated at a cut-off time and on the full frame. For every row strictly before the cut-off the
    feature values must be identical: if a feature used any later transaction, the two builds would disagree. Also, every feature
    named in `features` that the history builder produces must be covered (a missing one is a failure, not a skip).
    `builder` is the feature function under test (the production one by default). `txns` needs txn_id, ts, src, dst, amt_paid, from_bank, to_bank; it is sorted by ts here.
    """
    df = txns.sort(["ts", "txn_id"])
    if df.height < 20:
        return False, "too few rows to run the check"
    cut = df["ts"][int(df.height * 0.6)]
    full = builder(df)
    early = builder(df.filter(pl.col("ts") < cut))
    covered = [f for f in features if f in full.columns]
    missing = [f for f in features if f not in covered and f.startswith(("s_", "d_", "has_reverse"))]
    if missing:
        return False, f"history features not produced by the builder: {missing[:5]}"
    ref = full.filter(pl.col("ts") < cut).sort("txn_id")
    got = early.sort("txn_id")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(ref.height, size=min(probe_rows, ref.height), replace=False))
    bad = []
    for f in covered:
        a = ref[f].cast(pl.Float64).to_numpy()[idx]
        b = got[f].cast(pl.Float64).to_numpy()[idx]
        if not np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True):
            bad.append(f)
    if bad:
        return False, f"features changed when future rows were removed (leakage): {bad}"
    return True, f"{len(covered)} history features identical with and without future rows ({len(idx)} probe rows before the cut-off)"
