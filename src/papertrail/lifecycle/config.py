"""Lifecycle settings in one place. Defaults can be overridden by environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GateTolerances:
    """How much worse than the current model a candidate may be (absolute metric units) and still be promoted."""
    pr_auc: float = 0.01
    recall_at_1pct_fpr: float = 0.02


@dataclass(frozen=True)
class LifecycleConfig:
    tracking_uri: str = field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{Path('data/local/mlflow.db')}"))
    model_name: str = field(default_factory=lambda: os.environ.get("PAPERTRAIL_REGISTERED_MODEL", "papertrail-xgb"))
    experiment: str = field(default_factory=lambda: os.environ.get("PAPERTRAIL_EXPERIMENT", "papertrail-lifecycle"))
    tolerances: GateTolerances = field(default_factory=lambda: GateTolerances(
        pr_auc=float(os.environ.get("GATE_TOL_PR_AUC", "0.01")),
        recall_at_1pct_fpr=float(os.environ.get("GATE_TOL_RECALL_1FPR", "0.02"))))
