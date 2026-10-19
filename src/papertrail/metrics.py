"""Evaluation metrics for rare-event detection (laundering is about 0.1% of rows).

Accuracy is never reported. Every model is logged with the names in METRIC_NAMES.
The threshold is chosen on validation only (`best_f1_threshold`) and then applied
unchanged to test.
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

METRIC_NAMES = ("pr_auc", "f1", "recall_at_1pct_fpr", "precision_at_1000")


def _prepare(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y and p must have the same length, got {y.shape} and {p.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be binary (0/1)")
    return y, p


def pr_auc(y, p) -> float:
    """Area under the precision-recall curve (average precision)."""
    y, p = _prepare(y, p)
    if y.sum() == 0:
        raise ValueError("y has no positives, PR-AUC is undefined")
    return float(average_precision_score(y, p))


def f1_at(y, p, thr: float) -> float:
    """Minority-class (label 1) F1 when flagging every row with score >= thr."""
    y, p = _prepare(y, p)
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def best_f1_threshold(y_val, p_val) -> float:
    """Score threshold that maximises minority-class F1. Call on VALIDATION data only."""
    y, p = _prepare(y_val, p_val)
    if y.sum() == 0:
        raise ValueError("y_val has no positives, cannot choose a threshold")
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1))])


def recall_at_fpr(y, p, fpr: float = 0.01) -> float:
    """Highest recall reachable while the false-positive rate stays <= `fpr`."""
    y, p = _prepare(y, p)
    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("y needs both classes")
    fpr_curve, tpr_curve, _ = roc_curve(y, p, drop_intermediate=False)
    return float(tpr_curve[fpr_curve <= fpr].max())


def precision_at_k(y, p, k: int) -> float:
    """Share of true positives among the k highest-scored rows (k is capped at the data size)."""
    y, p = _prepare(y, p)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, y.size)
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].mean())


def evaluate(y, p, thr: float, k: int = 1000, fpr: float = 0.01) -> dict[str, float]:
    """The standard metric set, under the fixed names in METRIC_NAMES (log these to MLflow)."""
    return {
        "pr_auc": pr_auc(y, p),
        "f1": f1_at(y, p, thr),
        "recall_at_1pct_fpr": recall_at_fpr(y, p, fpr),
        "precision_at_1000": precision_at_k(y, p, k),
    }
