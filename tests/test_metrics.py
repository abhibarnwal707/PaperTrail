import numpy as np
import pytest

from papertrail.metrics import (
    METRIC_NAMES,
    best_f1_threshold,
    evaluate,
    f1_at,
    pr_auc,
    precision_at_k,
    recall_at_fpr,
)

# Ranked by score: 0.8(pos), 0.4(neg), 0.35(pos), 0.1(neg)
Y = np.array([0, 0, 1, 1])
P = np.array([0.1, 0.4, 0.35, 0.8])


def test_pr_auc_hand_computed():
    # recall 0.5 at precision 1, recall 1.0 at precision 2/3 -> 0.5*1 + 0.5*(2/3)
    assert pr_auc(Y, P) == pytest.approx(5 / 6)


def test_pr_auc_perfect_and_worst_ranking():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert pr_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) < 0.6


def test_f1_at():
    # thr 0.35 flags 0.8, 0.4, 0.35 -> tp=2 fp=1 fn=0 -> P=2/3, R=1 -> F1=0.8
    assert f1_at(Y, P, 0.35) == pytest.approx(0.8)
    # thr 0.8 flags only 0.8 -> tp=1 fp=0 fn=1 -> F1 = 2/3
    assert f1_at(Y, P, 0.8) == pytest.approx(2 / 3)


def test_f1_at_nothing_flagged_is_zero():
    assert f1_at(Y, P, 0.99) == 0.0


def test_best_f1_threshold():
    thr = best_f1_threshold(Y, P)
    assert thr == pytest.approx(0.35)
    assert f1_at(Y, P, thr) == pytest.approx(0.8)


def test_recall_at_fpr():
    # 2 negatives. At fpr=0 only the 0.8 row is flagged -> recall 0.5.
    assert recall_at_fpr(Y, P, fpr=0.01) == pytest.approx(0.5)
    # Allowing fpr=0.5 (one negative) lets 0.35 in -> recall 1.0.
    assert recall_at_fpr(Y, P, fpr=0.5) == pytest.approx(1.0)


def test_precision_at_k():
    assert precision_at_k(Y, P, 1) == 1.0          # {0.8}
    assert precision_at_k(Y, P, 2) == 0.5          # {0.8, 0.4}
    assert precision_at_k(Y, P, 3) == pytest.approx(2 / 3)
    assert precision_at_k(Y, P, 1000) == 0.5       # k capped at n=4


def test_evaluate_uses_fixed_metric_names():
    out = evaluate(Y, P, thr=0.35, k=2)
    assert tuple(out) == METRIC_NAMES
    assert out["pr_auc"] == pytest.approx(5 / 6)
    assert out["f1"] == pytest.approx(0.8)
    assert out["precision_at_1000"] == 0.5  # k=2 here


@pytest.mark.parametrize("bad", [
    lambda: pr_auc([0, 1], [0.1]),               # length mismatch
    lambda: pr_auc([0, 0], [0.1, 0.2]),          # no positives
    lambda: pr_auc([0, 2], [0.1, 0.2]),          # not binary
    lambda: best_f1_threshold([0, 0], [0.1, 0.2]),
    lambda: recall_at_fpr([1, 1], [0.1, 0.2]),   # one class only
    lambda: precision_at_k(Y, P, 0),
])
def test_bad_input_raises(bad):
    with pytest.raises(ValueError):
        bad()
