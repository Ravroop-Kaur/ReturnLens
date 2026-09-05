import numpy as np
from evaluation.metrics.classification import evaluate, confusion_counts


def test_confusion_counts_basic():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 1]
    c = confusion_counts(y_true, y_pred)
    assert c.tp == 1 and c.fn == 1 and c.tn == 1 and c.fp == 1


def test_precision_recall_f1_known_values():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.05, 0.6])
    m = evaluate(y_true, y_proba, threshold=0.5)
    # predicted positive: indices 0,1,3,7 -> TP=2 (0,1), FP=2 (3,7), FN=1 (2), TN=3
    assert m.counts.tp == 2
    assert m.counts.fp == 2
    assert m.counts.fn == 1
    assert m.counts.tn == 3
    assert abs(m.precision - 0.5) < 1e-9
    assert abs(m.recall - (2 / 3)) < 1e-9
    expected_f1 = 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)
    assert abs(m.f1 - expected_f1) < 1e-9


def test_zero_denominator_precision_when_no_predicted_positives():
    y_true = np.array([1, 0, 0])
    y_proba = np.array([0.1, 0.1, 0.1])  # nothing predicted positive at 0.5
    m = evaluate(y_true, y_proba, threshold=0.5)
    assert m.precision_defined is False
    assert np.isnan(m.precision)


def test_zero_denominator_recall_when_no_actual_positives():
    y_true = np.array([0, 0, 0])
    y_proba = np.array([0.9, 0.1, 0.8])
    m = evaluate(y_true, y_proba, threshold=0.5)
    assert m.recall_defined is False
    assert np.isnan(m.recall)


def test_fpr_fnr_zero_denominator_handling():
    # all actual negatives -> FNR undefined (no positives to miss)
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.9, 0.1, 0.8, 0.2])
    m = evaluate(y_true, y_proba, threshold=0.5)
    assert m.fnr_defined is False
    assert m.fpr_defined is True


def test_perfect_classifier():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.1, 0.2])
    m = evaluate(y_true, y_proba, threshold=0.5)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.fpr == 0.0
    assert m.fnr == 0.0
