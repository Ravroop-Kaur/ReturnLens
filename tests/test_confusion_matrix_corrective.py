"""
Focused corrective-pass tests for the confusion matrix / classification
evaluation (see the "confusion matrix corrective pass" spec). This suite
does NOT touch the model, features, training data, threshold, or
hyperparameters -- it only verifies the evaluation math, semantics, and
UI/API plumbing are correct, dynamic, and internally consistent.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from evaluation.metrics.classification import (
    confusion_counts,
    evaluate,
    evaluate_thresholds,
    POSITIVE_CLASS,
    NEGATIVE_CLASS,
)
from src.adapters.generic_csv import ingest_generic_csv
from src.model.train import train_return_risk_model
from src.features.engineering import build_features

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------
# 1-2. Class semantics: positive = return, negative = no_return, and
# TP/FP/FN/TN are never reversed.
# --------------------------------------------------------------------

def test_positive_class_is_return_negative_class_is_no_return():
    assert POSITIVE_CLASS == "return"
    assert NEGATIVE_CLASS == "no_return"


def test_tp_fp_fn_tn_semantics_explicit():
    # y_true: 1=actual return, 0=actual no-return.
    # y_pred: 1=predicted return, 0=predicted no-return.
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    #          TP FN FP TN TP TN
    c = confusion_counts(y_true, y_pred)
    assert c.tp == 2  # actual return + predicted return (indices 0, 4)
    assert c.fn == 1  # actual return + predicted no-return (index 1)
    assert c.fp == 1  # actual no-return + predicted return (index 2)
    assert c.tn == 2  # actual no-return + predicted no-return (indices 3, 5)


# --------------------------------------------------------------------
# 3. Metric calculations (formulas already covered in
# test_evaluation_metrics.py; here we specifically check the numbers
# named in the corrective-pass spec for the current synthetic result).
# --------------------------------------------------------------------

def test_reported_synthetic_metrics_are_self_consistent():
    """The report is the source of truth; never pin tests to stale golden numbers."""
    report = json.loads((REPO_ROOT / "evaluation/reports/full_report.json").read_text())
    ev = report["test_evaluation"]
    c = ev["counts"]
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == ev["n_total"]
    expected_precision = c["tp"] / (c["tp"] + c["fp"])
    expected_recall = c["tp"] / (c["tp"] + c["fn"])
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    assert abs(ev["precision"] - expected_precision) < 1e-12
    assert abs(ev["recall"] - expected_recall) < 1e-12
    assert abs(ev["f1"] - expected_f1) < 1e-12


# --------------------------------------------------------------------
# 4. sklearn cross-check.
# --------------------------------------------------------------------

def test_confusion_counts_matches_sklearn_on_synthetic_data():
    np.random.seed(0)
    y_true = np.random.randint(0, 2, size=2000)
    y_proba = np.clip(y_true * 0.5 + np.random.normal(0, 0.3, size=2000) + 0.3, 0, 1)
    threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)

    ours = confusion_counts(y_true, y_pred)
    # sklearn's confusion_matrix(y_true, y_pred) returns
    # [[tn, fp], [fn, tp]] for labels=[0, 1].
    sk = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    sk_tn, sk_fp, sk_fn, sk_tp = sk[0, 0], sk[0, 1], sk[1, 0], sk[1, 1]

    assert ours.tp == sk_tp
    assert ours.fp == sk_fp
    assert ours.fn == sk_fn
    assert ours.tn == sk_tn


def test_synthetic_holdout_confusion_matrix_matches_sklearn():
    """Cross-check the actual held-out predictions against sklearn.
    No fixed TP/FP/FN/TN values are assumed."""
    ingestion = ingest_generic_csv(REPO_ROOT / "data/sample/generic_merchant_orders.csv")
    bundle = train_return_risk_model(ingestion.canonical_df)
    y_test = bundle._test_y.values
    p_test = bundle.predict_proba(bundle._test_X)
    y_pred = (p_test >= bundle.threshold).astype(int)

    ours = confusion_counts(y_test, y_pred)
    sk = sk_confusion_matrix(y_test, y_pred, labels=[0, 1])
    sk_tn, sk_fp, sk_fn, sk_tp = sk[0, 0], sk[0, 1], sk[1, 0], sk[1, 1]

    assert ours.tp == sk_tp
    assert ours.fp == sk_fp
    assert ours.fn == sk_fn
    assert ours.tn == sk_tn


# --------------------------------------------------------------------
# 5. Sample-accounting invariants.
# --------------------------------------------------------------------

def test_matrix_sums_to_test_set_size():
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 0, 0, 0, 1]
    c = confusion_counts(y_true, y_pred)
    assert c.tp + c.fp + c.fn + c.tn == len(y_true)


def test_matrix_matches_actual_and_predicted_totals():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 1])
    c = confusion_counts(y_true, y_pred)
    assert c.tp + c.fn == int((y_true == 1).sum())
    assert c.fp + c.tn == int((y_true == 0).sum())
    assert c.tp + c.fp == int((y_pred == 1).sum())
    assert c.fn + c.tn == int((y_pred == 0).sum())


def test_invariants_hold_on_real_synthetic_holdout():
    report = json.loads((REPO_ROOT / "evaluation/reports/full_report.json").read_text())
    ev = report["test_evaluation"]
    c = ev["counts"]
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == ev["n_total"]
    assert c["tp"] + c["fn"] == ev["n_positive"]
    assert c["fp"] + c["tn"] == ev["n_negative"]


# --------------------------------------------------------------------
# 11. Threshold is recorded and tied to the reported matrix.
# --------------------------------------------------------------------

def test_threshold_recorded_on_result():
    y_true = [1, 0, 1, 0]
    y_proba = [0.9, 0.2, 0.6, 0.4]
    m = evaluate(y_true, y_proba, threshold=0.5)
    assert m.threshold == 0.5


def test_different_threshold_produces_different_matrix():
    y_true = [1, 1, 0, 0, 1]
    y_proba = [0.9, 0.55, 0.6, 0.2, 0.3]
    low = evaluate(y_true, y_proba, threshold=0.3)
    high = evaluate(y_true, y_proba, threshold=0.7)
    assert (low.counts.tp, low.counts.fp, low.counts.fn, low.counts.tn) != (
        high.counts.tp, high.counts.fp, high.counts.fn, high.counts.tn
    )


# --------------------------------------------------------------------
# 12. Threshold analysis is diagnostic-only and does not change the
# production threshold anywhere.
# --------------------------------------------------------------------

def test_threshold_analysis_covers_requested_thresholds():
    y_true = np.random.RandomState(1).randint(0, 2, 500)
    y_proba = np.random.RandomState(2).rand(500)
    results = evaluate_thresholds(y_true, y_proba, thresholds=(0.2, 0.5, 0.8))
    assert [m.threshold for m in results] == [0.2, 0.5, 0.8]
    for m in results:
        assert m.counts.n == 500


def test_threshold_analysis_does_not_mutate_production_threshold():
    report_before = json.loads((REPO_ROOT / "evaluation/reports/full_report.json").read_text())
    y_true = np.random.RandomState(3).randint(0, 2, 200)
    y_proba = np.random.RandomState(4).rand(200)
    evaluate_thresholds(y_true, y_proba)
    report_after = json.loads((REPO_ROOT / "evaluation/reports/full_report.json").read_text())
    assert report_before["model"]["threshold"] == report_after["model"]["threshold"]


# --------------------------------------------------------------------
# 13-14. Reusable evaluation object + structured API output.
# --------------------------------------------------------------------

def test_classification_metrics_carries_all_required_fields():
    m = evaluate([1, 0, 1, 0], [0.8, 0.2, 0.6, 0.4], threshold=0.5,
                 dataset_type="synthetic_holdout", model_version="v1")
    for attr in ("threshold", "positive_class", "negative_class", "precision",
                 "recall", "f1", "fpr", "fnr", "n_total", "dataset_type", "model_version"):
        assert hasattr(m, attr)


def test_to_api_dict_structured_shape():
    m = evaluate([1, 0, 1, 0], [0.8, 0.2, 0.6, 0.4], threshold=0.5,
                 dataset_type="synthetic_holdout", model_version="v1")
    d = m.to_api_dict()
    assert d["dataset_type"] == "synthetic_holdout"
    assert d["positive_class"] == "return"
    assert d["negative_class"] == "no_return"
    assert d["threshold"] == 0.5
    assert d["model_version"] == "v1"
    assert set(d["confusion_matrix"].keys()) == {"tp", "fp", "fn", "tn"}
    assert set(d["metrics"].keys()) == {"precision", "recall", "f1", "fpr", "fnr"}


def test_to_dict_includes_confusion_matrix_alias_without_removing_counts():
    m = evaluate([1, 0, 1, 0], [0.8, 0.2, 0.6, 0.4], threshold=0.5)
    d = m.to_dict()
    assert "counts" in d  # backward-compatible key, unchanged
    assert "confusion_matrix" in d  # new structured alias
    assert d["confusion_matrix"] == d["counts"]


def test_full_report_json_exposes_dataset_context():
    report = json.loads((REPO_ROOT / "evaluation/reports/full_report.json").read_text())
    ev = report["test_evaluation"]
    assert ev["dataset_type"] == "synthetic_holdout"
    assert ev["positive_class"] == "return"
    assert ev["negative_class"] == "no_return"


# --------------------------------------------------------------------
# 15. Financial FP/FN exposure is a separate calculation from the
# FP/FN counts, but consistent with the same predictions.
# --------------------------------------------------------------------

def test_financial_exposure_count_consistent_with_confusion_matrix():
    from src.exposure.financial import compute_exposure
    import pandas as pd

    amount = pd.Series([100, 200, 300, 400, 500])
    y_true = pd.Series([1, 0, 1, 0, 1])
    y_pred = pd.Series([1, 1, 0, 0, 1])
    #      idx:            0    1    2    3    4
    #      true/pred:    TP   FP   FN   TN   TP
    c = confusion_counts(y_true.values, y_pred.values)
    exposure = compute_exposure(amount, y_true, y_pred)

    # n_high_risk_orders (predicted positive) must equal TP + FP.
    assert exposure.n_high_risk_orders == c.tp + c.fp == 3
    # False-positive/negative EXPOSURE (money, from order amounts) is a
    # distinct calculation from the false-positive/negative COUNT (from
    # the confusion matrix) -- they must not be conflated, and here they
    # even happen to differ in scale: FP count is 1 order but its
    # transaction value is 200; FN count is 1 order but its value is 300.
    assert c.fp == 1 and exposure.false_positive_exposure == 200
    assert c.fn == 1 and exposure.false_negative_exposure == 300


# --------------------------------------------------------------------
# 16 (UI). The dashboard must never hard-code confusion-matrix numbers;
# it must always read them from whatever report dict it's given.
# --------------------------------------------------------------------

def test_dashboard_source_has_no_hardcoded_confusion_matrix_numbers():
    src = (REPO_ROOT / "app/ui/generate_dashboard.py").read_text()
    # A renderer must consume report counts, not embed a particular
    # evaluation result. Check for the old golden-number literals.
    for literal in ("243", "361", "3596", "3,596"):
        assert literal not in src, f"generate_dashboard.py must not hard-code {literal}"


def test_dashboard_renders_different_matrices_for_different_reports():
    from app.ui.generate_dashboard import render

    def _report(tp, fp, fn, tn):
        n = tp + fp + fn + tn
        return {
            "status": "ok", "dataset_label": "x", "is_synthetic_demo": True,
            "data_readiness": None,
            "model": {"model_name": "logistic_regression", "threshold": 0.5,
                      "feature_count": 1, "split": {"train_n": 1, "val_n": 1, "test_n": n}},
            "test_evaluation": {
                "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
                "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
                "f1": 0.5, "fpr": 0.1, "fnr": 0.1,
                "n_total": n, "n_positive": tp + fn,
                "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            },
            "financial_exposure": {"n_high_risk_orders": tp + fp, "n_total_orders": n,
                                    "pct_orders_high_risk": (tp + fp) / n,
                                    "predicted_return_exposure": 0, "observed_historical_return_value": 0},
            "diagnosis": {}, "top_finding": None, "recommendation": None, "verification": None,
        }

    html_a = render(_report(10, 20, 30, 40))
    html_b = render(_report(99, 88, 77, 66))
    assert "10" in html_a and "20" in html_a and "30" in html_a and "40" in html_a
    assert "99" in html_b and "88" in html_b and "77" in html_b and "66" in html_b
    assert html_a != html_b
