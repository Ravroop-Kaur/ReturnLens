import pandas as pd
from src.exposure.financial import compute_exposure


def test_exposure_basic_math():
    amount = pd.Series([100, 200, 300, 400])
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 1, 0, 0])
    # TP: idx0 (100), FP: idx1 (200), FN: idx2 (300), TN: idx3 (400)
    report = compute_exposure(amount, y_true, y_pred)
    assert report.predicted_return_exposure == 300  # idx0+idx1 predicted positive
    assert report.observed_return_value == 400  # idx0+idx2 actually returned
    assert report.false_positive_exposure == 200
    assert report.false_negative_exposure == 300
    assert report.n_high_risk_orders == 2
    assert report.n_total_orders == 4


def test_exposure_terminology_never_claims_savings():
    amount = pd.Series([100, 200])
    y_true = pd.Series([1, 0])
    y_pred = pd.Series([1, 0])
    report = compute_exposure(amount, y_true, y_pred)
    d = report.to_dict()
    banned_terms = ["revenue lost", "savings", "profit", "roi", "recovered revenue"]
    serialized = str(d).lower()
    for term in banned_terms:
        assert term not in serialized


def test_exposure_with_no_high_risk_orders():
    amount = pd.Series([100, 200])
    y_true = pd.Series([0, 1])
    y_pred = pd.Series([0, 0])
    report = compute_exposure(amount, y_true, y_pred)
    assert report.predicted_return_exposure == 0
    assert report.pct_orders_high_risk == 0.0
