import pandas as pd

from src.quality.drift import check_drift, NORMAL, MILD_DRIFT, SIGNIFICANT_DRIFT


def _make(n, return_rate, amount_mean, category_split):
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "return_event": rng.random(n) < return_rate,
        "amount": rng.normal(amount_mean, 10, n),
        "category": rng.choice(list(category_split.keys()), size=n, p=list(category_split.values())),
    })


def test_no_drift_when_distributions_match():
    ref = _make(2000, 0.1, 100, {"A": 0.5, "B": 0.5})
    cur = _make(2000, 0.1, 100, {"A": 0.5, "B": 0.5})
    report = check_drift(ref, cur)
    assert report.overall_status == NORMAL


def test_significant_drift_on_return_prevalence_spike():
    ref = _make(2000, 0.05, 100, {"A": 0.5, "B": 0.5})
    cur = _make(2000, 0.35, 100, {"A": 0.5, "B": 0.5})
    report = check_drift(ref, cur)
    prevalence_signal = next(s for s in report.signals if s.signal == "return_prevalence")
    assert prevalence_signal.status == SIGNIFICANT_DRIFT
    assert report.overall_status == SIGNIFICANT_DRIFT


def test_categorical_drift_detects_new_category():
    ref = pd.DataFrame({"category": ["A"] * 100 + ["B"] * 100})
    cur = pd.DataFrame({"category": ["A"] * 50 + ["C"] * 150})
    report = check_drift(ref, cur)
    cat_signal = next(s for s in report.signals if s.signal == "category_distribution")
    assert cat_signal.status in (MILD_DRIFT, SIGNIFICANT_DRIFT)
    assert "C" in cat_signal.detail["new_categories_not_in_reference"]


def test_missing_data_on_one_side_is_normal_not_a_false_alarm():
    ref = pd.DataFrame({"return_event": []})
    cur = pd.DataFrame({"return_event": [True, False]})
    report = check_drift(ref, cur)
    sig = next(s for s in report.signals if s.signal == "return_prevalence")
    assert sig.status == NORMAL


def test_missingness_drift_detected():
    ref = pd.DataFrame({"category": ["A"] * 100})
    cur = pd.DataFrame({"category": ["A"] * 70 + [None] * 30})
    report = check_drift(ref, cur)
    miss_signal = next(s for s in report.signals if s.signal == "category_missingness")
    assert miss_signal.status in (MILD_DRIFT, SIGNIFICANT_DRIFT)
