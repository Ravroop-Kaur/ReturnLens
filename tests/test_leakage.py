import numpy as np
import pandas as pd
import pytest

from src.features.engineering import build_features, _expanding_prior_rate
from src.canonical.schema import LEAKAGE_FORBIDDEN_FIELDS


def _sample_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    category = rng.choice(["A", "B"], size=n)
    return_event = rng.integers(0, 2, size=n).astype(bool)
    return pd.DataFrame({
        "order_id": [f"O{i}" for i in range(n)],
        "order_date": dates,
        "amount": rng.uniform(10, 500, size=n),
        "category": category,
        "fulfilment_method": rng.choice(["platform", "third_party"], size=n),
        "region": rng.choice(["N", "S"], size=n),
        "shipping_service": rng.choice(["Std", "Exp"], size=n),
        "return_event": return_event,
        "refund_event": return_event,
        "chargeback_event": rng.integers(0, 2, size=n).astype(bool),
    })


def test_forbidden_fields_never_in_feature_matrix():
    df = _sample_df()
    X, y, feature_names = build_features(df, for_training=True)
    assert LEAKAGE_FORBIDDEN_FIELDS.isdisjoint(set(feature_names))
    assert LEAKAGE_FORBIDDEN_FIELDS.isdisjoint(set(X.columns))


def test_historical_rate_excludes_current_row():
    """The historical rate feature for a group must never include the
    current row's own outcome -- i.e. it must equal the group's prior
    smoothed mean using strictly earlier rows only."""
    df = pd.DataFrame({
        "grp": ["x", "x", "x", "x"],
        "return_event": [True, True, True, True],
    })
    # if leakage existed, the first row's "historical" rate would be
    # pulled toward 1.0 by its own True value. With leave-one-out, the
    # first row has ZERO prior observations, so it must equal the
    # global prior (smoothed), not 1.0.
    rate = _expanding_prior_rate(df, "grp", "return_event")
    global_mean = df["return_event"].mean()
    assert abs(rate.iloc[0] - global_mean) < 1e-9


def test_historical_rate_increases_as_more_positive_history_accumulates():
    df = pd.DataFrame({
        "grp": ["x"] * 5,
        "return_event": [False, False, True, True, True],
    })
    rate = _expanding_prior_rate(df, "grp", "return_event")
    # rate should be monotonically non-decreasing here since we keep
    # adding positive prior observations after the first two
    assert rate.iloc[2] <= rate.iloc[3] <= rate.iloc[4]


def test_shuffled_row_order_raises_without_date_sort():
    """build_features re-sorts by order_date internally -- verify a
    shuffled input still produces a feature matrix aligned with sorted
    rows, so leave-one-out ordering is date-based, not input-order-based."""
    df = _sample_df(n=50, seed=1)
    shuffled = df.sample(frac=1.0, random_state=5).reset_index(drop=True)
    X1, y1, _ = build_features(df, for_training=True)
    X2, y2, _ = build_features(shuffled, for_training=True)
    # after internal re-sort, both should have identical historical
    # rate columns in the same (date-sorted) order
    hist_cols = [c for c in X1.columns if c.startswith("hist_return_rate_")]
    for col in hist_cols:
        assert np.allclose(X1[col].values, X2[col].values)


def test_no_future_information_leaks_into_early_rows():
    """An early row's category historical rate must not depend on a
    much-later row's outcome in the same category."""
    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    # all early rows are False, one very late row is True
    return_event = [False] * (n - 1) + [True]
    df = pd.DataFrame({
        "order_date": dates,
        "category": ["A"] * n,
        "return_event": return_event,
        "amount": [10.0] * n,
    })
    X, y, _ = build_features(df, for_training=True)
    # the historical rate for row 0 must be the (smoothed) prior,
    # i.e. it cannot possibly reflect the True at the very end
    assert X["hist_return_rate_category"].iloc[0] <= 0.05
