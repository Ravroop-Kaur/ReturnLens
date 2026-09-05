import pandas as pd
from src.model.train import train_return_risk_model, LIGHTGBM_AVAILABLE
from src.model.split import temporal_split


def _load_sample():
    return pd.read_csv(
        "data/sample/generic_merchant_orders.csv",
        parse_dates=["order_date"],
    )


def test_temporal_split_is_chronological_and_non_overlapping():
    df = _load_sample()
    train, val, test, bounds = temporal_split(df)
    assert train["order_date"].max() <= val["order_date"].min()
    assert val["order_date"].max() <= test["order_date"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_model_trains_and_produces_valid_probabilities():
    df = _load_sample()
    bundle = train_return_risk_model(df, compare_lightgbm=False)
    p = bundle.predict_proba(bundle._test_X)
    assert (p >= 0).all() and (p <= 1).all()
    assert 0.0 <= bundle.threshold <= 1.0


def test_threshold_is_frozen_from_validation_only():
    df = _load_sample()
    bundle = train_return_risk_model(df, compare_lightgbm=False)
    # the recorded threshold selection stats must reference the
    # validation set size, not the test set size
    sel = bundle._val_threshold_selection
    n_from_selection = sel["tp"] + sel["fp"] + sel["fn"] + sel["tn"]
    assert n_from_selection == len(bundle._val_y)
    assert n_from_selection != len(bundle._test_y) or len(bundle._val_y) == len(bundle._test_y)


def test_reproducibility_same_seed_same_result():
    df = _load_sample()
    b1 = train_return_risk_model(df, compare_lightgbm=False)
    b2 = train_return_risk_model(df, compare_lightgbm=False)
    p1 = b1.predict_proba(b1._test_X)
    p2 = b2.predict_proba(b2._test_X)
    assert (abs(p1 - p2) < 1e-9).all()
    assert b1.threshold == b2.threshold


def test_model_selection_uses_validation_not_test():
    df = _load_sample()
    bundle = train_return_risk_model(df, compare_lightgbm=True)
    # val_metrics_at_candidates must contain entries computed on
    # validation-sized predictions, not test
    assert "logistic_regression" in bundle.val_metrics_at_candidates
    # LightGBM is an optional dependency (see src.model.train) -- some
    # environments (e.g. sandboxes with no network access to install
    # it) won't have it. When it's genuinely available, it must have
    # been compared; when it's not, training must still fall back
    # gracefully to logistic-regression-only rather than crashing.
    if LIGHTGBM_AVAILABLE:
        assert "lightgbm" in bundle.val_metrics_at_candidates
    else:
        assert "lightgbm" not in bundle.val_metrics_at_candidates
        assert bundle.model_name == "logistic_regression"


def test_demo_operating_point_reports_metrics_without_metric_gaming():
    """The demo must produce valid metrics without requiring a hard-coded
    precision/recall target on the frozen test set."""
    df = _load_sample()
    bundle = train_return_risk_model(df, compare_lightgbm=True)
    p = bundle.predict_proba(bundle._test_X)
    pred = (p >= bundle.threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(bundle._test_y, pred, zero_division=0)
    recall = recall_score(bundle._test_y, pred, zero_division=0)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert bundle._val_threshold_selection["selection_rule"].startswith("validation_only:")
