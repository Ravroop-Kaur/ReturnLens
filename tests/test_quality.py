import pandas as pd
import pytest

from src.quality.dedup import analyze_duplicates, drop_exact_duplicates
from src.quality.order_level import aggregate_to_order_level, detect_granularity
from src.quality.feature_contract import evaluate_feature_contract, FeatureStatus
from src.quality.lifecycle import assign_label_state, usable_for_supervision, RETURNED, NO_RETURN, PENDING
from src.quality.readiness import run_data_readiness_pipeline


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------

def test_exact_duplicate_rows_dropped():
    df = pd.DataFrame({"order_id": ["A", "A", "B"], "amount": [10, 10, 20]})
    deduped, n_dropped = drop_exact_duplicates(df)
    assert n_dropped == 1
    assert len(deduped) == 2


def test_legitimate_line_items_not_flagged_as_conflicting():
    df = pd.DataFrame({
        "order_id": ["123", "123", "123"],
        "product_id": ["A", "B", "C"],
        "amount": [10, 20, 30],
    })
    report = analyze_duplicates(df)
    assert report.n_line_item_groups == 1
    assert report.n_conflicting_groups == 0


def test_conflicting_outcome_records_flagged():
    df = pd.DataFrame({
        "order_id": ["123", "123"],
        "product_id": ["A", "A"],
        "return_event": [True, False],
    })
    report = analyze_duplicates(df)
    assert report.n_conflicting_groups == 1
    assert "123" in report.conflicting_order_ids


def test_repeated_identical_event_is_not_a_conflict():
    df = pd.DataFrame({
        "order_id": ["123", "123"],
        "return_event": [True, True],
        "amount": [10, 10],
    })
    # identical rows -> dropped by drop_exact_duplicates, never surfaced
    # as a conflict.
    report = analyze_duplicates(df)
    assert report.n_conflicting_groups == 0


# ---------------------------------------------------------------------------
# order-level aggregation
# ---------------------------------------------------------------------------

def test_detect_granularity_line_item():
    df = pd.DataFrame({"order_id": ["1", "1", "2"]})
    assert detect_granularity(df) == "line_item_level"


def test_detect_granularity_order_level():
    df = pd.DataFrame({"order_id": ["1", "2", "3"]})
    assert detect_granularity(df) == "order_level"


def test_aggregate_sums_amount_and_any_outcome():
    df = pd.DataFrame({
        "order_id": ["1", "1", "2"],
        "amount": [10, 20, 30],
        "return_event": [False, True, False],
        "order_date": ["2025-01-01", "2025-01-01", "2025-01-02"],
    })
    order_df, report = aggregate_to_order_level(df)
    assert report.detected_granularity == "line_item_level"
    row = order_df[order_df["order_id"] == "1"].iloc[0]
    assert row["amount"] == 30
    assert bool(row["return_event"]) is True
    assert len(order_df) == 2


def test_aggregate_no_double_counting_order_level_eval():
    df = pd.DataFrame({
        "order_id": ["1", "1", "1"],
        "amount": [10, 10, 10],
        "order_date": ["2025-01-01"] * 3,
    })
    order_df, report = aggregate_to_order_level(df)
    assert report.n_orders == 1
    assert len(order_df) == 1  # not 3


def test_disagreeing_non_special_column_becomes_nan():
    df = pd.DataFrame({
        "order_id": ["1", "1"],
        "category": ["Books", "Electronics"],
        "amount": [10, 20],
    })
    order_df, _ = aggregate_to_order_level(df)
    assert pd.isna(order_df.iloc[0]["category"])


# ---------------------------------------------------------------------------
# feature contract
# ---------------------------------------------------------------------------

def test_all_required_present_and_usable():
    df = pd.DataFrame({
        "order_id": range(10), "order_date": ["2025-01-01"] * 10,
        "amount": range(10), "return_event": [True, False] * 5,
    })
    result = evaluate_feature_contract(df)
    assert result.has_all_required()
    assert result.has_usable_target()
    assert result.readiness_label() in ("FULLY_SUPPORTED", "PARTIALLY_SUPPORTED")


def test_missing_required_field_marked_required_missing():
    df = pd.DataFrame({"order_id": range(5), "amount": range(5)})
    result = evaluate_feature_contract(df)
    assert result.statuses["order_date"] == FeatureStatus.REQUIRED_MISSING
    assert not result.has_all_required()
    assert result.readiness_label() == "NOT_READY"


def test_sparse_column_marked_not_usable():
    df = pd.DataFrame({
        "order_id": range(10), "order_date": ["2025-01-01"] * 10, "amount": range(10),
        "return_event": [True] * 10,
        "region": [None] * 9 + ["North"],  # 90% missing
    })
    result = evaluate_feature_contract(df)
    assert result.statuses["region"] == FeatureStatus.NOT_USABLE


def test_absent_optional_field_marked_not_available():
    df = pd.DataFrame({"order_id": range(5), "order_date": ["2025-01-01"] * 5, "amount": range(5), "return_event": [True] * 5})
    result = evaluate_feature_contract(df)
    assert result.statuses["region"] == FeatureStatus.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# label lifecycle
# ---------------------------------------------------------------------------

def test_returned_order_is_final_regardless_of_recency():
    df = pd.DataFrame({"order_date": [pd.Timestamp.now()], "return_event": [True]})
    state = assign_label_state(df, as_of=pd.Timestamp.now())
    assert state.iloc[0] == RETURNED


def test_recent_order_with_no_return_is_pending_not_no_return():
    df = pd.DataFrame({"order_date": [pd.Timestamp.now() - pd.Timedelta(days=1)], "return_event": [False]})
    state = assign_label_state(df, as_of=pd.Timestamp.now(), return_window_days=30)
    assert state.iloc[0] == PENDING


def test_old_order_with_no_return_is_finalized_no_return():
    df = pd.DataFrame({"order_date": [pd.Timestamp.now() - pd.Timedelta(days=60)], "return_event": [False]})
    state = assign_label_state(df, as_of=pd.Timestamp.now(), return_window_days=30)
    assert state.iloc[0] == NO_RETURN


def test_unknown_order_date_is_pending_not_fabricated():
    df = pd.DataFrame({"order_date": [pd.NaT], "return_event": [False]})
    state = assign_label_state(df, as_of=pd.Timestamp.now())
    assert state.iloc[0] == PENDING


def test_usable_for_supervision_excludes_pending():
    df = pd.DataFrame({
        "order_date": [pd.Timestamp.now() - pd.Timedelta(days=60), pd.Timestamp.now() - pd.Timedelta(days=1)],
        "return_event": [False, False],
    })
    state = assign_label_state(df, as_of=pd.Timestamp.now(), return_window_days=30)
    trainable = usable_for_supervision(df, state)
    assert len(trainable) == 1


# ---------------------------------------------------------------------------
# full readiness pipeline
# ---------------------------------------------------------------------------

def _big_ready_dataset(n=250):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "order_id": [f"O{i}" for i in range(n)],
        "order_date": dates,
        "amount": [100.0 + i for i in range(n)],
        "category": ["Apparel", "Books"] * (n // 2),
        "return_event": [i % 5 == 0 for i in range(n)],
    })


def test_readiness_ready_on_sufficient_clean_data():
    df = _big_ready_dataset()
    order_df, report = run_data_readiness_pipeline(df, as_of=pd.Timestamp("2025-06-01"))
    assert report.model_status == "READY"
    assert report.n_orders == len(df)


def test_readiness_not_ready_too_few_rows():
    df = _big_ready_dataset(n=10)
    order_df, report = run_data_readiness_pipeline(df, as_of=pd.Timestamp("2025-06-01"))
    assert report.model_status == "NOT_READY"
    assert any("Only" in r and "orders available" in r for r in report.reasons_not_ready)


def test_readiness_not_ready_missing_required_field():
    df = _big_ready_dataset().drop(columns=["amount"])
    order_df, report = run_data_readiness_pipeline(df, as_of=pd.Timestamp("2025-06-01"))
    assert report.model_status == "NOT_READY"


def test_readiness_not_ready_insufficient_labels():
    df = _big_ready_dataset()
    df["return_event"] = None  # nothing finalized
    order_df, report = run_data_readiness_pipeline(df, as_of=pd.Timestamp("2024-01-10"))
    assert report.model_status == "NOT_READY"


def test_line_item_all_missing_return_label_stays_unknown():
    df = pd.DataFrame({
        "order_id": ["1", "1"], "amount": [10, 20],
        "return_event": [pd.NA, pd.NA],
        "order_date": ["2025-01-01", "2025-01-01"],
    })
    order_df, _ = aggregate_to_order_level(df)
    assert pd.isna(order_df.iloc[0]["return_event"])


def test_historical_return_feature_does_not_use_unmatured_future_label():
    from src.features.engineering import build_features
    df = pd.DataFrame({
        "order_id": ["A", "B"],
        "order_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-05")],
        "return_event": [True, False],
        "return_date": [pd.Timestamp("2025-01-20"), pd.NaT],
        "customer_id": ["C1", "C1"],
        "amount": [100, 100],
    })
    X, _, _ = build_features(df)
    # A's return was not known on B's prediction date (Jan 5), so
    # B must receive only the smoothed global prior, not a 100% history.
    assert X.loc[1, "hist_return_rate_customer_id"] < 1.0
