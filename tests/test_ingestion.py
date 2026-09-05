import pandas as pd
import tempfile
import os
from src.adapters.generic_csv import ingest_generic_csv
from src.adapters.amazon_adapter import ingest_amazon_csv


def _write_csv(df: pd.DataFrame) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    df.to_csv(path, index=False)
    return path


def test_generic_csv_full_capability():
    df = pd.DataFrame({
        "order_id": ["A1", "A2", "A3"],
        "order_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "amount": [100, 200, 300],
        "category": ["Apparel", "Books", "Apparel"],
        "return_event": ["yes", "no", "yes"],
    })
    path = _write_csv(df)
    result = ingest_generic_csv(path)
    assert result.can_train_detector()
    assert result.capabilities.has_target
    assert result.capabilities.has_category
    assert list(result.canonical_df["return_event"]) == [True, False, True]


def test_generic_csv_missing_target_reports_unavailable():
    df = pd.DataFrame({
        "order_id": ["A1", "A2"],
        "order_date": ["2025-01-01", "2025-01-02"],
        "amount": [100, 200],
    })
    path = _write_csv(df)
    result = ingest_generic_csv(path)
    assert not result.capabilities.has_target
    assert not result.can_train_detector()
    assert "unavailable" in result.message().lower()


def test_generic_csv_missing_required_field_detected():
    df = pd.DataFrame({
        "order_id": ["A1", "A2"],
        "amount": [100, 200],
        # no date column at all, and no synonym present
    })
    path = _write_csv(df)
    result = ingest_generic_csv(path)
    assert "order_date" in result.capabilities.missing_required
    assert not result.can_train_detector()


def test_ambiguous_column_names_mapped_via_synonyms():
    df = pd.DataFrame({
        "Order ID": ["A1", "A2"],
        "Date": ["2025-01-01", "2025-01-02"],
        "Total": [10, 20],
    })
    path = _write_csv(df)
    result = ingest_generic_csv(path)
    assert "order_id" in result.canonical_df.columns
    assert "order_date" in result.canonical_df.columns
    assert "amount" in result.canonical_df.columns


def test_amazon_adapter_maps_status_to_return_event():
    df = pd.DataFrame({
        "Order ID": ["X1", "X2", "X3"],
        "Date": ["01-01-25", "01-02-25", "01-03-25"],
        "Status": [
            "Shipped - Returned to seller",
            "Shipped - Delivered to Buyer",
            "Some Unknown Status",
        ],
        "Amount": [500, 700, 900],
        "Fulfilment": ["Amazon", "Merchant", "Amazon"],
        "Category": ["Apparel", "Electronics", "Home"],
    })
    path = _write_csv(df)
    result = ingest_amazon_csv(path)
    returns = result.canonical_df["return_event"].tolist()
    assert returns[0] == True
    assert returns[1] == False
    # unknown status must NOT be fabricated into a definite label
    assert pd.isna(returns[2])


def test_amazon_is_only_a_source_adapter_not_required():
    """The generic path must work without any Amazon-specific concept."""
    df = pd.DataFrame({
        "order_id": ["G1"],
        "order_date": ["2025-05-01"],
        "amount": [42.0],
        "return_event": [False],
    })
    path = _write_csv(df)
    result = ingest_generic_csv(path)
    assert result.can_train_detector()
    assert "Status" not in result.canonical_df.columns


def test_generic_mapping_preserves_customer_id_for_behavioral_features(tmp_path):
    import pandas as pd
    from src.adapters.generic_csv import ingest_generic_csv
    path = tmp_path / "orders.csv"
    pd.DataFrame({
        "order_id": ["o1"], "order_date": ["2025-01-01"], "amount": [100],
        "customer_id": ["c1"], "return_event": [False]
    }).to_csv(path, index=False)
    result = ingest_generic_csv(str(path))
    assert "customer_id" in result.canonical_df.columns
