import os
import tempfile

import pandas as pd
import pytest

from src.connectors.base import MerchantDataConnector
from src.connectors.mock import MockMerchantConnector
from src.connectors.csv_connector import CSVMerchantConnector


def test_connector_is_an_abstract_interface():
    with pytest.raises(TypeError):
        MerchantDataConnector()  # abstract -- cannot be instantiated directly


def test_mock_connector_returns_canonical_dataframe():
    connector = MockMerchantConnector()
    health = connector.test_connection()
    assert health.healthy
    df = connector.fetch_historical_data()
    assert "order_id" in df.columns
    assert "order_date" in df.columns
    assert len(df) > 0
    assert connector.connector_type == "mock"


def test_mock_connector_incremental_filters_by_date():
    connector = MockMerchantConnector()
    full = connector.fetch_historical_data()
    cutoff = full["order_date"].median()
    incremental = connector.fetch_incremental_data(since=cutoff)
    assert (incremental["order_date"] > cutoff).all()
    assert len(incremental) < len(full)


def _write_csv(df: pd.DataFrame) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    df.to_csv(path, index=False)
    return path


def test_csv_connector_maps_to_canonical():
    df = pd.DataFrame({
        "order_id": ["A1", "A2"],
        "order_date": ["2025-01-01", "2025-01-02"],
        "amount": [100, 200],
        "return_event": ["yes", "no"],
    })
    path = _write_csv(df)
    connector = CSVMerchantConnector(csv_path=path)
    assert connector.test_connection().healthy
    canonical = connector.fetch_historical_data()
    assert list(canonical["return_event"]) == [True, False]
    assert connector.connector_type == "csv"


def test_csv_connector_reports_unhealthy_for_missing_file():
    connector = CSVMerchantConnector(csv_path="/no/such/file.csv")
    health = connector.test_connection()
    assert not health.healthy
