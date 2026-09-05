"""
Mock/reference connector.

Serves the bundled synthetic demo dataset (data/sample/generic_merchant_orders.csv)
through the same MerchantDataConnector interface every real connector
uses. This is what lets the whole product run end-to-end in demo mode
without any production credentials (PART J). Every result derived
from this connector must be labelled DEMO / SYNTHETIC by callers
(connector_type == "mock" is exactly the flag callers check).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.connectors.base import MerchantDataConnector, ConnectionHealth
from src.canonical.mapping import propose_mapping, apply_mapping, coerce_types

DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample" / "generic_merchant_orders.csv"


class MockMerchantConnector(MerchantDataConnector):
    connector_type = "mock"

    def __init__(self, sample_path: Optional[str] = None):
        self.sample_path = Path(sample_path) if sample_path else DEFAULT_SAMPLE_PATH

    def test_connection(self) -> ConnectionHealth:
        if self.sample_path.exists():
            return ConnectionHealth(healthy=True, detail=f"Demo dataset found at {self.sample_path}.")
        return ConnectionHealth(healthy=False, detail=f"Demo dataset not found at {self.sample_path}.")

    def _load(self) -> pd.DataFrame:
        raw = pd.read_csv(self.sample_path)
        mapping = propose_mapping(list(raw.columns))
        canonical = apply_mapping(raw, mapping)
        return coerce_types(canonical)

    def fetch_historical_data(self) -> pd.DataFrame:
        return self._load()

    def fetch_incremental_data(self, since: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        df = self._load()
        if since is not None and "order_date" in df.columns:
            df = df[df["order_date"] > since]
        return df.reset_index(drop=True)
