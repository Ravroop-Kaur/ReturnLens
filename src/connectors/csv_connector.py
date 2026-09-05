"""
CSV connector -- fallback / manual import path (PART G1: "CSV =
fallback/import, NOT the product").

Wraps src.adapters.generic_csv (and, when a mapping override signals
Amazon-style export, src.adapters.amazon_adapter) behind the common
MerchantDataConnector interface, so the readiness/model pipeline can
treat a manually uploaded CSV exactly like any other data source.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.connectors.base import MerchantDataConnector, ConnectionHealth
from src.adapters.generic_csv import ingest_generic_csv


class CSVMerchantConnector(MerchantDataConnector):
    connector_type = "csv"

    def __init__(self, csv_path: str, mapping_override: Optional[Dict[str, Optional[str]]] = None):
        self.csv_path = csv_path
        self.mapping_override = mapping_override

    def test_connection(self) -> ConnectionHealth:
        try:
            pd.read_csv(self.csv_path, nrows=1)
            return ConnectionHealth(healthy=True, detail=f"CSV file readable at {self.csv_path}.")
        except Exception as exc:
            return ConnectionHealth(healthy=False, detail=f"Could not read CSV: {exc}")

    def fetch_historical_data(self) -> pd.DataFrame:
        result = ingest_generic_csv(self.csv_path, mapping_override=self.mapping_override)
        return result.canonical_df

    def fetch_incremental_data(self, since: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        # A flat CSV import has no native notion of "since last sync" --
        # every import is treated as a fresh full snapshot. Callers
        # that need true incremental behavior should use an API-based
        # connector (Razorpay, or a future REST/warehouse connector).
        df = self.fetch_historical_data()
        if since is not None and "order_date" in df.columns:
            df = df[df["order_date"] > since]
        return df.reset_index(drop=True)
