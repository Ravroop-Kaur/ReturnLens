"""
Razorpay connector.

Wires src.integrations.razorpay (client + mapper) behind the common
MerchantDataConnector interface. Works in Razorpay Test Mode or fully
offline demo mode (RazorpayClient.is_demo) without any change to the
pipeline that consumes it.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.connectors.base import MerchantDataConnector, ConnectionHealth
from src.integrations.razorpay.config import RazorpayConfig
from src.integrations.razorpay.client import RazorpayClient, RazorpayAPIError
from src.integrations.razorpay.mapper import map_payments_to_canonical, refunded_ids_from_refunds_page


class RazorpayConnector(MerchantDataConnector):
    connector_type = "razorpay"

    def __init__(self, config: Optional[RazorpayConfig] = None, force_mock: bool = False):
        self.client = RazorpayClient(config=config, force_mock=force_mock)

    def test_connection(self) -> ConnectionHealth:
        if self.client.is_demo:
            return ConnectionHealth(healthy=True, detail="Razorpay demo/mock mode (no credentials configured).")
        try:
            self.client.fetch_payments(count=1)
            mode = "Test Mode" if self.client.config.is_test_mode else "Live Mode"
            return ConnectionHealth(healthy=True, detail=f"Connected to Razorpay ({mode}).")
        except RazorpayAPIError as exc:
            return ConnectionHealth(healthy=False, detail=str(exc))

    def _fetch_all_payments(self) -> dict:
        all_items = []
        skip = 0
        page = self.client.fetch_payments(count=100, skip=skip)
        all_items.extend(page.get("items", []))
        while len(page.get("items", [])) == 100:
            skip += 100
            page = self.client.fetch_payments(count=100, skip=skip)
            all_items.extend(page.get("items", []))
        return {"entity": "collection", "count": len(all_items), "items": all_items}

    def fetch_historical_data(self) -> pd.DataFrame:
        payments_page = self._fetch_all_payments()
        refunds = []
        skip = 0
        refunds_page = self.client.fetch_refunds(count=100, skip=skip)
        refunds.extend(refunds_page.get("items", []))
        while len(refunds_page.get("items", [])) == 100:
            skip += 100
            refunds_page = self.client.fetch_refunds(count=100, skip=skip)
            refunds.extend(refunds_page.get("items", []))
        refunded_ids = refunded_ids_from_refunds_page({"items": refunds})
        return map_payments_to_canonical(payments_page, refunded_payment_ids=refunded_ids)

    def fetch_incremental_data(self, since: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        df = self.fetch_historical_data()
        if since is not None and "order_date" in df.columns:
            df = df[df["order_date"] > since]
        return df.reset_index(drop=True)
