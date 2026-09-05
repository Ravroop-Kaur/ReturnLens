"""
Generic merchant data connector abstraction (PART C).

The ML engine and every downstream layer only ever consume
`fetch_historical_data()` / `fetch_incremental_data()`'s canonical
DataFrame output -- never a source-specific client. This is what lets
future connectors (a real REST API, a read-only database replica, a
warehouse export, an ERP/OMS integration) be added without touching
src.quality, src.model, src.diagnosis, src.exposure, etc.

For the current MVP, three implementations exist:
  - MockMerchantConnector   (src.connectors.mock)      -- synthetic demo data
  - CSVMerchantConnector    (src.connectors.csv_connector) -- CSV import/fallback
  - RazorpayConnector       (src.connectors.razorpay_connector) -- Razorpay Test Mode

IMPORTANT: connectors must never require unrestricted production
database access or a merchant's normal database admin password. Every
implementation here uses either static demo data, a user-supplied
read-only file, or a scoped API key/webhook secret from environment
variables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ConnectionHealth:
    healthy: bool
    detail: str


class MerchantDataConnector(ABC):
    """Every connector implementation must declare a connector_type
    (used purely for labelling in reports/UI, e.g. "mock", "csv",
    "razorpay" -- never used to change ML behavior)."""

    connector_type: str = "unknown"

    @abstractmethod
    def test_connection(self) -> ConnectionHealth:
        """Cheap, side-effect-free check that credentials/config are
        valid and the source is reachable. Must never raise for an
        ordinary failure -- return ConnectionHealth(healthy=False, ...)."""
        raise NotImplementedError

    @abstractmethod
    def fetch_historical_data(self) -> pd.DataFrame:
        """Full historical canonical order data available from this
        source, already mapped to the canonical schema (see
        src.canonical.schema). Must NEVER fabricate columns that are
        not actually available from the source."""
        raise NotImplementedError

    @abstractmethod
    def fetch_incremental_data(self, since: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Only orders/events new or changed since `since` (or the
        connector's own notion of "since last sync" if None). Used for
        incremental risk updates without re-pulling full history."""
        raise NotImplementedError

    def health_check(self) -> ConnectionHealth:
        """Default implementation defers to test_connection(); a
        connector with a cheaper liveness probe may override this."""
        return self.test_connection()
