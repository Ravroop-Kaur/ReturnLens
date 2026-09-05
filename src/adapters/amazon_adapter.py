"""
Amazon-style export adapter.

Amazon is NOT the product. This adapter exists only to demonstrate
that a specific, real-world export format (the widely used "Amazon
Sale Report" style CSV, with columns like Order ID / Date / Status /
Fulfilment / Category / Amount / ship-state / ship-service-level) can
be mapped into the same canonical schema that the generic CSV path
produces. Everything downstream of this adapter is merchant-agnostic.

Amazon's raw "Status" column mixes fulfilment state and outcome
state (e.g. "Shipped - Returned to seller", "Cancelled",
"Shipped - Delivered to Buyer"). We interpret it ONLY to construct
the return_event outcome label -- never as a predictive feature,
since it is partially a mechanical function of the outcome itself
(see docs/leakage_notes.md).
"""

from __future__ import annotations
import pandas as pd
from typing import Optional

from src.canonical.mapping import validate_and_summarize, coerce_types
from src.adapters.generic_csv import IngestionResult

AMAZON_COLUMN_MAP = {
    "Order ID": "order_id",
    "Date": "order_date",
    "Amount": "amount",
    "Category": "category",
    "Fulfilment": "fulfilment_method",
    "ship-service-level": "shipping_service",
    "ship-state": "region",
    "SKU": "product_id",
    "Status": "_raw_status",  # interpreted below, not passed through raw
}

# Status values that indicate the order was returned. This mapping is
# specific to this export format's vocabulary and lives only here.
RETURN_STATUS_VALUES = {
    "returned to seller",
    "shipped - returned to seller",
    "shipped - rejected by buyer",
    "shipped - returning to seller",
}
NON_RETURN_TERMINAL_STATUS_VALUES = {
    "shipped - delivered to buyer",
    "shipped",
    "delivered",
}


def ingest_amazon_csv(csv_path: str) -> IngestionResult:
    raw = pd.read_csv(csv_path, low_memory=False)

    present_cols = {src: dst for src, dst in AMAZON_COLUMN_MAP.items() if src in raw.columns}
    canonical = pd.DataFrame(index=raw.index)
    for src, dst in present_cols.items():
        canonical[dst] = raw[src]

    if "_raw_status" in canonical.columns:
        status_lower = canonical["_raw_status"].astype(str).str.strip().str.lower()
        return_event = status_lower.isin(RETURN_STATUS_VALUES)
        known_terminal = status_lower.isin(RETURN_STATUS_VALUES | NON_RETURN_TERMINAL_STATUS_VALUES)
        # Only assert a label where the status is a known terminal state.
        # Unknown/ambiguous status values become NaN (unlabeled), not a
        # fabricated negative.
        canonical["return_event"] = pd.NA
        canonical.loc[known_terminal, "return_event"] = return_event[known_terminal]
        canonical = canonical.drop(columns=["_raw_status"])

    canonical = coerce_types(canonical)
    caps = validate_and_summarize(canonical)

    return IngestionResult(
        canonical_df=canonical,
        capabilities=caps,
        mapping_used={v: k for k, v in present_cols.items()},
        source_columns=list(raw.columns),
        n_rows_raw=len(raw),
    )
