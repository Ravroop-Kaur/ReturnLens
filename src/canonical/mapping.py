"""
Maps an arbitrary merchant CSV (with unknown column names) into the
canonical schema, and validates what capabilities the result supports.

This module deliberately does NOT know about "Amazon" or any other
specific merchant. Source-specific quirks belong in src/adapters/.
This module only knows about the canonical field names and a set of
common synonyms used to auto-suggest a mapping to a human reviewer /
caller.
"""

from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional

from src.canonical.schema import (
    REQUIRED_FIELDS,
    OPTIONAL_FIELDS,
    OUTCOME_FIELDS,
    CanonicalCapabilities,
)

# Synonym table used only to PROPOSE a mapping. A human/caller still
# supplies or confirms the final mapping -- we never silently guess
# on required fields without validation.
SYNONYMS: Dict[str, List[str]] = {
    "order_id": ["order_id", "order id", "transaction_id", "order-id", "id"],
    "order_date": ["order_date", "date", "order date", "created_at", "order_time"],
    "amount": ["amount", "order_amount", "total", "price", "value", "amount_inr"],
    "product_id": ["product_id", "sku", "asin", "product"],
    "customer_id": ["customer_id", "customer id", "customer", "buyer_id", "user_id"],
    "category": ["category", "product_category"],
    "region": ["region", "ship_state", "state", "city", "customer_region"],
    "fulfilment_method": ["fulfilment_method", "fulfilment", "fulfillment", "fulfilled_by"],
    "shipping_service": ["shipping_service", "courier", "carrier", "ship_service_level"],
    "payment_status": ["payment_status", "payment", "b2b_or_payment"],
    "delivery_status": ["delivery_status", "status", "order_status"],
    "review_text": ["review_text", "review", "comments"],
    "return_event": ["return_event", "is_return", "returned", "return_flag"],
    "return_date": ["return_date", "date_returned"],
    "refund_event": ["refund_event", "is_refund", "refunded"],
    "chargeback_event": ["chargeback_event", "is_chargeback", "chargeback"],
}


def propose_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    """Suggest a canonical_field -> source_column mapping using synonyms.

    Returns None for a field with no confident match. The caller
    (adapter or generic ingestion UI) is responsible for confirming or
    overriding this before it is used.
    """
    lower_cols = {c.lower().strip(): c for c in columns}
    mapping: Dict[str, Optional[str]] = {}
    for canonical_field, synonyms in SYNONYMS.items():
        match = None
        for syn in synonyms:
            if syn in lower_cols:
                match = lower_cols[syn]
                break
        mapping[canonical_field] = match
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Rename/select source columns into a canonical DataFrame.

    Only columns with a non-null mapping are copied over. Missing
    optional/outcome fields are simply absent from the result -- they
    are NOT fabricated with placeholder values.
    """
    out = pd.DataFrame(index=df.index)
    for canonical_field, source_col in mapping.items():
        if source_col is not None and source_col in df.columns:
            out[canonical_field] = df[source_col]
    return out


def validate_and_summarize(canonical_df: pd.DataFrame) -> CanonicalCapabilities:
    """Check which required fields are missing and which optional
    capabilities are available. Never fabricates data -- only reports.
    """
    caps = CanonicalCapabilities()

    missing_required = [f for f in REQUIRED_FIELDS if f not in canonical_df.columns]
    caps.missing_required = missing_required

    caps.has_target = "return_event" in canonical_df.columns and canonical_df["return_event"].notna().any()
    caps.has_category = "category" in canonical_df.columns
    caps.has_region = "region" in canonical_df.columns
    caps.has_fulfilment = "fulfilment_method" in canonical_df.columns
    caps.has_shipping_service = "shipping_service" in canonical_df.columns
    caps.has_review_text = "review_text" in canonical_df.columns

    return caps


def coerce_types(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort, non-fabricating type coercion.

    - order_date -> datetime (invalid values become NaT, not silently
      dropped, so callers can decide how to handle them)
    - amount -> numeric
    - boolean-ish outcome columns -> nullable boolean
    """
    df = canonical_df.copy()

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    if "return_date" in df.columns:
        df["return_date"] = pd.to_datetime(df["return_date"], errors="coerce")

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    for bool_field in ["return_event", "refund_event", "chargeback_event"]:
        if bool_field in df.columns:
            df[bool_field] = _coerce_bool(df[bool_field])

    return df


def _coerce_bool(series: pd.Series) -> pd.Series:
    truthy = {"true", "1", "yes", "y", "returned", "return", "refunded", "refund", "1.0"}
    falsy = {"false", "0", "no", "n", "not returned", "0.0", ""}

    def conv(v):
        if pd.isna(v):
            return pd.NA
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        if s in truthy:
            return True
        if s in falsy:
            return False
        return pd.NA

    return series.apply(conv)
