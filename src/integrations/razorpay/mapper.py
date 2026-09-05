"""
Maps Razorpay-shaped payloads into the canonical merchant schema.

The ML/business logic (src.model, src.quality, src.diagnosis, etc.)
must never depend on a Razorpay-specific field name -- everything
downstream of this module only ever sees canonical columns
(order_id, order_date, amount, category, ..., return_event, ...).

Razorpay itself has no native "return" concept (it is a payments
API, not an order-management/returns system). A refund is therefore
kept as a separate `refund_event` signal, never converted into
`return_event`. A merchant with a real returns/OMS system should
supply the actual `return_event` field; Razorpay data alone mainly
supplies payment/order value and refund signals.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.canonical.mapping import coerce_types


def map_payments_to_canonical(
    payments_page: dict,
    refunded_payment_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Map Razorpay payments into canonical orders.

    Refunds are financial events, not proof that an order was returned.
    Therefore `refund_event` may be populated, while `return_event`
    remains unknown unless a separate returns/OMS source supplies it.
    Missing return labels are intentionally left as ``pd.NA``.
    """
    refunded_payment_ids = refunded_payment_ids or set()
    items = payments_page.get("items", [])

    rows = []
    for item in items:
        payment_id = item.get("id")
        amount_paise = item.get("amount")
        amount = (amount_paise / 100.0) if amount_paise is not None else None
        created_at = item.get("created_at")
        order_date = (
            pd.to_datetime(created_at, unit="s", errors="coerce")
            if created_at is not None else None
        )
        notes = item.get("notes") or {}

        rows.append({
            "order_id": payment_id,
            "order_date": order_date,
            "amount": amount,
            "category": notes.get("category"),
            "region": notes.get("region"),
            "return_event": pd.NA,
            "refund_event": payment_id in refunded_payment_ids,
        })

    canonical = pd.DataFrame(rows)
    if canonical.empty:
        canonical = pd.DataFrame(columns=[
            "order_id", "order_date", "amount", "category", "region",
            "return_event", "refund_event",
        ])
    return coerce_types(canonical)


def refunded_ids_from_refunds_page(refunds_page: dict) -> set:
    ids = set()
    for item in refunds_page.get("items", []):
        payment_id = item.get("payment_id")
        if payment_id:
            ids.add(payment_id)
    return ids
