"""
Return-claim data representation (PART E1).

Every field is Optional except claim_id/organization_id/order_id/
claimed_reason -- a real merchant integration will rarely supply every
field, and this module must handle that gracefully rather than
assuming the demo shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReturnClaim:
    claim_id: str
    organization_id: str
    order_id: str
    claimed_reason: str
    claim_timestamp: Optional[str] = None
    image_references: list = field(default_factory=list)

    # Order/product/delivery context, where available -- never
    # fabricated if the merchant's data doesn't have it.
    order_amount: Optional[float] = None
    order_date: Optional[str] = None
    product_category: Optional[str] = None
    delivery_status: Optional[str] = None
    delivery_date: Optional[str] = None

    # Customer/return history signals, where available.
    customer_prior_return_count: Optional[int] = None
    customer_prior_order_count: Optional[int] = None

    final_status: Optional[str] = None  # set once a human reviewer resolves the claim
