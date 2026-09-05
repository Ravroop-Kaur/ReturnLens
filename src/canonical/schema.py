"""
Canonical merchant order schema.

This is the single internal representation that every merchant data
source (generic CSV, Amazon-style export, Razorpay Test Mode, etc.)
must be mapped into before any feature engineering, modelling,
diagnosis or exposure calculation happens.

The rest of the system (features, model, diagnosis, exposure,
recommendation, verification) NEVER reads a source-specific column
name. It only ever reads these canonical fields. This is what makes
the product merchant-agnostic instead of "an Amazon dashboard".
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Canonical column names
# ---------------------------------------------------------------------------
# Only fields that are actually used downstream are included. We do not
# model concepts "just in case" -- every field here is either a feature
# input, the prediction target, an identifier, or used for exposure /
# diagnosis grouping.

REQUIRED_FIELDS = [
    "order_id",        # unique order / transaction identifier
    "order_date",      # date/time the order was placed (prediction point)
    "amount",          # order value, used for exposure calculations
]

OPTIONAL_FIELDS = [
    "product_id",           # product identifier
    "customer_id",          # stable customer identifier for historical behaviour
    "category",             # product category
    "region",                # shipping / customer region
    "fulfilment_method",     # e.g. "merchant_fulfilled" / "platform_fulfilled"
    "shipping_service",      # named shipping/courier service
    "payment_status",        # payment status at prediction time (e.g. paid, cod, pending)
    "delivery_status",       # delivery status at prediction time, if known pre-outcome
    "review_text",           # optional free-text review (NOT used for prediction features
                              # in this MVP -- see docs/leakage_notes.md)
]

# Outcome / target-related fields. These are NEVER used as model
# features. They exist only to construct the label and for
# post-hoc evaluation / exposure calculation.
OUTCOME_FIELDS = [
    "return_event",     # bool: did this order eventually result in a return?
    "return_date",       # date the return was recorded, if any
    "refund_event",      # bool: was a refund issued (may lag return_event)
    "chargeback_event",  # bool: was a chargeback filed
]

# Multi-tenancy identifiers. Not modelled on in this MVP, but present
# in the schema so a future multi-merchant deployment does not require
# a schema migration.
TENANCY_FIELDS = [
    "organization_id",
    "merchant_id",
    "data_source_id",
]

ALL_CANONICAL_FIELDS = (
    REQUIRED_FIELDS + OPTIONAL_FIELDS + OUTCOME_FIELDS + TENANCY_FIELDS
)

# Fields that must NEVER be used as a predictive feature because they
# are either the target itself or only known after the outcome.
LEAKAGE_FORBIDDEN_FIELDS = {
    "return_event",
    "return_date",
    "refund_event",
    "chargeback_event",
}


@dataclass
class CanonicalCapabilities:
    """
    What can this dataset actually support, given the columns that were
    successfully mapped? The UI and the pipeline must both consult this
    instead of assuming every merchant dataset looks like the demo data.
    """
    has_target: bool = False
    has_category: bool = False
    has_region: bool = False
    has_fulfilment: bool = False
    has_shipping_service: bool = False
    has_review_text: bool = False
    missing_required: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "return_outcome_available": self.has_target,
            "category_available": self.has_category,
            "region_available": self.has_region,
            "fulfilment_available": self.has_fulfilment,
            "shipping_service_available": self.has_shipping_service,
            "review_text_available": self.has_review_text,
            "missing_required_fields": self.missing_required,
        }
