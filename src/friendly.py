"""
Small, explicit mapping from raw canonical field values / dimension
names to merchant-friendly display labels.

Deliberately a fixed lookup table, not a generic "snake_case ->
Title Case" transformer: guessed transformations can produce wrong or
awkward text for values we haven't checked (e.g. acronyms). Only
values actually produced by the canonical schema (see
src/canonical/schema.py) are listed here; anything else is returned
unchanged so a new/unknown value never crashes the UI, it just isn't
prettified yet.
"""

from __future__ import annotations

_FRIENDLY_VALUES = {
    "platform_fulfilled": "Platform-fulfilled",
    "third_party_fulfilled": "Third-party fulfilled",
    "merchant_fulfilled": "Merchant-fulfilled",
}

_FRIENDLY_DIMENSIONS = {
    "fulfilment_method": "Fulfilment method",
    "shipping_service": "Shipping service",
    "category": "Product category",
    "region": "Region",
}


def friendly_value(value) -> str:
    """Map a raw field value to a merchant-friendly label. Values not
    in the lookup table (already human-readable strings like
    "Apparel" or "Economy") are returned unchanged."""
    if value is None:
        return value
    text = str(value)
    return _FRIENDLY_VALUES.get(text, text)


def friendly_dimension(dimension: str) -> str:
    """Map a raw dimension/field name to a merchant-friendly label."""
    if dimension is None:
        return dimension
    return _FRIENDLY_DIMENSIONS.get(dimension, dimension)
