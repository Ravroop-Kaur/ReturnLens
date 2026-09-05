"""
Explicit feature contract.

The product must never assume a merchant's data looks like the
synthetic demo. Every canonical field is classified into exactly one
of five states, and the model is only allowed to run when the
REQUIRED fields are all usable:

    REQUIRED         -- required for training/scoring, and usable
    REQUIRED_MISSING -- required, but absent or unusable -> hard abstain
    RECOMMENDED      -- meaningfully improves the model, present and usable
    OPTIONAL         -- nice-to-have, present and usable
    NOT_AVAILABLE    -- column absent from this merchant's data entirely
    NOT_USABLE       -- column present, but too much of it is missing to
                        trust (crosses NOT_USABLE_MISSING_THRESHOLD)

Nothing here fabricates a value for a missing/unusable field -- it is
purely a report the readiness pipeline and the UI both consult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import pandas as pd

# Fraction of nulls at or above which a *present* column is
# considered too sparse to be trusted, regardless of its tier.
NOT_USABLE_MISSING_THRESHOLD = 0.8

REQUIRED_FIELDS = ["order_id", "order_date", "amount"]
TARGET_FIELDS = ["return_event"]
RECOMMENDED_FIELDS = ["category", "region", "product_id"]
OPTIONAL_FIELDS = [
    "fulfilment_method",
    "shipping_service",
    "payment_status",
    "delivery_status",
    "review_text",
]


class FeatureStatus(str, Enum):
    REQUIRED = "REQUIRED"
    REQUIRED_MISSING = "REQUIRED_MISSING"
    RECOMMENDED = "RECOMMENDED"
    OPTIONAL = "OPTIONAL"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    NOT_USABLE = "NOT_USABLE"


@dataclass
class FeatureContractResult:
    statuses: Dict[str, FeatureStatus]
    missing_fractions: Dict[str, float] = field(default_factory=dict)
    require_target: bool = True

    def has_all_required(self) -> bool:
        return all(
            self.statuses.get(f) not in (FeatureStatus.REQUIRED_MISSING, FeatureStatus.NOT_USABLE)
            for f in REQUIRED_FIELDS
        )

    def has_usable_target(self) -> bool:
        target_status = self.statuses.get("return_event")
        if not self.require_target:
            return target_status not in (FeatureStatus.NOT_USABLE,)
        return target_status not in (FeatureStatus.REQUIRED_MISSING, FeatureStatus.NOT_USABLE)

    def readiness_label(self) -> str:
        if not self.has_all_required() or not self.has_usable_target():
            return "NOT_READY"
        secondary = [
            status
            for field_name, status in self.statuses.items()
            if field_name not in REQUIRED_FIELDS and field_name not in TARGET_FIELDS
        ]
        if any(status in (FeatureStatus.NOT_AVAILABLE, FeatureStatus.NOT_USABLE) for status in secondary):
            return "PARTIALLY_SUPPORTED"
        return "FULLY_SUPPORTED"

    def summary_table(self) -> dict:
        return {field_name: status.value for field_name, status in self.statuses.items()}

    def rows_for_ui(self) -> list:
        """Ordered list of {field, status, missingness} for rendering,
        e.g. in the DATA READINESS UI (spec G2)."""
        order = REQUIRED_FIELDS + TARGET_FIELDS + RECOMMENDED_FIELDS + OPTIONAL_FIELDS
        return [
            {
                "field": field_name,
                "status": self.statuses[field_name].value,
                "missingness": self.missing_fractions.get(field_name),
            }
            for field_name in order
            if field_name in self.statuses
        ]


def _missing_fraction(df: pd.DataFrame, field_name: str) -> float:
    if len(df) == 0:
        return 1.0
    return float(df[field_name].isna().mean())


def _status_for_field(df: pd.DataFrame, field_name: str, tier: str, require_target: bool) -> FeatureStatus:
    if field_name not in df.columns:
        if tier == "required":
            return FeatureStatus.REQUIRED_MISSING
        if tier == "target":
            return FeatureStatus.REQUIRED_MISSING if require_target else FeatureStatus.NOT_AVAILABLE
        return FeatureStatus.NOT_AVAILABLE

    missing_frac = _missing_fraction(df, field_name)
    if missing_frac >= NOT_USABLE_MISSING_THRESHOLD:
        return FeatureStatus.NOT_USABLE

    if tier == "required":
        return FeatureStatus.REQUIRED
    if tier == "target":
        return FeatureStatus.REQUIRED if require_target else FeatureStatus.OPTIONAL
    if tier == "recommended":
        return FeatureStatus.RECOMMENDED
    return FeatureStatus.OPTIONAL


def evaluate_feature_contract(df: pd.DataFrame, require_target: bool = True) -> FeatureContractResult:
    statuses: Dict[str, FeatureStatus] = {}
    missing_fractions: Dict[str, float] = {}

    field_tiers = (
        [(f, "required") for f in REQUIRED_FIELDS]
        + [(f, "target") for f in TARGET_FIELDS]
        + [(f, "recommended") for f in RECOMMENDED_FIELDS]
        + [(f, "optional") for f in OPTIONAL_FIELDS]
    )

    for field_name, tier in field_tiers:
        statuses[field_name] = _status_for_field(df, field_name, tier, require_target)
        if field_name in df.columns:
            missing_fractions[field_name] = _missing_fraction(df, field_name)

    return FeatureContractResult(statuses=statuses, missing_fractions=missing_fractions, require_target=require_target)
