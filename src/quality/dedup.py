"""
Duplicate detection and handling.

Real merchant exports commonly contain three very different things
that all *look* like "duplicate order IDs" at a glance:

1. Exact duplicate rows (a re-export, a retried upload) -- safe to drop.
2. Legitimate multi-line orders (order 123 has three products, so it
   has three rows sharing order_id 123) -- must NOT be collapsed by a
   naive drop_duplicates(subset=["order_id"]).
3. Genuinely conflicting records for the *same* line (e.g. two rows
   for the same order_id + product_id disagree on whether it was
   returned) -- these are a data-quality problem to surface, not to
   silently resolve one way or the other.

This module only classifies and reports; see src.quality.order_level
for how line items get aggregated into an order-level row once this
report says it is safe to do so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# Fields that represent an order's outcome. Two rows for the same
# order_id (and, where present, the same product_id) that disagree on
# one of these are a genuine conflict, not a legitimate line item.
OUTCOME_FIELDS = ["return_event", "refund_event", "chargeback_event"]


@dataclass
class DuplicateReport:
    n_duplicate_rows: int
    duplicate_order_ids: list = field(default_factory=list)
    n_line_item_groups: int = 0
    n_conflicting_groups: int = 0
    conflicting_order_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_duplicate_rows": self.n_duplicate_rows,
            "n_duplicate_order_id_groups": len(self.duplicate_order_ids),
            "n_line_item_groups": self.n_line_item_groups,
            "n_conflicting_groups": self.n_conflicting_groups,
            "conflicting_order_ids": self.conflicting_order_ids[:50],
        }


def drop_exact_duplicates(df: pd.DataFrame) -> "tuple[pd.DataFrame, int]":
    """Drop rows that are exact duplicates across every column. This is
    always safe -- it never removes a legitimate line item, since a
    real second line item will differ in at least product_id/amount."""
    deduped = df.drop_duplicates().reset_index(drop=True)
    n_dropped = len(df) - len(deduped)
    return deduped, n_dropped


def analyze_duplicates(df: pd.DataFrame, order_id_col: str = "order_id") -> DuplicateReport:
    """Classify every order_id that appears more than once."""
    if order_id_col not in df.columns or len(df) == 0:
        _, n_exact = drop_exact_duplicates(df)
        return DuplicateReport(n_duplicate_rows=n_exact)

    _, n_exact_dropped = drop_exact_duplicates(df)

    duplicate_order_ids: list = []
    n_line_item_groups = 0
    n_conflicting_groups = 0
    conflicting_order_ids: list = []

    for order_id, group in df.groupby(order_id_col, dropna=False):
        if len(group) <= 1:
            continue
        duplicate_order_ids.append(order_id)

        is_line_item = False
        if "product_id" in group.columns:
            n_unique_products = group["product_id"].nunique(dropna=False)
            if n_unique_products == len(group) and n_unique_products > 1:
                is_line_item = True

        if is_line_item:
            n_line_item_groups += 1
            continue

        # Same product (or no product_id to distinguish rows at all) --
        # this is either a re-delivered duplicate event or a genuine
        # conflict. Only outcome-field disagreement is flagged as a
        # conflict; identical repeated rows were already handled by
        # drop_exact_duplicates.
        has_conflict = False
        for field_name in OUTCOME_FIELDS:
            if field_name in group.columns:
                values = group[field_name].dropna().unique()
                if len(values) > 1:
                    has_conflict = True
                    break

        if has_conflict:
            n_conflicting_groups += 1
            conflicting_order_ids.append(order_id)

    return DuplicateReport(
        n_duplicate_rows=n_exact_dropped,
        duplicate_order_ids=duplicate_order_ids,
        n_line_item_groups=n_line_item_groups,
        n_conflicting_groups=n_conflicting_groups,
        conflicting_order_ids=conflicting_order_ids,
    )
