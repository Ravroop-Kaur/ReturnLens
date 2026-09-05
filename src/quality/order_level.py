"""
Line-item vs order-level granularity.

The primary ML target is "will this ORDER eventually result in a
return", so a merchant export that has one row per line item (three
rows for a three-item order) must be aggregated to one row per order
before features/training/evaluation ever see it -- otherwise a single
three-item order gets counted as three independent training/test
examples, which both distorts the class balance and would let
order-level evaluation double- or triple-count outcomes.

This module only aggregates; it does not decide whether the result is
trainable (see src.quality.readiness for that).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Columns that get special aggregation treatment. Everything else is
# kept only if every row in the order agrees on its value; otherwise
# it becomes NaN (never fabricated, never arbitrarily picked).
SUM_COLS = {"amount"}
ANY_COLS = {"return_event", "refund_event", "chargeback_event"}
MIN_DATE_COLS = {"order_date"}
MAX_DATE_COLS = {"return_date"}


@dataclass
class GranularityReport:
    detected_granularity: str  # "order_level" | "line_item_level"
    n_rows: int
    n_orders: int
    max_rows_per_order: int

    def to_dict(self) -> dict:
        return {
            "detected_granularity": self.detected_granularity,
            "n_rows": self.n_rows,
            "n_orders": self.n_orders,
            "max_rows_per_order": self.max_rows_per_order,
        }


def detect_granularity(df: pd.DataFrame, order_id_col: str = "order_id") -> str:
    if order_id_col not in df.columns or len(df) == 0:
        return "order_level"
    counts = df.groupby(order_id_col, dropna=False).size()
    return "line_item_level" if counts.max() > 1 else "order_level"


def _agree_or_nan(series: pd.Series):
    values = series.dropna().unique()
    if len(values) == 0:
        return np.nan
    if len(values) == 1:
        return values[0]
    return np.nan


def aggregate_to_order_level(
    df: pd.DataFrame, order_id_col: str = "order_id"
) -> "tuple[pd.DataFrame, GranularityReport]":
    """
    Returns (order_level_df, report). If the data is already order
    level, the frame is returned unchanged (aside from a defensive
    copy). Otherwise every order_id's rows are collapsed into one row:

      - amount:           summed (total order value)
      - return/refund/chargeback event: True if ANY line item shows it
      - order_date:       earliest (the order's actual placement time)
      - return_date:      latest (final return event for the order)
      - everything else:  kept only if every line item agrees, else NaN
    """
    n_rows = len(df)
    n_orders = int(df[order_id_col].nunique(dropna=False)) if order_id_col in df.columns and n_rows else 0
    max_per_order = (
        int(df.groupby(order_id_col, dropna=False).size().max()) if order_id_col in df.columns and n_rows else 0
    )
    granularity = detect_granularity(df, order_id_col)
    report = GranularityReport(
        detected_granularity=granularity,
        n_rows=n_rows,
        n_orders=n_orders,
        max_rows_per_order=max_per_order,
    )

    if granularity == "order_level" or order_id_col not in df.columns:
        return df.copy(), report

    working = df.copy()
    if "order_date" in working.columns:
        working["order_date"] = pd.to_datetime(working["order_date"], errors="coerce")
    if "return_date" in working.columns:
        working["return_date"] = pd.to_datetime(working["return_date"], errors="coerce")

    agg_spec = {}
    for col in working.columns:
        if col == order_id_col:
            continue
        if col in SUM_COLS:
            agg_spec[col] = "sum"
        elif col in ANY_COLS:
            def _any_event(s):
                non_null = s.dropna()
                if len(non_null) == 0:
                    return pd.NA
                return bool(non_null.astype(bool).any())
            agg_spec[col] = _any_event
        elif col in MIN_DATE_COLS:
            agg_spec[col] = "min"
        elif col in MAX_DATE_COLS:
            agg_spec[col] = "max"
        else:
            agg_spec[col] = _agree_or_nan

    aggregated = working.groupby(order_id_col, as_index=False, dropna=False).agg(agg_spec)
    return aggregated, report
