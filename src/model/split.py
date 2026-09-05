"""
Time-aware train/validation/test split.

Because returns unfold over time and merchant behaviour drifts
(seasonality, catalog changes, courier performance), a random
row-level split would let the model "see the future" relative to
individual test orders that were placed earlier than some training
orders. We instead split strictly by order_date into three
contiguous, non-overlapping periods.

The test period is the LATEST period and must never be touched for
model selection, feature selection, or threshold tuning.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass


@dataclass
class SplitBounds:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "order_date",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitBounds]":
    """
    Split df (must contain date_col) into (train, val, test) by
    chronological quantile cut points. test_frac = 1 - train_frac - val_frac.
    """
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1

    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end_idx].copy()
    val = df.iloc[train_end_idx:val_end_idx].copy()
    test = df.iloc[val_end_idx:].copy()

    bounds = SplitBounds(
        train_end=train[date_col].max(),
        val_end=val[date_col].max() if len(val) else train[date_col].max(),
        test_end=test[date_col].max() if len(test) else val[date_col].max(),
    )
    return train, val, test, bounds
