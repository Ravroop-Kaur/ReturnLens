"""
Return-label lifecycle.

Returns happen *after* orders. A recent order that has not returned
yet is not evidence of "no return" -- its return window simply hasn't
closed. Treating it as a negative label would poison training with
fabricated negatives and poison evaluation with an inflated apparent
precision/recall on labels that were never really settled.

Every order's label is one of:

    RETURNED   -- return_event is True. Final, regardless of recency.
    NO_RETURN  -- return_event is False/absent AND the return window
                  (return_window_days from order_date) has closed.
    PENDING    -- return_event is False/absent AND the window has NOT
                  closed yet, or order_date itself is unknown.

Only RETURNED and NO_RETURN are "finalized" and may be used for
supervised training/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_RETURN_WINDOW_DAYS = 30

RETURNED = "RETURNED"
NO_RETURN = "NO_RETURN"
PENDING = "PENDING"


@dataclass
class LifecycleReport:
    n_returned: int
    n_no_return: int
    n_pending: int
    n_total: int

    def to_dict(self) -> dict:
        return {
            "n_returned": self.n_returned,
            "n_no_return": self.n_no_return,
            "n_pending": self.n_pending,
            "n_total": self.n_total,
            "target_completeness": round((self.n_returned + self.n_no_return) / self.n_total, 4)
            if self.n_total
            else 0.0,
        }


def assign_label_state(
    df: pd.DataFrame,
    as_of: "pd.Timestamp | None" = None,
    return_window_days: int = DEFAULT_RETURN_WINDOW_DAYS,
) -> pd.Series:
    if as_of is None:
        as_of = pd.Timestamp.now()

    order_date = (
        pd.to_datetime(df["order_date"], errors="coerce")
        if "order_date" in df.columns
        else pd.Series([pd.NaT] * len(df), index=df.index)
    )
    return_event = (
        df["return_event"] if "return_event" in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
    )

    states = []
    for idx in df.index:
        oe = return_event.loc[idx]
        if pd.notna(oe) and bool(oe):
            states.append(RETURNED)
            continue
        od = order_date.loc[idx]
        if pd.isna(od):
            # Can't determine age -- do not assume the window has
            # closed just because we lack the date.
            states.append(PENDING)
            continue
        age_days = (as_of - od).days
        states.append(PENDING if age_days < return_window_days else NO_RETURN)

    return pd.Series(states, index=df.index)


def report(state: pd.Series) -> LifecycleReport:
    return LifecycleReport(
        n_returned=int((state == RETURNED).sum()),
        n_no_return=int((state == NO_RETURN).sum()),
        n_pending=int((state == PENDING).sum()),
        n_total=len(state),
    )


def usable_for_supervision(df: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
    """Only RETURNED / NO_RETURN rows may be used for supervised
    training or evaluation -- PENDING rows are excluded, never
    silently coerced into a negative."""
    mask = state != PENDING
    return df[mask].reset_index(drop=True)
