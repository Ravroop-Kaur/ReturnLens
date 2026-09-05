"""
Lightweight data drift monitoring.

Compares a "reference" window (typically the training set) against a
"current" window (typically the most recent held-out/test set, or a
later batch of incoming data) across a handful of business-meaningful
signals: return prevalence, order amount distribution, and the
distribution of a few categorical dimensions.

This module NEVER triggers an automatic retrain. It only classifies
each signal as NORMAL / MILD_DRIFT / SIGNIFICANT_DRIFT and leaves the
decision of whether to re-evaluate or retrain to a human (see
PART A10 / M of the product spec: no autonomous retraining).

Thresholds here are simple, explainable heuristics (population
stability-style bucketing + a proportion difference), not a claim of
statistical rigor -- the goal is an early, honest warning signal, not
a p-value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

NORMAL = "NORMAL"
MILD_DRIFT = "MILD_DRIFT"
SIGNIFICANT_DRIFT = "SIGNIFICANT_DRIFT"

# Absolute difference in proportion/rate at or above which a signal is
# flagged. These are deliberately simple, documented cutoffs -- not a
# tuned statistical model.
MILD_THRESHOLD = 0.05
SIGNIFICANT_THRESHOLD = 0.15

CATEGORICAL_DIMENSIONS = ["category", "region", "fulfilment_method", "shipping_service"]


@dataclass
class SignalDrift:
    signal: str
    status: str
    detail: dict = field(default_factory=dict)


@dataclass
class DriftReport:
    signals: list
    overall_status: str

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status,
            "signals": [{"signal": s.signal, "status": s.status, "detail": s.detail} for s in self.signals],
        }


def _classify(diff: float) -> str:
    diff = abs(diff)
    if diff >= SIGNIFICANT_THRESHOLD:
        return SIGNIFICANT_DRIFT
    if diff >= MILD_THRESHOLD:
        return MILD_DRIFT
    return NORMAL


def _rate_drift(name: str, ref: pd.Series, cur: pd.Series) -> SignalDrift:
    ref_rate = float(pd.Series(ref).dropna().astype(float).mean()) if len(ref.dropna()) else None
    cur_rate = float(pd.Series(cur).dropna().astype(float).mean()) if len(cur.dropna()) else None
    if ref_rate is None or cur_rate is None:
        return SignalDrift(signal=name, status=NORMAL, detail={"reason": "insufficient data on one side"})
    diff = cur_rate - ref_rate
    return SignalDrift(
        signal=name,
        status=_classify(diff),
        detail={"reference_rate": round(ref_rate, 4), "current_rate": round(cur_rate, 4), "abs_diff": round(abs(diff), 4)},
    )


def _numeric_distribution_drift(name: str, ref: pd.Series, cur: pd.Series, n_bins: int = 5) -> SignalDrift:
    """Bucket the reference window into quantile bins, then compare
    what fraction of the current window falls in the top and bottom
    bin vs. the expected ~1/n_bins each -- a simple, explainable proxy
    for distribution shift without requiring scipy KS-test machinery
    (which would give a false sense of statistical certainty here)."""
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    cur = pd.to_numeric(cur, errors="coerce").dropna()
    if len(ref) < 20 or len(cur) < 20:
        return SignalDrift(signal=name, status=NORMAL, detail={"reason": "insufficient data on one side"})

    try:
        quantile_edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    except Exception:
        return SignalDrift(signal=name, status=NORMAL, detail={"reason": "could not bin reference distribution"})

    quantile_edges[0] = -np.inf
    quantile_edges[-1] = np.inf
    expected_frac = 1.0 / n_bins

    cur_bins = pd.cut(cur, bins=quantile_edges, include_lowest=True)
    observed_fracs = cur_bins.value_counts(normalize=True, sort=False)
    max_deviation = float((observed_fracs - expected_frac).abs().max()) if len(observed_fracs) else 0.0

    return SignalDrift(
        signal=name,
        status=_classify(max_deviation),
        detail={
            "reference_median": round(float(np.median(ref)), 2),
            "current_median": round(float(np.median(cur)), 2),
            "max_bin_deviation": round(max_deviation, 4),
        },
    )


def _categorical_distribution_drift(name: str, ref: pd.Series, cur: pd.Series) -> SignalDrift:
    ref = ref.dropna().astype(str)
    cur = cur.dropna().astype(str)
    if len(ref) == 0 or len(cur) == 0:
        return SignalDrift(signal=name, status=NORMAL, detail={"reason": "insufficient data on one side"})

    ref_dist = ref.value_counts(normalize=True)
    cur_dist = cur.value_counts(normalize=True)
    all_categories = set(ref_dist.index) | set(cur_dist.index)

    # Total variation distance between the two categorical distributions.
    tv_distance = 0.5 * sum(abs(ref_dist.get(c, 0.0) - cur_dist.get(c, 0.0)) for c in all_categories)
    unseen_in_ref = sorted(set(cur_dist.index) - set(ref_dist.index))

    return SignalDrift(
        signal=name,
        status=_classify(tv_distance),
        detail={
            "total_variation_distance": round(tv_distance, 4),
            "new_categories_not_in_reference": unseen_in_ref[:10],
        },
    )


def check_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftReport:
    """Compare `current_df` (e.g. the latest held-out test window, or
    a fresh batch of merchant data) against `reference_df` (e.g. the
    training window) across return prevalence, order amount, and any
    available categorical dimensions."""
    signals = []

    if "return_event" in reference_df.columns and "return_event" in current_df.columns:
        signals.append(_rate_drift("return_prevalence", reference_df["return_event"], current_df["return_event"]))

    if "amount" in reference_df.columns and "amount" in current_df.columns:
        signals.append(_numeric_distribution_drift("amount_distribution", reference_df["amount"], current_df["amount"]))

    for dim in CATEGORICAL_DIMENSIONS:
        if dim in reference_df.columns and dim in current_df.columns:
            signals.append(_categorical_distribution_drift(f"{dim}_distribution", reference_df[dim], current_df[dim]))

    for col in ["category", "region", "fulfilment_method", "shipping_service", "amount"]:
        if col in reference_df.columns and col in current_df.columns:
            ref_missing = float(reference_df[col].isna().mean())
            cur_missing = float(current_df[col].isna().mean())
            diff = cur_missing - ref_missing
            signals.append(SignalDrift(
                signal=f"{col}_missingness",
                status=_classify(diff),
                detail={"reference_missing_rate": round(ref_missing, 4), "current_missing_rate": round(cur_missing, 4)},
            ))

    if not signals:
        overall = NORMAL
    elif any(s.status == SIGNIFICANT_DRIFT for s in signals):
        overall = SIGNIFICANT_DRIFT
    elif any(s.status == MILD_DRIFT for s in signals):
        overall = MILD_DRIFT
    else:
        overall = NORMAL

    return DriftReport(signals=signals, overall_status=overall)
