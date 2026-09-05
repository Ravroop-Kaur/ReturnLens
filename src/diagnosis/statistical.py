"""
Statistical diagnosis engine.

Answers: "Where is return risk concentrated, and what evidence
supports that?" -- kept deliberately separate from the ML model, and
deliberately observational. This module NEVER claims causation. It
reports observed association, effect size, a practical-significance
threshold, and a confidence interval, then ranks segments by
strength of evidence.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class SegmentFinding:
    dimension: str
    segment: str
    segment_return_rate: float
    baseline_return_rate: float
    relative_risk: float
    segment_n: int
    baseline_n: int
    ci_low: float
    ci_high: float
    practically_significant: bool
    statistically_supported: bool

    def plain_english(self) -> str:
        rr = self.relative_risk
        return (
            f"Orders in segment '{self.segment}' ({self.dimension}) show an observed "
            f"return rate of {self.segment_return_rate:.1%}, compared with {self.baseline_return_rate:.1%} "
            f"for the rest of the data -- about {rr:.1f}x higher. This is an observed association, not a "
            f"proven cause."
        )


def _wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def diagnose_dimension(
    df: pd.DataFrame,
    dimension: str,
    target_col: str = "return_event",
    min_segment_size: int = 30,
    practical_effect_threshold: float = 1.3,
) -> list:
    """
    For every value of `dimension` (e.g. every fulfilment_method),
    compare its return rate against the rest of the data (the
    baseline = everyone NOT in that segment), using a Wilson interval
    for the segment rate and a simple relative-risk effect size.

    A finding is flagged `statistically_supported` if the segment's
    confidence interval does not include the baseline rate, AND
    `practically_significant` if the relative risk exceeds the given
    threshold. Both must hold for a finding to be surfaced as a
    strong pattern; segments below min_segment_size are excluded
    entirely (too little evidence to say anything).
    """
    if dimension not in df.columns or target_col not in df.columns:
        return []

    findings = []
    y = df[target_col].astype(int)

    for segment_value in df[dimension].dropna().unique():
        seg_mask = df[dimension] == segment_value
        seg_n = int(seg_mask.sum())
        if seg_n < min_segment_size:
            continue

        seg_successes = int(y[seg_mask].sum())
        seg_rate = seg_successes / seg_n

        base_mask = ~seg_mask
        base_n = int(base_mask.sum())
        if base_n == 0:
            continue
        base_successes = int(y[base_mask].sum())
        base_rate = base_successes / base_n if base_n else 0.0

        ci_low, ci_high = _wilson_ci(seg_successes, seg_n)

        relative_risk = (seg_rate / base_rate) if base_rate > 0 else float("inf")

        statistically_supported = ci_low > base_rate  # segment CI entirely above baseline
        practically_significant = relative_risk >= practical_effect_threshold

        findings.append(SegmentFinding(
            dimension=dimension,
            segment=str(segment_value),
            segment_return_rate=seg_rate,
            baseline_return_rate=base_rate,
            relative_risk=relative_risk,
            segment_n=seg_n,
            baseline_n=base_n,
            ci_low=ci_low,
            ci_high=ci_high,
            practically_significant=practically_significant,
            statistically_supported=statistically_supported,
        ))

    findings.sort(key=lambda f: (f.statistically_supported, f.practically_significant, f.relative_risk), reverse=True)
    return findings


def run_full_diagnosis(df: pd.DataFrame, target_col: str = "return_event") -> dict:
    dimensions = ["fulfilment_method", "category", "region", "shipping_service"]
    results = {}
    for dim in dimensions:
        if dim in df.columns:
            results[dim] = diagnose_dimension(df, dim, target_col=target_col)
    return results


def top_finding(diagnosis_results: dict):
    """Return the single strongest, evidence-backed finding across all
    dimensions, or None if nothing meets the bar."""
    candidates = []
    for dim, findings in diagnosis_results.items():
        for f in findings:
            if f.statistically_supported and f.practically_significant:
                candidates.append(f)
    if not candidates:
        return None
    candidates.sort(key=lambda f: f.relative_risk, reverse=True)
    return candidates[0]
