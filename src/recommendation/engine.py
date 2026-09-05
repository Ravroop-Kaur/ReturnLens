"""
Bounded recommendation engine.

Turns a statistical finding into a SUGGESTED, human-actioned review
step. This module never executes any real-world action (no refunds,
no courier changes, no autonomous merchant account changes). It only
produces text + the evidence that supports it.
"""

from __future__ import annotations
from dataclasses import dataclass


DIMENSION_TEMPLATES = {
    "fulfilment_method": "Review fulfilment performance for the '{segment}' segment "
                          "(e.g. packaging, listing accuracy, or handling quality).",
    "shipping_service": "Investigate delivery performance for the '{segment}' shipping "
                         "service (e.g. transit damage or delivery delays).",
    "category": "Review product quality, sizing/listing accuracy, or return drivers "
                "for the '{segment}' category.",
    "region": "Investigate regional delivery or courier performance for the '{segment}' region.",
}

DEFAULT_TEMPLATE = "Review the '{segment}' segment of {dimension}, which shows an elevated observed return rate."


@dataclass
class Recommendation:
    text: str
    dimension: str
    segment: str
    relative_risk: float
    evidence_summary: str
    bounded: bool = True
    autonomous: bool = False


def recommend(finding) -> "Recommendation | None":
    if finding is None:
        return None
    template = DIMENSION_TEMPLATES.get(finding.dimension, DEFAULT_TEMPLATE)
    text = template.format(segment=finding.segment, dimension=finding.dimension)
    return Recommendation(
        text=text,
        dimension=finding.dimension,
        segment=finding.segment,
        relative_risk=finding.relative_risk,
        evidence_summary=finding.plain_english(),
    )
