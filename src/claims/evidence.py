"""
Return-claim evidence aggregation (PART E2 / E3).

This is a SECONDARY signal layer, entirely separate from the primary
ML return-risk detector (src.model). It never declares a customer
fraudulent -- the only possible statuses are SUPPORTED, NEEDS_REVIEW,
SUSPICIOUS, and INDETERMINATE, all of which are explicitly
human-review framings, never a fraud verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.claims.model import ReturnClaim

SUPPORTED = "SUPPORTED"
NEEDS_REVIEW = "NEEDS_REVIEW"
SUSPICIOUS = "SUSPICIOUS"
INDETERMINATE = "INDETERMINATE"

DISCLAIMER = (
    "These are evidence signals for human review. They do not prove fraud, "
    "customer intent, or responsibility for the damage."
)

# Controlled taxonomy. Only reasons with a confident, unambiguous
# synonym are normalized; anything else is left as the merchant's
# original free text rather than force-fit into a category.
REASON_TAXONOMY = {
    "DAMAGED": ["damaged", "broken", "defective", "arrived damaged", "product fault", "not working", "faulty"],
    "WRONG_ITEM": ["wrong item", "wrong product", "incorrect item", "mismatched item"],
    "SIZE_FIT": ["wrong size", "doesn't fit", "size issue", "too small", "too large"],
    "NOT_AS_DESCRIBED": ["not as described", "different from listing", "misleading listing"],
    "CHANGED_MIND": ["changed my mind", "no longer needed", "ordered by mistake"],
    "LATE_DELIVERY": ["arrived late", "late delivery", "delayed"],
}

HIGH_RETURN_HISTORY_THRESHOLD = 3  # prior returns at/above this is "elevated" history


def normalize_reason(raw_reason: Optional[str]) -> Optional[str]:
    if not raw_reason:
        return None
    text = raw_reason.strip().lower()
    for canonical, synonyms in REASON_TAXONOMY.items():
        if any(syn in text for syn in synonyms):
            return canonical
    return None  # ambiguous -- do not silently force a category


@dataclass
class EvidenceAggregate:
    claim_id: str
    status: str
    confidence: str  # "low" | "medium" | "high" -- qualitative, not a fabricated number
    normalized_reason: Optional[str]
    raw_reason: str
    supporting_signals: list = field(default_factory=list)
    contradictory_signals: list = field(default_factory=list)
    missing_evidence: list = field(default_factory=list)
    human_review_recommended: bool = True
    image_signal: Optional[dict] = None
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "status": self.status,
            "confidence": self.confidence,
            "normalized_reason": self.normalized_reason,
            "raw_reason": self.raw_reason,
            "supporting_signals": self.supporting_signals,
            "contradictory_signals": self.contradictory_signals,
            "missing_evidence": self.missing_evidence,
            "human_review_recommended": self.human_review_recommended,
            "image_signal": self.image_signal,
            "disclaimer": self.disclaimer,
        }


def aggregate_evidence(claim: ReturnClaim, image_result=None) -> EvidenceAggregate:
    """`image_result`, if provided, is an
    src.evidence.image_analyzer.ImageEvidenceResult."""
    supporting = []
    contradictory = []
    missing = []

    normalized = normalize_reason(claim.claimed_reason)
    if normalized is None:
        missing.append("Claimed reason could not be confidently normalized into the controlled taxonomy.")

    # ORDER TIMELINE: consistent if we have both an order date and a
    # claim timestamp and the claim comes after the order.
    if claim.order_date and claim.claim_timestamp:
        supporting.append("Order timeline consistent (claim filed after order placement).")
    else:
        missing.append("Order timeline could not be verified (order_date or claim_timestamp unavailable).")

    # DELIVERY: no contradictory event if delivery_status doesn't
    # conflict with a damage/defect-style claim (e.g. claim says
    # "damaged" but delivery_status says "returned to sender/undelivered").
    if claim.delivery_status:
        conflicting_statuses = {"undelivered", "lost", "returned_to_sender"}
        if claim.delivery_status.strip().lower() in conflicting_statuses and normalized in ("DAMAGED", "WRONG_ITEM", "SIZE_FIT"):
            contradictory.append(
                f"Delivery status '{claim.delivery_status}' is inconsistent with a claim about the received product."
            )
        else:
            supporting.append("No contradictory delivery event found.")
    else:
        missing.append("Delivery status unavailable.")

    # RETURN HISTORY
    if claim.customer_prior_return_count is not None:
        if claim.customer_prior_return_count >= HIGH_RETURN_HISTORY_THRESHOLD:
            contradictory.append(
                f"Customer has {claim.customer_prior_return_count} prior returns -- elevated return history."
            )
        else:
            supporting.append("Customer return history is not elevated.")
    else:
        missing.append("Customer return history unavailable.")

    # IMAGE EVIDENCE (secondary signal only)
    image_signal_dict = None
    if image_result is not None:
        image_signal_dict = {
            "overall_signal": image_result.overall_signal(),
            "authenticity_score": image_result.authenticity_score,
            "manipulation_score": image_result.manipulation_score,
            "duplicate_score": image_result.duplicate_score,
            "is_demo": image_result.is_demo,
            "disclaimer": image_result.disclaimer,
        }
        overall = image_result.overall_signal()
        if overall == "SUSPICIOUS":
            contradictory.append("Image authenticity signal suspicious.")
        elif overall == "NORMAL":
            supporting.append("Image authenticity signal normal.")
        else:
            missing.append("Image authenticity signal inconclusive (low confidence).")
    elif claim.image_references:
        missing.append("Images were provided but no image analyzer was run.")
    else:
        # Image evidence is a secondary, optional signal -- most claims
        # never carry photos, so its absence is noted for transparency
        # but must not by itself block a status of SUPPORTED.
        missing.append("No images provided with this claim (optional signal).")

    # Gaps that genuinely block confidence in the claim (excludes the
    # image-absence note above, since images are an optional signal).
    blocking_missing = [m for m in missing if "optional signal" not in m]

    # Overall status -- never a fraud verdict.
    if contradictory:
        status = SUSPICIOUS if len(contradictory) >= 2 else NEEDS_REVIEW
        confidence = "medium" if len(contradictory) < 2 else "high"
    elif len(blocking_missing) >= 3:
        status = INDETERMINATE
        confidence = "low"
    elif supporting and not blocking_missing:
        status = SUPPORTED
        confidence = "medium"
    else:
        status = NEEDS_REVIEW
        confidence = "low"

    return EvidenceAggregate(
        claim_id=claim.claim_id,
        status=status,
        confidence=confidence,
        normalized_reason=normalized,
        raw_reason=claim.claimed_reason,
        supporting_signals=supporting,
        contradictory_signals=contradictory,
        missing_evidence=missing,
        human_review_recommended=(status != SUPPORTED),
        image_signal=image_signal_dict,
    )
