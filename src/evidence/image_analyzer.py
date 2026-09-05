"""
Image evidence analyzer.

This is a SECONDARY signal for the return-claim evidence layer -- it
is NOT the primary ML return-risk detector (src.model), and its
output is never proof of anything on its own. An AI-generated or
manipulated image signal does not prove customer fraud, customer
intent, who caused any damage, or whether the product was actually
damaged. See IMAGE_DISCLAIMER, which every consumer of this module's
output must surface alongside the signal.

ImageEvidenceAnalyzer is an abstraction so a real pretrained
detector (clearly documented as pretrained, with no fabricated
accuracy claim -- see PART F2 of the spec) can be swapped in later
without changing src.claims.evidence. MockImageEvidenceAnalyzer is the
demo/mock implementation used when no such provider is configured; it
is always labelled is_demo=True and never emits an accuracy claim.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

IMAGE_DISCLAIMER = (
    "Image authenticity signals may indicate that additional review is "
    "useful. They do not prove fraud or customer intent."
)

# Below this confidence, the analyzer itself is telling us it cannot
# meaningfully distinguish normal from suspicious for this image.
LOW_CONFIDENCE_THRESHOLD = 0.3

# Any of these crossing threshold marks the image as worth a second look.
SUSPICIOUS_SCORE_THRESHOLD = 0.7
SUSPICIOUS_AUTHENTICITY_FLOOR = 0.3


@dataclass
class ImageEvidenceResult:
    authenticity_score: float
    manipulation_score: float
    duplicate_score: float
    confidence: float
    provider: str
    signals: list = field(default_factory=list)
    is_demo: bool = False
    disclaimer: str = IMAGE_DISCLAIMER

    def overall_signal(self) -> str:
        """NORMAL | SUSPICIOUS | INCONCLUSIVE -- never a fraud verdict."""
        if self.confidence < LOW_CONFIDENCE_THRESHOLD:
            return "INCONCLUSIVE"
        if (
            self.manipulation_score >= SUSPICIOUS_SCORE_THRESHOLD
            or self.duplicate_score >= SUSPICIOUS_SCORE_THRESHOLD
            or self.authenticity_score <= SUSPICIOUS_AUTHENTICITY_FLOOR
        ):
            return "SUSPICIOUS"
        return "NORMAL"


class ImageEvidenceAnalyzer(ABC):
    """Provider abstraction. A real implementation (e.g. wrapping a
    pretrained synthetic-image detector) must document that it is
    pretrained and must not claim a specific real-world accuracy
    unless backed by a genuine held-out evaluation."""

    @abstractmethod
    def analyze(self, image_ref: str) -> ImageEvidenceResult:
        raise NotImplementedError


class MockImageEvidenceAnalyzer(ImageEvidenceAnalyzer):
    """Deterministic, clearly-labelled demo analyzer. Scores are
    derived from a hash of the image reference so the same reference
    always analyzes the same way (useful for demos and tests), but
    they carry no genuine forensic signal and no accuracy claim."""

    PROVIDER_NAME = "mock_demo_v1"
    DEMO_CONFIDENCE = 0.7

    def analyze(self, image_ref: str) -> ImageEvidenceResult:
        digest = hashlib.sha256(str(image_ref).encode("utf-8")).hexdigest()
        authenticity = int(digest[0:8], 16) / 0xFFFFFFFF
        manipulation = int(digest[8:16], 16) / 0xFFFFFFFF
        duplicate = int(digest[16:24], 16) / 0xFFFFFFFF

        signals = []
        if manipulation >= SUSPICIOUS_SCORE_THRESHOLD:
            signals.append("elevated manipulation-likelihood signal (demo heuristic, not a real detector)")
        if duplicate >= SUSPICIOUS_SCORE_THRESHOLD:
            signals.append("elevated duplicate-image signal (demo heuristic, not a real detector)")
        if not signals:
            signals.append("no notable signals (demo heuristic, not a real detector)")

        return ImageEvidenceResult(
            authenticity_score=round(authenticity, 4),
            manipulation_score=round(manipulation, 4),
            duplicate_score=round(duplicate, 4),
            confidence=self.DEMO_CONFIDENCE,
            provider=self.PROVIDER_NAME,
            signals=signals,
            is_demo=True,
        )
