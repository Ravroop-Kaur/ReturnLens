from src.claims.model import ReturnClaim
from src.claims.evidence import (
    aggregate_evidence, normalize_reason,
    SUPPORTED, NEEDS_REVIEW, SUSPICIOUS, INDETERMINATE,
)
from src.evidence.image_analyzer import MockImageEvidenceAnalyzer


def test_normalize_reason_confident_synonym():
    assert normalize_reason("Arrived damaged") == "DAMAGED"
    assert normalize_reason("product fault") == "DAMAGED"


def test_normalize_reason_ambiguous_returns_none():
    assert normalize_reason("it just wasn't right for me somehow") is None


def _base_claim(**overrides) -> ReturnClaim:
    defaults = dict(
        claim_id="c1", organization_id="org_1", order_id="o1",
        claimed_reason="Arrived damaged",
        claim_timestamp="2025-01-05", order_date="2025-01-01",
        delivery_status="delivered", customer_prior_return_count=0,
    )
    defaults.update(overrides)
    return ReturnClaim(**defaults)


def test_never_declares_fraud_confirmed():
    claim = _base_claim()
    evidence = aggregate_evidence(claim)
    assert evidence.status in (SUPPORTED, NEEDS_REVIEW, SUSPICIOUS, INDETERMINATE)
    text_blob = str(evidence.to_dict()).lower()
    assert "fraud confirmed" not in text_blob
    assert "customer is fraudulent" not in text_blob


def test_well_evidenced_claim_is_supported():
    claim = _base_claim()
    evidence = aggregate_evidence(claim)
    assert evidence.status == SUPPORTED


def test_elevated_return_history_flagged_needs_review_or_suspicious():
    claim = _base_claim(customer_prior_return_count=5)
    evidence = aggregate_evidence(claim)
    assert evidence.status in (NEEDS_REVIEW, SUSPICIOUS)
    assert any("elevated" in s.lower() for s in evidence.contradictory_signals)


def test_missing_fields_produce_indeterminate_not_a_guess():
    claim = ReturnClaim(
        claim_id="c2", organization_id="org_1", order_id="o2",
        claimed_reason="not sure what happened",
    )
    evidence = aggregate_evidence(claim)
    assert evidence.status == INDETERMINATE
    assert len(evidence.missing_evidence) >= 3


def test_suspicious_image_signal_surfaces_as_contradictory():
    claim = _base_claim(image_references=["img_high_manip_ref"])
    analyzer = MockImageEvidenceAnalyzer()
    # find a reference whose deterministic hash yields a suspicious signal
    result = None
    for i in range(50):
        candidate = analyzer.analyze(f"ref_{i}")
        if candidate.overall_signal() == "SUSPICIOUS":
            result = candidate
            break
    assert result is not None, "expected at least one suspicious mock result in 50 tries"
    evidence = aggregate_evidence(claim, image_result=result)
    assert evidence.image_signal["overall_signal"] == "SUSPICIOUS"
    assert any("image" in s.lower() for s in evidence.contradictory_signals)


def test_disclaimer_always_present():
    claim = _base_claim()
    evidence = aggregate_evidence(claim)
    assert "do not prove fraud" in evidence.disclaimer.lower()
