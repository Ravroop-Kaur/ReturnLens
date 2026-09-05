from src.evidence.image_analyzer import (
    ImageEvidenceAnalyzer, MockImageEvidenceAnalyzer, IMAGE_DISCLAIMER,
)


def test_mock_analyzer_is_labelled_demo():
    result = MockImageEvidenceAnalyzer().analyze("some_ref")
    assert result.is_demo is True
    assert result.provider == "mock_demo_v1"


def test_mock_analyzer_deterministic_for_same_reference():
    analyzer = MockImageEvidenceAnalyzer()
    r1 = analyzer.analyze("same_ref_123")
    r2 = analyzer.analyze("same_ref_123")
    assert r1.authenticity_score == r2.authenticity_score
    assert r1.manipulation_score == r2.manipulation_score


def test_mock_analyzer_different_refs_can_differ():
    analyzer = MockImageEvidenceAnalyzer()
    r1 = analyzer.analyze("ref_a")
    r2 = analyzer.analyze("ref_b")
    assert (r1.authenticity_score, r1.manipulation_score, r1.duplicate_score) != (
        r2.authenticity_score, r2.manipulation_score, r2.duplicate_score
    )


def test_overall_signal_is_one_of_the_three_labels():
    analyzer = MockImageEvidenceAnalyzer()
    for i in range(30):
        result = analyzer.analyze(f"ref_{i}")
        assert result.overall_signal() in ("NORMAL", "SUSPICIOUS", "INCONCLUSIVE")


def test_disclaimer_never_claims_fraud_proof():
    assert "do not prove fraud" in IMAGE_DISCLAIMER.lower()


def test_analyzer_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        ImageEvidenceAnalyzer()


def test_no_accuracy_claim_anywhere_in_result():
    analyzer = MockImageEvidenceAnalyzer()
    result = analyzer.analyze("ref_x")
    text_blob = " ".join(result.signals).lower()
    assert "% accurate" not in text_blob
    assert "95%" not in text_blob
