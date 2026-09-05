import numpy as np
import pandas as pd
from src.diagnosis.statistical import diagnose_dimension, run_full_diagnosis, top_finding


def _biased_df(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    fulfilment = rng.choice(["platform", "third_party"], size=n, p=[0.7, 0.3])
    base_p = np.where(fulfilment == "third_party", 0.30, 0.10)
    return_event = rng.binomial(1, base_p).astype(bool)
    return pd.DataFrame({"fulfilment_method": fulfilment, "return_event": return_event})


def test_diagnosis_detects_planted_effect():
    df = _biased_df()
    findings = diagnose_dimension(df, "fulfilment_method")
    third_party = [f for f in findings if f.segment == "third_party"][0]
    assert third_party.relative_risk > 1.5
    assert third_party.statistically_supported
    assert third_party.practically_significant


def test_diagnosis_ignores_tiny_segments():
    df = pd.DataFrame({
        "fulfilment_method": ["rare"] * 5 + ["common"] * 500,
        "return_event": [True] * 5 + [False] * 500,
    })
    findings = diagnose_dimension(df, "fulfilment_method", min_segment_size=30)
    segments = [f.segment for f in findings]
    assert "rare" not in segments


def test_plain_english_never_uses_causal_language():
    df = _biased_df()
    findings = diagnose_dimension(df, "fulfilment_method")
    for f in findings:
        text = f.plain_english().lower()
        assert "causes" not in text
        assert "caused by" not in text
        assert "association" in text or "observed" in text


def test_top_finding_requires_both_statistical_and_practical_support():
    df = pd.DataFrame({
        "fulfilment_method": ["a"] * 100 + ["b"] * 100,
        "return_event": [False] * 100 + [False] * 95 + [True] * 5,
    })
    results = run_full_diagnosis(df)
    finding = top_finding(results)
    # tiny relative effect shouldn't surface as the top actionable finding
    if finding is not None:
        assert finding.practically_significant and finding.statistically_supported


def test_no_finding_when_no_dimensions_present():
    df = pd.DataFrame({"return_event": [True, False, True]})
    results = run_full_diagnosis(df)
    assert all(len(v) == 0 for v in results.values())
