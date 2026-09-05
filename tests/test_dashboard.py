"""
Tests for app/ui/generate_dashboard.py.

The dashboard generator must render both:
  - the original benchmark report shape (evaluation/reports/full_report.json)
  - the newer per-organization pipeline result shape (src.pipeline result)
without hard-coding synthetic values, and it must render the DATA SOURCES
(G1) and RETURN CLAIMS / EVIDENCE (G4/G5) sections the spec asked for.
"""
import json
from pathlib import Path

from app.ui.generate_dashboard import (
    render,
    render_data_sources,
    render_claim_card,
    render_claims_section,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _new_shape_report(is_synthetic_demo=False):
    return {
        "status": "ok",
        "dataset_label": "Acme Corp",
        "connector_type": "razorpay",
        "is_synthetic_demo": is_synthetic_demo,
        "data_readiness": {"feature_contract": [], "granularity": "order", "n_rows_raw": 100, "n_orders": 100},
        "model": {
            "model_name": "logistic_regression",
            "threshold": 0.5,
            "feature_count": 10,
            "split": {"train_n": 60, "val_n": 20, "test_n": 20},
        },
        "test_evaluation": {
            "precision": 0.5, "recall": 0.4, "f1": 0.44, "fpr": 0.1, "fnr": 0.6,
            "n_total": 20, "n_positive": 5,
            "counts": {"tp": 2, "fp": 3, "fn": 3, "tn": 12},
        },
        "financial_exposure": {
            "n_high_risk_orders": 5, "n_total_orders": 20, "pct_orders_high_risk": 0.25,
            "predicted_return_exposure": 1000, "observed_historical_return_value": 900,
        },
        "diagnosis": {}, "top_finding": None, "recommendation": None, "verification": None,
    }


def test_render_handles_new_pipeline_shape_without_crashing():
    # This is the shape src.pipeline.run_organization_pipeline actually
    # returns -- no top-level feature_importance, no model.split.train_end
    # /val_end/test_end/random_seed. Must not raise.
    html = render(_new_shape_report())
    assert "<html" in html


def test_render_still_handles_original_benchmark_shape():
    report = json.loads((REPO_ROOT / "evaluation/reports/full_report.json").read_text())
    html = render(report)
    assert "<html" in html


def test_confusion_matrix_labelled_merchant_held_out_when_not_demo():
    html = render(_new_shape_report(is_synthetic_demo=False))
    assert "MERCHANT HELD-OUT TEST" in html


def test_confusion_matrix_labelled_demo_synthetic_when_demo():
    html = render(_new_shape_report(is_synthetic_demo=True))
    assert "DEMO / SYNTHETIC" in html


def test_confusion_matrix_values_are_dynamic_not_hardcoded():
    report = _new_shape_report()
    report["test_evaluation"]["counts"] = {"tp": 111, "fp": 22, "fn": 33, "tn": 444}
    html = render(report)
    assert "111" in html and "22" in html and "33" in html and "444" in html


def test_data_sources_section_marks_active_connector():
    html = render_data_sources(active_connector_type="razorpay")
    assert "Razorpay" in html
    assert "CSV" in html
    assert "Fallback / manual import" in html


def test_data_sources_rendered_only_when_requested():
    report = _new_shape_report()
    assert "DATA SOURCES" not in render(report)
    assert "DATA SOURCES" in render(report, show_data_sources=True)


def test_claim_card_never_declares_fraud():
    evidence = {
        "claim_id": "C1", "status": "SUSPICIOUS", "confidence": "high",
        "normalized_reason": "DAMAGED", "raw_reason": "arrived damaged",
        "supporting_signals": [], "contradictory_signals": ["Elevated return history."],
        "missing_evidence": [], "human_review_recommended": True, "image_signal": None,
        "disclaimer": "These are evidence signals for human review. They do not prove fraud, customer intent, or responsibility for the damage.",
    }
    html = render_claim_card(evidence)
    assert "fraud confirmed" not in html.lower()
    assert "customer is fraudulent" not in html.lower()
    assert "do not prove fraud" in html  # the mandated disclaimer is present
    assert "SUSPICIOUS" in html


def test_claims_section_counts_need_review():
    claims = [
        {"claim_id": "C1", "status": "SUPPORTED", "disclaimer": "d", "supporting_signals": [],
         "contradictory_signals": [], "missing_evidence": [], "raw_reason": "r"},
        {"claim_id": "C2", "status": "NEEDS_REVIEW", "disclaimer": "d", "supporting_signals": [],
         "contradictory_signals": [], "missing_evidence": [], "raw_reason": "r"},
    ]
    html = render_claims_section(claims)
    assert "1 claim need review" in html or "1 claim" in html


def test_claims_section_empty_when_no_claims():
    assert render_claims_section([]) == ""


def test_render_includes_claims_section_when_provided():
    report = _new_shape_report()
    claims = [{"claim_id": "C1", "status": "NEEDS_REVIEW", "disclaimer": "d",
               "supporting_signals": [], "contradictory_signals": [], "missing_evidence": [],
               "raw_reason": "arrived damaged"}]
    html = render(report, claims=claims)
    assert "RETURN CLAIMS" in html
    assert "arrived damaged" in html
