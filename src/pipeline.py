"""
Orchestration layer: wires authentication, tenant isolation, a
connector, the data-readiness pipeline, model scope decision,
training/evaluation, exposure, diagnosis, recommendation,
verification, drift monitoring, and the model registry into the two
flows the UI actually calls:

    run_organization_pipeline(...)  -- PREDICT/ABSTAIN -> EXPLAIN ->
                                        QUANTIFY -> ACT -> VERIFY
    review_return_claim(...)        -- the secondary evidence layer

Every call here requires a valid session token and every persisted
result is written through TenantScopedStore, so cross-tenant access
is denied by the data layer itself, not by caller discipline.

This module intentionally contains no autonomous action: it only
predicts, explains, quantifies, recommends, and (for demo purposes)
simulates verification. It never refunds, blocks a customer, cancels
an order, or changes courier/payment configuration.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.auth.service import AuthService
from src.tenancy.store import TenantScopedStore, TenantIsolationError
from src.quality.readiness import run_data_readiness_pipeline
from src.quality.lifecycle import assign_label_state, usable_for_supervision
from src.quality.drift import check_drift
from src.model.registry import decide_model_scope, ModelRegistry
from src.model.train import train_return_risk_model
from src.connectors.base import MerchantDataConnector
from evaluation.metrics.classification import evaluate
from src.exposure.financial import compute_exposure
from src.diagnosis.statistical import run_full_diagnosis, top_finding
from src.recommendation.engine import recommend
from src.verification.simulate import simulate_intervention
from src.claims.model import ReturnClaim
from src.claims.evidence import aggregate_evidence
from src.evidence.image_analyzer import ImageEvidenceAnalyzer


class PipelineAuthError(Exception):
    """Raised whenever a caller is not authenticated, or is
    authenticated but not authorized for the record/organization it
    is asking about. Never silently swallowed -- callers must treat
    this as a hard failure, not fall back to a default organization."""


def _require_organization(token: Optional[str], auth: AuthService) -> str:
    result = auth.require_session(token)
    if not result.success:
        raise PipelineAuthError(result.error or "Authentication required.")
    return result.session.organization_id


_ORDER_SIGNAL_DIMENSIONS = ["fulfilment_method", "shipping_service", "category", "region"]


def _build_order_risk_table(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    p_test,
    y_pred_test,
    threshold: float,
    diagnosis_results: dict,
) -> list[dict]:
    """Package the model's already-computed per-order test predictions
    into a UI-friendly list (order_id, amount, risk score/level, and
    the canonical field values already on the order -- never a new
    calculation, never a claim of feature attribution/SHAP that the
    model does not actually produce).

    Risk level bands are derived from the single frozen threshold the
    model already uses: HIGH at/above threshold, MEDIUM in the band
    just below it, LOW further below. No separate threshold is
    trained or tuned here.
    """
    medium_floor = max(0.0, float(threshold) - 0.20)

    # Segments flagged by the diagnosis engine, used only to describe
    # (never to compute) which *known* signal an order's own field
    # values overlap with -- e.g. an order is in a segment the
    # diagnosis engine already reported as elevated-risk.
    flagged_segments: dict[str, set] = {}
    for dimension, findings in (diagnosis_results or {}).items():
        flagged_segments[dimension] = {
            f.segment for f in findings if getattr(f, "practically_significant", False)
        }

    rows = []
    test_df = test_df.reset_index(drop=True)
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        score = float(p_test[i])
        if score >= threshold:
            level = "high"
        elif score >= medium_floor:
            level = "medium"
        else:
            level = "low"

        signals = []
        for dim in _ORDER_SIGNAL_DIMENSIONS:
            if dim in test_df.columns:
                value = row.get(dim)
                if pd.notna(value):
                    is_flagged = str(value) in flagged_segments.get(dim, set())
                    signals.append({"dimension": dim, "value": str(value), "flagged": is_flagged})

        flagged_labels = [s["value"] for s in signals if s["flagged"]]
        if flagged_labels:
            key_reason = " + ".join(flagged_labels[:2])
        else:
            key_reason = "Elevated model-predicted risk"

        order_date_value = row.get("order_date") if "order_date" in test_df.columns else None
        rows.append({
            "order_id": str(row.get("order_id", f"row_{i}")),
            "order_date": str(order_date_value) if pd.notna(order_date_value) else None,
            "amount": float(row.get("amount", 0.0)) if pd.notna(row.get("amount", None)) else None,
            "risk_score": round(score, 4),
            "risk_level": level,
            "predicted_high_risk": bool(y_pred_test[i]),
            "actual_return": bool(y_test.values[i]) if y_test is not None else None,
            "signals": signals,
            "key_reason": key_reason,
        })

    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    return rows


def run_organization_pipeline(
    token: Optional[str],
    auth: AuthService,
    connector: MerchantDataConnector,
    tenant_store: TenantScopedStore,
    registry: ModelRegistry,
    dataset_label: str = "merchant data",
) -> dict:
    """
    RAW MERCHANT DATA -> SCHEMA DETECTION -> DATA QUALITY CHECK ->
    CANONICAL MAPPING -> FEATURE AVAILABILITY -> MODEL READINESS ->
    PREDICT / ABSTAIN -> EXPLAIN -> QUANTIFY -> ACT -> VERIFY.

    Always returns a dict (never raises) once authentication has
    passed, and always persists its result under
    ("risk_results", organization_id, "latest") so the dashboard has
    something to read even when the model abstained.
    """
    organization_id = _require_organization(token, auth)

    raw_df = connector.fetch_historical_data()
    order_df, readiness = run_data_readiness_pipeline(raw_df)
    decision = decide_model_scope(readiness)

    if decision.status == "INSUFFICIENT_DATA":
        result = {
            "status": "abstained",
            "organization_id": organization_id,
            "dataset_label": dataset_label,
            "connector_type": getattr(connector, "connector_type", "unknown"),
            "data_readiness": readiness.to_dict(),
            "model_decision": {"status": decision.status, "scope": decision.scope, "reason": decision.reason},
            "message": "MODEL STATUS: NOT READY. " + decision.reason,
        }
        tenant_store.put("risk_results", organization_id, "latest", result)
        tenant_store.put("risk_orders", organization_id, "latest", [])
        return result

    # Only finalized labels (RETURNED / NO_RETURN) may be used for
    # supervised training or evaluation -- PENDING orders are excluded.
    state = assign_label_state(order_df)
    trainable_df = usable_for_supervision(order_df, state)

    is_synthetic_demo = getattr(connector, "connector_type", "") == "mock"

    bundle = train_return_risk_model(trainable_df)

    X_test, y_test, test_df = bundle._test_X, bundle._test_y, bundle._test_df
    p_test = bundle.predict_proba(X_test)
    y_pred_test = (p_test >= bundle.threshold).astype(int)

    # Registered before evaluate() purely so its version label can be
    # attached to the evaluation record below (spec: "Model version:
    # [version]" in the dataset-context UI). Registration itself is
    # unaffected by, and does not affect, the model/features/threshold.
    model_version = registry.register(
        organization_id=organization_id,
        model_scope=decision.scope,
        model_name=bundle.model_name,
        n_train_rows=len(bundle._train_df),
        feature_names=bundle.feature_names,
        threshold=bundle.threshold,
        is_synthetic_demo=is_synthetic_demo,
    )

    metrics = evaluate(
        y_test.values,
        p_test,
        bundle.threshold,
        dataset_type="synthetic_holdout" if is_synthetic_demo else "merchant_holdout",
        model_version=model_version.label(),
    )
    exposure = compute_exposure(test_df["amount"], y_test, y_pred_test)

    diagnosis_results = run_full_diagnosis(test_df.assign(return_event=y_test.values))
    best_finding = top_finding(diagnosis_results)
    recommendation = recommend(best_finding)

    order_risk_table = _build_order_risk_table(
        test_df=test_df,
        y_test=y_test,
        p_test=p_test,
        y_pred_test=y_pred_test,
        threshold=bundle.threshold,
        diagnosis_results=diagnosis_results,
    )
    tenant_store.put("risk_orders", organization_id, "latest", order_risk_table)

    verification = None
    if best_finding is not None:
        seg_mask = test_df[best_finding.dimension].astype(str) == best_finding.segment
        seg_n = int(seg_mask.sum())
        seg_returns = int(y_test.values[seg_mask.values].sum())
        if seg_n >= 20:
            verification = simulate_intervention(before_rate=seg_returns / seg_n, before_n=seg_n)

    drift_report = check_drift(bundle._train_df, test_df.assign(return_event=y_test.values))

    result = {
        "status": "ok",
        "organization_id": organization_id,
        "dataset_label": dataset_label,
        "connector_type": getattr(connector, "connector_type", "unknown"),
        "is_synthetic_demo": is_synthetic_demo,
        "data_readiness": readiness.to_dict(),
        "model_decision": {"status": decision.status, "scope": decision.scope, "reason": decision.reason},
        "model": {
            "model_name": bundle.model_name,
            "threshold": bundle.threshold,
            "feature_count": len(bundle.feature_names),
            "split": {
                "train_n": len(bundle._train_df),
                "val_n": len(bundle._val_df),
                "test_n": len(test_df),
            },
        },
        "test_evaluation": metrics.to_dict(),
        "financial_exposure": exposure.to_dict(),
        "diagnosis": {dim: [f.__dict__ for f in findings[:5]] for dim, findings in diagnosis_results.items()},
        "top_finding": best_finding.__dict__ if best_finding else None,
        "recommendation": recommendation.__dict__ if recommendation else None,
        "verification": verification.__dict__ if verification else None,
        "drift": drift_report.to_dict(),
        "model_version": {
            "organization_id": model_version.organization_id,
            "version": model_version.version,
            "label": model_version.label(),
            "model_scope": model_version.model_scope,
            "trained_at": model_version.trained_at,
            "is_synthetic_demo": model_version.is_synthetic_demo,
        },
    }

    tenant_store.put("risk_results", organization_id, "latest", result)
    return result


def review_return_claim(
    token: Optional[str],
    auth: AuthService,
    claim: ReturnClaim,
    tenant_store: TenantScopedStore,
    image_analyzer: Optional[ImageEvidenceAnalyzer] = None,
) -> dict:
    """
    Run the secondary return-claim evidence layer for one claim.
    Never declares a fraud verdict -- see src.claims.evidence.
    """
    organization_id = _require_organization(token, auth)

    if claim.organization_id != organization_id:
        raise PipelineAuthError("This return claim does not belong to the authenticated organization.")

    image_result = None
    if image_analyzer is not None and claim.image_references:
        image_result = image_analyzer.analyze(claim.image_references[0])

    evidence = aggregate_evidence(claim, image_result=image_result)
    result = evidence.to_dict()

    tenant_store.put("claims", organization_id, claim.claim_id, result)
    return result
