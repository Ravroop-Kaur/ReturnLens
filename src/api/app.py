"""
Minimal Flask application wiring auth, the pipeline, and the Razorpay
webhook endpoint together.

Deliberately small (PART M: "do not overbuild") -- this is enough to
demonstrate login, protected routes, tenant-scoped results, and a
webhook endpoint for the hackathon MVP, not a production API gateway.

Routes:
    POST /auth/login            -> {token} or 401
    POST /auth/logout           -> 200
    GET  /risk/latest           -> tenant-scoped latest risk result (requires Bearer token)
    POST /webhooks/razorpay     -> Razorpay webhook receiver

Run with: FLASK_APP=src.api.app flask run   (or `python -m src.api.app`)
"""

from __future__ import annotations

import json

from flask import Flask, request, jsonify

from src.auth.service import AuthService
from src.tenancy.store import TenantScopedStore, TenantIsolationError
from src.model.registry import ModelRegistry
from src.connectors.mock import MockMerchantConnector
from src.connectors.razorpay_connector import RazorpayConnector
from src.connectors.csv_connector import CSVMerchantConnector
from src.integrations.razorpay.config import RazorpayConfig
from src.pipeline import run_organization_pipeline, review_return_claim, PipelineAuthError
from src.integrations.razorpay.webhook import RazorpayWebhookHandler, InvalidWebhookSignature, WebhookIntegrationRegistry
from src.claims.model import ReturnClaim
from src.evidence.image_analyzer import MockImageEvidenceAnalyzer

app = Flask(__name__)


def _to_jsonable(obj):
    """Recursively convert numpy/pandas scalar types to native Python
    types. Several downstream modules (statistical diagnosis,
    verification, drift) intentionally use numpy/scipy directly for
    correctness and are left untouched -- this conversion happens only
    at the API boundary, where JSON serialization actually requires
    plain Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    return obj


# Process-lifetime singletons. A real deployment would back these
# with a database; the in-memory implementations are explicitly
# documented as MVP/demo (see each module's own docstring).
auth_service = AuthService()
tenant_store = TenantScopedStore()
model_registry = ModelRegistry()
webhook_handler = RazorpayWebhookHandler()
webhook_registry = WebhookIntegrationRegistry()
razorpay_connections = {}  # MVP: in-memory integration config; secrets never returned to clients


def _bearer_token() -> str | None:
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[len("Bearer "):]
    return None




def _require_auth_session():
    token = _bearer_token()
    result = auth_service.require_session(token)
    if not result.success:
        return None, jsonify({"error": result.error}), 401
    return result.session, None, None


@app.get("/")
def dashboard():
    from pathlib import Path
    html_path = Path(__file__).resolve().parents[2] / "app" / "ui" / "dashboard.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/integrations/razorpay/test-connection")
def razorpay_test_connection():
    """Small real Razorpay Test Mode flow: validate credentials by making
    a read-only API request, then register the data source for webhooks.
    No live-mode action is performed."""
    session, error_response, status = _require_auth_session()
    if error_response is not None:
        return error_response, status
    body = request.get_json(force=True, silent=True) or {}
    key_id = str(body.get("key_id", "")).strip()
    key_secret = str(body.get("key_secret", ""))
    webhook_secret = str(body.get("webhook_secret", "")).strip()
    data_source_id = str(body.get("data_source_id", "")).strip() or f"rzp_{session.organization_id}"
    if not key_id or not key_secret:
        return jsonify({"error": "Test Mode Key ID and Key Secret are required."}), 400
    if not key_id.startswith("rzp_test_"):
        return jsonify({"error": "For this MVP flow, use a Razorpay Test Mode key (rzp_test_...)."}), 400
    config = RazorpayConfig(key_id=key_id, key_secret=key_secret, webhook_secret=webhook_secret)
    connector = RazorpayConnector(config=config)
    health = connector.test_connection()
    if not health.healthy:
        return jsonify({"connected": False, "mode": "TEST", "detail": health.detail}), 400
    razorpay_connections[data_source_id] = {"organization_id": session.organization_id, "config": config}
    if webhook_secret:
        webhook_registry.register(data_source_id, session.organization_id, webhook_secret)
    sample = connector.fetch_historical_data().head(5)
    return jsonify({
        "connected": True,
        "mode": "TEST",
        "data_source_id": data_source_id,
        "detail": health.detail,
        "payments_preview_count": len(sample),
        "webhook_endpoint": f"/webhooks/razorpay/{data_source_id}",
        "return_labels": "NOT PROVIDED BY RAZORPAY PAYMENTS API",
        "refund_signal": "AVAILABLE",
    })


@app.post("/integrations/razorpay/import")
def razorpay_import():
    session, error_response, status = _require_auth_session()
    if error_response is not None:
        return error_response, status
    body = request.get_json(force=True, silent=True) or {}
    data_source_id = str(body.get("data_source_id", "")).strip() or f"rzp_{session.organization_id}"
    connection = razorpay_connections.get(data_source_id)
    if not connection or connection["organization_id"] != session.organization_id:
        return jsonify({"error": "Razorpay Test Mode data source is not connected for this organization."}), 404
    connector = RazorpayConnector(config=connection["config"])
    df = connector.fetch_historical_data()
    tenant_store.put("data_sources", session.organization_id, data_source_id, {
        "type": "razorpay", "mode": "TEST", "rows": len(df),
        "return_labels_available": bool("return_event" in df.columns and df["return_event"].notna().any()),
        "refund_signals_available": bool("refund_event" in df.columns),
    })
    return jsonify({
        "status": "imported", "mode": "TEST", "data_source_id": data_source_id,
        "rows": len(df), "return_labels_available": False, "refund_signals_available": True,
        "next_step": "Connect an OMS/returns source for actual return outcomes before training the return-risk model."
    })

@app.post("/data-sources/csv/import")
def csv_import():
    """CSV = fallback / manual import path (PART G1), never the primary
    product story. A successfully imported CSV is run through the same
    readiness -> model pipeline as every other connector, via
    CSVMerchantConnector, so results are never a second, hand-rolled
    calculation."""
    session, error_response, status = _require_auth_session()
    if error_response is not None:
        return error_response, status
    token = _bearer_token()

    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "No CSV file was uploaded."}), 400

    import tempfile, os
    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    try:
        os.close(fd)
        uploaded.save(tmp_path)
        connector = CSVMerchantConnector(csv_path=tmp_path)
        health = connector.test_connection()
        if not health.healthy:
            return jsonify({"error": health.detail}), 400
        try:
            result = run_organization_pipeline(
                token=token, auth=auth_service, connector=connector,
                tenant_store=tenant_store, registry=model_registry,
                dataset_label=uploaded.filename,
            )
        except PipelineAuthError as exc:
            return jsonify({"error": str(exc)}), 401
        tenant_store.put("data_sources", session.organization_id, "csv", {
            "type": "csv", "filename": uploaded.filename,
        })
        return jsonify(_to_jsonable({
            "status": "imported",
            "filename": uploaded.filename,
            "model_decision": result.get("model_decision"),
            "data_readiness": result.get("data_readiness"),
        }))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/auth/register")
def register():
    """Lightweight merchant signup for the hackathon demo.
    Production identity persistence can replace this service later."""
    body = request.get_json(force=True, silent=True) or {}
    email = str(body.get("email", "")).strip()
    password = str(body.get("password", ""))
    organization_id = str(body.get("organization_id", "")).strip()
    if not email or len(password) < 8 or not organization_id:
        return jsonify({"error": "email, organization_id and a password of at least 8 characters are required."}), 400
    try:
        account = auth_service.register(email, password, organization_id, body.get("display_name", ""))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"user_id": account.user_id, "organization_id": account.organization_id}), 201


@app.post("/auth/login")
def login():
    body = request.get_json(force=True, silent=True) or {}
    result = auth_service.login(body.get("email", ""), body.get("password", ""))
    if not result.success:
        return jsonify({"error": result.error}), 401
    return jsonify({"token": result.session.token, "organization_id": result.session.organization_id})


@app.post("/auth/logout")
def logout():
    token = _bearer_token()
    auth_service.logout(token)
    return jsonify({"status": "logged_out"})


@app.get("/risk/latest")
def risk_latest():
    token = _bearer_token()
    auth_result = auth_service.require_session(token)
    if not auth_result.success:
        return jsonify({"error": auth_result.error}), 401

    organization_id = auth_result.session.organization_id
    try:
        result = tenant_store.get("risk_results", organization_id, "latest")
    except TenantIsolationError as exc:
        return jsonify({"error": str(exc)}), 403

    if result is None:
        # Nothing computed yet for this organization -- run the demo
        # pipeline against the mock connector so the dashboard has
        # something to show. A real deployment would trigger this from
        # a background job when a real connector is configured instead.
        try:
            result = run_organization_pipeline(
                token=token,
                auth=auth_service,
                connector=MockMerchantConnector(),
                tenant_store=tenant_store,
                registry=model_registry,
                dataset_label="synthetic demo data",
            )
        except PipelineAuthError as exc:
            return jsonify({"error": str(exc)}), 401

    return jsonify(_to_jsonable(result))


@app.get("/risk/orders")
def risk_orders():
    """Tenant-scoped per-order risk table already computed by the
    pipeline (src.pipeline._build_order_risk_table). Never recomputes
    scores in the API layer -- reads the persisted result only."""
    token = _bearer_token()
    auth_result = auth_service.require_session(token)
    if not auth_result.success:
        return jsonify({"error": auth_result.error}), 401
    organization_id = auth_result.session.organization_id
    try:
        orders = tenant_store.get("risk_orders", organization_id, "latest")
    except TenantIsolationError as exc:
        return jsonify({"error": str(exc)}), 403
    if orders is None:
        # Ensure the pipeline has run at least once (mirrors /risk/latest).
        try:
            run_organization_pipeline(
                token=token, auth=auth_service, connector=MockMerchantConnector(),
                tenant_store=tenant_store, registry=model_registry, dataset_label="synthetic demo data",
            )
        except PipelineAuthError as exc:
            return jsonify({"error": str(exc)}), 401
        orders = tenant_store.get("risk_orders", organization_id, "latest") or []
    n_high = sum(1 for o in orders if o.get("risk_level") == "high")
    n_medium = sum(1 for o in orders if o.get("risk_level") == "medium")
    n_low = sum(1 for o in orders if o.get("risk_level") == "low")
    return jsonify(_to_jsonable({
        "orders": orders,
        "n_high": n_high,
        "n_medium": n_medium,
        "n_low": n_low,
    }))


_DEMO_CLAIM_SEEDS = [
    {"reason": "Item arrived damaged", "delivery_status": "delivered", "prior_returns": 1, "has_image": True},
    {"reason": "Wrong item received", "delivery_status": "delivered", "prior_returns": 0, "has_image": False},
    {"reason": "Doesn't fit / wrong size", "delivery_status": "delivered", "prior_returns": 4, "has_image": False},
    {"reason": "Changed my mind", "delivery_status": "delivered", "prior_returns": 5, "has_image": False},
    {"reason": "Package never arrived", "delivery_status": "lost", "prior_returns": 0, "has_image": False},
]


def _seed_demo_claims(token: str, organization_id: str) -> list[dict]:
    """Runs the existing evidence-aggregation engine (src.claims.evidence,
    via src.pipeline.review_return_claim) against a handful of
    representative claims built from the organization's own order
    table, so the Claims screen has something real to show in demo
    mode. Every status produced is exactly what the evidence engine
    already returns -- nothing here is fabricated beyond the input
    claim text. Returns [] (never raises) if there is no order table
    yet to build claims from."""
    orders = tenant_store.get("risk_orders", organization_id, "latest") or []
    if not orders:
        return []

    analyzer = MockImageEvidenceAnalyzer()
    results = []
    for i, seed in enumerate(_DEMO_CLAIM_SEEDS):
        if i >= len(orders):
            break
        order = orders[i]
        claim = ReturnClaim(
            claim_id=f"CLM-{organization_id}-{i+1:04d}",
            organization_id=organization_id,
            order_id=order["order_id"],
            claimed_reason=seed["reason"],
            claim_timestamp=order.get("order_date"),
            image_references=["demo-image-1"] if seed["has_image"] else [],
            order_amount=order.get("amount"),
            order_date=order.get("order_date"),
            delivery_status=seed["delivery_status"],
            customer_prior_return_count=seed["prior_returns"],
        )
        result = review_return_claim(
            token=token, auth=auth_service, claim=claim,
            tenant_store=tenant_store, image_analyzer=analyzer,
        )
        results.append(result)
    return results


@app.get("/claims")
def list_claims():
    """All return claims already reviewed for this organization. This
    never invents a fraud verdict -- see src.claims.evidence for the
    only statuses ever produced.

    In demo mode (synthetic connector, no claims filed yet), this
    seeds a handful of representative demo claims the first time the
    Claims screen is opened, so it never shows an empty page purely
    because nothing has been manually seeded -- exactly the same
    evidence engine output a real filed claim would get."""
    token = _bearer_token()
    session, error_response, status = _require_auth_session()
    if error_response is not None:
        return error_response, status
    organization_id = session.organization_id
    claim_ids = tenant_store.list_kind("claims", organization_id)
    claims = [tenant_store.get("claims", organization_id, cid) for cid in claim_ids]
    claims = [c for c in claims if c is not None]

    if not claims:
        risk_result = tenant_store.get("risk_results", organization_id, "latest")
        if risk_result and risk_result.get("is_synthetic_demo"):
            claims = _seed_demo_claims(token, organization_id)

    needs_review = sum(1 for c in claims if c.get("status") != "SUPPORTED")
    return jsonify(_to_jsonable({
        "claims": claims,
        "n_total": len(claims),
        "n_needs_review": needs_review,
        "needs_review_count": needs_review,
    }))


@app.post("/claims/seed-demo")
def seed_demo_claims():
    """Manual trigger for the same demo-claim seeding /claims does
    automatically on first load. Kept for direct/manual use."""
    token = _bearer_token()
    session, error_response, status = _require_auth_session()
    if error_response is not None:
        return error_response, status
    organization_id = session.organization_id
    results = _seed_demo_claims(token, organization_id)
    if not results:
        return jsonify({"error": "No orders available yet. Load Risk analysis first."}), 400
    return jsonify(_to_jsonable({"claims": results}))


@app.get("/data-sources")
def data_sources_status():
    """Summarizes the three data-source stories the product tells
    (Razorpay payments/refunds, merchant returns/OMS, CSV fallback) so
    the Data Sources screen never has to infer this from other
    endpoints' shapes."""
    session, error_response, status = _require_auth_session()
    if error_response is not None:
        return error_response, status
    organization_id = session.organization_id

    razorpay_connected = any(
        c["organization_id"] == organization_id for c in razorpay_connections.values()
    )
    risk_result = tenant_store.get("risk_results", organization_id, "latest")
    oms_connected = bool(risk_result and not risk_result.get("is_synthetic_demo", True))
    readiness = risk_result.get("data_readiness") if risk_result else None

    return jsonify(_to_jsonable({
        "sources": {
            "razorpay": {
                "connected": razorpay_connected,
                "mode": "TEST",
                "capabilities": {"payments": True, "refunds": True, "return_labels": False},
            },
            "merchant_oms": {
                "connected": oms_connected,
                "note": (
                    "Return outcomes from your OMS are used to train and evaluate the return-risk model."
                    if oms_connected else
                    "Not connected yet. Currently using CSV import / synthetic demo data for return outcomes."
                ),
            },
            "csv": {
                "available": True,
                "role": "Historical import / fallback",
                "last_import": tenant_store.get("data_sources", organization_id, "csv"),
            },
        },
        "data_readiness": readiness,
        "model_decision": risk_result.get("model_decision") if risk_result else None,
        "is_synthetic_demo": risk_result.get("is_synthetic_demo") if risk_result else None,
    }))


@app.post("/webhooks/razorpay/<data_source_id>")
def razorpay_webhook(data_source_id):
    """Receive a webhook for a configured merchant data source.

    Tenant identity is derived server-side from the configured
    data_source_id. An arbitrary X-Organization-Id header is never
    trusted. In production the registry is persistent integration
    configuration; for the MVP it is in-memory.
    """
    raw_body = request.get_data()
    signature = request.headers.get("X-Razorpay-Signature", "")
    try:
        payload_json = json.loads(raw_body or b"{}")
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON payload."}), 400

    try:
        organization_id, webhook_secret = webhook_registry.resolve(data_source_id)
        event = webhook_handler.handle(
            organization_id=organization_id,
            payload_body=raw_body,
            received_signature=signature,
            payload_json=payload_json,
            webhook_secret=webhook_secret,
        )
    except InvalidWebhookSignature as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "status": event.processing_status,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "organization_id": event.organization_id,
    })


@app.post("/webhooks/razorpay")
def razorpay_webhook_legacy():
    """Legacy route intentionally refuses client-supplied tenant IDs.

    Integrations must use /webhooks/razorpay/<data_source_id>.
    """
    return jsonify({
        "error": "Tenant identity must come from a configured data source. "
                 "Use /webhooks/razorpay/<data_source_id>."
    }), 410


if __name__ == "__main__":
    app.run(debug=True)
