import hashlib
import hmac
import json

from src.api.app import app, auth_service


def _client():
    app.config["TESTING"] = True
    return app.test_client()


def _register(email="merchant@shop.com", password="pw123456", organization_id="org_api_test"):
    try:
        auth_service.register(email=email, password=password, organization_id=organization_id)
    except ValueError:
        pass  # already registered by an earlier test in this process


def test_login_success_and_failure():
    _register()
    client = _client()
    ok = client.post("/auth/login", json={"email": "merchant@shop.com", "password": "pw123456"})
    assert ok.status_code == 200
    assert "token" in ok.get_json()

    bad = client.post("/auth/login", json={"email": "merchant@shop.com", "password": "wrong"})
    assert bad.status_code == 401


def test_protected_route_requires_token():
    client = _client()
    resp = client.get("/risk/latest")
    assert resp.status_code == 401


def test_protected_route_with_valid_token_returns_result():
    _register(email="merchant2@shop.com", organization_id="org_api_test_2")
    client = _client()
    login = client.post("/auth/login", json={"email": "merchant2@shop.com", "password": "pw123456"})
    token = login.get_json()["token"]
    resp = client.get("/risk/latest", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.get_json()["organization_id"] == "org_api_test_2"


def test_webhook_rejects_invalid_signature():
    from src.api.app import webhook_registry
    webhook_registry.register("ds_api_1", "org_api_test", "whsec_api_test")
    client = _client()
    payload = {"id": "evt_api_1", "event": "refund.processed"}
    resp = client.post(
        "/webhooks/razorpay/ds_api_1",
        data=json.dumps(payload),
        headers={"X-Organization-Id": "attacker_org",
                 "X-Razorpay-Signature": "bad",
                 "Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_webhook_does_not_trust_client_org_header():
    from src.api.app import webhook_registry
    webhook_registry.register("ds_api_2", "real_org", "whsec_api_test")
    client = _client()
    payload = {"id": "evt_api_2", "event": "payment.captured"}
    body = json.dumps(payload).encode()
    sig = hmac.new(b"whsec_api_test", body, hashlib.sha256).hexdigest()
    resp = client.post(
        "/webhooks/razorpay/ds_api_2",
        data=body,
        headers={"X-Organization-Id": "attacker_org",
                 "X-Razorpay-Signature": sig,
                 "Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.get_json()["organization_id"] == "real_org"


def test_webhook_unknown_data_source_rejected():
    client = _client()
    resp = client.post("/webhooks/razorpay/unknown", data=b"{}",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


def test_legacy_webhook_route_rejects_client_tenant_identity():
    client = _client()
    resp = client.post("/webhooks/razorpay", data=b"{}",
                       headers={"X-Organization-Id": "attacker_org",
                                "Content-Type": "application/json"})
    assert resp.status_code == 410
