import hashlib
import hmac
import json

import pytest

from src.integrations.razorpay.config import RazorpayConfig
from src.integrations.razorpay.client import RazorpayClient
from src.integrations.razorpay.mapper import map_payments_to_canonical, refunded_ids_from_refunds_page
from src.integrations.razorpay.webhook import (
    RazorpayWebhookHandler, InvalidWebhookSignature, verify_signature,
)


def test_config_never_hardcodes_secrets(monkeypatch):
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_WEBHOOK_SECRET", raising=False)
    config = RazorpayConfig.from_env()
    assert not config.is_configured


def test_config_test_mode_detection(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_abc123")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
    config = RazorpayConfig.from_env()
    assert config.is_configured
    assert config.is_test_mode


def test_client_falls_back_to_demo_mode_without_credentials(monkeypatch):
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    client = RazorpayClient(config=RazorpayConfig.from_env())
    assert client.is_demo
    page = client.fetch_payments(count=5)
    assert page["is_demo"] is True
    assert len(page["items"]) == 5


def test_mapper_maps_payments_to_canonical_schema():
    client = RazorpayClient(force_mock=True)
    page = client.fetch_payments(count=3)
    canonical = map_payments_to_canonical(page, refunded_payment_ids=set())
    assert "order_id" in canonical.columns
    assert "order_date" in canonical.columns
    assert "amount" in canonical.columns
    assert canonical["return_event"].isna().all()  # no refunds known -> never fabricated False


def test_mapper_sets_refund_event_without_inventing_return_event():
    client = RazorpayClient(force_mock=True)
    page = client.fetch_payments(count=3)
    refunded_id = page["items"][0]["id"]
    canonical = map_payments_to_canonical(page, refunded_payment_ids={refunded_id})
    row = canonical[canonical["order_id"] == refunded_id].iloc[0]
    assert bool(row["refund_event"]) is True
    assert canonical.loc[canonical["order_id"] == refunded_id, "return_event"].isna().all()


def test_refunded_ids_extraction():
    refunds_page = {"items": [{"payment_id": "pay_1"}, {"payment_id": "pay_2"}]}
    assert refunded_ids_from_refunds_page(refunds_page) == {"pay_1", "pay_2"}


# ---------------------------------------------------------------------------
# webhook signature validation + idempotency
# ---------------------------------------------------------------------------

def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def test_verify_signature_accepts_correct_signature():
    body = b'{"event": "payment.captured"}'
    sig = _sign(body, "whsec_123")
    assert verify_signature(body, sig, "whsec_123")


def test_verify_signature_rejects_tampered_body():
    body = b'{"event": "payment.captured"}'
    sig = _sign(body, "whsec_123")
    tampered = b'{"event": "payment.captured", "amount": 999999}'
    assert not verify_signature(tampered, sig, "whsec_123")


def test_webhook_handler_rejects_invalid_signature():
    config = RazorpayConfig(key_id=None, key_secret=None, webhook_secret="whsec_123")
    handler = RazorpayWebhookHandler(config=config)
    body = json.dumps({"id": "evt_1", "event": "refund.processed"}).encode()
    with pytest.raises(InvalidWebhookSignature):
        handler.handle(organization_id="org_1", payload_body=body, received_signature="bad-sig",
                        payload_json=json.loads(body))


def test_webhook_handler_processes_valid_event_and_is_idempotent():
    config = RazorpayConfig(key_id=None, key_secret=None, webhook_secret="whsec_123")
    handler = RazorpayWebhookHandler(config=config)
    payload = {"id": "evt_dup_1", "event": "refund.processed", "created_at": 1700000000}
    body = json.dumps(payload).encode()
    sig = _sign(body, "whsec_123")

    first = handler.handle(organization_id="org_1", payload_body=body, received_signature=sig, payload_json=payload)
    assert first.processing_status == "processed"

    second = handler.handle(organization_id="org_1", payload_body=body, received_signature=sig, payload_json=payload)
    assert second.processing_status == "duplicate"

    assert len(handler.events_for_org("org_1")) == 2  # both recorded, but only one "processed"


def test_webhook_missing_secret_refuses_to_process():
    config = RazorpayConfig(key_id=None, key_secret=None, webhook_secret=None)
    handler = RazorpayWebhookHandler(config=config)
    body = b'{"id": "evt_1"}'
    with pytest.raises(InvalidWebhookSignature):
        handler.handle(organization_id="org_1", payload_body=body, received_signature="anything",
                        payload_json={"id": "evt_1"})


def test_client_uses_real_test_mode_http_path(monkeypatch):
    from src.integrations.razorpay import client as client_module

    class FakeResponse:
        status_code = 200
        text = "ok"
        def json(self):
            return {"entity": "collection", "count": 1, "items": [{"id": "pay_test_1"}]}

    class FakeRequests:
        @staticmethod
        def get(url, params, auth, timeout):
            assert url.endswith("/payments")
            assert params == {"count": 1, "skip": 0}
            assert auth == ("rzp_test_abc", "test_secret")
            assert timeout == 15
            return FakeResponse()

    monkeypatch.setattr(client_module, "requests", FakeRequests)
    cfg = RazorpayConfig(key_id="rzp_test_abc", key_secret="test_secret", webhook_secret="whsec")
    c = RazorpayClient(config=cfg)
    page = c.fetch_payments(count=1)
    assert c.is_demo is False
    assert page["items"][0]["id"] == "pay_test_1"
