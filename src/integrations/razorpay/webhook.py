"""
Razorpay webhook handling.

Flow (PART B1):

    RAZORPAY -> WEBHOOK -> SIGNATURE VALIDATION -> IDEMPOTENCY CHECK
    -> EVENT STORAGE -> CANONICAL EVENT -> MERCHANT DATA -> RISK UPDATE

This module implements everything up through "canonical event";
"risk update" is the caller's job (re-running the readiness/model
pipeline), not this module's.

CRITICAL: webhooks are DATA INGESTION, never autonomous action (PART
B2). Nothing here refunds money, blocks a customer, cancels an order,
or changes courier/payment configuration -- it only records an event
and, if it is a refund event, feeds refunded_payment_ids into the
Razorpay mapper so the next risk computation reflects it.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Optional

from src.integrations.razorpay.config import RazorpayConfig


class InvalidWebhookSignature(Exception):
    pass


def verify_signature(payload_body: bytes, received_signature: str, webhook_secret: str) -> bool:
    """Razorpay signs the raw request body with HMAC-SHA256 using the
    webhook secret configured in the Razorpay dashboard. Comparison
    uses hmac.compare_digest to avoid timing side-channels."""
    if not webhook_secret or not received_signature:
        return False
    expected = hmac.new(webhook_secret.encode("utf-8"), payload_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, received_signature)


@dataclass
class StoredWebhookEvent:
    event_id: str
    event_type: str
    organization_id: str
    received_at: float
    occurred_at: Optional[float]
    source: str
    processing_status: str  # "received" | "processed" | "duplicate" | "rejected"
    payload_reference: dict = field(default_factory=dict)



class WebhookIntegrationRegistry:
    """Server-side mapping from a public data-source endpoint to a
    merchant organization and its webhook secret.

    The incoming request never supplies organization_id. In production
    this mapping belongs in persistent integration configuration. This
    in-memory version is sufficient for the demo and preserves the
    security boundary.
    """

    def __init__(self):
        self._sources: dict[str, tuple[str, str]] = {}

    def register(self, data_source_id: str, organization_id: str, webhook_secret: str) -> None:
        if not data_source_id or not organization_id or not webhook_secret:
            raise ValueError("data_source_id, organization_id and webhook_secret are required.")
        self._sources[data_source_id] = (organization_id, webhook_secret)

    def resolve(self, data_source_id: str) -> tuple[str, str]:
        value = self._sources.get(data_source_id)
        if value is None:
            raise InvalidWebhookSignature("Unknown Razorpay data source.")
        return value

class RazorpayWebhookHandler:
    """Tracks processed event IDs in-memory per organization to
    guarantee idempotency: the same Razorpay event_id delivered twice
    (Razorpay retries on timeout) must never create a duplicate
    business event or be processed twice."""

    def __init__(self, config: Optional[RazorpayConfig] = None):
        self.config = config or RazorpayConfig.from_env()
        self._seen_event_ids: dict[str, set] = {}  # organization_id -> {event_id}
        self._events: list[StoredWebhookEvent] = []

    def handle(
        self,
        organization_id: str,
        payload_body: bytes,
        received_signature: str,
        payload_json: dict,
        webhook_secret: Optional[str] = None,
    ) -> StoredWebhookEvent:
        secret = webhook_secret or self.config.webhook_secret
        if not secret:
            raise InvalidWebhookSignature(
                "Razorpay webhook secret is not configured; refusing to process webhook."
            )
        if not verify_signature(payload_body, received_signature, secret):
            event = StoredWebhookEvent(
                event_id=payload_json.get("id", "unknown"),
                event_type=payload_json.get("event", "unknown"),
                organization_id=organization_id,
                received_at=time.time(),
                occurred_at=payload_json.get("created_at"),
                source="razorpay",
                processing_status="rejected",
                payload_reference={"reason": "invalid_signature"},
            )
            self._events.append(event)
            raise InvalidWebhookSignature("Webhook signature validation failed.")

        event_id = payload_json.get("id") or f"no-id-{time.time()}"
        event_type = payload_json.get("event", "unknown")
        seen = self._seen_event_ids.setdefault(organization_id, set())

        if event_id in seen:
            event = StoredWebhookEvent(
                event_id=event_id,
                event_type=event_type,
                organization_id=organization_id,
                received_at=time.time(),
                occurred_at=payload_json.get("created_at"),
                source="razorpay",
                processing_status="duplicate",
                payload_reference={"payload_keys": list(payload_json.keys())},
            )
            self._events.append(event)
            return event

        seen.add(event_id)
        event = StoredWebhookEvent(
            event_id=event_id,
            event_type=event_type,
            organization_id=organization_id,
            received_at=time.time(),
            occurred_at=payload_json.get("created_at"),
            source="razorpay",
            processing_status="processed",
            payload_reference={"payload_keys": list(payload_json.keys())},
        )
        self._events.append(event)
        return event

    def events_for_org(self, organization_id: str) -> list:
        return [e for e in self._events if e.organization_id == organization_id]
