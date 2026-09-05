"""
Thin Razorpay API client.

Deliberately minimal -- only the read endpoints the return-risk
pipeline actually needs (payments and refunds, for historical
ingestion), not a full Razorpay SDK reimplementation.

If no credentials are configured (RazorpayConfig.is_configured is
False), or `force_mock=True`, the client returns a small, clearly
labelled synthetic payload instead of making a network call. This is
what lets PART J (demo mode) work without production Razorpay
credentials -- see RazorpayClient.is_demo.
"""

from __future__ import annotations

from typing import Optional

from src.integrations.razorpay.config import RazorpayConfig

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


class RazorpayAPIError(Exception):
    pass


class RazorpayClient:
    def __init__(self, config: Optional[RazorpayConfig] = None, force_mock: bool = False):
        self.config = config or RazorpayConfig.from_env()
        self.is_demo = force_mock or not self.config.is_configured

    def _auth(self):
        return (self.config.key_id, self.config.key_secret)

    def fetch_payments(self, count: int = 100, skip: int = 0) -> dict:
        if self.is_demo:
            return _mock_payments_page(count=count, skip=skip)
        if requests is None:
            raise RazorpayAPIError("The 'requests' package is not installed in this environment.")
        resp = requests.get(
            f"{self.config.base_url}/payments",
            params={"count": count, "skip": skip},
            auth=self._auth(),
            timeout=15,
        )
        if resp.status_code != 200:
            raise RazorpayAPIError(f"Razorpay API error {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def fetch_refunds(self, count: int = 100, skip: int = 0) -> dict:
        if self.is_demo:
            return {"count": 0, "items": []}
        if requests is None:
            raise RazorpayAPIError("The 'requests' package is not installed in this environment.")
        resp = requests.get(
            f"{self.config.base_url}/refunds",
            params={"count": count, "skip": skip},
            auth=self._auth(),
            timeout=15,
        )
        if resp.status_code != 200:
            raise RazorpayAPIError(f"Razorpay API error {resp.status_code}: {resp.text[:500]}")
        return resp.json()


def _mock_payments_page(count: int, skip: int) -> dict:
    """A small, deterministic, clearly-synthetic page of Razorpay-shaped
    payment objects, used only when no real credentials are configured."""
    items = []
    n = max(0, min(count, 20) - skip) if skip < 20 else 0
    for i in range(n):
        idx = skip + i
        items.append({
            "id": f"pay_MOCKDEMO{idx:04d}",
            "amount": 50000 + idx * 137,  # paise
            "currency": "INR",
            "status": "captured",
            "created_at": 1700000000 + idx * 86400,
            "notes": {"category": ["electronics", "apparel", "home"][idx % 3]},
        })
    return {"entity": "collection", "count": len(items), "items": items, "is_demo": True}
