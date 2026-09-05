"""
Razorpay configuration.

Never hard-code API keys, secrets, or webhook secrets. Everything is
read from environment variables, and this module is the ONLY place
that reads them -- the client and webhook handler both go through
RazorpayConfig rather than calling os.environ directly, so there is
one obvious place to check when auditing what credentials the app
uses.

Supports Razorpay Test Mode: test keys look like "rzp_test_..." and
work identically to live keys against Razorpay's API, so no
special-casing is needed here beyond documenting it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RazorpayConfig:
    key_id: Optional[str]
    key_secret: Optional[str]
    webhook_secret: Optional[str]
    base_url: str = "https://api.razorpay.com/v1"

    @property
    def is_configured(self) -> bool:
        return bool(self.key_id and self.key_secret)

    @property
    def is_test_mode(self) -> bool:
        return bool(self.key_id and self.key_id.startswith("rzp_test_"))

    @classmethod
    def from_env(cls) -> "RazorpayConfig":
        return cls(
            key_id=os.environ.get("RAZORPAY_KEY_ID"),
            key_secret=os.environ.get("RAZORPAY_KEY_SECRET"),
            webhook_secret=os.environ.get("RAZORPAY_WEBHOOK_SECRET"),
            base_url=os.environ.get("RAZORPAY_BASE_URL", "https://api.razorpay.com/v1"),
        )
