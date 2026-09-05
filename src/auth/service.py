"""
Lightweight merchant authentication.

Deliberately NOT an enterprise identity platform (no SSO, no OAuth
provider, no external IdP) -- this is a hackathon MVP per PART M of
the spec ("do not overbuild"). It provides exactly what PART D asks
for:

  - login / logout
  - authenticated sessions (opaque bearer tokens, held in memory)
  - protected routes (via require_session)
  - password hashing (PBKDF2-HMAC-SHA256, salted; no plaintext
    passwords are ever stored)

Every account belongs to exactly one organization_id, which is what
makes every downstream read/write tenant-scoped (see
src.tenancy.store). This module never itself reads or writes business
data -- it only establishes "who is this, and which organization do
they belong to".
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional

PBKDF2_ITERATIONS = 200_000
SESSION_TTL_SECONDS = 12 * 60 * 60  # 12 hours


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Returns "salt_hex$hash_hex". Never returns or stores the
    plaintext password itself."""
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return hmac.compare_digest(candidate.hex(), digest_hex)


@dataclass
class MerchantAccount:
    user_id: str
    email: str
    organization_id: str
    password_hash: str
    display_name: str = ""


@dataclass
class Session:
    token: str
    user_id: str
    organization_id: str
    created_at: float
    expires_at: float


@dataclass
class AuthResult:
    success: bool
    session: Optional[Session] = None
    error: Optional[str] = None


class AuthService:
    """In-memory account + session store. Swappable for a real
    database-backed implementation later without changing any caller
    (every caller only ever uses login/logout/require_session)."""

    def __init__(self, session_ttl_seconds: int = SESSION_TTL_SECONDS):
        self._accounts_by_email: dict[str, MerchantAccount] = {}
        self._sessions: dict[str, Session] = {}
        self._session_ttl = session_ttl_seconds

    # -- account management -------------------------------------------------
    def register(self, email: str, password: str, organization_id: str, display_name: str = "") -> MerchantAccount:
        email = email.strip().lower()
        if email in self._accounts_by_email:
            raise ValueError(f"An account already exists for {email}.")
        account = MerchantAccount(
            user_id=secrets.token_hex(8),
            email=email,
            organization_id=organization_id,
            password_hash=hash_password(password),
            display_name=display_name,
        )
        self._accounts_by_email[email] = account
        return account

    # -- login / logout -------------------------------------------------
    def login(self, email: str, password: str) -> AuthResult:
        email = email.strip().lower()
        account = self._accounts_by_email.get(email)
        if account is None or not verify_password(password, account.password_hash):
            # Same generic error for "no such account" and "wrong
            # password" -- never reveal which one it was.
            return AuthResult(success=False, error="Invalid email or password.")

        now = time.time()
        session = Session(
            token=secrets.token_urlsafe(32),
            user_id=account.user_id,
            organization_id=account.organization_id,
            created_at=now,
            expires_at=now + self._session_ttl,
        )
        self._sessions[session.token] = session
        return AuthResult(success=True, session=session)

    def logout(self, token: Optional[str]) -> bool:
        if not token:
            return False
        return self._sessions.pop(token, None) is not None

    # -- protected-route support -------------------------------------------------
    def require_session(self, token: Optional[str]) -> AuthResult:
        if not token:
            return AuthResult(success=False, error="Authentication required. No session token provided.")
        session = self._sessions.get(token)
        if session is None:
            return AuthResult(success=False, error="Invalid or expired session.")
        if session.expires_at < time.time():
            self._sessions.pop(token, None)
            return AuthResult(success=False, error="Session expired. Please log in again.")
        return AuthResult(success=True, session=session)
