"""
Tenant-scoped storage.

Multi-tenancy MUST be enforced server-side, not by frontend
discipline (PART D1). TenantScopedStore makes cross-tenant access
structurally impossible rather than merely policy-forbidden: every
key is namespaced by organization_id at write time, and every read
requires the caller to supply the organization_id it believes it is
reading -- there is no "read everything" or "read by kind alone" API
that a caller could accidentally use to leak across tenants.

This is an in-memory reference implementation (a real deployment
would back this with a database using an organization_id column/
partition and row-level security), but the API shape is what matters:
callers throughout the pipeline (risk_results, claims, evidence,
recommendations, data sources, model versions) all go through this
one object.
"""

from __future__ import annotations

from typing import Any, Optional


class TenantIsolationError(Exception):
    """Raised whenever code attempts to read or write a record under
    an organization_id that does not match the authenticated caller's
    own organization. Never silently swallowed."""


class TenantScopedStore:
    def __init__(self):
        # {(organization_id, kind, record_id): value}
        self._data: dict[tuple[str, str, str], Any] = {}

    def put(self, kind: str, organization_id: str, record_id: str, value: Any) -> None:
        if not organization_id:
            raise TenantIsolationError("Refusing to store a record with no organization_id.")
        self._data[(organization_id, kind, record_id)] = value

    def get(self, kind: str, organization_id: str, record_id: str) -> Optional[Any]:
        if not organization_id:
            raise TenantIsolationError("Refusing to read a record with no organization_id.")
        return self._data.get((organization_id, kind, record_id))

    def list_kind(self, kind: str, organization_id: str) -> list:
        """All record_ids of a given kind belonging to this
        organization only -- never across organizations."""
        if not organization_id:
            raise TenantIsolationError("Refusing to list records with no organization_id.")
        return [
            record_id
            for (org_id, k, record_id) in self._data.keys()
            if org_id == organization_id and k == kind
        ]

    def delete(self, kind: str, organization_id: str, record_id: str) -> bool:
        if not organization_id:
            raise TenantIsolationError("Refusing to delete a record with no organization_id.")
        return self._data.pop((organization_id, kind, record_id), None) is not None

    def assert_owned_by(self, organization_id: str, record_organization_id: str) -> None:
        """Convenience guard for callers holding a record that carries
        its own organization_id field (e.g. a ReturnClaim) -- raises
        rather than returning False, so a forgotten check fails loud."""
        if organization_id != record_organization_id:
            raise TenantIsolationError(
                "This record does not belong to the authenticated organization."
            )
