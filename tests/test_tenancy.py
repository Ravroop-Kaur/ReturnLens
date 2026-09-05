import pytest

from src.tenancy.store import TenantScopedStore, TenantIsolationError


def test_put_and_get_roundtrip():
    store = TenantScopedStore()
    store.put("risk_results", "org_a", "latest", {"status": "ok"})
    assert store.get("risk_results", "org_a", "latest") == {"status": "ok"}


def test_cross_tenant_read_returns_nothing_not_other_tenants_data():
    store = TenantScopedStore()
    store.put("risk_results", "org_a", "latest", {"secret": "a-data"})
    store.put("risk_results", "org_b", "latest", {"secret": "b-data"})
    assert store.get("risk_results", "org_a", "latest") == {"secret": "a-data"}
    assert store.get("risk_results", "org_b", "latest") == {"secret": "b-data"}
    # org_a's key never leaks org_b's value even under the same record_id
    assert store.get("risk_results", "org_a", "latest") != store.get("risk_results", "org_b", "latest")


def test_list_kind_only_returns_own_organization():
    store = TenantScopedStore()
    store.put("claims", "org_a", "c1", {})
    store.put("claims", "org_a", "c2", {})
    store.put("claims", "org_b", "c3", {})
    assert sorted(store.list_kind("claims", "org_a")) == ["c1", "c2"]
    assert store.list_kind("claims", "org_b") == ["c3"]


def test_empty_organization_id_rejected():
    store = TenantScopedStore()
    with pytest.raises(TenantIsolationError):
        store.put("claims", "", "c1", {})
    with pytest.raises(TenantIsolationError):
        store.get("claims", "", "c1")


def test_assert_owned_by_raises_on_mismatch():
    store = TenantScopedStore()
    with pytest.raises(TenantIsolationError):
        store.assert_owned_by("org_a", "org_b")
    store.assert_owned_by("org_a", "org_a")  # does not raise


def test_delete_is_tenant_scoped():
    store = TenantScopedStore()
    store.put("claims", "org_a", "c1", {})
    store.put("claims", "org_b", "c1", {})
    assert store.delete("claims", "org_a", "c1") is True
    assert store.get("claims", "org_a", "c1") is None
    # org_b's identically-named record is untouched
    assert store.get("claims", "org_b", "c1") == {}
