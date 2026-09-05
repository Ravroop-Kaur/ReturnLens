import pandas as pd
import pytest

from src.auth.service import AuthService
from src.tenancy.store import TenantScopedStore, TenantIsolationError
from src.model.registry import ModelRegistry
from src.connectors.mock import MockMerchantConnector
from src.connectors.base import MerchantDataConnector, ConnectionHealth
from src.pipeline import run_organization_pipeline, review_return_claim, PipelineAuthError
from src.claims.model import ReturnClaim
from src.evidence.image_analyzer import MockImageEvidenceAnalyzer


class _TinyConnector(MerchantDataConnector):
    """A connector with far too little data -- exercises the
    INSUFFICIENT_DATA / abstain path end-to-end."""
    connector_type = "mock"

    def test_connection(self):
        return ConnectionHealth(healthy=True, detail="tiny demo")

    def fetch_historical_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            "order_id": ["1", "2", "3"],
            "order_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "amount": [10.0, 20.0, 30.0],
            "return_event": [True, False, False],
        })

    def fetch_incremental_data(self, since=None) -> pd.DataFrame:
        return self.fetch_historical_data()


def _authed_env():
    auth = AuthService()
    auth.register(email="merchant@shop.com", password="pw123456", organization_id="org_1")
    token = auth.login("merchant@shop.com", "pw123456").session.token
    return auth, token


def test_pipeline_requires_authentication():
    auth = AuthService()
    tenant_store = TenantScopedStore()
    registry = ModelRegistry()
    with pytest.raises(PipelineAuthError):
        run_organization_pipeline(
            token="not-a-real-token", auth=auth, connector=MockMerchantConnector(),
            tenant_store=tenant_store, registry=registry,
        )


def test_pipeline_abstains_on_insufficient_data():
    auth, token = _authed_env()
    tenant_store = TenantScopedStore()
    registry = ModelRegistry()
    result = run_organization_pipeline(
        token=token, auth=auth, connector=_TinyConnector(),
        tenant_store=tenant_store, registry=registry,
    )
    assert result["status"] == "abstained"
    assert result["model_decision"]["status"] == "INSUFFICIENT_DATA"
    # abstained result is still persisted, tenant-scoped, so the
    # dashboard always has something to read.
    stored = tenant_store.get("risk_results", "org_1", "latest")
    assert stored["status"] == "abstained"


def test_pipeline_runs_end_to_end_on_synthetic_demo_data():
    auth, token = _authed_env()
    tenant_store = TenantScopedStore()
    registry = ModelRegistry()
    result = run_organization_pipeline(
        token=token, auth=auth, connector=MockMerchantConnector(),
        tenant_store=tenant_store, registry=registry, dataset_label="synthetic demo",
    )
    assert result["status"] == "ok"
    assert result["is_synthetic_demo"] is True
    assert "precision" in result["test_evaluation"]
    assert "confusion" not in result  # matrix lives inside counts, not a bare top-level key
    assert result["model_version"]["organization_id"] == "org_1"
    assert registry.latest("org_1") is not None


def test_cross_tenant_results_never_mix():
    auth = AuthService()
    auth.register(email="a@shop.com", password="pw123456", organization_id="org_a")
    auth.register(email="b@shop.com", password="pw123456", organization_id="org_b")
    token_a = auth.login("a@shop.com", "pw123456").session.token
    token_b = auth.login("b@shop.com", "pw123456").session.token

    tenant_store = TenantScopedStore()
    registry = ModelRegistry()

    run_organization_pipeline(token=token_a, auth=auth, connector=MockMerchantConnector(),
                               tenant_store=tenant_store, registry=registry)
    run_organization_pipeline(token=token_b, auth=auth, connector=_TinyConnector(),
                               tenant_store=tenant_store, registry=registry)

    result_a = tenant_store.get("risk_results", "org_a", "latest")
    result_b = tenant_store.get("risk_results", "org_b", "latest")
    assert result_a["status"] == "ok"
    assert result_b["status"] == "abstained"
    assert result_a["organization_id"] == "org_a"
    assert result_b["organization_id"] == "org_b"


def test_claim_review_rejects_cross_tenant_claim():
    auth, token = _authed_env()
    tenant_store = TenantScopedStore()
    claim = ReturnClaim(
        claim_id="c1", organization_id="some_other_org", order_id="o1",
        claimed_reason="Arrived damaged",
    )
    with pytest.raises(PipelineAuthError):
        review_return_claim(token=token, auth=auth, claim=claim, tenant_store=tenant_store)


def test_claim_review_succeeds_for_own_organization():
    auth, token = _authed_env()
    tenant_store = TenantScopedStore()
    claim = ReturnClaim(
        claim_id="c1", organization_id="org_1", order_id="o1",
        claimed_reason="Arrived damaged", claim_timestamp="2025-01-05",
        order_date="2025-01-01", delivery_status="delivered",
        customer_prior_return_count=0, image_references=["ref_1"],
    )
    result = review_return_claim(
        token=token, auth=auth, claim=claim, tenant_store=tenant_store,
        image_analyzer=MockImageEvidenceAnalyzer(),
    )
    assert result["claim_id"] == "c1"
    assert result["status"] in ("SUPPORTED", "NEEDS_REVIEW", "SUSPICIOUS", "INDETERMINATE")
    stored = tenant_store.get("claims", "org_1", "c1")
    assert stored == result
