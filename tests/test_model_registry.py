from dataclasses import dataclass

from src.model.registry import (
    decide_model_scope, ModelRegistry,
    MERCHANT_SPECIFIC, GLOBAL_BASELINE, INSUFFICIENT_DATA,
)


@dataclass
class _FakeReadiness:
    model_status: str
    n_orders: int
    n_returned: int
    n_no_return: int
    history_days: float
    reasons_not_ready: list


def test_abstains_when_not_ready():
    readiness = _FakeReadiness("NOT_READY", 10, 1, 1, 5, ["Only 10 orders available (need at least 200)."])
    decision = decide_model_scope(readiness)
    assert decision.status == INSUFFICIENT_DATA


def test_global_baseline_when_ready_but_thin_merchant_history():
    readiness = _FakeReadiness("READY", 300, 100, 100, 60, [])
    decision = decide_model_scope(readiness)
    assert decision.status == GLOBAL_BASELINE


def test_merchant_specific_when_deep_history():
    readiness = _FakeReadiness("READY", 2000, 800, 800, 200, [])
    decision = decide_model_scope(readiness)
    assert decision.status == MERCHANT_SPECIFIC


def test_registry_tracks_versions_incrementally():
    registry = ModelRegistry()
    v1 = registry.register(
        organization_id="org_1", model_scope=GLOBAL_BASELINE, model_name="logistic_regression",
        n_train_rows=100, feature_names=["a", "b"], threshold=0.5, is_synthetic_demo=True,
    )
    v2 = registry.register(
        organization_id="org_1", model_scope=MERCHANT_SPECIFIC, model_name="lightgbm",
        n_train_rows=2000, feature_names=["a", "b", "c"], threshold=0.4, is_synthetic_demo=False,
    )
    assert v1.version == 1
    assert v2.version == 2
    assert registry.latest("org_1").version == 2
    assert len(registry.history("org_1")) == 2
    assert v1.label() == "DEMO / SYNTHETIC"
    assert v2.label() == "MERCHANT HELD-OUT TEST"


def test_registry_isolates_organizations():
    registry = ModelRegistry()
    registry.register("org_a", GLOBAL_BASELINE, "logistic_regression", 100, ["a"], 0.5)
    assert registry.latest("org_b") is None
