"""
Model scope decision + model version registry.

Two responsibilities (PART A7 / A8 / 13 of the spec):

1. decide_model_scope(readiness) -- given a DataReadinessReport,
   decide whether this organization gets:
     - a MERCHANT_SPECIFIC model (enough of their own labelled
       history to train and hold out a genuine test set), or
     - the GLOBAL_BASELINE model (not enough history, but required
       features are available), or
     - INSUFFICIENT_DATA (abstain -- required fields missing/unusable,
       or not even enough data for the global baseline path).

   This module never trains anything itself -- it only decides which
   path src.pipeline should take, based purely on the readiness report
   (never on wishful thinking about what the data "probably" contains).

2. ModelRegistry -- tracks every trained model version per
   organization (PART 13: "Model versions are tracked"), purely as a
   record-keeping ledger. It does not itself decide when to retrain
   (see src.quality.drift -- drift never triggers automatic retraining).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

MERCHANT_SPECIFIC = "MERCHANT_SPECIFIC"
GLOBAL_BASELINE = "GLOBAL_BASELINE"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# A merchant needs at least this many finalized-label orders AND this
# many days of history before we trust training a model on their data
# alone rather than falling back to the global baseline.
MIN_ORDERS_FOR_MERCHANT_MODEL = 1000
MIN_LABELED_ROWS_FOR_MERCHANT_MODEL = 500
MIN_HISTORY_DAYS_FOR_MERCHANT_MODEL = 90


@dataclass
class ModelScopeDecision:
    status: str  # MERCHANT_SPECIFIC | GLOBAL_BASELINE | INSUFFICIENT_DATA
    scope: str   # same vocabulary, kept as a separate field for readability in reports
    reason: str


def decide_model_scope(readiness) -> ModelScopeDecision:
    """`readiness` is a src.quality.readiness.DataReadinessReport."""
    if readiness.model_status == "NOT_READY":
        reason = " ".join(readiness.reasons_not_ready) or "Required data is missing or unusable."
        return ModelScopeDecision(status=INSUFFICIENT_DATA, scope=INSUFFICIENT_DATA, reason=reason)

    n_labeled = readiness.n_returned + readiness.n_no_return
    history_days = readiness.history_days or 0

    if (
        readiness.n_orders >= MIN_ORDERS_FOR_MERCHANT_MODEL
        and n_labeled >= MIN_LABELED_ROWS_FOR_MERCHANT_MODEL
        and history_days >= MIN_HISTORY_DAYS_FOR_MERCHANT_MODEL
    ):
        return ModelScopeDecision(
            status=MERCHANT_SPECIFIC,
            scope=MERCHANT_SPECIFIC,
            reason=(
                f"Sufficient merchant-specific history ({readiness.n_orders} orders, "
                f"{n_labeled} finalized labels, {history_days:.0f} days) to train and "
                f"hold out a merchant-specific model."
            ),
        )

    return ModelScopeDecision(
        status=GLOBAL_BASELINE,
        scope=GLOBAL_BASELINE,
        reason=(
            f"Required fields are usable, but merchant history is not yet large enough "
            f"({readiness.n_orders} orders, {n_labeled} finalized labels, {history_days:.0f} days) "
            f"for a merchant-specific model. Using the global baseline model."
        ),
    )


@dataclass
class ModelVersion:
    organization_id: str
    version: int
    model_scope: str
    model_name: str
    n_train_rows: int
    feature_names: list
    threshold: float
    is_synthetic_demo: bool
    trained_at: float = field(default_factory=time.time)

    def label(self) -> str:
        if self.is_synthetic_demo:
            return "DEMO / SYNTHETIC"
        return "MERCHANT HELD-OUT TEST" if self.model_scope == MERCHANT_SPECIFIC else "GLOBAL BASELINE"


class ModelRegistry:
    """In-memory ledger of every trained model version per
    organization. Never deletes history -- each register() call
    appends a new version; nothing is overwritten."""

    def __init__(self):
        self._versions: dict[str, list[ModelVersion]] = {}

    def register(
        self,
        organization_id: str,
        model_scope: str,
        model_name: str,
        n_train_rows: int,
        feature_names: list,
        threshold: float,
        is_synthetic_demo: bool = False,
    ) -> ModelVersion:
        history = self._versions.setdefault(organization_id, [])
        version = ModelVersion(
            organization_id=organization_id,
            version=len(history) + 1,
            model_scope=model_scope,
            model_name=model_name,
            n_train_rows=n_train_rows,
            feature_names=list(feature_names),
            threshold=threshold,
            is_synthetic_demo=is_synthetic_demo,
        )
        history.append(version)
        return version

    def latest(self, organization_id: str) -> Optional[ModelVersion]:
        history = self._versions.get(organization_id, [])
        return history[-1] if history else None

    def history(self, organization_id: str) -> list:
        return list(self._versions.get(organization_id, []))
