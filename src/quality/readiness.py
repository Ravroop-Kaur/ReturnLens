"""
Data readiness pipeline.

RAW MERCHANT DATA -> SCHEMA DETECTION -> DATA QUALITY CHECK ->
CANONICAL MAPPING -> FEATURE AVAILABILITY -> MODEL READINESS ->
PREDICT / ABSTAIN

This module is the single place that decides whether the pipeline is
allowed to train or score a model for a given ingested dataset. It
never fabricates data; every field it reports on is either present in
the merchant's export or explicitly marked absent/unusable/pending.

Nothing here is merchant-specific (no Amazon-isms, no assumptions
about which optional fields exist) -- that is the point of running
this on canonical_df, after src.adapters/* and src.canonical.mapping
have already done source-specific translation.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional

from src.quality.dedup import analyze_duplicates, drop_exact_duplicates, DuplicateReport
from src.quality.order_level import aggregate_to_order_level, detect_granularity, GranularityReport
from src.quality.feature_contract import evaluate_feature_contract, FeatureContractResult
from src.quality.lifecycle import assign_label_state, report as lifecycle_report, DEFAULT_RETURN_WINDOW_DAYS


MIN_ROWS_FOR_TRAINING = 200          # below this, a model would be trained on noise
MIN_LABELED_ROWS_FOR_TRAINING = 100  # RETURNED + NO_RETURN rows, post label-lifecycle filtering
MIN_HISTORY_DAYS = 30                # a merchant with < 30 days of history has almost no
                                      # temporal signal and no way to hold out a later test period


@dataclass
class DataReadinessReport:
    n_rows_raw: int
    n_orders: int
    granularity: GranularityReport
    duplicates: DuplicateReport
    feature_contract: FeatureContractResult
    date_validity: dict
    numeric_validity: dict
    history_days: Optional[float]
    n_returned: int
    n_no_return: int
    n_pending: int
    target_completeness: float  # fraction of orders with a FINALIZED label (not pending)
    model_status: str           # "READY" | "NOT_READY"
    model_readiness_label: str  # "FULLY_SUPPORTED" | "PARTIALLY_SUPPORTED" | "NOT_READY"
    reasons_not_ready: list

    def to_dict(self) -> dict:
        return {
            "n_rows_raw": self.n_rows_raw,
            "n_orders": self.n_orders,
            "granularity": self.granularity.to_dict(),
            "duplicates": self.duplicates.to_dict(),
            "feature_contract": self.feature_contract.rows_for_ui(),
            "date_validity": self.date_validity,
            "numeric_validity": self.numeric_validity,
            "history_days": self.history_days,
            "label_lifecycle": {
                "n_returned": self.n_returned,
                "n_no_return": self.n_no_return,
                "n_pending": self.n_pending,
                "target_completeness": round(self.target_completeness, 4),
            },
            "model_status": self.model_status,
            "model_readiness_label": self.model_readiness_label,
            "reasons_not_ready": self.reasons_not_ready,
        }


def _date_validity(df: pd.DataFrame, col: str = "order_date") -> dict:
    if col not in df.columns:
        return {"present": False}
    parsed = pd.to_datetime(df[col], errors="coerce")
    n = len(df)
    n_invalid = int(parsed.isna().sum())
    return {
        "present": True,
        "n_invalid": n_invalid,
        "pct_invalid": round(n_invalid / n, 4) if n else 0.0,
        "min_date": str(parsed.min()) if n_invalid < n else None,
        "max_date": str(parsed.max()) if n_invalid < n else None,
    }


def _numeric_validity(df: pd.DataFrame, col: str = "amount") -> dict:
    if col not in df.columns:
        return {"present": False}
    parsed = pd.to_numeric(df[col], errors="coerce")
    n = len(df)
    n_invalid = int(parsed.isna().sum())
    n_negative = int((parsed < 0).sum())
    return {
        "present": True,
        "n_invalid": n_invalid,
        "pct_invalid": round(n_invalid / n, 4) if n else 0.0,
        "n_negative": n_negative,
    }


def run_data_readiness_pipeline(
    raw_or_canonical_df: pd.DataFrame,
    order_id_col: str = "order_id",
    require_target: bool = True,
    return_window_days: int = DEFAULT_RETURN_WINDOW_DAYS,
    as_of: pd.Timestamp = None,
) -> "tuple[pd.DataFrame, DataReadinessReport]":
    """
    Runs SCHEMA DETECTION -> DATA QUALITY CHECK -> feature availability
    -> MODEL READINESS on an already-canonicalized dataframe (i.e. after
    src.canonical.mapping has renamed source columns to canonical
    names). Returns the order-level, deduplicated dataframe the model
    should actually be trained/scored on, plus the full report.
    """
    n_raw = len(raw_or_canonical_df)

    # --- DATA QUALITY CHECK: duplicates ---
    dup_report = analyze_duplicates(raw_or_canonical_df, order_id_col=order_id_col)
    deduped, _ = drop_exact_duplicates(raw_or_canonical_df)

    # --- SCHEMA DETECTION: order-level vs line-item granularity ---
    order_df, gran_report = aggregate_to_order_level(deduped, order_id_col=order_id_col)

    date_validity = _date_validity(order_df)
    numeric_validity = _numeric_validity(order_df)

    history_days = None
    if "order_date" in order_df.columns:
        dates = pd.to_datetime(order_df["order_date"], errors="coerce").dropna()
        if len(dates):
            history_days = float((dates.max() - dates.min()).days)

    # --- LABEL LIFECYCLE: never treat pending as a fabricated negative ---
    n_returned = n_no_return = n_pending = 0
    target_completeness = 0.0
    if "order_date" in order_df.columns:
        state = assign_label_state(order_df, as_of=as_of, return_window_days=return_window_days)
        lc = lifecycle_report(state)
        n_returned, n_no_return, n_pending = lc.n_returned, lc.n_no_return, lc.n_pending
        target_completeness = (n_returned + n_no_return) / lc.n_total if lc.n_total else 0.0

    # --- FEATURE AVAILABILITY: explicit REQUIRED/RECOMMENDED/OPTIONAL contract ---
    contract = evaluate_feature_contract(order_df, require_target=require_target)

    # --- MODEL READINESS ---
    reasons = []
    if not contract.has_all_required():
        reasons.append("One or more REQUIRED fields are missing or unusable.")
    # Required fields must also be semantically valid, not merely present.
    if date_validity.get("n_invalid", 0) > 0:
        reasons.append(f"{date_validity['n_invalid']} rows have invalid order_date values.")
    if numeric_validity.get("n_invalid", 0) > 0:
        reasons.append(f"{numeric_validity['n_invalid']} rows have non-numeric amount values.")
    if numeric_validity.get("n_negative", 0) > 0:
        reasons.append(f"{numeric_validity['n_negative']} rows have negative amount values.")
    if "order_id" in order_df.columns and order_df["order_id"].isna().any():
        reasons.append("Some orders have missing order_id values.")
    if require_target:
        n_labeled = n_returned + n_no_return
        if n_labeled < MIN_LABELED_ROWS_FOR_TRAINING:
            reasons.append(
                f"Only {n_labeled} orders have a finalized return label "
                f"(need at least {MIN_LABELED_ROWS_FOR_TRAINING}); the rest are pending "
                f"or unlabeled. Insufficient return labels."
            )
    if len(order_df) < MIN_ROWS_FOR_TRAINING:
        reasons.append(f"Only {len(order_df)} orders available (need at least {MIN_ROWS_FOR_TRAINING}).")
    if history_days is not None and history_days < MIN_HISTORY_DAYS:
        reasons.append(f"Only {history_days:.0f} days of order history available (need at least {MIN_HISTORY_DAYS}).")
    elif history_days is None:
        reasons.append("order_date could not be parsed for any row.")

    model_status = "NOT_READY" if reasons else "READY"
    readiness_label = "NOT_READY" if reasons else contract.readiness_label()

    report = DataReadinessReport(
        n_rows_raw=n_raw,
        n_orders=len(order_df),
        granularity=gran_report,
        duplicates=dup_report,
        feature_contract=contract,
        date_validity=date_validity,
        numeric_validity=numeric_validity,
        history_days=history_days,
        n_returned=n_returned,
        n_no_return=n_no_return,
        n_pending=n_pending,
        target_completeness=target_completeness,
        model_status=model_status,
        model_readiness_label=readiness_label,
        reasons_not_ready=reasons,
    )
    return order_df, report
