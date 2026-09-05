"""
Generic merchant CSV ingestion. This is the PRIMARY ingestion path.

Flow:
    Upload CSV -> inspect columns -> propose mapping -> validate
    -> report capabilities -> (caller decides whether to proceed)

Any merchant whose export roughly matches "one row per order, with a
date, an amount, and optionally a return/refund flag" can use this
path directly. Merchant-specific quirks (like Amazon's column names
or status codes) belong in their own adapter, not here.
"""

from __future__ import annotations
import pandas as pd
from typing import Dict, Optional

from src.canonical.mapping import (
    propose_mapping,
    apply_mapping,
    validate_and_summarize,
    coerce_types,
)
from src.canonical.schema import CanonicalCapabilities


class IngestionResult:
    def __init__(
        self,
        canonical_df: pd.DataFrame,
        capabilities: CanonicalCapabilities,
        mapping_used: Dict[str, Optional[str]],
        source_columns: list,
        n_rows_raw: int,
    ):
        self.canonical_df = canonical_df
        self.capabilities = capabilities
        self.mapping_used = mapping_used
        self.source_columns = source_columns
        self.n_rows_raw = n_rows_raw

    def can_train_detector(self) -> bool:
        return self.capabilities.has_target and not self.capabilities.missing_required

    def message(self) -> str:
        if self.capabilities.missing_required:
            return (
                "Required fields missing from this dataset: "
                f"{self.capabilities.missing_required}. Cannot proceed."
            )
        if not self.capabilities.has_target:
            return "Return outcomes are unavailable in this dataset."
        return "Dataset supports full return-risk analysis."


def ingest_generic_csv(
    csv_path: str,
    mapping_override: Optional[Dict[str, Optional[str]]] = None,
) -> IngestionResult:
    """Load a generic merchant CSV and map it into the canonical schema.

    mapping_override lets a caller (UI or adapter) fix / extend the
    auto-proposed mapping, e.g. when column names are ambiguous.
    """
    raw = pd.read_csv(csv_path)
    proposed = propose_mapping(list(raw.columns))
    if mapping_override:
        proposed.update(mapping_override)

    canonical = apply_mapping(raw, proposed)
    canonical = coerce_types(canonical)
    caps = validate_and_summarize(canonical)

    return IngestionResult(
        canonical_df=canonical,
        capabilities=caps,
        mapping_used=proposed,
        source_columns=list(raw.columns),
        n_rows_raw=len(raw),
    )
