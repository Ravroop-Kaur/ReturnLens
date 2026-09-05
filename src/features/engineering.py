"""
Leakage-safe feature engineering for the return-risk model.

CRITICAL RULE: every feature computed here must be something that
would genuinely be known at the moment the order is placed (the
"prediction point"). In particular, historical return-rate features
for a product/category/fulfilment/region must only be computed from
orders that occurred STRICTLY BEFORE the current order, using an
expanding window ordered by order_date. This is equivalent to
leave-one-out / "as of" computation and prevents the classic mistake
of computing a group's return rate over the whole dataset (which
would leak the current order's own outcome into its own feature).

Forbidden inputs (see src.canonical.schema.LEAKAGE_FORBIDDEN_FIELDS):
return_event, return_date, refund_event, chargeback_event are NEVER
read here except to build the historical (past-only) aggregates,
and even then only using data strictly before each row's order_date.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List

from src.canonical.schema import LEAKAGE_FORBIDDEN_FIELDS

GLOBAL_PRIOR_SMOOTHING = 3  # Bayesian smoothing strength (pseudo-observations)


def _historical_known_arrays(df: pd.DataFrame, target_col: str, order_date_col: str,
                             return_date_col: str, return_window_days: int):
    """Return normalized dates, targets and the timestamp when each label became usable."""
    dates = pd.to_datetime(df[order_date_col], errors="coerce") if order_date_col in df.columns else pd.Series(
        pd.date_range("1970-01-01", periods=len(df), freq="ns"), index=df.index
    )
    target = pd.to_numeric(df[target_col], errors="coerce")
    ret = pd.to_datetime(df[return_date_col], errors="coerce") if return_date_col in df.columns else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    positive = target.eq(1) & ret.notna() & dates.notna() & ret.ge(dates)
    negative = target.eq(0) & dates.notna()
    info = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    info.loc[positive] = ret.loc[positive]
    info.loc[negative] = dates.loc[negative] + pd.to_timedelta(return_window_days, unit="D")
    return dates, target, info


def _prior_stats_by_time(current_times, event_times, event_values):
    """Vectorized prior sum/count using strictly event_time < current_time."""
    if len(event_times) == 0:
        return np.zeros(len(current_times), dtype=float), np.zeros(len(current_times), dtype=float)
    order = np.argsort(event_times.astype("datetime64[ns]"))
    et = event_times[order]
    ev = np.asarray(event_values, dtype=float)[order]
    counts = np.arange(1, len(ev) + 1, dtype=float)
    sums = np.cumsum(ev, dtype=float)
    pos = np.searchsorted(et, current_times.astype("datetime64[ns]"), side="left")
    out_count = np.where(pos > 0, counts[pos - 1], 0.0)
    out_sum = np.where(pos > 0, sums[pos - 1], 0.0)
    return out_sum, out_count


def _expanding_prior_rate(
    df: pd.DataFrame, group_col: str, target_col: str,
    order_date_col: str = "order_date", return_date_col: str = "return_date",
    return_window_days: int = 30,
) -> pd.Series:
    """Leakage-safe historical group rate, vectorized over timestamps.

    If called directly without an order-date column (legacy unit-test/helper
    usage), positional order is used as the explicit fallback. Production
    build_features still requires real order_date values.
    """
    work = df.copy()
    if order_date_col not in work.columns:
        # Legacy direct-helper compatibility only. build_features() refuses
        # date-less production data. Without timestamps there is no honest
        # temporal ordering, so return the global prior rather than inventing
        # one from row position. This keeps the helper's historical contract
        # explicit while preventing accidental use in production.
        target = pd.to_numeric(work[target_col], errors="coerce")
        mean = float(target.mean()) if target.notna().any() else 0.0
        return pd.Series(mean, index=work.index, dtype=float)
    dates, target, info = _historical_known_arrays(work, target_col, order_date_col, return_date_col, return_window_days)
    groups = work[group_col].astype(str).fillna("UNKNOWN")
    valid = info.notna() & target.notna()
    result = np.zeros(len(work), dtype=float)
    current = dates.to_numpy(dtype="datetime64[ns]")
    valid_idx = np.flatnonzero(valid.to_numpy())
    global_sum, global_count = _prior_stats_by_time(
        current,
        info.iloc[valid_idx].to_numpy(dtype="datetime64[ns]"),
        target.iloc[valid_idx].to_numpy(dtype=float),
    )
    global_mean = np.divide(global_sum, global_count, out=np.zeros_like(global_sum), where=global_count > 0)
    # Grouped searchsorted: no per-row pandas .loc access.
    group_values = groups.to_numpy()
    for g in pd.unique(group_values):
        mask = group_values == g
        event_mask = mask & valid.to_numpy()
        idx = np.flatnonzero(mask)
        eidx = np.flatnonzero(event_mask)
        sums, counts = _prior_stats_by_time(
            current[idx], info.iloc[eidx].to_numpy(dtype="datetime64[ns]"), target.iloc[eidx].to_numpy(dtype=float)
        )
        denom = counts + GLOBAL_PRIOR_SMOOTHING
        result[idx] = (sums + GLOBAL_PRIOR_SMOOTHING * global_mean[idx]) / denom
    return pd.Series(result, index=work.index)


def _expanding_known_count(
    df: pd.DataFrame, group_col: str, target_col: str, order_date_col: str = "order_date",
    return_date_col: str = "return_date", return_window_days: int = 30
) -> pd.Series:
    """Count mature historical labels in the same group, strictly before T."""
    work = df.copy()
    if order_date_col not in work.columns:
        return pd.Series(0.0, index=work.index, dtype=float)
    dates, target, info = _historical_known_arrays(work, target_col, order_date_col, return_date_col, return_window_days)
    groups = work[group_col].astype(str).fillna("UNKNOWN").to_numpy()
    valid = info.notna() & target.notna()
    current = dates.to_numpy(dtype="datetime64[ns]")
    result = np.zeros(len(work), dtype=float)
    for g in pd.unique(groups):
        mask = groups == g
        idx = np.flatnonzero(mask)
        eidx = np.flatnonzero(mask & valid.to_numpy())
        _, counts = _prior_stats_by_time(current[idx], info.iloc[eidx].to_numpy(dtype="datetime64[ns]"), np.ones(len(eidx)))
        result[idx] = counts
    return pd.Series(result, index=work.index)

def build_features(
    canonical_df: pd.DataFrame,
    target_col: str = "return_event",
    for_training: bool = True,
) -> "tuple[pd.DataFrame, pd.Series, list]":
    """
    Build the leakage-safe feature matrix.

    Data MUST already be sorted by order_date ascending before calling
    this (the historical-rate computation depends on row order within
    each group reflecting chronological order).

    Returns (X, y, feature_names). If for_training is False, y may be
    all-NaN (e.g. scoring new orders with unknown outcome).
    """
    df = canonical_df.copy()
    if "order_date" not in df.columns:
        raise ValueError("order_date is required for leakage-safe feature engineering")
    df = df.sort_values("order_date").reset_index(drop=True)

    if for_training:
        if target_col not in df.columns:
            raise ValueError(f"target column '{target_col}' not present -- cannot train")
        df = df[df[target_col].notna()].reset_index(drop=True)
        y = df[target_col].astype(int)
    else:
        y = df[target_col].astype("float") if target_col in df.columns else pd.Series(
            [np.nan] * len(df), index=df.index
        )

    features = pd.DataFrame(index=df.index)

    # --- direct, legitimately pre-outcome fields ---
    if "amount" in df.columns:
        features["amount"] = df["amount"]
        features["log_amount"] = np.log1p(df["amount"].clip(lower=0))

    if "order_date" in df.columns:
        dt = pd.DatetimeIndex(df["order_date"])
        features["order_dow"] = dt.dayofweek
        features["order_month"] = dt.month
        features["order_hour"] = dt.hour

    # --- historical (expanding, leave-one-out) group return rates ---
    # These are the ONLY features that touch return_event, and only
    # ever via a strictly-past expanding window.
    if target_col in df.columns:
        for group_col in ["customer_id", "product_id", "category", "fulfilment_method", "region", "shipping_service"]:
            if group_col in df.columns:
                col_name = f"hist_return_rate_{group_col}"
                # temporarily attach target to compute prior rate, then drop
                tmp_df = df[[group_col, "order_date"]].copy()
                if "return_date" in df.columns:
                    tmp_df["return_date"] = df["return_date"]
                tmp_df[target_col] = df[target_col].astype(float)
                features[col_name] = _expanding_prior_rate(tmp_df, group_col, target_col)
                # Prior sample size tells the model how trustworthy a historical rate is.
                # Count only labels that were actually mature/known at the
                # prediction timestamp. Raw cumcount would include pending
                # outcomes and would misstate the amount of historical evidence.
                features[f"hist_count_{group_col}"] = _expanding_known_count(
                    tmp_df, group_col, target_col
                )

    # --- label-free customer/product history available at prediction time ---
    # Counts and prior average order value do not depend on future outcomes.
    # They give the model behavioural context even when mature return labels
    # are sparse or delayed.
    for group_col in ["customer_id", "product_id"]:
        if group_col in df.columns:
            features[f"prior_order_count_{group_col}"] = df.groupby(group_col).cumcount().astype(float)
            if "amount" in df.columns:
                amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
                prior_sum = amt.groupby(df[group_col]).cumsum() - amt
                prior_count = df.groupby(group_col).cumcount().replace(0, np.nan)
                features[f"prior_avg_amount_{group_col}"] = (prior_sum / prior_count).fillna(amt.mean())

    # --- deterministic interactions available at prediction time ---
    if "category" in df.columns and "fulfilment_method" in df.columns:
        features["is_apparel_merchant"] = (
            (df["category"].astype(str) == "Apparel") &
            (df["fulfilment_method"].astype(str) == "merchant_fulfilled")
        ).astype(int)
    if "fulfilment_method" in df.columns and "shipping_service" in df.columns:
        features["is_third_party_economy"] = (
            (df["fulfilment_method"].astype(str) == "third_party_fulfilled") &
            (df["shipping_service"].astype(str) == "Economy")
        ).astype(int)

    # --- categorical one-hot (no leakage risk, purely descriptive) ---
    for cat_col in ["category", "fulfilment_method", "shipping_service", "region", "payment_status"]:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
            features = pd.concat([features, dummies], axis=1)

    features = features.fillna(0)
    feature_names = list(features.columns)

    # Defensive check: make sure no forbidden column ended up directly
    # in the feature matrix.
    leaked = LEAKAGE_FORBIDDEN_FIELDS.intersection(set(feature_names))
    if leaked:
        raise RuntimeError(f"Leakage-forbidden fields ended up in features: {leaked}")

    return features, y, feature_names
