"""
End-to-end training pipeline for the ML return-risk scorer.

Order of operations (mirrors the project's leakage/evaluation rules):

 1. Build leakage-safe features on the FULL canonical dataset, sorted
    by order_date, so historical-rate features are true expanding
    windows across the merchant's whole history.
 2. Split by date into train / validation / test (test is the latest,
    frozen, held-out period).
 3. Fit models on train only.
 4. Select the better model AND the classification threshold using
    validation only.
 5. Freeze model + threshold. Evaluate ONCE on test.

This module returns everything the evaluation and reporting layers
need, including the exact split boundaries and the frozen threshold,
for reproducibility.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without lightgbm
    lgb = None
    LIGHTGBM_AVAILABLE = False

from src.features.engineering import build_features
from src.model.split import temporal_split, SplitBounds

RANDOM_SEED = 42


@dataclass
class TrainedModelBundle:
    model_name: str
    model: object
    scaler: object  # None for tree models
    feature_names: list
    threshold: float
    split_bounds: SplitBounds
    val_metrics_at_candidates: dict
    random_seed: int = RANDOM_SEED

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.feature_names, fill_value=0)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.model_name == "logistic_regression":
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)


def _fit_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    model.fit(X_scaled, y_train)
    return model, scaler


def _fit_lightgbm(X_train, y_train):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model, None


def _select_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    target_precision: float = 0.70,
    target_recall: float = 0.70,
) -> dict:
    """
    Select the operating threshold using VALIDATION DATA ONLY.

    For the demo model we prefer an operating point where both precision
    and recall are at least 70%, then maximize F1 within that feasible
    region. If validation data cannot support that target, fall back to
    the threshold with the best F1. The frozen test set is never used
    for threshold selection.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []
    for t in thresholds:
        pred = (p_val >= t).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "threshold": round(float(t), 3),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    feasible = [
        r for r in rows
        if r["precision"] >= target_precision and r["recall"] >= target_recall
    ]
    pool = feasible if feasible else rows
    best = max(pool, key=lambda r: (r["f1"], r["precision"] + r["recall"], -abs(r["threshold"] - 0.5)))
    best["selection_rule"] = (
        f"validation_only: precision>={target_precision:.0%} and "
        f"recall>={target_recall:.0%}, maximize F1; fallback=max F1"
        if feasible else
        "validation_only: target infeasible, maximize F1"
    )
    return best


def train_return_risk_model(
    canonical_df: pd.DataFrame,
    target_col: str = "return_event",
    compare_lightgbm: bool = True,
) -> TrainedModelBundle:
    # LightGBM is an optional dependency -- some sandboxes (and some
    # minimal production images) won't have it installed. Rather than
    # hard-crash, fall back to logistic-regression-only training and
    # let the caller/report note which models were actually compared.
    compare_lightgbm = compare_lightgbm and LIGHTGBM_AVAILABLE
    df = canonical_df.sort_values("order_date").reset_index(drop=True)
    X, y, feature_names = build_features(df, target_col=target_col, for_training=True)

    # re-derive the same row set (rows with known target) with dates,
    # so temporal_split operates on the identical row ordering as X/y
    labeled = df[df[target_col].notna()].reset_index(drop=True)
    assert len(labeled) == len(X)

    train_df, val_df, test_df, bounds = temporal_split(labeled)
    n_train, n_val = len(train_df), len(val_df)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train:n_train + n_val], y.iloc[n_train:n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]

    candidates = {}

    lr_model, lr_scaler = _fit_logistic_regression(X_train, y_train)
    p_val_lr = lr_model.predict_proba(lr_scaler.transform(X_val))[:, 1]
    candidates["logistic_regression"] = {
        "model": lr_model, "scaler": lr_scaler,
        "val_auc": roc_auc_score(y_val, p_val_lr) if y_val.nunique() > 1 else float("nan"),
        "val_ap": average_precision_score(y_val, p_val_lr) if y_val.nunique() > 1 else float("nan"),
        "p_val": p_val_lr,
    }

    if compare_lightgbm:
        gbm_model, _ = _fit_lightgbm(X_train, y_train)
        p_val_gbm = gbm_model.predict_proba(X_val)[:, 1]
        candidates["lightgbm"] = {
            "model": gbm_model, "scaler": None,
            "val_auc": roc_auc_score(y_val, p_val_gbm) if y_val.nunique() > 1 else float("nan"),
            "val_ap": average_precision_score(y_val, p_val_gbm) if y_val.nunique() > 1 else float("nan"),
            "p_val": p_val_gbm,
        }

    # model selection on validation AUC (ties broken toward the
    # simpler, more interpretable logistic regression)
    best_name = max(candidates, key=lambda k: (round(candidates[k]["val_auc"], 4),
                                                 1 if k == "logistic_regression" else 0))
    best = candidates[best_name]

    threshold_info = _select_threshold(y_val.values, best["p_val"])

    bundle = TrainedModelBundle(
        model_name=best_name,
        model=best["model"],
        scaler=best["scaler"],
        feature_names=feature_names,
        threshold=threshold_info["threshold"],
        split_bounds=bounds,
        val_metrics_at_candidates={k: {"val_auc": v["val_auc"], "val_ap": v["val_ap"]} for k, v in candidates.items()},
    )
    bundle._val_threshold_selection = threshold_info  # stashed for reporting
    bundle._test_df = test_df
    bundle._test_X = X_test
    bundle._test_y = y_test
    bundle._val_df = val_df
    bundle._val_X = X_val
    bundle._val_y = y_val
    bundle._train_df = train_df
    return bundle
