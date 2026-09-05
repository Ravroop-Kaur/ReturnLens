"""
Runs the full PREDICT -> EXPLAIN -> QUANTIFY -> ACT -> VERIFY pipeline
on a merchant dataset (defaults to the synthetic demo dataset) and
writes a single JSON report consumed by the merchant-facing UI.

This is the "real-data model evaluation" referenced in the spec: the
model is trained and evaluated on a genuine temporal held-out portion
of whichever dataset is passed in. If that dataset is the synthetic
demo file, the report is clearly labeled as demo/synthetic data, not
a real merchant result.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.adapters.generic_csv import ingest_generic_csv
from src.model.train import train_return_risk_model
from src.features.engineering import build_features
from evaluation.metrics.classification import evaluate
from src.exposure.financial import compute_exposure
from src.diagnosis.statistical import run_full_diagnosis, top_finding
from src.recommendation.engine import recommend
from src.verification.simulate import simulate_intervention

REPO_ROOT = Path(__file__).resolve().parents[2]


def feature_importance_report(bundle) -> list:
    if bundle.model_name == "logistic_regression":
        coefs = bundle.model.coef_[0]
        pairs = sorted(zip(bundle.feature_names, coefs), key=lambda x: -abs(x[1]))
        return [{"feature": f, "weight": float(w), "direction": "increases risk" if w > 0 else "decreases risk"}
                for f, w in pairs[:10]]
    else:
        importances = bundle.model.feature_importances_
        pairs = sorted(zip(bundle.feature_names, importances), key=lambda x: -x[1])
        total = sum(importances) or 1
        return [{"feature": f, "weight": float(w) / float(total), "direction": "contributes to risk"}
                for f, w in pairs[:10]]


def explain_order(bundle, X_row: pd.Series, top_k: int = 3) -> list:
    """Very small, honest local explanation: which of this order's
    active features have the largest model weight (LR) or largest
    global importance (GBM) among features that are actually "on"
    for this row. Not a full SHAP explanation -- documented as such."""
    if bundle.model_name == "logistic_regression":
        coefs = dict(zip(bundle.feature_names, bundle.model.coef_[0]))
    else:
        coefs = dict(zip(bundle.feature_names, bundle.model.feature_importances_))

    contributions = []
    for feat in bundle.feature_names:
        val = X_row.get(feat, 0)
        if isinstance(val, (int, float, np.floating, np.integer)) and val not in (0, False):
            weight = coefs.get(feat, 0)
            contributions.append((feat, weight * (val if abs(val) <= 1 else 1)))
    contributions.sort(key=lambda x: -abs(x[1]))
    return [c[0] for c in contributions[:top_k]]


def run(csv_path: str, out_path: str, dataset_label: str):
    ingestion = ingest_generic_csv(csv_path)
    caps = ingestion.capabilities

    report = {
        "dataset_label": dataset_label,
        "ingestion": {
            "n_rows_raw": ingestion.n_rows_raw,
            "capabilities": caps.summary(),
            "message": ingestion.message(),
        },
    }

    if not ingestion.can_train_detector():
        report["status"] = "cannot_train"
        Path(out_path).write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        return report

    df = ingestion.canonical_df
    bundle = train_return_risk_model(df)

    X_test, y_test, test_df = bundle._test_X, bundle._test_y, bundle._test_df
    p_test = bundle.predict_proba(X_test)
    y_pred_test = (p_test >= bundle.threshold).astype(int)

    metrics = evaluate(y_test.values, p_test, bundle.threshold, dataset_type="synthetic_holdout")
    exposure = compute_exposure(test_df["amount"], y_test, y_pred_test)

    diagnosis_results = run_full_diagnosis(test_df.assign(return_event=y_test.values))
    best_finding = top_finding(diagnosis_results)
    recommendation = recommend(best_finding)

    verification = None
    if best_finding is not None:
        seg_mask = test_df[best_finding.dimension].astype(str) == best_finding.segment
        seg_n = int(seg_mask.sum())
        seg_returns = int(y_test.values[seg_mask.values].sum())
        if seg_n >= 20:
            verification = simulate_intervention(before_rate=seg_returns / seg_n, before_n=seg_n)

    report.update({
        "status": "ok",
        "model": {
            "model_name": bundle.model_name,
            "threshold": bundle.threshold,
            "feature_count": len(bundle.feature_names),
            "val_model_comparison": bundle.val_metrics_at_candidates,
            "threshold_selection_on_validation": bundle._val_threshold_selection,
            "split": {
                "train_n": len(bundle._train_df),
                "val_n": len(bundle._val_df),
                "test_n": len(test_df),
                "train_end": str(bundle.split_bounds.train_end),
                "val_end": str(bundle.split_bounds.val_end),
                "test_end": str(bundle.split_bounds.test_end),
            },
            "random_seed": bundle.random_seed,
        },
        "test_evaluation": metrics.to_dict(),
        "financial_exposure": exposure.to_dict(),
        "feature_importance": feature_importance_report(bundle),
        "diagnosis": {
            dim: [f.__dict__ for f in findings[:5]] for dim, findings in diagnosis_results.items()
        },
        "top_finding": best_finding.__dict__ if best_finding else None,
        "recommendation": recommendation.__dict__ if recommendation else None,
        "verification": verification.__dict__ if verification else None,
    })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"Report written to {out_path}")
    print(json.dumps(report["test_evaluation"], indent=2, default=str))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(REPO_ROOT / "data/sample/generic_merchant_orders.csv"))
    parser.add_argument("--out", default=str(REPO_ROOT / "evaluation/reports/full_report.json"))
    parser.add_argument("--label", default="SYNTHETIC DEMO DATA -- not a real merchant")
    args = parser.parse_args()
    run(args.csv, args.out, args.label)
