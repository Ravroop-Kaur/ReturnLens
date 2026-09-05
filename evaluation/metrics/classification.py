"""
Standard classification evaluation metrics for the return-risk
detector, computed on a frozen held-out set at a frozen threshold.

All formulas match the spec exactly:

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * Precision * Recall / (Precision + Recall)
    FPR       = FP / (FP + TN)
    FNR       = FN / (FN + TP)

Every denominator is checked explicitly; a zero denominator produces
a documented "not defined" result rather than a crash or a silently
wrong 0.0.

Class semantics (fixed, never reversed):
    Positive class = RETURN     (y == 1)
    Negative class = NO_RETURN  (y == 0)
    TP = actual return   & predicted return
    FP = actual no-return & predicted return
    FN = actual return   & predicted no-return
    TN = actual no-return & predicted no-return

This module intentionally computes everything from y_true/y_pred at
one frozen threshold -- nothing here selects, tunes, or changes a
production threshold. See evaluate_thresholds() for the separate,
explicitly diagnostic-only multi-threshold utility.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional

POSITIVE_CLASS = "return"
NEGATIVE_CLASS = "no_return"


@dataclass
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


@dataclass
class ClassificationMetrics:
    counts: ConfusionCounts
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    precision_defined: bool
    recall_defined: bool
    fpr_defined: bool
    fnr_defined: bool
    n_positive: int
    n_negative: int
    n_total: int
    # Metadata, additive only -- optional so every existing call site
    # (evaluate(y_true, y_proba, threshold) with no extra args) keeps
    # working unchanged. positive_class/negative_class are fixed, not
    # configurable, so class semantics can never be silently reversed.
    positive_class: str = POSITIVE_CLASS
    negative_class: str = NEGATIVE_CLASS
    dataset_type: Optional[str] = None
    model_version: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Additive convenience aliases for API consumers (spec PART A9 /
        # section 14): a nested `confusion_matrix` object alongside the
        # existing flat `counts` key (kept for backward compatibility
        # with the dashboard/pipeline, which already read `counts`).
        d["confusion_matrix"] = self.counts.to_dict()
        d["test_size"] = self.n_total
        return d

    def to_api_dict(self) -> dict:
        """The exact structured shape requested for API exposure: no
        formatted-text parsing required on the client side, and every
        value read straight from this same object -- never a second,
        separately-maintained calculation."""
        return {
            "dataset_type": self.dataset_type,
            "positive_class": self.positive_class,
            "negative_class": self.negative_class,
            "threshold": self.threshold,
            "test_size": self.n_total,
            "model_version": self.model_version,
            "confusion_matrix": self.counts.to_dict(),
            "metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "fpr": self.fpr,
                "fnr": self.fnr,
            },
        }


def _safe_div(numerator: float, denominator: float):
    if denominator == 0:
        return None, False
    return numerator / denominator, True


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionCounts:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    # Sample-accounting invariants (spec section 5). These are
    # mathematically guaranteed by construction (each example falls
    # into exactly one of the four boolean-and masks above), so they
    # can never fail -- they exist as an explicit, permanent guard
    # against a future refactor accidentally breaking that guarantee,
    # not because a failure is expected today.
    n = len(y_true)
    assert tp + fp + fn + tn == n, "confusion matrix counts must sum to the number of test examples"
    assert tp + fn == int((y_true == 1).sum()), "TP + FN must equal the actual positive count"
    assert fp + tn == int((y_true == 0).sum()), "FP + TN must equal the actual negative count"
    assert tp + fp == int((y_pred == 1).sum()), "TP + FP must equal the predicted positive count"
    assert fn + tn == int((y_pred == 0).sum()), "FN + TN must equal the predicted negative count"

    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    dataset_type: Optional[str] = None,
    model_version: Optional[str] = None,
) -> ClassificationMetrics:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)

    precision, precision_defined = _safe_div(counts.tp, counts.tp + counts.fp)
    recall, recall_defined = _safe_div(counts.tp, counts.tp + counts.fn)
    fpr, fpr_defined = _safe_div(counts.fp, counts.fp + counts.tn)
    fnr, fnr_defined = _safe_div(counts.fn, counts.fn + counts.tp)

    if precision_defined and recall_defined and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (precision_defined and recall_defined) else None

    n_positive = int((np.asarray(y_true) == 1).sum())
    n_negative = int((np.asarray(y_true) == 0).sum())

    return ClassificationMetrics(
        counts=counts,
        threshold=float(threshold),
        precision=precision if precision_defined else float("nan"),
        recall=recall if recall_defined else float("nan"),
        f1=f1 if f1 is not None else float("nan"),
        fpr=fpr if fpr_defined else float("nan"),
        fnr=fnr if fnr_defined else float("nan"),
        precision_defined=precision_defined,
        recall_defined=recall_defined,
        fpr_defined=fpr_defined,
        fnr_defined=fnr_defined,
        n_positive=n_positive,
        n_negative=n_negative,
        n_total=len(y_true),
        dataset_type=dataset_type,
        model_version=model_version,
    )


def evaluate_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> list[ClassificationMetrics]:
    """Diagnostic-only multi-threshold sweep (spec section 12).

    Returns one ClassificationMetrics per threshold so it's possible to
    see later whether poor performance is (A) a threshold-selection
    problem -- precision/recall trade off sharply across thresholds --
    or (B) a model-discrimination problem -- performance stays poor at
    every threshold. This function does NOT select, recommend, or set
    a production threshold; the model's existing frozen threshold is
    untouched by calling this.
    """
    return [evaluate(y_true, y_proba, t) for t in thresholds]
