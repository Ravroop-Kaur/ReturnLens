"""
Financial exposure calculations.

Terminology is deliberately conservative:

- "Predicted return exposure": the transaction value of orders the
  model currently classifies as high risk (whether or not they end
  up returning). This is a FORWARD-LOOKING estimate, not a fact.

- "Observed historical return value": the transaction value of
  orders in the held-out set that ACTUALLY returned. This is
  descriptive of the past, not a forecast.

- "False-positive exposure": transaction value of orders flagged
  high-risk that did NOT actually return (the cost of an unnecessary
  intervention, if the merchant acts on every flag).

- "False-negative exposure": transaction value of orders that DID
  return but were NOT flagged (the cost of a missed detection).

We never call any of these "revenue lost", "savings", "profit", or
"ROI" -- none of those claims are supportable from order amount
alone.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ExposureReport:
    predicted_return_exposure: float
    observed_return_value: float
    false_positive_exposure: float
    false_negative_exposure: float
    n_high_risk_orders: int
    n_total_orders: int
    pct_orders_high_risk: float

    def to_dict(self) -> dict:
        return {
            "predicted_return_exposure": round(self.predicted_return_exposure, 2),
            "observed_historical_return_value": round(self.observed_return_value, 2),
            "false_positive_exposure": round(self.false_positive_exposure, 2),
            "false_negative_exposure": round(self.false_negative_exposure, 2),
            "n_high_risk_orders": self.n_high_risk_orders,
            "n_total_orders": self.n_total_orders,
            "pct_orders_high_risk": round(self.pct_orders_high_risk, 4),
            "note": "Predicted exposure is an estimate, not a guaranteed loss. "
                    "See docs for exact definitions.",
        }


def compute_exposure(
    amount: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> ExposureReport:
    amount = pd.Series(amount).reset_index(drop=True)
    y_true = pd.Series(y_true).reset_index(drop=True).astype(int)
    y_pred = pd.Series(y_pred).reset_index(drop=True).astype(int)

    predicted_return_exposure = float(amount[y_pred == 1].sum())
    observed_return_value = float(amount[y_true == 1].sum())

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    false_positive_exposure = float(amount[fp_mask].sum())
    false_negative_exposure = float(amount[fn_mask].sum())

    n_high_risk = int((y_pred == 1).sum())
    n_total = len(y_pred)

    return ExposureReport(
        predicted_return_exposure=predicted_return_exposure,
        observed_return_value=observed_return_value,
        false_positive_exposure=false_positive_exposure,
        false_negative_exposure=false_negative_exposure,
        n_high_risk_orders=n_high_risk,
        n_total_orders=n_total,
        pct_orders_high_risk=(n_high_risk / n_total if n_total else 0.0),
    )
