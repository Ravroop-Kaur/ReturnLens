# Leakage Notes

This document records every field that was considered as a feature
and either included or explicitly rejected, with the reasoning.

## Rejected outright (target / outcome fields)

These are never used as model features under any circumstances:

- `return_event` -- this IS the target.
- `return_date` -- only exists once a return has happened.
- `refund_event` -- lags/co-occurs with the outcome; not known at
  order time.
- `chargeback_event` -- a different loss class, and like the above,
  only known after the fact.

## Fields interpreted only to construct the label (adapters)

- Amazon-style `Status` column (e.g. "Shipped - Returned to seller",
  "Shipped - Delivered to Buyer"). This column mixes fulfilment
  progress and outcome state. We use it ONLY inside
  `src/adapters/amazon_adapter.py` to construct `return_event`, and
  never expose it (raw or transformed) as a feature. Rows whose
  status is not clearly terminal (returned vs. not-returned) are left
  unlabeled (`NaN`) rather than guessed.

## Historical / aggregate features: leave-one-out required

Product-level, category-level, fulfilment-level, region-level and
shipping-level historical return rates are extremely predictive in
naive implementations because a naive "return rate of category X"
computed over the WHOLE dataset already includes each row's own
outcome. For a category with few orders, a single row can shift the
computed rate enough that the model is effectively told the answer.

To prevent this, `src/features/engineering.py::_expanding_prior_rate`
computes, for every row, the smoothed historical rate of its group
using ONLY rows with a strictly earlier `order_date`. The first order
of any group has no history yet, so it receives the smoothed global
prior, not its own outcome.

This was verified with automated tests in `tests/test_leakage.py`,
including a test that plants a single return far in the future of a
category's history and confirms it has zero effect on an earlier
row's feature value.

## Fields deliberately excluded

- `review_text` -- present in the canonical schema for completeness,
  but not used as a model feature in this MVP. Review text written
  after a purchase can itself reflect the return decision (e.g. a
  customer explains why they are returning something), which would
  be leakage if the review timestamp cannot be reliably shown to
  precede the return. Rather than build an unreliable "review must
  predate order X days" filter for the hackathon MVP, this field is
  excluded entirely. This is a documented limitation, not an
  oversight.
- `payment_status` at prediction time is included as a categorical
  feature, but any late-arriving payment status change (e.g. status
  updated after a return is initiated) would be leakage if it were
  re-fetched at report time instead of captured at order time. The
  generic ingestion path assumes the CSV represents the payment
  status as of order placement; this assumption is documented and
  should be confirmed with any new merchant data source.
