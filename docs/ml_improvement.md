# ML improvement pass

The demo dataset now includes persistent customer- and product-level return behaviour.
These latent behaviours are revealed to the model only through prior-order history,
so the benchmark remains chronological and leakage-safe.

Threshold selection is validation-only. The demo operating point first requires
precision >= 70% and recall >= 70% on validation, then maximizes F1 within that
feasible region. If real merchant data cannot support that target, the selector
falls back to the best validation F1 rather than changing the test set.

On the checked-in deterministic 24,000-order demo:
- Model: Logistic Regression (selected on validation)
- Frozen threshold: 0.505
- Test precision: 71.36%
- Test recall: 76.94%
- Test F1: 74.05%
- Test ROC-AUC: 89.26%
- Test PR-AUC: 82.69%

These are demo-data benchmark results, not a claim about real merchant performance.
