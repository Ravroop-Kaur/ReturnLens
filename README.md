# ReturnLens
Explainable Return Risk Intelligence
# AI Merchant Return-Risk Manager

**Razorpay AI Buildathon 2026 -- Track 02: AI Risk Manager**

| | |
|---|---|
| Track | 02 -- AI Risk Manager |
| Loss class | Returns |
| Detector | Supervised ML return-risk scorer (Logistic Regression, compared against LightGBM) |
| Primary evaluation | Held-out test-set precision and recall |
| Additional evaluation | F1, FPR, FNR, confusion matrix |
| Business evaluation | False-positive and false-negative financial exposure |
| Diagnosis | Statistical association engine, separate from the ML model |
| Action | Bounded, human-actioned recommendation |
| Verification | Simulated post-intervention before/after comparison |
| Defense-only | **Yes.** No autonomous refunds, courier changes, or payment actions. |

> **Important honesty note.** No real merchant dataset was provided for this
> build. The dataset used for development and for every number in this
> README (`data/sample/generic_merchant_orders.csv`) is **synthetically
> generated** (see `data/sample/generate_sample_data.py`) with a known,
> documented risk-generating process, specifically so that precision/recall
> measure genuine detection ability against a real (if synthetic) signal,
> rather than being fabricated. Every result below is labelled accordingly.
> Section 24 of the brief ("real-data model evaluation") could not be
> performed because no real dataset was supplied -- this is disclosed here
> rather than papered over.

---

## 1. What problem are we solving?

Merchants lose money not just to fraud, but to a steady, often-ignored
drain: **orders that come back**. Returns tie up working capital, cost
shipping twice, and are frequently concentrated in a few identifiable
segments (a fulfilment partner, a shipping method, a product category)
rather than being spread evenly. This product predicts which orders are
likely to be returned, explains why, quantifies the exposure, and
recommends a bounded, human-reviewed action.

## 2. Why returns (and not fraud or chargebacks)?

The brief explicitly asks for **one** loss class, done well, rather than
several done superficially. Returns were chosen because:
- they are the loss class best supported by a single, common e-commerce
  export format (order + eventual return outcome),
- the detection problem is genuinely learnable from pre-outcome order
  attributes (category, fulfilment, shipping, amount, timing),
- the statistical diagnosis and financial exposure story is easy for a
  non-technical merchant to understand end-to-end.

## 3. What does the ML model predict?

For each order, at the moment it is placed, the model predicts:

**"Will this order eventually result in a return?"**

Output: a probability `P(return)`, a HIGH/LOW risk label (via a frozen
threshold), and a binary prediction.

## 4. What data does it use?

The canonical schema (`src/canonical/schema.py`) is merchant-agnostic. A
generic merchant CSV is mapped into it via `src/adapters/generic_csv.py`
(the **primary** ingestion path); an isolated `src/adapters/amazon_adapter.py`
demonstrates that a real-world Amazon-style export (Order ID / Date /
Status / Fulfilment / Category / Amount / ship-state / ship-service-level)
can be mapped into the exact same canonical form. Amazon is a source
adapter, never a hard-coded assumption -- confirmed by
`tests/test_ingestion.py::test_amazon_is_only_a_source_adapter_not_required`.

Features used by the model (see section 10 below for the full list):
order amount, category, fulfilment method, shipping service, region,
order timing (day of week / month / hour), and leakage-safe historical
return rates for product / category / fulfilment / region / shipping
service.

## 5. What information is excluded due to leakage?

`return_event`, `return_date`, `refund_event`, and `chargeback_event` are
**never** used as features -- only as the training target or for
post-hoc evaluation. `review_text` is excluded entirely from features in
this MVP because review timestamps relative to the return event cannot be
reliably verified from the available data (a review written to explain a
return would leak the outcome). Full reasoning and automated leakage
tests are in `docs/leakage_notes.md` and `tests/test_leakage.py`.

Historical group return-rate features (e.g. "this category's typical
return rate") are computed using an **as-of information window**. A
historical return is usable only after its return timestamp is known;
a historical no-return is usable only after its observation window has
closed. The current order's outcome and any future outcome are excluded.
This is stricter than merely shifting by order date and is covered by
leakage/maturity tests.

## 6. How is the model evaluated?

Chronological (time-aware) split: earliest 60% of orders = train, next
20% = validation, latest 20% = **frozen** test. The classification
threshold is chosen on validation only (minimizing an explicit,
documented cost function that weights a missed return 3x an unnecessary
flag -- an assumption, not a measured constant) and then frozen before
the test set is touched even once.

## 7. What are precision and recall?

- **Precision** = of the orders we flagged high-risk, what fraction
  actually returned? High precision means fewer wasted reviews.
- **Recall** = of the orders that actually returned, what fraction did we
  catch? High recall means fewer missed returns.

Both matter, and they trade off against each other via the threshold.

## 8. What is the held-out test set?

The most recent 20% of orders by date (test set ends 2025-12-31 in the
demo run), **never used** for feature selection, model selection, or
threshold tuning.

## 9. What are the actual results?

*(Demo/synthetic dataset -- see honesty note above. Reproduce with
`python -m evaluation.reports.run_full_pipeline`.)*

| Metric | Value |
|---|---|
| Model selected | Logistic Regression (validation ROC-AUC 0.883 vs LightGBM 0.874) |
| Threshold (frozen on validation) | 0.505 |
| Test set size | 4,800 orders (1,665 actual returns, 3,135 non-returns) |
| TP / FP / TN / FN | 1,248 / 454 / 2,681 / 417 |
| Precision | 0.733 |
| Recall | 0.750 |
| F1 | 0.741 |
| False Positive Rate | 0.145 |
| False Negative Rate | 0.250 |

### Synthetic-data boundary

The current benchmark and model metrics are generated from synthetic merchant data. They demonstrate the pipeline and evaluation method; they are not a guarantee of performance on real merchant data. Real-merchant results are reported separately when labelled, mature data is available.

These metrics are the result of the **stricter point-in-time feature
implementation** in this version. Historical return-rate features may
only use labels whose outcomes would already have been known at the
prediction timestamp: a positive return must have a recorded
`return_date`, while a negative label is usable only after the
configured return-observation window closes. This avoids label-maturity
leakage.

The result is intentionally reported without guaranteeing a target
precision/recall range. Performance on a real merchant may be higher or
lower depending on label quality, history depth, feature availability,
and distribution drift.

## 10. What does the statistical diagnosis do?

Separately from the ML model, `src/diagnosis/statistical.py` checks each
canonical dimension (fulfilment method, category, region, shipping
service) for segments whose return rate is both **statistically
supported** (a Wilson confidence interval for the segment sits entirely
above the baseline rate) and **practically significant** (relative risk
>= 1.3x), ignoring any segment with fewer than 30 orders. In the demo
run it correctly surfaces `fulfilment_method = third_party_fulfilled`
(30.9% observed return rate vs 14.3% baseline, ~2.2x, 947 orders) -- which
matches the effect actually planted in the synthetic generator, confirming
the diagnosis engine recovers real signal rather than noise.

## 11. How is financial exposure calculated?

`src/exposure/financial.py` computes, using order amount as the value
basis:
- **Predicted return exposure**: transaction value of orders currently
  flagged high-risk (forward-looking estimate, not a fact).
- **Observed historical return value**: transaction value of orders that
  actually returned in the evaluated period (descriptive of the past).
- **False-positive exposure** / **false-negative exposure**: transaction
  value tied to each error type.

We never use the words "revenue lost", "savings", "profit", or "ROI" --
enforced by an automated test
(`tests/test_financial_exposure.py::test_exposure_terminology_never_claims_savings`).

## 12. How does intervention verification work?

`src/verification/simulate.py` runs a two-proportion z-test comparing a
before-rate to an after-rate. In this MVP the "after" data is a
**simulated** sample (documented, explicit assumed relative reduction),
clearly labelled `DEMO / SYNTHETIC SIMULATION` everywhere it is shown.
The exact same function works unmodified on real before/after data if a
merchant ever supplies it -- only the label changes.

## 12A. Diagnosis benchmark size

The statistical diagnosis benchmark contains **600 independent scenarios**
across six scenario families. The family-stratified split produces **204
frozen test scenarios** (34 per family), with 198 scenarios in dev/validation
combined. This benchmark measures diagnosis reliability, not the ML order-level
return predictor, and its results are explicitly synthetic.

## 13. What is synthetic vs real?

- **Synthetic**: the entire demo dataset (`data/sample/generic_merchant_orders.csv`),
  the benchmark scenarios (`evaluation/benchmark/scenarios.py`), and the
  post-intervention "after" sample in verification.
- **Real (if supplied)**: the generic CSV / Amazon adapter ingestion path,
  the leakage-safe feature engineering, the model training and evaluation
  code, and the statistical diagnosis engine all operate identically on
  real merchant data -- nothing about them is specific to the synthetic
  generator.

## 14. What are the limitations?

- No real merchant dataset was available for this build; all reported
  numbers are on synthetic data with a known, documented generating
  process (see honesty note above).
- The demo metrics are synthetic and are not evidence that the same
  performance will hold on real merchants. The operating threshold is
  selected on validation only; the later held-out test set is used once
  for final reporting. Real merchant deployment should be gated by
  merchant-specific held-out evaluation and can abstain when data is
  insufficient.
- The `hist_return_rate_category` coefficient has a counter-intuitive
  (negative) sign in the fitted logistic regression, most likely due to
  collinearity with the category one-hot features. This is disclosed
  rather than hidden; a production version would likely drop the raw
  category one-hots when the historical-rate feature is present, or use
  a model less sensitive to collinearity.
- The local "explanation" in the report is feature-weight-based, not a
  full SHAP explanation; it is documented as an approximation.
- Multi-tenancy is enforced server-side by the in-memory tenant store and
  authenticated organization identity. Persistence is the remaining
  production upgrade.
- The Razorpay Test Mode adapter is implemented as a deliberately small
  real read-only flow: Test Mode credentials are verified against the real
  Razorpay API, payments/refunds can be imported, and a signed webhook
  endpoint can update the merchant event stream. True return labels still
  require an OMS/returns source because Razorpay payments do not define
  an order-return outcome.

## 15. What is future work?

- Evaluate on a real merchant export via the same generic CSV path.
- Expand the small Razorpay Test Mode flow only if the demo needs more
  payment/refund event types; keep it read-only and source-isolated.
- Expand the diagnosis engine to two-way interactions (e.g. fulfilment x
  shipping) once enough real segment-level sample size exists.
- Calibrate the FP/FN cost ratio used for threshold selection against a
  merchant's actual, disclosed costs instead of the current documented
  assumption (3x).

---

## Architecture

```
MERCHANT DATA (generic CSV | Amazon-style CSV)
        |
CANONICAL SCHEMA  (src/canonical)
        |
FEATURE ENGINEERING (src/features, leakage-safe, expanding windows)
        |
   +----+-----------------------+
   |                            |
ML RETURN-RISK SCORER   STATISTICAL DIAGNOSIS
(src/model)             (src/diagnosis)
   |                            |
   +----------+-----------------+
              |
      FINANCIAL EXPOSURE (src/exposure)
              |
      BOUNDED RECOMMENDATION (src/recommendation)
              |
      INTERVENTION (simulated)
              |
      VERIFICATION (src/verification)
              |
      MERCHANT-FACING UI (app/ui) + TECHNICAL EVIDENCE (secondary)
```

## Project layout

```
return-risk-manager/
├── app/ui/                    merchant-facing static HTML dashboard generator
├── src/
│   ├── canonical/              schema + column-mapping/validation
│   ├── adapters/                generic CSV (primary) + Amazon-style (isolated)
│   ├── features/                 leakage-safe feature engineering
│   ├── model/                     train/val/test split + LR/LightGBM training
│   │                               + registry.py (model scope decision + version ledger)
│   ├── quality/                    data readiness pipeline (dedup, order-level
│   │                               aggregation, feature contract, label lifecycle,
│   │                               drift monitoring)
│   ├── diagnosis/                  statistical association engine
│   ├── exposure/                    financial exposure calculations
│   ├── recommendation/               bounded recommendation engine
│   ├── verification/                  before/after simulation + real-data-ready z-test
│   ├── auth/                           lightweight login/session service
│   ├── tenancy/                         tenant-scoped storage
│   ├── connectors/                       MerchantDataConnector + mock/csv/razorpay
│   ├── integrations/razorpay/             isolated Razorpay client/config/mapper/webhook
│   ├── claims/                             return-claim model + evidence aggregation
│   ├── evidence/                            image-evidence analyzer (secondary signal)
│   ├── pipeline.py                          orchestration: auth -> readiness -> model
│   │                                        or abstain -> explain -> quantify -> act -> verify
│   └── api/app.py                            minimal Flask app (login, protected routes,
│                                              Razorpay webhook receiver)
├── evaluation/
│   ├── metrics/                classification metrics (precision/recall/F1/FPR/FNR)
│   ├── benchmark/                independent synthetic scenario harness
│   └── reports/                    full pipeline orchestrator + JSON output
├── data/sample/                synthetic demo dataset + generator + Amazon-style sample
├── docs/leakage_notes.md      full leakage reasoning
├── tools/                      minirunner.py -- DEV-ONLY fallback test runner
│                                sandboxes with no network access to install real pytest
├── tests/                      automated tests (ingestion, leakage, model, metrics,
│                                benchmark, exposure, diagnosis, verification, data
│                                quality/readiness, drift, auth, tenancy, connectors,
│                                claims/evidence, image evidence, Razorpay, API, e2e)
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# (re)generate the synthetic demo dataset
python data/sample/generate_sample_data.py

# run the full PREDICT -> EXPLAIN -> QUANTIFY -> ACT -> VERIFY pipeline
python -m evaluation.reports.run_full_pipeline

# generate the merchant-facing dashboard from the resulting report
python app/ui/generate_dashboard.py
# open app/ui/dashboard.html in a browser

# run the independent synthetic benchmark for the diagnosis engine
python -m evaluation.benchmark.run_benchmark

# run all automated tests (real pytest, when network/pip access is available)
pytest tests/ -q

# run the demo API (login, protected risk endpoint, Razorpay webhook receiver)
python -m src.api.app
# no Razorpay credentials needed -- src.connectors.mock backs /risk/latest until
# a real connector is configured, and src.integrations.razorpay.client falls back
# to a clearly-labelled demo payload when RAZORPAY_KEY_ID/KEY_SECRET are unset.
```

## Defense-only confirmation

This project contains no functionality capable of executing a real-world
action against a merchant account, payment, courier, or customer. All
"interventions" are simulated and explicitly labelled. There is no
offense-capable functionality anywhere in this codebase. Razorpay webhooks
are DATA INGESTION ONLY -- they update stored risk information and never
trigger a refund, block, cancellation, or configuration change.

---

## Part 2 — Real-merchant-readiness alteration pass

The sections above describe the original hackathon submission (synthetic
data, single-tenant, no login, no external connectors). Everything below
documents what was added on top of it to make the architecture
real-merchant-ready, organized the way any reviewer should read it: what
actually works today, what is demo/mocked, and what is explicitly out of
scope for this MVP.

### CURRENTLY IMPLEMENTED

- **Data readiness pipeline** (`src/quality/readiness.py`): runs duplicate
  analysis, order-level aggregation, an explicit feature contract
  (REQUIRED / RECOMMENDED / OPTIONAL / NOT_AVAILABLE / NOT_USABLE), label
  lifecycle (RETURNED / NO_RETURN / PENDING), date/numeric validity, and
  history-span checks, then decides `READY` vs `NOT_READY` with itemized
  reasons. Nothing is fabricated: a missing/unusable field is reported as
  such, never silently imputed with a guessed value.
- **Duplicate handling** (`src/quality/dedup.py`): distinguishes exact
  duplicate rows (safe to drop), legitimate multi-line orders (never
  collapsed), and genuinely conflicting records on the same order+product
  (surfaced, never silently resolved).
- **Order-level aggregation** (`src/quality/order_level.py`): line-item
  data is aggregated to one row per order (amount summed, outcome fields
  OR'd, dates min/maxed, everything else kept only on agreement) before
  training/evaluation ever sees it, so a 3-item order is never
  triple-counted.
- **Label lifecycle** (`src/quality/lifecycle.py`): a recent order whose
  return window hasn't closed is `PENDING`, never coerced into
  `NO_RETURN`. Only finalized labels are used for supervised
  training/evaluation (`usable_for_supervision`).
- **Model scope decision + abstention** (`src/model/registry.py`):
  `decide_model_scope` chooses `MERCHANT_SPECIFIC`, `GLOBAL_BASELINE`, or
  `INSUFFICIENT_DATA` (abstain) purely from the readiness report's
  numbers. The pipeline abstains (never trains/predicts) whenever
  required fields are missing/unusable or there isn't enough labelled
  history -- and always persists a `status: "abstained"` result so the
  dashboard has something honest to show.
- **Model version registry**: every trained model version (scope, feature
  set, threshold, row counts, demo/real flag) is appended to an
  in-memory, per-organization ledger -- never overwritten.
- **Drift monitoring** (`src/quality/drift.py`): compares a reference
  window against a current window on return prevalence, order-amount
  distribution, categorical distributions, and missingness, reporting
  `NORMAL` / `MILD_DRIFT` / `SIGNIFICANT_DRIFT` per signal. It only
  reports -- it never triggers an automatic retrain.
- **Login + sessions** (`src/auth/service.py`): PBKDF2-HMAC-SHA256
  password hashing (no plaintext, ever), opaque bearer session tokens,
  login/logout, and `require_session` for protecting routes.
- **Multi-tenancy** (`src/tenancy/store.py`): every persisted record
  (risk results, claims, evidence, model versions) is namespaced by
  `organization_id` at the data-layer level -- there is no "read
  everything" API a caller could misuse to cross tenants. Enforced
  server-side, never by frontend discipline.
- **Merchant connector abstraction** (`src/connectors/base.py`) with
  three implementations: `MockMerchantConnector` (bundled synthetic demo
  data), `CSVMerchantConnector` (manual import -- the fallback path, not
  the product), and `RazorpayConnector` (real API or demo-mode fallback).
  None of them require a merchant's production database admin password.
- **Razorpay integration** (`src/integrations/razorpay/`): an isolated
  client/config/mapper package. Credentials come only from environment
  variables (`RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`,
  `RAZORPAY_WEBHOOK_SECRET`); Test Mode keys (`rzp_test_...`) are
  detected automatically. The ML/business logic never sees a
  Razorpay-specific field name -- only the canonical schema.
- **Razorpay webhook receiver** (`src/integrations/razorpay/webhook.py`,
  wired at `POST /webhooks/razorpay` in `src/api/app.py`): validates the
  HMAC-SHA256 signature against the raw request body, rejects invalid
  signatures, and is idempotent per organization (a redelivered event ID
  is recorded as `duplicate` and never reprocessed). Webhooks are DATA
  INGESTION ONLY.
- **Return-claim evidence layer** (`src/claims/`): a `ReturnClaim`
  representation, controlled-taxonomy reason normalization (ambiguous
  reasons are left unnormalized rather than force-fit), and
  `aggregate_evidence`, which outputs one of `SUPPORTED` / `NEEDS_REVIEW`
  / `SUSPICIOUS` / `INDETERMINATE` -- never a fraud verdict -- with a
  disclaimer attached to every result.
- **Image evidence analyzer** (`src/evidence/image_analyzer.py`): a
  provider abstraction (`ImageEvidenceAnalyzer`) plus a deterministic,
  clearly-labelled `MockImageEvidenceAnalyzer` (`is_demo=True`, no
  accuracy claim anywhere in its output). Output is `NORMAL` /
  `SUSPICIOUS` / `INCONCLUSIVE`, always paired with the disclaimer that
  it does not prove fraud, intent, or fault.
- **Orchestration** (`src/pipeline.py`): `run_organization_pipeline` wires
  auth -> connector -> readiness -> model-scope decision -> (train +
  evaluate + exposure + diagnosis + recommendation + simulated
  verification + drift + registry) or abstain, and always persists a
  tenant-scoped result. `review_return_claim` runs the secondary evidence
  layer for one claim, enforcing that a claim's `organization_id` matches
  the authenticated caller's.
- **Minimal API** (`src/api/app.py`): `POST /auth/login`, `POST
  /auth/logout`, `GET /risk/latest` (protected, tenant-scoped), `POST
  /webhooks/razorpay`.
- **Automated tests cover ingestion, leakage, model, metrics, benchmark, exposure, diagnosis, verification, auth, tenancy, connectors, claims/evidence, image evidence, Razorpay, dashboard, and end-to-end flows.** covering all of the above (data
  quality edge cases -- malformed dates, unseen categories, missing
  columns, conflicting duplicates, insufficient history -- auth, tenancy
  including cross-tenant isolation, connectors, claims/evidence, image
  evidence, Razorpay signature/idempotency, dashboard rendering, and an
  end-to-end pipeline test including the abstention path). Run with
  `python3 -m pytest -q` from the repo root.
- **Dashboard generator** (`app/ui/generate_dashboard.py`): renders
  DATA SOURCES (`render_data_sources`, G1: Razorpay/Merchant
  API/Database/CSV/synthetic-demo, with the org's actual connector
  marked active and CSV explicitly labelled fallback/import, not the
  product), DATA READINESS (per-field checkmarks plus the READY /
  PARTIALLY SUPPORTED / NOT READY badge), and a dynamic confusion matrix
  + precision/recall/F1 block explicitly labelled `DEMO / SYNTHETIC` or
  `MERCHANT HELD-OUT TEST` depending on the result's `is_synthetic_demo`
  flag (A9's requirement, not just a generic badge). It also renders
  RETURN CLAIMS / EVIDENCE (`render_claim_card` / `render_claims_section`,
  G4/G5: reason, supporting/contradictory/missing signals, the
  SUPPORTED/NEEDS_REVIEW/SUSPICIOUS/INDETERMINATE tag, and the
  never-a-fraud-verdict disclaimer on every card). Critically, `render()`
  It accepts both the benchmark report shape and the per-organization
  pipeline result shape without requiring benchmark-only fields.
  The dashboard includes a lightweight login gate backed by the protected API;
  account/session persistence remains an MVP in-memory implementation.

### DEMO / MOCK

- `data/sample/generic_merchant_orders.csv` and the Amazon-style sample
  remain **synthetic** -- every number derived from them in this README
  or a report is labelled accordingly, never presented as real-world
  performance.
- `MockMerchantConnector` serves the synthetic dataset through the same
  interface a real connector would use; `connector_type == "mock"` is
  the flag every consumer checks to label results DEMO/SYNTHETIC.
- `RazorpayClient` runs in demo mode (`is_demo=True`) whenever
  `RAZORPAY_KEY_ID`/`RAZORPAY_KEY_SECRET` are not set, returning a small,
  clearly-synthetic page of Razorpay-shaped payment objects instead of
  making a network call.
- `MockImageEvidenceAnalyzer` is the only image analyzer implemented; a
  real pretrained detector would need to be documented as pretrained and
  would still not be allowed to claim a specific accuracy number without
  a genuine held-out evaluation.
- `simulate_intervention` (pre-existing, unchanged) remains a labelled
  synthetic before/after simulation, not a real intervention outcome.
- `src/auth/service.py` and `src/tenancy/store.py` are in-memory
  reference implementations. The API shape (register/login/logout/
  require_session; put/get/list_kind/delete, all organization_id-scoped)
  is what a production deployment would keep, backed by a real database
  with row-level security instead of a Python dict.

### FUTURE PRODUCTION (explicitly out of scope for this MVP)

- A real pretrained image-authenticity/manipulation detector.
- A production database/warehouse/ERP/OMS connector beyond CSV and
  Razorpay (the `MerchantDataConnector` abstraction is designed so this
  can be added without touching the ML engine).
- A real interactive frontend and login page (the dashboard generator
  now renders data sources, data readiness, the confusion matrix, and
  return-claims/evidence as static HTML sections from a JSON report; a
  real app would call `POST /auth/login` and the other API routes
  directly instead of regenerating a static file).
- Enterprise SSO, a distributed session/database backend, Kafka,
  microservices, autonomous refunds/blocks/cancellations, or any other
  item explicitly listed as "do not overbuild" in the product spec.



## Small real Razorpay Test Mode flow

1. In the Razorpay Dashboard switch to **Test Mode** and generate a Test Mode Key ID/Key Secret. Razorpay documents Test Mode as a sandbox; no real payments are processed.
2. Register/login to the app, then call `POST /integrations/razorpay/test-connection` with the Test Mode credentials and an optional webhook secret. The server makes a real read-only Razorpay API request and registers the data source.
3. Call `POST /integrations/razorpay/import` to pull Test Mode payments/refunds into the canonical schema.
4. Configure Razorpay Test Mode webhook delivery to `/webhooks/razorpay/<data_source_id>`. The endpoint verifies the raw-body HMAC signature and deduplicates `x-razorpay-event-id`.
5. Razorpay payments/refunds are operational signals. Razorpay does not provide a native order-return outcome through this payments flow, so `refund_event` stays separate from `return_event`. Use an OMS/returns source for the actual return label before training the return-risk model.

Razorpay's official documentation confirms that Test Mode uses separate API keys, test webhooks carry the same payload structure as live webhooks, webhook signatures use HMAC-SHA256, and duplicate events should be handled by event ID.


## Development disclosure

This submission was developed with AI-assisted iteration, including code review, debugging, test hardening, and documentation updates. The final implementation and design decisions remain the responsibility of the project author.
