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

# ReturnLens

**Explainable Return Risk Intelligence** — built for the Razorpay AI Buildathon 2026 (Track 02: AI Risk Manager).

## What this actually is

Most merchants lose money to returns quietly — it's not one big fraud event, it's death by a thousand cuts. An order gets placed, shipped, and then comes right back, and the merchant eats the shipping both ways plus the tied-up capital. Often these returns aren't random either — they cluster around a specific courier, a category, a shipping method. ReturnLens tries to catch that pattern early.

Given an order at the moment it's placed, it predicts the probability that it'll eventually be returned, tells you *why* using a separate statistical check (not just "the model said so"), works out how much money is actually at stake, and suggests a bounded action a human can review. It never takes action on its own — no auto-refunds, no auto-blocking couriers, nothing like that.

I picked returns instead of fraud/chargebacks because it's the loss class you can actually learn from a normal e-commerce order export, and it's easy to explain to a non-technical merchant end to end — from "here's the risky segment" to "here's what it's costing you."

## Being upfront about the data

I didn't have access to a real merchant dataset for this build, so every number in this README comes from a **synthetic** dataset (`data/sample/generic_merchant_orders.csv`) that I generated myself with a known, documented risk pattern baked in (see `data/sample/generate_sample_data.py`). I did this on purpose so precision/recall would still mean something — the model has to actually detect the planted signal, it's not just made-up numbers. But I want to be clear: this is not a claim about how it'll perform on a real merchant's data. Anywhere a number shows up, it's labelled as synthetic/demo.

## How the model works

For every order, at the time it's placed, it predicts one thing: **will this order eventually come back as a return?** Output is a probability, plus a HIGH/LOW label from a threshold that's frozen ahead of time (not tuned after the fact).

Features going in: order amount, category, fulfilment method, shipping service, region, and timing (day/month/hour), plus historical return rates by product/category/fulfilment/region/shipping — but computed carefully so we only ever use information that would genuinely have been known at prediction time. `return_event`, `return_date`, `refund_event`, `chargeback_event` never touch the feature set — those are only used as the label or for evaluation afterward. Review text is left out entirely for this version, since I couldn't reliably confirm a review wasn't written *after* the return (which would leak the answer). All the reasoning and the tests that enforce this live in `docs/leakage_notes.md` and `tests/test_leakage.py`.

Data comes in through a merchant-agnostic canonical schema (`src/canonical/schema.py`). The main path is a generic CSV adapter (`src/adapters/generic_csv.py`); there's also an Amazon-style adapter (`src/adapters/amazon_adapter.py`) just to prove the schema isn't hard-coded around one platform — Amazon is one source among possible others, not a requirement.

## How it's evaluated

Split by time, not randomly: earliest 60% of orders train the model, next 20% is validation (where the threshold gets picked, using a cost function that treats a missed return as 3x worse than a false alarm — that's a documented assumption, not something measured), and the final 20% is a **frozen** test set that's touched exactly once, at the end.

On the synthetic demo run:

- Logistic Regression was chosen over LightGBM (validation ROC-AUC 0.883 vs 0.874)
- Threshold: 0.505
- Test set: 4,800 orders, 1,665 of which actually returned
- Precision 0.733, Recall 0.750, F1 0.741
- False positive rate 0.145, false negative rate 0.250

You can reproduce this yourself with `python -m evaluation.reports.run_full_pipeline`. Again — synthetic data, so treat this as "the pipeline works and measures itself honestly," not "this is the accuracy you'll get."

## The diagnosis engine (separate from the ML model)

`src/diagnosis/statistical.py` looks at each dimension — fulfilment, category, region, shipping — and flags segments where the return rate is both statistically real (Wilson confidence interval sits clearly above baseline) and big enough to matter (at least 1.3x the baseline rate), ignoring anything with under 30 orders so it's not reacting to noise. On the demo data it correctly finds that third-party-fulfilled orders return at 30.9% vs a 14.3% baseline — which is exactly the effect I planted in the generator, so at least I know the engine is finding real signal and not just noise.

## Money, in plain terms

`src/exposure/financial.py` uses order value to work out: how much is tied up in orders currently flagged high-risk, how much actually came back as returns historically, and how much is riding on false positives vs false negatives. I'm deliberately careful with language here — the code never says "savings," "profit," or "ROI" (there's even a test enforcing that), because none of this is a promise, it's an estimate of exposure.

## Checking whether an intervention actually helped

`src/verification/simulate.py` runs a two-proportion z-test comparing a before-rate to an after-rate. Right now the "after" data is simulated (an assumed reduction, clearly labelled DEMO/SYNTHETIC everywhere it shows up) — but the same function works unmodified the day a merchant hands over real before/after numbers. Only the label changes.

## What's real vs what's demo, laid out plainly

**Actually working:**
- A data readiness pipeline that checks duplicates, aggregates multi-line orders correctly, validates a feature contract, tracks label lifecycle (a return that hasn't resolved yet is `PENDING`, never force-labelled), and decides `READY`/`NOT_READY` — never silently guessing at missing data.
- Model scope decisions that can genuinely abstain (`INSUFFICIENT_DATA`) rather than train on garbage, and a version registry that never overwrites past model versions.
- Drift monitoring that compares distributions over time and reports normal/mild/significant drift — it only reports, it never auto-retrains.
- A real login/session system (PBKDF2-HMAC-SHA256 hashing, bearer tokens) and multi-tenancy enforced at the data layer, so one organization's data is never reachable through another's session.
- A connector abstraction with three implementations — mock (bundled synthetic data), CSV (manual import, the fallback), and Razorpay (real API or a clearly-labelled demo fallback).
- A real, working Razorpay Test Mode integration: signed webhook receiver (HMAC-SHA256 verified against the raw body, idempotent per org so a redelivered event isn't reprocessed), and endpoints to test a connection and import payments/refunds. Worth noting — Razorpay's payment data doesn't include an actual "this order was returned" signal, so you'd still need an OMS/returns source for real training labels.
- A return-claim evidence layer that never issues a fraud verdict, only SUPPORTED / NEEDS_REVIEW / SUSPICIOUS / INDETERMINATE, always with a disclaimer attached.
- An image evidence analyzer — currently a deterministic mock, clearly labelled `is_demo=True`, no accuracy claims anywhere.
- A dashboard generator that shows data sources, data readiness, the confusion matrix (tagged DEMO/SYNTHETIC or MERCHANT HELD-OUT TEST depending on what generated it), and the claims/evidence cards.
- Tests covering ingestion edge cases, leakage, the model, metrics, the benchmark, exposure, diagnosis, verification, auth, tenancy (including cross-tenant isolation), connectors, claims/evidence, image evidence, Razorpay, the dashboard, and an end-to-end run. `python3 -m pytest -q` from the repo root.

**Demo/mocked, and labelled as such everywhere it appears:**
- The synthetic order dataset and the Amazon-style sample.
- The mock connector and the mock image analyzer.
- Razorpay client falling back to demo payloads when no real keys are configured.
- The "after" side of the intervention simulation.
- Auth and tenancy are in-memory right now — the shape of the API is what you'd want in production, just backed by an actual database instead of a Python dict.

**Deliberately not built** (out of scope for this MVP): a real pretrained image-authenticity detector, production database/ERP/OMS connectors beyond CSV and Razorpay, a real frontend beyond the generated static dashboard, SSO, distributed sessions, Kafka, microservices, or any autonomous action on refunds/blocks/cancellations.

## Known rough edges

- No real merchant data was available, so nothing here is validated against reality yet — everything is honest about being synthetic.
- One feature (historical category return-rate) ends up with a counter-intuitive negative coefficient in the fitted regression, most likely from collinearity with the category one-hot features. I'm disclosing it rather than hiding it — a production version would probably drop the raw one-hots once the historical-rate feature is present.
- The "explanation" shown is based on feature weights, not a full SHAP breakdown — it's an approximation and documented as one.
- The FP/FN cost ratio used to pick the threshold (3x) is an assumption, not calibrated against a real merchant's actual costs.

## Architecture, roughly

```
merchant data (generic CSV | Amazon-style CSV)
        │
canonical schema
        │
feature engineering (leakage-safe)
        │
   ┌────┴─────────────────┐
   │                       │
ML risk scorer      statistical diagnosis
   │                       │
   └──────────┬────────────┘
              │
      financial exposure
              │
      bounded recommendation
              │
      intervention (simulated)
              │
      verification
              │
      merchant-facing dashboard
```

## Project layout

```
├── app/ui/              merchant-facing dashboard generator (static HTML)
├── src/
│   ├── canonical/       schema + column mapping/validation
│   ├── adapters/        generic CSV (main path) + Amazon-style (example)
│   ├── features/        leakage-safe feature engineering
│   ├── model/           train/val/test split, LR + LightGBM, version registry
│   ├── quality/         readiness, dedup, order-level aggregation, drift
│   ├── diagnosis/       statistical association engine
│   ├── exposure/        financial exposure math
│   ├── recommendation/  bounded recommendation engine
│   ├── verification/    before/after simulation + real z-test
│   ├── auth/            login/session service
│   ├── tenancy/         tenant-scoped storage
│   ├── connectors/      mock / CSV / Razorpay connectors
│   ├── integrations/razorpay/   client, config, mapper, webhook
│   ├── claims/          return-claim model + evidence aggregation
│   ├── evidence/        image evidence analyzer (secondary signal)
│   ├── pipeline.py      wires the whole thing together
│   └── api/app.py       minimal Flask app (login, protected routes, webhook)
├── evaluation/          metrics, benchmark harness, full pipeline report
├── data/sample/         synthetic dataset + generator
├── docs/                leakage notes, ml improvement notes, Razorpay test mode notes
├── tests/               all the automated tests
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# (re)generate the synthetic demo dataset
python data/sample/generate_sample_data.py

# run predict → explain → quantify → act → verify, end to end
python -m evaluation.reports.run_full_pipeline

# turn that report into the dashboard
python app/ui/generate_dashboard.py
# then just open app/ui/dashboard.html

# run the diagnosis engine's own benchmark
python -m evaluation.benchmark.run_benchmark

# run all tests
pytest tests/ -q

# run the demo API (login, protected risk endpoint, Razorpay webhook)
python -m src.api.app
# no Razorpay keys needed to try this — it falls back to mock/demo data
```

## Trying it with real Razorpay Test Mode

1. Switch to Test Mode in the Razorpay Dashboard and grab a Test Mode Key ID/Secret — Razorpay's own docs describe Test Mode as a full sandbox, no real payments involved.
2. Log in to the app, then hit `POST /integrations/razorpay/test-connection` with those credentials (and a webhook secret if you have one). This makes a real, read-only call to Razorpay to confirm it works.
3. Call `POST /integrations/razorpay/import` to pull Test Mode payments/refunds into the canonical schema.
4. Point Razorpay's Test Mode webhook delivery at `/webhooks/razorpay/<data_source_id>`. Signatures are verified against the raw body, and duplicate deliveries are deduped by event ID.
5. One catch worth knowing: Razorpay's payments API tells you about payments and refunds, not whether an order was *returned*. So `refund_event` and `return_event` stay separate — you'd still need an actual OMS/returns feed to train the return-risk model on real labels.

## On the "no autonomous actions" thing

This is worth stating plainly: nothing in this codebase can take a real-world action against a merchant's account, payment, courier, or customer. Every "intervention" is simulated and labelled as such. Razorpay webhooks only ever bring data *in* — they update stored risk info and never trigger a refund, a block, or a cancellation.
