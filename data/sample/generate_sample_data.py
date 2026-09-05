"""
Generates a SYNTHETIC generic-merchant orders CSV for development and
demo purposes.

This is not real merchant data. It exists so the pipeline (ingestion,
features, model, diagnosis, exposure, recommendation, verification)
can be developed and demonstrated end-to-end without a real dataset.
The README and UI must always label results derived from this file
as coming from demo/sample data, not a real merchant.

The generation process deliberately builds in known, named risk
drivers (fulfilment method, category, shipping service) with
realistic effect sizes and noise, so that:
  - the ML model has genuine, learnable signal (not memorization)
  - the statistical diagnosis engine has something real to find
  - held-out precision/recall are measuring real detection ability,
    not evaluating against a trivial or fabricated target
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42


def generate(n_orders: int = 24000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    categories = np.array(["Apparel", "Electronics", "Home", "Beauty", "Sports", "Books"])
    category_p = np.array([0.30, 0.20, 0.18, 0.14, 0.10, 0.08])
    category = rng.choice(categories, size=n_orders, p=category_p)

    fulfilment = np.array(["platform_fulfilled", "merchant_fulfilled", "third_party_fulfilled"])
    fulfilment_p = np.array([0.55, 0.25, 0.20])
    fulfilment_method = rng.choice(fulfilment, size=n_orders, p=fulfilment_p)

    shipping = np.array(["Standard", "Expedited", "Economy"])
    shipping_p = np.array([0.55, 0.20, 0.25])
    shipping_service = rng.choice(shipping, size=n_orders, p=shipping_p)

    regions = np.array(["North", "South", "East", "West", "Central"])
    region_p = np.array([0.24, 0.22, 0.20, 0.18, 0.16])
    region = rng.choice(regions, size=n_orders, p=region_p)

    # Persistent customer behaviour: learnable from prior orders only.
    n_customers = max(1500, n_orders // 12)
    customer_id = np.array([f"CUST-{i:05d}" for i in rng.integers(0, n_customers, size=n_orders)])
    customer_propensity = rng.normal(0.0, 2.5, size=n_customers)

    # order dates spread across a full year, uniform-ish with mild weekly seasonality
    start = pd.Timestamp("2025-01-01")
    day_offsets = rng.integers(0, 365, size=n_orders)
    order_date = start + pd.to_timedelta(day_offsets, unit="D")
    order_date = order_date + pd.to_timedelta(rng.integers(0, 24, size=n_orders), unit="h")

    # product ids: 800 distinct products, category-consistent, power-law popularity
    product_pool = {}
    product_id = np.empty(n_orders, dtype=object)
    for cat in categories:
        product_pool[cat] = [f"{cat[:3].upper()}-{i:04d}" for i in range(1, 141)]
    for i in range(n_orders):
        pool = product_pool[category[i]]
        # zipf-like popularity skew
        idx = min(int(rng.zipf(1.8)), len(pool)) - 1
        product_id[i] = pool[idx]

    # Persistent product return propensity.
    unique_products = np.unique(product_id)
    product_propensity = {p: float(rng.normal(0.0, 1.5)) for p in unique_products}

    # order amount: category-dependent lognormal
    base_mu = {
        "Apparel": 6.6, "Electronics": 7.6, "Home": 6.9,
        "Beauty": 6.2, "Sports": 6.8, "Books": 5.6,
    }
    amount = np.array([
        rng.lognormal(mean=base_mu[c], sigma=0.55) for c in category
    ])
    amount = np.round(amount, 2)

    # ---- true (unknown-to-model) return probability function ----
    logit = np.full(n_orders, -2.35)  # base rate ~ 8.7%

    cat_effect = {
        "Apparel": 1.25, "Electronics": 0.10, "Home": -0.05,
        "Beauty": 0.15, "Sports": -0.10, "Books": -0.55,
    }
    logit += np.array([cat_effect[c] for c in category])

    fulfil_effect = {
        "platform_fulfilled": -0.20,
        "merchant_fulfilled": 0.05,
        "third_party_fulfilled": 0.65,
    }
    logit += np.array([fulfil_effect[f] for f in fulfilment_method])
    logit += np.array([customer_propensity[int(cid.split("-")[1])] for cid in customer_id])
    logit += np.array([product_propensity[p] for p in product_id])

    ship_effect = {"Standard": -0.05, "Expedited": -0.15, "Economy": 0.30}
    logit += np.array([ship_effect[s] for s in shipping_service])

    region_effect = {"North": 0.0, "South": 0.05, "East": -0.05, "West": 0.10, "Central": 0.0}
    logit += np.array([region_effect[r] for r in region])

    # amount effect: higher-value orders somewhat more likely to be returned
    amount_z = (np.log(amount) - np.log(amount).mean()) / np.log(amount).std()
    logit += 0.28 * amount_z

    # interaction: third-party fulfilment + Economy shipping is especially bad
    interaction = (fulfilment_method == "third_party_fulfilled") & (shipping_service == "Economy")
    logit += np.where(interaction, 0.45, 0.0)

    # mild temporal drift (post-festival-season return spike late in year)
    month = pd.DatetimeIndex(order_date).month
    logit += np.where(np.isin(month, [11, 12]), 0.20, 0.0)

    # idiosyncratic noise
    logit += rng.normal(0, 0.02, size=n_orders)

    p_return = 1 / (1 + np.exp(-logit))
    return_event = rng.binomial(1, p_return).astype(bool)

    # refund_event lags return_event closely but not identical (some
    # returns pending refund at time of data export)
    refund_event = return_event & (rng.random(n_orders) < 0.9)

    # chargeback: rare, weakly correlated with return but distinct
    chargeback_p = 0.003 + 0.01 * return_event
    chargeback_event = rng.binomial(1, chargeback_p).astype(bool)

    return_date = np.where(
        return_event,
        order_date + pd.to_timedelta(rng.integers(3, 21, size=n_orders), unit="D"),
        pd.NaT,
    )

    payment_status = rng.choice(["paid", "cod"], size=n_orders, p=[0.78, 0.22])

    df = pd.DataFrame({
        "order_id": [f"ORD-{i:07d}" for i in range(1, n_orders + 1)],
        "order_date": order_date,
        "amount": amount,
        "product_id": product_id,
        "customer_id": customer_id,
        "category": category,
        "region": region,
        "fulfilment_method": fulfilment_method,
        "shipping_service": shipping_service,
        "payment_status": payment_status,
        "return_event": return_event,
        "return_date": return_date,
        "refund_event": refund_event,
        "chargeback_event": chargeback_event,
    })

    df = df.sort_values("order_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    df = generate()
    out_path = out_dir / "generic_merchant_orders.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic orders to {out_path}")
    print(f"Overall return rate: {df['return_event'].mean():.3%}")
    print(df.groupby("category")["return_event"].mean().sort_values(ascending=False))
