"""
Intervention verification engine.

For this MVP, we do not execute a real merchant-side intervention
(that would require actually changing a courier, listing, etc. in
production, which is out of scope and would be an offense-capable /
autonomous action). Instead we SIMULATE a plausible post-intervention
return rate for the flagged segment and run the same statistical test
a real before/after comparison would use, so the mechanics of
verification are genuine even though the "after" data is synthetic.

Every output from this module must be labelled DEMO / SYNTHETIC
SIMULATION. If real post-intervention data is ever supplied, the same
`verify_before_after` function can be used unmodified -- the labelling
is the caller's responsibility (see the UI layer).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class VerificationResult:
    before_rate: float
    after_rate: float
    before_n: int
    after_n: int
    absolute_change: float
    relative_change: float
    z_stat: float
    p_value: float
    improved: "bool | None"
    is_simulation: bool
    label: str


def _two_proportion_z_test(x1, n1, x2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value


def verify_before_after(
    before_returns: int,
    before_n: int,
    after_returns: int,
    after_n: int,
    is_simulation: bool = True,
    alpha: float = 0.05,
) -> VerificationResult:
    before_rate = before_returns / before_n if before_n else float("nan")
    after_rate = after_returns / after_n if after_n else float("nan")

    z_stat, p_value = _two_proportion_z_test(before_returns, before_n, after_returns, after_n)

    if before_n < 20 or after_n < 20:
        improved = None  # not enough sample size to say anything
    elif np.isnan(p_value):
        improved = None
    elif p_value < alpha and after_rate < before_rate:
        improved = True
    elif p_value < alpha and after_rate > before_rate:
        improved = False
    else:
        improved = None  # indeterminate: change not statistically distinguishable from noise

    return VerificationResult(
        before_rate=before_rate,
        after_rate=after_rate,
        before_n=before_n,
        after_n=after_n,
        absolute_change=after_rate - before_rate,
        relative_change=((after_rate - before_rate) / before_rate) if before_rate else float("nan"),
        z_stat=z_stat,
        p_value=p_value,
        improved=improved,
        is_simulation=is_simulation,
        label="DEMO / SYNTHETIC SIMULATION" if is_simulation else "MEASURED (real data)",
    )


def simulate_intervention(
    before_rate: float,
    before_n: int,
    assumed_relative_reduction: float = 0.45,
    seed: int = 42,
) -> VerificationResult:
    """
    Simulate a plausible after-intervention sample of the SAME size as
    the before sample, drawing from a Bernoulli process at a reduced
    rate. `assumed_relative_reduction` is an explicit, documented
    assumption (not a measured effect) representing a moderate,
    plausible improvement if the recommended review is acted on.
    """
    rng = np.random.default_rng(seed)
    before_returns = int(round(before_rate * before_n))

    after_n = before_n
    simulated_after_rate = before_rate * (1 - assumed_relative_reduction)
    after_returns = int(rng.binomial(after_n, simulated_after_rate))

    return verify_before_after(
        before_returns=before_returns,
        before_n=before_n,
        after_returns=after_returns,
        after_n=after_n,
        is_simulation=True,
    )
