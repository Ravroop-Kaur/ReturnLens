"""
Synthetic benchmark scenario generator.

This is INDEPENDENT of the demo merchant dataset in data/sample/. Its
job is to stress-test the statistical diagnosis engine (the component
that decides "is return risk concentrated in this segment?") against
controlled scenarios where the ground truth ("is there really a
planted effect here, yes or no?") is fixed BEFORE the diagnosis engine
ever sees the data. The diagnosis engine never gets to define its own
ground truth -- that would make the benchmark circular.

Scenario families:
  - true_positive: a real, clearly-detectable effect is planted
  - true_negative: no effect planted (segment = baseline rate)
  - hard_negative: a small, non-planted random fluctuation that could
    look like an effect by chance, at a small sample size
  - subtle_positive: a real but small effect, near the detection
    threshold, at a moderate sample size
  - noisy: baseline rate is itself unstable / high-variance
  - distribution_shift: baseline rate differs between "early" and
    "late" halves of the scenario, unrelated to the segment itself
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BenchmarkScenario:
    scenario_id: str
    family: str
    df: pd.DataFrame
    ground_truth_has_effect: bool
    planted_relative_risk: float
    segment_n: int
    baseline_n: int


def _make_scenario(rng, scenario_id, family, relative_risk, baseline_rate,
                    segment_n, baseline_n, has_effect):
    seg_rate = min(0.95, baseline_rate * relative_risk) if has_effect else baseline_rate
    seg_events = rng.binomial(1, seg_rate, size=segment_n)
    base_events = rng.binomial(1, baseline_rate, size=baseline_n)

    df = pd.DataFrame({
        "segment_dim": ["target_segment"] * segment_n + ["other"] * baseline_n,
        "return_event": np.concatenate([seg_events, base_events]).astype(bool),
    })
    return BenchmarkScenario(
        scenario_id=scenario_id,
        family=family,
        df=df,
        ground_truth_has_effect=has_effect,
        planted_relative_risk=relative_risk if has_effect else 1.0,
        segment_n=segment_n,
        baseline_n=baseline_n,
    )


def generate_scenarios(seed: int = 123) -> list:
    """Generate a larger fixed scenario benchmark.

    The benchmark now contains 600 independent scenarios (100 per family),
    so individual scenario flips have much less influence on the result.
    Ground truth is fixed before diagnosis sees each scenario. This remains a diagnosis
    benchmark, not the ML order-level benchmark.
    """
    rng = np.random.default_rng(seed)
    scenarios = []

    # 20 clear positives: vary effect size and sample size.
    for i in range(100):
        rr = float(rng.choice([2.0, 2.5, 3.0, 3.5, 4.0]))
        n = int(rng.choice([100, 150, 200, 300, 400, 600]))
        scenarios.append(_make_scenario(rng, f"true_positive_{i:02d}", "true_positive", rr, 0.10, n, 1800, True))

    # 20 clean negatives.
    for i in range(100):
        n = int(rng.choice([150, 250, 400, 600, 800, 1200]))
        scenarios.append(_make_scenario(rng, f"true_negative_{i:02d}", "true_negative", 1.0, 0.10, n, 1800, False))

    # 20 small-sample negatives where random noise can look convincing.
    for i in range(100):
        n = int(rng.choice([25, 35, 50, 60, 80]))
        scenarios.append(_make_scenario(rng, f"hard_negative_{i:02d}", "hard_negative", 1.0, 0.10, n, 1800, False))

    # 20 subtle positives around the practical-effect boundary.
    for i in range(100):
        rr = float(rng.choice([1.25, 1.30, 1.35, 1.40, 1.50]))
        n = int(rng.choice([400, 600, 900, 1200, 1600]))
        scenarios.append(_make_scenario(rng, f"subtle_positive_{i:02d}", "subtle_positive", rr, 0.10, n, 2200, True))

    # 20 higher-variance baseline scenarios. Half have a genuine effect.
    for i in range(100):
        has_effect = i % 2 == 1
        rr = float(rng.choice([1.0, 1.8, 2.0])) if has_effect else 1.0
        n = int(rng.choice([200, 300, 500, 700]))
        scenarios.append(_make_scenario(rng, f"noisy_{i:02d}", "noisy", rr, 0.35, n, 1400, has_effect))

    # 20 distribution-shift negatives: segment and baseline drift together.
    for i in range(100):
        n = int(rng.choice([250, 350, 500, 700, 900]))
        early_rate, late_rate = rng.choice([(0.06, 0.12), (0.08, 0.14), (0.10, 0.18)])
        half = n // 2
        seg_events = np.concatenate([
            rng.binomial(1, early_rate, size=half),
            rng.binomial(1, late_rate, size=n - half),
        ])
        base_n = int(rng.choice([1500, 1800, 2200]))
        base_half = base_n // 2
        base_events = np.concatenate([
            rng.binomial(1, early_rate, size=base_half),
            rng.binomial(1, late_rate, size=base_n - base_half),
        ])
        df = pd.DataFrame({
            "segment_dim": ["target_segment"] * n + ["other"] * base_n,
            "return_event": np.concatenate([seg_events, base_events]).astype(bool),
        })
        scenarios.append(BenchmarkScenario(
            scenario_id=f"distribution_shift_{i:02d}", family="distribution_shift",
            df=df, ground_truth_has_effect=False, planted_relative_risk=1.0,
            segment_n=n, baseline_n=base_n,
        ))

    return scenarios


def split_scenarios(scenarios: list, seed: int = 7):
    """Fixed, family-stratified split of scenarios themselves.

    Each of the six scenario families contributes 33 scenarios to dev,
    33 to validation, and 34 to the frozen test set. This avoids a tiny
    test set or an accidental family imbalance driving the benchmark.
    """
    rng = np.random.default_rng(seed)
    by_family = {}
    for s in scenarios:
        by_family.setdefault(s.family, []).append(s)
    dev, val, test = [], [], []
    for family in sorted(by_family):
        items = list(by_family[family])
        rng.shuffle(items)
        if len(items) < 100:
            raise ValueError(f"Family {family} needs at least 30 scenarios for the stratified split.")
        dev.extend(items[:33])
        val.extend(items[33:66])
        test.extend(items[66:100])
    rng.shuffle(dev); rng.shuffle(val); rng.shuffle(test)
    return dev, val, test
