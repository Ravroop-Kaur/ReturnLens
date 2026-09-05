"""
Runs the statistical diagnosis engine's detection logic against the
frozen synthetic benchmark scenarios and reports precision/recall of
"did the engine correctly say this segment has / does not have a real
effect", against ground truth that was fixed before generation.

This benchmark measures the DIAGNOSIS component's detection
reliability under controlled conditions. It is reported separately
from, and must never be confused with, the ML model's real-data
evaluation in evaluation/reports/.
"""

from __future__ import annotations
import json
from pathlib import Path

from src.diagnosis.statistical import diagnose_dimension
from evaluation.benchmark.scenarios import generate_scenarios, split_scenarios
from evaluation.metrics.classification import confusion_counts, evaluate
import numpy as np


def _engine_says_effect(scenario) -> bool:
    findings = diagnose_dimension(
        scenario.df, dimension="segment_dim", target_col="return_event",
        min_segment_size=15,
    )
    target = [f for f in findings if f.segment == "target_segment"]
    if not target:
        return False
    f = target[0]
    return bool(f.statistically_supported and f.practically_significant)


def run_on_scenario_set(scenarios: list, bootstrap_iterations: int = 1000, seed: int = 42) -> dict:
    y_true = np.array([1 if s.ground_truth_has_effect else 0 for s in scenarios])
    y_pred = np.array([1 if _engine_says_effect(s) else 0 for s in scenarios])
    metrics = evaluate(y_true, y_pred, threshold=0.5)  # y_pred already 0/1
    # Scenario-level bootstrap interval: useful because a small benchmark can
    # make precision/recall look unstable when one scenario flips.
    rng = np.random.default_rng(seed)
    precisions, recalls = [], []
    n = len(scenarios)
    for _ in range(bootstrap_iterations):
        idx = rng.integers(0, n, size=n)
        m = evaluate(y_true[idx], y_pred[idx], threshold=0.5)
        if m.precision_defined:
            precisions.append(m.precision)
        if m.recall_defined:
            recalls.append(m.recall)
    ci = {
        "precision_95_ci": [float(np.quantile(precisions, 0.025)), float(np.quantile(precisions, 0.975))] if precisions else None,
        "recall_95_ci": [float(np.quantile(recalls, 0.025)), float(np.quantile(recalls, 0.975))] if recalls else None,
    }
    per_scenario = [
        {
            "scenario_id": s.scenario_id,
            "family": s.family,
            "ground_truth_has_effect": s.ground_truth_has_effect,
            "engine_detected_effect": bool(p),
            "correct": bool(s.ground_truth_has_effect == bool(p)),
        }
        for s, p in zip(scenarios, y_pred)
    ]
    return {"metrics": metrics.to_dict(), "bootstrap_95_ci": ci, "per_scenario": per_scenario}


def main():
    scenarios = generate_scenarios()
    dev, val, test = split_scenarios(scenarios)

    results = {
        "label": "Synthetic benchmark performance -- not real-world production performance.",
        "n_scenarios_total": len(scenarios),
        "n_dev": len(dev), "n_val": len(val), "n_test": len(test),
        "dev_results": run_on_scenario_set(dev),
        "val_results": run_on_scenario_set(val),
        "test_results": run_on_scenario_set(test),
    }

    out_path = Path(__file__).parents[1] / "reports" / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps({
        "label": results["label"],
        "test_metrics": results["test_results"]["metrics"],
    }, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
