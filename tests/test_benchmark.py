from evaluation.benchmark.scenarios import generate_scenarios, split_scenarios
from evaluation.benchmark.run_benchmark import run_on_scenario_set, _engine_says_effect


def test_scenarios_have_families_and_fixed_ground_truth():
    scenarios = generate_scenarios()
    families = {s.family for s in scenarios}
    assert "true_positive" in families
    assert "true_negative" in families
    assert "hard_negative" in families
    assert "subtle_positive" in families
    assert "distribution_shift" in families
    for s in scenarios:
        assert isinstance(s.ground_truth_has_effect, bool)


def test_true_positive_scenarios_are_usually_detected():
    scenarios = [s for s in generate_scenarios() if s.family == "true_positive"]
    detected = [_engine_says_effect(s) for s in scenarios]
    assert sum(detected) >= 2  # at least most large clear effects are caught


def test_true_negative_scenarios_do_not_trigger_false_detection():
    scenarios = [s for s in generate_scenarios() if s.family == "true_negative"]
    detected = [_engine_says_effect(s) for s in scenarios]
    # The benchmark is intentionally stochastic; require a low false-positive
    # rate rather than an impossible zero-error guarantee.
    assert sum(detected) / len(detected) <= 0.05


def test_reproducible_scenario_generation():
    s1 = generate_scenarios(seed=99)
    s2 = generate_scenarios(seed=99)
    assert [s.scenario_id for s in s1] == [s.scenario_id for s in s2]
    assert s1[0].df["return_event"].tolist() == s2[0].df["return_event"].tolist()


def test_split_is_disjoint_and_covers_all_scenarios():
    scenarios = generate_scenarios()
    dev, val, test = split_scenarios(scenarios)
    ids = set(s.scenario_id for s in dev) | set(s.scenario_id for s in val) | set(s.scenario_id for s in test)
    assert len(ids) == len(scenarios)
    dev_ids = set(s.scenario_id for s in dev)
    val_ids = set(s.scenario_id for s in val)
    test_ids = set(s.scenario_id for s in test)
    assert dev_ids.isdisjoint(val_ids)
    assert dev_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_run_on_scenario_set_returns_metrics_and_detail():
    scenarios = generate_scenarios()
    result = run_on_scenario_set(scenarios)
    assert "metrics" in result
    assert "per_scenario" in result
    assert len(result["per_scenario"]) == len(scenarios)
