from src.verification.simulate import verify_before_after, simulate_intervention


def test_verify_before_after_detects_improvement():
    result = verify_before_after(before_returns=300, before_n=1000, after_returns=150, after_n=1000)
    assert result.improved is True
    assert result.after_rate < result.before_rate
    assert result.is_simulation is True
    assert result.label == "DEMO / SYNTHETIC SIMULATION"


def test_verify_before_after_detects_worsening():
    result = verify_before_after(before_returns=100, before_n=1000, after_returns=300, after_n=1000)
    assert result.improved is False


def test_verify_before_after_indeterminate_with_small_sample():
    result = verify_before_after(before_returns=2, before_n=10, after_returns=1, after_n=10)
    assert result.improved is None  # not enough sample size to say anything


def test_verify_before_after_indeterminate_when_no_real_difference():
    result = verify_before_after(before_returns=100, before_n=1000, after_returns=98, after_n=1000)
    assert result.improved is None


def test_real_measurement_label_when_not_simulation():
    result = verify_before_after(before_returns=300, before_n=1000, after_returns=150, after_n=1000,
                                  is_simulation=False)
    assert result.label == "MEASURED (real data)"


def test_simulate_intervention_labels_as_synthetic():
    result = simulate_intervention(before_rate=0.30, before_n=500)
    assert result.is_simulation is True
    assert result.label == "DEMO / SYNTHETIC SIMULATION"
    assert result.before_n == result.after_n == 500
