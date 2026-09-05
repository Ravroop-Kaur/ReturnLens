Fallback test runner (optional)
================================
The test suite runs fine with real `pytest` (see the root requirements.txt).
This folder exists only as a fallback for a fully offline environment where
`pip install pytest` is not possible: `pytest.py` is a minimal stand-in for
the subset of pytest's API this suite uses (fixtures, monkeypatch, raises),
and `minirunner.py` is a tiny runner that executes test_*.py files directly.

To use it instead of real pytest:
    PYTHONPATH=tests/fallback_runner:. python3 tools/minirunner.py

This is not needed in any environment where `pytest` itself is installable.
