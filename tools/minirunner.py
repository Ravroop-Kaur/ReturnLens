"""
DEV-ONLY test runner for sandboxes with no network access to install
real pytest (see tools/pytest_shim/pytest.py). Not part of the
shipped product and not required in any normal development or CI
environment -- there, `pip install pytest && pytest` works as usual
and this script is simply unused.

Usage:
    python tools/minirunner.py [substring-filter]

Discovers tests/test_*.py, resolves simple fixtures (monkeypatch,
tmp_path, and any function-scoped fixture defined in the test module
itself via @pytest.fixture), and reports pass/fail counts.
"""
import sys
import os
import importlib
import importlib.util
import inspect
import traceback

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, os.path.join(TOOLS_DIR, "pytest_shim"))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pytest  # the shim (only used if real pytest isn't already on sys.path)

TEST_DIR = os.path.join(REPO, "tests")

results = {"passed": 0, "failed": 0, "errors": []}


def resolve_fixture(name, module, cache, mp_registry):
    if name == "monkeypatch":
        mp = pytest.MonkeyPatch()
        mp_registry.append(mp)
        return mp
    if name == "tmp_path":
        import tempfile
        from pathlib import Path
        return Path(tempfile.mkdtemp(prefix="minipytest_"))
    if name in cache:
        return cache[name]
    fixture_func = getattr(module, name, None)
    if fixture_func is None or not getattr(fixture_func, "_is_fixture", False):
        raise RuntimeError(f"Unknown fixture: {name}")
    sig = inspect.signature(fixture_func)
    kwargs = {}
    for pname in sig.parameters:
        kwargs[pname] = resolve_fixture(pname, module, cache, mp_registry)
    value = fixture_func(**kwargs)
    cache[name] = value
    return value


def run_module(path):
    mod_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        results["errors"].append((mod_name, "<import>", "".join(traceback.format_exception(type(e), e, e.__traceback__))))
        results["failed"] += 1
        return

    test_funcs = [
        (name, obj) for name, obj in vars(module).items()
        if name.startswith("test_") and inspect.isfunction(obj)
    ]

    for name, func in sorted(test_funcs):
        sig = inspect.signature(func)
        cache = {}
        mp_registry = []
        try:
            kwargs = {}
            for pname in sig.parameters:
                kwargs[pname] = resolve_fixture(pname, module, cache, mp_registry)
            func(**kwargs)
            results["passed"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append((mod_name, name, "".join(traceback.format_exception(type(e), e, e.__traceback__))))
        finally:
            for mp in mp_registry:
                mp.undo()


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(f for f in os.listdir(TEST_DIR) if f.startswith("test_") and f.endswith(".py"))
    if only:
        files = [f for f in files if only in f]
    for f in files:
        run_module(os.path.join(TEST_DIR, f))

    print(f"\n{'='*60}\nPASSED: {results['passed']}  FAILED: {results['failed']}\n{'='*60}")
    for mod, test, tb in results["errors"]:
        print(f"\n--- FAIL: {mod}::{test} ---")
        print(tb[-2000:])

    sys.exit(1 if results["failed"] else 0)


if __name__ == "__main__":
    main()
