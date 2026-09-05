"""Minimal stand-in for the pieces of pytest this test suite uses,
for environments without network access to install real pytest.
Not a general pytest replacement -- just enough surface: raises(),
fixture(), monkeypatch, tmp_path.

This is a DEV-ONLY convenience shim (see tools/minirunner.py). It is
never imported when real pytest is installed -- a real pytest install
takes priority on sys.path in any normal environment. It exists so
this test suite can still be exercised in sandboxes with no network
access to pip-install real pytest.
"""
import os
import shutil
import tempfile
from pathlib import Path


class _RaisesContext:
    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.exc_type}")
        if not issubclass(exc_type, self.exc_type):
            return False
        self.value = exc_val
        return True


def raises(exc_type):
    return _RaisesContext(exc_type)


def fixture(func=None, **kwargs):
    if func is None:
        def wrapper(f):
            f._is_fixture = True
            return f
        return wrapper
    func._is_fixture = True
    return func


class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setenv(self, name, value):
        old = os.environ.get(name)
        had = name in os.environ
        os.environ[name] = value
        self._undo.append((name, had, old))

    def delenv(self, name, raising=False):
        had = name in os.environ
        old = os.environ.get(name)
        if had:
            del os.environ[name]
        elif raising:
            raise KeyError(name)
        self._undo.append((name, had, old))

    def undo(self):
        for name, had, old in reversed(self._undo):
            if had:
                os.environ[name] = old
            else:
                os.environ.pop(name, None)
        self._undo.clear()


class _TmpPathFactory:
    def __init__(self):
        self._dirs = []

    def new(self):
        d = Path(tempfile.mkdtemp(prefix="minipytest_"))
        self._dirs.append(d)
        return d

    def cleanup(self):
        for d in self._dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._dirs.clear()
