"""A 5xx must never carry raw exception text to the client.

MIS-E2E-110. IDL-22 hardened error messages module by module, so every module
written afterwards escaped it — the classic "fixed one representative, never
generalized" shape. Rather than list the modules, this derives the rule from
the code: any `except` handler that raises a 5xx `HTTPException` whose detail
is built from the caught exception is a finding, wherever it lives.

The distinction that matters, and the reason a blanket ban would be wrong: all
27 `detail=str(e)` sites in this API were checked, and 26 are DOMAIN exceptions
on 4xx codes — `CircuitValidationError` on a 422, `ProtectedSettingError` on a
403, `ValueError` on a 400. There the message IS the explanation the caller
needs, and stripping it would make the API worse. Exactly one was a 500, and it
embedded a checkpoint's filesystem path plus a raw OSError on an unauthenticated
route. That one is fixed; this keeps the class closed.
"""

import ast
from pathlib import Path

import pytest

API = Path(__file__).resolve().parents[2] / "src" / "api"


def _modules():
    files = sorted(API.rglob("*.py"))
    assert len(files) > 15, f"only {len(files)} API modules found — the scan broke"
    return files


def _mentions_caught_name(node, caught: str) -> bool:
    """Does this expression read the caught exception variable?"""
    if caught is None:
        return False
    return any(
        isinstance(n, ast.Name) and n.id == caught
        for n in ast.walk(node)
    )


def _offenders_in(path: Path):
    tree = ast.parse(path.read_text())
    found = []
    for handler in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
        caught = handler.name
        for node in ast.walk(handler):
            if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
                continue
            call = node.exc
            if getattr(call.func, "id", None) != "HTTPException":
                continue
            status = detail = None
            for kw in call.keywords:
                if kw.arg == "status_code":
                    status = getattr(kw.value, "value", None)
                elif kw.arg == "detail":
                    detail = kw.value
            if not isinstance(status, int) or status < 500:
                continue
            if detail is not None and _mentions_caught_name(detail, caught):
                found.append(f"{path.name}:{node.lineno} ({status})")
    return found


class TestTheScanWorks:
    """A scan that parses nothing agrees with everything."""

    def test_it_finds_http_exceptions_at_all(self):
        total = 0
        for path in _modules():
            tree = ast.parse(path.read_text())
            total += sum(
                1 for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "HTTPException"
            )
        assert total > 100, f"only {total} HTTPException calls seen — scan is blind"

    def test_it_would_flag_a_leaking_handler(self):
        """Synthetic positive: the detector must catch the shape it bans."""
        import tempfile

        leaking = (
            "from fastapi import HTTPException\n"
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except Exception as e:\n"
            "        raise HTTPException(status_code=500, detail=str(e))\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(leaking)
            tmp = Path(fh.name)
        try:
            assert _offenders_in(tmp), "the detector does not catch its own example"
        finally:
            tmp.unlink()

    def test_it_does_not_flag_a_4xx_domain_message(self):
        """The 26 legitimate sites must stay legal."""
        import tempfile

        fine = (
            "from fastapi import HTTPException\n"
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except ValueError as e:\n"
            "        raise HTTPException(status_code=400, detail=str(e))\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(fine)
            tmp = Path(fh.name)
        try:
            assert not _offenders_in(tmp), (
                "a 400 carrying its domain message was flagged; that message is "
                "the explanation the caller needs"
            )
        finally:
            tmp.unlink()


class TestNo5xxCarriesExceptionText:
    def test_the_whole_api_is_clean(self):
        offenders = []
        for path in _modules():
            offenders.extend(_offenders_in(path))
        assert not offenders, (
            "these handlers put the caught exception's text into a 5xx response. "
            "Unexpected-exception text carries paths, errnos and internals to an "
            "unauthenticated caller — log it and return something generic:\n  "
            + "\n  ".join(offenders)
        )
