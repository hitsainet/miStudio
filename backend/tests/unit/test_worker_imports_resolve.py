"""Every deferred import inside a worker must actually resolve.

2026-08-24. `tqdm_websocket_bridge.py:202` did

    from ..models.dataset import DatasetTokenization

but that class lives in `models/dataset_tokenization.py`. The line above it
correctly imports `Dataset` from `..models.dataset`, and this one copied the
path. Because the import sits INSIDE a `try` whose handler is
`logger.warning("Failed to update database progress: ...")`, it raised on every
single progress tick and was swallowed every time.

Consequence: a tokenization ran to completion — 789,850 samples in 6m30s — while
its database row stayed frozen at 40%. The UI showed a stuck job, the user
cancelled and deleted finished work, and the failure was invisible for seven
months because a function-level import is never exercised by module import or
by any test that does not run that exact branch.

That last point is why this test walks the AST rather than importing modules:
`import src.workers.tqdm_websocket_bridge` succeeds fine. Only resolving each
deferred import individually finds it.

A SUBMODULE IS A NAME TOO (2026-09-14, multi-GPU Phase 3 ported onto main).
`from ..services import steering_worker_card` imports a submodule, which Python
does whether or not anything imported it before. The scan asked only
`hasattr(src.services, "steering_worker_card")`, which is true only AFTER some
earlier import has bound the submodule onto its package. So the guard passed in
the full suite (test_steering_worker_per_gpu sorts first and imports it) and
failed run alone or in any other order, on three correct imports. A name is now
resolved as the package's attribute or, failing that, as its submodule
(:func:`_unresolvable`), and a submodule that is missing, or raises on import, is
still reported.

MUTATION CONTROL (2026-09-14, applied alone, this module run, restored and
checked by sha256):
  W1  the submodule fallback removed (an attribute only)
        -> test_a_submodule_nothing_has_imported_yet_resolves, and the scan
           itself run alone
"""

import ast
import importlib
from pathlib import Path
from typing import Optional

import pytest

WORKERS = Path(__file__).resolve().parents[2] / "src" / "workers"
SERVICES = Path(__file__).resolve().parents[2] / "src" / "services"


def _deferred_imports(path: Path):
    """`from X import a, b` statements nested inside a function body."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.module and inner.level:
                out.append((inner.lineno, inner.level, inner.module,
                            [a.name for a in inner.names]))
    return out


def _modules():
    files = sorted(list(WORKERS.glob("*.py")) + list(SERVICES.glob("*.py")))
    assert len(files) > 40, f"only {len(files)} modules found — the scan broke"
    return files


def _unresolvable(target: str, name: str) -> Optional[str]:
    """Why ``from <target> import <name>`` would raise, or None when it resolves.

    ``name`` resolves as an attribute of ``target`` or, as Python's own import
    falls back to, as its submodule ``<target>.<name>`` — imported here, so the
    answer never depends on what an earlier test happened to import.
    """
    try:
        mod = importlib.import_module(target)
    except Exception as exc:  # noqa: BLE001
        return f"import {target} -> {exc}"
    if hasattr(mod, name):
        return None
    submodule = f"{target}.{name}"
    try:
        importlib.import_module(submodule)
    except ModuleNotFoundError as exc:
        if exc.name in (submodule, target):
            return f"{target} has no attribute {name!r}"
        return f"import {submodule} -> {exc}"
    except Exception as exc:  # noqa: BLE001 - a submodule that raises on import is broken
        return f"import {submodule} -> {exc}"
    return None


class TestTheScanSeesDeferredImports:
    def test_it_finds_a_meaningful_number(self):
        total = sum(len(_deferred_imports(p)) for p in _modules())
        assert total > 50, (
            f"only {total} function-level imports found; the scan is broken and "
            f"would pass regardless of what it checks"
        )

    def test_module_import_alone_would_not_catch_this(self):
        """Why this test exists: the module imports fine with the bug present."""
        mod = importlib.import_module("src.workers.tqdm_websocket_bridge")
        assert mod is not None


class TestEveryDeferredImportResolves:
    def test_no_worker_or_service_defers_an_unresolvable_import(self):
        broken = []
        for path in _modules():
            pkg_parts = ["src"] + list(path.relative_to(
                Path(__file__).resolve().parents[2] / "src").parts[:-1])
            for lineno, level, module, names in _deferred_imports(path):
                # Resolve the relative module against this file's package.
                base = pkg_parts[: len(pkg_parts) - (level - 1)] if level > 1 else pkg_parts
                target = ".".join(base + module.split("."))
                for name in names:
                    why = _unresolvable(target, name)
                    if why is not None:
                        broken.append(f"{path.name}:{lineno} `from {module} import {name}` -> {why}")
        assert not broken, (
            "these deferred imports raise at runtime, inside handlers that "
            "swallow the error:\n  " + "\n  ".join(broken)
        )

    def test_a_submodule_nothing_has_imported_yet_resolves(self, monkeypatch):
        """`from ..services import steering_worker_card` works at runtime whatever
        was imported before it; the scan must agree, in any test order."""
        import sys

        import src.services

        monkeypatch.delitem(sys.modules, "src.services.steering_worker_card", raising=False)
        monkeypatch.delattr(src.services, "steering_worker_card", raising=False)
        assert not hasattr(src.services, "steering_worker_card"), "the fixture did not unbind the submodule"

        assert _unresolvable("src.services", "steering_worker_card") is None

    def test_a_name_that_is_neither_attribute_nor_submodule_is_still_reported(self):
        assert "has no attribute 'no_such_module_here'" in (
            _unresolvable("src.services", "no_such_module_here") or ""
        )
        # The defect this module was written for: a class, not a submodule, of a module.
        assert _unresolvable("src.models.dataset", "DatasetTokenization") is not None

    def test_the_specific_one_that_bit(self):
        from src.models.dataset_tokenization import DatasetTokenization  # noqa: F401
        import src.models.dataset as dataset_module

        assert not hasattr(dataset_module, "DatasetTokenization"), (
            "DatasetTokenization is now re-exported from models.dataset, which "
            "would make the old broken import work by accident — fine, but this "
            "test's premise changed"
        )
        text = (WORKERS / "tqdm_websocket_bridge.py").read_text()
        assert "from ..models.dataset_tokenization import DatasetTokenization" in text


class TestAFrozenProgressRowIsNotSilent:
    """A completed job whose progress could never be written is useless.

    The import bug raised on EVERY tick for seven months and each one was
    logged at WARNING and dropped. "Don't let database errors break the
    operation" is right for one dropped tick and catastrophic as a standing
    policy: nothing escalates, so a permanently broken writer looks identical
    to a busy one. Tokenization finished 789,850 samples while its row sat at
    40%; the operator saw "stuck" and deleted completed work.
    """

    def _source(self):
        return (WORKERS / "tqdm_websocket_bridge.py").read_text()

    def test_repeated_failures_escalate_past_warning(self):
        src = self._source()
        assert "_db_write_failures" in src, (
            "consecutive progress-write failures are not counted, so a writer "
            "that never once succeeds is indistinguishable from a healthy one"
        )
        assert "logger.error" in src, (
            "repeated failures never escalate above WARNING; the operator has "
            "no signal that the row is frozen"
        )

    def test_the_counter_resets_on_success(self):
        """Parsed, not sliced: the `else` belongs to the `try`, and a fixed
        character window after the increment missed it."""
        import ast
        import inspect
        import importlib

        mod = importlib.import_module("src.workers.tqdm_websocket_bridge")
        cls = next(
            c for _n, c in vars(mod).items()
            if isinstance(c, type) and hasattr(c, "DB_FAILURE_ALARM")
        )
        # `cleandoc` on a class source mangles the body indentation; use
        # textwrap.dedent, which preserves relative indentation.
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
        tries = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Try) and n.orelse
                 and any("_db_write_failures" in ast.dump(h) for h in n.handlers)]
        assert tries, (
            "no try/except/else around the progress write, so a successful "
            "write never clears the failure run and an occasional blip will "
            "eventually trip the alarm"
        )
        assert any("_db_write_failures" in ast.dump(stmt)
                   for t in tries for stmt in t.orelse), (
            "the else branch does not reset the counter"
        )

    def test_a_single_failure_is_still_tolerated(self):
        """One dropped tick must not fail the job — that part was correct."""
        src = self._source()
        assert "if self._db_write_failures == 1:" in src
        assert "logger.warning" in src

    def test_the_threshold_is_a_real_number(self):
        import importlib

        mod = importlib.import_module("src.workers.tqdm_websocket_bridge")
        cls = next(
            c for _n, c in vars(mod).items()
            if isinstance(c, type) and hasattr(c, "DB_FAILURE_ALARM")
        )
        assert 1 < cls.DB_FAILURE_ALARM <= 20, (
            f"DB_FAILURE_ALARM={cls.DB_FAILURE_ALARM} is not a sane escalation "
            f"point: 1 alarms on a single blip, a large value never fires"
        )
