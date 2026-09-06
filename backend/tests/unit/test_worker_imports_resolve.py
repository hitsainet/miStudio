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
"""

import ast
import importlib
from pathlib import Path

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
                try:
                    mod = importlib.import_module(target)
                except Exception as exc:                     # noqa: BLE001
                    broken.append(f"{path.name}:{lineno} import {target} -> {exc}")
                    continue
                for name in names:
                    if not hasattr(mod, name):
                        broken.append(
                            f"{path.name}:{lineno} `from {module} import {name}` "
                            f"-> {target} has no attribute {name!r}"
                        )
        assert not broken, (
            "these deferred imports raise at runtime, inside handlers that "
            "swallow the error:\n  " + "\n  ".join(broken)
        )

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
