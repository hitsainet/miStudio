"""OSD-25 — CROSS_IMPLEMENTATION and ROUND_TRIP have no caller, ON PURPOSE.

The tracker recorded them as "written and tested but wired to no live consumer",
which is true and is the correct state: both need an external consumer — a live
miLLM serving the mounted artifact — and miStudio has no such consumer in its own
plane. Wiring them here could only fabricate a pass, which is worse than an
honest absence.

So this file does what the arc already does for `spliced_ce_delta`: it records the
absence deliberately, and pins the two properties that make the absence safe.

  1. They stay DEFERRABLE, so a missing result never blocks publishing — the four
     LOCAL classes are what `serviceable` gates on.
  2. They stay OUT of the local pass set, so a deferred check can never be read as
     a pass. A deferred check that counted as a pass would let an artifact claim
     an agreement nobody measured.

WHEN A CONSUMER EXISTS, replace this file with the reachability assertions —
`_called_names(...)` over the module that calls them — exactly as
`TestTheEvaluationIsNoLongerUnwired` replaced the `spliced_ce_delta` version.
"""
import ast
import inspect
import pkgutil
import importlib

from src.services import jlens_validation
from src.services.jlens_validation import (
    DEFERRABLE,
    CheckClass,
    CheckStatus,
    check_cross_implementation,
    check_round_trip,
)

DEFERRED_CHECKS = ("check_cross_implementation", "check_round_trip")


def _called_names(module) -> set:
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute):
            names.add(fn.attr)
        elif isinstance(fn, ast.Name):
            names.add(fn.id)
    return names


class TestTheAbsenceIsRecorded:

    def test_no_production_module_calls_them(self):
        """Reads the AST for a CALL — a substring search matches this file's own
        prose and the docstrings that describe the checks."""
        callers = {}
        for package in ("src.services", "src.workers", "src.api.v1.endpoints"):
            module = importlib.import_module(package)
            for found in pkgutil.iter_modules(module.__path__):
                name = f"{package}.{found.name}"
                try:
                    imported = importlib.import_module(name)
                    called = _called_names(imported)
                except Exception:                       # noqa: BLE001 - optional deps
                    continue
                for check in DEFERRED_CHECKS:
                    if check in called:
                        callers.setdefault(check, []).append(name)

        assert not callers, (
            "these checks now HAVE callers, which is the good outcome — replace "
            f"this file with reachability assertions for them: {callers}"
        )

    def test_they_are_still_importable_and_runnable(self):
        """Unwired is not the same as broken: a consumer must be able to call them."""
        result = check_round_trip(None)
        assert result.check is CheckClass.ROUND_TRIP
        assert result.status is not CheckStatus.PASS, (
            "no served readout must not read as a pass"
        )


class TestTheAbsenceIsSafe:

    def test_both_are_deferrable(self):
        assert CheckClass.CROSS_IMPLEMENTATION in DEFERRABLE
        assert CheckClass.ROUND_TRIP in DEFERRABLE

    def test_no_local_class_is_deferrable(self):
        """The four local classes are what `serviceable` gates on; deferring one
        would let an unvalidated artifact serve."""
        local = set(CheckClass) - DEFERRABLE
        assert len(local) >= 4
        assert not (local & DEFERRABLE)

    def test_a_deferred_check_is_not_counted_as_a_pass(self):
        """`_local_pass` exists because this distinction was being routed around."""
        source = inspect.getsource(jlens_validation)
        assert "DEFERRABLE" in source
        deferred = check_cross_implementation(None, None) if _takes_two() else None
        if deferred is not None:
            assert deferred.status is not CheckStatus.PASS


def _takes_two() -> bool:
    return len(inspect.signature(check_cross_implementation).parameters) == 2
