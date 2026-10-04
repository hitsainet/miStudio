"""
An unrun consumer check must not be recorded as a PASS.

`commit` requires `passed`, and `passed` requires all six classes. The two
consumer-interop classes cannot run without a live external consumer, so
`_local_pass` stamped them `CheckStatus.PASS` with a detail string beginning
"deferred: …". `_write_report` then serialised

    {"check": "cross_implementation", "status": "pass"}

into `validation.json` — the file whose entire purpose is to TRAVEL WITH THE
ARTIFACT to a consumer that pulled it off HuggingFace. That consumer sees a green
six-class pass and can only learn otherwise by reading English prose in a
neighbouring field.

`CheckStatus`'s own comment says counting an unrun check as a pass is "far worse"
than blocking a good artifact. The value was carefully protected and then routed
around under a different name. `DEFERRED` is that state, recorded truthfully.

MUTATION CONTROLS (each must turn this file red):
  * stamp PASS in `defer_consumer_checks`  -> "not recorded as a PASS"
  * accept DEFERRED for any class          -> "the four LOCAL classes are not deferrable"
  * accept NOT_RUN the way DEFERRED is     -> "NOT_RUN is still never a pass"
  * alias cleared_for_handover to passed   -> "requires a LITERAL pass"
  * overwrite an already-run result        -> "a check that actually RAN is left alone"
"""

from __future__ import annotations

import json

import pytest

from src.services.jlens_validation import (
    DEFERRABLE,
    CheckClass,
    CheckResult,
    CheckStatus,
    ValidationReport,
    defer_consumer_checks,
)

LOCAL = (
    CheckClass.STRUCTURAL,
    CheckClass.NAMING,
    CheckClass.ENVELOPE,
    CheckClass.SEMANTIC,
)


def _locals_pass():
    return ValidationReport([CheckResult(c, CheckStatus.PASS, "ok") for c in LOCAL])


class TestDeferredIsNotAPass:
    def test_a_deferred_consumer_check_is_NOT_recorded_as_a_pass(self):
        """Asserted on the SERIALISED string, not the in-memory enum.

        `validation.json` is what leaves this machine. An enum that stringifies
        to "pass" in the file is a claim to a downstream reader no matter what
        the Python value is called.
        """
        out = defer_consumer_checks(_locals_pass())
        wire = json.loads(
            json.dumps(
                [
                    {"check": r.check.value, "status": r.status.value}
                    for r in out.results
                ]
            )
        )
        interop = [r for r in wire if r["check"] in {c.value for c in DEFERRABLE}]
        assert len(interop) == 2, wire
        for row in interop:
            assert row["status"] == "deferred", (
                f"{row['check']} serialises as {row['status']!r}; a consumer "
                "reading validation.json would be told an interop check succeeded"
            )

    def test_the_artifact_is_still_PUBLISHABLE(self):
        """Publishing behaviour is unchanged — only the recorded status differs.

        If this regressed, every fit would become unpublishable and the Jacobian
        path unreachable, which is the failure the deferral exists to avoid.
        """
        assert defer_consumer_checks(_locals_pass()).passed is True

    def test_it_is_NOT_cleared_for_handover(self):
        """The real BR-030 gate. False everywhere today, which is correct: no
        A5/A6 harness exists, so nothing has been checked against a consumer."""
        assert defer_consumer_checks(_locals_pass()).cleared_for_handover is False

    def test_cleared_for_handover_requires_a_LITERAL_pass(self):
        """MUTATION CONTROL: alias it to `passed` and this fails."""
        all_pass = ValidationReport(
            [CheckResult(c, CheckStatus.PASS, "ok") for c in CheckClass]
        )
        assert all_pass.cleared_for_handover is True
        assert all_pass.passed is True


class TestTheValveCannotWiden:
    def test_the_four_LOCAL_classes_are_NOT_deferrable(self):
        """The property that stops this from becoming a general escape hatch.

        `serviceable` gates miStudio's own readout on these four. A deferrable
        SEMANTIC would let an artifact that was never read out at all be served.

        MUTATION CONTROL: add SEMANTIC to DEFERRABLE and this fails.
        """
        for check in LOCAL:
            assert check not in DEFERRABLE, f"{check.value} became deferrable"

    def test_a_DEFERRED_local_class_does_not_pass(self):
        """Belt and braces on the same rule, from the `passed` side."""
        results = [CheckResult(c, CheckStatus.PASS, "ok") for c in LOCAL[:3]]
        results.append(
            CheckResult(CheckClass.SEMANTIC, CheckStatus.DEFERRED, "handwave")
        )
        results += [CheckResult(c, CheckStatus.DEFERRED, "no consumer") for c in DEFERRABLE]
        assert ValidationReport(results).passed is False

    def test_NOT_RUN_is_still_never_a_pass(self):
        """DEFERRED says "known unrunnable here, and we publish anyway".
        NOT_RUN says "we do not know". They must not collapse.

        MUTATION CONTROL: accept NOT_RUN the way DEFERRED is accepted -> fails.
        """
        results = [CheckResult(c, CheckStatus.PASS, "ok") for c in LOCAL]
        results += [CheckResult(c, CheckStatus.NOT_RUN, "?") for c in DEFERRABLE]
        assert ValidationReport(results).passed is False

    def test_serviceable_is_unaffected_by_deferral(self):
        out = defer_consumer_checks(_locals_pass())
        assert out.serviceable is True


class TestAnAlreadyRunCheckSurvives:
    def test_a_check_that_actually_RAN_is_left_alone(self):
        """When an A5/A6 harness finally runs these for real, deferring over the
        top would erase the only evidence this project has ever had for them.

        MUTATION CONTROL: overwrite unconditionally and this fails.
        """
        results = [CheckResult(c, CheckStatus.PASS, "ok") for c in LOCAL]
        results.append(
            CheckResult(
                CheckClass.CROSS_IMPLEMENTATION,
                CheckStatus.PASS,
                "agreed with the live consumer on 8/8 tokens",
            )
        )
        out = defer_consumer_checks(ValidationReport(results))
        cross = next(r for r in out.results if r.check is CheckClass.CROSS_IMPLEMENTATION)
        assert cross.status is CheckStatus.PASS
        assert "8/8" in cross.detail, "a real interop result was overwritten"
        # And with one genuinely run, the OTHER is still deferred.
        rt = next(r for r in out.results if r.check is CheckClass.ROUND_TRIP)
        assert rt.status is CheckStatus.DEFERRED

    def test_a_real_FAILURE_is_not_deferred_away(self):
        """The dangerous direction: an interop check that ran and FAILED must
        not be laundered into a publishable deferral."""
        results = [CheckResult(c, CheckStatus.PASS, "ok") for c in LOCAL]
        results.append(
            CheckResult(CheckClass.CROSS_IMPLEMENTATION, CheckStatus.FAIL, "disagreed")
        )
        out = defer_consumer_checks(ValidationReport(results))
        assert out.passed is False, "a FAILED interop check was deferred into a pass"


class TestTheFitWorkerUsesTheSharedHelper:
    def test_local_pass_is_gone(self):
        """Two copies of "what publishable means" is how two workers come to
        disagree. The acquisition path must not import a private helper.

        MUTATION CONTROL: reintroduce `_local_pass` and this fails.
        """
        from src.workers import jlens_fit_tasks

        assert not hasattr(jlens_fit_tasks, "_local_pass"), (
            "the private copy is back; the acquire path will grow a second one"
        )

    def test_the_fit_worker_CALLS_the_shared_helper(self):
        """Reachability, from the COMPILED CODE rather than from source text.

        A regex over source is the failure mode this project has written down
        twice: it matches nothing on an unexpected layout and asserts nothing.
        `co_names` is what the interpreter will actually look up when the
        function runs, so reformatting, renaming the argument or splitting the
        call across lines cannot fool it — and deleting the call empties it.

        PINS BOTH LINKS OF THE CHAIN. The commit site moved into
        `_validate_and_commit` when that sequence was extracted so a
        re-validation could reach it, and this test then failed while the
        BEHAVIOUR was unchanged — a bytecode probe cannot distinguish "moved"
        from "deleted" any better than a regex can. Following the call one
        frame down without also asserting the fit worker still reaches it would
        have left a hole exactly the size of the refactor: `_validate_and_commit`
        could keep its wrapper forever while nothing called it.

        MUTATION CONTROL: drop the `defer_consumer_checks(...)` wrapper at the
        commit site, OR stop `_fit_and_publish` calling `_validate_and_commit`,
        and this fails.
        """
        import dis

        from src.workers.jlens_fit_tasks import (
            _fit_and_publish,
            _validate_and_commit,
        )

        def _loads(fn, name):
            # THE NAME BEING PRESENT IS NOT ENOUGH — and the first version of
            # this test made exactly that mistake. `co_names` contains the name
            # because the function IMPORTS it, so replacing
            # `defer_consumer_checks(report)` with a bare `report` at the commit
            # site left this green. A guard an unused import satisfies is the
            # fail-open shape all over again.
            #
            # An import BINDS the name (STORE_FAST); only a use LOADS it. So a
            # LOAD of this name is the evidence that it is actually invoked.
            return [
                i
                for i in dis.get_instructions(fn)
                if i.opname.startswith("LOAD") and i.argval == name
            ]

        assert _loads(_validate_and_commit, "defer_consumer_checks"), (
            "the commit path imports defer_consumer_checks but never uses it; "
            "the report reaching commit still carries whatever the checks left"
        )
        assert _loads(_fit_and_publish, "_validate_and_commit"), (
            "the fit worker no longer reaches the shared validate-and-commit "
            "helper, so a fit publishes by some other route — or not at all"
        )
        # SANITY: this is still the function that publishes, or the assertion
        # above is checking the wrong body. Retargeted with the extraction —
        # `commit` moved into `_validate_and_commit` alongside the wrapper it
        # guards, and a sanity check left pointing at the old body would have
        # gone false for the same harmless reason the real assertion did.
        assert any(
            i.argval == "commit" for i in dis.get_instructions(_validate_and_commit)
        ), "this is no longer the function that publishes"
