"""MIS-E2E-079 — a requested freeze that applies to nothing must fail loudly.

`CLAUDE.md` claimed, as a shipped property:

    "`affine_residual` refuses a fit whose freeze leaked. An incomplete freeze
     yields a matrix of the right shape and size that passes STRUCTURAL/NAMING/
     ENVELOPE and reads out plausible nonsense; fit time is the only point where
     it is detectable."

No such gate existed. `max_affine_residual` was a constructor argument, assigned
to an attribute and never read; `affine_residual` appeared only in docstrings.

**Reinstating it would have been the wrong fix.** Freezing does not make the map
affine — the MLP activation stays non-linear — so a global-affine gate reports a
large departure for every real model and would refuse every genuine fit. That is
exactly why the check had been replaced by `linearisation_residual`, a recorded
DIAGNOSTIC. The audit finding read the absence correctly and proposed the wrong
remedy.

So: the dead threshold is removed (a configured value nothing reads is worse
than none — it reads like a guard, so nobody looks for the missing one), and the
hazard it was aimed at is closed by a check that is actually sound. Verifying the
patch LANDED is direct and certain; inferring it from the resulting matrix is
neither.

The specific failure this catches: `_norm_modules` selects the modules to freeze
by predicate. It once used a substring match. A model whose norm layers do not
match yields an EMPTY list — a `freeze_norms=True` fit that froze nothing, while
the artifact records `freeze_norms: true`.
"""

import inspect

import pytest
import torch

from src.ml import jlens_fitter as fitter
from src.ml.jlens_fitter import frozen_attention_and_norms


@pytest.fixture(autouse=True)
def _freeze_lock_must_not_leak():
    """Turn a leaked `_FREEZE_LOCK` into a failure instead of a hang.

    Found by mutation control C45: deleting the refusal path's unwind loop made
    this file HANG rather than fail. `_FREEZE_LOCK` is a plain `threading.Lock`
    acquired before the norm modules are collected, so a refusal that does not
    release it blocks the next test on `acquire()` — forever, with no output.

    A test that hangs is worse than one that fails: CI reports a timeout with no
    indication of which invariant broke, and a developer running the file
    locally sees nothing at all. This makes the failure legible and lets the
    rest of the file run.
    """
    if fitter._FREEZE_LOCK.locked():
        fitter._FREEZE_LOCK.release()
        pytest.fail("_FREEZE_LOCK was already held on entry — an earlier test leaked it")
    yield
    if fitter._FREEZE_LOCK.locked():
        fitter._FREEZE_LOCK.release()
        pytest.fail(
            "this test leaked _FREEZE_LOCK — the next fit in this process would "
            "block forever, silently"
        )


class _NoNorms(torch.nn.Module):
    """A model with nothing `_norm_modules` will match."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class _WithNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(4)


# ── The dead threshold is gone ─────────────────────────────────────────────

def test_the_unread_affine_threshold_is_removed():
    """A configured threshold compared to nothing reads like a guard."""
    assert not hasattr(fitter, "MAX_AFFINE_RESIDUAL")
    sig = inspect.signature(fitter.JacobianFitter.__init__)
    assert "max_affine_residual" not in sig.parameters


def test_the_diagnostic_that_replaced_it_still_exists():
    """Negative control for the test above.

    Removing the threshold must not have removed the local-linearisation
    measurement with it — that is what the artifact records, and it is the only
    quantitative statement about how far the lens can be trusted.
    """
    assert callable(fitter.linearisation_residual)


# ── A freeze that applies to nothing is refused ────────────────────────────

def test_freeze_norms_on_a_model_with_no_norm_modules_raises():
    """The silent leak: nothing to freeze, and the artifact says frozen."""
    with pytest.raises(RuntimeError, match="no norm module"):
        with frozen_attention_and_norms(_NoNorms(), freeze_qk=False, freeze_norms=True):
            pass


def test_freeze_norms_succeeds_when_there_is_something_to_freeze():
    """A refusal-only test would pass against a gate that refuses everything."""
    with frozen_attention_and_norms(_WithNorm(), freeze_qk=False, freeze_norms=True):
        pass


def test_freeze_qk_verifies_the_sdpa_patch_is_in_place():
    """If something replaced SDPA after we patched it, the fit is not frozen."""
    original = torch.nn.functional.scaled_dot_product_attention
    try:
        with frozen_attention_and_norms(_WithNorm(), freeze_qk=True, freeze_norms=False):
            assert (
                torch.nn.functional.scaled_dot_product_attention is not original
            ), "freeze_qk must actually patch SDPA"
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


def test_freeze_qk_refuses_when_the_sdpa_patch_is_displaced(monkeypatch):
    """Pin the freeze_qk half of the gate — mutation control C44 SURVIVED.

    The check compares the global against the function just assigned to it, and
    the only code that runs in between is norm freezing. So the branch is
    unreachable unless something in that window replaces SDPA — which is exactly
    what it is there to catch, and what this simulates by having `_freeze_norm`
    steal it.

    Contrived, deliberately. The alternative was an assertion no test could ever
    reach, which is the same category of defect as the gate that was missing:
    something that reads like a control and is not one.
    """
    original = torch.nn.functional.scaled_dot_product_attention

    def _thief(module):
        torch.nn.functional.scaled_dot_product_attention = original
        return lambda: None

    monkeypatch.setattr(fitter, "_freeze_norm", _thief)

    try:
        with pytest.raises(RuntimeError, match="SDPA patch is not in place"):
            with frozen_attention_and_norms(_WithNorm(), freeze_qk=True, freeze_norms=True):
                pass
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


def test_no_freeze_requested_touches_nothing_and_does_not_raise():
    """With both flags off there is nothing to verify and no lock to take."""
    original = torch.nn.functional.scaled_dot_product_attention
    with frozen_attention_and_norms(_NoNorms(), freeze_qk=False, freeze_norms=False):
        assert torch.nn.functional.scaled_dot_product_attention is original


def test_a_refused_freeze_releases_the_lock():
    """The refusal path must undo what it already did.

    `_FREEZE_LOCK` is acquired before the norm modules are collected, so raising
    without releasing would deadlock the NEXT fit — turning a loud, correct
    refusal into a silent permanent hang.
    """
    with pytest.raises(RuntimeError):
        with frozen_attention_and_norms(_NoNorms(), freeze_qk=False, freeze_norms=True):
            pass

    assert not fitter._FREEZE_LOCK.locked(), (
        "the lock survived a refused freeze — the next fit would block forever"
    )

    # And the next fit still works.
    with frozen_attention_and_norms(_WithNorm(), freeze_qk=False, freeze_norms=True):
        pass


class TestTheFreezeContextLeaksNothingOnFailure:
    """MIS-E2E-082 / task 15.1.

    The lock acquisition and the process-wide SDPA patch sat ABOVE the `try`.
    An exception between them and the `yield` escaped without running the
    `finally`, so the attention patch stayed installed for every subsequent
    forward pass in that worker and `_FREEZE_LOCK` was never released, leaving
    every later fit blocked on a lock nobody held. Both silent and permanent;
    the second presents only as a hung worker.
    """

    def _model_with_no_norms(self):
        import torch.nn as nn

        class NoNorms(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)

        return NoNorms()

    def test_a_failure_before_yield_releases_the_lock(self):
        import pytest

        from src.ml.jlens_fitter import _FREEZE_LOCK, frozen_attention_and_norms

        assert not _FREEZE_LOCK.locked(), "the lock was already held on entry"
        with pytest.raises(RuntimeError, match="no norm module"):
            with frozen_attention_and_norms(self._model_with_no_norms(),
                                            freeze_qk=False, freeze_norms=True):
                pass  # pragma: no cover
        assert not _FREEZE_LOCK.locked(), (
            "_FREEZE_LOCK is still held after a failed freeze; every later fit "
            "in this process blocks on it forever"
        )

    def test_a_failure_before_yield_restores_sdpa(self):
        import pytest
        import torch

        from src.ml.jlens_fitter import frozen_attention_and_norms

        original = torch.nn.functional.scaled_dot_product_attention
        with pytest.raises(RuntimeError, match="no norm module"):
            with frozen_attention_and_norms(self._model_with_no_norms(),
                                            freeze_qk=True, freeze_norms=True):
                pass  # pragma: no cover
        assert torch.nn.functional.scaled_dot_product_attention is original, (
            "the process-wide SDPA patch survived a failed freeze; every model "
            "in this worker now runs under frozen-Q/K attention"
        )

    def test_the_lock_is_reusable_after_repeated_failures(self):
        import pytest

        from src.ml.jlens_fitter import _FREEZE_LOCK, frozen_attention_and_norms

        for _ in range(3):
            with pytest.raises(RuntimeError):
                with frozen_attention_and_norms(self._model_with_no_norms(),
                                                freeze_qk=False, freeze_norms=True):
                    pass  # pragma: no cover
        assert _FREEZE_LOCK.acquire(blocking=False), (
            "the lock cannot be taken after three failed freezes"
        )
        _FREEZE_LOCK.release()

    def test_it_does_not_double_release(self):
        """Moving the try up made the finally re-run the manual cleanup.

        That surfaced as `RuntimeError: release unlocked lock` masking the real
        error — a worse failure than the leak it replaced.
        """
        import pytest

        from src.ml.jlens_fitter import frozen_attention_and_norms

        with pytest.raises(RuntimeError) as exc:
            with frozen_attention_and_norms(self._model_with_no_norms(),
                                            freeze_qk=False, freeze_norms=True):
                pass  # pragma: no cover
        assert "release unlocked lock" not in str(exc.value)
