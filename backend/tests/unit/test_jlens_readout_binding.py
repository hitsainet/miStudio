"""
The bound readout endpoint (Phase 4.5) and the model registry behind it.

The endpoint used to return 501. Now that it serves, three things must hold and
each has a silent failure mode:

  * a JACOBIAN_LENS request must NEVER be answered with logit data. Falling
    back to identity produces a complete, plausible readout under the wrong
    label — a lower evidence rung in a higher rung's clothing (BR-019).
  * an artifact fitted for DIFFERENT WEIGHTS must be refused. An
    instruction-tuned variant is not the base model, and a lens from one
    applied to the other reads out fluently and wrongly.
  * only ONE model may be resident. This workbench shares a card with a serving
    process, and the previous logit-lens implementation failed outright the
    moment miLLM occupied it.

MUTATION CONTROLS (each must turn this file red):
  * fall back to IdentityTransport when no artifact exists -> "never logit under Jacobian" fails
  * drop the slug comparison in _validated_report          -> "weight identity" fails
  * make the model cache hold two entries                  -> "one model resident" fails
  * gate serving on `passed` instead of `serviceable`      -> "serviceable gate" fails
  * gate serving on nothing at all                         -> "refuses unvalidated" fails
"""

from __future__ import annotations

import pytest
import torch

from src.services import jlens_model_registry as registry
from src.services.jlens_validation import (
    CheckClass,
    CheckResult,
    CheckStatus,
    ValidationReport,
)


def result(check: CheckClass, status: CheckStatus = CheckStatus.PASS) -> CheckResult:
    return CheckResult(check, status, "x")


LOCAL = [
    CheckClass.STRUCTURAL,
    CheckClass.NAMING,
    CheckClass.ENVELOPE,
    CheckClass.SEMANTIC,
]
CONSUMER = [CheckClass.CROSS_IMPLEMENTATION, CheckClass.ROUND_TRIP]


# ── the two gates ──────────────────────────────────────────────────────────


def test_serviceable_and_passed_are_different_gates():
    """Local serving and external handover carry different risk.

    The consumer-interop classes need a live external consumer to run at all,
    so requiring them for local serving would make the Jacobian path
    permanently unreachable. Requiring only the local four for HANDOVER would
    let an unvalidated artifact reach a consumer that fails silently. Two
    gates, deliberately.
    """
    local_only = ValidationReport(
        [result(c) for c in LOCAL]
        + [result(c, CheckStatus.NOT_RUN) for c in CONSUMER]
    )
    assert local_only.serviceable is True
    assert local_only.passed is False

    everything = ValidationReport([result(c) for c in LOCAL + CONSUMER])
    assert everything.serviceable is True
    assert everything.passed is True


def test_serviceable_is_fail_closed_on_a_local_class_that_never_ran():
    partial = ValidationReport(
        [result(c) for c in LOCAL if c is not CheckClass.SEMANTIC]
        + [result(c, CheckStatus.NOT_RUN) for c in CONSUMER]
    )
    assert partial.serviceable is False


def test_a_failing_local_class_is_not_serviceable():
    """SEMANTIC is the one that catches a right-shaped, contentless artifact."""
    failing = ValidationReport(
        [result(c) for c in LOCAL if c is not CheckClass.SEMANTIC]
        + [result(CheckClass.SEMANTIC, CheckStatus.FAIL)]
        + [result(c, CheckStatus.NOT_RUN) for c in CONSUMER]
    )
    assert failing.serviceable is False


def test_serving_refuses_an_unvalidated_artifact(tmp_path):
    from src.services.jlens_artifact_service import (
        ArtifactNotValidated,
        JLensArtifactService,
    )

    svc = JLensArtifactService(tmp_path / "artifacts")
    with pytest.raises(ArtifactNotValidated):
        svc.load_for_readout("org/model", report=None)


# ── weight identity (Phase 4.2 / FPRD §3.4) ────────────────────────────────


def test_the_slug_separates_a_base_model_from_its_instruction_tuned_variant():
    """The check that stops a lens being applied to different weights.

    Comparing model NAMES loosely is what makes this mistake easy: the two
    repos differ by one suffix and the readout that results is fluent and
    wrong.
    """
    from src.services.jlens_artifact_service import slug_for

    assert slug_for("google/gemma-2-2b") != slug_for("google/gemma-2-2b-it")
    assert slug_for("LiquidAI/LFM2.5-1.2B") != slug_for("LiquidAI/LFM2.5-1.2B-Instruct")


def test_endpoint_refuses_an_artifact_id_that_is_not_this_models_slug():
    from src.api.v1.endpoints.jlens import _validated_report
    from src.services.jlens_artifact_service import ArtifactNotValidated

    class Loaded:
        name = "google/gemma-2-2b-it"
        d_model = 8
        n_layers = 3
        n_vocab = 512

    with pytest.raises(ArtifactNotValidated, match="different weights"):
        _validated_report(Loaded(), artifact_id="gemma-2-2b")


def test_a_missing_artifact_says_the_logit_lens_needs_none():
    """The error has to be actionable: the alternative lens requires nothing."""
    from src.api.v1.endpoints.jlens import _validated_report

    class Loaded:
        name = "org/absent-model"
        d_model = 8
        n_layers = 3
        n_vocab = 512

    with pytest.raises(FileNotFoundError, match="logit lens needs none"):
        _validated_report(Loaded(), artifact_id=None)


# ── the semantic fixture ───────────────────────────────────────────────────


def test_the_semantic_fixture_answer_is_absent_from_its_own_prompt():
    """Otherwise an artifact encoding NOTHING would pass the check.

    A token already present in the prompt is recoverable from the residual
    stream by the identity map, so a fixture whose answer appears in the prompt
    validates a broken lens.
    """
    from src.api.v1.endpoints.jlens import (
        SEMANTIC_FIXTURE_ANSWER,
        SEMANTIC_FIXTURE_PROMPT,
    )

    assert SEMANTIC_FIXTURE_ANSWER not in SEMANTIC_FIXTURE_PROMPT.lower()


# ── one model resident ─────────────────────────────────────────────────────


class FakeCacheEntry:
    def __init__(self, key):
        self.key = key


def test_the_model_cache_holds_exactly_one_entry():
    """Two resident models is the failure this cache exists to prevent.

    A readout needs the whole model for its forward pass, and this workbench
    shares a card with a serving process.
    """
    cache = registry._SingleEntryCache()
    a = cache.get_or_load("model-a", lambda: FakeCacheEntry("model-a"))
    assert cache.loaded_key == "model-a"

    b = cache.get_or_load("model-b", lambda: FakeCacheEntry("model-b"))
    assert cache.loaded_key == "model-b"
    assert b is not a

    # And the first is genuinely gone, not merely shadowed.
    reloaded = cache.get_or_load("model-a", lambda: FakeCacheEntry("model-a"))
    assert reloaded is not a


def test_evicting_releases_memory_rather_than_only_dropping_the_reference(monkeypatch):
    """Dropping the Python reference is NOT enough on CUDA.

    Without an explicit collect + empty_cache the freed blocks stay in torch's
    caching allocator, so the "one model resident" guarantee holds in Python
    while two models' worth of VRAM is still held — which is exactly the
    contention this cache exists to avoid, and it is invisible from the object
    graph.
    """
    released = []
    monkeypatch.setattr(registry, "_release_memory", lambda: released.append(1))

    cache = registry._SingleEntryCache()
    cache.get_or_load("a", lambda: FakeCacheEntry("a"))
    assert released == [], "nothing to release on the first load"

    cache.get_or_load("b", lambda: FakeCacheEntry("b"))
    assert released == [1], "eviction did not release the previous model's memory"


def test_clearing_the_cache_also_releases(monkeypatch):
    released = []
    monkeypatch.setattr(registry, "_release_memory", lambda: released.append(1))

    cache = registry._SingleEntryCache()
    cache.get_or_load("a", lambda: FakeCacheEntry("a"))
    cache.clear()
    assert released == [1]
    assert cache.loaded_key is None


def test_a_repeat_request_does_not_reload():
    """Loading per request would make every readout cost tens of seconds."""
    calls = []

    def loader():
        calls.append(1)
        return FakeCacheEntry("same")

    cache = registry._SingleEntryCache()
    cache.get_or_load("same", loader)
    cache.get_or_load("same", loader)
    assert len(calls) == 1


def test_capture_defaults_to_cpu_not_auto():
    """"auto" silently takes the GPU the moment one exists.

    A readout is an analysis operation that must never contend with serving for
    VRAM — the reason the readout itself is CPU-only.
    """
    import inspect

    default = inspect.signature(registry.load_for_readout).parameters["capture_device"].default
    assert default == "cpu"


def test_a_model_without_a_repo_id_is_refused_with_a_reason():
    class Record:
        id = "m_x"
        repo_id = None

    with pytest.raises(registry.ModelNotAvailable, match="repo_id"):
        registry.load_for_readout(Record())


def test_a_model_that_is_not_downloaded_is_refused_with_a_reason(monkeypatch):
    """409, not a crash: a forward pass needs the weights present."""

    class Record:
        id = "m_y"
        repo_id = "org/not-downloaded"
        file_path = None
        quantization = None

    registry.clear_cache()
    with pytest.raises(registry.ModelNotAvailable, match="not downloaded"):
        registry.load_for_readout(Record())


# ── rung discipline ────────────────────────────────────────────────────────


def test_jacobian_transport_refuses_to_fall_back_to_identity():
    """The defect BR-019 exists to prevent, at the one place it could happen."""
    from src.services.jlens_readout_service import JacobianTransport

    transport = JacobianTransport({0: torch.eye(4)})
    with pytest.raises(KeyError, match="identity"):
        transport.apply(torch.ones(4), layer=1)


def test_the_endpoint_no_longer_returns_501_for_readout():
    """The 501 was load-bearing while nothing was bound; it must be gone now.

    Asserted on the source rather than by calling the route, because calling it
    needs a database and a loaded model — and the thing at risk is that the
    stub survives a partial wiring.
    """
    import inspect

    from src.api.v1.endpoints import jlens

    source = inspect.getsource(jlens.readout)
    assert "501" not in source
    assert "NOT_IMPLEMENTED" not in source


# ── review round 3 ─────────────────────────────────────────────────────────


def test_validation_is_cached_per_artifact_identity():
    """Without a cache, EVERY Jacobian readout re-runs the whole suite.

    The SEMANTIC check is itself a full readout, so each request paid for two
    readouts plus a revalidation — correct, and unusably slow.

    The key includes mtime and size, so a REPLACED artifact is revalidated
    rather than served on a stale verdict. That is the property that makes the
    cache safe rather than merely fast.
    """
    import inspect

    from src.api.v1.endpoints import jlens

    source = inspect.getsource(jlens._validated_report)
    assert "_VALIDATION_CACHE" in source

    key_line = next(l for l in source.splitlines() if "key = (" in l)
    for part in ("st_mtime_ns", "st_size"):
        assert part in key_line, (
            f"the cache key omits {part}: a replaced artifact would be served "
            "on the previous artifact's verdict"
        )


def test_the_jacobian_transport_is_built_once_not_per_call():
    """The constructor casts every matrix; building it in the closure undoes that.

    JacobianTransport casts to the compute dtype ONCE on purpose, so `apply`
    does not copy a d_model^2 matrix per call — 8 MB at the reference model,
    thousands of times per readout. Constructing it inside the per-invocation
    closure moved that cost back, over the whole artifact.
    """
    import inspect

    from src.api.v1.endpoints import jlens

    source = inspect.getsource(jlens._semantic_check)
    build_line = next(
        i for i, l in enumerate(source.splitlines()) if "JacobianTransport(" in l
    )
    closure_line = next(
        i for i, l in enumerate(source.splitlines()) if "def top_at" in l
    )
    assert build_line < closure_line, (
        "JacobianTransport is constructed inside top_at, so every probe recasts "
        "the entire artifact"
    )


def test_an_empty_stream_fails_the_semantic_check_rather_than_raising_NameError():
    """The distinction this whole feature exists for.

    An empty readout must be a FAILED check with a reason, never an unbound
    local — and never an empty pass.
    """
    import inspect

    from src.api.v1.endpoints import jlens

    source = inspect.getsource(jlens._semantic_check)
    assert "last = None" in source
    assert "no token messages" in source


def test_the_model_is_loaded_in_ITS_OWN_dtype_not_a_forced_one():
    """Forcing fp16 onto a bfloat16 checkpoint leaves the model MIXED.

    The forward pass then dies with "expected scalar type BFloat16 but found
    Half" BEFORE any readout arithmetic runs — so the two casts in the readout
    path cannot save it. That is what gemma-2-2b-it did on the cluster.

    A readout does not need a particular precision, it needs the model to RUN,
    so the right dtype is whatever the checkpoint was saved in.
    """
    import inspect

    from src.services import jlens_model_registry

    source = inspect.getsource(jlens_model_registry.load_for_readout)
    assert 'dtype="auto"' in source, (
        "the readout loader forces a dtype; a checkpoint saved in another "
        "family will be internally mixed and its forward pass will raise"
    )
    # The forced-dtype path survives only as a FALLBACK, and says so.
    assert "falling back to the" in source


# ---------------------------------------------------------------------------
# A PARTIAL fit must be servable
#
# Found on the cluster with a published, semantically-valid artifact: the
# Jacobian readout was refused with "no serviceable validation report". The fit
# API, the MCP tool and the UI all accept a LAYER SUBSET, but both validation
# paths passed `expected_layers=range(model.n_layers)` — so a two-layer artifact
# failed STRUCTURAL with "missing layers [0..23]" and could never be served.
# The product offered a shape it then refused to honour.
#
# The semantic probe had the same assumption: it targeted mid-stack regardless
# of which layers the artifact contained, so on a partial fit it read out at a
# layer with no Jacobian to apply.
#
# MUTATION CONTROLS (each must turn this section red):
#   * expected_layers back to range(n_layers)   -> "partial fit validates" fails
#   * semantic layer back to a fixed mid        -> "probes a present layer" fails
# ---------------------------------------------------------------------------


def _committed_artifact(tmp_path, layers, d_model=8):
    """A real artifact directory, laid out the way `find` expects."""
    import torch

    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    ref = service.write_staged(
        "org/model",
        {l: torch.randn(d_model, d_model) for l in layers},
        "corpus: partial\n",
    )
    final = tmp_path / ref.slug
    ref.directory.rename(final)
    return service


class _Loaded:
    name = "org/model"
    d_model = 8
    n_layers = 26
    n_vocab = 256
    model = object()
    tokenizer = object()
    structure = object()
    unembedding = None


def test_a_partial_fit_validates_through_the_real_readout_binding(tmp_path, monkeypatch):
    """Calls `_validated_report` ITSELF — the function the readout uses.

    An earlier version of this test called `service.validate(...)` with the
    layer list already computed, which tested a reimplementation of the fix
    rather than the fix. Both mutation controls survived against it.
    """
    from src.api.v1.endpoints import jlens
    from src.services.jlens_validation import CheckClass, CheckResult, CheckStatus

    service = _committed_artifact(tmp_path, [24, 25])
    monkeypatch.setattr(jlens, "_service", lambda: service)
    jlens._VALIDATION_CACHE.clear()

    seen = {}

    def fake_semantic(loaded, ref, present=None):
        seen["present"] = list(present or [])
        return CheckResult(CheckClass.SEMANTIC, CheckStatus.PASS, "stubbed")

    monkeypatch.setattr(jlens, "_semantic_check", fake_semantic)

    report = jlens._validated_report(_Loaded(), None)

    structural = next(r for r in report.results if r.check.value == "structural")
    assert structural.status.value == "pass", (
        f"a two-layer artifact failed STRUCTURAL through the readout binding: "
        f"{structural.detail}. The fit API, the MCP tool and the UI all accept a "
        "layer subset, so a partial fit must be servable"
    )
    assert report.serviceable, f"partial artifact not serviceable: {report.summary()}"
    assert seen["present"] == [24, 25], (
        "the semantic check was not told which layers the artifact holds, so it "
        "cannot avoid probing one that is absent"
    )


def test_the_semantic_check_scans_only_layers_the_artifact_actually_holds(monkeypatch):
    """Calls `_semantic_check` ITSELF and records the layers it targets.

    The endpoint used to probe "about two thirds of the way up", with a comment
    asserting that was not a band constant. It was one — it asserts WHERE an
    unspoken intermediate must live, and BR-002 forbids this project assuming a
    band it has not measured for the model in front of it. It also cost a
    converged LFM2 artifact whose readout at that depth was the correct concept
    field with the token elsewhere in the stack.

    Scanning must still be confined to the layers PRESENT: reading out at a
    layer the artifact does not hold has no Jacobian to apply, and would fail
    for a reason that says nothing about the lens.

    MUTATION CONTROL: restore the `mid = max(0, int(n_layers * 2 / 3) - 1)`
    target, or pass `sorted(range(loaded.n_layers))` instead of `present`, and
    this fails.
    """
    from src.api.v1.endpoints import jlens
    from src.services import jlens_readout_service, jlens_validation

    targeted = {}

    def fake_check_semantic(
        readout, prompt, layers, expected_intermediate, top_k=8, control_prompt=None
    ):
        targeted["layers"] = layers
        targeted["control_prompt"] = control_prompt
        from src.services.jlens_validation import CheckClass, CheckResult, CheckStatus

        return CheckResult(CheckClass.SEMANTIC, CheckStatus.PASS, "stubbed")

    monkeypatch.setattr(jlens_validation, "check_semantic", fake_check_semantic)
    monkeypatch.setattr(
        jlens_readout_service, "ReadoutService", lambda **kw: object()
    )
    monkeypatch.setattr(
        jlens_readout_service, "JacobianTransport", lambda j, **kw: object()
    )

    class _Svc:
        def _load_payload(self, ref):
            return {24: None, 25: None}

        def layer_scales(self, ref):
            # No rescale recorded — the common case, and the one where an
            # artifact fitted before scales were written must still read.
            return {}

    monkeypatch.setattr(jlens, "_service", lambda: _Svc())

    # A PARTIAL artifact is scanned over exactly what it holds — never over the
    # model's full layer range.
    jlens._semantic_check(_Loaded(), object(), [24, 25])
    assert targeted["layers"] == [24, 25], (
        f"the probe scanned {targeted['layers']}, which is not the set the "
        "artifact holds — a readout there has no Jacobian to apply"
    )

    # A full fit scans the whole stack rather than picking a depth.
    jlens._semantic_check(_Loaded(), object(), list(range(26)))
    assert targeted["layers"] == list(range(26))

    # And the matched control travels with it, or the scan is a rubber stamp.
    assert targeted["control_prompt"] == jlens.SEMANTIC_FIXTURE_CONTROL
