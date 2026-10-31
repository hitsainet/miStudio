"""
Artifact lifecycle: stage, validate, commit, serve.

The property under test throughout is that NOTHING REACHES A CONSUMER WITHOUT A
FULL PASS. The consumer's lens loading is best-effort and fails at request time
without raising, so every shortcut here — publishing early, serving without a
report, leaving a half-written directory mounted — surfaces as a feature that
quietly returns nothing.

MUTATION CONTROLS (each must turn this file red):
  * commit without checking report.passed          -> "refuses to publish" fails
  * treat a NOT_RUN class as a pass                 -> "not run blocks publish" fails
  * load_for_readout defaulting report=None to ok   -> "serve refuses" fails
  * include staging dirs in list_artifacts          -> "staging is invisible" fails
  * accept a directory with two lens files          -> "ambiguous" fails
  * torch.load without weights_only                 -> "weights only" fails
"""

from __future__ import annotations

import inspect
import json
import pathlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.services.jlens_artifact_service import (
    ArtifactNotValidated,
    JLensArtifactService,
    slug_for,
)
from src.services.jlens_validation import (
    CheckClass,
    CheckResult,
    CheckStatus,
    ValidationReport,
)

D_MODEL = 8
LAYERS = [0, 1, 2]
N_VOCAB = 512


def jacobians():
    return {i: torch.eye(D_MODEL, dtype=torch.float16) for i in LAYERS}


def live_passes():
    """The three checks that need a loaded model or a running consumer."""
    return {
        "semantic_result": CheckResult(CheckClass.SEMANTIC, CheckStatus.PASS, "ok"),
        "cross_impl_result": CheckResult(
            CheckClass.CROSS_IMPLEMENTATION, CheckStatus.PASS, "ok"
        ),
        "round_trip_result": CheckResult(CheckClass.ROUND_TRIP, CheckStatus.PASS, "ok"),
    }


def service(tmp_path: Path) -> JLensArtifactService:
    return JLensArtifactService(tmp_path / "artifacts")


# ------------------------------------------------------------------- slug


def test_slug_matches_the_consumer_convention():
    assert slug_for("LiquidAI/LFM2.5-1.2B-Instruct") == "lfm2.5-1.2b-instruct"
    assert slug_for("google/gemma-2-2b-it") == "gemma-2-2b-it"


def test_an_unslugabble_id_is_refused_not_silently_emptied():
    with pytest.raises(ValueError):
        slug_for("///")


# ---------------------------------------------------------------- staging


def test_a_staged_artifact_is_invisible_to_discovery(tmp_path: Path):
    """Half-written artifacts must not be servable.

    The consumer mounts the directory and reads whatever is there, without
    reporting what it found.
    """
    svc = service(tmp_path)
    svc.write_staged("org/model", jacobians(), "recipe: test")

    assert svc.list_artifacts() == []
    assert svc.find("org/model") is None


def test_committing_makes_it_discoverable(tmp_path: Path):
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())

    published = svc.commit("org/model", report)
    assert published.lens_path.exists()
    assert [a.slug for a in svc.list_artifacts()] == ["model"]
    assert svc.find("org/model") is not None
    assert not svc.staging_dir("org/model").exists()


def test_restaging_replaces_a_previous_stage(tmp_path: Path):
    svc = service(tmp_path)
    svc.write_staged("org/model", jacobians(), "first")
    ref = svc.write_staged("org/model", jacobians(), "second")
    assert ref.config_path.read_text() == "second"


# -------------------------------------------------------------- publishing


def test_commit_refuses_to_publish_a_failing_report(tmp_path: Path):
    """The last point at which a bad artifact can be stopped by anything."""
    svc = service(tmp_path)
    svc.write_staged("org/model", jacobians(), "recipe: test")
    failing = ValidationReport(
        [CheckResult(c, CheckStatus.FAIL, "no") for c in CheckClass]
    )

    with pytest.raises(ArtifactNotValidated, match="fails silently|refusing"):
        svc.commit("org/model", failing)
    assert svc.find("org/model") is None


def test_a_not_run_class_blocks_publication(tmp_path: Path):
    """"We did not check" must never publish like "we checked and it was fine".

    This is the default path: the three live checks are absent unless supplied,
    so an artifact validated without a model or a consumer cannot be published
    at all.
    """
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB)  # no live checks

    assert report.passed is False
    assert set(report.missing) == set()
    statuses = {r.check: r.status for r in report.results}
    assert statuses[CheckClass.SEMANTIC] is CheckStatus.NOT_RUN
    assert statuses[CheckClass.ROUND_TRIP] is CheckStatus.NOT_RUN

    with pytest.raises(ArtifactNotValidated):
        svc.commit("org/model", report)


def test_commit_without_a_stage_is_an_error_not_a_silent_noop(tmp_path: Path):
    svc = service(tmp_path)
    passing = ValidationReport([CheckResult(c, CheckStatus.PASS, "ok") for c in CheckClass])
    with pytest.raises(FileNotFoundError):
        svc.commit("org/model", passing)


# -------------------------------------------------------------- validation


def test_validation_catches_a_wrong_d_model(tmp_path: Path):
    svc = service(tmp_path)
    wrong = {i: torch.eye(4, dtype=torch.float16) for i in LAYERS}
    ref = svc.write_staged("org/model", wrong, "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())

    assert report.passed is False
    structural = next(r for r in report.results if r.check is CheckClass.STRUCTURAL)
    assert "d_model is 8" in structural.detail


def test_validation_catches_a_missing_layer(tmp_path: Path):
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", {0: torch.eye(D_MODEL)}, "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())
    assert report.passed is False


def test_a_corrupt_artifact_fails_structurally_rather_than_raising(tmp_path: Path):
    """A file that does not deserialize is a FAIL, not an exception at serve."""
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    ref.lens_path.write_bytes(b"not a torch file")

    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())
    assert report.passed is False
    structural = next(r for r in report.results if r.check is CheckClass.STRUCTURAL)
    assert structural.status is CheckStatus.FAIL


def test_artifacts_are_loaded_weights_only():
    """An artifact is an untrusted file this process is about to load.

    The unrestricted loader executes pickled code, so this is a security
    property rather than a preference — asserted on the source because a
    behavioural test would need a malicious pickle.
    """
    source = inspect.getsource(JLensArtifactService._load_payload)
    assert "weights_only=True" in source


# ----------------------------------------------------------------- serving


def test_serving_refuses_without_a_passing_report(tmp_path: Path):
    """Serving an unvalidated artifact is the failure BR-030 exists for."""
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())
    svc.commit("org/model", report)

    with pytest.raises(ArtifactNotValidated):
        svc.load_for_readout("org/model", report=None)

    failing = ValidationReport([CheckResult(c, CheckStatus.FAIL, "no") for c in CheckClass])
    with pytest.raises(ArtifactNotValidated):
        svc.load_for_readout("org/model", report=failing)


def test_serving_returns_tensors_keyed_by_int_layer(tmp_path: Path):
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())
    svc.commit("org/model", report)

    loaded = svc.load_for_readout("org/model", report=report)
    assert sorted(loaded) == LAYERS
    assert all(isinstance(k, int) for k in loaded)
    assert loaded[0].shape == (D_MODEL, D_MODEL)


def test_serving_an_absent_artifact_raises(tmp_path: Path):
    svc = service(tmp_path)
    passing = ValidationReport([CheckResult(c, CheckStatus.PASS, "ok") for c in CheckClass])
    with pytest.raises(FileNotFoundError):
        svc.load_for_readout("org/absent", report=passing)


# --------------------------------------------------------------- ambiguity


def test_a_directory_with_two_lens_files_is_not_an_artifact(tmp_path: Path):
    """The consumer picks among them without saying which."""
    svc = service(tmp_path)
    ref = svc.write_staged("org/model", jacobians(), "recipe: test")
    report = svc.validate(ref, D_MODEL, LAYERS, N_VOCAB, **live_passes())
    published = svc.commit("org/model", report)

    (published.directory / "other_jacobian_lens.pt").write_bytes(b"x")
    assert svc.find("org/model") is None
    assert svc.list_artifacts() == []


# ---------------------------------------------------------------------------
# The verdict is recorded with the artifact, and identity-checked
#
# A published, semantically-valid partial artifact was refused at read time.
# The fit had validated it with a fixture the caller chose for the layers they
# fitted; that verdict was discarded and the readout re-validated with its own
# hard-coded fixture, which targets mid-stack — a question a top-of-stack fit
# was never fitted to answer. It failed a different test than the one it passed.
#
# MUTATION CONTROLS (each must turn this section red):
#   * commit stops writing validation.json      -> "records the verdict" fails
#   * stored_report skips the identity check     -> "a swapped file" fails
#   * the refusal drops the failing detail       -> "names the failing class" fails
# ---------------------------------------------------------------------------


def _passing_report():
    from src.services.jlens_validation import (
        CheckClass,
        CheckResult,
        CheckStatus,
        ValidationReport,
    )

    return ValidationReport(
        [
            CheckResult(c, CheckStatus.PASS, f"{c.value} ok")
            for c in CheckClass
        ]
    )


def _staged(tmp_path, service, layers=(24, 25)):
    import torch

    return service.write_staged(
        "org/model",
        {l: torch.randn(4, 4) for l in layers},
        "corpus: test\n",
    )


def test_commit_records_the_verdict_beside_the_artifact(tmp_path):
    from src.services.jlens_artifact_service import (
        VALIDATION_FILE,
        JLensArtifactService,
    )

    service = JLensArtifactService(tmp_path)
    _staged(tmp_path, service)
    ref = service.commit("org/model", _passing_report())

    assert (ref.directory / VALIDATION_FILE).is_file(), (
        "commit published without recording the verdict, so the readout must "
        "re-derive it — and re-derives it with a different fixture"
    )
    stored = service.stored_report(ref)
    assert stored is not None and stored["serviceable"] is True


def test_a_swapped_lens_file_invalidates_the_recorded_verdict(tmp_path):
    """Serving a lens fitted for other weights is what this gate prevents."""
    import torch

    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    _staged(tmp_path, service)
    ref = service.commit("org/model", _passing_report())
    assert service.stored_report(ref) is not None

    # Replace the weights, leaving the verdict beside them untouched.
    torch.save({24: torch.randn(4, 4), 25: torch.randn(4, 4)}, ref.lens_path)

    assert service.stored_report(ref) is None, (
        "a replaced lens file was still served on the OLD verdict; that verdict "
        "describes different weights, which produces a complete, plausible "
        "readout that is wrong"
    )


def test_the_refusal_names_the_failing_class(tmp_path):
    """A refusal the user cannot act on is only half a guard."""
    import pytest

    from src.api.v1.endpoints.jlens import _StoredReport
    from src.services.jlens_artifact_service import (
        ArtifactNotValidated,
        JLensArtifactService,
    )

    service = JLensArtifactService(tmp_path)
    failed = _StoredReport(
        {
            "passed": False,
            "serviceable": False,
            "summary": "semantic=fail",
            "results": [
                {
                    "check": "semantic",
                    "status": "fail",
                    "detail": "'spider' absent from the top-8 at layer 24",
                }
            ],
        }
    )

    with pytest.raises(ArtifactNotValidated) as excinfo:
        service.load_for_readout("org/model", report=failed)

    message = str(excinfo.value)
    assert "semantic" in message, f"the refusal names no failing class: {message}"
    assert "spider" in message, (
        f"the refusal drops the check's own detail: {message}. Without it the "
        "user cannot tell a missing report from a failed check"
    )


# ---------------------------------------------------------------------------
# A refit must not silently destroy coverage
#
# FROM THE CLUSTER LOGS, not from review:
#   2026-08-01 12:06:41  published lfm2.5-1.2b-instruct, layers 0..15, 134 MB
#   2026-08-02 12:57:52  published lfm2.5-1.2b-instruct, layers [1,2,3,10..15]
#
# The second `shutil.rmtree`'d the first. Nine minutes of GPU and the reference
# model's only FULL-STACK lens, gone with no warning, no backup and no way back
# — and the replacement does not dominate it (16 layers/120 prompts vs
# 9 layers/400 prompts, neither strictly better).
#
# MUTATION CONTROLS (each must turn this section red):
#   * commit rmtree's the old dir instead of archiving  -> "archives" fails
#   * the coverage guard is removed                      -> "refuses" fails
#   * .superseded is not excluded from discovery         -> "hidden" fails
#   * allow_coverage_loss is ignored                     -> "override" fails
# ---------------------------------------------------------------------------


def _commit_layers(tmp_path, layers, allow_loss=False):
    """Stage and commit an artifact covering exactly `layers`."""
    import torch

    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    service.write_staged(
        "org/model", {l: torch.randn(4, 4) for l in layers}, "corpus: t\n"
    )
    return service, service.commit(
        "org/model", _passing_report(), allow_coverage_loss=allow_loss
    )


def test_a_refit_that_loses_layers_is_REFUSED(tmp_path):
    from src.services.jlens_artifact_service import ArtifactCoverageLoss

    service, _ = _commit_layers(tmp_path, range(16))

    import torch

    service.write_staged(
        "org/model",
        {l: torch.randn(4, 4) for l in [1, 2, 3, 10, 11, 12, 13, 14, 15]},
        "corpus: t\n",
    )

    with pytest.raises(ArtifactCoverageLoss) as excinfo:
        service.commit("org/model", _passing_report())

    message = str(excinfo.value)
    # The refusal must NAME the layers at risk — "coverage would be reduced"
    # sends the user back to the logs to work out what they nearly lost.
    for layer in (0, 4, 5, 6, 7, 8, 9):
        assert str(layer) in message, f"layer {layer} not named in: {message}"

    # And the existing artifact must be untouched by a refused commit.
    payload = service._load_payload(service.find("org/model"))
    assert sorted(int(k) for k in payload) == list(range(16))


def test_the_refusal_can_be_overridden_deliberately(tmp_path):
    """Losing coverage is allowed — it just has to be a DECISION."""
    import torch

    service, _ = _commit_layers(tmp_path, range(16))
    service.write_staged("org/model", {l: torch.randn(4, 4) for l in [1, 2]}, "c: t\n")

    ref = service.commit("org/model", _passing_report(), allow_coverage_loss=True)
    payload = service._load_payload(ref)
    assert sorted(int(k) for k in payload) == [1, 2]


def test_a_superseded_artifact_is_ARCHIVED_not_deleted(tmp_path):
    """The replaced artifact survives one generation, so a mistake is undoable."""
    import torch

    from src.services.jlens_artifact_service import SUPERSEDED_SUFFIX

    service, _ = _commit_layers(tmp_path, range(16))
    service.write_staged("org/model", {l: torch.randn(4, 4) for l in [1, 2]}, "c: t\n")
    service.commit("org/model", _passing_report(), allow_coverage_loss=True)

    archive = tmp_path / f"model{SUPERSEDED_SUFFIX}"
    assert archive.is_dir(), (
        "the replaced artifact was deleted outright; nine minutes of GPU and a "
        "full-stack lens went with it the first time this happened"
    )
    recovered = torch.load(
        next(archive.glob("*_jacobian_lens.pt")), weights_only=True
    )
    # THROUGH THE NORMALISER, like every reader in the system. This asserted
    # layer indices at the TOP LEVEL, which coupled a test about ARCHIVAL to the
    # emitted on-disk layout — and that layout changed deliberately, so the
    # published form is loadable by a conformant consumer.
    from src.services.jlens_artifact_service import normalise_payload

    assert sorted(normalise_payload(recovered)) == list(range(16))


def test_the_archive_is_hidden_from_discovery(tmp_path):
    """Two directories for one model would let the consumer pick either."""
    import torch

    service, _ = _commit_layers(tmp_path, range(16))
    service.write_staged("org/model", {l: torch.randn(4, 4) for l in [1, 2]}, "c: t\n")
    service.commit("org/model", _passing_report(), allow_coverage_loss=True)

    slugs = [a.slug for a in service.list_artifacts()]
    assert slugs == ["model"], (
        f"discovery returned {slugs}; a superseded artifact must not be "
        "servable, or a stale lens is one directory listing away from being used"
    )


def test_a_refit_that_ADDS_layers_is_not_refused(tmp_path):
    """Negative control: the guard must not block a genuine upgrade."""
    import torch

    service, _ = _commit_layers(tmp_path, [1, 2])
    service.write_staged(
        "org/model", {l: torch.randn(4, 4) for l in range(16)}, "c: t\n"
    )
    ref = service.commit("org/model", _passing_report())
    payload = service._load_payload(ref)
    assert sorted(int(k) for k in payload) == list(range(16))


# ---------------------------------------------------------------------------
# The fp16 storage scale must SURVIVE to the readout (F2)
#
# `_to_storage_dtype` divides each matrix down so the fp16 cast cannot saturate
# — GPT-2's layer-6 Jacobian peaks around 1.7e7 against fp16's 65504 ceiling —
# and its docstring has always said "The scale is stored in the artifact's
# config.yaml". It was not. `FitResult.scales` was computed, returned, and
# dropped on the floor.
#
# Ranked readouts never noticed: the model's final norm divides a positive
# scalar straight back out, so `softmax(W_U @ norm(alpha * J @ h))` is exactly
# `softmax(W_U @ norm(J @ h))`. Everything that does NOT normalise did notice —
# probe scores and intervention magnitudes came out scaled by an unrecorded
# per-layer alpha and were not comparable across layers.
#
# MUTATION CONTROLS (each must turn this section red):
#   * stop writing layer_scales into config.yaml -> "round trip" fails
#   * JacobianTransport ignores `scales`         -> "unscales" fails
# ---------------------------------------------------------------------------


def test_layer_scales_round_trip_through_the_written_config(tmp_path):
    """Write a fit's scales, read them back off disk."""
    import torch

    from src.services.jlens_artifact_service import JLensArtifactService
    from src.workers.jlens_fit_tasks import _config_yaml

    class _Result:
        # `_config_yaml` records the fitted layers from this.
        jacobians = {24: None, 25: None}
        degenerate_layers = [25]
        scales = {24: 1.0, 25: 259.4}
        prompts_seen = 100
        converged = False
        convergence_delta = 1e-3
        position_spread_mean = {24: 0.01, 25: 0.02}
        position_spread_max = {24: 0.03, 25: 0.04}

    class _Loaded:
        name = "org/model"
        d_model = 4
        n_layers = 26
        n_vocab = 256
        model = None
        structure = type("S", (), {"num_layers": 26, "attention_module": None})()

    service = JLensArtifactService(tmp_path)
    ref = service.write_staged(
        "org/model",
        {24: torch.randn(4, 4), 25: torch.randn(4, 4)},
        _config_yaml(_Loaded(), _Result(), freeze_qk=True, corpus_name="t"),
    )

    recovered = service.layer_scales(ref)
    assert recovered == {24: 1.0, 25: 259.4}, (
        f"scales did not survive the write/read round trip: {recovered}. An "
        "unrecorded scale makes every probe and intervention magnitude wrong "
        "by a per-layer factor, and the artifact unreconstructible"
    )


def test_the_transport_undoes_the_storage_scale():
    """A scaled matrix read back unscaled is the stored J, not the fitted J."""
    import torch

    from src.services.jlens_readout_service import JacobianTransport

    true_j = torch.eye(4) * 3.0
    alpha = 100.0
    stored = true_j / alpha  # what _to_storage_dtype would have written

    unscaled = JacobianTransport({7: stored}, scales={7: alpha})
    h = torch.ones(4)
    assert torch.allclose(unscaled.apply(h, 7), true_j @ h, atol=1e-4), (
        "the transport did not undo the storage scale; probe scores and "
        "intervention magnitudes are off by that factor"
    )

    # And an artifact with no recorded scale is read as-is rather than refused,
    # so lenses fitted before the scale was written stay usable.
    plain = JacobianTransport({7: stored})
    assert torch.allclose(plain.apply(h, 7), stored @ h, atol=1e-6)


def test_ranking_is_invariant_to_the_scale_but_probing_is_not():
    """Why this went unnoticed, pinned so the reasoning is not re-derived."""
    import torch

    from src.services.jlens_readout_service import JacobianTransport

    j = torch.randn(6, 6)
    h = torch.randn(6)
    alpha = 250.0

    scaled = JacobianTransport({0: j / alpha}, scales={0: alpha}).apply(h, 0)
    unscaled = JacobianTransport({0: j / alpha}).apply(h, 0)

    # RMS-normalised, the two are identical — which is exactly why every ranked
    # readout looked correct while the magnitudes were wrong.
    def rms_norm(x):
        return x / x.pow(2).mean().sqrt().clamp_min(1e-6)

    assert torch.allclose(rms_norm(scaled), rms_norm(unscaled), atol=1e-4)
    # Unnormalised — a probe score — they differ by the factor.
    assert not torch.allclose(scaled, unscaled, atol=1e-3)


# ---------------------------------------------------------------------------
# A RECIPE CHANGE IS NOT DATA LOSS.
#
# The first paper-aligned LFM2 fit converged over 888 prompts, passed every
# local class including SEMANTIC, and was then refused publication because the
# previous artifact held layer 15 and this one did not. The refusal advised
# "fit the missing layers as well" — impossible: under a PENULTIMATE target,
# layer 15 is above the target, its Jacobian to that target is zero by
# causality, and the fitter refuses to fit it at all.
# ---------------------------------------------------------------------------


def _recipe(target: str, n_layers: int = 16) -> str:
    return f"corpus: t\nn_layers: {n_layers}\ntarget_layer: {target}\n"


def test_a_penultimate_refit_over_a_final_target_artifact_PUBLISHES(tmp_path):
    """The dropped layer is above the new target, so it is scope, not loss.

    MUTATION CONTROL: make `_coverage_delta` return `(missing, [])` and this
    fails with ArtifactCoverageLoss — the state that blocked a good artifact.
    """
    import torch

    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    service.write_staged(
        "org/model",
        {l: torch.randn(4, 4) for l in range(16)},
        _recipe("final"),
    )
    service.commit("org/model", _passing_report())

    # The aligned refit: 0..14, targeting the penultimate block of a 16-layer model.
    service.write_staged(
        "org/model",
        {l: torch.randn(4, 4) for l in range(15)},
        _recipe("penultimate"),
    )
    service.commit("org/model", _passing_report())  # must NOT raise

    payload = service._load_payload(service.find("org/model"))
    assert sorted(int(k) for k in payload) == list(range(15))

    # And the artifact it replaced is archived, not destroyed.
    archived = tmp_path / "model.superseded"
    assert archived.is_dir(), "the previous artifact was deleted rather than archived"


def test_a_GENUINE_partial_refit_is_still_refused_when_the_recipe_is_readable(tmp_path):
    """Excusing layers above the target must not excuse a hole below it.

    This is the guard the change above could have destroyed: same readable
    recipe, but the missing layers are ones this target could have covered.

    MUTATION CONTROL: excuse every missing layer regardless of the ceiling and
    this fails.
    """
    import torch

    import pytest as _pytest

    from src.services.jlens_artifact_service import (
        ArtifactCoverageLoss,
        JLensArtifactService,
    )

    service = JLensArtifactService(tmp_path)
    service.write_staged(
        "org/model", {l: torch.randn(4, 4) for l in range(16)}, _recipe("final")
    )
    service.commit("org/model", _passing_report())

    # Penultimate target, but only 0..8 fitted: layers 9..14 are BELOW the
    # ceiling of 14 and are real loss. 15 is above it and is not.
    service.write_staged(
        "org/model",
        {l: torch.randn(4, 4) for l in range(9)},
        _recipe("penultimate"),
    )
    with _pytest.raises(ArtifactCoverageLoss) as excinfo:
        service.commit("org/model", _passing_report())

    message = str(excinfo.value)
    for layer in range(9, 15):
        assert str(layer) in message, f"layer {layer} not named in: {message}"


def test_an_unreadable_recipe_excuses_NOTHING(tmp_path):
    """Fail closed: no recipe to appeal to means no layer may be dropped.

    Treating an unparseable config as permission to drop layers would make the
    guard decorative exactly when the artifact is least trustworthy.

    MUTATION CONTROL: return `([], missing)` when `_target_index` is None and
    this fails.
    """
    import torch

    import pytest as _pytest

    from src.services.jlens_artifact_service import (
        ArtifactCoverageLoss,
        JLensArtifactService,
    )

    service = JLensArtifactService(tmp_path)
    service.write_staged(
        "org/model", {l: torch.randn(4, 4) for l in range(16)}, _recipe("final")
    )
    service.commit("org/model", _passing_report())

    # No target_layer and no n_layers -> nothing to reason with.
    service.write_staged(
        "org/model", {l: torch.randn(4, 4) for l in range(15)}, "corpus: t\n"
    )
    with _pytest.raises(ArtifactCoverageLoss):
        service.commit("org/model", _passing_report())


# ---------------------------------------------------------------------------
# PUBLISHING IS LAST-WRITER-WINS, AND "LAST" IS NOT "BEST".
#
# 2026-08-04: a 400-prompt fit that never converged published over a
# 1097-prompt fit that did. The weaker job had been queued HOURS EARLIER, sat
# unclaimed in Redis through a series of pod rolls, and ran when the queue
# finally drained. Same code, same recipe — only the corpus differed. Nothing
# compared them, because the coverage guard protects LAYERS and nothing
# protected EVIDENCE.
# ---------------------------------------------------------------------------


def _quality(converged: str, n_prompts: int, target: str = "penultimate") -> str:
    return (
        f"corpus: t\nn_layers: 16\ntarget_layer: {target}\n"
        f"converged: {converged}\nn_prompts: {n_prompts}\n"
    )


def _stage(service, layers, config):
    import torch

    service.write_staged("org/model", {l: torch.randn(4, 4) for l in layers}, config)


def test_a_NON_CONVERGED_fit_may_not_replace_a_CONVERGED_one(tmp_path):
    """The exact shape that displaced a good LFM2 lens.

    MUTATION CONTROL: drop the convergence comparison from `_quality_regression`
    and this fails.
    """
    from src.services.jlens_artifact_service import (
        ArtifactQualityRegression,
        JLensArtifactService,
    )

    service = JLensArtifactService(tmp_path)
    _stage(service, range(15), _quality("true", 1097))
    service.commit("org/model", _passing_report())

    _stage(service, range(15), _quality("false", 400))
    with pytest.raises(ArtifactQualityRegression) as excinfo:
        service.commit("org/model", _passing_report())

    message = str(excinfo.value)
    assert "1097" in message and "400" in message, (
        f"the refusal must name both corpora so it can be judged: {message}"
    )

    # The incumbent is untouched by a refused commit.
    from src.services.jlens_artifact_service import JLensArtifactService as _S
    cfg = (tmp_path / "model" / "config.yaml").read_text()
    assert "n_prompts: 1097" in cfg


def test_a_SMALLER_corpus_may_not_replace_a_larger_one_at_equal_convergence(tmp_path):
    """Same convergence status, less evidence — still a regression.

    MUTATION CONTROL: drop the n_prompts comparison and this fails.
    """
    from src.services.jlens_artifact_service import (
        ArtifactQualityRegression,
        JLensArtifactService,
    )

    service = JLensArtifactService(tmp_path)
    _stage(service, range(15), _quality("true", 1097))
    service.commit("org/model", _passing_report())

    _stage(service, range(15), _quality("true", 400))
    with pytest.raises(ArtifactQualityRegression):
        service.commit("org/model", _passing_report())


def test_a_CONVERGED_fit_over_FEWER_prompts_publishes_freely(tmp_path):
    """Converging sooner is not worse. This guard must not block a better fit.

    A converged 400-prompt fit reached the threshold; a converged 1097-prompt
    fit merely took longer to. Refusing this would make the guard an obstacle
    to exactly the improvement it exists to protect.

    MUTATION CONTROL: compare n_prompts without the equal-convergence condition
    and this fails.
    """
    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    _stage(service, range(15), _quality("false", 1097))
    service.commit("org/model", _passing_report())

    _stage(service, range(15), _quality("true", 400))
    service.commit("org/model", _passing_report())  # must NOT raise

    cfg = (tmp_path / "model" / "config.yaml").read_text()
    assert "converged: true" in cfg and "n_prompts: 400" in cfg


def test_the_regression_can_be_overridden_deliberately(tmp_path):
    """A refusal the user cannot override is a wall, not a guard."""
    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    _stage(service, range(15), _quality("true", 1097))
    service.commit("org/model", _passing_report())

    _stage(service, range(15), _quality("false", 400))
    service.commit("org/model", _passing_report(), allow_quality_regression=True)

    cfg = (tmp_path / "model" / "config.yaml").read_text()
    assert "n_prompts: 400" in cfg


def test_an_unreadable_incumbent_recipe_does_not_block_publishing(tmp_path):
    """This guard fails OPEN, unlike the coverage guard, and on purpose.

    Coverage protects layers a user already paid GPU time for, so an unknown
    must not be discarded on a guess. Here an unreadable incumbent is not
    evidence worth defending — and failing closed would make an artifact with a
    corrupt config impossible to ever replace.

    MUTATION CONTROL: return a regression string when either recipe is
    unreadable and this fails.
    """
    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    _stage(service, range(15), "corpus: t\n")  # no converged, no n_prompts
    service.commit("org/model", _passing_report())

    _stage(service, range(15), _quality("false", 400))
    service.commit("org/model", _passing_report())  # must NOT raise


# ---------------------------------------------------------------------------
# RESTORING A DISPLACED ARTIFACT.
#
# `_quality_regression` stops the next stale fit from publishing over a better
# lens. It does nothing for one already displaced, which on 2026-08-04 could
# only be recovered by renaming directories inside the pod by hand.
# ---------------------------------------------------------------------------


def test_restore_swaps_the_archive_back_into_service(tmp_path):
    """The good lens serves again and the bad one is archived, not deleted.

    MUTATION CONTROL: make `restore_superseded` a plain move (drop the swap)
    and the displaced-artifact assertion fails — the 400-prompt fit would be
    gone with no way back.
    """
    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    _stage(service, range(15), _quality("true", 1097))
    service.commit("org/model", _passing_report())          # the good one
    _stage(service, range(15), _quality("false", 400))
    service.commit("org/model", _passing_report(), allow_quality_regression=True)

    # Precondition: the weak fit is serving, exactly as it was on hardware.
    assert "n_prompts: 400" in (tmp_path / "model" / "config.yaml").read_text()

    out = service.restore_superseded("model")

    assert "n_prompts: 1097" in (tmp_path / "model" / "config.yaml").read_text()
    assert out["restored"]["n_prompts"] == 1097 and out["restored"]["converged"] is True
    assert out["displaced"]["n_prompts"] == 400

    # NOTHING IS DELETED: the displaced fit is archived, so this is its own undo.
    archived = tmp_path / "model.superseded" / "config.yaml"
    assert archived.is_file() and "n_prompts: 400" in archived.read_text()


def test_restore_is_its_own_undo(tmp_path):
    """Called twice, you are back where you started."""
    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    _stage(service, range(15), _quality("true", 1097))
    service.commit("org/model", _passing_report())
    _stage(service, range(15), _quality("false", 400))
    service.commit("org/model", _passing_report(), allow_quality_regression=True)

    service.restore_superseded("model")
    service.restore_superseded("model")
    assert "n_prompts: 400" in (tmp_path / "model" / "config.yaml").read_text()


def test_restore_refuses_an_archive_whose_verdict_describes_other_weights(tmp_path):
    """A `.superseded` directory is not privileged.

    Promoting on a stale verdict would serve a lens validated against a
    different file — the failure the whole publish gate exists to prevent.

    MUTATION CONTROL: drop the `stored_report` check and this fails.
    """
    from src.services.jlens_artifact_service import (
        ArtifactNotValidated,
        JLensArtifactService,
    )

    service = JLensArtifactService(tmp_path)
    _stage(service, range(15), _quality("true", 1097))
    service.commit("org/model", _passing_report())
    _stage(service, range(15), _quality("false", 400))
    service.commit("org/model", _passing_report(), allow_quality_regression=True)

    # Tamper with the ARCHIVED lens so its recorded sha256 no longer matches.
    archived_lens = next((tmp_path / "model.superseded").glob("*_jacobian_lens.pt"))
    archived_lens.write_bytes(archived_lens.read_bytes() + b"tampered")

    with pytest.raises(ArtifactNotValidated, match="different file|no verdict"):
        service.restore_superseded("model")

    # And the serving artifact is untouched by the refusal.
    assert "n_prompts: 400" in (tmp_path / "model" / "config.yaml").read_text()


def test_restore_without_an_archive_is_an_error_not_a_silent_noop(tmp_path):
    """Silently doing nothing would read as "restored" to the caller."""
    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    _stage(service, range(15), _quality("true", 1097))
    service.commit("org/model", _passing_report())

    with pytest.raises(FileNotFoundError, match="no archived artifact"):
        service.restore_superseded("model")


# ---------------------------------------------------------------------------
# CAUSAL EVIDENCE TRAVELS WITH THE LENS.
#
# A lens is consumed by MOUNTING its directory. Published to HuggingFace and
# pulled down by a serving runtime, it arrives as files and nothing else — so
# evidence that lived only in a task result would not make the journey, and the
# consumer would have a dictionary it can read with no idea which directions
# actually move the model.
# ---------------------------------------------------------------------------


def _evidence(direction=" dog", layers=(9, 10), strength=1.0, separated=True):
    return {
        "steering_recipe": {
            "primitive": "additive",
            "direction_token": direction,
            "target_token": direction,
            "layers": list(layers),
            "positions": [-1],
            "strength": strength,
            "hook_target": "layers_module[L] (resid_post)",
        },
        "evidence": {"separated_from_control": separated},
        "evidence_rung": 2,
    }


def _published(tmp_path, layers=range(15)):
    import torch

    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    service.write_staged(
        "org/model", {l: torch.randn(4, 4) for l in layers}, _quality("true", 500)
    )
    service.commit("org/model", _passing_report())
    return service


def test_causal_evidence_is_written_beside_the_lens(tmp_path):
    """Next to the weights, not in a database the consumer never sees."""
    from src.services.jlens_artifact_service import INTERVENTION_FILE

    service = _published(tmp_path)
    service.record_intervention_result("org/model", _evidence())

    path = tmp_path / "model" / INTERVENTION_FILE
    assert path.is_file(), "the evidence did not land in the artifact directory"

    records = service.intervention_results(service.find("org/model"))
    assert len(records) == 1
    # THE RECIPE, not just the score: a consumer must be able to apply it.
    recipe = records[0]["steering_recipe"]
    assert recipe["direction_token"] == " dog"
    assert recipe["layers"] == [9, 10]
    assert recipe["hook_target"] == "layers_module[L] (resid_post)"


def test_evidence_for_DIFFERENT_weights_is_dropped(tmp_path):
    """A refit replaces the matrices; the old evidence describes a dead file.

    Carrying it forward would attribute one lens's demonstrated behaviour to
    another — worse than having no evidence at all.

    MUTATION CONTROL: stop comparing `lens_sha256` on read and this fails.
    """
    service = _published(tmp_path)
    service.record_intervention_result("org/model", _evidence())
    assert len(service.intervention_results(service.find("org/model"))) == 1

    lens = next((tmp_path / "model").glob("*_jacobian_lens.pt"))
    lens.write_bytes(lens.read_bytes() + b"refitted")

    assert service.intervention_results(service.find("org/model")) == [], (
        "evidence describing the previous weights survived a change to the lens"
    )


def test_rerunning_the_SAME_experiment_replaces_its_record(tmp_path):
    """Two runs of one experiment are not two findings.

    MUTATION CONTROL: append unconditionally and this fails with 2 records.
    """
    service = _published(tmp_path)
    service.record_intervention_result("org/model", _evidence(separated=False))
    service.record_intervention_result("org/model", _evidence(separated=True))

    records = service.intervention_results(service.find("org/model"))
    assert len(records) == 1
    assert records[0]["evidence"]["separated_from_control"] is True, (
        "the later run did not supersede the earlier one"
    )


def test_a_DIFFERENT_experiment_is_kept_alongside(tmp_path):
    """A lens can steer several concepts; each is its own finding.

    MUTATION CONTROL: key the replacement on the slug alone and this fails —
    every new direction would evict the last.
    """
    service = _published(tmp_path)
    service.record_intervention_result("org/model", _evidence(direction=" dog"))
    service.record_intervention_result("org/model", _evidence(direction=" cat"))
    service.record_intervention_result("org/model", _evidence(direction=" dog", layers=(3,)))

    records = service.intervention_results(service.find("org/model"))
    assert len(records) == 3, [r["steering_recipe"] for r in records]


def test_recording_against_an_unpublished_model_is_an_error(tmp_path):
    """Silently doing nothing would read as "recorded" to the caller."""
    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    with pytest.raises(FileNotFoundError, match="no published artifact"):
        service.record_intervention_result("org/nothing", _evidence())


def test_unreadable_evidence_does_not_break_the_listing(tmp_path):
    """A corrupt sidecar must not make the artifact unusable."""
    from src.services.jlens_artifact_service import INTERVENTION_FILE

    service = _published(tmp_path)
    (tmp_path / "model" / INTERVENTION_FILE).write_text("{not json")
    assert service.intervention_results(service.find("org/model")) == []


def test_two_DIFFERENT_swaps_are_both_kept(tmp_path):
    """A swap IS the pair; the partner is part of the experiment.

    The recipe key omitted `target_token` and `positions`, so swapping dog with
    cat and swapping dog with pet shared a key — and recording the second
    DELETED the first from the file whose whole purpose is carrying evidence off
    this machine. Each record is a completed GPU run.

    MUTATION CONTROL: drop target_token from `_recipe_key` and this fails at 1.
    """
    service = _published(tmp_path)
    swap = lambda partner: {  # noqa: E731
        "steering_recipe": {
            "primitive": "coordinate_swap",
            "direction_token": " dog",
            "target_token": partner,
            "layers": [9],
            "positions": [-1],
            "strength": 1.0,
        },
        "evidence": {"separated_from_control": True},
    }
    service.record_intervention_result("org/model", swap(" cat"))
    service.record_intervention_result("org/model", swap(" pet"))
    records = service.intervention_results(service.find("org/model"))
    assert len(records) == 2, [r["steering_recipe"]["target_token"] for r in records]


def test_records_differing_only_in_POSITIONS_are_both_kept(tmp_path):
    """Each field the key widened needs its OWN test, varied alone.

    `test_two_DIFFERENT_swaps_are_both_kept` varies `target_token` and holds
    `positions: [-1]` constant across both records, so the two fields agree by
    construction and only half the widening is pinned: dropping `"positions"`
    from `_recipe_key` left the whole suite green. Perturbing the last token is
    a different experiment from perturbing the first, and the loser of that
    collision is a completed GPU run deleted from the portability file.

    MUTATION CONTROL: drop `"positions"` from `_recipe_key` and this fails at 1.
    """
    service = _published(tmp_path)
    at = lambda pos: {  # noqa: E731
        "steering_recipe": {
            "primitive": "additive",
            "direction_token": " Paris",
            "target_token": " Paris",
            "layers": [9],
            "positions": pos,
            "strength": 1.0,
            "prompts_sha256": "abc",
        },
        "evidence": {"separated_from_control": True},
    }
    service.record_intervention_result("org/model", at([-1]))
    service.record_intervention_result("org/model", at([0]))
    records = service.intervention_results(service.find("org/model"))
    assert len(records) == 2, [r["steering_recipe"]["positions"] for r in records]


def test_a_ONE_TRIAL_click_does_not_evict_a_FIFTY_PROMPT_run(tmp_path):
    """The trial set is the most obvious variable, and it was not in the key.

    An agent runs 50 prompts on ' Paris' at L9 and files a record with
    `separated_from_control: true`. The user then reads out a DIFFERENT prompt
    in the panel and clicks Steer on the same token — same primitive, same
    direction, same target, same layers, same positions, same strength. Same
    key. The 50-prompt record is dropped, and the artifact now tells a miLLM
    consumer that the direction moves nothing.

    MUTATION CONTROL: drop `"prompts_sha256"` from `_recipe_key` and this fails
    at 1, having kept only the one-trial record.
    """
    service = _published(tmp_path)
    base = {
        "primitive": "additive",
        "direction_token": " Paris",
        "target_token": " Paris",
        "layers": [9],
        "positions": [-1],
        "strength": 1.0,
    }
    service.record_intervention_result(
        "org/model",
        {
            "steering_recipe": {**base, "n_trials": 50, "prompts_sha256": "fifty"},
            "evidence": {"separated_from_control": True, "n_trials": 50},
        },
    )
    service.record_intervention_result(
        "org/model",
        {
            "steering_recipe": {**base, "n_trials": 1, "prompts_sha256": "one"},
            "evidence": {"separated_from_control": False, "n_trials": 1},
        },
    )
    records = service.intervention_results(service.find("org/model"))
    assert len(records) == 2, "the one-trial click evicted the fifty-prompt run"
    # AND THE STRONGER ONE SURVIVED INTACT — not merely "two records exist".
    trials = sorted(r["evidence"]["n_trials"] for r in records)
    assert trials == [1, 50], trials


def test_the_same_swap_twice_still_supersedes(tmp_path):
    """Widening the key must not turn every re-run into a duplicate."""
    service = _published(tmp_path)
    rec = {
        "steering_recipe": {
            "primitive": "coordinate_swap",
            "direction_token": " dog",
            "target_token": " cat",
            "layers": [9],
            "positions": [-1],
            "strength": 1.0,
        },
        "evidence": {"separated_from_control": False},
    }
    service.record_intervention_result("org/model", rec)
    service.record_intervention_result("org/model", {**rec, "evidence": {"separated_from_control": True}})
    records = service.intervention_results(service.find("org/model"))
    assert len(records) == 1
    assert records[0]["evidence"]["separated_from_control"] is True


def test_the_record_file_is_written_ATOMICALLY(tmp_path):
    """A truncate-then-write loses every record if the pod dies mid-write.

    `_read_interventions` fails to `[]` on invalid JSON, so the next successful
    write puts a one-element list over the wreckage — total, permanent, invisible
    loss of measurements that cost GPU time and cannot be regenerated.

    MUTATION CONTROL: use `write_text` directly and this fails.
    """
    # BEHAVIOUR, NOT A SUBSTRING. The first version asserted `".replace(" in
    # inspect.getsource(...)`, which any unrelated `.replace(` in the method
    # satisfies — and which `shutil.move`, equally atomic, would fail. That is
    # the same source-scrape failure mode removed from the `owns_its_failure`
    # guard in this very arc: it fails open.
    #
    # THE TEST INSTEAD KILLS THE WRITE. A non-atomic implementation truncates
    # the real file first, so the existing records are already gone when the
    # write fails; an atomic one is still writing to a temp path and the real
    # file is untouched.
    service = _published(tmp_path)
    service.record_intervention_result("org/model", _evidence())
    ref = service.find("org/model")
    target = ref.directory / "interventions.json"
    before = target.read_text()
    assert json.loads(before), "nothing was recorded; the fixture is not exercising it"

    real_write = pathlib.Path.write_text

    def die_mid_write(self, *args, **kwargs):
        real_write(self, *args, **kwargs)
        raise OSError("the pod was evicted")

    with patch.object(pathlib.Path, "write_text", die_mid_write):
        with pytest.raises(OSError):
            service.record_intervention_result(
                "org/model",
                {
                    "steering_recipe": {
                        "primitive": "additive",
                        "direction_token": " other",
                        "layers": [1],
                    },
                    "evidence": {"separated_from_control": True},
                },
            )

    assert target.read_text() == before, (
        "the interrupted write destroyed the existing records; a truncating "
        "write loses every measurement in the file, and _read_interventions "
        "then fails to [] so the next success writes over the wreckage"
    )

    # And a normal write still works end to end, leaving no temp file behind.
    service = _published(tmp_path)
    service.record_intervention_result("org/model", _evidence())
    assert service.intervention_results(service.find("org/model"))
    assert not list((tmp_path / "model").glob("*.tmp")), "a temp file was left behind"

def test_a_leftover_SWAP_directory_is_invisible_to_discovery(tmp_path):
    """It briefly holds the only copy of a live lens, under a servable name.

    `restore_superseded` parks the displaced artifact under `<slug>.swap` for
    the duration of a three-way rename. Discovery skipped `.staging` and
    `.superseded` and not this one — so a pod evicted mid-rename left the
    displaced lens discoverable as a SECOND, differently-slugged artifact for
    the same model, which is exactly what the method's own docstring says must
    never happen.

    MUTATION CONTROL: drop SWAP_SUFFIX from the discovery skip tuple and this
    fails at 2.
    """
    from src.services.jlens_artifact_service import SWAP_SUFFIX

    service = _published(tmp_path)
    debris = tmp_path / f"model{SWAP_SUFFIX}"
    debris.mkdir()
    # A CONFORMANT directory, or discovery would skip it for the wrong reason
    # and the test would pass without the suffix rule existing.
    (debris / "model_jacobian_lens.pt").write_bytes(b"x")

    slugs = [a.slug for a in service.list_artifacts()]
    assert slugs == ["model"], slugs


def test_restore_REFUSES_rather_than_clearing_a_leftover_swap(tmp_path):
    """`rmtree` here made the recovery operation destroy what it recovered.

    Debris from an interrupted rename is not a stale row to be tidied — the
    filesystem IS the registry (PADR IDL-46), so it is data, and it may be the
    only copy of a lens.

    MUTATION CONTROL: restore the `shutil.rmtree(swap)` and this fails.
    """
    from src.services.jlens_artifact_service import (
        SWAP_SUFFIX,
        ArtifactConflict,
        JLensArtifactService,
    )

    # BUILT THROUGH THE REAL PUBLISH PATH, so the archive carries a verdict
    # matching its own weights. A hand-made directory is refused earlier, for a
    # different reason, and the swap guard is never reached.
    service = JLensArtifactService(tmp_path)
    _stage(service, range(15), _quality("true", 1097))
    service.commit("org/model", _passing_report())
    _stage(service, range(15), _quality("false", 400))
    service.commit("org/model", _passing_report(), allow_quality_regression=True)

    debris = tmp_path / f"model{SWAP_SUFFIX}"
    debris.mkdir()
    only_copy = debris / "model_jacobian_lens.pt"
    only_copy.write_bytes(b"the only copy")

    with pytest.raises(ArtifactConflict, match="by hand"):
        service.restore_superseded("model")

    # NOT MERELY REFUSED — STILL THERE, byte for byte.
    assert only_copy.read_bytes() == b"the only copy"

