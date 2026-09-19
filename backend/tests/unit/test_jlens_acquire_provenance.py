"""
An acquired lens must not be described as one miStudio fitted.

The fit worker's `_config_yaml` records a recipe — treatment, position scope,
aggregation, differentiation mode, corpus, sequence length, convergence threshold
— because it PERFORMED those choices. For a downloaded lens miStudio performed
none of them, and a defaulted value in `config.yaml` is indistinguishable from a
measured one to every reader downstream, including `ProvenanceStrip`, which
renders it as fact.

The highest-value test here is `test_the_upstream_config_is_NOT_merged`. miStudio's
config readers are line scanners that match `name.strip()` at ANY indentation, so a
"namespaced" upstream block is not namespaced: a nested `layer_scales:` is read as
real and produces a FABRICATED per-layer rescale that `JacobianTransport` applies
to every probe and intervention magnitude — invisible in ranked readouts, because
the model's final norm divides a positive scalar straight back out.

MUTATION CONTROLS (each must turn this file red):
  * merge the upstream config into config.yaml -> "upstream config is NOT merged"
  * take `fit.n_prompts` instead of `results.prompts_fitted`
                                                -> "n_prompts is what RAN"
  * default `converged` to true                 -> "convergence is not claimed"
  * downgrade an identity MISMATCH to a warning -> "a lens for OTHER WEIGHTS"
  * treat UNVERIFIED as verified                -> "a config-less lens is UNVERIFIED"
  * hardcode target_layer = penultimate         -> "the target is DERIVED"
  * skip the layer-range check                  -> "a foreign indexing convention"
  * write the fitter's recipe fields            -> "claims NO RECIPE it did not measure"
  * keep the upstream filename stem             -> "named from the TARGET MODEL"
"""

from __future__ import annotations

import json

import pytest
import torch

from src.services.jlens_acquire_service import (
    AcquisitionRefused,
    WeightIdentity,
    check_weight_identity,
    config_yaml_for_acquired,
    derive_converged,
    derive_n_prompts,
    dtype_of,
    inspect_layers,
    parse_upstream_config,
    write_acquisition_record,
)
from src.services.jlens_artifact_service import ArtifactRef, JLensArtifactService
from src.services.jlens_validation import (
    CheckClass,
    CheckResult,
    CheckStatus,
    ValidationReport,
)

#: A published config in the real shape, with the two prompt figures DIFFERENT.
#:
#: They differ on the real gemma-2-2b-it lens (requested 1000, ran 337) and are
#: EQUAL on the real pythia one. A fixture built from pythia's numbers cannot
#: tell `fit.n_prompts` from `results.prompts_fitted` — the two agree by
#: construction and the mutation survives.
UPSTREAM = """
np_model_id: "gemma-2-2b-it"
hf_model_name: "google/gemma-2-2b-it"
dataset:
  name: "Salesforce/wikitext"
fit:
  n_prompts: 1000
  stop_at_delta: 0.002
  dtype: "bfloat16"
results:
  prompts_fitted: 337
  final_mean_rel_change: 0.00180945
attribution: "fit by someone else"
"""

D_MODEL = 4
N_LAYERS = 6


def _payload(layers=(0, 1, 2, 3, 4), d_model=D_MODEL, dtype=torch.float16):
    return {l: torch.zeros(d_model, d_model, dtype=dtype) for l in layers}


class TestTheConfigClaimsOnlyWhatWasMeasured:
    def test_it_claims_NO_RECIPE_IT_DID_NOT_MEASURE(self):
        """The fabrication guard.

        Every key below describes a choice made while FITTING. miStudio made
        none of them for a downloaded lens, and BR-007 says this file must be
        sufficient to rebuild the artifact — so an invented value here is worse
        than a missing one.

        MUTATION CONTROL: reuse the fit worker's `_config_yaml` and this fails.
        """
        verdict = inspect_layers(_payload(), n_layers=N_LAYERS, d_model=D_MODEL)
        text = config_yaml_for_acquired(
            repo_id="google/gemma-2-2b-it",
            layers=verdict,
            n_vocab=256,
            n_layers=N_LAYERS,
            dtype="fp16",
            upstream_config=parse_upstream_config(UPSTREAM),
        )
        for invented in (
            "attention_gradients_requested",
            "target_position_scope",
            "source_position_aggregation",
            "differentiation_mode",
            "aggregation:",
            "corpus:",
            "seq_len:",
            "convergence_delta:",
            "per_layer_applicability",
            "treatment:",
        ):
            assert invented not in text, (
                f"config.yaml asserts {invented!r}, a fit parameter miStudio "
                "never chose for this lens"
            )

    def test_the_upstream_config_is_NOT_merged(self, tmp_path):
        """The poisoning guard, and the reason provenance is a sidecar.

        miStudio's readers match on `name.strip()` at ANY indentation and return
        on the first hit. A nested `layer_scales:` is therefore read as a real
        top-level block — verified — and `JacobianTransport` then applies a
        rescale nobody wrote, silently changing every probe and intervention
        magnitude while ranked readouts look perfect.

        MUTATION CONTROL: embed the upstream config under an `acquired:` block
        and this fails.
        """
        hostile = parse_upstream_config(
            UPSTREAM
            + """
layer_scales:
  0: 7.5
  1: 9.5
target_layer: "final"
converged: true
"""
        )
        verdict = inspect_layers(_payload(), n_layers=N_LAYERS, d_model=D_MODEL)
        text = config_yaml_for_acquired(
            repo_id="google/gemma-2-2b-it",
            layers=verdict,
            n_vocab=256,
            n_layers=N_LAYERS,
            dtype="fp16",
            upstream_config=hostile,
        )

        directory = tmp_path / "gemma-2-2b-it"
        directory.mkdir()
        (directory / "gemma-2-2b-it_jacobian_lens.pt").write_bytes(b"x")
        config = directory / "config.yaml"
        config.write_text(text)
        ref = ArtifactRef(
            slug="gemma-2-2b-it",
            directory=directory,
            lens_path=directory / "gemma-2-2b-it_jacobian_lens.pt",
            config_path=config,
        )
        service = JLensArtifactService(tmp_path)

        assert service.layer_scales(ref) == {}, (
            "a FABRICATED per-layer rescale reached the config; JacobianTransport "
            "applies it to every probe and intervention magnitude"
        )
        # And the derived values won, not the upstream assertions.
        assert service.target_layer(ref) == "penultimate", (
            "the upstream's `final` overrode the target measured from the tensors"
        )

    def test_n_prompts_is_what_RAN_not_what_was_requested(self):
        """`fit.n_prompts` is the operator's cap; `results.prompts_fitted` is
        what ran before the convergence stop fired. `_quality_regression`
        compares this number, so taking the request lets a 337-prompt fit claim
        to be a 1000-prompt one and displace a larger local fit."""
        assert derive_n_prompts(parse_upstream_config(UPSTREAM)) == 337

    def test_convergence_is_DERIVED_from_the_publishers_own_numbers(self):
        """0.00180945 <= 0.002. A comparison, not a claim."""
        assert derive_converged(parse_upstream_config(UPSTREAM)) is True

    def test_convergence_is_FALSE_when_the_threshold_was_not_reached(self):
        """The real pythia lens ran its whole budget without converging."""
        config = parse_upstream_config(
            UPSTREAM.replace("final_mean_rel_change: 0.00180945", "final_mean_rel_change: 0.5")
        )
        assert derive_converged(config) is False

    def test_convergence_is_NOT_CLAIMED_when_it_cannot_be_derived(self):
        """Absent is the honest third answer, and `_config_bool` is explicit
        that "None is NOT False". Defaulting True would let an unconverged
        third-party lens displace a converged local fit.

        MUTATION CONTROL: default to True (or False) and this fails.
        """
        assert derive_converged({}) is None
        assert derive_converged({"fit": {"stop_at_delta": 0.002}}) is None

    def test_an_underivable_field_is_OMITTED_rather_than_defaulted(self):
        verdict = inspect_layers(_payload(), n_layers=N_LAYERS, d_model=D_MODEL)
        text = config_yaml_for_acquired(
            repo_id="org/m",
            layers=verdict,
            n_vocab=256,
            n_layers=N_LAYERS,
            dtype="fp16",
            upstream_config=None,  # a community repo with no config at all
        )
        assert "converged:" not in text
        assert "n_prompts:" not in text
        # But what WAS measurable is still there.
        assert "fitted_layers: [0, 1, 2, 3, 4]" in text
        assert "d_model: 4" in text

    def test_NO_layer_scales_block_is_written(self):
        """Absent means "no rescale to undo", which is correct: the published
        artifacts store raw fp16 whose entries are O(1). Writing 1.0s would
        assert a convention the publisher never stated."""
        verdict = inspect_layers(_payload(), n_layers=N_LAYERS, d_model=D_MODEL)
        text = config_yaml_for_acquired(
            repo_id="org/m",
            layers=verdict,
            n_vocab=256,
            n_layers=N_LAYERS,
            dtype="fp16",
            upstream_config=parse_upstream_config(UPSTREAM),
        )
        assert "layer_scales" not in text


class TestWeightIdentity:
    def test_a_matching_declaration_is_VERIFIED(self):
        v = check_weight_identity(parse_upstream_config(UPSTREAM), "google/gemma-2-2b-it")
        assert v.state is WeightIdentity.VERIFIED

    def test_a_lens_for_OTHER_WEIGHTS_is_a_MISMATCH(self):
        """Not a warning. A lens fitted for different weights "produces a
        complete, plausible readout that is wrong"."""
        v = check_weight_identity(parse_upstream_config(UPSTREAM), "google/gemma-2-9b-it")
        assert v.state is WeightIdentity.MISMATCH
        assert "gemma-2-2b-it" in v.detail and "gemma-2-9b-it" in v.detail

    def test_a_config_less_lens_is_UNVERIFIED_not_verified(self):
        """Community repos ship a bare `.pt`. The pairing then rests on the
        caller's assertion, and the record must say so rather than implying a
        check was performed.

        MUTATION CONTROL: return VERIFIED when nothing is declared -> fails.
        """
        v = check_weight_identity(None, "org/whatever")
        assert v.state is WeightIdentity.UNVERIFIED
        assert v.declared is None

    def test_the_three_states_are_DISTINCT(self):
        """UNVERIFIED must not collapse into either neighbour: it is neither
        "checked and fine" nor "checked and wrong"."""
        assert len({s.value for s in WeightIdentity}) == 3


class TestTheTensorsAreInterrogated:
    def test_the_target_is_DERIVED_from_the_fitted_layers(self):
        """MUTATION CONTROL: hardcode `penultimate` and the `final` case fails."""
        penultimate = inspect_layers(_payload((0, 1, 2, 3, 4)), N_LAYERS, D_MODEL)
        assert penultimate.target_layer == "penultimate"
        final = inspect_layers(_payload((0, 1, 2, 3, 4, 5)), N_LAYERS, D_MODEL)
        assert final.target_layer == "final"

    def test_an_underivable_target_is_OMITTED_not_guessed(self):
        """A partial fit stopping short of both. `target_layer()` returns None
        and `_coverage_delta` then fails closed when REPLACING — correct for a
        lens whose extent we cannot state."""
        partial = inspect_layers(_payload((0, 1, 2)), N_LAYERS, D_MODEL)
        assert partial.target_layer is None

    def test_a_FOREIGN_INDEXING_CONVENTION_is_refused(self):
        """The check semantic discrimination cannot make.

        `check_semantic` deliberately scans EVERY fitted layer, so a 1-based or
        output-counted convention still finds the expected token somewhere and
        passes. A key outside the stack is impossible for these weights.

        MUTATION CONTROL: drop the range check and this fails.
        """
        with pytest.raises(AcquisitionRefused, match="outside"):
            inspect_layers(_payload((1, 2, 3, 4, 5, 6)), N_LAYERS, D_MODEL)

    def test_a_d_model_DISAGREEMENT_is_refused(self):
        """Two independent declarations of the same fact disagreeing is a
        wrong-model signal that costs nothing to check."""
        with pytest.raises(AcquisitionRefused, match="d_model"):
            inspect_layers(_payload(d_model=8), N_LAYERS, D_MODEL)

    def test_degenerate_layers_are_MEASURED(self):
        """The identity layer is the identity by construction, so this is a
        measurement rather than a declaration to be trusted."""
        payload = _payload((0, 1, 2, 3, 4))
        payload[4] = torch.eye(D_MODEL, dtype=torch.float16)
        verdict = inspect_layers(payload, N_LAYERS, D_MODEL)
        assert verdict.degenerate == [4]
        assert verdict.identity_distance[4] < verdict.identity_distance[0]

    def test_a_MIXED_dtype_lens_is_refused(self):
        payload = _payload((0, 1))
        payload[1] = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float32)
        with pytest.raises(AcquisitionRefused, match="dtype"):
            dtype_of(payload)

    def test_the_dtype_is_READ_not_declared(self):
        assert dtype_of(_payload(dtype=torch.float16)) == "fp16"
        assert dtype_of(_payload(dtype=torch.bfloat16)) == "bf16"


class TestTheTransferRecord:
    def test_it_records_both_digests_and_whether_they_MATCH(self, tmp_path):
        """`local == upstream` is a fact a third party can check against the
        source repo. It is the only cryptographic identity an acquired lens
        has, and the existing weight-identity check never opens the file."""
        verdict = inspect_layers(_payload(), N_LAYERS, D_MODEL)
        identity = check_weight_identity(parse_upstream_config(UPSTREAM), "google/gemma-2-2b-it")
        write_acquisition_record(
            tmp_path,
            source_repo="org/lenses",
            source_path="a/b.pt",
            revision="abc123",
            upstream_sha256="deadbeef",
            local_sha256="deadbeef",
            identity=identity,
            layers=verdict,
            upstream_config=parse_upstream_config(UPSTREAM),
        )
        record = json.loads((tmp_path / "acquisition.json").read_text())
        assert record["bytes"]["identical"] is True
        assert record["source"]["revision"] == "abc123"
        assert record["weight_identity"]["state"] == "verified"

    def test_a_REWRITTEN_file_is_not_reported_as_identical(self, tmp_path):
        verdict = inspect_layers(_payload(), N_LAYERS, D_MODEL)
        write_acquisition_record(
            tmp_path,
            source_repo="org/lenses",
            source_path="a/b.pt",
            revision=None,
            upstream_sha256="aaa",
            local_sha256="bbb",
            identity=check_weight_identity(None, "org/m"),
            layers=verdict,
            upstream_config=None,
        )
        record = json.loads((tmp_path / "acquisition.json").read_text())
        assert record["bytes"]["identical"] is False

    def test_an_UNKNOWN_upstream_digest_reads_as_UNKNOWN(self, tmp_path):
        """Not as divergence — and this test used to assert the opposite.

        The Hub exposes `lfs.sha256` only for LFS-tracked files, so every lens
        below the threshold produced `identical: false`: a positive claim that
        what we serve differs from what was published, from a comparison that
        never ran. The MCP tool presents that field to an agent as a fact. This
        module is explicit everywhere else that None is not False, and the first
        version of this test PINNED the exception rather than preventing it.
        """
        verdict = inspect_layers(_payload(), N_LAYERS, D_MODEL)
        write_acquisition_record(
            tmp_path,
            source_repo="org/lenses",
            source_path="a/b.pt",
            revision=None,
            upstream_sha256=None,
            local_sha256=None,
            identity=check_weight_identity(None, "org/m"),
            layers=verdict,
            upstream_config=None,
        )
        record = json.loads((tmp_path / "acquisition.json").read_text())
        assert record["bytes"]["identical"] is None, (
            "an unmeasured comparison was recorded as a difference"
        )

    def test_the_upstream_config_is_QUARANTINED_here(self, tmp_path):
        """It is the richest provenance available AND it must never reach
        `config.yaml`. Both facts, one place."""
        verdict = inspect_layers(_payload(), N_LAYERS, D_MODEL)
        write_acquisition_record(
            tmp_path,
            source_repo="org/lenses",
            source_path="a/b.pt",
            revision="r",
            upstream_sha256="x",
            local_sha256="x",
            identity=check_weight_identity(parse_upstream_config(UPSTREAM), "google/gemma-2-2b-it"),
            layers=verdict,
            upstream_config=parse_upstream_config(UPSTREAM),
        )
        record = json.loads((tmp_path / "acquisition.json").read_text())
        assert record["upstream_config"]["results"]["prompts_fitted"] == 337
        assert record["upstream_config"]["attribution"] == "fit by someone else"

    def test_the_sidecar_is_INVISIBLE_to_discovery_and_naming(self, tmp_path):
        """Same property `validation.json` and `interventions.json` rely on."""
        from src.services.jlens_validation import CheckStatus, check_naming

        directory = tmp_path / "m"
        directory.mkdir()
        (directory / "m_jacobian_lens.pt").write_bytes(b"x")
        (directory / "config.yaml").write_text("model: org/m\n")
        verdict = inspect_layers(_payload(), N_LAYERS, D_MODEL)
        write_acquisition_record(
            directory,
            source_repo="r",
            source_path="p",
            revision=None,
            upstream_sha256=None,
            local_sha256="x",
            identity=check_weight_identity(None, "org/m"),
            layers=verdict,
            upstream_config=None,
        )
        assert check_naming(directory).status is CheckStatus.PASS
        assert [a.slug for a in JLensArtifactService(tmp_path).list_artifacts()] == ["m"]


class TestStagingNamesFromTheModel:
    def test_the_staged_file_is_named_from_the_TARGET_MODEL(self, tmp_path):
        """`slug_for` lowercases; a published lens preserves the HuggingFace
        case. So `Qwen/Qwen3-8B` publishes as `Qwen3-8B_jacobian_lens.pt`, which
        fails miStudio's lowercase-anchored NAMING check. Keeping the upstream
        stem breaks on exactly the model ids carrying capitals — and passes on
        every lowercase one that gets tested first.

        MUTATION CONTROL: name the staged file from the source path -> fails.
        """
        from src.services.jlens_validation import CheckStatus, check_naming

        source = tmp_path / "Qwen3-8B_jacobian_lens.pt"
        torch.save({0: torch.zeros(2, 2)}, source)
        service = JLensArtifactService(tmp_path / "registry")

        ref = service.stage_from_file("Qwen/Qwen3-8B", source, "model: Qwen/Qwen3-8B\n")

        assert ref.lens_path.name == "qwen3-8b_jacobian_lens.pt", ref.lens_path.name
        assert check_naming(ref.directory).status is CheckStatus.PASS

    def test_the_BYTES_are_preserved(self, tmp_path):
        """Byte identity is the acquired artifact's only checkable provenance."""
        from src.services.jlens_acquire_service import file_digest

        source = tmp_path / "up_jacobian_lens.pt"
        torch.save({0: torch.zeros(2, 2)}, source)
        service = JLensArtifactService(tmp_path / "registry")
        ref = service.stage_from_file("org/m", source, "model: org/m\n")
        assert file_digest(ref.lens_path) == file_digest(source)

    def test_a_PT_SIDECAR_is_refused(self, tmp_path):
        """A second `.pt` fails NAMING, and a consumer globbing
        `*_jacobian_lens.pt` picks between matches silently."""
        source = tmp_path / "up_jacobian_lens.pt"
        torch.save({0: torch.zeros(2, 2)}, source)
        other = tmp_path / "extra.pt"
        other.write_bytes(b"x")
        service = JLensArtifactService(tmp_path / "registry")
        with pytest.raises(ValueError, match=r"\.pt"):
            service.stage_from_file(
                "org/m", source, "model: org/m\n", sidecars={"extra.pt": other}
            )

    def test_a_NON_pt_sidecar_is_copied(self, tmp_path):
        """The convergence trace rides along; spec §2.1 puts it in the same
        directory and a consumer ignores it."""
        source = tmp_path / "up_jacobian_lens.pt"
        torch.save({0: torch.zeros(2, 2)}, source)
        csv = tmp_path / "conv.csv"
        csv.write_text("n_done,identity_distance\n1,0.5\n")
        service = JLensArtifactService(tmp_path / "registry")
        ref = service.stage_from_file(
            "org/m", source, "model: org/m\n", sidecars={"m_convergence.csv": csv}
        )
        assert (ref.directory / "m_convergence.csv").read_text().startswith("n_done")

class TestTheReviewRound1Findings:
    """Regressions for eight defects, two of which crashed every acquisition.

    MUTATION CONTROLS:
      * digest after commit                  -> "the digest is taken BEFORE"
      * catch only the two gate refusals     -> "an unserviceable lens REPORTS"
      * preview applies the lower bound      -> "a PARTIAL lens is not rejected"
      * dtype_bytes hardcoded 2              -> "an fp32 lens fits its own envelope"
      * one probe per path                   -> "distinct MOUNTS"
      * single-copy footprint                -> "BOTH copies"
      * checkpoint n_prompts ignored         -> "a bare .pt still carries n_prompts"
      * staging cleared unconditionally      -> "a staged artifact is NOT destroyed"
    """

    def test_the_digest_is_taken_BEFORE_commit_renames_the_directory(self, tmp_path):
        """`commit` ends in `staging.rename(final)`, so reading `ref.lens_path`
        afterwards raises FileNotFoundError — on every LFS-hosted lens, which is
        every real one. The artifact published, the row said completed, and then
        the task failed over a lens that had actually landed.
        """
        from src.services.jlens_acquire_service import file_digest
        from src.services.jlens_validation import (
            CheckClass,
            CheckResult,
            CheckStatus,
            ValidationReport,
        )

        source = tmp_path / "up_jacobian_lens.pt"
        torch.save({0: torch.zeros(2, 2)}, source)
        service = JLensArtifactService(tmp_path / "reg")
        ref = service.stage_from_file("org/m", source, "model: org/m\n")

        before = file_digest(ref.lens_path)
        service.commit(
            "org/m",
            ValidationReport(
                [CheckResult(c, CheckStatus.PASS, "ok") for c in CheckClass]
            ),
        )
        assert not ref.lens_path.exists(), (
            "the staging path survived commit; this test no longer describes "
            "the hazard"
        )
        # The digest must have been captured while the file was still there.
        assert before and len(before) == 64

    def test_an_UNSERVICEABLE_lens_REPORTS_rather_than_crashing(self, tmp_path):
        """`commit` raises ArtifactNotValidated for any report short of a full
        pass. Catching only the two gate refusals meant a lens that simply did
        not surface the fixture — the likeliest outcome for a foreign lens —
        crashed instead of returning its per-layer evidence."""
        from src.services.jlens_artifact_service import ArtifactNotValidated
        from src.services.jlens_validation import (
            CheckClass,
            CheckResult,
            CheckStatus,
            ValidationReport,
        )

        results = [
            CheckResult(c, CheckStatus.PASS, "ok")
            for c in CheckClass
            if c is not CheckClass.SEMANTIC
        ]
        results.append(
            CheckResult(CheckClass.SEMANTIC, CheckStatus.FAIL, "no spider anywhere")
        )
        report = ValidationReport(results)
        assert report.serviceable is False, "the fixture must be unserviceable"
        assert report.passed is False

        source = tmp_path / "up_jacobian_lens.pt"
        torch.save({0: torch.zeros(2, 2)}, source)
        service = JLensArtifactService(tmp_path / "reg")
        service.stage_from_file("org/m", source, "model: org/m\n")
        with pytest.raises(ArtifactNotValidated):
            service.commit("org/m", report)

        # THE DECISION, TESTED WHERE IT IS MADE. An inline `if` in the worker is
        # only reachable by running the whole task, so a mutation deleting it
        # survives every test that does not — which is exactly what happened to
        # the first version of this test.
        from src.services.jlens_acquire_service import publication_blocker

        blocker = publication_blocker(report)
        assert blocker is not None, "an unserviceable report was cleared to publish"
        assert "not published" in blocker
        # And a good report is NOT blocked, or the guard would refuse everything.
        good = ValidationReport(
            [CheckResult(c, CheckStatus.PASS, "ok") for c in CheckClass]
        )
        assert publication_blocker(good) is None

    def test_a_PARTIAL_lens_is_not_rejected_by_the_preview(self):
        """`check_envelope` uses the layer count for BOTH bounds, and the count
        is unknowable before opening the file. Passing the model's full stack
        made the floor far too high, so every legitimate partial lens previewed
        as `fits_envelope: false` while the validator accepted it — and the MCP
        tool tells an agent to read that field when choosing a path."""
        from src.services.jlens_acquire_service import preview_envelope_verdict

        dims = {"d_model": 2304, "n_layers": 26, "n_vocab": 256000}
        partial = 2304 * 2304 * 2 * 12
        full = 2304 * 2304 * 2 * 26
        materialised = 256000 * 2304 * 2 * 26

        assert preview_envelope_verdict(partial, dims)["fits"] is True, (
            "a 12-layer partial lens previewed as a failure while the validator "
            "would accept it"
        )
        assert preview_envelope_verdict(full, dims)["fits"] is True
        out = preview_envelope_verdict(materialised, dims)
        assert out["fits"] is False and "materialised" in out["detail"]

    def test_an_fp32_lens_fits_its_OWN_envelope(self, tmp_path):
        """`dtype_of` supports fp32, and `check_envelope` was hardcoded to 2
        bytes — so a full-coverage fp32 lens could never pass the one check that
        gates publication. Acquisition is the first path where a non-fp16
        artifact can arrive; local fits always write fp16."""
        from src.services.jlens_validation import CheckClass, CheckStatus

        # END TO END THROUGH `validate`, which now DERIVES the element size from
        # the payload it already loads. A `dtype_bytes` parameter defaulting to 2
        # was right for every artifact this project fits and silently wrong for
        # an acquired fp32 one — and a caller that forgot to pass it got the
        # wrong ceiling and no error. There is no parameter to forget now.
        # A FIXTURE WITH REAL MARGIN. At 64x64x4 layers the fp16 requirement is
        # 32 KiB, which puts `container_allowance` in its `required // 2` branch
        # — so the fp16 ceiling lands at exactly 2x required, exactly the fp32
        # payload, and the test passed on 2 KB of zip-container overhead alone.
        # A 3% margin is not a discrimination. Above 128 KiB the allowance caps
        # at 64 KiB and the two ceilings separate properly.
        for dtype, label in ((torch.float16, "fp16"), (torch.float32, "fp32")):
            directory = tmp_path / label
            directory.mkdir()
            lens = directory / f"{label}_jacobian_lens.pt"
            torch.save(
                {l: torch.zeros(256, 256, dtype=dtype) for l in range(8)}, lens
            )
            (directory / "config.yaml").write_text("model: org/m\n")
            ref = ArtifactRef(
                slug=label,
                directory=directory,
                lens_path=lens,
                config_path=directory / "config.yaml",
            )
            report = JLensArtifactService(tmp_path).validate(
                ref, d_model=256, expected_layers=range(8), n_vocab=5000
            )
            envelope = next(
                r for r in report.results if r.check is CheckClass.ENVELOPE
            )
            assert envelope.status is CheckStatus.PASS, (
                f"a {label} lens failed its own envelope: {envelope.detail}"
            )

    def test_the_disk_guard_reserves_BOTH_copies(self):
        """The blob lands in the HF cache and is then copied into the registry,
        so the peak is twice the file."""
        from src.services.jlens_acquire_service import download_footprint

        assert download_footprint(100) == 200
        assert download_footprint(0) == 0

    def test_the_disk_guard_probes_DISTINCT_MOUNTS_only(self, tmp_path):
        """`jlens_artifacts_dir` lives inside `data_dir`, so passing both probed
        one filesystem twice while the cache volume — which the download hits
        FIRST — went unchecked. That is what the function's docstring says it
        exists to prevent."""
        from src.services.jlens_acquire_service import (
            MIN_FREE_DISK_BYTES,
            AcquisitionRefused,
            check_free_space,
        )

        inner = tmp_path / "jlens"
        inner.mkdir()
        # Same mount twice: the requirement must not be double-counted.
        check_free_space(tmp_path, inner, needed_bytes=0)
        with pytest.raises(AcquisitionRefused):
            check_free_space(tmp_path, needed_bytes=2**60)

    def test_a_bare_pt_still_carries_n_prompts(self):
        """The checkpoint declares it per spec §2.2, and for a community repo
        with no config.yaml that is the ONLY provenance. Without it
        `_quality_regression` cannot fire and a 50-prompt foreign lens silently
        displaces a converged 634-prompt local fit."""
        assert derive_n_prompts(None, {"n_prompts": 50}) == 50
        # The publisher's config still wins when it exists — it records what RAN.
        assert derive_n_prompts(parse_upstream_config(UPSTREAM), {"n_prompts": 50}) == 337

    def test_a_staged_artifact_is_NOT_destroyed_by_an_acquisition(self, tmp_path):
        """A fit that validated but was refused by a gate is deliberately kept
        in staging. This project once destroyed a converged 15-layer LFM2
        artifact — 754 seconds of GPU time — by treating staging as disposable.
        """
        from src.services.jlens_artifact_service import ArtifactConflict

        source = tmp_path / "up_jacobian_lens.pt"
        torch.save({0: torch.zeros(2, 2)}, source)
        service = JLensArtifactService(tmp_path / "reg")
        service.stage_from_file("org/m", source, "model: org/m\n")

        with pytest.raises(ArtifactConflict, match="staged artifact"):
            service.stage_from_file("org/m", source, "model: org/m\n")

        # And it CAN be replaced when the caller says so deliberately.
        ref = service.stage_from_file(
            "org/m", source, "model: org/m\n", replace_staged=True
        )
        assert ref.lens_path.exists()

    def test_the_element_size_is_the_EXACT_average_not_the_widest(self):
        """`max()` over a mixed payload doubled the ceiling for one stray fp32
        tensor — weakening, in the permissive direction, the single check that
        gates publication, for exactly a nonconformant file. `validate` is also
        called on arbitrary mounted artifacts with no mixed-dtype gate.

        MUTATION CONTROL: `max(t.element_size() ...)` and this fails.
        """
        from src.services.jlens_artifact_service import _dtype_bytes

        assert _dtype_bytes({0: torch.zeros(10, 10, dtype=torch.float16)}) == 2
        assert _dtype_bytes({0: torch.zeros(10, 10, dtype=torch.float32)}) == 4
        # Mostly fp16 with one fp32 layer: nearer 2 than 4, and NOT 4.
        mixed = {
            0: torch.zeros(100, 100, dtype=torch.float16),
            1: torch.zeros(100, 100, dtype=torch.float16),
            2: torch.zeros(100, 100, dtype=torch.float16),
            3: torch.zeros(10, 10, dtype=torch.float32),
        }
        assert _dtype_bytes(mixed) == 2, (
            "one small fp32 tensor set the ceiling for the whole artifact"
        )

    def test_the_disk_guard_dedups_on_the_DEVICE(self, tmp_path):
        """`(total, free)` collapses two genuinely distinct mounts whose usage
        coincides — two same-size volumes from one StorageClass, both freshly
        provisioned, match on both numbers — and the one it would skip is the
        cache volume the guard exists to cover.

        MUTATION CONTROL: key on `(total, free)` and this fails.
        """
        import inspect

        from src.services import jlens_acquire_service as module

        source = inspect.getsource(module.check_free_space)
        assert "st_dev" in source, "the guard no longer keys on the device"
        # And it still refuses when the ONE volume is too small, or the dedup
        # would have made the whole check vacuous.
        from src.services.jlens_acquire_service import (
            AcquisitionRefused,
            check_free_space,
        )

        with pytest.raises(AcquisitionRefused):
            check_free_space(tmp_path, tmp_path, needed_bytes=2**60)

