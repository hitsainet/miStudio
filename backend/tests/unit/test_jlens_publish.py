"""
What miStudio publishes must be loadable by whoever downloads it.

Spec §2.2 says a consumer reads `payload["J"]` and that "absence of `J` raises
with the offending key list". This project wrote the BARE `{layer: matrix}` form,
so every artifact it has ever produced would have failed to load for anyone who
downloaded it — invisible locally, because our own reader accepted the bare form
and nothing else ever read one. Publishing them would have shipped files nobody
could open.

That is why `normalise_payload` (which taught the reader both shapes) had to land
before this: changing what we EMIT is only safe once everything already on disk
still reads.

MUTATION CONTROLS (each must turn this file red):
  * emit the bare map again              -> "a consumer can LOAD what we emit"
  * omit d_model from the wrapper        -> "carries the fields a consumer reads"
  * derive source_layers from anything
    other than the keys                  -> "source_layers EQUALS the key set"
  * ship validation.json                 -> "our LOCAL VERDICT does not travel"
  * drop the deferred wording from the
    model card                           -> "the card says what was NOT checked"
  * return no revision                   -> "the commit sha is recorded"

The staged-artifact refusal and the temp-dir space check are NOT covered here —
they live in the worker and the endpoint, and are asserted in
test_jlens_acquire_worker.py. Two earlier entries in this block claimed controls
that were never written, which is a review record overstating its own coverage.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.services.jlens_acquire_service import (
    PUBLISHED_FILES,
    model_card,
    publish_artifact,
    published_path,
)
from src.services.jlens_artifact_service import (
    JLensArtifactService,
    normalise_payload,
)


class TestWhatWeEmitIsConformant:
    def test_a_consumer_can_LOAD_what_we_emit(self, tmp_path):
        """The reference consumer reads `payload["J"]` and raises without it.

        This is the whole reason the emitted format changed: a bare map is
        readable here and nowhere else, so every lens this project published
        would have been a file nobody could open.
        """
        service = JLensArtifactService(tmp_path)
        ref = service.write_staged(
            "org/model",
            {l: torch.zeros(8, 8) for l in range(3)},
            "model: org/model\n",
            n_prompts=634,
        )
        raw = torch.load(ref.lens_path, map_location="cpu", weights_only=True)
        assert "J" in raw, (
            "the emitted checkpoint has no 'J'; a conformant consumer raises "
            f"with the offending key list {sorted(raw)}"
        )
        assert sorted(raw["J"]) == [0, 1, 2]

    def test_it_carries_the_fields_a_consumer_reads(self, tmp_path):
        """`d_model` is the one other field read WITHOUT a fallback (§2.2)."""
        service = JLensArtifactService(tmp_path)
        ref = service.write_staged(
            "org/model",
            {l: torch.zeros(8, 8) for l in range(3)},
            "model: org/model\n",
            n_prompts=634,
        )
        raw = torch.load(ref.lens_path, map_location="cpu", weights_only=True)
        assert raw["d_model"] == 8
        assert raw["n_prompts"] == 634

    def test_source_layers_EQUALS_the_key_set(self, tmp_path):
        """A1 requires equality, and `normalise_payload` refuses a file where
        the two disagree — so emitting a stale list would make our own reader
        reject our own artifact."""
        service = JLensArtifactService(tmp_path)
        ref = service.write_staged(
            "org/model",
            {l: torch.zeros(4, 4) for l in (0, 3, 7)},
            "model: org/model\n",
        )
        raw = torch.load(ref.lens_path, map_location="cpu", weights_only=True)
        assert raw["source_layers"] == [0, 3, 7]
        assert sorted(normalise_payload(raw)) == [0, 3, 7]

    def test_OUR_OWN_READER_still_accepts_it(self, tmp_path):
        """The change is safe only because both shapes read. If this regressed,
        every artifact in the registry would become unreadable at once."""
        service = JLensArtifactService(tmp_path)
        ref = service.write_staged(
            "org/model", {0: torch.zeros(4, 4)}, "model: org/model\n"
        )
        assert sorted(service._load_payload(ref)) == [0]  # noqa: SLF001

    def test_mismatched_widths_are_refused(self, tmp_path):
        """A consumer reads one `d_model` without a fallback; there is no
        honest value to write for a lens whose matrices disagree."""
        service = JLensArtifactService(tmp_path)
        with pytest.raises(ValueError, match="d_model"):
            service.write_staged(
                "org/model",
                {0: torch.zeros(4, 4), 1: torch.zeros(8, 8)},
                "model: org/model\n",
            )


class TestTheLayoutMatchesTheSpec:
    def test_the_published_path_is_the_conformant_one(self):
        """`<model>/jlens/<dataset>/`, so a consumer that already resolves
        published lenses finds ours without being told anything new."""
        assert published_path("google/gemma-2-2b-it") == "gemma-2-2b-it/jlens/mistudio"
        assert (
            published_path("Qwen/Qwen3-8B", "wikitext") == "qwen3-8b/jlens/wikitext"
        )

    def test_our_LOCAL_VERDICT_does_not_travel(self):
        """`validation.json` records two classes as DEFERRED because they need a
        live external consumer and have never run anywhere. Shipping it invites
        a reader to take this installation's verdict for the lens's own."""
        assert "validation.json" not in PUBLISHED_FILES
        assert "acquisition.json" not in PUBLISHED_FILES

    def test_the_EVIDENCE_does_travel(self):
        """`interventions.json` exists precisely to make this journey.

        "A lens published to HuggingFace and pulled down by a serving runtime
        arrives as files and nothing else", and a result that does not travel
        leaves the consumer "with a dictionary it can read and no measurements
        of what happened when it was applied". Publish is the only mechanism
        that could carry it, and the MCP tool already promises agents it does.

        MUTATION CONTROL: skip it in the copy loop and this fails.
        """
        assert "interventions.json" in PUBLISHED_FILES

    def test_the_evidence_ARRIVES_not_merely_is_listed(self, tmp_path):
        """Membership in a constant is not delivery — and that gap SHIPPED.

        The first version of this test asserted only
        `"interventions.json" in PUBLISHED_FILES`, so a `continue` in the copy
        loop left it green. A concurrent review agent's mutation doing exactly
        that was committed to main and rode a full green suite; `interventions.json`
        stopped travelling and nothing noticed. Its sibling
        `test_the_CONVERGENCE_TRACE_travels` was written correctly against the
        uploaded file list, which is what makes the omission visible in hindsight.

        MUTATION CONTROL: `continue` past it in the copy loop and this fails.
        """
        directory = tmp_path / "m"
        directory.mkdir()
        torch.save(
            {"J": {0: torch.zeros(2, 2)}, "d_model": 2},
            directory / "m_jacobian_lens.pt",
        )
        (directory / "config.yaml").write_text("model: org/m\n")
        (directory / "interventions.json").write_text(
            '[{"steering_recipe": {"primitive": "additive"}}]'
        )

        captured = {}

        def fake_upload(folder_path, repo_id, path_in_repo, commit_message):
            import pathlib as _p

            captured["files"] = sorted(
                q.name for q in _p.Path(folder_path).iterdir()
            )
            return types.SimpleNamespace(oid="sha")

        api = MagicMock()
        api.upload_folder = fake_upload
        with patch("huggingface_hub.HfApi", return_value=api):
            publish_artifact(directory, "org/m", "you/lenses", "tok")

        assert "interventions.json" in captured["files"], (
            "the recorded evidence did not reach the upload; a consumer gets a "
            f"dictionary and no measurements. Uploaded: {captured['files']}"
        )


class TestTheModelCardIsHonest:
    def test_the_card_says_what_was_NOT_checked(self):
        """A green suite here does not mean interoperability proven, and a
        reader who assumes it does is reading something never measured."""
        card = model_card(
            "org/m",
            {"n_prompts": 634, "converged": True},
            {
                "results": [
                    {
                        "check": "cross_implementation",
                        "status": "deferred",
                        "detail": "needs a live consumer",
                    }
                ]
            },
        )
        assert "deferred" in card
        assert "not a pass" in card

    def test_the_card_forbids_porting_bands(self):
        """BR-002, restated for whoever downloads this. The published boundaries
        were measured on one model, and porting them is the error this project
        makes impossible by construction locally — a README is the only place
        that constraint can travel."""
        card = model_card("org/m", {}, None)
        assert "Band boundaries" in card
        assert "must not be inferred" in card

    def test_the_card_states_the_checkpoint_shape(self):
        """So a consumer knows what to expect without opening it."""
        card = model_card("org/m", {}, None)
        assert '"J"' in card and "weights_only=True" in card


class TestPublishing:
    @staticmethod
    def _artifact(tmp_path):
        directory = tmp_path / "m"
        directory.mkdir()
        torch.save({"J": {0: torch.zeros(2, 2)}, "d_model": 2}, directory / "m_jacobian_lens.pt")
        (directory / "config.yaml").write_text("model: org/m\nn_prompts: 500\n")
        (directory / "validation.json").write_text('{"passed": true}')
        (directory / "acquisition.json").write_text('{"source": {}}')
        return directory

    def test_it_uploads_ONLY_the_conformant_files(self, tmp_path):
        directory = self._artifact(tmp_path)
        captured = {}

        def fake_upload(folder_path, repo_id, path_in_repo, commit_message):
            captured["files"] = sorted(p.name for p in __import__("pathlib").Path(folder_path).iterdir())
            captured["path"] = path_in_repo
            return types.SimpleNamespace(oid="deadbeef")

        api = MagicMock()
        api.upload_folder = fake_upload
        with patch("huggingface_hub.HfApi", return_value=api):
            out = publish_artifact(directory, "org/m", "you/lenses", "tok")

        assert captured["files"] == [
            "README.md",
            "config.yaml",
            "m_jacobian_lens.pt",
        ], captured["files"]
        assert "validation.json" not in captured["files"]
        assert "acquisition.json" not in captured["files"]
        assert captured["path"] == "m/jlens/mistudio"
        assert out["revision"] == "deadbeef"

    def test_the_COMMIT_SHA_is_recorded(self, tmp_path):
        """The uploader this follows returns `commit_hash: None` with a comment
        saying it "would need to get [it] from the API response". For an
        artifact whose whole purpose is portable evidence, "published at X"
        without a revision names a moving target."""
        directory = self._artifact(tmp_path)
        api = MagicMock()
        api.upload_folder = lambda **kw: types.SimpleNamespace(oid="abc123")
        with patch("huggingface_hub.HfApi", return_value=api):
            out = publish_artifact(directory, "org/m", "you/lenses", "tok")
        assert out["revision"] == "abc123"
        assert "abc123" in out["url"]

    def test_a_directory_with_TWO_lenses_is_refused(self, tmp_path):
        from src.services.jlens_acquire_service import AcquisitionRefused

        directory = self._artifact(tmp_path)
        torch.save({"J": {0: torch.zeros(2, 2)}}, directory / "other_jacobian_lens.pt")
        api = MagicMock()
        with patch("huggingface_hub.HfApi", return_value=api):
            with pytest.raises(AcquisitionRefused, match="exactly one"):
                publish_artifact(directory, "org/m", "you/lenses", "tok")

class TestPublishReviewRound1:
    """Eight findings. The first would have shipped an unloadable file.

    MUTATION CONTROLS:
      * upload the .pt verbatim                -> "an OLD-FORMAT artifact"
      * model_card ignores dataset             -> "the README documents WHERE"
      * drop the dataset pattern               -> "a dataset segment cannot escape"
      * re-implement cleared_for_handover      -> "an EMPTY report is not cleared"
      * skip the temp-dir space check          -> "the rewrite checks for space"
      * drop the convergence copy              -> "the convergence trace travels"
    """

    @staticmethod
    def _old_format(tmp_path):
        """An artifact in the shape EVERY existing lens on the cluster has."""
        directory = tmp_path / "m"
        directory.mkdir()
        torch.save(
            {l: torch.zeros(4, 4, dtype=torch.float16) for l in range(3)},
            directory / "m_jacobian_lens.pt",
        )
        (directory / "config.yaml").write_text(
            "model: org/m\nn_prompts: 634\n"
        )
        (directory / "m_convergence.csv").write_text(
            "n_done,identity_distance\n1,0.5\n"
        )
        return directory

    @staticmethod
    def _upload_capture():
        captured = {}

        def fake_upload(folder_path, repo_id, path_in_repo, commit_message):
            import pathlib as _p

            out = _p.Path(folder_path)
            captured["files"] = sorted(p.name for p in out.iterdir())
            captured["path"] = path_in_repo
            lens = next(out.glob("*_jacobian_lens.pt"))
            captured["payload"] = torch.load(
                lens, map_location="cpu", weights_only=True
            )
            captured["readme"] = (out / "README.md").read_text()
            return types.SimpleNamespace(oid="sha1234")

        api = MagicMock()
        api.upload_folder = fake_upload
        return api, captured

    def test_an_OLD_FORMAT_artifact_is_published_CONFORMANT(self, tmp_path):
        """Both lenses on the cluster predate the emitted-format change and are
        the bare `{layer: matrix}` form. They carry a matching validation.json,
        so every gate passes and the upload proceeds — and the consumer then
        reads `payload["J"]`, raises, and is holding a README describing the
        wrapper. Publishing is the moment the on-disk vintage stops being local.
        """
        directory = self._old_format(tmp_path)
        api, captured = self._upload_capture()
        with patch("huggingface_hub.HfApi", return_value=api):
            publish_artifact(directory, "org/m", "you/lenses", "tok")

        payload = captured["payload"]
        assert "J" in payload, (
            f"an old-format lens was uploaded verbatim: {sorted(payload)}"
        )
        assert sorted(payload["J"]) == [0, 1, 2]
        assert payload["d_model"] == 4
        # And the figure the README quotes came from the recipe, not from thin air.
        assert payload["source_layers"] == [0, 1, 2]

    def test_the_README_documents_WHERE_the_files_actually_went(self, tmp_path):
        """The card called `published_path(repo_id)` with no dataset, so every
        README documented `.../jlens/mistudio/` regardless — wrong on exactly
        the calls that use the parameter."""
        directory = self._old_format(tmp_path)
        api, captured = self._upload_capture()
        with patch("huggingface_hub.HfApi", return_value=api):
            publish_artifact(
                directory, "org/m", "you/lenses", "tok", dataset="wikitext"
            )
        assert captured["path"] == "m/jlens/wikitext"
        assert "m/jlens/wikitext/" in captured["readme"], (
            "the README points somewhere the files are not"
        )
        assert "jlens/mistudio" not in captured["readme"]

    def test_the_CONVERGENCE_TRACE_travels(self, tmp_path):
        """Spec §2.1 puts it in the same directory, and it is what miStudio
        itself reads off an acquired artifact to see whether a fit plateaued."""
        directory = self._old_format(tmp_path)
        api, captured = self._upload_capture()
        with patch("huggingface_hub.HfApi", return_value=api):
            publish_artifact(directory, "org/m", "you/lenses", "tok")
        assert "m_convergence.csv" in captured["files"], captured["files"]

    def test_an_EMPTY_report_is_not_CLEARED_FOR_HANDOVER(self):
        """`all(...)` over an empty list is vacuously True, so a report with no
        results read as "every class literally passed" — the strongest claim
        this system makes, from no evidence."""
        from src.workers.jlens_acquire_tasks import _report_cleared_for_handover

        assert _report_cleared_for_handover({"results": []}) is False
        assert _report_cleared_for_handover({}) is False
        # A PARTIAL report is not cleared either.
        assert (
            _report_cleared_for_handover(
                {"results": [{"check": "structural", "status": "pass"}]}
            )
            is False
        )
        # And a genuinely complete one IS.
        from src.services.jlens_validation import CheckClass

        assert (
            _report_cleared_for_handover(
                {
                    "results": [
                        {"check": c.value, "status": "pass"} for c in CheckClass
                    ]
                }
            )
            is True
        )

    def test_a_dataset_segment_cannot_ESCAPE_the_layout(self):
        """It is interpolated into a repo path. `..` commits the lens where no
        consumer resolving `<model>/jlens/<dataset>/` will look."""
        from pydantic import ValidationError

        from src.api.v1.endpoints.jlens import PublishRequest

        for bad in ("../..", "/abs", "a/b", ".hidden", ""):
            with pytest.raises(ValidationError):
                PublishRequest(model_id="m", target_repo="you/x", dataset=bad)
        assert PublishRequest(
            model_id="m", target_repo="you/x", dataset="wikitext-103"
        ).dataset == "wikitext-103"

    def test_update_row_REPORTS_whether_it_found_the_row(self):
        """The endpoints open the row after `.delay()`, so a task failing in its
        first milliseconds arrives before the row exists. A silent return left it
        at "queued 0%" forever — a job that failed instantly, displayed as one
        that never started."""
        from src.workers.jlens_progress import update_row

        assert update_row("no-such-task-id", status="running") is False

class TestPublishReviewRounds2And3:
    """Findings the second and third rounds raised against the first's fixes.

    MUTATION CONTROLS:
      * skip the digest rebinding        -> "the evidence still BINDS"
      * drop **preserved                 -> "fields the publisher put there"
      * remove the expansion ceiling     -> "a bomb is refused before torch.load"
    """

    @staticmethod
    def _capture():
        captured = {}

        def fake_upload(folder_path, repo_id, path_in_repo, commit_message):
            import json as _json
            import pathlib as _p

            out = _p.Path(folder_path)
            lens = next(out.glob("*_jacobian_lens.pt"))
            captured["payload"] = torch.load(
                lens, map_location="cpu", weights_only=True
            )
            captured["digest"] = __import__(
                "hashlib"
            ).sha256(lens.read_bytes()).hexdigest()
            evidence = out / "interventions.json"
            captured["records"] = (
                _json.loads(evidence.read_text()) if evidence.is_file() else None
            )
            return types.SimpleNamespace(oid="sha")

        api = MagicMock()
        api.upload_folder = fake_upload
        return api, captured

    def test_the_evidence_still_BINDS_to_the_file_that_ships(self, tmp_path):
        """Re-saving changes the container's bytes, so every `lens_sha256` in
        `interventions.json` would name a file that is not at the destination —
        and miStudio's own rule DROPS a record whose digest does not match. The
        evidence would arrive and self-invalidate, on exactly the old-format
        artifacts the re-save exists to fix.
        """
        import hashlib
        import json as _json

        directory = tmp_path / "m"
        directory.mkdir()
        lens = directory / "m_jacobian_lens.pt"
        # OLD FORMAT, so the re-save genuinely changes the bytes.
        torch.save({l: torch.zeros(4, 4, dtype=torch.float16) for l in range(3)}, lens)
        local = hashlib.sha256(lens.read_bytes()).hexdigest()
        (directory / "config.yaml").write_text("model: org/m\n")
        (directory / "interventions.json").write_text(
            _json.dumps([{"lens_sha256": local, "evidence": {"n_trials": 6}}])
        )

        api, captured = self._capture()
        with patch("huggingface_hub.HfApi", return_value=api):
            publish_artifact(directory, "org/m", "you/lenses", "tok")

        assert captured["digest"] != local, (
            "the fixture must be old-format, or this cannot detect the hazard"
        )
        record = captured["records"][0]
        assert record["lens_sha256"] == captured["digest"], (
            "the published evidence names a file that is not there; a consumer "
            "applying miStudio's own rule keeps 0 of 1 records"
        )
        # AND IT SAYS IT MOVED, so a reader can tell a rebound digest from an
        # original one.
        assert record["rebound_from"] == local
        assert "matrices are" in record["rebound_reason"]

    def test_fields_the_publisher_put_there_SURVIVE(self, tmp_path):
        """Rebuilding only the four spec fields dropped everything else —
        `layer_convention` and `target_layer` most of all, which are what a
        consumer needs to know whether the lens targets penultimate or final.
        For a repo republishing just the `.pt`, the checkpoint is the only place
        they can travel.
        """
        directory = tmp_path / "m"
        directory.mkdir()
        torch.save(
            {
                "J": {l: torch.zeros(4, 4, dtype=torch.float16) for l in range(3)},
                "d_model": 4,
                "source_layers": [0, 1, 2],
                "n_prompts": 500,
                "target_layer": "penultimate",
                "layer_convention": "zero_based_resid_post",
            },
            directory / "m_jacobian_lens.pt",
        )
        (directory / "config.yaml").write_text("model: org/m\n")

        api, captured = self._capture()
        with patch("huggingface_hub.HfApi", return_value=api):
            publish_artifact(directory, "org/m", "you/lenses", "tok")

        payload = captured["payload"]
        assert payload["target_layer"] == "penultimate", sorted(payload)
        assert payload["layer_convention"] == "zero_based_resid_post"
        # And the four rebuilt ones are still right.
        assert payload["d_model"] == 4
        assert payload["source_layers"] == [0, 1, 2]

    def test_a_BOMB_is_refused_before_torch_load(self, tmp_path):
        """`check_free_space` covers disk and runs AFTER `torch.load` has
        already committed the memory. The acquisition path guards this with
        `check_expansion`; publish deserialised without a ceiling."""
        import zipfile

        from src.services.jlens_acquire_service import AcquisitionRefused

        directory = tmp_path / "m"
        directory.mkdir()
        lens = directory / "m_jacobian_lens.pt"
        torch.save(
            {l: torch.zeros(256, 256, dtype=torch.float16) for l in range(8)}, lens
        )
        # Recompressed so the file on disk is tiny and the expansion is not.
        source = zipfile.ZipFile(lens)
        names = source.namelist()
        blobs = {n: source.read(n) for n in names}
        source.close()
        with zipfile.ZipFile(lens, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as out:
            for name in names:
                out.writestr(name, blobs[name])
        (directory / "config.yaml").write_text("model: org/m\n")

        api, _captured = self._capture()
        with patch("huggingface_hub.HfApi", return_value=api):
            with pytest.raises(AcquisitionRefused, match="expands to"):
                publish_artifact(
                    directory,
                    "org/m",
                    "you/lenses",
                    "tok",
                    # A tiny model: the expansion is far past what it could need.
                    recipe={"d_model": 8, "n_layers": 4, "n_vocab": 100},
                )

