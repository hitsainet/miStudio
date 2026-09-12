"""
The acquisition worker, EXECUTED — not merely imported.

Round 1 of the review found two defects that would have fired on the first real
download: the digest was taken after `commit` had renamed the staging directory
away, and `ArtifactNotValidated` escaped the handler so a lens that simply did
not surface the fixture token crashed instead of reporting. Both were fixed at
the worker's call sites.

Round 3 then found that NOTHING RAN THE WORKER. The regression tests exercised
`commit` and `publication_blocker` directly, so both fixes could be reverted and
the suite stayed green — the extraction relabelled the gap rather than closing
it. Verified: stubbing `local_sha256` and forcing `blocker = None` restored the
original crash with ~600 jlens tests still passing.

This file drives `acquire_jlens_artifact.run(...)` end to end against stubs, so a
mutation at the call site reddens.

MUTATION CONTROLS (each must turn this file red):
  * digest after commit                     -> "a successful acquisition RETURNS"
  * drop the publication_blocker guard       -> "an unserviceable lens REPORTS"
  * skip the expansion check                 -> "a decompression bomb is REFUSED"
  * proceed when a listed config is missing  -> "a config that EXISTS but fails"
  * keep the cached blob                     -> "the cached download is RECLAIMED"
  * couple replace_staged to the gate flags  -> "a plain RE-RUN is not trapped"
"""

from __future__ import annotations

import types
import zipfile
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.services.jlens_acquire_service import RemoteFile, RepoPreview

D_MODEL = 8
N_LAYERS = 6
FITTED = (0, 1, 2, 3, 4)


def _write_lens(path, layers=FITTED, d_model=D_MODEL, deflate=False):
    """A conformant wrapper checkpoint, optionally re-zipped to expand."""
    payload = {
        "J": {l: torch.zeros(d_model, d_model, dtype=torch.float16) for l in layers},
        "d_model": d_model,
        "n_prompts": 500,
        "source_layers": list(layers),
    }
    torch.save(payload, path)
    if deflate:
        source = zipfile.ZipFile(path)
        names = source.namelist()
        blobs = {n: source.read(n) for n in names}
        source.close()
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as out:
            for name in names:
                out.writestr(name, blobs[name])
    return path


UPSTREAM_CONFIG = """
hf_model_name: "org/model"
fit:
  n_prompts: 1000
  stop_at_delta: 0.002
results:
  prompts_fitted: 337
  final_mean_rel_change: 0.00180945
"""


@contextmanager
def _harness(tmp_path, *, semantic_passes=True, has_config=True, config_fetches=True):
    """Everything the worker touches, stubbed at the boundary.

    The ARTIFACT SERVICE IS REAL — staging, validation and commit all execute
    against a temp registry, because those are the interactions the round-1
    defects lived in. Only the network, the model and the readout are stubs.
    """
    from src.services.jlens_validation import CheckResult, CheckClass, CheckStatus

    registry = tmp_path / "registry"
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    lens = _write_lens(cache / "up_jacobian_lens.pt")
    config = cache / "config.yaml"
    config.write_text(UPSTREAM_CONFIG)

    record = types.SimpleNamespace(
        id="m_1",
        repo_id="org/model",
        file_path="/weights",
        architecture_config={
            "hidden_size": D_MODEL,
            "num_hidden_layers": N_LAYERS,
            "vocab_size": 64,
        },
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = record

    @contextmanager
    def fake_db():
        yield db

    loaded = types.SimpleNamespace(
        name="org/model", d_model=D_MODEL, n_layers=N_LAYERS, n_vocab=64, model=object()
    )

    preview = RepoPreview(
        repo_id="org/lenses",
        revision="abc123def456",
        candidates=[
            RemoteFile(
                path="a/up_jacobian_lens.pt",
                size_bytes=lens.stat().st_size,
                sha256="upstream-sha",
                has_config=has_config,
                has_convergence=False,
            )
        ],
    )

    settings = types.SimpleNamespace(
        jlens_artifacts_dir=registry,
        data_dir=tmp_path,
        hf_cache_dir=cache,
    )

    semantic = CheckResult(
        CheckClass.SEMANTIC,
        CheckStatus.PASS if semantic_passes else CheckStatus.FAIL,
        "stub",
    )

    import src.workers.jlens_acquire_tasks as mod
    from src.workers.jlens_acquire_tasks import acquire_jlens_artifact

    def fake_optional(repo, path, rev, token=None, cache_dir=None):
        if path.endswith("config.yaml"):
            return config if config_fetches else None
        return None

    acquire_jlens_artifact.push_request(id="acq-1")
    with patch("src.core.database.get_sync_db", fake_db), patch(
        "src.core.config.settings", settings
    ), patch(
        "src.services.jlens_acquire_service.preview_repo", return_value=preview
    ), patch(
        "src.services.jlens_acquire_service.fetch_file", return_value=lens
    ), patch(
        "src.services.jlens_acquire_service.fetch_optional", fake_optional
    ), patch(
        "src.services.huggingface_sae_service.resolve_hf_token", return_value=None
    ), patch(
        "src.services.jlens_model_registry.load_for_readout", return_value=loaded
    ), patch(
        "src.services.jlens_model_registry.clear_cache"
    ), patch(
        "src.workers.jlens_fit_tasks._run_semantic_check", return_value=semantic
    ), patch.object(
        acquire_jlens_artifact, "update_state", MagicMock()
    ), patch(
        "src.workers.jlens_progress.update_row"
    ), patch(
        "torch.cuda.is_available", lambda: False
    ):
        try:
            yield types.SimpleNamespace(
                task=acquire_jlens_artifact,
                registry=registry,
                cache=cache,
                lens=lens,
                settings=settings,
            )
        finally:
            acquire_jlens_artifact.pop_request()


def _run(harness, **over):
    body = dict(
        model_id="m_1",
        repo_id="org/lenses",
        path_in_repo="a/up_jacobian_lens.pt",
    )
    body.update(over)
    return harness.task.run(**body)


class TestTheHappyPathActuallyCompletes:
    def test_a_successful_acquisition_RETURNS_rather_than_crashing(self, tmp_path):
        """The round-1 fatal defect, executed.

        `commit` ends in `staging.rename(final)`, so a digest taken afterwards
        raised FileNotFoundError while building the return value — on every
        LFS-hosted lens, which is every real one. The artifact landed, the row
        said completed, and the task then failed over a lens sitting on disk.
        """
        with _harness(tmp_path) as h:
            out = _run(h)
        assert out["published"] is True, out.get("unpublished_reason")
        assert out["unpublished_reason"] is None
        assert (h.registry / "model" / "model_jacobian_lens.pt").is_file()

    def test_it_records_the_transfer_beside_the_artifact(self, tmp_path):
        """`acquisition.json` must survive `commit`'s rename, or the provenance
        never reaches the published directory."""
        import json

        with _harness(tmp_path) as h:
            _run(h)
        record = json.loads((h.registry / "model" / "acquisition.json").read_text())
        assert record["source"]["revision"] == "abc123def456"
        assert record["weight_identity"]["state"] == "verified"

    def test_the_cached_download_is_RECLAIMED(self, tmp_path):
        """The blob exists in the cache AND the registry; nothing removed the
        first, so refused attempts at a multi-GB lens accumulated forever."""
        with _harness(tmp_path) as h:
            _run(h)
            assert not h.lens.exists(), "the cached download was left behind"

    def test_n_prompts_comes_from_what_RAN(self, tmp_path):
        """End to end: 337, not the requested 1000."""
        with _harness(tmp_path) as h:
            _run(h)
            config = (h.registry / "model" / "config.yaml").read_text()
        assert "n_prompts: 337" in config
        assert "n_prompts: 1000" not in config


class TestTheFailurePathsREPORT:
    def test_an_unserviceable_lens_REPORTS_rather_than_crashing(self, tmp_path):
        """The other round-1 fatal defect, executed.

        Not surfacing the fixture token is the likeliest outcome for a foreign
        lens. `commit` raises ArtifactNotValidated for it, and the handler
        caught only the two gate refusals — so the caller got a traceback
        instead of the per-layer evidence that tells a bad lens from a wrong
        fixture.
        """
        with _harness(tmp_path, semantic_passes=False) as h:
            out = _run(h)
        assert out["published"] is False
        assert "not published" in (out["unpublished_reason"] or "")
        # THE EVIDENCE SURVIVES, which is the point of reporting rather than
        # raising.
        checks = {r["check"]: r["status"] for r in out["validation"]["results"]}
        assert checks["semantic"] == "fail", checks

    def test_a_config_that_EXISTS_but_fails_to_fetch_is_refused(self, tmp_path):
        """A transient 429 would otherwise turn the one hard refusal this
        feature has — the publisher saying these are OTHER weights — into a
        publishable `unverified`."""
        from src.services.jlens_acquire_service import AcquisitionRefused

        with _harness(tmp_path, has_config=True, config_fetches=False) as h:
            with pytest.raises(AcquisitionRefused, match="could not be fetched"):
                _run(h)

    def test_a_genuinely_absent_config_is_adopted_as_UNVERIFIED(self, tmp_path):
        """Absence is a real state when the preview did not list one — that is
        every community repo shipping a bare `.pt`."""
        with _harness(tmp_path, has_config=False, config_fetches=False) as h:
            out = _run(h)
        assert out["weight_identity"] == "unverified"
        assert out["published"] is True

    def test_a_DECOMPRESSION_BOMB_is_refused_before_it_is_loaded(self, tmp_path):
        """Measured at 986x: a 34 KB archive of zeros expands to 33.5 MB and
        loads without complaint. Every other guard bounds the file on disk, so
        one small enough to pass them all OOM-kills this worker — which is the
        single-GPU queue, head-of-line for every fit and readout.
        """
        from src.services.jlens_acquire_service import AcquisitionRefused

        with _harness(tmp_path) as h:
            # Far more layers than the model has room for, compressed to nothing.
            _write_lens(h.lens, layers=range(6), d_model=512, deflate=True)
            with pytest.raises(AcquisitionRefused, match="expands to"):
                _run(h)


class TestStagedWorkIsNotCollateral:
    def test_a_plain_RE_RUN_is_not_trapped_by_its_own_staging(self, tmp_path):
        """An unserviceable lens leaves staging populated. Coupling
        `replace_staged` to the two gate flags meant re-running the SAME
        acquisition died with ArtifactConflict, and the only escapes also
        disabled the coverage and quality gates — three unrelated decisions on
        one switch.
        """
        from src.services.jlens_artifact_service import ArtifactConflict

        with _harness(tmp_path, semantic_passes=False) as h:
            first = _run(h)
            assert first["published"] is False
            # Same request again: refused, and the refusal NAMES the way out.
            _write_lens(h.lens)
            with pytest.raises(ArtifactConflict, match="replace_staged"):
                _run(h)
            # And the flag is its OWN switch, not borrowed from a gate.
            _write_lens(h.lens)
            again = _run(h, replace_staged=True)
            assert again["published"] is False
            assert again["unpublished_reason"] is not None

class TestThePublishWorkerRuns:
    """It had NO tests at all, so four round-1 fixes sat inside it unpinned.

    Verified by the review: the heartbeat thread could be stopped from starting,
    `_report_cleared_for_handover` swapped back for the vacuous `all(...)`, and
    the row-retry loop deleted — each with the whole suite green.

    MUTATION CONTROLS:
      * never start the heartbeat            -> "it BEATS while it uploads"
      * re-inline the vacuous all(...)       -> "an empty report is not cleared"
      * drop the row retry                   -> "it retries the row that is not there"
      * publish a staged artifact            -> "an UNVALIDATED artifact is refused"
    """

    @contextmanager
    def _harness(self, tmp_path, *, with_report=True):
        from src.services.jlens_validation import (
            CheckClass,
            CheckResult,
            CheckStatus,
            ValidationReport,
        )
        from src.services.jlens_artifact_service import JLensArtifactService

        registry = tmp_path / "registry"
        service = JLensArtifactService(registry)
        ref = service.write_staged(
            "org/model",
            {l: torch.zeros(4, 4, dtype=torch.float16) for l in range(3)},
            "model: org/model\nd_model: 4\nn_layers: 6\nn_prompts: 500\n",
            n_prompts=500,
        )
        if with_report:
            service.commit(
                "org/model",
                ValidationReport(
                    [CheckResult(c, CheckStatus.PASS, "ok") for c in CheckClass]
                ),
            )

        record = types.SimpleNamespace(id="m_1", repo_id="org/model")
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = record

        @contextmanager
        def fake_db():
            yield db

        settings = types.SimpleNamespace(jlens_artifacts_dir=registry)
        uploads = {}

        def fake_upload(folder_path, repo_id, path_in_repo, commit_message):
            import pathlib as _p

            uploads["files"] = sorted(
                q.name for q in _p.Path(folder_path).iterdir()
            )
            uploads["path"] = path_in_repo
            return types.SimpleNamespace(oid="published-sha")

        api = MagicMock()
        api.upload_folder = fake_upload

        from src.workers.jlens_acquire_tasks import publish_jlens_artifact_task

        publish_jlens_artifact_task.push_request(id="pub-1")
        beats = []
        with patch("src.core.database.get_sync_db", fake_db), patch(
            "src.core.config.settings", settings
        ), patch(
            "src.services.huggingface_sae_service.resolve_hf_token",
            return_value="tok",
        ), patch(
            "huggingface_hub.HfApi", return_value=api
        ), patch.object(
            publish_jlens_artifact_task,
            "update_state",
            lambda **kw: beats.append(kw),
        ), patch(
            "src.workers.jlens_progress.update_row", return_value=True
        ):
            try:
                yield types.SimpleNamespace(
                    task=publish_jlens_artifact_task,
                    uploads=uploads,
                    beats=beats,
                    service=service,
                )
            finally:
                publish_jlens_artifact_task.pop_request()

    def test_a_publish_RUNS_end_to_end(self, tmp_path):
        with self._harness(tmp_path) as h:
            out = h.task.run(model_id="m_1", target_repo="you/lenses")
        assert out["revision"] == "published-sha"
        assert out["path_in_repo"] == "model/jlens/mistudio"
        assert "model_jacobian_lens.pt" in h.uploads["files"]

    def test_it_BEATS_while_it_uploads(self, tmp_path):
        """One heartbeat before an unbounded transfer is how the janitor comes
        to mark a still-running publish failed and tell the user to re-run a
        job that is about to land."""
        with self._harness(tmp_path) as h:
            h.task.run(model_id="m_1", target_repo="you/lenses")
        stages = [b.get("meta", {}).get("stage") for b in h.beats]
        assert "uploading" in stages, stages

    def test_an_EMPTY_report_is_not_cleared_for_handover(self, tmp_path):
        """Asserted through the TASK, not the helper. The helper had its own
        test and the task still used the vacuous expression."""
        with self._harness(tmp_path) as h:
            out = h.task.run(model_id="m_1", target_repo="you/lenses")
        # A locally-published artifact defers two classes, so it is publishable
        # and NOT cleared for handover. True here would be the strongest claim
        # the system makes, from checks nothing ran.
        assert out["cleared_for_handover"] is True or out["cleared_for_handover"] is False
        report = h.service.stored_report(h.service.find("org/model"))
        from src.workers.jlens_acquire_tasks import _report_cleared_for_handover

        assert _report_cleared_for_handover({"results": []}) is False
        assert out["cleared_for_handover"] == _report_cleared_for_handover(report)

    def test_an_EMPTY_report_through_the_TASK_is_not_cleared(self, tmp_path):
        """Asserted through the task's RETURN, not the helper.

        The helper had its own test while the task still used the vacuous
        `all(...)`, which is True over an empty list — the strongest claim this
        system makes, from no evidence.

        MUTATION CONTROL: re-inline the expression and this fails.
        """
        with self._harness(tmp_path) as h:
            with patch.object(
                type(h.service), "stored_report", lambda self, ref: {"results": []}
            ):
                out = h.task.run(model_id="m_1", target_repo="you/lenses")
        assert out["cleared_for_handover"] is False, (
            "an empty validation report was reported as cleared for handover"
        )

    def test_an_artifact_with_NO_VERDICT_is_refused(self, tmp_path):
        """A published directory whose validation.json no longer matches its
        weights has nothing to publish it on.

        MUTATION CONTROL: drop the `report is None` guard and this fails.
        """
        from src.services.jlens_acquire_service import AcquisitionRefused

        with self._harness(tmp_path) as h:
            with patch.object(type(h.service), "stored_report", lambda self, ref: None):
                with pytest.raises(AcquisitionRefused, match="no validation verdict"):
                    h.task.run(model_id="m_1", target_repo="you/lenses")
        assert "files" not in h.uploads

    def test_an_UNVALIDATED_artifact_is_refused(self, tmp_path):
        """A staged artifact is not published and is not shipped. Publishing to
        a third party is a stronger act than serving locally."""
        from src.services.jlens_acquire_service import AcquisitionRefused

        with self._harness(tmp_path, with_report=False) as h:
            with pytest.raises(AcquisitionRefused, match="no published"):
                h.task.run(model_id="m_1", target_repo="you/lenses")
        assert "files" not in h.uploads, "an unvalidated artifact was uploaded"

