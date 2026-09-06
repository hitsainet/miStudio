"""
Adopt a J-lens someone else fitted, as a background task.

WHY A WORKER AND NOT A REQUEST. The artifact is hundreds of megabytes to several
gigabytes, and adopting it requires a real readout — `ValidationReport.serviceable`
demands SEMANTIC, and `load_for_readout` gates on `serviceable`, so an artifact
committed without one is unreadable by the very panel that would display it. That
means loading the model, which means the GPU.

ON THE `extraction` QUEUE, WITH A COST. That is the single-GPU queue, so a
multi-gigabyte download head-of-line-blocks fits and readouts while it runs. It
is still the right queue: the semantic check needs the model, and the readout
service's single-entry model cache lives here — routing the download elsewhere
would either duplicate the model load or hand the artifact to a worker that
cannot check it. The alternative is publishing an unvalidated lens, which is the
one outcome BR-030 exists to prevent.

THE TASK NAME IS FULLY QUALIFIED ON PURPOSE. `celery_app.task_routes` and
`autodiscover_tasks` both hold ENUMERATED entries per J-space module — there is
no `jlens_*` glob. A short name lands on the default queue silently, and a
missing autodiscover entry means the worker never imports this module at all, so
`.delay()` publishes a message nothing will ever consume. Both are asserted in
`tests/unit/test_jlens_reachable.py`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..core.celery_app import celery_app
from . import jlens_progress
from .task_heartbeat import beat

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.workers.jlens_acquire_tasks.acquire_jlens_artifact",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def acquire_jlens_artifact(
    self,
    model_id: str,
    repo_id: str,
    path_in_repo: str,
    revision: Optional[str] = None,
    access_token: Optional[str] = None,
    allow_coverage_loss: bool = False,
    allow_quality_regression: bool = False,
    replace_staged: bool = False,
) -> Dict[str, Any]:
    """Download a published lens, describe it honestly, validate it, publish it.

    The measurement this performs is the SEMANTIC check — a real readout through
    the downloaded artifact, on the same fixture a local fit is held to. Nothing
    else about the transfer is taken on trust: weight identity comes from the
    publisher's own declaration, the layer convention and target from the
    tensors, and byte identity from a digest of what actually landed.
    """
    import torch

    from ..core.config import settings
    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.jlens_acquire_service import (
        AcquisitionRefused,
        WeightIdentity,
        bytes_identity,
        check_expansion,
        check_free_space,
        check_weight_identity,
        download_footprint,
        config_yaml_for_acquired,
        dtype_of,
        fetch_file,
        fetch_optional,
        file_digest,
        full_fit_ceiling,
        inspect_layers,
        model_dims,
        parse_upstream_config,
        preview_repo,
        publication_blocker,
        sibling_paths,
        write_acquisition_record,
    )
    from ..services.jlens_artifact_service import (
        ArtifactCoverageLoss,
        ArtifactNotValidated,
        ArtifactQualityRegression,
        JLensArtifactService,
        normalise_payload,
    )
    from ..services.jlens_model_registry import ModelNotAvailable, load_for_readout
    from ..services.jlens_validation import defer_consumer_checks
    from ..services.huggingface_sae_service import resolve_hf_token

    jlens_progress.mark_running(self.request.id, progress=1.0)

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")
        repo_of_model = getattr(record, "repo_id", None)
        if not repo_of_model:
            raise ValueError(f"Model {model_id!r} has no repo_id to attach a lens to")

    token = resolve_hf_token(access_token)

    # ------------------------------------------------------------ the source
    self.update_state(state="PROGRESS", meta=beat({"stage": "resolving_source"}))
    preview = preview_repo(repo_id, revision=revision, token=token)
    remote = next((c for c in preview.candidates if c.path == path_in_repo), None)
    if remote is None:
        raise AcquisitionRefused(
            f"{path_in_repo!r} is not a downloadable lens candidate in "
            f"{repo_id}@{preview.revision[:8]}. Candidates: "
            f"{[c.path for c in preview.candidates][:5]}"
        )

    # REFUSED BEFORE A BYTE MOVES. A download that cannot fit fails halfway and
    # leaves the volume full — and this volume also holds every model, dataset
    # and checkpoint.
    # BOTH COPIES, ON EVERY VOLUME THEY TOUCH. The blob lands in the HF cache
    # and is then copied into the registry, so the peak is twice the file — and
    # the cache may be a different mount, which the previous call never probed.
    needed = download_footprint(int(remote.size_bytes or 0))
    if not remote.size_bytes:
        logger.warning(
            "%s reports no size; the disk guard falls back to the bare floor",
            path_in_repo,
        )
    check_free_space(
        settings.jlens_artifacts_dir,
        settings.hf_cache_dir,
        needed_bytes=needed,
    )

    self.update_state(
        state="PROGRESS", meta=beat({"stage": "downloading", "bytes": needed})
    )
    jlens_progress.update_row(self.request.id, progress=10.0)

    # PINNED TO THE REVISION THE PREVIEW RESOLVED, so the file that was
    # inspected is the file that arrives, and "acquired from X@Y" is a
    # statement someone else can check.
    lens_file = fetch_file(repo_id, path_in_repo, preview.revision, token)
    siblings = sibling_paths(path_in_repo)
    config_file = fetch_optional(repo_id, siblings["config"], preview.revision, token)
    convergence_file = fetch_optional(
        repo_id, siblings["convergence"], preview.revision, token
    )
    if config_file is None and remote.has_config:
        # THE PREVIEW SAW ONE. A transient 429, an expired token or a 5xx would
        # otherwise turn "the publisher says these are OTHER weights" — the one
        # hard refusal this feature has — into a publishable `unverified`, and
        # the lens would serve a complete, plausible readout that is wrong.
        # Absence is a real state only when the file is genuinely absent.
        raise AcquisitionRefused(
            f"{siblings['config']} is listed in {repo_id}@{preview.revision[:8]} "
            "but could not be fetched, so weight identity cannot be checked. "
            "Retry rather than adopting it unverified."
        )
    upstream_config = (
        parse_upstream_config(config_file.read_text(encoding="utf-8"))
        if config_file
        else None
    )

    jlens_progress.update_row(self.request.id, progress=40.0)
    self.update_state(state="PROGRESS", meta=beat({"stage": "inspecting"}))

    # BOUNDED BEFORE IT IS OPENED. Every other guard bounds the file on disk;
    # this bounds what it becomes. A 34 KB archive of zeros expands to 33.5 MB —
    # measured — and at that ratio a file small enough to pass the envelope
    # OOM-kills this worker, which is the single-GPU queue.
    dims = model_dims(record)
    if dims:
        check_expansion(lens_file, full_fit_ceiling(dims))

    checkpoint = torch.load(lens_file, map_location="cpu", weights_only=True)
    payload = normalise_payload(checkpoint)
    # THE WRAPPER'S OWN DECLARATIONS, KEPT. `n_prompts` and `d_model` live in the
    # checkpoint per spec §2.2, and for a community repo shipping a bare `.pt`
    # with no config.yaml they are the ONLY provenance there is.
    declared = checkpoint if isinstance(checkpoint, dict) else {}

    # ---------------------------------------------------------- the model
    # AFTER the download, because the endpoint already refused synchronously
    # when the weights are absent; this load is belt-and-braces rather than the
    # primary guard, and doing it first would pay a model load to discover a
    # bad path.
    # LOADED INSIDE THE GUARDED BLOCK. Outside it, a CUDA OOM part-way through
    # `from_pretrained` left whatever had been allocated with no `clear_cache()`
    # — and the failure path is the one this task takes most often.
    device = "cuda" if torch.cuda.is_available() else None
    loaded = None
    try:
        try:
            loaded = load_for_readout(record, capture_device=device)
        except ModelNotAvailable as exc:
            raise AcquisitionRefused(str(exc)) from exc

        identity = check_weight_identity(upstream_config, repo_of_model)
        if identity.state is WeightIdentity.MISMATCH:
            # A REFUSAL, NOT A BADGE. The readout would be complete, plausible
            # and wrong, and nothing downstream can tell that from a good one.
            raise AcquisitionRefused(identity.detail)

        observed_dtype = dtype_of(payload)
        # THE CHECKPOINT'S OWN d_model, CROSS-CHECKED. Spec §2.2 makes it the
        # one field read without a fallback, so it is a claim by the publisher —
        # and two independent declarations of the same fact disagreeing is a
        # wrong-model signal that costs nothing to test.
        declared_d_model = declared.get("d_model")
        if declared_d_model is not None and int(declared_d_model) != int(loaded.d_model):
            raise AcquisitionRefused(
                f"the checkpoint declares d_model={int(declared_d_model)} but "
                f"{repo_of_model} has {int(loaded.d_model)}; these are not the "
                "same weights"
            )

        layers = inspect_layers(
            payload, n_layers=int(loaded.n_layers), d_model=int(loaded.d_model)
        )
        config_yaml = config_yaml_for_acquired(
            repo_id=repo_of_model,
            layers=layers,
            n_vocab=int(loaded.n_vocab),
            n_layers=int(loaded.n_layers),
            dtype=observed_dtype,
            upstream_config=upstream_config,
            checkpoint=declared,
        )

        service = JLensArtifactService(settings.jlens_artifacts_dir)
        sidecars = {}
        if convergence_file is not None:
            sidecars[f"{loaded.name.split('/')[-1].lower()}_convergence.csv"] = (
                convergence_file
            )
        ref = service.stage_from_file(
            repo_of_model,
            lens_file,
            config_yaml,
            sidecars=sidecars or None,
            # ITS OWN FLAG. Coupling this to the two gate flags trapped the case
            # the worker calls likeliest: an unserviceable lens leaves staging
            # populated, so a plain RE-RUN of the same acquisition died with
            # ArtifactConflict, and the only escapes also disabled the coverage
            # and quality gates at commit — three unrelated decisions on one
            # switch, in both directions.
            replace_staged=replace_staged,
        )

        # DIGESTED ONCE, AND BEFORE `commit`. `commit` ends in
        # `staging.rename(final)`, so `ref.lens_path` no longer exists
        # afterwards — reading it again to build the return value raised
        # FileNotFoundError on every successful acquisition of an LFS-hosted
        # lens, i.e. every real one. The artifact published, the row was
        # stamped completed, and then `owns_its_failure` flipped it to failed
        # over a lens that had actually landed.
        local_sha256 = file_digest(ref.lens_path)
        bytes_identical = bytes_identity(remote.sha256, local_sha256)

        # THE CACHE COPY IS NOW REDUNDANT. `download_footprint` reserves twice
        # the file because both exist at once — but nothing ever removed the
        # first, so ten refused attempts at a 1.5 GB lens left 15 GB behind with
        # nothing pointing at it. Removed AFTER the copy has landed, and failure
        # to remove is a warning: the bytes we serve are already safe.
        try:
            lens_file.unlink()
        except OSError as exc:  # noqa: BLE001 - the artifact is already staged
            logger.warning("Could not reclaim the cached download: %s", exc)

        write_acquisition_record(
            ref.directory,
            source_repo=repo_id,
            source_path=path_in_repo,
            revision=preview.revision,
            upstream_sha256=remote.sha256,
            local_sha256=local_sha256,
            identity=identity,
            layers=layers,
            upstream_config=upstream_config,
        )

        jlens_progress.update_row(self.request.id, progress=65.0)
        self.update_state(state="PROGRESS", meta=beat({"stage": "validating"}))

        # THE SAME FIXTURE A LOCAL FIT IS HELD TO, through the same helper. A
        # separate probe here would let an acquired artifact be published on an
        # easier test than one miStudio fitted itself.
        from .jlens_fit_tasks import _run_semantic_check
        from ..api.v1.endpoints.jlens import (
            SEMANTIC_FIXTURE_ANSWER,
            SEMANTIC_FIXTURE_CONTROL,
            SEMANTIC_FIXTURE_PROMPT,
        )

        semantic_result = _run_semantic_check(
            service=service,
            ref=ref,
            loaded=loaded,
            probe={
                "prompt": SEMANTIC_FIXTURE_PROMPT,
                "expected_intermediate": SEMANTIC_FIXTURE_ANSWER,
                "control_prompt": SEMANTIC_FIXTURE_CONTROL,
            },
            fitted_layers=layers.fitted,
        )

        report = service.validate(
            ref,
            d_model=int(loaded.d_model),
            expected_layers=layers.fitted,
            n_vocab=int(loaded.n_vocab),
            semantic_result=semantic_result,
        )

        jlens_progress.update_row(self.request.id, progress=85.0)

        published = False
        unpublished_reason: Optional[str] = None
        displaced: Optional[Dict[str, Any]] = None

        # GUARDED LIKE THE FIT WORKER IS. `commit` raises ArtifactNotValidated
        # for any report short of a full pass, and the handler below caught only
        # the two gate refusals — so a lens that simply did not surface the
        # fixture token, which is the LIKELIEST outcome for a foreign lens,
        # crashed the task instead of returning its report. The caller then
        # never saw `check_semantic`'s per-layer evidence, which is the only way
        # to tell "bad lens" from "wrong fixture".
        blocker = publication_blocker(report)
        if blocker is not None:
            unpublished_reason = blocker
            logger.warning("Acquired lens not serviceable: %s", report.summary())
        else:
            try:
                incumbent = service.find(repo_of_model)
                if incumbent is not None:
                    displaced = service._recipe_summary(incumbent)  # noqa: SLF001
                service.commit(
                    repo_of_model,
                # ITS OWN WORDING. Copying the fit worker's sentence would put
                # "requires a live external consumer" over an artifact nobody
                # fitted here, implying a local fit was performed.
                    defer_consumer_checks(report),
                    allow_coverage_loss=allow_coverage_loss,
                    allow_quality_regression=allow_quality_regression,
                )
                published = True
            except (
                ArtifactCoverageLoss,
                ArtifactQualityRegression,
                ArtifactNotValidated,
            ) as exc:
                # THE STAGED ARTIFACT SURVIVES. It is a completed download;
                # throwing it away because the gate refused means paying the
                # bandwidth again to make the same decision with a flag set.
                unpublished_reason = str(exc)
                logger.warning("Acquired lens staged but not published: %s", exc)

        jlens_progress.update_row(
            self.request.id, status="completed", progress=100.0
        )
        return {
            "model": repo_of_model,
            "source": {
                "repo": repo_id,
                "path": path_in_repo,
                "revision": preview.revision,
            },
            "published": published,
            "unpublished_reason": unpublished_reason,
            # WHAT WAS ARCHIVED, NAMED. `_quality_regression` cannot refuse an
            # acquired lens carrying no `converged` key over a converged local
            # fit — `True == None` is False, so the rule does not fire. The
            # backstop is `.superseded` plus `restore_superseded`, and the
            # caller has to be TOLD, or the backstop is one nobody reaches for.
            "displaced": displaced,
            "weight_identity": identity.state.value,
            "weight_identity_detail": identity.detail,
            "bytes_identical": bytes_identical,
            "fitted_layers": layers.fitted,
            "target_layer": layers.target_layer,
            "degenerate_layers": layers.degenerate,
            "validation": {
                "passed": report.passed,
                "serviceable": report.serviceable,
                # FALSE, AND CORRECTLY SO. No interop harness has run against
                # this or any other artifact this project holds.
                "cleared_for_handover": report.cleared_for_handover,
                "results": [
                    {"check": r.check.value, "status": r.status.value, "detail": r.detail}
                    for r in report.results
                ],
            },
            "evidence_rung": 0,
            "caveat": (
                "Adopting a lens checks that it is conformant and that it "
                "discriminates on our fixture. It does not reproduce the fit. "
                "Weight identity rests on the publisher's own declaration, and "
                "is recorded as unverified when they made none."
            ),
        }
    finally:
        if device == "cuda":
            from ..services.jlens_model_registry import clear_cache

            # DROP EVERY REFERENCE, NOT JUST `loaded`. `clear_cache` runs gc then
            # `empty_cache`, so anything still holding the model keeps every
            # block allocated. The intervention task documents this as
            # hardware-proven — 2570 MiB stayed resident when only `loaded` was
            # nulled — and `_run_semantic_check` builds a `ReadoutService` with
            # `model=loaded.model`, an independent strong reference that lives in
            # its frame for as long as a propagating traceback holds it.
            #
            # THE TRACEBACK'S FRAMES ARE CLEARED, not merely inspected. The
            # first version called `sys.exc_info()` and a comment claimed that
            # cleared it — Python 3 has no `exc_clear`, and `exc_info()` is a
            # pure read, so the fix was a no-op on the exact path it named. On a
            # propagating exception the traceback keeps every frame between here
            # and the raise alive, and one of them is `_run_semantic_check`'s,
            # which holds a `ReadoutService` built with `model=loaded.model`.
            import sys
            import traceback as _traceback

            pending = sys.exc_info()[1]
            if pending is not None and pending.__traceback__ is not None:
                _traceback.clear_frames(pending.__traceback__)

            loaded = None  # noqa: F841 - the assignment IS the release
            payload = None  # noqa: F841 - d_model x d_model per layer, on CPU
            checkpoint = None  # noqa: F841 - the undivided original
            clear_cache()
            logger.info("Released the acquisition model from GPU memory")

def _report_cleared_for_handover(report: Dict[str, Any]) -> bool:
    """Rebuild `ValidationReport.cleared_for_handover` from a stored verdict.

    FAILS CLOSED ON A MISSING CLASS, which is the whole point: a bare
    `all(status == "pass")` is vacuously True over an empty or partial result
    list, and this value is what an MCP caller and the UI read as the handover
    verdict.
    """
    from ..services.jlens_validation import CheckClass

    rows = report.get("results") or []
    seen = {r.get("check") for r in rows}
    if seen != {c.value for c in CheckClass}:
        return False
    return all(r.get("status") == "pass" for r in rows)


@celery_app.task(
    name="src.workers.jlens_acquire_tasks.publish_jlens_artifact",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def publish_jlens_artifact_task(
    self,
    model_id: str,
    target_repo: str,
    access_token: Optional[str] = None,
    dataset: str = "mistudio",
    create_repo: bool = False,
    private: bool = False,
) -> Dict[str, Any]:
    """Upload this model's published lens to HuggingFace.

    ON THE SAME QUEUE AS ITS SIBLINGS AND NOT BECAUSE IT NEEDS A GPU — it does
    not. It is here because it reads the artifact directory, which the fit and
    acquire tasks write, and putting a writer of that directory on a different
    worker would make the interleaving something to reason about rather than
    something the queue already serialises.

    ONLY A VALIDATED, PUBLISHED ARTIFACT. `load_for_readout` gates local serving
    on `serviceable`; publishing to a third party is a stronger act than serving
    it here, so a staged or unvalidated artifact is refused outright.
    """
    from ..core.config import settings
    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.huggingface_sae_service import resolve_hf_token
    from ..services.jlens_acquire_service import AcquisitionRefused, publish_artifact
    from ..services.jlens_artifact_service import JLensArtifactService

    # THE ROW MAY NOT EXIST YET. `open_row` runs in the endpoint AFTER
    # `.delay()`, so a task that fails in its first milliseconds can reach
    # `fail_row` before the row is written — `update_row` then finds nothing,
    # returns silently, and the row lands as "queued 0%" and never moves.
    # Retried briefly rather than assumed: the work is the point and the row is
    # the narration, so this must never raise.
    jlens_progress.mark_running(self.request.id, progress=5.0)

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")
        repo_of_model = getattr(record, "repo_id", None)
        if not repo_of_model:
            raise ValueError(f"Model {model_id!r} has no repo_id")

    service = JLensArtifactService(settings.jlens_artifacts_dir)
    ref = service.find(repo_of_model)
    if ref is None:
        raise AcquisitionRefused(
            f"no published J-lens artifact for {repo_of_model}. Fit or acquire "
            "one first — a staged artifact is not published and is not shipped."
        )

    report = service.stored_report(ref)
    if report is None:
        raise AcquisitionRefused(
            f"the artifact for {repo_of_model} carries no validation verdict "
            "matching its current weights, so there is nothing to publish it on."
        )

    # A WRITE TOKEN IS NOT OPTIONAL AND HAS NO SAFE FALLBACK. The read path
    # degrades to anonymous; an upload cannot, and an empty string reaches the
    # Hub as a malformed credential rather than as "no credential".
    token = resolve_hf_token(access_token)
    if not token:
        raise AcquisitionRefused(
            "publishing needs a HuggingFace token with write access; none was "
            "supplied and none is configured"
        )

    jlens_progress.update_row(self.request.id, progress=30.0)

    # BEATEN WHILE IT UPLOADS. One heartbeat before a transfer of unbounded
    # duration is how `cleanup_orphaned_tasks` comes to mark a still-running
    # publish as failed: it sees a `running` row whose Celery heartbeat is older
    # than STALE_AFTER_SECONDS and writes "presumed gone… Any work it completed
    # was not saved. Re-run it." A 276 MB lens over an ordinary uplink passes
    # ten minutes routinely, so the user would be told to re-run a publish that
    # is in flight and about to land.
    import threading

    done = threading.Event()

    def _heartbeat() -> None:
        while not done.wait(30.0):
            try:
                self.update_state(
                    state="PROGRESS", meta=beat({"stage": "uploading"})
                )
            except Exception:  # noqa: BLE001 - a blip must not stop the pulse
                # UNGUARDED, THIS THREAD DIES SILENTLY. One backend hiccup ends
                # the loop, heartbeats stop mid-upload, and the janitor marks a
                # still-running publish failed — restoring by transient exactly
                # the bug this thread exists to prevent.
                logger.debug("Heartbeat write failed; continuing", exc_info=True)

    self.update_state(state="PROGRESS", meta=beat({"stage": "uploading"}))
    pulse = threading.Thread(target=_heartbeat, daemon=True)
    pulse.start()
    try:
        outcome = publish_artifact(
            ref.directory,
            repo_of_model,
            target_repo,
            token,
            dataset=dataset,
            create=create_repo,
            private=private,
            recipe=service._recipe_summary(ref),  # noqa: SLF001 - same package
            validation=report,
        )
    finally:
        done.set()
        # JOINED WITHOUT A TIMEOUT ESCAPE. A pulse still in flight when the task
        # stores SUCCESS would write PROGRESS over it, leaving a completed
        # publish reported as running forever. The loop only ever sleeps or
        # writes, so it cannot outlive `done` by more than one write.
        pulse.join(timeout=60.0)
        if pulse.is_alive():
            logger.warning(
                "The upload heartbeat did not stop; a late PROGRESS write may "
                "land after the terminal state"
            )

    jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
    return {
        "model": repo_of_model,
        **outcome,
        # WHAT THE READER OF THAT REPO IS AND IS NOT BEING TOLD. The two
        # consumer-interop classes travel as DEFERRED, so nobody downstream can
        # mistake this project's local verdict for proven interoperability.
        # THE AUTHORITATIVE PROPERTY, not a re-implementation. `all(...)` over an
        # empty list is vacuously True, so a report with no results at all read
        # as "every class literally passed" — the strongest claim this system
        # makes, from no evidence. The sibling acquire task in this file already
        # uses the real one.
        "cleared_for_handover": _report_cleared_for_handover(report),
        "caveat": (
            "Published with the validation report this installation produced. "
            "The two consumer-interop classes are DEFERRED — they need a live "
            "external consumer and have never been run here."
        ),
    }

