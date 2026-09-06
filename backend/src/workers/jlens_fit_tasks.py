"""
Celery task for fitting a J-lens artifact (Phase 4.3).

GPU-BOUND AND SINGLE-FLIGHT. Fitting runs a forward and a linearised pass per
layer over a corpus, with the whole model resident. It shares the `extraction`
queue for the same reason circuit validation and calibration do: one GPU, and
these are the jobs that occupy it.

THE TASK NAME IS EXPLICIT AND FULLY QUALIFIED. `task_routes` globs match the
TASK NAME, not the module path, so a task registered under a short name
silently lands on the default queue — a defect this project has already shipped
once. The name here matches the route glob in `celery_app.py` exactly.

STAGE, VALIDATE, THEN COMMIT. The fit writes to a staging directory that
discovery excludes; it is moved into the mounted registry only if validation is
serviceable. A half-written or unvalidated artifact in the mounted directory is
served, and the consumer says nothing about it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from ..core.celery_app import celery_app
from .task_heartbeat import beat
from . import jlens_progress
from ..core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.workers.jlens_fit_tasks.fit_jlens_artifact",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def fit_jlens_artifact(
    self,
    model_id: str,
    prompts: List[str],
    layers: Optional[List[int]] = None,
    # FULL BACKWARD IS THE STANDARD RECIPE (D4). Frozen Q/K is an
    # ABLATION — useful, and not the default. This defaulted to True, so
    # every fit started from the API or MCP silently ran the ablation while
    # the fitter class it constructs defaults to False; the aligned recipe
    # was unreachable without naming the flag explicitly, and the artifact
    # published on 2026-08-03 records `frozen_qk` for exactly this reason.
    freeze_qk: bool = False,
    corpus_name: str = "unspecified",
    semantic_probe: Optional[Dict[str, Any]] = None,
    allow_coverage_loss: bool = False,
    allow_quality_regression: bool = False,
    freeze_norms: bool = False,
    target_layer: str = "penultimate",
    convergence_delta: Optional[float] = None,
) -> Dict[str, Any]:
    """Fit, validate and publish a J-lens artifact for one model.

    Returns a dict rather than raising on a validation failure: a fit that
    produced a real artifact which then failed validation is a RESULT the user
    needs to see per-check, not an opaque task error. A fit that could not run
    at all still raises.

    `max_retries=0` deliberately. A fit takes minutes on a GPU shared with
    serving; an automatic retry of a job that OOMed would take the card again
    at the worst possible moment.
    """
    from ..ml.jlens_fitter import JacobianFitter
    from ..models.model import Model
    from ..services.jlens_artifact_service import (
        ArtifactCoverageLoss,
        ArtifactQualityRegression,
        JLensArtifactService,
    )
    from ..services.jlens_model_registry import load_for_readout
    from ..core.database import get_sync_db

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")
        repo_id = record.repo_id

        # Capture on GPU when one is free: fitting is the one J-space operation
        # that genuinely needs it. The READOUT stays on CPU regardless.
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = load_for_readout(record, capture_device=device)

    # RELEASE THE CARD WHEN THE FIT IS DONE. `load_for_readout` caches the model
    # so a fit does not reload it per prompt, and NOTHING dropped it afterwards:
    # `clear_cache` existed with zero callers. An LFM2 fit left 4.0 GB resident
    # on the shared 3090 at 0% utilisation from the moment it finished until the
    # pod was next restarted — on a card miLLM serves from.
    #
    # `try/finally` around everything after the load, because the release must
    # happen on the failure paths too: an OOM or a validation raise is exactly
    # when the card most needs to come back.
    try:
        return _fit_and_publish(
            self,
            loaded=loaded,
            model_id=model_id,
            repo_id=repo_id,
            prompts=prompts,
            layers=layers,
            freeze_qk=freeze_qk,
            corpus_name=corpus_name,
            semantic_probe=semantic_probe,
            allow_coverage_loss=allow_coverage_loss,
            allow_quality_regression=allow_quality_regression,
            freeze_norms=freeze_norms,
            target_layer=target_layer,
            convergence_delta=convergence_delta,
        )
    finally:
        if device == "cuda":
            from ..services.jlens_model_registry import clear_cache

            # DROP OUR OWN REFERENCE FIRST. `clear_cache` nulls the CACHE's
            # entry and then runs gc + `torch.cuda.empty_cache()` — but this
            # frame still held `loaded`, so there was nothing for gc to collect
            # and `empty_cache` had no free blocks to return. The first version
            # of this release looked correct and recovered only the activation
            # workspace: 7706 MiB fell to 2608 MiB, which is LFM2's fp16 weights
            # still sitting on a card miLLM serves from.
            #
            # The three tests written with it all passed, because they asserted
            # `clear_cache` was CALLED. Being called is not being effective.
            loaded = None  # noqa: F841 - the assignment IS the release
            clear_cache()
            logger.info("Released the fitted model from GPU memory")


@celery_app.task(
    name="src.workers.jlens_fit_tasks.revalidate_staged_artifact",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def revalidate_staged_artifact(
    self,
    model_id: str,
    semantic_probe: Optional[Dict[str, Any]] = None,
    allow_coverage_loss: bool = False,
    allow_quality_regression: bool = False,
) -> Dict[str, Any]:
    """Re-run validation on an ALREADY-FITTED staged artifact, and commit it.

    THE CAPABILITY THE REFUSAL MESSAGE ALREADY PROMISED. When validation
    refuses, `_validate_and_commit` keeps the staging directory and logs "so it
    can be re-validated without refitting" — and until this task existed there
    was no route, worker or MCP tool that could do that. The only way back was
    to pay for the entire fit again, which is the opposite of what keeping the
    directory was for. Documentation over an unreachable capability is this
    repo's signature failure and this one was in an operator-facing log line.

    Two things make a re-validation worth having rather than a convenience:

      * A FIXTURE IS NOT A LENS. Most SEMANTIC failures are a bad probe — a
        token the model would not reach, or one that appears in the prompt. The
        artifact is fine and only the question was wrong.
      * A READOUT BUG IS NOT A LENS EITHER. gemma-4-12B's 53-minute fit was
        refused because `_resolve_final_norm` missed the norm on a nested
        unified architecture, so every readout dropped the learned per-channel
        gain. The Jacobians were correct throughout.

    It still needs the GPU: SEMANTIC runs a real forward pass. It is far
    cheaper than a fit — one model load and a handful of prompts against
    minutes to hours of accumulation.

    Returns the same `validation` block a fit returns, so a caller can read one
    shape from either path.
    """
    from ..models.model import Model
    from ..services.jlens_artifact_service import JLensArtifactService
    from ..services.jlens_model_registry import load_for_readout
    from ..core.database import get_sync_db

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")
        repo_id = record.repo_id

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = load_for_readout(record, capture_device=device)

    # SAME RELEASE DISCIPLINE AS THE FIT, and for the same reason: this holds
    # the whole model on a card miLLM serves from, and the release has to happen
    # on the failure paths too. See `fit_jlens_artifact` for what a missing
    # release cost once.
    try:
        service = JLensArtifactService(settings.jlens_artifacts_dir)
        staging = service.staging_dir(repo_id)
        ref = service._ref_for(staging) if staging.is_dir() else None  # noqa: SLF001
        if ref is None:
            raise ValueError(
                f"No staged J-lens artifact for {repo_id} at {staging}. A "
                "re-validation reads work a fit already did; there is nothing "
                "staged to re-validate."
            )

        # THE LAYERS COME FROM THE ARTIFACT, never from the caller. `validate`
        # compares what it finds against `expected_layers`, so letting a caller
        # name them would let a wrong list turn a complete artifact into a
        # missing-layers failure — or, worse, pass a partial one.
        payload = service._load_payload(ref)  # noqa: SLF001 - same package concern
        if not payload:
            raise ValueError(
                f"The staged artifact at {staging} did not deserialize, so "
                "there is nothing to validate."
            )
        fitted_layers = sorted(int(k) for k in payload)

        # THROUGH `beat()`, like every other report in this module. A bare
        # meta dict REPLACES the previous one, so it erases the liveness
        # timestamp and the janitor reaps work that is still running
        # (MIS-E2E-096). Caught by test_wave7_group3 on the first run of
        # this task — the guard bites.
        self.update_state(state="PROGRESS", meta=beat({"stage": "validating"}))
        published, coverage_refusal, report, semantic_result = _validate_and_commit(
            service=service,
            ref=ref,
            loaded=loaded,
            repo_id=repo_id,
            fitted_layers=fitted_layers,
            semantic_probe=semantic_probe,
            allow_coverage_loss=allow_coverage_loss,
            allow_quality_regression=allow_quality_regression,
        )

        jlens_progress.update_row(
            self.request.id, status="completed", progress=100.0
        )
        return {
            "model_id": model_id,
            "repo_id": repo_id,
            "slug": ref.slug,
            "layers": fitted_layers,
            "revalidated": True,
            "published": published,
            "unpublished_reason": _unpublished_reason(
                published=published,
                coverage_refusal=coverage_refusal,
                semantic_result=semantic_result,
                report=report,
            ),
            "validation": _validation_block(report),
        }
    finally:
        if device == "cuda":
            from ..services.jlens_model_registry import clear_cache

            loaded = None  # noqa: F841 - the assignment IS the release
            clear_cache()
            logger.info("Released the model from GPU memory after re-validation")


def _fit_and_publish(
    self,
    loaded,
    model_id: str,
    repo_id: str,
    prompts: List[str],
    layers: Optional[List[int]],
    freeze_qk: bool,
    corpus_name: str,
    semantic_probe: Optional[Dict[str, Any]],
    allow_coverage_loss: bool,
    allow_quality_regression: bool,
    freeze_norms: bool,
    target_layer: str,
    convergence_delta: Optional[float],
) -> Dict[str, Any]:
    """The fit itself, split out so the caller can guarantee the GPU release."""
    from ..ml.jlens_fitter import DEFAULT_CONVERGENCE_DELTA, JacobianFitter
    from ..services.jlens_artifact_service import JLensArtifactService

    # None means the fitter's default. Passed explicitly rather than defaulted
    # in the signature so the value that was ACTUALLY used is the one written
    # into config.yaml below — a recipe naming a threshold the fit did not use
    # is worse than one naming none.
    fitter = JacobianFitter(
        loaded.model,
        loaded.tokenizer,
        loaded.structure,
        freeze_qk=freeze_qk,
        freeze_norms=freeze_norms,
        target_layer=target_layer,
        convergence_delta=(
            convergence_delta
            if convergence_delta is not None
            else DEFAULT_CONVERGENCE_DELTA
        ),
    )

    # MIS-E2E-096: through `beat()`. A bare meta dict REPLACES the previous
    # meta, so a plain `update_state` here erases the liveness timestamp the
    # stuck-task janitor reads — the task then looks like it has not
    # reported since whenever `beat()` last ran, and a long fit gets
    # reaped mid-flight.
    self.update_state(state="PROGRESS", meta=beat({"stage": "fitting", "prompts_seen": 0}))

    # SAY IT IS RUNNING BEFORE THE FIRST PROMPT FINISHES. The row is opened as
    # `queued` and, without this, only flips when `on_progress` first fires —
    # which is after a whole prompt's Jacobians are accumulated. On a 12B model
    # that is MINUTES per prompt (measured 2026-09-05: 3.4 min/prompt across 47
    # layers), so the longest-running job in the product was the one that
    # displayed "queued · 0%" while the GPU sat at 100%.
    #
    # jlens_acquire_tasks already does this in both of its entry points; the
    # fit task was the one that did not, which is exactly backwards — acquire
    # finishes in seconds and a fit runs for hours.
    jlens_progress.mark_running(self.request.id, progress=0.5)

    total_prompts = max(len(prompts), 1)

    # The cancellation channel. `on_progress` fires once per prompt, which is
    # the natural checkpoint: seconds on a small model, minutes on a large one,
    # and in both cases far cheaper than the alternative of not being able to
    # stop at all.
    _cancelled = jlens_progress.cancel_checker(self.request.id)

    def on_progress(progress):
        if _cancelled():
            # SCOPE AND TARGET, NOT A BARE MESSAGE. `TaskCancelled` is now an
            # alias of `OperatorCancelled`, whose signature is
            # (scope, target_id, reason, detail). Passing one argument raised
            # TypeError here — which `except TaskCancelled` below does NOT
            # catch, so it reached `owns_its_failure`, which marks the row
            # FAILED. The operator's cancellation became a crash report, on
            # the one path the module docstring cites as hardware-verified.
            raise jlens_progress.TaskCancelled(
                "jlens_task", self.request.id,
                detail=(
                    f"cancelled after {progress.prompts_seen} of "
                    f"{total_prompts} prompts"
                ),
            )
        # The SAME numbers the heartbeat carries, written where Active
        # Operations and the J-Lens panel can see them. Without this a fit is
        # visible only to the browser tab that started it.
        jlens_progress.update_row(
            self.request.id,
            status="running",
            progress=100.0 * progress.prompts_seen / total_prompts,
        )
        self.update_state(
            state="PROGRESS",
            meta=beat({
                "stage": "fitting",
                "prompts_seen": progress.prompts_seen,
                # THE DENOMINATOR TRAVELS TOO. It existed only as a local used
                # to compute the percentage and was then dropped, so a reader
                # could show "53%" but not "634 / 1200" without reconstructing
                # the total from a rounded percentage.
                "total_prompts": total_prompts,
                "last_delta": progress.last_delta,
                # The threshold the delta is racing. A delta with no target is
                # a number nobody can judge.
                "convergence_delta": fitter.convergence_delta,
                "converged": progress.converged,
            }),
        )

    try:
        result = fitter.fit(prompts, layers=layers, on_progress=on_progress)
    except jlens_progress.TaskCancelled as cancelled:
        # The row is ALREADY "cancelled" — the endpoint set it, which is how the
        # task found out. Do not write status here: `update_row` now refuses to
        # move a terminal row, and re-writing it would only add noise. Record
        # the stopping point and let the finally-block free the GPU.
        logger.info("J-lens fit %s stopped: %s", self.request.id, cancelled)
        jlens_progress.update_row(
            self.request.id, error_message=str(cancelled)
        )
        return {
            "status": "cancelled",
            "model_id": model_id,
            "detail": str(cancelled),
        }

    service = JLensArtifactService(settings.jlens_artifacts_dir)
    config_yaml = _config_yaml(
        loaded,
        result,
        freeze_qk,
        corpus_name,
        freeze_norms=freeze_norms,
        target_layer=target_layer,
    )
    #  TRAVELS IN THE CHECKPOINT, not only in config.yaml. A
    # consumer that mounts the directory reads the .pt; a community repo that
    # republishes just the .pt carries this and nothing else.
    ref = service.write_staged(
        repo_id, result.jacobians, config_yaml, n_prompts=result.prompts_seen
    )

    self.update_state(state="PROGRESS", meta=beat({"stage": "validating"}))
    published, coverage_refusal, report, semantic_result = _validate_and_commit(
        service=service,
        ref=ref,
        loaded=loaded,
        repo_id=repo_id,
        fitted_layers=sorted(result.jacobians),
        semantic_probe=semantic_probe,
        allow_coverage_loss=allow_coverage_loss,
        allow_quality_regression=allow_quality_regression,
    )
    unpublished_reason = _unpublished_reason(
        published=published,
        coverage_refusal=coverage_refusal,
        semantic_result=semantic_result,
        report=report,
    )

    jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
    return {
        "model_id": model_id,
        "repo_id": repo_id,
        "slug": ref.slug,
        "prompts_seen": result.prompts_seen,
        "converged": result.converged,
        # WHICH TEST THAT FLAG REFERS TO (MIS-E2E-080). "Converged" used to mean
        # "the running mean's own increment got small", which happens because n
        # grows, not because the estimate stabilised. A reader of an existing
        # artifact cannot tell the two apart unless the criterion is named, and
        # the word is doing evidential work in the artifact, the docs and the
        # gate decision.
        "convergence_criterion": result.convergence_criterion,
        "convergence_delta": result.convergence_delta,
        "layers": sorted(result.jacobians),
        "size_bytes": result.size_bytes(),
        "published": published,
        "unpublished_reason": unpublished_reason,
        "validation": _validation_block(report),
    }


def _validation_block(report) -> Dict[str, Any]:
    """The per-check report, in ONE shape for every path that produces one.

    Shared with `revalidate_staged_artifact` deliberately: a caller that reads
    a fit result and a re-validation result should not have to know which
    produced it, and two hand-built copies of this dict are how the two come to
    disagree about a field name.
    """
    return {
        "serviceable": report.serviceable,
        "passed": report.passed,
        "summary": report.summary(),
        "results": [
            {
                "check": r.check.value,
                "status": r.status.value,
                "detail": r.detail,
                # WHAT THE CHECK ACTUALLY SAW. Dropped until now, which made a
                # failed SEMANTIC check unactionable: it said the intermediate
                # was absent from the top-k and never said what WAS there, so
                # the only way to learn was to publish the artifact the check
                # had just refused.
                "evidence": r.evidence,
            }
            for r in report.results
        ],
    }



def _validate_and_commit(
    *,
    service,
    ref,
    loaded,
    repo_id: str,
    fitted_layers: List[int],
    semantic_probe: Optional[Dict[str, Any]],
    allow_coverage_loss: bool,
    allow_quality_regression: bool,
):
    """Validate a STAGED artifact and commit it when it is serviceable.

    EXTRACTED SO A RE-VALIDATION CAN REACH IT. This sequence lived only inside
    `_fit_and_publish`, which meant the refusal message below — "the staged fit
    is kept ... so it can be re-validated without refitting" — described a
    capability that did not exist. There was no route, task or MCP tool that
    could validate a staged artifact and commit it; the only way back was to
    pay for the whole fit again, which is precisely what keeping the staging
    directory was supposed to avoid. Observed on gemma-4-12B (2026-09-05): a
    53-minute fit was refused by SEMANTIC over a readout bug, the artifact was
    intact, and the documented recovery path was unreachable.

    Returns `(published, coverage_refusal, report, semantic_result)`.
    """
    from ..services.jlens_artifact_service import (
        ArtifactCoverageLoss,
        ArtifactQualityRegression,
    )
    # USED AT THE COMMIT SITE BELOW, and it moved here with it. The extraction
    # left this import behind in `_fit_and_publish` while the only call to it
    # came here — a NameError on the first commit of every fit, which no test
    # exercising the fit path could see because none of them reach the commit
    # line. `test_no_undefined_names` caught it.
    from ..services.jlens_validation import defer_consumer_checks

    # SEMANTIC runs HERE or nowhere. The check needs a loaded model, and this
    # task is the one place in the system that has one alongside a freshly
    # written artifact — so leaving it NOT_RUN made `serviceable` false on every
    # successful fit, and the artifact was discarded seconds after being built.
    # It still needs a FIXTURE, which cannot be invented: the intermediate must
    # be one this model would plausibly reach and must not appear in the prompt.
    # So it is the caller's to supply, and its absence fails closed with a
    # stated reason rather than publishing on an unrun check.
    semantic_result = None
    if semantic_probe:
        semantic_result = _run_semantic_check(
            service=service,
            ref=ref,
            loaded=loaded,
            probe=semantic_probe,
            fitted_layers=sorted(fitted_layers),
        )

    report = service.validate(
        ref,
        d_model=loaded.d_model,
        expected_layers=sorted(fitted_layers),
        n_vocab=loaded.n_vocab,
        semantic_result=semantic_result,
    )

    published = False
    coverage_refusal = None
    if report.serviceable:
        # `commit` requires the FULL pass, which needs a live external consumer.
        # Serviceable-but-not-passed still publishes locally, because the two
        # consumer-interop classes cannot run here and gating on them would
        # make every fit unusable. The report travels with the result so the
        # distinction is visible rather than implied.
        try:
            service.commit(
                repo_id,
                defer_consumer_checks(report),
                allow_coverage_loss=allow_coverage_loss,
                allow_quality_regression=allow_quality_regression,
            )
            published = True
        except ArtifactQualityRegression as exc:
            # SAME SHAPE AS A COVERAGE REFUSAL, and for the same reason: the fit
            # succeeded and publishing was refused to protect something the user
            # already has. The staged fit survives so it can be published
            # deliberately rather than refitted.
            logger.warning("Refused to publish %s: %s", repo_id, exc)
            coverage_refusal = str(exc)
        except ArtifactCoverageLoss as exc:
            # NOT an error the user should have to read in a log. Publishing was
            # refused to protect layers they already paid GPU time for, and the
            # staged fit is kept so they can publish it deliberately.
            logger.warning("Refused to publish %s: %s", repo_id, exc)
            coverage_refusal = str(exc)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            logger.error("Publishing %s failed: %s", repo_id, exc)
    else:
        # THE STAGED FIT SURVIVES A FAILED VALIDATION. This used to be
        # `service.discard_staged(repo_id)`, which destroyed a converged
        # 15-layer LFM2 artifact — 754 seconds of GPU time — because one
        # fixture token did not appear at one layer. The lens was fine; the
        # fixture was wrong, and there was then nothing left to re-validate
        # against, so proving that required paying for the whole fit again.
        #
        # Keeping it is safe: staging is excluded from discovery, so nothing
        # serves it, and `write_staged` clears the directory before the next
        # fit, so it cannot accumulate beyond one per model.
        logger.warning(
            "Not publishing %s: %s. The staged fit is kept at %s so it can be "
            "re-validated without refitting.",
            repo_id,
            report.summary(),
            service.staging_dir(repo_id),
        )

    return published, coverage_refusal, report, semantic_result


def _unpublished_reason(
    *,
    published: bool,
    coverage_refusal: Optional[str],
    semantic_result,
    report,
) -> Optional[str]:
    """Why nothing was published, when the cause is a missing fixture.

    Without this the result reads "semantic=not_run" and the user is left to
    infer that the fit failed, when in fact it succeeded and was discarded
    for want of one prompt.
    """
    unpublished_reason = None
    if not published:
        if coverage_refusal is not None:
            unpublished_reason = coverage_refusal
        elif semantic_result is None:
            unpublished_reason = (
                "No semantic_probe was supplied, so the SEMANTIC check could not "
                "run and the artifact was not published. Supply "
                "{prompt, expected_intermediate, layer} — an intermediate the "
                "model should reach that does NOT appear in the prompt."
            )
        elif not report.serviceable:
            unpublished_reason = (
                "The artifact failed a local validation class; see `validation`."
            )

    return unpublished_reason


def _run_semantic_check(service, ref, loaded, probe: Dict[str, Any], fitted_layers):
    """Read out the STAGED artifact and check for a known unspoken intermediate.

    Deliberately reads the file that was just written rather than the tensors
    still in memory. The in-memory ones are known good — they came straight out
    of the fitter — so checking them would confirm the fit and prove nothing
    about the artifact anyone else will load. A truncated or mis-keyed write is
    only visible on the way back in.

    A layer outside the fitted set is a fixture error, not a lens failure, and
    is reported as such: reading out at an unfitted layer has no Jacobian to
    apply and would fail for a reason that has nothing to do with the artifact.
    """
    from ..services.jlens_readout_service import JacobianTransport, ReadoutService
    from ..services.jlens_validation import (
        CheckClass,
        CheckResult,
        CheckStatus,
        check_semantic,
    )
    from ..schemas.jlens import LensTokenMessage

    prompt = str(probe.get("prompt", ""))
    expected = str(probe.get("expected_intermediate", ""))
    top_k = int(probe.get("top_k", 8))
    control_prompt = probe.get("control_prompt")
    if control_prompt is not None:
        control_prompt = str(control_prompt)
    layer = probe.get("layer")
    if layer is not None:
        # An EXPLICIT layer is honoured exactly. A caller naming a layer is
        # making a claim about that layer, and quietly scanning around it would
        # answer a question they did not ask.
        scan: Sequence[int] = [int(layer)]
    else:
        # EVERY FITTED LAYER. Two earlier defaults were both wrong for the same
        # reason: they asserted WHERE an unspoken intermediate must appear. The
        # top of the stack was wrong because the model has moved on to the
        # answer by then; "two thirds up" was wrong because that is a band
        # constant, and BR-002 forbids this project assuming a band it has not
        # measured for the model in front of it.
        #
        # Observed on hardware: the aligned LFM2 lens reads
        # ' tourism'/' located'/' geography' at L9 for an Eiffel-Tower fixture —
        # the concept field, correct and useful — and a single-layer check
        # discarded a converged 15-layer artifact over it.
        #
        # Scanning is weaker than a single layer, which is why the fixture's
        # control prompt matters; see `check_semantic`.
        scan = list(fitted_layers)

    unfitted = [c for c in scan if c not in fitted_layers]
    if unfitted:
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.FAIL,
            (
                f"probe layer(s) {unfitted} were not fitted (fitted: "
                f"{list(fitted_layers)}); there is no Jacobian to read out through"
            ),
        )

    payload = service._load_payload(ref)
    if payload is None:
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.FAIL,
            "the staged artifact did not deserialize, so it cannot be read out",
        )

    # The STAGED artifact's scales, read from the config written beside it.
    transport = JacobianTransport(
        {int(k): v for k, v in payload.items()},
        scales=service.layer_scales(ref),
    )
    readout_service = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )

    def readout(text: str, at_layers: Sequence[int], k: int) -> Dict[int, List[str]]:
        """Top-k at the LAST position, for EVERY requested layer, in one pass.

        The last position is where the next token is being formed. One stream
        call covers the whole layer list: `stream` captures residuals once and
        reads each requested layer off them, so asking per layer would re-run
        the model once per layer.
        """
        ordered = list(at_layers)
        last = None
        for message in readout_service.stream(
            text, [transport], layers=ordered, top_n=k
        ):
            if isinstance(message, LensTokenMessage):
                last = message
        if last is None:
            raise ValueError("readout produced no tokens")
        rows = last.results[0].top_tokens
        # `top_tokens` is indexed by POSITION IN THE REQUESTED LIST, not by
        # absolute layer number. Zipping against `ordered` is what keeps a
        # partial artifact from reading the wrong layer's row.
        return {layer: rows[i] for i, layer in enumerate(ordered) if i < len(rows)}

    return check_semantic(
        readout, prompt, scan, expected, top_k=top_k, control_prompt=control_prompt
    )




def _config_yaml(
    loaded,
    result,
    freeze_qk: bool,
    corpus_name: str,
    freeze_norms: bool = False,
    target_layer: str = "penultimate",
) -> str:
    """The construction recipe, sufficient to rebuild the artifact (BR-007).

    Per-layer applicability is recorded because a recipe choice can be
    INAPPLICABLE to a layer rather than merely unset: on a hybrid model
    frozen-Q/K is undefined wherever the layer does not attend, and an artifact
    must not be described as "frozen_qk" wholesale when the treatment reached a
    subset.
    """
    from ..services.jlens_readout_service import build_layer_applicability

    applicability = build_layer_applicability(
        loaded.structure, getattr(loaded.model, "config", None)
    )
    attended = [e.layer for e in applicability if e.has_attention]
    treatment = "frozen_qk" if freeze_qk else "full"

    lines = [
        f"model: {loaded.name}",
        f"d_model: {loaded.d_model}",
        f"n_layers: {loaded.n_layers}",
        f"n_vocab: {loaded.n_vocab}",
        "dtype: fp16",
        # THE RECIPE'S OWN VOCABULARY (BR-007). `JLensArtifactRecipe` declares
        # these fields and this writer emitted none of them, so the schema was a
        # contract nothing honoured and the provenance said nothing about how the
        # lens was built.
        #
        # THE TARGET IS THE CALLER'S CHOICE AND MUST BE REPORTED AS RUN. This
        # line was the literal string "final" — correct when the fitter always
        # ran to the last block, and a LIE the moment the target became
        # selectable. The parameter was added to this function's signature and
        # then never read, so a 15-of-16-layer penultimate fit published a
        # recipe claiming it targeted the final block. Nothing objected: the
        # value was threaded the whole way here and dropped on the last line.
        f"target_layer: {target_layer}",
        # THE PAPER'S DEFINITION, recorded as it was run: an expectation over
        # SOURCE positions of the summed effect on all subsequent target
        # positions. The previous fitter recorded `self_only_isolated` because
        # that is what it did — one source position, length-1 sub-network.
        "target_position_scope: all_subsequent",
        "source_position_aggregation: mean_over_all_positions",
        "differentiation_mode: reverse",
        "aggregation: mean",
        f"seq_len: {getattr(result, 'mean_seq_len', 0.0):.1f}",
        # PER-LAYER, NOT WHOLESALE. Describing a hybrid model's lens as
        # "frozen_qk" when the treatment reached 6 of 16 layers is the exact
        # overstatement this file's own docstring warns about. The requested
        # treatment and where it actually applied are separate facts.
        f"attention_gradients_requested: {treatment}",
        f"norm_statistics: {'frozen' if freeze_norms else 'differentiated'}",
        f"attention_gradients_applied_to_layers: {attended}",
        # THE LAYERS THIS ARTIFACT ACTUALLY COVERS, stated once and cheaply.
        # Everything else that needs them had to deserialise the whole tensor
        # file — 276 MB to answer "which layers?" — which is why the artifact
        # listing never carried the fact and a partial fit looked identical to
        # a full one right up until the readout refused.
        f"fitted_layers: {sorted(result.jacobians)}",
        # LAYERS WHERE THE LENS IS THE LOGIT LENS, exactly. The last decoder
        # layer has no blocks after it, so its sub-network is the identity and
        # J = I by construction. A Diff there is empty because the two lenses
        # ARE the same lens — not because they happen to agree — and an empty
        # top row read without that context looks like a finding.
        f"degenerate_layers: {result.degenerate_layers}",
        f"corpus: {corpus_name}",
        f"n_prompts: {result.prompts_seen}",
        f"converged: {str(result.converged).lower()}",
        f"convergence_delta: {result.convergence_delta}",
        # THE SCALE, WITHOUT WHICH THE ARTIFACT IS WRONG. `_to_storage_dtype`
        # divides each matrix down so the fp16 cast cannot saturate, and its
        # docstring has always said the factor is recorded here — it was not.
        # Ranked readouts never noticed, because the model's final norm divides
        # a positive scalar straight back out. Everything that does NOT
        # normalise did: probe scores and intervention magnitudes were off by
        # an unrecorded per-layer factor, so they were not comparable across
        # layers, and an external consumer multiplying by W_U got the wrong
        # magnitudes — the exact case that docstring names.
        "layer_scales:",
    ]
    for layer in sorted(result.scales):
        lines.append(f"  {layer}: {result.scales[layer]!r}")

    # HOW POSITIONALLY STABLE the lens is, over the whole corpus. Both figures,
    # because the mean says what is typical and the max says how bad it gets —
    # and a lens is judged on the second. This used to be a single number taken
    # from whichever prompt happened to be last, and it used to be published
    # under a name belonging to a quantity nothing computes (MIS-E2E-081).
    # MIS-E2E-081: named for what it measures. These keys used to be
    # `linearisation_residual_{mean,max}`, which is a different quantity — one
    # this fitter never computes in production. The value is the spread of the
    # Jacobian's rows across SOURCE POSITIONS, normalised by |J|.mean().
    if result.position_spread_mean:
        lines.append("source_position_spread_mean:")
        for layer in sorted(result.position_spread_mean):
            lines.append(f"  {layer}: {result.position_spread_mean[layer]:.6g}")
    if result.position_spread_max:
        lines.append("source_position_spread_max:")
        for layer in sorted(result.position_spread_max):
            lines.append(f"  {layer}: {result.position_spread_max[layer]:.6g}")

    lines += [
        "per_layer_applicability:",
    ]
    for entry in applicability:
        lines.append(f"  - layer: {entry.layer}")
        lines.append(f"    has_attention: {str(entry.has_attention).lower()}")
        # Absent, never false: inapplicable is not the same as "checked and no".
        if entry.frozen_qk_applicable is None:
            lines.append("    frozen_qk_applicable: null  # INAPPLICABLE here")
        else:
            lines.append(
                f"    frozen_qk_applicable: {str(entry.frozen_qk_applicable).lower()}"
            )
    return "\n".join(lines) + "\n"
