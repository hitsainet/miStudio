"""
J-space readout endpoints.

Emits the upstream lens wire format verbatim (BR-029, PADR IDL-45) so a
miStudio stream and a Neuronpedia stream are interchangeable at the client and
the readout panel is driven by either with no adaptation layer.

The LOGIT lens needs no artifact and is the default (BR-005). Requesting
JACOBIAN_LENS without an artifact is refused at the schema, not silently served
as logit data under a Jacobian label — that would breach rung discipline
(BR-019).
"""

import logging
import os
from typing import Any, Dict, List, Literal, Optional, Sequence

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.config import settings
from ....core.deps import get_db
from ....schemas.jlens import (
    LensDoneMessage,
    LensMetaMessage,
    LensTokenMessage,
    ProbeRequest,
    ProbeScore,
    ReadoutRequest,
)
from ....services.jlens_artifact_service import (
    ArtifactConflict,
    ArtifactNotValidated,
    JLensArtifactService,
)
from ....services.jlens_model_registry import ModelNotAvailable, load_for_readout

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jlens", tags=["jlens"])

#: The per-prompt character bound, shared by `prompt` and every entry of
#: `prompts`. One bound, one place: they are the same thing to the worker, which
#: runs a forward pass per arm per trial over whichever it is given.
MAX_PROMPT_CHARS = 8000

#: How many layers one intervention may hook. Each is a hook on every forward
#: pass of every arm of every trial; the intervened arm additionally perturbs
#: the output of the hook before it, so this is not merely a cost bound.
MAX_INTERVENED_LAYERS = 64


def _service() -> JLensArtifactService:
    return JLensArtifactService(settings.jlens_artifacts_dir)


def _jacobian_transport(loaded: Any, artifact_id: Optional[str] = None):
    """Build a JacobianTransport for `loaded`, WITH its recorded scales.

    One helper rather than five construction sites. The scale has to travel
    with the matrices — an unscaled transport reads out ranked results that
    look perfect and probe/intervention magnitudes that are silently wrong by a
    per-layer factor — and five call sites is five chances to forget it.
    """
    from ....services.jlens_readout_service import JacobianTransport

    service = _service()
    report = _validated_report(loaded, artifact_id)
    jacobians = service.load_for_readout(loaded.name, report=report)
    ref = service.find(loaded.name)
    scales = service.layer_scales(ref) if ref is not None else {}
    return JacobianTransport(jacobians, scales=scales)


class ArtifactSummary(BaseModel):
    """One artifact as it exists ON DISK — presence, not validity.

    `validated` is deliberately absent from this shape: an artifact's validity
    is the outcome of running the suite, not a property of the file, and a
    field here would be read as a verdict the listing never computed.
    """

    slug: str
    directory: str
    lens_file: str
    size_bytes: int
    has_config: bool
    #: Layers this artifact covers. EMPTY MEANS UNKNOWN, not "none" — an
    #: artifact whose config cannot be read still holds whatever it holds, and
    #: rendering unknown as zero coverage would be a claim the listing did not
    #: check. A partial fit was previously indistinguishable from a full one
    #: here, which is how a 9-of-16-layer lens reached a readout asking for 16.
    layers: List[int] = []
    #: Layers where J is the identity — the lens there IS the logit lens. A
    #: Diff at such a layer is empty by construction, and saying so is the
    #: difference between "no signal" and "the same lens twice".
    degenerate_layers: List[int] = []
    #: Which block the Jacobian was taken TO. With a `penultimate` target a
    #: COMPLETE fit covers 0..N-2, so a client comparing coverage against the
    #: model's layer count would render a full artifact as incomplete.
    target_layer: Optional[str] = None
    #: How many intervention results are recorded beside this lens and still
    #: describe its current weights. A COUNT, not the records: a listing should
    #: not carry every experiment, and zero is a real answer meaning "nothing
    #: has been demonstrated about this lens yet" — not "it does not work".
    intervention_records: int = 0


class CheckOutcome(BaseModel):
    check: str
    status: str
    detail: str
    evidence: Dict[str, Any] = {}


class ValidationResponse(BaseModel):
    """The suite's verdict, with every class reported individually.

    `passed` is FAIL-CLOSED: a class that could not run is not a pass. The
    three live checks need a loaded model or a running consumer, so a
    validation performed from here reports them NOT_RUN and `passed` is False
    — which is the honest answer, not a defect.
    """

    slug: str
    passed: bool
    summary: str
    results: List[CheckOutcome]


@router.get(
    "/artifacts",
    response_model=List[ArtifactSummary],
    summary="J-lens artifacts present in the mounted registry",
)
async def list_artifacts() -> List[ArtifactSummary]:
    """List conformant artifact directories.

    Staging directories are excluded — an artifact still being written is not
    an artifact, and the whole point of staging is that it is invisible until
    it commits.
    """
    service = _service()
    return [
        ArtifactSummary(
            slug=ref.slug,
            directory=str(ref.directory),
            lens_file=ref.lens_path.name,
            size_bytes=ref.size_bytes,
            has_config=ref.config_path is not None,
            layers=service.fitted_layers(ref),
            degenerate_layers=service.degenerate_layers(ref),
            target_layer=service.target_layer(ref),
            intervention_records=len(service.intervention_results(ref)),
        )
        for ref in service.list_artifacts()
    ]


@router.post(
    "/artifacts/{slug}/validate",
    response_model=ValidationResponse,
    summary="Run the artifact validation suite (BR-030)",
)
async def validate_artifact(
    slug: str,
    d_model: int,
    n_layers: int,
    n_vocab: int,
) -> ValidationResponse:
    """Run every check that does not require a loaded model or a live consumer.

    The model's dimensions are REQUIRED parameters rather than looked up,
    because the envelope bound must come from the model the artifact was fitted
    for. Defaulting them would produce a bound derived from nothing, and a
    wrong envelope bound passes on one model while missing a real
    materialisation on another.
    """
    service = _service()
    ref = next((a for a in service.list_artifacts() if a.slug == slug), None)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conformant J-lens artifact directory named {slug!r}",
        )

    # Same correction as the readout path: the artifact declares which layers it
    # holds. `range(n_layers)` failed every PARTIAL fit with "missing layers
    # [0..23]" — a fit shape the API, the MCP tool and the UI all offer. The
    # envelope bound is derived from the count actually present, so a partial
    # artifact is measured against what it contains rather than a full stack it
    # never claimed to be.
    payload = service._load_payload(ref)  # noqa: SLF001 - same package concern
    present = sorted(int(k) for k in payload) if isinstance(payload, dict) else []

    report = service.validate(
        ref,
        d_model=d_model,
        expected_layers=present or range(n_layers),
        n_vocab=n_vocab,
    )
    return ValidationResponse(
        slug=slug,
        passed=report.passed,
        summary=report.summary(),
        results=[
            CheckOutcome(
                check=r.check.value,
                status=r.status.value,
                detail=r.detail,
                evidence=r.evidence,
            )
            for r in report.results
        ],
    )


class InterventionRecordsResponse(BaseModel):
    """Interventions that were run against this lens, and what they measured.

    Separate from the artifact listing on purpose: the listing answers "what is
    on disk", and a measurement of behaviour is not a property of a file. It is
    also the shape a serving runtime wants — a list of recipes it could apply,
    each with the evidence for applying it.
    """

    slug: str
    lens_sha256: str
    records: List[Dict[str, Any]]


@router.get(
    "/artifacts/{slug}/interventions",
    response_model=InterventionRecordsResponse,
    summary="Intervention results recorded beside this lens",
)
async def intervention_records(slug: str) -> InterventionRecordsResponse:
    """Records that describe THIS lens file; others are dropped.

    Each record carries a `steering_recipe` — primitive, direction, layers,
    positions, strength and hook target — beside the rates that justify it. A
    consumer can therefore pick a direction whose intervened rate separated from
    its matched control and apply exactly what was tested, rather than inferring
    a recipe from a score.

    AN EMPTY LIST IS AN ANSWER. It means no intervention has been run against
    this lens yet, which is different from one that was run and moved nothing —
    the latter appears here with overlapping intervals.
    """
    service = _service()
    ref = next((a for a in service.list_artifacts() if a.slug == slug), None)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conformant J-lens artifact directory named {slug!r}",
        )
    return InterventionRecordsResponse(
        slug=slug,
        lens_sha256=service._lens_digest(ref.lens_path),  # noqa: SLF001
        records=service.intervention_results(ref),
    )


class RestoreResponse(BaseModel):
    """What was promoted and what was archived in its place.

    Both recipes travel back, because "restored" without saying WHAT it
    replaced is unverifiable — the caller cannot tell whether the swap did what
    they intended, and this operation exists precisely because the wrong lens
    was serving.
    """

    slug: str
    restored: Dict[str, Any]
    displaced: Dict[str, Any]


@router.post(
    "/artifacts/{slug}/restore-superseded",
    response_model=RestoreResponse,
    summary="Promote the archived artifact back into service",
)
async def restore_superseded(slug: str) -> RestoreResponse:
    """Swap `<slug>.superseded` back into `<slug>`.

    A SWAP, so this is its own undo and nothing is deleted. Call it twice and
    you are back where you started.

    Publishing is last-writer-wins, and "last" means finished last, not best. A
    stale 400-prompt fit that never converged once published over a 1097-prompt
    fit that did; `allow_quality_regression` now refuses that, but an artifact
    already displaced could only be recovered by a shell rename inside the pod
    — no audit trail, no digest check, and a typo away from destroying the
    archive. This is that operation, supported.
    """
    from ....services.jlens_artifact_service import ArtifactNotValidated

    service = _service()
    try:
        outcome = service.restore_superseded(slug)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ArtifactConflict as exc:
        # 409 FOR THE SAME REASON THE CLAUSE BELOW IS ONE: the request is
        # well-formed and the conflict is with the state on disk. As a bare
        # RuntimeError it matched neither handler and left as an opaque 500 —
        # so the actionable text the refusal was written to deliver ("inspect
        # it and move it aside by hand") never reached the operator retrying
        # the recovery.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except ArtifactNotValidated as exc:
        # 409, not 400: the request is well-formed and the CONFLICT is with the
        # state on disk. A 400 would read as "you asked wrongly".
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    logger.info("Restored the archived J-lens artifact for %s", slug)
    return RestoreResponse(**outcome)



class CancelResponse(BaseModel):
    task_id: str
    cancelled: bool
    was_running: bool
    detail: str


@router.post(
    "/tasks/{task_id}/cancel",
    response_model=CancelResponse,
    summary="Cancel a running or queued J-space task",
)
async def cancel_task(task_id: str) -> CancelResponse:
    """Stop a J-space task. Cooperative, because it has to be.

    THE OBVIOUS IMPLEMENTATION DOES NOT WORK HERE, and every other cancel in
    this codebase uses it. `celery_app.control.revoke(terminate=True)` only
    signals a POOL CHILD, and the GPU worker runs `--pool=solo` because CUDA and
    fork do not mix — there is no child to signal. Worse, a solo worker busy in
    a task is not reading the control queue, so the revoke is never delivered:
    it returns cleanly, changes nothing, and the worker does not even appear in
    `inspect()`. Verified on hardware 2026-09-05 against a running gemma-4-12B
    fit, which then needed a SIGKILL on the worker PID.

    So this writes "cancelled" to the row and the task notices at its next
    checkpoint — one prompt for a fit. The revoke below is still issued, for the
    ONE case it does handle: a task that has not started yet never will.
    """
    from ....core.celery_app import celery_app
    from ....core.database import get_sync_db
    from ....models.task_queue import TaskQueue
    from ....workers import jlens_progress

    with get_sync_db() as db:
        row = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No task_queue row for task {task_id!r}",
            )
        prior = row.status

    if prior in ("completed", "failed", "cancelled"):
        return CancelResponse(
            task_id=task_id,
            cancelled=False,
            was_running=False,
            detail=f"Task is already {prior}; nothing to cancel.",
        )

    # Stops a task that has NOT started. Harmless and useless for one that has,
    # which is exactly why it is not the whole implementation.
    try:
        celery_app.control.revoke(task_id)
    except Exception:  # noqa: BLE001 - the row is the channel that matters
        logger.warning("revoke() failed for %s; the row still carries the request", task_id)

    ok = jlens_progress.request_cancel(task_id)
    return CancelResponse(
        task_id=task_id,
        cancelled=ok,
        was_running=(prior == "running"),
        detail=(
            "Cancellation requested. A running fit stops at its next prompt "
            "boundary, which is minutes on a large model."
            if prior == "running"
            else "Task had not started; it will not run."
        ),
    )


class FitRequest(BaseModel):
    """Start a fit. The corpus is named or supplied, never chosen server-side.

    The corpus is part of the recipe (BR-007), so it is the caller's choice and
    is recorded in `config.yaml`. A server-chosen DEFAULT corpus would produce
    artifacts whose provenance says nothing — which is why there is no default
    here and exactly one of `prompts` or `dataset_id` is required.

    NAMING A DATASET IS BETTER PROVENANCE THAN INLINING ITS TEXT, not worse.
    `dataset_id + n_prompts + max_chars + sample_seed` re-derives the exact
    corpus; 1200 opaque strings in a request body cannot be re-derived from the
    artifact at all. It is also the only way to reach a reference-sized corpus
    over MCP or HTTP: 1200 documents at the reference's 2000-char cap is 2.2 MB
    and the body cap is 1024 KB, so the inline path 413s on the recipe it is
    meant to reproduce.
    """

    model_id: str
    #: Supply the corpus inline, OR name a registered dataset below. Exactly one.
    prompts: Optional[List[str]] = None
    #: A registered miStudio dataset to sample the fitting corpus from.
    dataset_id: Optional[str] = None
    #: How many documents to sample when `dataset_id` is used.
    n_prompts: int = Field(1200, ge=100, le=20000)
    #: Truncate each document to this many characters. The reference recipe
    #: passes max_chars 2000 with max_seq_len 128, so the model sees ~128 tokens
    #: regardless; 550 chars is about that and keeps the sample honest about
    #: what the model actually read.
    max_chars: int = Field(550, ge=50, le=8000)
    #: Documents shorter than this are skipped — a near-empty prompt linearises
    #: around almost no context and drags the mean J toward the null input.
    min_chars: int = Field(400, ge=0, le=8000)
    #: Recorded in the recipe so the sample is reproducible.
    sample_seed: int = 0
    layers: Optional[List[int]] = None
    #: FULL BACKWARD IS THE STANDARD RECIPE (D4); frozen Q/K is an ABLATION.
    #: Defaulting this to True made the ablation the only thing an API or
    #: MCP caller could get without naming the flag, and the recipe written
    #: beside the artifact then honestly recorded `frozen_qk` — an aligned
    #: request producing an unaligned lens, with the provenance telling the
    #: truth about it and nobody reading.
    freeze_qk: bool = False
    corpus_name: str = "unspecified"
    #: Fixture for the SEMANTIC validation class: {prompt, expected_intermediate,
    #: layer?, top_k?}. Optional, and its absence FAILS CLOSED — the artifact is
    #: fitted and then discarded unpublished, because publishing on a check that
    #: never ran is the failure this suite exists to prevent. The intermediate
    #: must not appear in the prompt: a token already present is recovered by an
    #: artifact encoding nothing at all.
    semantic_probe: Optional[Dict[str, Any]] = None
    #: Publish even though the existing artifact covers layers this fit does
    #: not. Off by default: a refit is not automatically an upgrade, and a
    #: 16-layer lens was destroyed by a 9-layer one with no warning at all.
    allow_coverage_loss: bool = False
    #: Publish even when this fit is WEAKER evidence than the artifact it
    #: replaces — fewer prompts, or not converged where the incumbent was.
    #: Default False because publishing is otherwise last-writer-wins, and a
    #: stale job that finishes last is still last.
    allow_quality_regression: bool = False
    #: Relative Frobenius change in the accumulated J below which the fit is
    #: considered settled, sustained across consecutive shards.
    #:
    #: EXPOSED BECAUSE BUYING CONVERGENCE WITH GPU HOURS IS NOT THE ONLY
    #: OPTION. Both 400-prompt fits reported converged=false at the built-in
    #: 1e-3, and the only lever was more corpus. A caller who wants a looser
    #: criterion should be able to ask for one and have it RECORDED, rather
    #: than have the artifact silently describe a threshold it was not fitted
    #: against.
    #:
    #: None keeps the fitter's own default. Bounded below at 0 because a
    #: non-positive delta can never be met and would fit the entire corpus
    #: while reporting "not converged" — indistinguishable from a genuine
    #: failure to settle.
    convergence_delta: Optional[float] = Field(None, gt=0)
    #: Freeze normalisation statistics as well. Off by default: freezing makes
    #: the map exactly affine, which is convenient and is NOT what the paper
    #: computes — its J is a local linearisation whose departure is reported.
    freeze_norms: bool = False
    #: Which block's output the Jacobian runs TO. Penultimate by default per
    #: BRD A.2: the last block is specialised for next-token calibration and
    #: adds noise. Layers ABOVE the target are refused — their gradient to it is
    #: zero by causality and a zero lens reads out as confident uniform noise.
    target_layer: Literal["final", "penultimate"] = "penultimate"

    @model_validator(mode="after")
    def _exactly_one_corpus_source(self):
        """Neither is a server-chosen corpus; both is an ambiguous recipe."""
        if bool(self.prompts) == bool(self.dataset_id):
            raise ValueError(
                "supply exactly one of `prompts` (inline corpus) or "
                "`dataset_id` (name a registered dataset). Supplying neither "
                "would make the server pick a corpus, which BR-007 forbids; "
                "supplying both leaves the recipe ambiguous about which one "
                "the artifact was fitted on."
            )
        return self


class FitAccepted(BaseModel):
    task_id: str
    model_id: str
    queue: str


async def _sample_dataset_prompts(db: AsyncSession, request: "FitRequest"):
    """Draw a reproducible fitting corpus from a registered dataset.

    Returns (prompts, corpus_name). The name records everything needed to
    re-derive the sample, which inline prompts cannot express.

    Sampling is a DETERMINISTIC STRIDE from a seeded offset, not `random`:
    the artifact records the seed, so the same request must reproduce the same
    corpus on any machine and after any library upgrade.
    """
    from datasets import load_from_disk

    from ....models.dataset import Dataset

    row = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
    dataset = row.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No dataset with id {request.dataset_id!r}",
        )
    if not dataset.raw_path or not os.path.isdir(dataset.raw_path):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Dataset {dataset.name!r} has no readable raw_path "
                f"({dataset.raw_path!r}); it cannot be sampled for a fit."
            ),
        )

    try:
        ds = load_from_disk(dataset.raw_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Could not open dataset {dataset.name!r}: {exc}",
        ) from exc
    if hasattr(ds, "keys"):
        ds = ds[list(ds.keys())[0]]
    field = "text" if "text" in ds.column_names else ds.column_names[0]

    total = len(ds)
    stride = max(1, total // max(request.n_prompts, 1))
    prompts: List[str] = []
    idx = request.sample_seed % max(stride, 1)
    while idx < total and len(prompts) < request.n_prompts:
        text_value = (ds[idx].get(field) or "").strip()
        if len(text_value) >= request.min_chars:
            prompts.append(text_value[: request.max_chars])
        idx += stride

    # REFUSE A SHORT SAMPLE rather than fit on it. The fitter's own floor is
    # 100, but a caller who asked for 1200 and would silently get 300 has been
    # given a different experiment under the same name.
    if len(prompts) < request.n_prompts:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Asked for {request.n_prompts} prompts of at least "
                f"{request.min_chars} characters but {dataset.name!r} yielded "
                f"only {len(prompts)} at stride {stride} over {total} rows. "
                f"Lower n_prompts or min_chars rather than fitting on a "
                f"smaller corpus than the recipe claims."
            ),
        )

    name = (
        f"{dataset.name}-{len(prompts)}docs"
        f"-{request.max_chars}chars-seed{request.sample_seed}"
    )
    return prompts, name


@router.post(
    "/fit",
    response_model=FitAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Fit a J-lens artifact for a model",
)
async def fit(request: FitRequest, db: AsyncSession = Depends(get_db)) -> FitAccepted:
    """Queue a fit. GPU-bound and long-running, so it never runs inline.

    The prompt floor (Appendix A.2) is enforced by the fitter itself and
    REFUSED rather than warned about: an under-fitted lens is indistinguishable
    from a fitted one by inspection.
    """
    from ....models.model import Model
    from ....workers.jlens_fit_tasks import fit_jlens_artifact

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    prompts = request.prompts
    corpus_name = request.corpus_name
    if request.dataset_id:
        prompts, resolved_name = await _sample_dataset_prompts(db, request)
        # Only fill a corpus name the caller did not give. Overwriting theirs
        # would put a name in the recipe they never chose.
        if corpus_name in (None, "", "unspecified"):
            corpus_name = resolved_name

    from ....workers import jlens_progress

    task = fit_jlens_artifact.delay(
        model_id=request.model_id,
        prompts=prompts,
        layers=request.layers,
        freeze_qk=request.freeze_qk,
        corpus_name=corpus_name,
        semantic_probe=request.semantic_probe,
        allow_coverage_loss=request.allow_coverage_loss,
        allow_quality_regression=request.allow_quality_regression,
        freeze_norms=request.freeze_norms,
        target_layer=request.target_layer,
        convergence_delta=request.convergence_delta,
    )
    # VISIBLE WHILE IT RUNS. A 45-minute fit used to burn the GPU with nothing
    # in the product saying so — the panel's fit card only knew about a fit the
    # same browser tab had started.
    jlens_progress.open_row(jlens_progress.FIT, request.model_id, task.id)
    return FitAccepted(task_id=task.id, model_id=request.model_id, queue="extraction")


class RevalidateRequest(BaseModel):
    """Re-run validation against an already-fitted STAGED artifact."""

    model_id: str
    #: The fixture the SEMANTIC class needs. Usually the whole point of a
    #: re-validation: the previous refusal was a bad probe, not a bad lens.
    semantic_probe: Optional[Dict[str, Any]] = None
    allow_coverage_loss: bool = False
    allow_quality_regression: bool = False


class RevalidateAccepted(BaseModel):
    task_id: str
    model_id: str
    slug: str


@router.post(
    "/artifacts/revalidate",
    response_model=RevalidateAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-validate a staged J-lens artifact and commit it if it passes",
)
async def revalidate_staged(
    request: RevalidateRequest, db: AsyncSession = Depends(get_db)
) -> RevalidateAccepted:
    """Validate work a fit already did, without paying for the fit again.

    THE ROUTE THE REFUSAL MESSAGE ALREADY ASSUMED. When validation refuses, the
    worker keeps the staging directory and logs "so it can be re-validated
    without refitting" — and nothing could. `/artifacts/{slug}/validate` sees
    only COMMITTED artifacts and cannot run SEMANTIC (no loaded model);
    `/publish` is a HuggingFace upload and explicitly refuses a staged
    artifact. The only path back was a full refit.

    Refuses HERE what is knowable here — a missing model and an empty staging
    directory both cost a slot on the single-GPU queue to discover in the
    worker, possibly behind a running fit.
    """
    from ....models.model import Model
    from ....services.jlens_artifact_service import JLensArtifactService
    from ....workers import jlens_progress
    from ....workers.jlens_fit_tasks import revalidate_staged_artifact

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )
    repo_of_model = getattr(record, "repo_id", None)
    if not repo_of_model:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model {request.model_id!r} has no repo_id",
        )

    service = JLensArtifactService(settings.jlens_artifacts_dir)
    staging = service.staging_dir(repo_of_model)
    ref = service._ref_for(staging) if staging.is_dir() else None  # noqa: SLF001
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"No staged J-lens artifact for {repo_of_model}. A "
                "re-validation reads work a fit already did; fit one first."
            ),
        )

    task = revalidate_staged_artifact.delay(
        model_id=request.model_id,
        semantic_probe=request.semantic_probe,
        allow_coverage_loss=request.allow_coverage_loss,
        allow_quality_regression=request.allow_quality_regression,
    )
    # VISIBLE WHILE IT RUNS, like every other J-space task. It takes the GPU and
    # can queue behind a fit, so a job with no row looks like nothing happened.
    jlens_progress.open_row(
        jlens_progress.REVALIDATE, request.model_id, task.id
    )
    return RevalidateAccepted(
        task_id=task.id, model_id=request.model_id, slug=ref.slug
    )


class BandReportResponse(BaseModel):
    """A model's measured profile and the boundaries it does or does not support.

    `boundaries` is nullable and a null is the HONEST answer, not a missing
    value: bands are drawn only from a report computed for this model, and
    there is no default anywhere in the product (BR-002). The client renders
    nothing when this is null.
    """

    model_id: str
    has_bands: bool
    boundaries: Optional[Dict[str, int]]
    derivation: str
    control_seed: Optional[int]
    profiles: List[Dict[str, Any]]


class GateResponse(BaseModel):
    model_id: str
    decision: str
    rationale: str
    blocking: bool
    has_bands: bool


class BandReportRequest(BaseModel):
    """Compute a band report for one model, from ITS OWN measured profile.

    `control_seed` is REQUIRED and not defaulted, all the way from here down:
    the autocorrelation null is drawn from it, and a report whose control cannot
    be reproduced is not evidence.

    There is no `boundaries` field to supply and never will be. Bands come from
    the model's own kurtosis profile or they do not exist for it — BR-002
    requires that porting another model's boundaries be impossible by
    construction, not merely discouraged.
    """

    model_id: str
    prompts: List[str] = Field(..., min_length=1)
    control_seed: int
    layers: Optional[List[int]] = None
    #: Use the fitted lens dictionary when one exists. Effective dimensionality
    #: is a property of that dictionary; for the logit lens it is the identity,
    #: whose effective dimensionality says nothing and is recorded ABSENT.
    use_artifact: bool = True


class BandTaskAccepted(BaseModel):
    task_id: str
    model_id: str
    queue: str = "extraction"


@router.post(
    "/band-report",
    response_model=BandTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Compute this model's band report (BR-002)",
)
async def compute_band_report(
    request: BandReportRequest, db: AsyncSession = Depends(get_db)
) -> BandTaskAccepted:
    """Queue a band-report computation.

    THE ONLY THING IN THE PRODUCT THAT CAN MAKE BANDS APPEAR. Until this runs
    for a given model, every band surface renders nothing and
    `classify_behaviour` returns UNKNOWN — which is the honest answer, not a
    defect, because the published boundaries were measured on one specific
    model and do not transfer.
    """
    from ....models.model import Model
    from ....workers.jlens_band_tasks import compute_band_report_task

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    from ....workers import jlens_progress

    task = compute_band_report_task.delay(
        model_id=request.model_id,
        prompts=request.prompts,
        control_seed=request.control_seed,
        layers=request.layers,
        use_artifact=request.use_artifact,
    )
    jlens_progress.open_row(
        jlens_progress.BAND_REPORT, request.model_id, task.id
    )
    return BandTaskAccepted(task_id=task.id, model_id=request.model_id)


class GateRequest(BaseModel):
    """Record the Phase-0 decision (BR-003).

    THE INPUTS ARE FINDINGS, NOT SCORES. `claim_set_replicated` is the question
    BR-003 actually asks and is supplied by the analysis; there is deliberately
    no numeric criterion anywhere on this path. A threshold on any single metric
    would become the definition of the gate, and the one most likely to be
    reached for is next-token agreement, which BR-004 forbids scoring on.
    """

    model_id: str
    claim_set_replicated: bool
    larger_scale_indicated: bool = False
    #: Mandatory. A recorded decision without its reasoning is not a record.
    rationale: str = Field(..., min_length=1)
    replication_report_id: Optional[str] = None


@router.post(
    "/gate",
    response_model=BandTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Record the Phase-0 GO / NO-GO decision (BR-003)",
)
async def record_gate(
    request: GateRequest, db: AsyncSession = Depends(get_db)
) -> BandTaskAccepted:
    """Queue a gate record. REFUSES without a band report to weigh."""
    from ....models.model import Model
    from ....workers.jlens_band_tasks import record_gate_task

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    task = record_gate_task.delay(
        model_id=request.model_id,
        claim_set_replicated=request.claim_set_replicated,
        larger_scale_indicated=request.larger_scale_indicated,
        rationale=request.rationale,
        replication_report_id=request.replication_report_id,
    )
    return BandTaskAccepted(task_id=task.id, model_id=request.model_id)


@router.get(
    "/artifacts/{slug}/band-report",
    response_model=Optional[BandReportResponse],
    summary="This model's own sensory / workspace / motor boundaries",
)
async def band_report(slug: str) -> Optional[BandReportResponse]:
    """Return the stored band report, or NULL when there is none.

    A null body is the honest answer and the client draws no bands (BR-002).
    It is not a 404: the artifact exists, it simply has no report yet, and
    those are different facts. Boundaries measured on another model are never
    substituted — there is no default anywhere in the product.
    """
    from ....services.jlens_band_service import load_band_report

    ref = next((a for a in _service().list_artifacts() if a.slug == slug), None)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conformant J-lens artifact directory named {slug!r}",
        )

    stored = load_band_report(ref.directory)
    if stored is None:
        return None

    return BandReportResponse(
        model_id=stored.get("model_id", slug),
        has_bands=stored.get("boundaries") is not None,
        boundaries=stored.get("boundaries"),
        derivation=stored.get("derivation", ""),
        control_seed=stored.get("control_seed"),
        profiles=stored.get("profiles", []),
    )


@router.get(
    "/artifacts/{slug}/gate",
    response_model=Optional[GateResponse],
    summary="The recorded Phase-0 GO / NO-GO decision",
)
async def gate(slug: str) -> Optional[GateResponse]:
    """Return the recorded gate decision, or NULL when none has been made.

    NO_GO reads back exactly like GO and is a complete, publishable outcome
    (BR-003) — not an error state and not an absence.
    """
    from ....services.jlens_band_service import load_gate

    ref = next((a for a in _service().list_artifacts() if a.slug == slug), None)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conformant J-lens artifact directory named {slug!r}",
        )

    stored = load_gate(ref.directory)
    if stored is None:
        return None
    return GateResponse(
        model_id=stored.get("model_id", slug),
        decision=stored["decision"],
        rationale=stored.get("rationale", ""),
        blocking=stored.get("blocking", True),
        has_bands=stored.get("has_bands", False),
    )


class ReadoutResponse(BaseModel):
    """Non-streaming envelope: the meta message plus its token messages.

    CONTAINS a meta message rather than INHERITING one. Subclassing
    LensMetaMessage carried its `kind: "meta"` discriminator onto the envelope,
    so the response announced itself as a meta message while also carrying a
    `tokens` array — a client dispatching on `kind` would mis-handle it, in the
    one format this feature exists to conform to.

    A streaming transport (SSE/WebSocket) can be added later without changing
    the message shapes, which is the point of adopting the upstream format.
    """

    meta: LensMetaMessage
    tokens: List[LensTokenMessage]


# Validation is cached per (slug, mtime, size). Without this, EVERY Jacobian
# readout re-runs the suite — including the SEMANTIC check, which is itself a
# full readout — so each request paid for two readouts plus a revalidation. The
# key includes mtime and size so a replaced artifact is revalidated rather than
# served on a stale verdict.
_VALIDATION_CACHE: Dict[tuple, Any] = {}


def _validated_report(loaded: Any, artifact_id: Optional[str]):
    """Locate and validate the artifact for THIS model, or refuse.

    WEIGHT IDENTITY IS PART OF THE CHECK (BR-031, FPRD §3.4). The artifact slug
    is derived from the repo id, so an artifact fitted for a base model has a
    different slug from its instruction-tuned variant. Accepting an
    `artifact_id` that does not match this model's own slug would serve a lens
    fitted for DIFFERENT WEIGHTS — which produces a complete, plausible readout
    and is undetectable downstream. Checking the model NAME alone is what makes
    that mistake easy, so the comparison is on the slug the fit would produce.

    The SEMANTIC check runs here because it needs the loaded model; the two
    consumer-interop classes cannot run without a live external consumer and are
    reported NOT_RUN, which is why serving is gated on `serviceable` rather than
    `passed`.
    """
    from ....services.jlens_artifact_service import slug_for

    service = _service()
    expected = slug_for(loaded.name)
    if artifact_id and artifact_id != expected:
        raise ArtifactNotValidated(
            f"artifact {artifact_id!r} was not fitted for {loaded.name} "
            f"(expected slug {expected!r}). A lens fitted for different weights "
            "produces a complete, plausible readout that is wrong."
        )

    ref = service.find(loaded.name)
    if ref is None:
        raise FileNotFoundError(
            f"No J-lens artifact for {loaded.name}. The logit lens needs none; "
            "the Jacobian lens does — fit and validate one first."
        )

    stat = ref.lens_path.stat()
    key = (ref.slug, stat.st_mtime_ns, stat.st_size, loaded.d_model, loaded.n_layers)
    cached = _VALIDATION_CACHE.get(key)
    if cached is not None:
        return cached

    # THE ARTIFACT IS THE AUTHORITY ON WHICH LAYERS IT HAS, not the model's
    # config. `range(loaded.n_layers)` demanded a full-stack fit, so every
    # PARTIAL fit failed STRUCTURAL with "missing layers [0..23]" and could
    # never be served — while the fit API, the MCP tool and the UI all accept a
    # layer subset. The product offered something it then refused to honour.
    #
    # Deriving the expectation from the payload does not weaken the check: a
    # matrix of the wrong shape, a non-square one, or a non-integer key still
    # fails, and reading out at a layer the artifact lacks is refused at read
    # time by the transport rather than papered over here.
    # HONOUR THE VERDICT RECORDED AT PUBLISH TIME, when it describes this exact
    # file. The fit validated the artifact with a fixture the caller chose for
    # the layers they fitted, and that verdict was being thrown away: this path
    # re-validated with its own hard-coded fixture, which targets mid-stack and
    # therefore asks a question a top-of-stack partial fit was never fitted to
    # answer. A published, semantically-valid artifact was refused for failing
    # a different test than the one it passed.
    #
    # `stored_report` returns None if the lens file changed since — a swapped
    # artifact is revalidated rather than served on a stale verdict.
    stored = service.stored_report(ref)
    if stored is not None:
        report = _StoredReport(stored)
        _VALIDATION_CACHE[key] = report
        return report

    payload = service._load_payload(ref)  # noqa: SLF001 - same package concern
    present = sorted(int(k) for k in payload) if isinstance(payload, dict) else []

    report = service.validate(
        ref,
        d_model=loaded.d_model,
        expected_layers=present or range(loaded.n_layers),
        n_vocab=loaded.n_vocab,
        semantic_result=_semantic_check(loaded, ref, present),
    )
    _VALIDATION_CACHE[key] = report
    return report


class _StoredReport:
    """A published verdict, presented with the same surface as a fresh one.

    Only the fields consumers actually read are reconstructed. Rebuilding a real
    ValidationReport would mean re-deriving `serviceable` from parsed enum
    strings, which is a second implementation of the rule that decides whether
    an artifact may be served — and two implementations of that rule is exactly
    how one of them ends up wrong.
    """

    def __init__(self, stored: Dict[str, Any]) -> None:
        self._stored = stored
        self.passed = bool(stored.get("passed"))
        self.serviceable = bool(stored.get("serviceable"))
        self.results = stored.get("results", [])

    def summary(self) -> str:
        return self._stored.get("summary", "recorded at publish time")

    def failing_detail(self) -> str:
        """Which classes failed, and why — so a refusal is actionable."""
        bad = [
            f"{r.get('check')}: {r.get('detail')}"
            for r in self.results
            if r.get("status") == "fail"
        ]
        return "; ".join(bad) or self.summary()


def _semantic_check(loaded: Any, ref: Any, present: Optional[Sequence[int]] = None):
    """Does the artifact recover a known UNSPOKEN intermediate?

    Structure can be perfect while content is absent — a shuffled or
    zero-initialised J is the right shape and the right size and passes every
    other local class. The fixture's answer deliberately appears in neither the
    prompt nor the output, because a token present in the prompt is recoverable
    by an artifact that encodes nothing.
    """
    from ....services.jlens_readout_service import JacobianTransport, ReadoutService
    from ....services.jlens_validation import check_semantic

    service = _service()
    payload = service._load_payload(ref)  # noqa: SLF001 - same package concern
    if payload is None:
        from ....services.jlens_validation import CheckClass, CheckResult, CheckStatus

        return CheckResult(
            CheckClass.SEMANTIC, CheckStatus.FAIL, "artifact did not deserialize"
        )

    jacobians = {int(k): v for k, v in payload.items()}
    readout = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )
    # Built ONCE. JacobianTransport casts every matrix to the compute dtype in
    # its constructor — deliberately, so `apply` does not copy a d_model^2
    # matrix per call — and constructing it inside the closure moved that cost
    # back to per-invocation, over the whole artifact.
    # Scales travel here too: the SEMANTIC check reads out through this
    # transport, and a check run against an unscaled lens is not checking the
    # artifact anyone else will load.
    scales = service.layer_scales(ref)
    transport = JacobianTransport(jacobians, scales=scales)

    def top_at(prompt: str, at_layers, top_k: int):
        """Top-k at the last position for every requested layer, in ONE pass.

        `stream` captures residuals once and reads each requested layer off
        them, so passing the whole scan here costs one forward pass rather than
        one per layer.
        """
        ordered = list(at_layers)
        last = None
        for message in readout.stream(
            prompt, [transport], layers=ordered, top_n=top_k
        ):
            if isinstance(message, LensTokenMessage):
                last = message
        if last is None:
            # An empty stream is a FAILED semantic check, not a NameError and
            # not an empty pass — the distinction this feature exists for.
            raise ValueError("readout produced no token messages")
        rows = last.results[0].top_tokens
        # Indexed by POSITION IN THE REQUESTED LIST, not by absolute layer.
        return {layer: rows[i] for i, layer in enumerate(ordered) if i < len(rows)}

    # SCAN THE LAYERS THE ARTIFACT ACTUALLY HAS. This used to probe "about two
    # thirds of the way up", with a comment insisting that was not a band
    # constant. It was one: it asserts where in the stack an unspoken
    # intermediate must live, which is exactly the kind of imported boundary
    # BR-002 exists to forbid. It cost a converged LFM2 artifact, whose L9
    # readout was the right concept field with the token elsewhere.
    #
    # The scan is weaker than a single layer, so it carries a matched control:
    # the same token must NOT surface for an unrelated prompt.
    layers = sorted(present) if present else []

    return check_semantic(
        top_at,
        prompt=SEMANTIC_FIXTURE_PROMPT,
        layers=layers,
        expected_intermediate=SEMANTIC_FIXTURE_ANSWER,
        control_prompt=SEMANTIC_FIXTURE_CONTROL,
    )


# The intermediate appears in NEITHER the prompt nor the expected output, so
# recovering it cannot be explained by the artifact encoding nothing.
SEMANTIC_FIXTURE_PROMPT = "The number of legs on the animal that spins webs is"
SEMANTIC_FIXTURE_ANSWER = "spider"

# The MATCHED CONTROL for the scan above. A prompt for which the expected
# answer would be an absurd continuation: if the lens surfaces 'spider' here
# too, then surfacing it for the real prompt says nothing about the artifact,
# and the check fails however well the real prompt scored.
SEMANTIC_FIXTURE_CONTROL = "The interest rate set by the central bank was raised to"


class ReadoutAccepted(BaseModel):
    """A queued readout. The result arrives via the task, not this response.

    202, not 200: the readout has been ACCEPTED and not performed. Returning a
    body that looked like a readout would be the same lie the 501 refused to
    tell.
    """

    task_id: str
    model_id: str
    status: str = "queued"


@router.post(
    "/readout",
    response_model=ReadoutAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a position x layer lens readout",
)
async def readout(
    request: ReadoutRequest, db: AsyncSession = Depends(get_db)
) -> ReadoutAccepted:
    """Queue a readout and return its task id. Poll `/jlens/readout/{task_id}`.

    ASYNCHRONOUS BECAUSE IT MEASURABLY HAD TO BE. Bound synchronously, this
    endpoint 502'd at the ingress twice on a real model — 64.9s and 54.0s
    against nginx's 60s ceiling — because a J-space readout needs the whole
    model resident for its forward pass and loading it takes about a minute on
    CPU. Raising the proxy timeout would not bound it: readout cost is
    O(positions x layers x top_n) ON TOP of the load.

    Queueing also puts the readout in the process that can CACHE the loaded
    model across requests. A cache in the API process cannot help the worker
    and vice versa, so this is what makes the first-load cost payable once.

    Every other model-bound operation here already works this way.
    """
    from ....models.model import Model
    from ....workers.jlens_readout_tasks import compute_readout

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    from ....workers import jlens_progress

    task = compute_readout.delay(
        model_id=request.model_id,
        prompt=request.prompt,
        types=list(request.types),
        layers=request.layers,
        top_n=request.top_n,
        artifact_id=request.artifact_id,
    )
    return ReadoutAccepted(task_id=task.id, model_id=request.model_id)


class ReadoutResult(BaseModel):
    """A readout task's state, and its payload once ready.

    `readout` is null until the task succeeds. A caller that treats a pending
    task as an empty readout reproduces exactly the confusion this feature
    exists to prevent, so `status` is always present and always authoritative.
    """

    task_id: str
    status: str
    stage: Optional[str] = None
    readout: Optional[ReadoutResponse] = None
    error: Optional[str] = None


@router.get(
    "/readout/{task_id}",
    response_model=ReadoutResult,
    summary="Poll a queued readout",
)
async def readout_result(task_id: str) -> ReadoutResult:
    """Report a readout task's state.

    A FAILED task reports its reason rather than an empty readout — the
    distinction the 501 was protecting and that survives here.
    """
    import asyncio

    from ....core.celery_app import celery_app

    def _read():
        async_result = celery_app.AsyncResult(task_id)
        return async_result.state, async_result.info

    state, info = await asyncio.to_thread(_read)

    if state == "SUCCESS":
        return ReadoutResult(
            task_id=task_id, status=state, readout=ReadoutResponse(**info)
        )
    if state == "FAILURE":
        return ReadoutResult(task_id=task_id, status=state, error=str(info))
    return ReadoutResult(
        task_id=task_id,
        status=state,
        stage=(info or {}).get("stage") if isinstance(info, dict) else None,
    )


class ProbeAccepted(BaseModel):
    """Queued probe. Same two-step contract as the readout, for the same reason."""

    task_id: str
    model_id: str
    status: str = "queued"


class ProbeResult(BaseModel):
    """A polled probe. `scores` is null until `status` is SUCCESS."""

    task_id: str
    status: str
    stage: Optional[str] = None
    scores: Optional[List[ProbeScore]] = None
    #: Which mode produced these numbers. Probe and full-ranking scores can
    #: disagree, so an analysis that does not say which it used cannot be
    #: compared against one that does (BR-008).
    mode: Optional[str] = None
    lens_type: Optional[str] = None
    error: Optional[str] = None


@router.post(
    "/probe",
    response_model=ProbeAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Score named directions without ranking the vocabulary",
)
async def probe(
    request: ProbeRequest, db: AsyncSession = Depends(get_db)
) -> ProbeAccepted:
    """Probe mode (BR-008).

    Distinct from the full ranked readout: the two can disagree because ranking
    applies a data-dependent normalisation this does not, so which mode is
    canonical is RECORDED on the result rather than left to the caller.

    Queued rather than inline for the readout's measured reason — a real model
    takes about a minute to load, and nginx gives up at 60s.
    """
    from ....models.model import Model
    from ....workers.jlens_probe_tasks import compute_probe

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    task = compute_probe.delay(
        model_id=request.model_id,
        prompt=request.prompt,
        tokens=request.tokens,
        layers=request.layers,
        artifact_id=request.artifact_id,
    )
    return ProbeAccepted(task_id=task.id, model_id=request.model_id)


@router.get(
    "/probe/{task_id}",
    response_model=ProbeResult,
    summary="Poll a queued probe",
)
async def probe_result(task_id: str) -> ProbeResult:
    """Poll a probe. A FAILURE carries its REASON, never an empty score list.

    An empty `scores` on a failed task is indistinguishable from a real probe
    that found nothing — the confusion this whole feature exists to prevent.
    """
    from celery.result import AsyncResult

    from ....core.celery_app import celery_app

    async_result = AsyncResult(task_id, app=celery_app)
    state = async_result.state

    if state == "SUCCESS":
        payload = async_result.result or {}
        return ProbeResult(
            task_id=task_id,
            status="SUCCESS",
            scores=[ProbeScore(**row) for row in payload.get("scores", [])],
            mode=payload.get("mode"),
            lens_type=payload.get("lens_type"),
        )
    if state == "FAILURE":
        return ProbeResult(
            task_id=task_id, status="FAILURE", error=str(async_result.info)
        )

    info = async_result.info if isinstance(async_result.info, dict) else {}
    return ProbeResult(task_id=task_id, status=state, stage=info.get("stage"))


# ── annotation, interventions, watchlists, replication ─────────────────────
#
# These four services were implemented, unit-tested and documented while NO
# user or agent could call any of them — the same shape as the 16 MCP tools
# this project once shipped registered with nothing. The reachability harness
# does not catch it, because a harness asserts the surfaces that EXIST.


class AnnotateRequest(BaseModel):
    """Annotate one SAE feature's decoder direction (BR-012..015)."""

    #: All three are OPTIONAL and resolved from `feature_id` when omitted.
    #:
    #: The caller who wants this is looking at a FEATURE, and a feature knows
    #: its own SAE, that SAE's model and its layer. Demanding they be restated
    #: made the endpoint unusable from the one screen where features live —
    #: the modal has `training_id` and `neuron_index`, not these.
    model_id: Optional[str] = None
    sae_id: Optional[str] = None
    layer: Optional[int] = None
    #: A miStudio feature id (`feat_sae_<sae>_<index>`), or a bare index when
    #: sae_id is given explicitly.
    feature_id: str
    #: OPTIONAL now. Omit it and the server resolves this feature's decoder
    #: column from `sae_id`. A d_model vector is not something a browser can
    #: produce, so requiring it here is what kept this endpoint UI-less.
    direction: Optional[List[float]] = None
    label_tokens: List[str] = []
    top_k: int = 8


class AnnotationResponse(BaseModel):
    feature_id: str
    layer: int
    lens_kurtosis: Optional[float]
    #: UNKNOWN when no band report exists for this model. That is a real
    #: answer, not a failure: without boundaries measured HERE there is no
    #: principled middle of the stack to classify against.
    workspace_class: str
    top_tokens: List[str]
    disagreement_score: Optional[float] = None
    has_disagreement: bool = False
    #: Rung 0. Carried so a caller cannot receive an annotation stripped of
    #: what it is (BR-019).
    evidence_rung: int = 0


@router.post(
    "/annotate",
    response_model=AnnotationResponse,
    summary="Annotate a weight-space direction through the lens",
)
async def annotate(
    request: AnnotateRequest, db: AsyncSession = Depends(get_db)
) -> AnnotationResponse:
    """Project a feature's decoder direction and describe it in J-space.

    TWO INDEPENDENT FIELDS (BR-012). The geometric field alone labels every
    MOTOR feature a workspace feature, because a motor direction is sharp too —
    so `workspace_class` is reported separately and is UNKNOWN without a band
    report rather than guessed.
    """
    import torch

    from ....models.model import Model
    from ....services.jlens_annotation import annotate_direction, label_disagreement
    from ....services.jlens_band_service import load_band_report
    from ....services.jlens_readout_service import IdentityTransport

    # RESOLVE WHAT THE CALLER DID NOT RESTATE. A feature knows its own SAE,
    # that SAE's model and its layer; requiring all three made this endpoint
    # unusable from the screen where features actually live.
    sae_id, feature_index, model_id, layer = _resolve_feature_context(request)

    result = await db.execute(select(Model).where(Model.id == model_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"No model {model_id!r}")

    try:
        loaded = load_for_readout(record)
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    if request.direction is not None:
        direction = torch.tensor(request.direction, dtype=torch.float32)
    else:
        # RESOLVED SERVER-SIDE from the SAE's decoder column. A d_model vector
        # is not something a browser can produce, so requiring one here is what
        # left this endpoint without any UI at all.
        direction = _feature_direction(sae_id, feature_index)

    if direction.numel() != loaded.d_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"direction has {direction.numel()} entries but this model's "
                f"d_model is {loaded.d_model}"
            ),
        )

    ref = _service().find(loaded.name)
    band_report = _band_report_object(ref.directory) if ref else None

    annotation = annotate_direction(
        direction,
        IdentityTransport(),
        loaded.unembedding,
        layer=layer,
        feature_id=request.feature_id,
        decode=lambda ids: loaded.tokenizer.convert_ids_to_tokens(ids),
        top_k=request.top_k,
        band_report=band_report,
    )

    score = None
    if request.label_tokens:
        score = label_disagreement(request.label_tokens, annotation.top_tokens)

    return AnnotationResponse(
        feature_id=annotation.feature_id,
        layer=annotation.layer,
        lens_kurtosis=annotation.lens_kurtosis,
        workspace_class=annotation.workspace_class.value,
        top_tokens=annotation.top_tokens,
        disagreement_score=score,
        has_disagreement=bool(score is not None and score >= 0.8),
    )


def _resolve_feature_context(request):
    """(sae_id, feature_index, model_id, layer) from whatever the caller gave.

    Explicit values always win — a caller who states them is answering for
    them. Anything omitted is derived from the feature row and its SAE, which
    is where those facts already live. Nothing is DEFAULTED: an unresolvable
    field is refused with what was missing, because annotating the wrong
    layer's direction produces a complete, plausible, wrong answer.
    """
    from ....core.database import get_sync_db
    from ....models.external_sae import ExternalSAE
    from ....models.feature import Feature

    sae_id = request.sae_id
    model_id = request.model_id
    layer = request.layer
    feature_index = request.feature_id

    needs_lookup = not (sae_id and model_id and layer is not None)
    if needs_lookup:
        with get_sync_db() as db:
            feature = db.query(Feature).filter(Feature.id == request.feature_id).first()
            if feature is None and needs_lookup and not sae_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=(
                        f"No feature {request.feature_id!r}. Pass sae_id, "
                        "model_id and layer explicitly, or a feature id this "
                        "installation knows."
                    ),
                )
            if feature is not None:
                feature_index = str(feature.neuron_index)
                sae_id = sae_id or feature.external_sae_id
            if sae_id:
                sae = db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
                if sae is not None:
                    model_id = model_id or sae.model_id
                    layer = layer if layer is not None else sae.layer

    missing = [
        name
        for name, value in (("sae_id", sae_id), ("model_id", model_id), ("layer", layer))
        if value is None
    ]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"could not resolve {missing} for feature {request.feature_id!r}. "
                "Annotating against the wrong layer's direction produces a "
                "complete, plausible, wrong answer, so it is refused rather "
                "than guessed."
            ),
        )
    return sae_id, feature_index, model_id, int(layer)


def _feature_direction(sae_id: str, feature_id: str):
    """One SAE feature's decoder direction, as a d_model vector.

    Reuses `resolve_decoder_weight` — the same resolver steering and circuit
    intervention use — rather than reading the checkpoint again here. A second
    reader is a second chance to disagree about which matrix is the decoder,
    and the two would disagree silently: both produce a d_model vector.
    """
    import torch

    from ....services.steering_service import resolve_decoder_weight
    from ....services.circuit_capture_service import _load_sae_sync
    from ....core.database import get_sync_db
    from ....models.external_sae import ExternalSAE

    with get_sync_db() as db:
        record = db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No SAE with id {sae_id!r}",
            )
        sae = _load_sae_sync(record, "cpu")

    w_dec = resolve_decoder_weight(sae)  # [d_model, d_sae]
    if w_dec is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"SAE {sae_id!r} exposes no decoder weight to annotate against",
        )
    try:
        index = int(feature_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"feature_id {feature_id!r} is not an index into this SAE",
        )
    if index < 0 or index >= w_dec.shape[1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"feature {index} is outside this SAE's {w_dec.shape[1]} features"
            ),
        )
    return w_dec[:, index].detach().to(torch.float32).cpu()


def _band_report_object(directory):
    """Adapt a stored band report to the shape `classify_behaviour` reads.

    Returns None when there is none, so the behavioural field stays UNKNOWN
    rather than being classified against boundaries that do not exist.
    """
    from ....services.jlens_band_service import load_band_report

    stored = load_band_report(directory)
    if stored is None or stored.get("boundaries") is None:
        return None

    class _Bands:
        boundaries = stored["boundaries"]

    return _Bands()


class WatchlistRequest(BaseModel):
    name: str
    artifact_ref: str
    scoring_definition: str
    concepts: List[Dict[str, Any]]
    control_set: List[str] = []


class WatchlistResponse(BaseModel):
    name: str
    artifact_ref: str
    scoring_definition: str
    concept_count: int


class InterventionRequest(BaseModel):
    """Run one intervention AND its size-matched control (BR-016..018).

    There is no way to request an intervention WITHOUT a control. `control_seed`
    and `k` define it, and the result carries both outcomes plus their
    difference — an intervention that moves the output says nothing until
    compared with what a random direction of the same size does.
    """

    model_id: str
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    #: MORE PROMPTS FOR THE SAME EXPERIMENT — one TRIAL each. The source paper
    #: reports a fraction of trials (50 two-hop prompts, 192 swap trials), never
    #: a single number from a single prompt. One trial produces a Wilson
    #: interval spanning almost the whole range, which is the honest rendering
    #: of one observation.
    #: BOUNDED PER ITEM, not only in count. `prompt` was capped at 8000 and
    #: `prompts` only at 512 entries of unlimited length, so 512 x 400 000
    #: characters passed validation from one unauthenticated POST — and the
    #: worker budgets against `prompt`, which it then discards whenever
    #: `prompts` is present. That is the whole single-GPU queue, for hours,
    #: behind which every fit and readout blocks.
    prompts: Optional[List[str]] = Field(
        None, max_length=512, description="One trial each; each capped like `prompt`."
    )
    #: The token whose RANK is scored in the model's output. Defaults to
    #: `direction_token`. A coordinate swap wants them different: push
    #: direction A, ask whether answer B arrives.
    target_token: Optional[str] = None
    #: CONSTRAINED, not free text. A typo — "aditive" from a script or an agent
    #: — used to pass the schema, return 202 with a task id, take a slot on the
    #: single-GPU queue behind a possible 45-minute fit, and fail before its
    #: first progress report. The enum makes the unknown case a 422 at the door.
    primitive: Literal[
        "additive", "projective_ablation", "coordinate_swap", "dynamic_topk_ablation"
    ] = Field(
        ...,
        description=(
            "additive | projective_ablation | coordinate_swap. A "
            "coordinate_swap needs TWO different tokens — `direction_token` is "
            "the coordinate to move and `target_token` the one to exchange it "
            "with; one token would run an additive steer under a swap's name. "
            "dynamic_topk_ablation is not implemented on this path: it needs "
            "the lens coordinates at the intervened site, which this "
            "measurement does not compute."
        ),
    )
    layers: List[int] = Field(..., min_length=1)
    #: A raw d_model vector. Usable from a script; NOT usable from a browser,
    #: which has no access to the unembedding or to SAE decoder weights — which
    #: is why `direction_token` exists and why this whole surface had no UI.
    direction: Optional[List[float]] = None
    #: Resolve the direction SERVER-SIDE from a token string: the direction is
    #: that token's unembedding row. This is what makes the intervention
    #: reachable from the readout panel, where tokens are the thing on screen.
    #: Supplying both is refused rather than silently preferring one.
    direction_token: Optional[str] = None
    strength: float = 1.0
    #: Control size. Size-matched to the intervention, never 0.
    k: int = Field(1, ge=1)
    #: Required in practice: a control nobody can reconstruct is not a control.
    control_seed: int = 0
    positions: Optional[List[int]] = None
    artifact_id: Optional[str] = None

    @model_validator(mode="after")
    def _swap_needs_two_tokens(self) -> "InterventionRequest":
        """Refuse a swap with one token HERE, not in the worker.

        The worker already refuses it — but it does so after this endpoint has
        returned 202 with a task id. The caller is told the request was
        accepted, the job takes a slot on a single-GPU queue, and the refusal
        arrives a minute later behind a poll. Whether two tokens were supplied
        is knowable at request time and needs no model, so the honest answer is
        a 400 before anything is queued.
        """
        if self.primitive == "coordinate_swap":
            if not self.target_token or self.target_token == self.direction_token:
                raise ValueError(
                    "coordinate_swap needs TWO different tokens: "
                    "`direction_token` is the coordinate to move and "
                    "`target_token` the one to exchange it with. One token "
                    "would run an additive steer under a swap's name."
                )
        if self.primitive == "dynamic_topk_ablation":
            raise ValueError(
                "dynamic_topk_ablation is not implemented for the forward-pass "
                "path: it needs the lens coordinates at the intervened site, "
                "which this measurement does not compute. Use additive, "
                "projective_ablation or coordinate_swap."
            )

        # EVERY TRIAL PROMPT BOUNDED LIKE `prompt`. The worker runs one forward
        # pass per trial per arm, so an unbounded list item is 3 x 512 passes
        # over a text nothing measured.
        for i, p in enumerate(self.prompts or []):
            if not p.strip():
                raise ValueError(f"prompts[{i}] is empty; a trial needs a prompt")
            if len(p) > MAX_PROMPT_CHARS:
                raise ValueError(
                    f"prompts[{i}] is {len(p)} characters; the limit is "
                    f"{MAX_PROMPT_CHARS}, the same bound `prompt` carries"
                )

        # LAYERS DISTINCT. The worker registers one forward hook per entry, and
        # each hook reads the ALREADY-PERTURBED hidden state — so [9, 9, 9] at
        # strength 1.0 applies 3.0 and records the recipe as strength 1.0 at
        # layer 9. Reproducing that recipe cannot reproduce the result.
        if len(set(self.layers)) != len(self.layers):
            dupes = sorted({l for l in self.layers if self.layers.count(l) > 1})
            raise ValueError(
                f"layers {dupes} appear more than once. Each entry registers its "
                "own hook and each hook perturbs the output of the one before, "
                "so a repeat multiplies the strength while the recipe still "
                "reports the nominal value."
            )
        if len(self.layers) > MAX_INTERVENED_LAYERS:
            raise ValueError(
                f"{len(self.layers)} layers requested; the limit is "
                f"{MAX_INTERVENED_LAYERS}. Every layer is another hook on every "
                "forward pass of every trial and every arm."
            )

        # POSITIONS DISTINCT, for the same reason: the hook loops over them and
        # writes into the tensor it is reading.
        if self.positions is not None and len(set(self.positions)) != len(
            self.positions
        ):
            raise ValueError(
                "positions repeat; each is perturbed in turn and a repeat "
                "perturbs an already-perturbed activation"
            )
        return self


class TokenCheckRequest(BaseModel):
    """Is this string a single token in THIS model's vocabulary?"""

    model_id: str
    #: Bounded: this exists to check a handful of hand-typed strings, not to
    #: tokenise a corpus. `/datasets/tokenize-preview` is the endpoint for that.
    tokens: List[str] = Field(..., min_length=1, max_length=32)


class TokenCheck(BaseModel):
    token: str
    #: The ids it encodes to. Present even when there is more than one, because
    #: seeing "[ 4874, 883 ]" is what makes "this is two tokens" concrete.
    ids: List[int]
    n_tokens: int
    #: Usable as a lens direction. A direction is `W_U[id]`, which is defined for
    #: exactly one id.
    usable: bool
    detail: str


@router.post(
    "/token-check",
    response_model=List[TokenCheck],
    summary="Is a string a single token in this model's vocabulary?",
)
async def token_check(
    request: TokenCheckRequest, db: AsyncSession = Depends(get_db)
) -> List[TokenCheck]:
    """Resolve hand-typed tokens against the model's OWN tokenizer.

    WHY THIS EXISTS. A lens direction is an unembedding row, so ANY single token
    has one — including tokens the readout never surfaced, which are precisely
    the interesting swap targets ("does ' Rome' arrive if I put ' Paris' where
    it was?"). Restricting the UI to tokens already on screen was a limit the
    server never had.

    But whether a string is ONE token is a property of the model's vocabulary,
    not of the string: `' Rome'` is single on one model and two on another, and
    a leading space usually decides it. The worker refuses a multi-token
    direction — correctly, since it has no single row — but only after a 202 and
    a slot on a single-GPU queue that may be behind a 45-minute fit. That is
    knowable here, from the tokenizer alone, without loading any weights.
    """
    from ....models.model import Model
    from ....services.jlens_model_registry import tokenizer_for

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    try:
        tok = tokenizer_for(record)
    except ModelNotAvailable as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - a 500 tells the caller nothing
        # NEVER A BARE 500. This endpoint exists to save a caller a wasted GPU
        # slot; an opaque error here just moves the confusion earlier.
        logger.exception("Tokenizer check failed for %s", request.model_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Could not read the vocabulary for {request.model_id}: {exc}",
        ) from exc

    out: List[TokenCheck] = []
    for raw in request.tokens:
        ids = tok.encode(raw, add_special_tokens=False)
        n = len(ids)
        if n == 1:
            detail = "One token — usable as a direction."
        elif n == 0:
            detail = "Encodes to nothing; there is no direction to act along."
        else:
            # THE LEADING-SPACE HINT, because it is the cause almost every time
            # and it is invisible in a text box.
            alt = tok.encode(f" {raw.strip()}", add_special_tokens=False)
            detail = (
                f"{n} tokens. A lens direction is defined for a SINGLE token."
                + (
                    f" Try ' {raw.strip()}' with a leading space — that is one token."
                    if len(alt) == 1
                    else ""
                )
            )
        out.append(
            TokenCheck(
                token=raw, ids=ids, n_tokens=n, usable=(n == 1), detail=detail
            )
        )
    return out


def _queue_without_leaking_token(task: Any, **kwargs: Any) -> Any:
    """Queue a task whose kwargs include a credential.

    CELERY PUTS `kwargsrepr` IN THE MESSAGE HEADERS, and `task_send_sent_event`
    is on — so every `task-sent` event published to `celeryev` carries the
    rendered kwargs, readable by Flower, `celery events`, or any monitoring
    consumer attached to the broker. Verified: a HuggingFace token passed to
    `.delay()` appears in `message.headers` verbatim.

    `kwargsrepr` is what headers and events render; the BODY still carries the
    real values, so the worker is unaffected. Redacting it is the difference
    between a write credential sitting in a monitoring stream and not.
    """
    redacted = {
        k: ("***" if "token" in k else v) for k, v in kwargs.items()
    }
    return task.apply_async(kwargs=kwargs, kwargsrepr=repr(redacted))


class AcquirePreviewRequest(BaseModel):
    """Look before fetching. Read-only, and that is the point."""

    repo_id: str = Field(..., min_length=1, max_length=200)
    revision: Optional[str] = Field(None, max_length=100)
    #: Which model this would be attached to. Optional, and when given the
    #: response carries a per-file envelope verdict for THAT model's dimensions
    #: — which is what makes a generic any-repo flow usable rather than a guess.
    model_id: Optional[str] = None
    access_token: Optional[str] = Field(None, max_length=500)


class AcquireCandidate(BaseModel):
    path: str
    size_bytes: Optional[int] = None
    #: Beside a `config.yaml`, so weight identity can be CHECKED rather than
    #: asserted by the caller.
    has_config: bool = False
    has_convergence: bool = False
    #: None when no model was named. Otherwise whether this file's size is
    #: plausible for that model's dimensions — BR-006's guard, applied
    #: pre-flight and for free.
    fits_envelope: Optional[bool] = None
    envelope_detail: Optional[str] = None


class AcquirePreviewResponse(BaseModel):
    repo_id: str
    #: The RESOLVED commit. `main` moves, so an acquisition pinned to it is not
    #: a reproducible statement.
    revision: str
    candidates: List[AcquireCandidate]


@router.post(
    "/acquire/preview",
    response_model=AcquirePreviewResponse,
    summary="List downloadable lens candidates in a HuggingFace repo",
)
async def acquire_preview(
    request: AcquirePreviewRequest, db: AsyncSession = Depends(get_db)
) -> AcquirePreviewResponse:
    """What is in this repo that could be a lens, and would it fit this model?

    POST RATHER THAN GET because it carries a token, and a token in a query
    string lands in access logs and browser history.

    It lists every `*.pt`/`*.safetensors`, not just conformant
    `*_jacobian_lens.pt` names: community repos publish `qwen3_8b_lens.pt` and
    `gemma2_9b_jlens.pt`, and filtering to the conformant name would make this
    useless for exactly the repos it exists to reach.
    """
    from ....models.model import Model
    from ....services.huggingface_sae_service import resolve_hf_token
    from ....services.jlens_acquire_service import (
        AcquisitionRefused,
        model_dims,
        preview_envelope_verdict,
        preview_repo,
    )
    from ....services.jlens_validation import CheckStatus, check_envelope

    dims = None
    if request.model_id:
        result = await db.execute(select(Model).where(Model.id == request.model_id))
        record = result.scalar_one_or_none()
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No model with id {request.model_id!r}",
            )
        dims = model_dims(record)

    try:
        preview = preview_repo(
            request.repo_id,
            revision=request.revision,
            token=resolve_hf_token(request.access_token),
        )
    except AcquisitionRefused as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - a bad repo id is a 404, not a 500
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not read {request.repo_id!r}: {exc}",
        ) from exc

    candidates: List[AcquireCandidate] = []
    for c in preview.candidates:
        fits: Optional[bool] = None
        detail: Optional[str] = None
        if dims and c.size_bytes:
            verdict = preview_envelope_verdict(c.size_bytes, dims)
            fits = verdict["fits"]
            detail = verdict["detail"]
        candidates.append(
            AcquireCandidate(
                path=c.path,
                size_bytes=c.size_bytes,
                has_config=c.has_config,
                has_convergence=c.has_convergence,
                fits_envelope=fits,
                envelope_detail=detail,
            )
        )
    return AcquirePreviewResponse(
        repo_id=preview.repo_id, revision=preview.revision, candidates=candidates
    )


class AcquireRequest(BaseModel):
    """Adopt one published lens for one local model."""

    model_id: str
    repo_id: str = Field(..., min_length=1, max_length=200)
    path_in_repo: str = Field(..., min_length=1, max_length=500)
    #: Pinned when given. When absent the preview resolves `main` to a sha and
    #: the worker uses THAT, so the acquisition is reproducible either way.
    revision: Optional[str] = Field(None, max_length=100)
    access_token: Optional[str] = Field(None, max_length=500)
    #: The incumbent covers layers this one does not. Refused by default.
    allow_coverage_loss: bool = False
    #: The incumbent is a stronger fit. Refused by default.
    allow_quality_regression: bool = False
    #: Overwrite an artifact already sitting in this model's staging directory.
    #: Refused by default — it may be completed work that a gate declined to
    #: publish, and the documented recovery is to re-run with a flag rather than
    #: to have it deleted underneath you.
    replace_staged: bool = False


@router.post(
    "/acquire",
    response_model=BandTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Download a published J-lens and validate it for a local model",
)
async def acquire_artifact(
    request: AcquireRequest, db: AsyncSession = Depends(get_db)
) -> BandTaskAccepted:
    """Queue an acquisition, after refusing everything refusable from here.

    THREE SYNCHRONOUS REFUSALS, and they are the point of this endpoint rather
    than a nicety. Everything below is knowable without the GPU, and a request
    that gets a 202 takes a slot on the single-GPU queue — possibly behind a
    45-minute fit — before it can discover it was doomed. This project has a
    written doctrine about it after a production incident.
    """
    from ....models.model import Model
    from ....services.jlens_acquire_service import (
        MIN_FREE_DISK_BYTES,
        check_free_space,
        AcquisitionRefused,
    )
    from ....services.jlens_artifact_service import JLensArtifactService
    from ....services.jlens_model_registry import ModelNotAvailable, locate_weights
    from ....workers import jlens_progress
    from ....workers.jlens_acquire_tasks import acquire_jlens_artifact

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    # 1. THE WEIGHTS MUST BE HERE. A lens is unusable without them: the readout
    #    runs a real forward pass and needs the unembedding, and the validation
    #    this acquisition performs IS a readout. Discovering that after a
    #    265 MB download is the expensive way to learn it.
    repo_of_model = getattr(record, "repo_id", None) or ""
    try:
        locate_weights(record)
    except ModelNotAvailable as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    # 2. IT MUST FIT ON DISK. No download path in this project checked, and the
    #    data volume also holds every model, dataset and checkpoint.
    try:
        # THE CACHE VOLUME TOO. `jlens_artifacts_dir` lives inside `data_dir`,
        # so the original pair probed one filesystem twice — and after the
        # device-dedup that became a single probe of a single volume, while the
        # download lands in the HuggingFace cache FIRST.
        check_free_space(
            settings.jlens_artifacts_dir,
            settings.hf_cache_dir,
            needed_bytes=0,
        )
    except AcquisitionRefused as exc:
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE, detail=str(exc)
        ) from exc

    # 3. STAGING MUST BE FREE. Verified on hardware: leftover debris from an
    #    interrupted fit refused this acquisition — correctly, it was a
    #    converged 549-prompt artifact — but only after the worker had already
    #    downloaded 265 MB, because `stage_from_file` runs after the fetch.
    #    Whether a conformant artifact is sitting in staging is a `stat` away.
    service = JLensArtifactService(settings.jlens_artifacts_dir)
    if not request.replace_staged:
        staging = service.staging_dir(repo_of_model)
        if staging.is_dir() and service._ref_for(staging) is not None:  # noqa: SLF001
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"{staging.name} already holds a staged artifact for this "
                    "model. It may be completed work a gate refused; inspect it, "
                    "or pass replace_staged to overwrite it deliberately."
                ),
            )

    task = _queue_without_leaking_token(
        acquire_jlens_artifact,
        model_id=request.model_id,
        repo_id=request.repo_id,
        path_in_repo=request.path_in_repo,
        revision=request.revision,
        access_token=request.access_token,
        allow_coverage_loss=request.allow_coverage_loss,
        allow_quality_regression=request.allow_quality_regression,
        replace_staged=request.replace_staged,
    )
    # VISIBLE WHILE IT RUNS, like every other J-space task. Opened here rather
    # than in the worker so a job that never gets picked up still appears.
    jlens_progress.open_row(jlens_progress.ACQUIRE, request.model_id, task.id)
    return BandTaskAccepted(task_id=task.id, model_id=request.model_id)


class PublishRequest(BaseModel):
    """Upload this model's published lens to a HuggingFace repo."""

    model_id: str
    target_repo: str = Field(..., min_length=3, max_length=200)
    #: REQUIRED IN PRACTICE. The read path may run anonymously; a write cannot,
    #: and the worker refuses rather than sending an empty credential.
    access_token: Optional[str] = Field(None, max_length=500)
    #: The corpus segment of the published path, per spec §2.1.
    #:
    #: CONSTRAINED, because it is interpolated into a repo path. `..` or a
    #: leading `/` places the artifact outside the conformance layout the whole
    #: feature rests on — the Hub accepts it, so the lens commits somewhere no
    #: consumer resolving `<model>/jlens/<dataset>/` will look, under a README
    #: describing a third location.
    dataset: str = Field(
        "mistudio",
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    create_repo: bool = False
    private: bool = False


@router.post(
    "/publish",
    response_model=BandTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Publish a validated J-lens to HuggingFace",
)
async def publish_artifact_endpoint(
    request: PublishRequest, db: AsyncSession = Depends(get_db)
) -> BandTaskAccepted:
    """Queue an upload, refusing here what is knowable here.

    A STAGED ARTIFACT IS NOT PUBLISHED AND IS NOT SHIPPED. Publishing to a third
    party is a stronger act than serving locally, and `find` returns only what
    has been committed — so this refuses before spending a task slot on a model
    whose lens was never validated.
    """
    from ....models.model import Model
    from ....services.jlens_acquire_service import AcquisitionRefused
    from ....services.huggingface_sae_service import resolve_hf_token
    from ....services.jlens_artifact_service import JLensArtifactService
    from ....workers import jlens_progress
    from ....workers.jlens_acquire_tasks import publish_jlens_artifact_task

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )
    repo_of_model = getattr(record, "repo_id", None)
    if not repo_of_model:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model {request.model_id!r} has no repo_id",
        )

    service = JLensArtifactService(settings.jlens_artifacts_dir)
    ref = service.find(repo_of_model)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"No published J-lens artifact for {repo_of_model}. Fit or "
                "acquire one first; a staged artifact is not published."
            ),
        )
    if service.stored_report(ref) is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "The artifact carries no validation verdict matching its current "
                "weights, so there is nothing to publish it on."
            ),
        )

    # A WRITE TOKEN IS KNOWABLE HERE. Deferring it to the worker spends a slot
    # on the single-GPU queue — possibly behind a 45-minute fit — to discover
    # something a sync function already answers, which is the doctrine this file
    # states 130 lines above.
    if not resolve_hf_token(request.access_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Publishing needs a HuggingFace token with write access. None "
                "was supplied and none is configured."
            ),
        )

    task = _queue_without_leaking_token(
        publish_jlens_artifact_task,
        model_id=request.model_id,
        target_repo=request.target_repo,
        access_token=request.access_token,
        dataset=request.dataset,
        create_repo=request.create_repo,
        private=request.private,
    )
    jlens_progress.open_row(jlens_progress.PUBLISH, request.model_id, task.id)
    return BandTaskAccepted(task_id=task.id, model_id=request.model_id)


@router.post(
    "/interventions",
    response_model=BandTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run a J-space intervention with its matched control (BR-018)",
)
async def run_intervention(
    request: InterventionRequest, db: AsyncSession = Depends(get_db)
) -> BandTaskAccepted:
    """Queue an intervention. The control runs in the same task, always."""
    from ....models.model import Model
    from ....workers.jlens_intervention_tasks import run_intervention_task

    result = await db.execute(select(Model).where(Model.id == request.model_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with id {request.model_id!r}",
        )

    if request.direction is not None and request.direction_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "supply direction OR direction_token, not both — silently "
                "preferring one would run an intervention along a direction "
                "the caller did not choose"
            ),
        )

    from ....workers import jlens_progress

    task = run_intervention_task.delay(
        model_id=request.model_id,
        prompt=request.prompt,
        primitive=request.primitive,
        prompts=request.prompts,
        target_token=request.target_token,
        layers=request.layers,
        direction=request.direction,
        direction_token=request.direction_token,
        strength=request.strength,
        k=request.k,
        control_seed=request.control_seed,
        positions=request.positions,
        artifact_id=request.artifact_id,
    )
    jlens_progress.open_row(
        jlens_progress.INTERVENTION, request.model_id, task.id
    )
    return BandTaskAccepted(task_id=task.id, model_id=request.model_id)


@router.post(
    "/watchlists",
    response_model=WatchlistResponse,
    summary="Author a watchlist for runtime handoff",
)
async def create_watchlist(request: WatchlistRequest) -> WatchlistResponse:
    """Validate and echo a watchlist definition (BR-025).

    A watchlist missing its scoring definition or its artifact reference is
    REFUSED here rather than exported and discovered later: a threshold applied
    to a differently computed score is a different detector, and the consumer
    has no way to notice.
    """
    from ....services.jlens_watchlist import WatchedConcept, Watchlist

    try:
        watchlist = Watchlist(
            name=request.name,
            concepts=[
                WatchedConcept(token=c["token"], threshold=float(c["threshold"]))
                for c in request.concepts
            ],
            scoring_definition=request.scoring_definition,
            artifact_ref=request.artifact_ref,
            control_set=request.control_set,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return WatchlistResponse(
        name=watchlist.name,
        artifact_ref=watchlist.artifact_ref,
        scoring_definition=watchlist.scoring_definition,
        concept_count=len(watchlist.concepts),
    )


class CostEstimateResponse(BaseModel):
    operation: str
    order_of_magnitude_seconds: float
    order_of_magnitude_peak_bytes: int
    basis: str
    is_estimate: bool = True


@router.get(
    "/cost-estimate",
    response_model=CostEstimateResponse,
    summary="Estimate an operation's cost BEFORE committing to it",
)
async def cost_estimate(
    operation: str,
    d_model: int,
    n_layers: int,
    n_positions: int = 1,
    n_prompts: int = 1,
    n_features: int = 1,
) -> CostEstimateResponse:
    """Order-of-magnitude cost for one J-space operation class (BR-028).

    An unknown class is a 400 rather than a cheap default: a small number
    invites exactly the run it should warn about, and a caller cannot tell
    "cheap" from "unmeasured".
    """
    from ....services.jlens_watchlist import OperationClass, estimate_cost

    try:
        op = OperationClass(operation)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"unknown operation {operation!r}. Known: "
                f"{[o.value for o in OperationClass]}"
            ),
        )

    est = estimate_cost(
        op,
        d_model=d_model,
        n_layers=n_layers,
        n_positions=n_positions,
        n_prompts=n_prompts,
        n_features=n_features,
    )
    return CostEstimateResponse(
        operation=est.operation.value,
        order_of_magnitude_seconds=est.order_of_magnitude_seconds,
        order_of_magnitude_peak_bytes=est.order_of_magnitude_peak_bytes,
        basis=est.basis,
    )


@router.get(
    "/reports/replication",
    response_model=Optional[Dict[str, Any]],
    summary="The recorded replication report (BR-001)",
)
async def replication_report(slug: str) -> Optional[Dict[str, Any]]:
    """Return the stored replication report, or null when none was recorded.

    Published whether favourable or not — there is no filter here, which is the
    structural half of BR-001.
    """
    from ....services.jlens_replication import load_replication_report

    ref = next((a for a in _service().list_artifacts() if a.slug == slug), None)
    if ref is None:
        raise HTTPException(status_code=404, detail=f"No artifact {slug!r}")
    return load_replication_report(ref.directory)

