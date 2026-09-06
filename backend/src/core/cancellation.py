"""Cooperative cancellation: the row is the channel, and the task polls it.

WHY THIS IS NOT `revoke(terminate=True)`.
Every Celery worker in this deployment is `--pool=solo -c 1` — the GPU worker
(`k8s/base/backend.yaml`), the CPU worker (same manifest, see its comment), and
the steering worker. Celery's `terminate` signals a POOL CHILD, and solo has
none. Worse, a solo worker executing a task is not reading the control queue at
all, so the revoke is never even delivered: it returns cleanly, changes nothing,
and the worker does not appear in `inspect()`. Verified on hardware 2026-09-05
against a running gemma-4-12B J-lens fit, which then needed a SIGKILL on the
worker PID.

SIGKILL IS NOT THE FALLBACK. `workers/steering_worker_state.py` records what
happens: killing a solo worker mid-task crashed the pool with celery's "cannot
unpack non-iterable ExceptionInfo" and stranded the in-flight `acks_late`
message for the full 12-hour `visibility_timeout`, leaving zombie rows. PID-kill
remains an operator escape hatch, never an automated one.

So: an endpoint writes the request to a row, and the task reads that row at a
checkpoint it chooses. This module is the one implementation of that, and
`SCOPES` below is the one place the project's several terminal vocabularies are
written down.

THIS DOCSTRING IS THE SINGLE HOME FOR THE ABOVE. The solo-pool explanation was
duplicated, with drift, across seven files — every copy correct, none
authoritative, and each one an invitation to rediscover the same finding a
ninth time. Modules that need it now point here rather than restating it.

────────────────────────────────────────────────────────────────────────────
OPERATOR PROCEDURE — killing a wedged worker. NOT A CODE PATH.

Cooperative cancellation cannot help with a task that is stuck below its
finest checkpoint: a single `from_pretrained` on a 70B model, an NCCL collective
that never returns. For those the only recourse is a signal, and it is an
OPERATOR action taken deliberately, never something the product does on a
timer.

  1. Find the worker:  `ps -o pid,etime,cmd -C python | grep celery`
     Match on `/proc/<pid>/cmdline`, NOT on a `pgrep -f` pattern — a pattern
     that appears in your own command line matches your own shell, and a wait
     loop written that way never exits. That has cost real time here twice.
  2. Prefer SIGTERM first. Dataset tokenization installs an owner-bound handler
     that reaps its `Dataset.map` child pool and saves completed work.
  3. SIGKILL only if SIGTERM does nothing after a minute, and EXPECT THE COST:
     the pool dies with celery's "cannot unpack non-iterable ExceptionInfo",
     and the in-flight `acks_late` message is redelivered only after the full
     12-hour `visibility_timeout`. Clear the stranded row by hand.
  4. Never `pkill -f`. Use the tracked PIDs — for steering,
     `api/v1/endpoints/steering.py` keeps them precisely so a pattern kill is
     never needed.

If you find yourself doing this routinely for one task, that task is missing a
checkpoint. Add one here instead.
────────────────────────────────────────────────────────────────────────────

THE GUARD IS NOT OPTIONAL. Between the endpoint writing "cancelled" and the task
noticing, the task is still reporting progress. Without `record_progress`'s
terminal guard its next status write overwrites the cancellation — which is
worse than having no cancel at all, because the operator is told it worked.
Generalised from two independent rediscoveries of the same rule:
`workers/jlens_progress.update_row` and `workers/training_tasks.py:90-98`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Literal, Optional

logger = logging.getLogger(__name__)

#: Poll at most this often. See `cancel_checker` for why this is time and not a
#: call count.
DEFAULT_MIN_INTERVAL_S = 2.0


class OperatorCancelled(BaseException):
    """The operator asked this job to stop. NOT an error.

    DERIVES FROM BaseException, DELIBERATELY. An `Exception` here is swallowed
    by code that already exists on the very paths this must work on:

      * `services/activation_service.py` wraps the progress callback in
        `except Exception: logger.warning("Progress callback failed")` — the
        natural checkpoint of the worst-broken task would silently ignore it.
      * MIS-E2E-058: labeling's outer `except Exception` caught its own
        cancellation and wrote FAILED.
      * `workers/neuronpedia_tasks.py` has a bare `except Exception` that calls
        `mark_export_failed`.

    Three independent sites that would turn "the operator stopped it" into
    "it crashed". BaseException cannot be caught by any of them by accident.

    Two consequences to handle deliberately rather than discover later:
      1. `except Exception` cleanup is SKIPPED. Cleanup that must run on
         cancellation belongs in `finally`, not in an exception handler.
      2. Celery's `autoretry_for` tuples are all Exception subclasses, so a
         cancellation can never trigger a retry — which is what we want.
    """

    def __init__(
        self,
        scope: str,
        target_id: Any,
        reason: str = "cancelled",
        detail: str = "",
    ) -> None:
        self.scope = scope
        self.target_id = target_id
        self.reason = reason          # "cancelled" | "deleted"
        self.detail = detail
        super().__init__(
            f"{scope}:{target_id} {reason}" + (f" — {detail}" if detail else "")
        )


@dataclass(frozen=True)
class CancelScope:
    """How one lifecycle expresses "cancelled", "terminal" and "gone".

    `model` is a zero-arg callable rather than a class so the registry can be
    imported from `src.core` without dragging in `src.models` — those import
    `src.core.database.Base`, so a module-level import here is circular. The
    same lazy-import discipline `workers/jlens_progress.py` already follows.
    """

    kind: str
    model: Callable[[], type]
    id_field: str = "id"
    status_field: str = "status"
    #: Set when the status column cannot carry a cancelled value — the three
    #: tables whose status is a NATIVE POSTGRES ENUM with no CANCELLED member
    #: (datasets, models, dataset_tokenizations). Extending a PG enum needs a
    #: non-transactional ALTER TYPE; a nullable timestamp is one additive
    #: migration and separates "the operator asked" from "the job stopped",
    #: which is the conflation behind today's ERROR + "Cancelled by user".
    request_field: Optional[str] = None
    cancelled_values: frozenset = frozenset({"cancelled"})
    terminal_values: frozenset = frozenset({"completed", "failed", "cancelled"})
    #: What a vanished row means. "continue" for a lifecycle where deletion is
    #: unrelated; "cancelled" where deleting the row IS the stop signal — the
    #: labeling DELETE path deletes the row, which currently makes the check
    #: find nothing and return, so the job runs on against a deleted row.
    missing_row: Literal["continue", "cancelled"] = "continue"
    error_field: Optional[str] = "error_message"
    progress_field: Optional[str] = "progress"
    started_at_field: Optional[str] = None
    completed_at_field: Optional[str] = "completed_at"
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S


def _coerce_status(model: type, field: str, value: Any) -> Any:
    """Translate the registry's lowercase vocabulary into the column's own type.

    Scopes are written in plain lowercase strings because that is the only
    vocabulary all fifteen lifecycles share. Half the status columns are bare
    `String`, and the string goes straight in. The other half are `SQLEnum(...)`
    — and `activation_extractions.status` is declared WITHOUT `values_callable`,
    so SQLAlchemy persists the member NAME ("CANCELLED"), not the value. A raw
    "cancelled" assigned there is not a key SQLAlchemy can look up, and it fails
    at flush, inside a write nobody is watching, on the cancellation path.

    Unknown values pass through untouched: an invalid status must fail at the
    column, loudly, rather than be silently dropped here.
    """
    if value is None or not isinstance(value, str):
        return value
    try:
        enum_class = getattr(model.__table__.columns[field].type, "enum_class", None)
    except Exception:  # noqa: BLE001 - a fake row in a test has no __table__
        return value
    if enum_class is None:
        return value
    # MATCHED ON THE VALUE ONLY. A second branch matching `member.name` was here
    # and was pure redundancy — every status enum in this project spells its
    # value as the lowercase of its name, so the two branches always agree and
    # neither could be mutation-tested; a control that disabled one passed
    # silently on the other. Values are the vocabulary the scopes are written
    # in, so that is the one thing translated here. A member whose value does
    # not match falls through to the column, which rejects it loudly.
    wanted = value.lower()
    for member in enum_class:
        if str(getattr(member, "value", member)).lower() == wanted:
            return member
    return value


def _norm(value: Any) -> Optional[str]:
    """Compare statuses as plain lowercase strings.

    Columns are a mix of `SQLEnum(...)` (returning an enum member) and bare
    `String` (returning the value). `str, Enum` members compare equal to their
    value, but `.value` is explicit and survives a non-str enum later.
    """
    if value is None:
        return None
    inner = getattr(value, "value", value)
    return str(inner).lower()


# --------------------------------------------------------------------------
# The registry — the ONE place the vocabularies are reconciled
# --------------------------------------------------------------------------

SCOPES: Dict[str, CancelScope] = {}


def register(scope: CancelScope) -> CancelScope:
    if scope.kind in SCOPES:
        raise ValueError(f"duplicate cancel scope {scope.kind!r}")
    SCOPES[scope.kind] = scope
    return scope


def get_scope(kind: str) -> CancelScope:
    """A missing scope is a PROGRAMMING error, not a 404 — fail loudly."""
    try:
        return SCOPES[kind]
    except KeyError:
        raise KeyError(
            f"no cancel scope registered for {kind!r}. Known: "
            f"{sorted(SCOPES)}. Add one in src/core/cancellation.py rather "
            f"than inventing a second convention."
        ) from None


def _task_queue():
    from ..models.task_queue import TaskQueue
    return TaskQueue


def _activation_extraction():
    from ..models.activation_extraction import ActivationExtraction
    return ActivationExtraction


def _extraction_job():
    from ..models.extraction_job import ExtractionJob
    return ExtractionJob


def _labeling_job():
    from ..models.labeling_job import LabelingJob
    return LabelingJob


def _training():
    from ..models.training import Training
    return Training


def _export_job():
    from ..models.neuronpedia_export import NeuronpediaExportJob
    return NeuronpediaExportJob


def _circuit():
    from ..models.circuit import Circuit
    return Circuit


def _capture_run():
    from ..models.circuit_runs import CircuitCaptureRun
    return CircuitCaptureRun


def _discovery_run():
    from ..models.circuit_runs import CircuitDiscoveryRun
    return CircuitDiscoveryRun


def _record_run():
    from ..models.steering_record_run import SteeringRecordRun
    return SteeringRecordRun


def _enhanced_labeling_job():
    from ..models.enhanced_labeling_job import EnhancedLabelingJob
    return EnhancedLabelingJob


def _grouping_run():
    from ..models.feature_grouping import FeatureGroupingRun
    return FeatureGroupingRun


def _dataset():
    from ..models.dataset import Dataset
    return Dataset


def _model():
    from ..models.model import Model
    return Model


def _tokenization():
    from ..models.dataset_tokenization import DatasetTokenization
    return DatasetTokenization


#: J-space work is tracked in `task_queue`, keyed by the CELERY id. That is one
#: registered scope, NOT the universal channel: task_queue is populated by only
#: three lifecycles, and its key does not exist until after `.delay()` — the
#: race `jlens_progress.mark_running`'s retry loop exists to paper over.
register(CancelScope(
    kind="jlens_task",
    model=_task_queue,
    id_field="task_id",
    started_at_field="started_at",
))

register(CancelScope(
    kind="activation_extraction",
    model=_activation_extraction,
    #: EXTRACTING and LOADING and SAVING are all live; only these three end it.
    terminal_values=frozenset({"completed", "failed", "cancelled"}),
))

register(CancelScope(
    kind="sae_extraction",
    model=_extraction_job,
    #: `saes.py` currently writes FAILED + "Cancelled by user" on cancel while
    #: `models.py` writes CANCELLED for the sibling lifecycle. The enum has
    #: CANCELLED; this scope is what makes the two agree.
    terminal_values=frozenset({"completed", "failed", "cancelled"}),
))

register(CancelScope(
    kind="nlp_analysis",
    model=_extraction_job,
    status_field="nlp_status",
    #: A DELETED extraction is how the NLP pass learns to stop today, and it
    #: reports "aborted" rather than "cancelled". `missing_row` keeps that
    #: distinction instead of flattening it.
    missing_row="cancelled",
    terminal_values=frozenset({"completed", "failed", "cancelled"}),
    completed_at_field=None,
))

register(CancelScope(
    kind="labeling",
    model=_labeling_job,
    #: THE ROW BEING GONE IS A STOP SIGNAL HERE. `delete_labeling_job` revokes
    #: (inert) and deletes the row, so `_raise_if_cancelled` finds nothing and
    #: returns — the job then runs to completion against a deleted row.
    missing_row="cancelled",
))

#: PENDING/COMPUTING/PACKAGING are all live. The export writer reports
#: progress and a stage and NEVER a status, so this scope is only guarded at
#: all because a terminal row refuses a progress move as well as a status one.
register(CancelScope(
    kind="neuronpedia_export",
    model=_export_job,
    #: `DELETE /export/{job_id}` removes the row outright, so a vanished row is
    #: a stop signal here exactly as it is for labeling — there is nothing left
    #: to write results to and nobody waiting for them. With the default
    #: "continue" the export would run to completion against a deleted row.
    missing_row="cancelled",
    terminal_values=frozenset({"completed", "failed", "cancelled"}),
))

#: Faithfulness (rung 3) runs on a CIRCUIT, not on a discovery run — its own
#: lifecycle with its own status column. `pending|running|completed|failed`
#: were the documented values; `cancelled` was already written by the task's
#: `_FaithfulnessCancelled` handler, which until now could never fire because
#: the task passed `cancel_check=None`.
register(CancelScope(
    kind="circuit_faithfulness",
    model=_circuit,
    status_field="faithfulness_status",
    #: The circuit row long outlives any one faithfulness run, and `error_message`
    #: on a Circuit is not about the run. The manifest is the record.
    error_field=None,
    progress_field=None,
    completed_at_field=None,
))

#: THE THREE NATIVE-ENUM LIFECYCLES. `datasets.status`, `models.status` and
#: `dataset_tokenizations.status` are native Postgres enums with no CANCELLED
#: member, and `ALTER TYPE … ADD VALUE` is non-transactional. They carry
#: `cancel_requested_at` instead (migration f3c8a92b1e07), which is also the
#: better model: it separates "the operator asked" from "the job stopped".
#:
#: Their `terminal_values` are their OWN vocabularies, not the default one —
#: `ready` is this family's success state, and `error` its failure. Using the
#: default {completed, failed, cancelled} here would mean the guard never
#: considered any of these rows terminal, so a straggling progress write could
#: revive a finished download.
#:
#: WHAT THE STATUS ENDS UP AS. These three still finish at `error`, because the
#: enum offers nothing better and extending it is the thing being avoided. That
#: is not a regression — it is what they already did — and `cancel_requested_at`
#: is precisely what makes it survivable: a row with `status = error` AND a
#: non-null request is a stop, one without is a crash. Before this column the
#: two were the same row and no reader could tell them apart.
register(CancelScope(
    kind="dataset_download",
    model=_dataset,
    request_field="cancel_requested_at",
    cancelled_values=frozenset({"cancelled"}),
    terminal_values=frozenset({"ready", "error", "cancelled"}),
    started_at_field=None,
    completed_at_field=None,
))

register(CancelScope(
    kind="model_download",
    model=_model,
    request_field="cancel_requested_at",
    cancelled_values=frozenset({"cancelled"}),
    terminal_values=frozenset({"ready", "error", "cancelled"}),
    started_at_field=None,
    completed_at_field=None,
))

register(CancelScope(
    kind="dataset_tokenization",
    model=_tokenization,
    request_field="cancel_requested_at",
    cancelled_values=frozenset({"cancelled"}),
    terminal_values=frozenset({"ready", "error", "cancelled"}),
    started_at_field=None,
))

#: THE CIRCUITS ARC's THREE LIFECYCLES. All healthy already — this is the
#: Phase-5 shim, so the behaviour is unchanged and only the implementation is
#: shared. Their statuses are plain String(16) columns, and each stage has its
#: OWN column on the discovery run so a failed pass never corrupts a completed
#: earlier one.
register(CancelScope(
    kind="circuit_capture",
    model=_capture_run,
    error_field=None,
    progress_field=None,
    completed_at_field=None,
))

register(CancelScope(
    kind="circuit_discovery",
    model=_discovery_run,
    error_field=None,
    progress_field=None,
    completed_at_field=None,
))

register(CancelScope(
    kind="circuit_attribution",
    model=_discovery_run,
    status_field="attribution_status",
    error_field=None,
    progress_field=None,
    completed_at_field=None,
))

register(CancelScope(
    kind="circuit_validation",
    model=_discovery_run,
    status_field="validation_status",
    error_field=None,
    progress_field=None,
    completed_at_field=None,
))

#: PHASE 6 — the five lifecycles that had no cancel route at all. Their jobs
#: were startable and not stoppable: an operator could launch a faithfulness
#: pass or a feature-grouping run and then had no way to reach it short of
#: restarting the pod.
register(CancelScope(
    kind="circuit_calibration",
    model=_circuit,
    status_field="calibration_status",
    error_field=None,
    progress_field=None,
    completed_at_field=None,
))

register(CancelScope(
    kind="steering_record",
    model=_record_run,
    #: This table spells it `error`, not `error_message`. Declared rather than
    #: defaulted, because a wrong name here fails silently: `setattr` would
    #: happily create the attribute on the instance and never persist it.
    error_field="error",
    progress_field=None,
    completed_at_field=None,
))

register(CancelScope(
    kind="enhanced_labeling",
    model=_enhanced_labeling_job,
    progress_field=None,
))

register(CancelScope(
    kind="feature_grouping",
    model=_grouping_run,
    progress_field=None,
))

register(CancelScope(
    kind="training",
    model=_training,
    #: PAUSED is terminal FOR THE GUARD's purposes: a straggling progress write
    #: must not un-pause a job. Training keeps its own dict-return convention
    #: (Feature 21 — cancel must run finalize, not unwind); it adopts the
    #: registry and the guard only.
    terminal_values=frozenset({"completed", "failed", "cancelled", "paused"}),
    started_at_field="started_at",
))


# --------------------------------------------------------------------------
# Reading: the checker
# --------------------------------------------------------------------------

class CancelCheck:
    """Two surfaces on one object, so both existing conventions are drop-in.

    `check()` is the predicate the circuit and J-lens code inject as
    `cancel_check=`; `raise_if_cancelled()` is the labeling shape. Same state,
    same throttle, so a caller can mix them.
    """

    def __init__(
        self,
        scope: CancelScope,
        target_id: Any,
        db: Any = None,
        min_interval_s: Optional[float] = None,
    ) -> None:
        self._scope = scope
        self._target_id = target_id
        self._db = db
        self._min_interval_s = (
            scope.min_interval_s if min_interval_s is None else min_interval_s
        )
        self._calls = 0
        # SEEDED TO NOW, not 0.0. With a zero sentinel the elapsed-time test is
        # trivially true on the first call, which duplicates the explicit
        # `n == 0` branch below — and two mechanisms for one guarantee means
        # neither is load-bearing, so neither can be mutation-tested. Found
        # exactly that way: a mutation that broke the first-call rule left the
        # suite green because the sentinel silently covered for it.
        self._last_poll = time.monotonic()
        self._cancelled = False
        self._reason: Optional[str] = None

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def _should_poll(self) -> bool:
        # THE FIRST CALL ALWAYS POLLS. Throttling from zero skips the opening
        # checks, so a job cancelled before it started would run to its Nth
        # checkpoint before noticing.
        n = self._calls
        self._calls += 1
        if n == 0:
            return True
        return (time.monotonic() - self._last_poll) >= self._min_interval_s

    def poll_now(self) -> bool:
        """Ignore the throttle. For the boundary just before an expensive
        indivisible step, where being 2 seconds stale is the wrong trade."""
        return self._poll()

    def _poll(self) -> bool:
        if self._cancelled:
            return True
        self._last_poll = time.monotonic()
        scope = self._scope
        try:
            if self._db is not None:
                row = self._fetch(self._db)
            else:
                # A SHORT-LIVED SESSION PER POLL, by default. The circuit
                # convention polls on the caller's task session, so an aborted
                # transaction makes the POLL raise — the cancellation check
                # becomes the thing that breaks. A fresh session is immune to
                # the caller's transaction state.
                from .database import get_sync_db
                with get_sync_db() as db:
                    row = self._fetch(db)
        except Exception as exc:  # noqa: BLE001 - a failed poll must not kill the work
            logger.warning(
                "cancel poll failed for %s:%s: %s", scope.kind, self._target_id, exc
            )
            return False

        if row is None:
            if scope.missing_row == "cancelled":
                self._cancelled = True
                self._reason = "deleted"
                return True
            return False

        if scope.request_field is not None:
            if getattr(row, scope.request_field, None) is not None:
                self._cancelled = True
                self._reason = "cancelled"
                return True

        if _norm(getattr(row, scope.status_field, None)) in scope.cancelled_values:
            self._cancelled = True
            self._reason = "cancelled"
            return True
        return False

    def _fetch(self, db: Any) -> Any:
        model = self._scope.model()
        column = getattr(model, self._scope.id_field)
        # populate_existing() defeats the identity map: without it a long-lived
        # task session returns the row as it looked when the task started and
        # can never observe a write from the API process (MIS-E2E-057).
        return (
            db.query(model)
            .filter(column == self._target_id)
            .populate_existing()
            .first()
        )

    def __call__(self) -> bool:
        if self._cancelled:
            return True
        if not self._should_poll():
            return False
        return self._poll()

    def raise_if_cancelled(self, detail: str = "") -> None:
        if self():
            raise OperatorCancelled(
                self._scope.kind, self._target_id, self._reason or "cancelled", detail
            )


def cancel_checker(
    kind: str,
    target_id: Any,
    *,
    db: Any = None,
    min_interval_s: Optional[float] = None,
) -> CancelCheck:
    """A checker for the work loop.

    THROTTLED ON TIME, NOT CALL COUNT. Count throttles are wrong in both
    directions and this codebase proves it: the circuit convention's `% 5` over
    attribution batches can be twenty minutes of latency on a large model, while
    training's `% 25` over steps can be milliseconds. Each author guessed a
    number against their own unit cost and none of them travel.

    A time budget makes the caller's rule trivial: CALL THIS AT THE FINEST
    BOUNDARY AT WHICH YOU CAN CLEANLY ABANDON WORK — per token, per prompt, per
    batch. The throttle decides whether the call touches the database. At 2s a
    poll is ~0.05% overhead at any loop rate, and latency is bounded by
    `min_interval_s` plus one indivisible unit of work.

    THE COUNT THROTTLE IS GONE. `every=` existed only so the circuit and J-lens
    shims could keep their old semantics through the migration; both now use
    the time budget and nothing passes it.
    """
    return CancelCheck(get_scope(kind), target_id, db=db, min_interval_s=min_interval_s)


# --------------------------------------------------------------------------
# Writing: the guard, and the request
# --------------------------------------------------------------------------

def is_cancelled(kind: str, status: Any) -> bool:
    """Does this status mean the operator stopped it, in this scope's terms?

    For the handful of callers that must distinguish "cancelled" from the other
    terminal states — `mark_completed` is the one that matters, since a finished
    run must not overwrite a cancellation with COMPLETED. Goes through the
    registry rather than comparing to a literal, so the vocabulary stays in the
    one place it is written down.
    """
    return _norm(status) in get_scope(kind).cancelled_values


def guard_allows(
    kind: str,
    current_status: Any,
    incoming_status: Any = None,
    *,
    writes_progress: bool = False,
) -> bool:
    """Would a write of this shape be accepted onto a row in this state?

    THE ONE STATEMENT OF THE RULE, so that the writers which cannot borrow
    `record_progress`'s session mechanics still share its semantics rather than
    growing a fourth nearly-identical guard. Two such writers exist and neither
    is going away: `ExtractionService.update_extraction_status` is async and
    holds an `AsyncSession`, and `NeuronpediaTask.update_export_progress`
    already has the row open in a context manager it also emits from.

    THE RULE: a terminal row accepts only an error message, or a deliberate
    terminal -> terminal transition.

      * live row                            -> anything
      * terminal + non-terminal status      -> refused entirely; this is the
        case that loses a cancellation
      * terminal + terminal status          -> allowed, so `cleanup_orphaned_*`
        can still fail an abandoned row
      * terminal + no status, but progress  -> refused; a cancelled export must
        not go on announcing "packaging, 60%"
      * terminal + error_message only       -> allowed, so a stopping task can
        record WHERE it stopped
    """
    scope = get_scope(kind)
    current = _norm(current_status)
    if current not in scope.terminal_values:
        return True
    incoming = _norm(incoming_status)
    if incoming is not None:
        return incoming in scope.terminal_values
    return not writes_progress


def record_progress(
    kind: str,
    target_id: Any,
    *,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    error_message: Optional[str] = None,
    db: Any = None,
    **fields: Any,
) -> bool:
    """Write progress, refusing to resurrect a terminal row.

    THIS IS THE LOAD-BEARING HALF OF COOPERATIVE CANCELLATION. Cancellation is
    asynchronous by construction: the endpoint writes "cancelled" and the task
    notices at its next checkpoint. In between, the task is still reporting.
    Without this guard its next status write overwrites the cancellation and the
    request is silently lost — which is worse than having no cancel at all,
    because the operator was told it worked.

    Concretely: `services/extraction_db_service.update_progress` assigns status
    unconditionally and fires roughly every 10 samples, so a checker added to
    `extract_activations` without this would usually never see the flag.

    The rules, unchanged from the J-lens original because it already has the
    right shape:
      * terminal row + non-terminal status -> refuse, write NOTHING else, False
      * terminal -> terminal -> allow, so the janitors can still fail an
        abandoned row (`cleanup_orphaned_tasks`)
      * error_message-only onto a terminal row -> allow, so a cancelled task can
        record WHERE it stopped

    Returns whether a row was written. A caller that can retry needs to tell
    "no row yet" from "refused"; one that cannot simply ignores it.
    """
    scope = get_scope(kind)

    def _apply(session: Any) -> bool:
        model = scope.model()
        column = getattr(model, scope.id_field)
        row = (
            session.query(model)
            .filter(column == target_id)
            .populate_existing()
            .first()
        )
        if row is None:
            # LOUD, NOT SILENT. A progress write against a row that is gone
            # means the job is narrating into nothing — a deleted extraction
            # whose task is still holding the GPU. That warning was ignored 300
            # times in production once, which is why the task-start guard
            # exists too, but removing it entirely would take away the only
            # signal that the phantom job is running at all.
            logger.warning(
                "%s %s not found for progress update", scope.kind, target_id
            )
            return False

        current = _norm(getattr(row, scope.status_field, None))
        incoming = _norm(status)
        if not guard_allows(
            kind,
            current,
            status,
            writes_progress=progress is not None or bool(fields),
        ):
            logger.info(
                "Ignoring %s update for %s:%s — row is already %s",
                incoming or "progress", scope.kind, target_id, current,
            )
            return False

        now = datetime.now(timezone.utc)
        if status is not None:
            if (
                scope.started_at_field
                and incoming == "running"
                and getattr(row, scope.started_at_field, None) is None
            ):
                setattr(row, scope.started_at_field, now)
            if (
                scope.completed_at_field
                and incoming in scope.terminal_values
                and getattr(row, scope.completed_at_field, None) is None
            ):
                setattr(row, scope.completed_at_field, now)
            setattr(
                row,
                scope.status_field,
                _coerce_status(model, scope.status_field, status),
            )
        if progress is not None and scope.progress_field:
            # Clamped: a bar past 100% reads as a bug in the bar rather than in
            # whatever produced the number.
            setattr(row, scope.progress_field, max(0.0, min(100.0, float(progress))))
        if error_message is not None and scope.error_field:
            setattr(row, scope.error_field, str(error_message)[:2000])
        for key, value in fields.items():
            setattr(row, key, value)
        session.commit()
        return True

    try:
        if db is not None:
            return _apply(db)
        from .database import get_sync_db
        with get_sync_db() as session:
            return _apply(session)
    except Exception as exc:  # noqa: BLE001 - narration must not break the work
        logger.warning(
            "Could not record progress for %s:%s: %s", kind, target_id, exc
        )
        return False


@dataclass(frozen=True)
class CancelOutcome:
    requested: bool
    was_running: bool
    prior_status: Optional[str]
    detail: str


def request_cancel(
    kind: str,
    target_id: Any,
    *,
    reason: str = "cancelled by operator",
    celery_task_id: Optional[str] = None,
    db: Any = None,
) -> CancelOutcome:
    """Ask a job to stop. Writes the flag; the task notices at its checkpoint.

    ALSO ISSUES `revoke()`, ONCE, CENTRALLY — not as the mechanism, but for the
    one case it genuinely handles: a task that has not started never will.
    Doing it here means no endpoint has to remember it, and no endpoint can
    mistake it for the mechanism. `terminate=` is deliberately NOT passed: it
    only signals a pool child, solo has none, and passing it invites the reader
    to believe it stops a running task.
    """
    scope = get_scope(kind)

    def _apply(session: Any) -> CancelOutcome:
        model = scope.model()
        column = getattr(model, scope.id_field)
        row = (
            session.query(model)
            .filter(column == target_id)
            .populate_existing()
            .first()
        )
        if row is None:
            return CancelOutcome(False, False, None, "no such job")

        prior = _norm(getattr(row, scope.status_field, None))
        if prior in scope.terminal_values:
            return CancelOutcome(
                False, False, prior, f"already {prior}; nothing to cancel"
            )

        now = datetime.now(timezone.utc)
        if scope.request_field is not None:
            setattr(row, scope.request_field, now)
        else:
            setattr(
                row,
                scope.status_field,
                _coerce_status(model, scope.status_field, "cancelled"),
            )
            if scope.completed_at_field and getattr(row, scope.completed_at_field, None) is None:
                setattr(row, scope.completed_at_field, now)
        if scope.error_field:
            setattr(row, scope.error_field, str(reason)[:2000])
        session.commit()

        return CancelOutcome(
            True,
            prior not in (None, "queued", "pending"),
            prior,
            (
                "Cancellation requested. A running job stops at its next "
                "checkpoint, which is bounded by one indivisible unit of work."
                if prior not in (None, "queued", "pending")
                else "Job had not started; it will not run."
            ),
        )

    try:
        if db is not None:
            outcome = _apply(db)
        else:
            from .database import get_sync_db
            with get_sync_db() as session:
                outcome = _apply(session)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not request cancel for %s:%s: %s", kind, target_id, exc)
        return CancelOutcome(False, False, None, f"could not write the request: {exc}")

    if celery_task_id:
        try:
            from .celery_app import celery_app
            celery_app.control.revoke(celery_task_id)
        except Exception:  # noqa: BLE001 - the row is the channel that matters
            logger.warning(
                "revoke() failed for %s; the row still carries the request",
                celery_task_id,
            )
    return outcome


def clear_cancel_request(kind: str, target_id: Any, *, db: Any = None) -> bool:
    """Clear a stale request so a RETRY of this job can run.

    THE MISSING HALF OF `request_field`. The three native-enum lifecycles record
    the operator's request in `cancel_requested_at`, and nothing ever cleared
    it — so once a download had been cancelled, every retry of it read the old
    timestamp on its first tqdm tick and abandoned immediately. Cancel a
    download once and it could never be downloaded again.

    Verified before fixing: a checker over a row whose `cancel_requested_at` is
    a leftover returns True on tick one.

    Called at task START, not at cancel time: the flag must survive until the
    task that is running has seen it.
    """
    scope = get_scope(kind)
    if scope.request_field is None:
        return False

    def _apply(session: Any) -> bool:
        model = scope.model()
        column = getattr(model, scope.id_field)
        row = (
            session.query(model)
            .filter(column == target_id)
            .populate_existing()
            .first()
        )
        if row is None or getattr(row, scope.request_field, None) is None:
            return False
        setattr(row, scope.request_field, None)
        session.commit()
        logger.info(
            "Cleared a stale cancellation request on %s:%s before starting",
            kind, target_id,
        )
        return True

    try:
        if db is not None:
            return _apply(db)
        from .database import get_sync_db
        with get_sync_db() as session:
            return _apply(session)
    except Exception as exc:  # noqa: BLE001 - must not block the work
        logger.warning("Could not clear the cancel request for %s:%s: %s",
                       kind, target_id, exc)
        return False


def cooperative_cancel(kind: str):
    """Task-boundary decorator: turn OperatorCancelled into a canonical result.

    ONE return shape replaces the four in use today, so a caller never has to
    know which lifecycle it is reading:

        {"status": "cancelled", "scope": ..., "target_id": ..., "reason": ...,
         "detail": ...}

    The task RETURNING (rather than raising) is what acks the `acks_late`
    message — the property that matters, given a 12-hour visibility timeout.
    """
    def decorate(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except OperatorCancelled as cancelled:
                tid = cancelled.target_id
                logger.info(
                    "%s:%s stopped by operator (%s) %s",
                    cancelled.scope, tid, cancelled.reason, cancelled.detail,
                )
                # The row is ALREADY terminal — the endpoint set it, which is
                # how the task found out. Record only WHERE it stopped;
                # record_progress refuses to move a terminal row anyway.
                if cancelled.detail:
                    record_progress(cancelled.scope, tid, error_message=cancelled.detail)
                return {
                    "status": "cancelled",
                    "scope": cancelled.scope,
                    "target_id": tid,
                    "reason": cancelled.reason,
                    "detail": cancelled.detail,
                }

        # Read by the Shape-D registry test, which must locate tasks through the
        # imported object rather than a regex over source — a source-scraping
        # guard fails open, twice observed in this repo.
        wrapper.__cooperative_cancel_scope__ = kind
        return wrapper

    return decorate
