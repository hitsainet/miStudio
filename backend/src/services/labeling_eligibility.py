"""What still needs labeling, and what must never be labeled again.

This module holds ONE definition of eligibility, used by the coverage endpoint
and by the resume batch it hands back. Two definitions would drift, and the
drift would be silent in the expensive direction: coverage would report work
remaining that resume then declines to do, or — far worse — resume would redo
verdicts coverage called finished.

THE DISTINCTION THAT COSTS MONEY.

    label_status='succeeded'    a verdict exists            NEVER redo
    label_status='skipped'      deliberately left alone     NEVER redo
    label_status='pending'      the judge never ran         redo
    label_status='failed'       the judge ran and crashed   redo
    label_status='in_progress'  a job holds it right now    leave it alone

`succeeded` includes `uninterpretable`, which `_enforce_refusal` produces
deliberately when the fit ratio is below 0.5. It is the judge's honest answer
that a feature has no coherent pattern — an ADJUDICATION, not an absence.
Treating it as a gap turns resume into a relabel-everything button and
re-derives, at ~8 s/feature, verdicts already reached. On the L46 extraction
that is 196 features; on a full estate sweep it is the difference between a
resume that finishes and one that never does.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func, literal_column, or_, select
from sqlalchemy.dialects.postgresql import aggregate_order_by

from ..models.feature import Feature

#: Outcomes a resume must revisit. Everything else is either adjudicated or
#: claimed. Deliberately a frozenset of the literals rather than "not these",
#: so a NEW status is ineligible by default: an unknown outcome must not be
#: silently relabelled just because nobody taught this module about it.
RETRYABLE_STATUSES = frozenset({"pending", "failed"})

#: Outcomes that count as settled. `skipped` belongs here: an aqua feature was
#: deliberately left alone and is not outstanding work.
ADJUDICATED_STATUSES = frozenset({"succeeded", "skipped"})

#: Outcomes a JUDGE CHANGE can invalidate. Narrower than ADJUDICATED_STATUSES,
#: and the difference is load-bearing: a `skipped` feature has no fingerprint
#: because no judge was ever asked about it, so a naive "adjudicated and the
#: fingerprint does not match" test marks every aqua feature stale the moment a
#: template is edited — offering hand-verified labels up for relabelling, which
#: is the exact promise the aqua star makes to the user. Nothing a template
#: change does can invalidate a decision not to run the judge at all.
STALEABLE_STATUSES = frozenset({"succeeded"})

#: Attempts after which a feature stops being offered. A feature that fails for
#: a reason no retry can fix — a corrupt activation row, a token that breaks the
#: judge's parser — would otherwise be re-queued by every resume forever, and
#: because it is retried first it would crowd out work that can succeed.
DEFAULT_MAX_ATTEMPTS = 3


def eligibility_filter(
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    prompt_fingerprint: Optional[str] = None,
    judge_model: Optional[str] = None,
):
    """The predicate for "this feature still needs a labeling attempt".

    With `prompt_fingerprint`/`judge_model` supplied, adjudicated features whose
    verdict came from a DIFFERENT judge are eligible too. That is what makes
    improving a template self-servicing: edit it, and exactly the features it
    would now answer differently come back into scope, with no hand-written SQL
    and no "relabel everything".

    A NULL fingerprint is stale by construction — it is what the backfill writes
    over historical rows, and it means "we do not know which judge produced
    this". Honest, and re-adjudicable on request, but never by a routine resume:
    routine resume passes no fingerprint at all.
    """
    retryable = and_(
        Feature.label_status.in_(sorted(RETRYABLE_STATUSES)),
        Feature.label_attempts < max_attempts,
    )
    # BOTH, OR NEITHER — a one-sided predicate is a tautology.
    #
    # Compiled against PostgreSQL, `judge_model=None` with a fingerprint gives
    #     ... OR features.label_model IS NULL OR features.label_model IS NOT NULL
    # which is true for every row, so the "stale" arm selects the ENTIRE
    # staleable set. The same holds with the halves swapped.
    #
    # The React card already refuses to send half a predicate, and
    # `stale_count_query` already gates on both being present — but the REST
    # request schema declares the two independently, and the sweep now forwards
    # them, so a caller supplying one silently books the whole estate. On a
    # 53,088-feature extraction that is ~118 GPU-hours requested by omission.
    #
    # Refusing loudly rather than falling back to `retryable`: a caller that
    # sent one half meant to filter on something, and quietly ignoring it would
    # hand back a different answer than they asked for.
    if (prompt_fingerprint is None) != (judge_model is None):
        raise ValueError(
            "prompt_fingerprint and judge_model must be given together or not "
            "at all. A fingerprint alone treats every same-template/"
            "different-model verdict as fresh; a model alone cannot see a "
            "template edit. Either way the staleness arm degenerates to "
            "'every row' and the caller is offered the whole extraction."
        )
    if prompt_fingerprint is None and judge_model is None:
        return retryable

    stale = and_(
        Feature.label_status.in_(sorted(STALEABLE_STATUSES)),
        or_(
            Feature.label_prompt_fingerprint.is_(None),
            Feature.label_prompt_fingerprint != prompt_fingerprint,
            Feature.label_model.is_(None),
            Feature.label_model != judge_model,
        ),
    )
    return or_(retryable, stale)


def coverage_query(extraction_job_id: str):
    """Counts per `label_status` for one extraction, in one round trip."""
    return (
        select(Feature.label_status, func.count(Feature.id))
        .where(Feature.extraction_job_id == extraction_job_id)
        .group_by(Feature.label_status)
    )


def stale_count_query(
    extraction_job_id: str, *, prompt_fingerprint: str, judge_model: str
):
    """Adjudicated features whose verdict came from a different judge."""
    return (
        select(func.count(Feature.id))
        .where(Feature.extraction_job_id == extraction_job_id)
        .where(
            Feature.label_status.in_(sorted(STALEABLE_STATUSES)),
            or_(
                Feature.label_prompt_fingerprint.is_(None),
                Feature.label_prompt_fingerprint != prompt_fingerprint,
                Feature.label_model.is_(None),
                Feature.label_model != judge_model,
            ),
        )
    )


#: The placeholder `b8e4a2d0c517` wrote over historical failures whose reason was
#: discarded at a log line before per-feature capture existed.
#:
#: IT IS NOT A REASON, and the count below must not treat it as one. It did: the
#: live endpoint reported `failures_without_a_recorded_reason: 0` for an
#: extraction whose 14,560 failures every one predate capture. `c1f5b3e9a204`
#: nulls these rows, and this constant is the belt to that migration's braces —
#: a database between the two revisions, or one that has taken the downgrade,
#: still has them, and must still tell the truth about them.
#:
#: Matched on the prefix; the migration carries the same literal, because a
#: migration must not import application code that will have moved on by the
#: time someone replays it. `test_the_sentinel_literal_matches_the_migration`
#: keeps the two honest.
UNRECORDED_REASON_PREFIX = "(reason not recorded"


def has_no_actionable_reason():
    """Predicate: this failure tells an operator nothing they can act on."""
    return or_(
        Feature.label_error.is_(None),
        Feature.label_error == "",
        Feature.label_error.like(f"{UNRECORDED_REASON_PREFIX}%"),
    )


def unreported_failure_count_query(extraction_job_id: str):
    """Failures carrying no reason an operator can act on.

    Every failure written from now on carries a real one. A nonzero count here
    is therefore historical, and the coverage response says so rather than
    presenting these as diagnosable.
    """
    return (
        select(func.count(Feature.id))
        .where(Feature.extraction_job_id == extraction_job_id)
        .where(Feature.label_status == "failed")
        .where(has_no_actionable_reason())
    )


#: How many distinct failure reasons the breakdown reports before collapsing the
#: tail into `other`. Ten is enough to see the shape; an uncapped list on a bad
#: day is a thousand rows in a response body.
MAX_FAILURE_REASON_GROUPS = 10

#: Sample feature ids carried per reason, so an operator can open a real example
#: rather than take the count on trust.
FAILURE_REASON_SAMPLES = 3


def _reason_group():
    """The reason, normalised to its first colon-delimited segment.

    THE GROUPING RULE, AND WHY IT IS NOT `GROUP BY label_error`.

    Every reason is built as either `f"{type(e).__name__}: {e}"` or a fixed
    sentence, so the first segment is the exception class or the whole sentence,
    and the variable tail — host, port, feature id, retry count — is dropped.
    Grouping on the raw column instead produces ONE GROUP PER FEATURE the moment
    a reason carries an address in it, which is most of them:

        APIError: connection reset to 10.0.0.5:41288
        APIError: connection reset to 10.0.0.5:41291      <- a second "reason"

    That failure would look correct on a small sample and be useless on the
    14,560 that matter. `test_twelve_failure_paths_do_not_become_twelve_thousand
    _groups` is what actually holds it.
    """
    return func.split_part(Feature.label_error, ":", 1)


def failure_reason_query(extraction_job_id: str, *, limit: int = MAX_FAILURE_REASON_GROUPS):
    """Counts AND samples per normalised failure reason, commonest first.

    ONE query, not 1 + N.

    MEASURED ON THE LIVE L46 EXTRACTION (14,560 failures), and the honest result
    is not a clean win at today's data:

        groups   1 + N samples        this query
        1        ~24 ms               ~33 ms      <- slower
        3        ~42 ms               ~33 ms
        10       ~105 ms              ~33 ms

    Today L46 has exactly ONE group, because every failure predates error
    capture and shares the placeholder — so this is ~8 ms slower right now, on a
    ~113 ms endpoint. It is kept anyway, for two reasons that outlive today's
    data. The cost is FLAT in group count rather than linear, so it cannot
    degrade as failures start carrying real reasons — which is the whole point
    of having recorded them. And the 9 ms measured per samples query is
    in-database time; each was also a separate round trip through asyncpg, so
    the 1 + N figures above understate it.

    The grouping needs NO index on `label_error`: `ix_features_label_status`
    narrows to 14,560 rows before the aggregate runs, and the sort is 18 ms of
    the 33.

    A row whose reason is absent or a known placeholder groups under one explicit
    label rather than being dropped: on live data that is 14,560 undiagnosable
    failures, the most important row on the report.
    """
    classified = (
        select(
            Feature.id.label("id"),
            Feature.neuron_index.label("neuron_index"),
            func.coalesce(
                func.nullif(_reason_group(), ""),
                literal_column("'(no reason recorded)'"),
            ).label("reason"),
        )
        .where(Feature.extraction_job_id == extraction_job_id)
        .where(Feature.label_status == "failed")
        .cte("classified")
    )

    ranked = select(
        classified.c.reason,
        classified.c.id,
        func.row_number()
        .over(partition_by=classified.c.reason, order_by=classified.c.neuron_index.asc())
        .label("rn"),
        func.count().over(partition_by=classified.c.reason).label("reason_count"),
    ).cte("ranked")

    return (
        select(
            ranked.c.reason,
            ranked.c.reason_count,
            # Deterministic: two operators comparing notes see the same examples.
            func.array_agg(aggregate_order_by(ranked.c.id, ranked.c.rn.asc()))
            .filter(ranked.c.rn <= FAILURE_REASON_SAMPLES)
            .label("samples"),
        )
        .group_by(ranked.c.reason, ranked.c.reason_count)
        .order_by(ranked.c.reason_count.desc())
        .limit(limit)
    )

def eligible_count_query(
    extraction_job_id: str,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    prompt_fingerprint: Optional[str] = None,
    judge_model: Optional[str] = None,
):
    """How many features a resume would ACTUALLY take.

    Uses `eligibility_filter` — the same predicate as `resume_batch_query` — so
    the number on the button and the work a job takes cannot disagree.

    THEY DID DISAGREE. `summarise` derived `remaining` from status counts alone
    and claimed in its own docstring that a button "cannot offer N while a job
    labels a different N". It shared the STATUS constants but not the ATTEMPT
    CAP, so an extraction whose 39 failures had each been tried three times
    reported 39 remaining and produced a batch of zero: the UI said
    "Resume 0 of 39", and clicking it answered "Nothing left to label".
    """
    return (
        select(func.count(Feature.id))
        .where(Feature.extraction_job_id == extraction_job_id)
        .where(
            eligibility_filter(
                max_attempts=max_attempts,
                prompt_fingerprint=prompt_fingerprint,
                judge_model=judge_model,
            )
        )
    )


def exhausted_count_query(
    extraction_job_id: str, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS
):
    """Failures that have used up their retries.

    Reported separately rather than folded into `remaining` or hidden. These
    features genuinely still need a label and genuinely will not be attempted
    again without raising the cap — an operator has to be able to see both
    halves of that, or the arithmetic on screen looks broken.
    """
    return (
        select(func.count(Feature.id))
        .where(Feature.extraction_job_id == extraction_job_id)
        .where(Feature.label_status.in_(sorted(RETRYABLE_STATUSES)))
        .where(Feature.label_attempts >= max_attempts)
    )


def total_failure_count_query(extraction_job_id: str):
    """Every failure, so the breakdown can report an honest `other` remainder."""
    return (
        select(func.count(Feature.id))
        .where(Feature.extraction_job_id == extraction_job_id)
        .where(Feature.label_status == "failed")
    )


def resume_batch_query(
    extraction_job_id: str,
    *,
    limit: int,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    prompt_fingerprint: Optional[str] = None,
    judge_model: Optional[str] = None,
    only: Optional[str] = None,
):
    """The ids a resume should take next.

    Ordered by `label_attempts` then `neuron_index`: never-attempted features go
    first, so a batch cannot be filled entirely with features that have already
    failed twice while untouched ones wait behind them.

    `only` narrows to ONE outcome, which is what makes a sample worth taking.
    "Do the failures still fail?" and "does the judge work at all?" are different
    questions, and a mixed sample of 20 answers neither: 15 fresh successes and 5
    repeat failures reads as 75%, which is true of nothing.

    A value this function does not recognise RAISES. Falling through to the full
    eligible set would silently answer a different question than the one asked,
    and the caller would have no way to tell.
    """
    query = (
        select(Feature.id)
        .where(Feature.extraction_job_id == extraction_job_id)
        .where(
            eligibility_filter(
                max_attempts=max_attempts,
                prompt_fingerprint=prompt_fingerprint,
                judge_model=judge_model,
            )
        )
    )
    if only is not None:
        if only not in RETRYABLE_STATUSES:
            raise ValueError(
                f"cannot sample {only!r}: expected one of "
                f"{sorted(RETRYABLE_STATUSES)}. An adjudicated feature is not "
                "eligible work and must not be resampled."
            )
        query = query.where(Feature.label_status == only)
    return (
        query
        .order_by(Feature.label_attempts.asc(), Feature.neuron_index.asc())
        .limit(limit)
    )


def summarise(counts: Dict[str, int], *, eligible: Optional[int] = None) -> Dict[str, Any]:
    """Shape the raw per-status counts into the coverage response body.

    `outstanding` is every feature that still lacks a verdict. `remaining` is
    what a resume would ACTUALLY take, and is the number the UI puts on a
    button — so it must be measured with the same predicate the batch query
    uses, which is what `eligible` carries in.

    This function CANNOT compute `remaining` from counts alone, and an earlier
    version's docstring claimed otherwise. Status counts know nothing about the
    attempt cap, so an extraction whose failures had each been tried three
    times reported them as remaining and produced an empty batch: the button
    read "Resume 0 of 39" and clicking it said "Nothing left to label".
    """
    total = sum(counts.values())
    adjudicated = sum(counts.get(s, 0) for s in ADJUDICATED_STATUSES)
    outstanding = sum(counts.get(s, 0) for s in RETRYABLE_STATUSES)
    return {
        "total": total,
        "by_status": dict(counts),
        "adjudicated": adjudicated,
        #: Still lacking a verdict, whether or not a resume will take it.
        "outstanding": outstanding,
        #: What a resume takes. Falls back to `outstanding` only when no
        #: measurement was supplied, which no production caller does.
        "remaining": outstanding if eligible is None else eligible,
        "in_progress": counts.get("in_progress", 0),
        # Not `total - adjudicated`: that silently absorbs any status this
        # module has not been taught about, and would report a feature stuck in
        # an unknown state as finished.
        "unclassified": total - adjudicated - outstanding - counts.get("in_progress", 0),
    }
