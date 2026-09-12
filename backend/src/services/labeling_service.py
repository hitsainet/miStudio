"""
Feature labeling service for semantic labeling of SAE features.

This service manages semantic labeling of features extracted from SAE models.
Labeling is independent from extraction, allowing re-labeling without re-extraction.
"""

import logging
import os
import random
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import desc, select
from collections import defaultdict
import asyncio

from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.labeling_job import LabelingJob, LabelingStatus, LabelingMethod
from src.models.feature import Feature, LabelSource
from src.models.feature_activation import FeatureActivation
from src.core.config import settings
from src.core.encryption import decrypt_value, encrypt_value
from src.services.local_labeling_service import LocalLabelingService
from src.services.openai_labeling_service import OpenAILabelingService
from src.services.labeling_fingerprint import prompt_fingerprint
from src.services.labeling_eligibility import eligibility_filter
from src.services.labeling_judge_health import JudgeUnavailable, is_job_level_failure
from src.services.labeling_trial_service import TEMPLATE_CONFIG_KEYS
from src.workers.websocket_emitter import emit_labeling_progress, emit_labeling_result
from src.utils.token_filters import filter_token_stats
from src.utils.millm_utils import ensure_model_loaded

logger = logging.getLogger(__name__)


# ── which stored examples reach the judge ─────────────────────────────────────
#
# Every retrieval in this codebase was `ORDER BY max_activation DESC LIMIT
# max_examples`, so a feature's label was an inference from the extreme upper
# tail of its activation distribution — the top 10 of a stored top-100, or of a
# stored top-20 on the eb48 extraction. A feature that reads as "legal language"
# in its top decile may be "formal register" across its range, and nothing could
# show that.
#
# ONE DEFINITION, TWO CALLERS. The async and sync retrievals held byte-identical
# copies of this SQL. They have not drifted yet; the point is that they cannot.

#: Columns every strategy returns, so the row dict's shape never depends on the
#: sampling strategy. `rank` and `stored_n` are included under BOTH — a consumer
#: that behaved differently depending on the strategy would reintroduce exactly
#: the variable this change exists to isolate.
_EXAMPLE_COLUMNS = """
                fa.feature_id,
                fa.sample_index,
                fa.max_activation,
                fa.prefix_tokens,
                fa.prime_token,
                fa.suffix_tokens,
                fa.prime_activation_index,
                fa.activations,
                fa.tokens"""

_EXAMPLE_OUTPUT = """
                feature_id,
                sample_index,
                max_activation,
                prefix_tokens,
                prime_token,
                suffix_tokens,
                prime_activation_index,
                activations,
                tokens,
                rank,
                stored_n"""

#: The historical behaviour, unchanged apart from reporting `rank`/`stored_n`.
TOP_K_EXAMPLES_SQL = f"""
    WITH ranked_examples AS (
        SELECT {_EXAMPLE_COLUMNS},
            ROW_NUMBER() OVER (
                PARTITION BY fa.feature_id
                ORDER BY fa.max_activation DESC, fa.id ASC
            ) AS rank,
            COUNT(*) OVER (PARTITION BY fa.feature_id) AS stored_n
        FROM feature_activations fa
        WHERE fa.feature_id = ANY(:feature_ids)
    )
    SELECT {_EXAMPLE_OUTPUT}
    FROM ranked_examples
    WHERE rank <= :max_examples
    ORDER BY feature_id, rank;
"""

#: PROPORTIONAL BANDS over whatever is stored, taking the strongest row of each.
#:
#: The band is computed from each feature's OWN stored count, which is what
#: makes this safe on every extraction without a version branch:
#:   * 100 stored rows, K=10  -> ranks 1, 11, 21 ... 91
#:   * 20 stored rows,  K=10  -> ranks 1, 3, 5 ... 19
#:   * 4 stored rows,   K=10  -> all four, no error, no duplicates
#: There is no lookup of the extraction's retention policy and nothing to keep
#: in sync — a legacy top-K extraction simply stratifies over a narrower range.
#:
#: RANK 1 IS ALWAYS IN BAND 0 (integer floor of 0*K/n), so the peak example can
#: never be dropped and `max(returned max_activation)` still equals
#: `features.max_activation`. The percent-of-peak display depends on that.
#:
#: `:max_examples` is cast to int explicitly: the driver may bind it as numeric,
#: and numeric division would make every band fractional and collapse the
#: LEAST() clamp.
#:
#: `GREATEST(r.stored_n, 1)` IS UNREACHABLE, AND KEPT ANYWAY.
#: `COUNT(*) OVER (PARTITION BY feature_id)` is at least 1 within any partition
#: that exists, and a feature with no rows produces no partition — so the divisor
#: is never zero and no test can cover this branch. Removing it was tried and no
#: test went red, which is the honest reason it is documented rather than
#: asserted. It stays as insurance against a future restructuring (a LEFT JOIN
#: from `features`, say) that would make an empty partition reachable; a
#: division-by-zero there would take down every labeling run.
STRATIFIED_EXAMPLES_SQL = f"""
    WITH ranked_examples AS (
        SELECT {_EXAMPLE_COLUMNS},
            ROW_NUMBER() OVER (
                PARTITION BY fa.feature_id
                ORDER BY fa.max_activation DESC, fa.id ASC
            ) AS rank,
            COUNT(*) OVER (PARTITION BY fa.feature_id) AS stored_n
        FROM feature_activations fa
        WHERE fa.feature_id = ANY(:feature_ids)
    ),
    banded AS (
        SELECT r.*,
            LEAST(
                CAST(:max_examples AS int) - 1,
                ((r.rank - 1) * CAST(:max_examples AS int))
                    / GREATEST(r.stored_n, 1)
            ) AS band
        FROM ranked_examples r
    ),
    picked AS (
        SELECT b.*,
            ROW_NUMBER() OVER (
                PARTITION BY b.feature_id, b.band ORDER BY b.rank ASC
            ) AS within_band
        FROM banded b
    )
    SELECT {_EXAMPLE_OUTPUT}
    FROM picked
    WHERE within_band = 1
    ORDER BY feature_id, rank;
"""

#: Positive list. An unrecognised strategy FAILS CLOSED to the historical
#: behaviour rather than raising: a template carrying a value this build does
#: not know about must still be able to label, and silently labelling from the
#: top-K is the conservative reading. The log line is how anyone finds out.
_SAMPLING_SQL = {
    "top_k": TOP_K_EXAMPLES_SQL,
    "stratified": STRATIFIED_EXAMPLES_SQL,
}



#: The feature's WEAKEST retained activations, excluding anything already shown.
#:
#: NOT "negative examples", whatever the column is called. These are the
#: feature's own stored rows taken from the bottom of what was retained, so they
#: are weak POSITIVES — on L46 the smallest stored activation is 0.67, strictly
#: positive, and `labeling_detection_scorer`'s docstring records that with no
#: encode-on-text service nothing can certify a passage as non-activating.
#:
#: THE EXCLUSION IS LOAD-BEARING, not tidiness. Without it a feature with fewer
#: than `max_examples + num_negative_examples` stored rows shows the SAME
#: passage as both a strong example and a weak one in one prompt. Every feature
#: in the eb48 extraction stores exactly 20 rows, so at K=10 and N=5 that is not
#: an edge case — it is the common case.
#:
#: The exclusion is passed as two flat arrays and unnested, rather than a
#: per-feature loop: the caller batches 1000 feature ids at a time, and an N+1
#: here would turn one query into a thousand.
WEAKEST_EXAMPLES_SQL = f"""
    WITH excluded AS (
        SELECT * FROM unnest(
            CAST(:excl_feature_ids AS text[]),
            CAST(:excl_sample_indices AS int[])
        ) AS t(feature_id, sample_index)
    ),
    ranked_examples AS (
        SELECT {_EXAMPLE_COLUMNS},
            ROW_NUMBER() OVER (
                PARTITION BY fa.feature_id
                ORDER BY fa.max_activation ASC, fa.id ASC
            ) AS rank,
            COUNT(*) OVER (PARTITION BY fa.feature_id) AS stored_n
        FROM feature_activations fa
        WHERE fa.feature_id = ANY(:feature_ids)
          AND NOT EXISTS (
              SELECT 1 FROM excluded e
              WHERE e.feature_id = fa.feature_id
                AND e.sample_index = fa.sample_index
          )
    )
    SELECT {_EXAMPLE_OUTPUT}
    FROM ranked_examples
    WHERE rank <= :num_negative_examples
    ORDER BY feature_id, rank;
"""

#: NULL means "the documented default", not "none".
#:
#: `num_negative_examples` is nullable with no server default and three of the
#: five live templates leave it NULL, including the default one. Resolving it
#: with `or 0` would make `include_negative_examples=True` a silent no-op on the
#: template everything actually uses — the same shape of inert switch this whole
#: change exists to remove.
DEFAULT_NUM_NEGATIVE_EXAMPLES = 5


def resolve_num_negative(template_config: Dict[str, Any]) -> int:
    """How many weak examples this template asks for. One definition."""
    if not template_config.get("include_negative_examples"):
        return 0
    requested = template_config.get("num_negative_examples")
    if requested is None:
        return DEFAULT_NUM_NEGATIVE_EXAMPLES
    return max(0, int(requested))


def examples_sql(sampling: Optional[str]) -> str:
    """The retrieval SQL for one sampling strategy."""
    if sampling and sampling not in _SAMPLING_SQL:
        logger.warning(
            "unknown example_sampling %r; falling back to top_k", sampling
        )
    return _SAMPLING_SQL.get(sampling or "top_k", TOP_K_EXAMPLES_SQL)


#: Bands 3-9 of the stored set, for the mid-range scoring ruler.
#:
#: Deliberately NOT bands 0-9 (that is `STRATIFIED_EXAMPLES_SQL`): band 0 is the
#: peak, which is already the top-K ruler's territory, and the point of a second
#: ruler is that its passages are disjoint from the first's. Bands 3-9 sit below
#: anything a top-10 retrieval reaches on a 100-row feature while still being
#: strongly-activating text.
MIDRANGE_EXAMPLES_SQL = f"""
    WITH ranked_examples AS (
        SELECT {_EXAMPLE_COLUMNS},
            ROW_NUMBER() OVER (
                PARTITION BY fa.feature_id
                ORDER BY fa.max_activation DESC, fa.id ASC
            ) AS rank,
            COUNT(*) OVER (PARTITION BY fa.feature_id) AS stored_n
        FROM feature_activations fa
        WHERE fa.feature_id = ANY(:feature_ids)
    ),
    banded AS (
        SELECT r.*,
            LEAST(
                9,
                ((r.rank - 1) * 10) / GREATEST(r.stored_n, 1)
            ) AS band
        FROM ranked_examples r
    ),
    picked AS (
        SELECT b.*,
            ROW_NUMBER() OVER (
                PARTITION BY b.feature_id ORDER BY b.rank ASC
            ) AS within_range
        FROM banded b
        WHERE b.band >= 3
    )
    SELECT {_EXAMPLE_OUTPUT}
    FROM picked
    WHERE within_range <= :max_examples
    ORDER BY feature_id, rank;
"""


def assemble_batch_negatives(
    batch_features: List[Any],
    negatives_by_feature_id: Dict[str, List[Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    """The weak examples for one labeling batch, in the batch's own order.

    EXTRACTED SO IT CAN BE TESTED, and that is the entire point. This was two
    lines inline, and the regression test written for it re-implemented the same
    comprehension in the test file — so reverting the production code to a
    parallel list left the test green. A test that executes its own copy of the
    code under test is not a test.

    The lookup is by id, never by position: `features`, `features_examples` and
    `all_features_examples` are re-indexed by the junk filter, and a fourth
    parallel list would not be. After any junk drop, positional slicing hands
    one feature's passages to another feature's prompt — under a heading that
    says "same feature" — and the length check downstream cannot catch it,
    because the stale list is LONGER, so the slice comes back full-length.
    """
    return [
        negatives_by_feature_id.get(feature.id, []) for feature in batch_features
    ]


def example_row_to_dict(row) -> Dict[str, Any]:
    """One retrieved row, in the shape every consumer expects.

    `rank` and `stored_n` are carried so a caller can reason about WHERE in the
    feature's distribution an example sat. They must never reach the judge or
    the detection scorer — a rank is a direct answer leak, and `stored_n` is a
    retention count that would read as a corpus percentile.
    """
    return {
        "sample_index": row.sample_index,
        "max_activation": float(row.max_activation),
        "prefix_tokens": row.prefix_tokens or [],
        "prime_token": row.prime_token or "",
        "suffix_tokens": row.suffix_tokens or [],
        "prime_activation_index": row.prime_activation_index,
        "activations": row.activations or [],
        "tokens": row.tokens or [],  # legacy fallback
        "rank": row.rank,
        "stored_n": row.stored_n,
    }


def _column_default(key: str):
    """The declared default of one `labeling_prompt_templates` column.

    Used by the no-template fallback so it cannot disagree with the schema.
    Reads the ORM column's `default`, which is where the real answer lives —
    retyping the values here is how the fallback came to mark prime tokens with
    '>>>' while every seeded template used '<<>>'.
    """
    from src.models.labeling_prompt_template import LabelingPromptTemplate

    column = LabelingPromptTemplate.__table__.columns.get(key)
    if column is None or column.default is None:
        return None
    return getattr(column.default, "arg", None)




def create_example_tokens_summary(
    token_stats: Dict[str, Dict],
    filter_special: bool = True,
    filter_single_char: bool = True,
    filter_punctuation: bool = True,
    filter_numbers: bool = True,
    filter_fragments: bool = True,
    filter_stop_words: bool = False,
    top_n: int = 7
) -> Optional[Dict]:
    """
    Create example tokens summary from token statistics with filtering.

    Args:
        token_stats: Dict mapping token to {'count': N, 'total_activation': X}
        filter_*: Token filtering flags
        top_n: Number of top tokens to include (default 7)

    Returns:
        Dict with keys: 'tokens', 'counts', 'activations', 'max_activation'
        Returns None if no tokens remain after filtering
    """
    # Apply filters to token_stats
    filtered_stats = filter_token_stats(
        token_stats,
        filter_special=filter_special,
        filter_single_char=filter_single_char,
        filter_punctuation=filter_punctuation,
        filter_numbers=filter_numbers,
        filter_fragments=filter_fragments,
        filter_stop_words=filter_stop_words
    )

    if not filtered_stats:
        return None

    # Sort by count descending
    sorted_tokens = sorted(
        filtered_stats.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:top_n]

    # Extract tokens, counts, and average activations
    tokens = []
    counts = []
    activations = []

    for token, stats in sorted_tokens:
        tokens.append(token)
        counts.append(stats['count'])
        # Calculate average activation: total_activation / count
        avg_activation = stats['total_activation'] / stats['count'] if stats['count'] > 0 else 0.0
        activations.append(float(avg_activation))

    max_activation = max(activations) if activations else 0.0

    return {
        'tokens': tokens,
        'counts': counts,
        'activations': activations,
        'max_activation': float(max_activation)
    }


class LabelingService:
    """
    Service for semantic labeling of SAE features.

    Manages the feature labeling workflow:
    1. Create labeling job for an extraction
    2. Fetch features and their activations
    3. Aggregate token statistics for each feature
    4. Generate semantic labels using OpenAI or local LLM
    5. Update feature names and track labeling job
    6. Emit WebSocket progress events
    """

    def __init__(self, db: Union[AsyncSession, Session]):
        """Initialize labeling service with either async or sync session."""
        self.db = db
        self.is_async = isinstance(db, AsyncSession)

    async def start_labeling(
        self,
        extraction_job_id: str,
        config: Dict[str, Any]
    ) -> LabelingJob:
        """
        Start a feature labeling job for a completed extraction.

        Args:
            extraction_job_id: ID of the extraction to label features from
            config: Labeling configuration (labeling_method, openai_model, prompt_template_id, etc.)

        Returns:
            LabelingJob: Created labeling job record

        Raises:
            ValueError: If extraction not found, not completed, or active labeling exists
        """
        from sqlalchemy import func

        # Validate extraction exists and is completed
        result = await self.db.execute(
            select(ExtractionJob).where(ExtractionJob.id == extraction_job_id)
        )
        extraction_job = result.scalar_one_or_none()

        if not extraction_job:
            raise ValueError(f"Extraction job {extraction_job_id} not found")

        if extraction_job.status != ExtractionStatus.COMPLETED.value:
            raise ValueError(
                f"Extraction {extraction_job_id} must be completed before labeling "
                f"(current status: {extraction_job.status})"
            )

        # Check for active labeling on this extraction
        result = await self.db.execute(
            select(LabelingJob).where(
                LabelingJob.extraction_job_id == extraction_job_id,
                LabelingJob.status.in_([
                    LabelingStatus.QUEUED.value,
                    LabelingStatus.LABELING.value
                ])
            )
        )
        active_labeling = result.scalar_one_or_none()

        if active_labeling:
            raise ValueError(
                f"Extraction {extraction_job_id} already has an active labeling job: "
                f"{active_labeling.id}"
            )

        # Count features to label. When a panel is supplied the count MUST be
        # scoped too: an unscoped count leaves total_features as the whole
        # extraction, so progress crawls to 1% then jumps to 1.0 and every ETA
        # computed from the row is wrong by an order of magnitude.
        panel_ids = list(dict.fromkeys(config.get("feature_ids") or [])) or None
        count_q = select(func.count()).select_from(Feature).where(
            Feature.extraction_job_id == extraction_job_id
        )
        if panel_ids:
            count_q = count_q.where(Feature.id.in_(panel_ids))
        count_result = await self.db.execute(count_q)
        total_features = count_result.scalar_one()

        if total_features == 0:
            raise ValueError(f"Extraction {extraction_job_id} has no features to label")

        if panel_ids and total_features != len(panel_ids):
            # A shrunken panel is not the panel that was requested. Labelling a
            # subset silently would make two runs incomparable and any rate
            # computed from them wrong, so refuse and name the gap.
            raise ValueError(
                f"panel resolved to {total_features} of {len(panel_ids)} requested "
                f"features — the rest are absent from extraction {extraction_job_id}"
            )

        # Create labeling job ID: label_{extraction_id}_{timestamp}_{rand}
        #
        # The timestamp is second-resolution, so two starts within the same second
        # produced the SAME primary key and the insert below died with an opaque
        # IntegrityError->500. The active-job 409 masked this only while the first
        # job was still QUEUED/LABELING — a job that COMPLETED inside the same
        # second left the collision fully exposed. Nothing parses this id, so a
        # short random suffix is safe.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = f"label_{extraction_job_id}_{timestamp}_{uuid.uuid4().hex[:6]}"

        labeling_job = self.build_labeling_job_row(
            job_id=job_id,
            extraction_job_id=extraction_job_id,
            config=config,
            total_features=total_features,
            panel_ids=panel_ids,
        )

        self.db.add(labeling_job)
        await self.db.commit()
        await self.db.refresh(labeling_job)

        logger.info(
            f"Created labeling job {job_id} for extraction {extraction_job_id} "
            f"with {total_features} features using method: {labeling_job.labeling_method}"
        )

        return labeling_job

    async def _retrieve_top_examples_batch(
        self,
        session: AsyncSession,
        feature_ids: List[str],
        max_examples: int = 10,
        sampling: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve activation examples for a batch of features.

        `sampling` selects WHICH stored examples come back — see
        `examples_sql`. None means the historical top-K.
        """
        from sqlalchemy import text

        if not feature_ids:
            return {}

        result = await session.execute(
            text(examples_sql(sampling)),
            {"feature_ids": feature_ids, "max_examples": max_examples},
        )

        examples_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in result:
            examples_map.setdefault(row.feature_id, []).append(
                example_row_to_dict(row)
            )
        return examples_map

    def _retrieve_top_examples_batch_sync(
        self,
        session: Session,
        feature_ids: List[str],
        max_examples: int = 10,
        sampling: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Synchronous: retrieve activation examples for a batch of features.

        `sampling` selects WHICH stored examples come back — see
        `examples_sql`. None means the historical top-K.
        """
        from sqlalchemy import text

        if not feature_ids:
            return {}

        result = session.execute(
            text(examples_sql(sampling)),
            {"feature_ids": feature_ids, "max_examples": max_examples},
        )

        examples_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in result:
            examples_map.setdefault(row.feature_id, []).append(
                example_row_to_dict(row)
            )
        return examples_map

    def _retrieve_midrange_examples_batch_sync(
        self,
        session: Session,
        feature_ids: List[str],
        max_examples: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Passages from bands 3-9 — the mid-range scoring ruler.

        Pinned exactly as the top-K ruler is: no `sampling` parameter, because
        an arm must never be graded on its own sampling strategy.
        """
        from sqlalchemy import text

        if not feature_ids:
            return {}

        result = session.execute(
            text(MIDRANGE_EXAMPLES_SQL),
            {"feature_ids": feature_ids, "max_examples": max_examples},
        )
        examples_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in result:
            examples_map.setdefault(row.feature_id, []).append(
                example_row_to_dict(row)
            )
        return examples_map

    @staticmethod
    def _exclusion_arrays(
        exclude: Optional[Dict[str, List[int]]],
    ) -> Dict[str, List[Any]]:
        """Flatten a {feature_id: [sample_index]} map into two parallel arrays.

        Two scalar arrays keep the parameter binding simple and the query plan
        stable; a per-feature loop would be an N+1 over a 1000-feature batch.
        """
        feature_ids: List[str] = []
        sample_indices: List[int] = []
        for fid, indices in (exclude or {}).items():
            for index in indices:
                if index is None:
                    continue
                feature_ids.append(fid)
                sample_indices.append(int(index))
        return {
            "excl_feature_ids": feature_ids,
            "excl_sample_indices": sample_indices,
        }

    async def _retrieve_bottom_examples_batch(
        self,
        session: AsyncSession,
        feature_ids: List[str],
        num_negative_examples: int = 5,
        exclude_sample_indices_by_feature: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """The feature's WEAKEST retained activations — see WEAKEST_EXAMPLES_SQL.

        `exclude_sample_indices_by_feature` must carry whatever the prompt is
        already showing, or a feature with few stored rows will present the same
        passage twice under contradictory headings.
        """
        from sqlalchemy import text

        if not feature_ids or num_negative_examples <= 0:
            return {}

        params: Dict[str, Any] = {
            "feature_ids": feature_ids,
            "num_negative_examples": num_negative_examples,
        }
        params.update(self._exclusion_arrays(exclude_sample_indices_by_feature))

        result = await session.execute(text(WEAKEST_EXAMPLES_SQL), params)

        examples_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in result:
            examples_map.setdefault(row.feature_id, []).append(
                example_row_to_dict(row)
            )
        return examples_map

    def _retrieve_bottom_examples_batch_sync(
        self,
        session: Session,
        feature_ids: List[str],
        num_negative_examples: int = 5,
        exclude_sample_indices_by_feature: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Synchronous twin of `_retrieve_bottom_examples_batch`.

        This is the one the Celery worker calls.
        """
        from sqlalchemy import text

        if not feature_ids or num_negative_examples <= 0:
            return {}

        params: Dict[str, Any] = {
            "feature_ids": feature_ids,
            "num_negative_examples": num_negative_examples,
        }
        params.update(self._exclusion_arrays(exclude_sample_indices_by_feature))

        result = session.execute(text(WEAKEST_EXAMPLES_SQL), params)

        examples_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in result:
            examples_map.setdefault(row.feature_id, []).append(
                example_row_to_dict(row)
            )
        return examples_map

    #: PERMANENT ALIAS, not a subclass. `_LabelingCancelled` is caught by name
    #: in `workers/labeling_tasks.py` and asserted by name in two behavioural
    #: test files; pointing it at `OperatorCancelled` keeps every one of those
    #: working while upgrading it from `Exception` to `BaseException`.
    #:
    #: That upgrade IS the MIS-E2E-058 fix, generalised: labeling's own outer
    #: `except Exception` used to catch this and write FAILED, turning an
    #: operator's deliberate stop into a crash report.
    from ..core.cancellation import OperatorCancelled as _LabelingCancelled

    def _raise_if_cancelled(self, labeling_job_id: str) -> None:
        """Cooperative cancellation check — call once per batch.

        NOW A SHIM over `core.cancellation`. The reasoning that used to be
        spelled out here — solo pool, revoke signals a child that does not
        exist, the main process never services control messages while a task
        runs — lives once in that module's docstring.

        TWO THINGS THE SHIM FIXES FOR FREE.

        `populate_existing()` was already right here (MIS-E2E-057) and is now
        the shared default rather than one service's hard-won knowledge.

        And A DELETED ROW IS NOW A STOP. This returned silently when the row
        was gone — but `delete_labeling_job` DELETES THE ROW as its stop
        signal, so the job ran to completion against a row that no longer
        existed, writing results nobody could read. The `labeling` scope
        carries `missing_row="cancelled"` for exactly that.
        """
        from ..core.cancellation import cancel_checker

        # A FRESH CHECKER PER CALL, WHICH THEREFORE ALWAYS POLLS.
        #
        # `CancelCheck`'s first call always polls, so constructing one here is
        # the same thing the old implementation did: query once per batch. An
        # earlier version of this shim cached the checker per job so the
        # 2-second budget would throttle — and that was a behaviour change
        # dressed as a refactor, because a fast batch loop then ran its whole
        # length inside one window and never re-polled. The time budget exists
        # for per-token loops; a per-batch caller is already at the right
        # granularity, and `min_interval_s` here would be untestable
        # redundancy on top of the first-call rule.
        cancel_checker(
            "labeling", labeling_job_id, db=self.db
        ).raise_if_cancelled(f"labeling job {labeling_job_id}")

    def _label_batch(
        self,
        labeling_service,
        loop,
        batch_features,
        batch_examples,
        batch_all_examples,
        feature_logit_effects,
        template_config,
        user_prompt_template,
        system_message,
        batch_negatives=None,
    ):
        """Label one batch of features, batched through miLLM when enabled.

        Returns one label per feature in order. Never raises for a single
        feature: the batched client falls back to serial on any batch failure,
        and the serial path returns error labels, so the caller's
        isinstance(label, Exception) guard stays valid either way.

        BULK LABELING ONLY. Batch composition changes greedy output under int8
        quantisation, so a labeling trial must not come through here — and does
        not: LabelingTrialService calls generate_label_from_examples directly.
        """
        requests = []
        # `batch_negatives` is optional so every existing caller keeps working,
        # but it must line up with the other two lists when present — the zip
        # below is positional, and a short list would silently pair one
        # feature's weak examples with another feature's prompt.
        negatives = batch_negatives or [[]] * len(batch_features)
        if len(negatives) != len(batch_features):
            raise ValueError(
                f"batch_negatives has {len(negatives)} entries for "
                f"{len(batch_features)} features; a positional mismatch would "
                f"attach one feature's weak examples to another's prompt"
            )
        for feature, examples, all_ex, negs in zip(
            batch_features, batch_examples, batch_all_examples, negatives
        ):
            requests.append({
                "examples": examples,
                "template_config": template_config,
                "user_prompt_template": user_prompt_template,
                "system_message": system_message,
                "feature_id": feature.id,
                "neuron_index": feature.neuron_index,
                "logit_effects": feature_logit_effects.get(feature.id),
                "all_examples": all_ex,
                "nlp_analysis": feature.nlp_analysis,
                "negative_examples": negs,
            })

        batch_size = getattr(settings, "labeling_batch_size", 1) or 1
        can_batch = batch_size > 1 and hasattr(
            labeling_service, "generate_labels_from_examples_batched"
        )

        if can_batch:
            return loop.run_until_complete(
                labeling_service.generate_labels_from_examples_batched(
                    requests, batch_size=batch_size
                )
            )

        # Per-feature requests, concurrent within the shared loop.
        return loop.run_until_complete(
            asyncio.gather(
                *[
                    labeling_service.generate_label_from_examples(**req)
                    for req in requests
                ],
                return_exceptions=True,
            )
        )

    @staticmethod
    def build_labeling_job_row(
        *,
        job_id: str,
        extraction_job_id: str,
        config: Dict[str, Any],
        total_features: int,
        panel_ids: Optional[List[str]] = None,
    ) -> LabelingJob:
        """The ONE place a labeling job row is shaped.

        `start_labeling` is async and `resume_sweep_step` is a sync Celery task,
        so both need this and neither can call the other. A second construction
        site would drift — and the drift would be invisible, because a job with a
        subtly different config still runs and still produces labels. That is how
        `retryLabeling` came to drop the endpoint, the template and every filter
        flag while looking like it worked.
        """
        return LabelingJob(
            id=job_id,
            extraction_job_id=extraction_job_id,
            labeling_method=config.get("labeling_method", "openai"),
            openai_model=config.get("openai_model"),
            openai_api_key=(
                encrypt_value(config["openai_api_key"])
                if config.get("openai_api_key")
                else None
            ),
            openai_compatible_endpoint=config.get("openai_compatible_endpoint"),
            openai_compatible_model=config.get("openai_compatible_model"),
            local_model=config.get("local_model"),
            prompt_template_id=config.get("prompt_template_id"),
            filter_special=config.get("filter_special", True),
            filter_single_char=config.get("filter_single_char", True),
            filter_punctuation=config.get("filter_punctuation", True),
            filter_numbers=config.get("filter_numbers", True),
            filter_fragments=config.get("filter_fragments", True),
            filter_stop_words=config.get("filter_stop_words", False),
            save_requests_for_testing=config.get("save_requests_for_testing", False),
            export_format=config.get("export_format", "both"),
            save_poor_quality_labels=config.get("save_poor_quality_labels", False),
            poor_quality_sample_rate=config.get("poor_quality_sample_rate", 1.0),
            max_tokens=config.get("max_tokens", 300),
            api_timeout=config.get("api_timeout", 120.0),
            # Defaults FALSE — a full relabel, exactly as this has always
            # behaved. Only an explicit request changes that.
            skip_adjudicated=config.get("skip_adjudicated", False),
            status=LabelingStatus.QUEUED.value,
            progress=0.0,
            features_labeled=0,
            total_features=total_features,
            statistics={
                "max_examples": config.get("max_examples"),
                "batch_size": config.get("batch_size", 10),
            },
            # A real column, not a statistics key: the completion write replaces
            # `statistics` wholesale, which would erase the panel at the moment
            # the run finished and take reproducibility with it.
            feature_ids=panel_ids,
        )

    def _persist_filtered_out(
        self, features: List[Feature], labeled_at: datetime
    ) -> int:
        """Record features the pre-labeling junk filter removed.

        `skipped` and not `pending`, because the judge was deliberately not
        asked — the same distinction the aqua star gets. `skipped` counts as
        adjudicated, so a resume stops offering them and a sweep stops
        re-selecting them.

        Leaving them `pending` is what R4 caught: they were eligible forever,
        the attempt cap never engaged because no attempt was recorded, and a
        sweep burned batches labelling one feature in five with nothing on the
        row to say why.
        """
        if not features:
            return 0
        for feature in features:
            # An aqua feature is already `skipped` for a better reason; do not
            # overwrite the record of a hand-verified label being protected.
            if feature.star_color == 'aqua':
                continue
            feature.label_status = self.LABEL_STATUS_SKIPPED
            feature.label_error = (
                "skipped before labeling: the prime tokens are predominantly "
                "punctuation, whitespace or single non-alphanumeric characters, "
                "so there is no text for a judge to interpret"
            )
            feature.label_error_at = labeled_at
            feature.updated_at = labeled_at
        self.db.commit()
        logger.info(
            "Recorded %d features as skipped by the pre-labeling junk filter",
            len(features),
        )
        return len(features)

    def _preflight_judge(self, labeling_job: LabelingJob) -> None:
        """Refuse to start when the judge is not there to be asked.

        One HTTP call before any feature is touched, against the endpoint the
        job will actually use. Without it, a job configured for a model that no
        longer exists discovers this once PER FEATURE — 39 identical 404s in
        fourteen seconds, each one spending a retry.

        This is a courtesy, not the guarantee: `_persist_label_outcome` still
        refuses to blame a feature for a judge-level failure, which is what
        covers a model that disappears MID-RUN. A preflight alone would not.

        Deliberately soft on its own failure. If the check itself cannot run —
        no endpoint configured, a transport error reaching the model list — it
        returns and lets the job proceed. A preflight that blocks work because
        it could not confirm anything is worse than no preflight.
        """
        endpoint = labeling_job.openai_compatible_endpoint
        wanted = labeling_job.openai_compatible_model
        if not endpoint or not wanted:
            return

        import httpx

        try:
            response = httpx.get(f"{endpoint.rstrip('/')}/models", timeout=20.0)
            response.raise_for_status()
            served = {m.get("id") for m in (response.json().get("data") or [])}
        except Exception as exc:  # noqa: BLE001 — see the docstring
            logger.warning(
                "Judge preflight could not reach %s (%s); starting anyway",
                endpoint, exc,
            )
            return

        if served and wanted not in served:
            raise JudgeUnavailable(
                f"NotFoundError: the judge '{wanted}' is not served by {endpoint}. "
                f"Available: {', '.join(sorted(served)[:6])}"
                + ("…" if len(served) > 6 else "")
                + ". Nothing was labeled and no retries were spent."
            )

    def _claim_features(self, features: List[Feature], labeled_at: datetime) -> int:
        """Mark these features as being worked on, and return how many were taken.

        WHY CLAIMING EXISTS. `label_status='in_progress'` is excluded from
        eligibility, so a second resume started while the first is running skips
        whatever the first already holds. Without this write the exclusion is
        decorative: nothing would ever be `in_progress`, both jobs would compute
        the same batch from the same `pending` rows, and the estate would pay
        twice — at ~8 s a feature — for one result.

        An aqua feature is NOT claimed. It is going to be skipped, and moving it
        through `in_progress` would make it briefly indistinguishable from work
        in flight to anything reading coverage.

        This is a claim, not a lock. Two jobs racing on the same row can still
        both read `pending` before either writes; the commit here narrows that to
        the width of one batch rather than a whole run, which is the difference
        between a rare duplicated feature and a wholly duplicated job.

        A row left `in_progress` by a worker that died is released by
        `cleanup_stuck_labeling`, which marks it `failed` with a reason.

        THAT SENTENCE USED TO BE FALSE. It claimed the sweeper already reclaimed
        such rows; it did not — it failed the JOB and never touched the features,
        so a claimed feature stayed `in_progress` forever and was invisible to
        every resume. R4 caught it by restarting a pod mid-panel and finding 5 of
        15 stranded. An unverified sentence in a docstring is how a capability
        gets believed into existence.
        """
        claimed = 0
        for feature in features:
            if feature.star_color == 'aqua':
                continue
            feature.label_status = self.LABEL_STATUS_IN_PROGRESS
            feature.updated_at = labeled_at
            claimed += 1
        self.db.commit()
        return claimed

    # ── the single writer of a labeling outcome ──────────────────────────────
    #
    # There were THREE near-identical copies of this loop (the local, OpenAI and
    # OpenAI-compatible paths), with the aqua-skip guard triplicated alongside
    # them. Adding six columns to three places is how they drift, and the drift
    # would be invisible: two of the three copies already differed from the
    # first, silently, by an exception handler.
    #
    # It is also where the estate's 16,824 phantom-complete features were
    # written. A judge failure arrived here as a dict shaped exactly like a
    # verdict — category='error_feature', name='feature_{n}' — and this loop
    # stamped `label_source` and `labeled_at` over it like any other label. The
    # exception text was logged upstream and discarded. Nothing downstream could
    # tell the two apart afterwards, and nothing ever read 'error_feature'.

    #: Statuses this writer may record. `skipped` is a real outcome, not a
    #: non-event: an aqua feature was deliberately left alone and must not be
    #: counted as labeled, nor offered to a later resume.
    LABEL_STATUS_SUCCEEDED = "succeeded"
    LABEL_STATUS_IN_PROGRESS = "in_progress"
    LABEL_STATUS_FAILED = "failed"
    LABEL_STATUS_SKIPPED = "skipped"

    #: `label_error` is TEXT, but an unbounded provider traceback in a column
    #: read by a list endpoint is its own problem. Matches cancellation's
    #: `record_progress`.
    MAX_LABEL_ERROR_CHARS = 2000

    @staticmethod
    def failure_reason(label: Any) -> Optional[str]:
        """The reason this label is a failure, or None if it is a verdict.

        Keyed on an explicit `error` field, never on `category`. Category is the
        judge's SEMANTIC answer, and overloading it is what made a crash
        indistinguishable from a verdict — one local path is worse still and
        reports a parse failure as `category='semantic'`, which no string test
        could ever separate from a real semantic label.
        """
        if isinstance(label, BaseException):
            return f"{type(label).__name__}: {label}"
        if isinstance(label, dict):
            reason = label.get("error")
            if reason:
                return str(reason)
        if not isinstance(label, dict) or not label.get("specific"):
            return "judge returned no label"
        return None

    def _persist_label_outcome(
        self,
        feature: Feature,
        label: Any,
        examples: List[Dict[str, Any]],
        *,
        labeling_job: LabelingJob,
        labeled_at: datetime,
        label_source_value: str,
        prompt_fingerprint: Optional[str],
        judge_model: Optional[str],
    ) -> str:
        """Record what happened to one feature, and return the status recorded.

        The ONLY place a labeling outcome reaches the `features` table. Callers
        pass what the judge returned, whatever shape it came back in — including
        a raised exception — and this decides what that means.
        """
        # Never overwrite features completed by enhanced labeling. The aqua star
        # is a user-visible promise that a hand-verified label survives a bulk
        # run.
        if feature.star_color == 'aqua':
            logger.debug(
                "Skipping feature %s (star_color=aqua, enhanced labeling result preserved)",
                feature.id,
            )
            feature.label_status = self.LABEL_STATUS_SKIPPED
            feature.updated_at = labeled_at
            return self.LABEL_STATUS_SKIPPED

        feature.label_attempts = (feature.label_attempts or 0) + 1

        reason = self.failure_reason(label)

        # A JUDGE THAT CANNOT ANSWER IS NOT THIS FEATURE'S FAILURE.
        #
        # A missing model, refused credentials or an unreachable endpoint says
        # nothing about the feature — so it must not spend the feature's retry
        # budget or leave a reason on its row. Raising here aborts the JOB
        # instead, and the attempt increment above is undone.
        #
        # Observed live: a resume against a judge deleted months earlier took 39
        # features from one attempt to three in fourteen seconds of 404s, then
        # refused to retry them because they had "used up" retries they never
        # received.
        if reason is not None and is_job_level_failure(reason):
            feature.label_attempts = (feature.label_attempts or 0) - 1
            raise JudgeUnavailable(reason)

        if reason is not None:
            # A failure writes NO name, NO category and NO `labeled_at`. The
            # feature keeps whatever it had — usually its auto-generated
            # placeholder — and is truthfully still unlabeled.
            logger.error(
                "Labeling failed for feature %s (attempt %s): %s",
                feature.id, feature.label_attempts, reason,
            )
            feature.label_status = self.LABEL_STATUS_FAILED
            feature.label_error = reason[: self.MAX_LABEL_ERROR_CHARS]
            feature.label_error_at = labeled_at
            feature.labeling_job_id = labeling_job.id
            feature.updated_at = labeled_at
            return self.LABEL_STATUS_FAILED

        feature.category = label["category"]
        feature.name = label["specific"]
        feature.description = label.get("description", "")
        feature.label_source = label_source_value
        feature.labeling_job_id = labeling_job.id
        feature.labeled_at = labeled_at
        feature.updated_at = labeled_at

        feature.label_status = self.LABEL_STATUS_SUCCEEDED
        feature.label_error = None
        feature.label_error_at = None
        # WHICH judge reached this verdict. Without both, a later resume can
        # only ask whether a verdict exists, not whether it is still the verdict
        # this template and this model would produce.
        feature.label_prompt_fingerprint = prompt_fingerprint
        feature.label_model = judge_model

        # Quick preview of top prime tokens, for visual scanning in the UI.
        prime_tokens = [
            ex.get('prime_token', '') for ex in examples[:7] if ex.get('prime_token')
        ]
        feature.example_tokens_summary = ', '.join(prime_tokens) if prime_tokens else ''

        # Emit individual result for real-time display: first 10 full examples
        # with prefix/prime/suffix context.
        example_data = [
            {
                "prefix_tokens": ex.get('prefix_tokens', []),
                "prime_token": ex.get('prime_token', ''),
                "suffix_tokens": ex.get('suffix_tokens', []),
                "max_activation": ex.get('max_activation', 0.0),
            }
            for ex in examples[:10]
        ]
        emit_labeling_result(
            labeling_job_id=labeling_job.id,
            feature_data={
                "feature_id": feature.neuron_index,
                "label": feature.name,
                "category": feature.category,
                "description": feature.description or "",
                "examples": example_data,
            },
        )
        return self.LABEL_STATUS_SUCCEEDED

    def label_features_for_extraction(
        self,
        labeling_job_id: str
    ) -> Dict[str, Any]:
        """
        Execute semantic labeling for features from an extraction job.

        This is the core labeling logic that:
        1. Fetches features and their activations
        2. Aggregates token statistics for each feature (using efficient SQL batching)
        3. Generates semantic labels using specified method
        4. Updates feature names and tracks progress
        5. Calculates statistics and marks job complete

        Args:
            labeling_job_id: ID of the labeling job to execute

        Returns:
            Dict with labeling statistics

        Raises:
            ValueError: If labeling job not found or extraction invalid
        """
        # This method uses sync SQLAlchemy Session (.query() calls throughout).
        # It must be called from a Celery worker that injects a sync session,
        # not from an async FastAPI endpoint that uses AsyncSession.
        assert isinstance(self.db, Session), (
            "label_features_for_extraction requires a sync SQLAlchemy Session; "
            f"got {type(self.db).__name__}. Call from a Celery worker, not an async endpoint."
        )

        # Fetch labeling job
        labeling_job = self.db.query(LabelingJob).filter(
            LabelingJob.id == labeling_job_id
        ).first()

        if not labeling_job:
            raise ValueError(f"Labeling job {labeling_job_id} not found")

        # Validate extraction job and features BEFORE transitioning to LABELING.
        # This prevents the job being stuck in LABELING if the extraction is missing.
        extraction_job = self.db.query(ExtractionJob).filter(
            ExtractionJob.id == labeling_job.extraction_job_id
        ).first()
        if not extraction_job:
            raise ValueError(f"Extraction job {labeling_job.extraction_job_id} not found")

        # Chained .filter() rather than .limit()/.join(): the strict Mock in
        # tests/unit/test_labeling_service.py stubs only filter/order_by/all,
        # so keeping this shape means those guards keep guarding.
        _q = self.db.query(Feature).filter(
            Feature.extraction_job_id == labeling_job.extraction_job_id
        )
        # Only a real list of ids counts as a panel. A malformed column (or a
        # test double) is treated as "no panel" rather than being handed to
        # in_(), which raises an opaque ArgumentError deep in SQLAlchemy.
        _panel = labeling_job.feature_ids
        if isinstance(_panel, (list, tuple)) and _panel and all(
            isinstance(i, str) for i in _panel
        ):
            # Panel run. The extraction predicate stays, so a foreign id cannot
            # pull in a feature from a different extraction.
            _q = _q.filter(Feature.id.in_(list(_panel)))

        # Fill the gaps rather than redo the work — but ONLY when asked.
        #
        # The default is false and must stay false: this is the existing Label
        # button's query, and an operator pressing it expects the full relabel
        # it has always done. Applying the filter unconditionally would repurpose
        # a control silently, which is the one thing worse than not having the
        # option at all.
        #
        # Shares `eligibility_filter` with coverage and resume, so the count the
        # UI shows and the work this job takes cannot disagree.
        if getattr(labeling_job, "skip_adjudicated", False):
            _q = _q.filter(eligibility_filter())

        all_features = _q.order_by(Feature.neuron_index).all()
        if not all_features:
            raise ValueError(f"No features found for extraction {labeling_job.extraction_job_id}")

        # Before any feature is touched: is the judge actually there?
        self._preflight_judge(labeling_job)

        # Validation passed — safe to transition to LABELING
        labeling_job.status = LabelingStatus.LABELING.value
        labeling_job.updated_at = datetime.now(timezone.utc)
        self.db.commit()

        start_time = datetime.now(timezone.utc)

        # Counted from what was actually WRITTEN to each feature. The previous
        # count tested `specific.startswith('feature_')`, so a judge that
        # crashed and a judge that legitimately named a feature
        # 'feature_store_checkout' scored identically — and every one of the
        # 16,824 failed features in the estate was named exactly that way.
        # Bound HERE, before the method dispatch, so a labeling_method matching
        # no branch reports zero rather than raising NameError in the summary.
        outcome_counts = {
            self.LABEL_STATUS_SUCCEEDED: 0,
            self.LABEL_STATUS_FAILED: 0,
            self.LABEL_STATUS_SKIPPED: 0,
        }

        try:
            # NOTE: Batch commits throughout this method mean that if labeling fails
            # mid-run, some features will already be committed as labeled while others
            # are not. The job status is set to FAILED below, but the partial labels
            # remain. A full transactional rollback would require buffering all writes
            # until completion — deferred as a future improvement (high memory cost).

            # All features are retrieved for examples; filtering happens after retrieval
            # using prime tokens from activation examples (context-based approach).
            features = all_features

            total_features = len(features)
            logger.info(f"Labeling {total_features} features for extraction {labeling_job.extraction_job_id}")

            # Fetch template configuration - use specified template or fall back to DB default
            template_config = None
            max_examples = 10  # Default for miStudio Internal
            from src.models.labeling_prompt_template import LabelingPromptTemplate

            template = None
            if labeling_job.prompt_template_id:
                template = self.db.query(LabelingPromptTemplate).filter(
                    LabelingPromptTemplate.id == labeling_job.prompt_template_id
                ).first()
            else:
                # No template specified - look up the default template from DB
                template = self.db.query(LabelingPromptTemplate).filter(
                    LabelingPromptTemplate.is_default == True  # noqa: E712
                ).first()
                if template:
                    logger.info(f"No template specified in job - using DB default: {template.name}")

            # REFUSE A SCORING TEMPLATE AT JOB START, not once per feature.
            #
            # A detection template has no examples block to render, so every
            # feature in the run would fail individually and the job would
            # report thousands of per-feature errors for one job-level mistake.
            # That is the distinction this codebase already draws elsewhere: a
            # job-level fault says nothing about any feature and must not spend
            # a feature's retry budget.
            #
            # `LabelingTrialService.start_trial` refuses these the same way
            # (labeling_trial_service.py:168-172); the bulk path simply never
            # learned to.
            # GATE ON EITHER SIGNAL, because they are independent columns.
            #
            # The renderer refuses on `template_type == 'eleutherai_detection'`;
            # this gated only on `is_detection_template`. A template setting one
            # and not the other passed here and then raised per feature — and
            # `NotImplementedError` is a RuntimeError subclass, so the caller's
            # broad `except Exception` swallowed it into an error label. The
            # refusal was "loud" only for templates that happened to set both.
            _detection_type = (
                (getattr(template, "template_type", "") or "").strip().lower()
                == "eleutherai_detection"
            )
            if template is not None and (
                template.is_detection_template or _detection_type
            ):
                raise ValueError(
                    f"template {template.id} ({template.name!r}) is a "
                    f"detection/scoring template, not a labeling template. It "
                    f"scores an explanation that already exists and cannot "
                    f"produce one."
                )

            # BOUND BEFORE THE BRANCH (MIS-E2E-059).
            #
            # `job_batch_size` was assigned only inside `if template:` and read
            # unconditionally at three later points, so the explicitly-supported
            # "no template found" path died with UnboundLocalError — a labeling
            # run against a deleted template crashed with a Python error instead
            # of falling back, and surfaced as a generic 500 / FAILED job.
            job_max_examples = None
            job_batch_size = 10
            if labeling_job.statistics and isinstance(labeling_job.statistics, dict):
                job_max_examples = labeling_job.statistics.get('max_examples')
                job_batch_size = labeling_job.statistics.get('batch_size', 10)

            if template:
                # Check for job-level overrides in statistics
                if labeling_job.statistics and isinstance(labeling_job.statistics, dict):
                    job_max_examples = labeling_job.statistics.get('max_examples')
                    job_batch_size = labeling_job.statistics.get('batch_size', 10)

                # Use job override if provided, otherwise use template default
                max_examples = job_max_examples if job_max_examples is not None else template.max_examples

                # DERIVED FROM THE SHARED KEY LIST, not hand-written here.
                #
                # This was the third of three copies of the renderer's field
                # list, and it had already drifted: it omitted
                # `include_negative_examples` and `num_negative_examples`
                # entirely, so the bulk path could not see them however a
                # template was configured. The trial path carried them, which
                # means a trial did not predict the bulk run it was trialling.
                template_config = {
                    key: getattr(template, key, None)
                    for key in TEMPLATE_CONFIG_KEYS
                }
                # `max_examples` is the one field a JOB may override.
                template_config['max_examples'] = max_examples

                override_msg = f" (job override)" if job_max_examples is not None else ""
                logger.info(f"Using template: {template.name} (type: {template.template_type}, K={max_examples}{override_msg})")

            # Provide hardcoded fallback if no template found in DB at all
            if template_config is None:
                # THE COLUMN DEFAULTS, read off the model rather than retyped.
                #
                # A hand-typed fallback is a fourth copy of the field list with
                # nobody watching it, and it had already disagreed with the
                # seeded templates on `prime_token_marker` ('>>>' here versus
                # '<<>>' everywhere else) — so a run that fell back marked the
                # prime token differently from every other run, invisibly.
                template_config = {
                    key: _column_default(key) for key in TEMPLATE_CONFIG_KEYS
                }
                template_config['max_examples'] = max_examples
                logger.warning(f"No template found (specified or default) - using hardcoded fallback (K={max_examples})")

            # Retrieve activation examples using efficient SQL batching
            # For NLP analysis, we need ALL examples (up to 100)
            # For display in LLM prompt, we only show top max_examples (default 10)
            BATCH_SIZE = 1000
            NLP_ANALYSIS_EXAMPLES = 100
            #: The junk verdict reads the feature's ENTIRE STORED SET.
            #:
            #: Not the display set, and not a fixed slice of it either. Pinning
            #: to a constant K only moved the one-way door: pre-arc the verdict
            #: came from whatever the template displayed (10 for the active
            #: default, 25 and 50 for the seeded Anthropic ones), so a fixed 10
            #: would permanently retire features that the K=25 template had
            #: labelled the day before — reproduced against the real filter on a
            #: register feature whose top-10 primes are 9/10 punctuation and
            #: whose top-25 are 9/25.
            #:
            #: The decision writes `label_status='skipped'`, which is in
            #: ADJUDICATED_STATUSES, is excluded from STALEABLE_STATUSES, carries
            #: no fingerprint, and is recoverable only by hand-written SQL. A
            #: verdict that permanent must not depend on ANY prompt setting.
            #:
            #: The full stored set is the only basis with that property: it is
            #: fixed by extraction, cannot be moved by editing a template, and is
            #: the largest evidence base available — so the ratio is the best
            #: estimate of the feature's true junk rate that exists.
            #:
            #: This IS a behaviour change on the estate, in the direction of
            #: more evidence, and it is the last one this basis can undergo.
            JUNK_VERDICT_EXAMPLES = NLP_ANALYSIS_EXAMPLES
            include_nlp = template_config.get('include_nlp_analysis', False)
            retrieval_count = NLP_ANALYSIS_EXAMPLES if include_nlp else max_examples
            features_examples = []  # Top max_examples for LLM display
            all_features_examples = []  # All examples for NLP analysis (empty when NLP disabled)
            #: Weak contrast examples KEYED BY FEATURE ID. Never a parallel
            #: list — see the comment at its population site.
            negatives_by_feature_id: Dict[str, List[Dict[str, Any]]] = {}

            # Phase 1: Examples Retrieval with progress tracking
            logger.info(f"Starting examples retrieval phase for {total_features} features (display K={max_examples}, NLP={'enabled (K='+str(retrieval_count)+')' if include_nlp else 'disabled'}) in batches of {BATCH_SIZE}")

            for batch_start in range(0, total_features, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_features)
                batch_features = features[batch_start:batch_end]
                batch_size = len(batch_features)

                logger.info(f"Retrieving batch {batch_start//BATCH_SIZE + 1}/{(total_features + BATCH_SIZE - 1)//BATCH_SIZE}: features {batch_start+1}-{batch_end}")

                # Get feature IDs for this batch
                batch_feature_ids = [f.id for f in batch_features]

                # TWO RETRIEVALS WHEN NLP IS ON, because they answer two
                # different questions.
                #
                # The judge is shown `max_examples` rows, chosen by the
                # template's sampling strategy. The NLP summariser reads the
                # feature's full stored set, always in activation order.
                #
                # Collapsing these into one retrieval of 100-then-slice-10 broke
                # BOTH capabilities at once, and silently:
                #   * stratification became a no-op — banding 100 rows into 100
                #     bands returns ranks 1..100 in order, and taking the first
                #     ten is exactly top-K. Two arms of an A/B trial would have
                #     produced identical prompts while reporting distinct
                #     fingerprints.
                #   * the contrast pool was fully excluded — every stored row
                #     had been retrieved, so excluding "what was retrieved" left
                #     the weak query nothing to return.
                display_map = self._retrieve_top_examples_batch_sync(
                    session=self.db,
                    feature_ids=batch_feature_ids,
                    max_examples=max_examples,
                    sampling=template_config.get('example_sampling'),
                )
                all_examples_map: Dict[str, List[Dict[str, Any]]] = {}
                if include_nlp:
                    all_examples_map = self._retrieve_top_examples_batch_sync(
                        session=self.db,
                        feature_ids=batch_feature_ids,
                        max_examples=retrieval_count,
                        sampling=None,  # the summariser wants the real order
                    )

                # THE WEAK CONTRAST BLOCK.
                #
                # `include_negative_examples` defaults True on every template in
                # the estate and had NO effect: `_retrieve_bottom_examples_batch`
                # existed, was tested, and had zero callers, so the switch was
                # inert everywhere it was set.
                #
                # The exclusion is what the judge is SHOWN, not everything
                # retrieved. Excluding the NLP set would remove the entire donor
                # pool, since retention is at most 100 rows per feature.
                n_negative = resolve_num_negative(template_config)
                negatives_map: Dict[str, List[Dict[str, Any]]] = {}
                # No positives means nothing to contrast against, and the query
                # would return rows for features whose examples were all
                # filtered out — a prompt made entirely of weak examples, which
                # is not what the contrast block is for.
                if n_negative and display_map:
                    negatives_map = self._retrieve_bottom_examples_batch_sync(
                        session=self.db,
                        feature_ids=batch_feature_ids,
                        num_negative_examples=n_negative,
                        # `.get`, not `[...]`: an example dict that lacks
                        # `sample_index` cannot be excluded, and failing the
                        # whole labeling run over an un-excludable row would
                        # trade a duplicate passage for no labels at all.
                        # `_exclusion_arrays` drops the Nones.
                        exclude_sample_indices_by_feature={
                            fid: [e.get("sample_index") for e in rows]
                            for fid, rows in display_map.items()
                        },
                    )

                # KEYED BY FEATURE ID, NOT A PARALLEL LIST.
                #
                # `features`, `features_examples` and `all_features_examples`
                # are re-indexed by the junk filter below
                # (`filter_features_from_examples` returns SHORTENED lists), and
                # a fourth parallel list would not be. Every later slice would
                # then pair one feature's weak examples with another feature's
                # prompt — under a heading that says "same feature" — and the
                # length guard in `_label_batch` would NOT catch it, because the
                # stale list is longer, not shorter.
                #
                # A dict cannot desynchronise. It is looked up by id at the
                # point of use.
                for feature in batch_features:
                    all_examples = all_examples_map.get(feature.id, [])
                    negatives_by_feature_id[feature.id] = negatives_map.get(
                        feature.id, []
                    )
                    llm_examples = list(display_map.get(feature.id, []))
                    # Shuffle before display to break primacy bias.
                    random.shuffle(llm_examples)
                    features_examples.append(llm_examples)
                    all_features_examples.append(all_examples if include_nlp else [])

                # Update progress in database
                retrieval_progress = batch_end / total_features
                labeling_job.progress = retrieval_progress * 0.3  # Retrieval is ~30% of total work
                labeling_job.updated_at = datetime.now(timezone.utc)
                self.db.commit()

                # Emit WebSocket progress update
                emit_labeling_progress(
                    labeling_job_id=labeling_job.id,
                    event="labeling:progress",
                    data={
                        "labeling_job_id": labeling_job.id,
                        "extraction_job_id": labeling_job.extraction_job_id,
                        "progress": labeling_job.progress,
                        "features_labeled": 0,
                        "total_features": total_features,
                        "status": "labeling",
                        "phase": "examples_retrieval",
                        "message": f"Retrieved top-{max_examples} examples for {batch_end}/{total_features} features"
                    }
                )

                logger.info(f"Batch {batch_start//BATCH_SIZE + 1} complete: {batch_end}/{total_features} features processed ({retrieval_progress*100:.1f}%)")

            logger.info(f"Examples retrieval complete for {len(features_examples)} features (K={max_examples})")

            # Apply context-based pre-labeling filter: skip features whose prime tokens
            # are predominantly junk (punctuation, whitespace, single non-alphanumeric chars).
            from src.utils.token_filter import get_feature_filter
            feature_filter = get_feature_filter()
            labeled_at_for_skips = datetime.now(timezone.utc)
            _before_filter = list(features)
            # THE JUNK VERDICT IS TAKEN ON A PINNED TOP-K SET.
            #
            # `features_examples` is the display set, so judging on it let
            # `example_sampling` decide which features are PERMANENTLY retired
            # — `skipped` is never redone and carries no fingerprint. Retrieved
            # separately, unsampled, so the verdict is a property of the feature
            # rather than of the arm.
            junk_verdict_map = self._retrieve_top_examples_batch_sync(
                session=self.db,
                feature_ids=[f.id for f in features],
                max_examples=JUNK_VERDICT_EXAMPLES,
                sampling=None,
            )
            features, features_examples, all_features_examples, filter_stats = (
                feature_filter.filter_features_from_examples(
                    features,
                    features_examples,
                    all_features_examples,
                    verdict_examples=[
                        junk_verdict_map.get(f.id, []) for f in features
                    ],
                )
            )
            # RECORD WHAT THE FILTER DROPPED.
            #
            # Found by the R4 hardware round. A junk-filtered feature was left
            # `pending` with `label_attempts` still 0, so it stayed eligible
            # FOREVER: every resume offered it again, every sweep batch
            # re-selected it, and the attempt cap could never engage because no
            # attempt was ever recorded. On the L46 extraction a sweep batch of 5
            # labelled 1 and silently dropped 4, twice in a row, and would have
            # gone on doing that for its whole ceiling.
            #
            # This is the same shape as the aqua skip: a deliberate decision NOT
            # to ask the judge, which is an outcome and has to be written down.
            # AN ID SET, NOT `f not in features`.
            #
            # `in` over a list of ORM objects is O(N) identity comparison per
            # element, so at 53,088 features that was ~1.4 billion comparisons
            # of pure Python, inside the retrieval phase, before a single label
            # was requested.
            _kept_ids = {id(f) for f in features}
            self._persist_filtered_out(
                [f for f in _before_filter if id(f) not in _kept_ids], labeled_at_for_skips
            )
            total_features = len(features)
            logger.info(
                f"Pre-labeling filter: {filter_stats['features_to_label']}/{filter_stats['total_features']} "
                f"features pass ({filter_stats['features_skipped']} skipped as junk, "
                f"{filter_stats['skip_percentage']:.1f}%)"
            )

            # A filter that removed EVERYTHING is a failure, not a completed job.
            #
            # total_features is 0 here, so every `range(0, total_features, ...)`
            # label loop below is a no-op, `labels` stays empty, and the terminal
            # write records COMPLETED / progress=1.0 / features_labeled=0 with
            # avg_label_length=0 — a silent success that looks like a finished run
            # and labeled nothing. Raising converts it into a FAILED job carrying
            # the reason. The smaller the working set the likelier this is, so a
            # scoped trial panel is the case that most needs it.
            if total_features == 0:
                raise ValueError(
                    f"pre-labeling junk filter removed all "
                    f"{filter_stats['total_features']} features; nothing to label"
                )

            # Phase 2: Label Generation with progress tracking
            logger.info("Starting label generation phase")

            # Define progress callback for label generation
            def labeling_progress_callback(current: int, total: int):
                """
                Callback for label generation progress.
                Updates database and emits WebSocket events.

                Args:
                    current: Number of features labeled so far
                    total: Total number of features to label
                """
                # Calculate progress: aggregation was 0-30%, labeling is 30-100%
                labeling_progress = current / total if total > 0 else 0
                overall_progress = 0.3 + (labeling_progress * 0.7)

                # Update database
                labeling_job.progress = overall_progress
                labeling_job.features_labeled = current
                labeling_job.updated_at = datetime.now(timezone.utc)
                self.db.commit()

                # Emit WebSocket progress
                emit_labeling_progress(
                    labeling_job_id=labeling_job.id,
                    event="labeling:progress",
                    data={
                        "labeling_job_id": labeling_job.id,
                        "extraction_job_id": labeling_job.extraction_job_id,
                        "progress": overall_progress,
                        "features_labeled": current,
                        "total_features": total_features,
                        "status": "labeling",
                        "phase": "labeling",
                        "message": f"Generated labels for {current}/{total_features} features"
                    }
                )

            # Initialize appropriate labeling service
            labeling_method = labeling_job.labeling_method
            labels = []

            try:
                if labeling_method == LabelingMethod.LOCAL.value:
                    local_model = labeling_job.local_model or "meta-llama/Llama-3.2-1B"
                    logger.info(f"Initializing local labeling service with model: {local_model}")
                    labeling_service = LocalLabelingService(model_name=local_model)

                    # Load model once for the entire job
                    labeling_service.load_model()

                    try:
                        # Generate and persist labels in batches using context examples
                        # This ensures progress is saved incrementally if the job fails
                        label_source_value = LabelSource.LOCAL_LLM.value
                        labeled_at = datetime.now(timezone.utc)
                        # WHICH judge this job's verdicts came from. Computed once:
                        # it is a property of the job, not of a feature.
                        label_fingerprint = prompt_fingerprint(template)
                        judge_model = local_model
                        LABEL_BATCH_SIZE = job_batch_size

                        logger.info(f"Starting incremental labeling: {total_features} features in batches of {LABEL_BATCH_SIZE}")

                        for batch_start in range(0, total_features, LABEL_BATCH_SIZE):
                            # Stop promptly when the user cancels (see
                            # _raise_if_cancelled: revoke cannot kill a solo-pool task).
                            self._raise_if_cancelled(labeling_job_id)
                            batch_end = min(batch_start + LABEL_BATCH_SIZE, total_features)
                            batch_features = features[batch_start:batch_end]

                            # Claim before asking the judge. `in_progress` is excluded from
                            # eligibility, so a second resume started mid-run skips what this
                            # one already holds; without the write, that exclusion is
                            # decorative and both jobs pay for the same features.
                            self._claim_features(batch_features, labeled_at)
                            batch_examples = features_examples[batch_start:batch_end]
                            batch_all_examples = all_features_examples[batch_start:batch_end]
                            # NO CONTRAST BLOCK ON THE LOCAL PATH, stated rather
                            # than half-wired.
                            #
                            # `LocalLabelingService.generate_label` takes no
                            # `negative_examples` — it does not take a
                            # `template_config` at all — so this branch cannot
                            # render the weak-example block however a template
                            # is configured.
                            #
                            # A `batch_negatives` computed here and never passed
                            # would read as if it were wired, which is exactly
                            # the shape of `token_positions`: built on every
                            # example, stored nowhere, and mistaken for a
                            # feature for months. The gap is recorded in the
                            # arc notes instead.

                            # Generate labels for this batch (model already loaded)
                            # LOCAL service uses synchronous generation, not async
                            # Pass all examples for NLP analysis to improve labeling
                            batch_labels = []
                            for feature, examples, all_examples in zip(batch_features, batch_examples, batch_all_examples):
                                label = labeling_service.generate_label(
                                    examples=examples,
                                    neuron_index=feature.neuron_index,
                                    feature_id=feature.id,
                                    all_examples=all_examples,  # Pass full 100 examples for NLP analysis
                                    nlp_analysis=feature.nlp_analysis  # Use pre-computed NLP if available
                                )
                                batch_labels.append(label)

                            # Persist this batch immediately
                            for feature, label, examples in zip(batch_features, batch_labels, batch_examples):
                                # ONE writer. See `_persist_label_outcome`: this loop used to be
                                # copy-pasted three times, and it is where a judge failure was
                                # stamped into `features` as though it were a verdict.
                                outcome = self._persist_label_outcome(
                                    feature,
                                    label,
                                    examples,
                                    labeling_job=labeling_job,
                                    labeled_at=labeled_at,
                                    label_source_value=label_source_value,
                                    prompt_fingerprint=label_fingerprint,
                                    judge_model=judge_model,
                                )
                                outcome_counts[outcome] += 1

                            # Commit this batch
                            self.db.commit()

                            # Update progress
                            current_labeled = batch_end
                            labeling_progress_callback(current_labeled, total_features)

                            logger.info(f"Batch {batch_start//LABEL_BATCH_SIZE + 1}/{(total_features + LABEL_BATCH_SIZE - 1)//LABEL_BATCH_SIZE}: Labeled and persisted features {batch_start+1}-{batch_end}/{total_features}")

                        logger.info(f"All {total_features} features labeled and persisted successfully")

                        # Create labels list for statistics calculation (now we need to query back from DB)
                        labels = [{"category": f.category, "specific": f.name} for f in features]

                    finally:
                        # Always unload model to free GPU memory
                        logger.info("Unloading local labeling model from GPU memory")
                        labeling_service.unload_model()

                elif labeling_method == LabelingMethod.OPENAI.value:
                    # Decrypt key stored in labeling_job (encrypt_value on write, decrypt here).
                    # decrypt_value() gracefully handles legacy plaintext rows.
                    # Fallback chain: labeling_job → DB app_settings → env.
                    openai_api_key = None
                    if labeling_job.openai_api_key:
                        openai_api_key = decrypt_value(labeling_job.openai_api_key, setting_key="openai_api_key")

                    if not openai_api_key:
                        logger.warning("No API key in labeling job, checking DB app_settings")
                        try:
                            from src.models.app_setting import AppSetting
                            db_setting = self.db.query(AppSetting).filter(AppSetting.key == "openai_api_key").first()
                            if db_setting:
                                openai_api_key = decrypt_value(db_setting.value, setting_key="openai_api_key")
                                logger.info("Using OpenAI API key from DB app_settings")
                        except Exception as e:
                            logger.warning(f"Failed to read API key from DB app_settings: {e}")

                    if not openai_api_key:
                        logger.warning("Falling back to OPENAI_API_KEY environment variable")
                        openai_api_key = getattr(settings, 'openai_api_key', None)

                    if not openai_api_key:
                        raise ValueError("OpenAI API key not provided and not found in settings")

                    openai_model = labeling_job.openai_model or "gpt-4o-mini"

                    # Fetch prompt template - use specified or fall back to DB default
                    system_message = None
                    user_prompt_template = None
                    temperature = 0.3
                    max_tokens = labeling_job.max_tokens or 300
                    top_p = 0.9

                    from src.models.labeling_prompt_template import LabelingPromptTemplate
                    template = None
                    if labeling_job.prompt_template_id:
                        template = self.db.query(LabelingPromptTemplate).filter(
                            LabelingPromptTemplate.id == labeling_job.prompt_template_id
                        ).first()
                    else:
                        template = self.db.query(LabelingPromptTemplate).filter(
                            LabelingPromptTemplate.is_default == True  # noqa: E712
                        ).first()
                        if template:
                            logger.info(f"No template in job - using DB default: {template.name}")

                    if template:
                        system_message = template.system_message
                        user_prompt_template = template.user_prompt_template
                        temperature = template.temperature
                        # THE JOB'S VALUE WINS (MIS-E2E-060).
                        #
                        # `max_tokens` is exposed on the API and in the UI as a
                        # per-job setting and was then unconditionally replaced
                        # by the template's (default 50). A user raising it to
                        # get longer descriptions had the value accepted and
                        # every description still truncated — a control that
                        # appears to work and does nothing.
                        #
                        # The sibling `max_examples` already gets this
                        # precedence right; this is the same rule.
                        if labeling_job.max_tokens:
                            max_tokens = labeling_job.max_tokens
                        else:
                            max_tokens = template.max_tokens
                        top_p = template.top_p
                        logger.info(f"Using prompt template: {template.name} (ID: {template.id})")

                    api_timeout = labeling_job.api_timeout

                    # Generate and persist labels in batches of 10
                    # This ensures progress is saved incrementally if the job fails
                    label_source_value = LabelSource.OPENAI.value
                    labeled_at = datetime.now(timezone.utc)
                    label_fingerprint = prompt_fingerprint(template)
                    judge_model = openai_model
                    LABEL_BATCH_SIZE = job_batch_size

                    logger.info(f"Starting incremental labeling: {total_features} features in batches of {LABEL_BATCH_SIZE}")

                    # Create event loop BEFORE the OpenAI service so that httpx
                    # AsyncClient and asyncio.Semaphore bind to this loop.
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        logger.info(f"Initializing OpenAI labeling service with model: {openai_model}")
                        labeling_service = OpenAILabelingService(
                            api_key=openai_api_key,
                            model=openai_model,
                            system_message=system_message,
                            user_prompt_template=user_prompt_template,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            timeout=api_timeout,
                            filter_special=labeling_job.filter_special,
                            filter_single_char=labeling_job.filter_single_char,
                            filter_punctuation=labeling_job.filter_punctuation,
                            filter_numbers=labeling_job.filter_numbers,
                            filter_fragments=labeling_job.filter_fragments,
                            filter_stop_words=labeling_job.filter_stop_words,
                            save_requests_for_testing=labeling_job.save_requests_for_testing,
                            export_format=labeling_job.export_format,
                            save_poor_quality_labels=labeling_job.save_poor_quality_labels,
                            poor_quality_sample_rate=labeling_job.poor_quality_sample_rate,
                            save_requests_sample_rate=labeling_job.save_requests_sample_rate,
                            labeling_job_id=labeling_job.id
                        )

                        # Pre-load logit effects for all features (one bulk query, avoids N+1)
                        feature_logit_effects: Dict[str, Optional[Dict]] = {}
                        if template_config.get('include_logit_effects'):
                            from src.models.feature_dashboard import FeatureDashboardData
                            n_promoted = template_config.get('top_promoted_tokens_count', 10)
                            n_suppressed = template_config.get('top_suppressed_tokens_count', 10)
                            feature_ids = [f.id for f in features]
                            dashboard_rows = self.db.query(
                                FeatureDashboardData.feature_id,
                                FeatureDashboardData.logit_lens_data
                            ).filter(
                                FeatureDashboardData.feature_id.in_(feature_ids)
                            ).all()
                            for row in dashboard_rows:
                                lens = row.logit_lens_data or {}
                                top_pos = lens.get('top_positive', [])[:n_promoted]
                                top_neg = lens.get('top_negative', [])[:n_suppressed]
                                feature_logit_effects[row.feature_id] = {
                                    'top_promoted': [t['token'] for t in top_pos],
                                    'top_suppressed': [t['token'] for t in top_neg],
                                }
                            logger.info(f"Pre-loaded logit effects for {len(feature_logit_effects)}/{len(feature_ids)} features")

                        for batch_start in range(0, total_features, LABEL_BATCH_SIZE):
                            # Stop promptly when the user cancels (see
                            # _raise_if_cancelled: revoke cannot kill a solo-pool task).
                            self._raise_if_cancelled(labeling_job_id)
                            batch_end = min(batch_start + LABEL_BATCH_SIZE, total_features)
                            batch_features = features[batch_start:batch_end]

                            # Claim before asking the judge. `in_progress` is excluded from
                            # eligibility, so a second resume started mid-run skips what this
                            # one already holds; without the write, that exclusion is
                            # decorative and both jobs pay for the same features.
                            self._claim_features(batch_features, labeled_at)
                            batch_examples = features_examples[batch_start:batch_end]
                            batch_all_examples = all_features_examples[batch_start:batch_end]
                            batch_negatives = assemble_batch_negatives(
                                batch_features, negatives_by_feature_id
                            )

                            # Generate labels for this batch using context-based examples
                            # Create concurrent tasks for all features in batch
                            # Pass all examples for NLP analysis to improve labeling
                            batch_labels = self._label_batch(
                                labeling_service=labeling_service,
                                loop=loop,
                                batch_features=batch_features,
                                batch_examples=batch_examples,
                                batch_all_examples=batch_all_examples,
                                batch_negatives=batch_negatives,
                                feature_logit_effects=feature_logit_effects,
                                template_config=template_config,
                                user_prompt_template=user_prompt_template,
                                system_message=system_message,
                            )

                            # Persist this batch immediately
                            for feature, label, examples in zip(batch_features, batch_labels, batch_examples):
                                # ONE writer. See `_persist_label_outcome`: this loop used to be
                                # copy-pasted three times, and it is where a judge failure was
                                # stamped into `features` as though it were a verdict.
                                outcome = self._persist_label_outcome(
                                    feature,
                                    label,
                                    examples,
                                    labeling_job=labeling_job,
                                    labeled_at=labeled_at,
                                    label_source_value=label_source_value,
                                    prompt_fingerprint=label_fingerprint,
                                    judge_model=judge_model,
                                )
                                outcome_counts[outcome] += 1

                            # Commit this batch
                            self.db.commit()

                            # Update progress
                            current_labeled = batch_end
                            labeling_progress_callback(current_labeled, total_features)

                            logger.info(f"Batch {batch_start//LABEL_BATCH_SIZE + 1}/{(total_features + LABEL_BATCH_SIZE - 1)//LABEL_BATCH_SIZE}: Labeled and persisted features {batch_start+1}-{batch_end}/{total_features}")
                    finally:
                        # Clean up: close httpx client, then shut down the loop
                        loop.run_until_complete(labeling_service._http_client.aclose())
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.close()

                    logger.info(f"All {total_features} features labeled and persisted successfully")

                    # Create labels list for statistics calculation (now we need to query back from DB)
                    labels = [{"category": f.category, "specific": f.name} for f in features]

                elif labeling_method == LabelingMethod.OPENAI_COMPATIBLE.value:
                    # OpenAI-compatible endpoint (Ollama, vLLM, etc.)
                    endpoint = labeling_job.openai_compatible_endpoint
                    model_name = labeling_job.openai_compatible_model

                    if not endpoint:
                        raise ValueError("OpenAI-compatible endpoint not provided")
                    if not model_name:
                        raise ValueError("OpenAI-compatible model name not provided")

                    # Fetch prompt template - use specified or fall back to DB default
                    system_message = None
                    user_prompt_template = None
                    temperature = 0.3
                    max_tokens = labeling_job.max_tokens or 300
                    top_p = 0.9

                    from src.models.labeling_prompt_template import LabelingPromptTemplate
                    template = None
                    if labeling_job.prompt_template_id:
                        template = self.db.query(LabelingPromptTemplate).filter(
                            LabelingPromptTemplate.id == labeling_job.prompt_template_id
                        ).first()
                    else:
                        template = self.db.query(LabelingPromptTemplate).filter(
                            LabelingPromptTemplate.is_default == True  # noqa: E712
                        ).first()
                        if template:
                            logger.info(f"No template in job - using DB default: {template.name}")

                    if template:
                        system_message = template.system_message
                        user_prompt_template = template.user_prompt_template
                        temperature = template.temperature
                        # THE JOB'S VALUE WINS (MIS-E2E-060).
                        #
                        # `max_tokens` is exposed on the API and in the UI as a
                        # per-job setting and was then unconditionally replaced
                        # by the template's (default 50). A user raising it to
                        # get longer descriptions had the value accepted and
                        # every description still truncated — a control that
                        # appears to work and does nothing.
                        #
                        # The sibling `max_examples` already gets this
                        # precedence right; this is the same rule.
                        if labeling_job.max_tokens:
                            max_tokens = labeling_job.max_tokens
                        else:
                            max_tokens = template.max_tokens
                        top_p = template.top_p
                        logger.info(f"Using prompt template: {template.name} (ID: {template.id})")

                    api_timeout = labeling_job.api_timeout

                    # Generate and persist labels in batches of 10
                    # This ensures progress is saved incrementally if the job fails
                    label_source_value = LabelSource.OPENAI.value  # Use OPENAI source for compatible endpoints
                    labeled_at = datetime.now(timezone.utc)
                    label_fingerprint = prompt_fingerprint(template)
                    judge_model = labeling_job.openai_compatible_model
                    LABEL_BATCH_SIZE = job_batch_size

                    logger.info(f"Starting incremental labeling: {total_features} features in batches of {LABEL_BATCH_SIZE}")

                    # Ensure the miLLM model is loaded before making inference calls.
                    # No-ops silently for non-miLLM endpoints (Ollama, vLLM, etc.).
                    logger.info(f"Ensuring model {model_name!r} is loaded at {endpoint}")
                    ensure_model_loaded(endpoint, model_name)

                    # Create event loop BEFORE the OpenAI service so that httpx
                    # AsyncClient and asyncio.Semaphore bind to this loop.
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        logger.info(f"Initializing OpenAI-compatible labeling service with endpoint: {endpoint}, model: {model_name}")
                        labeling_service = OpenAILabelingService(
                            api_key="dummy-key-not-required",  # Most local endpoints don't require auth
                            model=model_name,
                            base_url=endpoint,
                            system_message=system_message,
                            user_prompt_template=user_prompt_template,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            timeout=api_timeout,
                            filter_special=labeling_job.filter_special,
                            filter_single_char=labeling_job.filter_single_char,
                            filter_punctuation=labeling_job.filter_punctuation,
                            filter_numbers=labeling_job.filter_numbers,
                            filter_fragments=labeling_job.filter_fragments,
                            filter_stop_words=labeling_job.filter_stop_words,
                            save_requests_for_testing=labeling_job.save_requests_for_testing,
                            export_format=labeling_job.export_format,
                            save_poor_quality_labels=labeling_job.save_poor_quality_labels,
                            poor_quality_sample_rate=labeling_job.poor_quality_sample_rate,
                            save_requests_sample_rate=labeling_job.save_requests_sample_rate,
                            labeling_job_id=labeling_job.id
                        )

                        # Pre-load logit effects for all features (one bulk query, avoids N+1)
                        feature_logit_effects: Dict[str, Optional[Dict]] = {}
                        if template_config.get('include_logit_effects'):
                            from src.models.feature_dashboard import FeatureDashboardData
                            n_promoted = template_config.get('top_promoted_tokens_count', 10)
                            n_suppressed = template_config.get('top_suppressed_tokens_count', 10)
                            feature_ids = [f.id for f in features]
                            dashboard_rows = self.db.query(
                                FeatureDashboardData.feature_id,
                                FeatureDashboardData.logit_lens_data
                            ).filter(
                                FeatureDashboardData.feature_id.in_(feature_ids)
                            ).all()
                            for row in dashboard_rows:
                                lens = row.logit_lens_data or {}
                                top_pos = lens.get('top_positive', [])[:n_promoted]
                                top_neg = lens.get('top_negative', [])[:n_suppressed]
                                feature_logit_effects[row.feature_id] = {
                                    'top_promoted': [t['token'] for t in top_pos],
                                    'top_suppressed': [t['token'] for t in top_neg],
                                }
                            logger.info(f"Pre-loaded logit effects for {len(feature_logit_effects)}/{len(feature_ids)} features")

                        for batch_start in range(0, total_features, LABEL_BATCH_SIZE):
                            # Stop promptly when the user cancels (see
                            # _raise_if_cancelled: revoke cannot kill a solo-pool task).
                            self._raise_if_cancelled(labeling_job_id)
                            batch_end = min(batch_start + LABEL_BATCH_SIZE, total_features)
                            batch_features = features[batch_start:batch_end]

                            # Claim before asking the judge. `in_progress` is excluded from
                            # eligibility, so a second resume started mid-run skips what this
                            # one already holds; without the write, that exclusion is
                            # decorative and both jobs pay for the same features.
                            self._claim_features(batch_features, labeled_at)
                            batch_examples = features_examples[batch_start:batch_end]
                            batch_all_examples = all_features_examples[batch_start:batch_end]
                            batch_negatives = assemble_batch_negatives(
                                batch_features, negatives_by_feature_id
                            )

                            # Generate labels for this batch using context-based examples
                            # Create concurrent tasks for all features in batch
                            # Pass all examples for NLP analysis to improve labeling
                            batch_labels = self._label_batch(
                                labeling_service=labeling_service,
                                loop=loop,
                                batch_features=batch_features,
                                batch_examples=batch_examples,
                                batch_all_examples=batch_all_examples,
                                batch_negatives=batch_negatives,
                                feature_logit_effects=feature_logit_effects,
                                template_config=template_config,
                                user_prompt_template=user_prompt_template,
                                system_message=system_message,
                            )

                            # Persist this batch immediately
                            for feature, label, examples in zip(batch_features, batch_labels, batch_examples):
                                # ONE writer. See `_persist_label_outcome`: this loop used to be
                                # copy-pasted three times, and it is where a judge failure was
                                # stamped into `features` as though it were a verdict.
                                outcome = self._persist_label_outcome(
                                    feature,
                                    label,
                                    examples,
                                    labeling_job=labeling_job,
                                    labeled_at=labeled_at,
                                    label_source_value=label_source_value,
                                    prompt_fingerprint=label_fingerprint,
                                    judge_model=judge_model,
                                )
                                outcome_counts[outcome] += 1

                            # Commit this batch
                            self.db.commit()

                            # Update progress
                            current_labeled = batch_end
                            labeling_progress_callback(current_labeled, total_features)

                            logger.info(f"Batch {batch_start//LABEL_BATCH_SIZE + 1}/{(total_features + LABEL_BATCH_SIZE - 1)//LABEL_BATCH_SIZE}: Labeled and persisted features {batch_start+1}-{batch_end}/{total_features}")
                    finally:
                        # Clean up: close httpx client, then shut down the loop
                        loop.run_until_complete(labeling_service._http_client.aclose())
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.close()

                    logger.info(f"All {total_features} features labeled and persisted successfully")

                    # Create labels list for statistics calculation (now we need to query back from DB)
                    labels = [{"category": f.category, "specific": f.name} for f in features]

                else:
                    raise ValueError(f"Unsupported labeling method: {labeling_method}")

                # Note: Feature persistence now happens incrementally in each method branch above
                logger.info(f"Successfully labeled and persisted {len(features)} features using {labeling_method}")

                # Unload OpenAI-compatible model (Ollama) from VRAM after completion
                if labeling_method == LabelingMethod.OPENAI_COMPATIBLE.value:
                    logger.info("Unloading OpenAI-compatible model from VRAM")
                    asyncio.run(self._unload_ollama_model(
                        labeling_job.openai_compatible_endpoint,
                        labeling_job.openai_compatible_model
                    ))

                # Calculate statistics
                end_time = datetime.now(timezone.utc)
                duration_seconds = (end_time - start_time).total_seconds()

                successfully_labeled = outcome_counts[self.LABEL_STATUS_SUCCEEDED]
                failed_labels = outcome_counts[self.LABEL_STATUS_FAILED]
                skipped_labels = outcome_counts[self.LABEL_STATUS_SKIPPED]
                avg_label_length = sum(len(l.get("specific", "")) for l in labels) / len(labels) if labels else 0

                statistics = {
                    "total_features": len(features),
                    "successfully_labeled": successfully_labeled,
                    "failed_labels": failed_labels,
                    # Aqua features the run deliberately left alone. Previously
                    # invisible: they were `logger.debug`'d and then counted as
                    # labeled, inflating `features_labeled`.
                    "skipped_labels": skipped_labels,
                    "avg_label_length": round(avg_label_length, 2),
                    "labeling_duration_seconds": round(duration_seconds, 2),
                    "labeling_method": labeling_method
                }

                # Mark labeling job as completed
                labeling_job.status = LabelingStatus.COMPLETED.value
                labeling_job.progress = 1.0
                # Features actually LABELED. `len(labels)` counted everything
                # attempted, including failures and aqua skips.
                labeling_job.features_labeled = successfully_labeled
                labeling_job.completed_at = end_time
                labeling_job.updated_at = end_time
                labeling_job.statistics = statistics
                self.db.commit()

                logger.info(f"Labeling job {labeling_job_id} completed successfully")

                # Emit completion event via WebSocket
                emit_labeling_progress(
                    labeling_job_id=labeling_job.id,
                    event="labeling:completed",
                    data={
                        "labeling_job_id": labeling_job.id,
                        "extraction_job_id": labeling_job.extraction_job_id,
                        "status": "completed",
                        "features_labeled": successfully_labeled,
                        "total_features": total_features,
                        "statistics": statistics,
                        "message": f"Successfully labeled {successfully_labeled}/{total_features} features in {duration_seconds:.1f}s"
                    }
                )

                return statistics

            except Exception as e:
                logger.error(f"Batch labeling failed: {e}", exc_info=True)
                raise

        except JudgeUnavailable as exc:
            # THE JUDGE, NOT THE FEATURES.
            #
            # Fail the job and leave every feature exactly as it was — no
            # status change, no reason, and no attempt spent. Nothing was
            # learned about any of them, so nothing should be recorded against
            # them. The operator fixes the judge and resumes with the retry
            # budget intact.
            logger.error("Labeling job %s: judge unavailable — %s", labeling_job_id, exc)
            labeling_job.status = LabelingStatus.FAILED.value
            labeling_job.error_message = str(exc)[:1000]
            labeling_job.completed_at = datetime.now(timezone.utc)
            labeling_job.updated_at = labeling_job.completed_at
            self.db.commit()

            emit_labeling_progress(
                labeling_job_id=labeling_job.id,
                event="labeling:failed",
                data={
                    "labeling_job_id": labeling_job.id,
                    "extraction_job_id": labeling_job.extraction_job_id,
                    "status": LabelingStatus.FAILED.value,
                    "error": str(exc)[:1000],
                    "message": "The labeling model is unavailable — no features were changed.",
                },
            )
            raise

        except LabelingService._LabelingCancelled:
            # A DELIBERATE CANCELLATION IS NOT A FAILURE (MIS-E2E-058).
            #
            # This fell through to the handler below, which set status=FAILED
            # and emitted `labeling:failed` before re-raising — so a user
            # pressing Cancel saw the job reported as broken. Worse,
            # `labeling_tasks.py` carries a comment asserting "the job row is
            # already CANCELLED", which was false precisely because this
            # handler had just overwritten it, so the next reader would not
            # look.
            #
            # Only reachable now that MIS-E2E-057 is fixed: before that the
            # cancellation was never raised at all.
            logger.info(f"Labeling job {labeling_job_id} was cancelled by the user")
            labeling_job.status = LabelingStatus.CANCELLED.value
            labeling_job.updated_at = datetime.now(timezone.utc)
            self.db.commit()

            emit_labeling_progress(
                labeling_job_id=labeling_job.id,
                event="labeling:cancelled",
                data={
                    "labeling_job_id": labeling_job.id,
                    "extraction_job_id": labeling_job.extraction_job_id,
                    "status": LabelingStatus.CANCELLED.value,
                    "message": "Labeling cancelled",
                },
            )
            raise

        except Exception as e:
            logger.error(f"Feature labeling failed for job {labeling_job_id}: {e}", exc_info=True)

            # Mark labeling job as failed
            labeling_job.status = LabelingStatus.FAILED.value
            labeling_job.error_message = str(e)
            labeling_job.updated_at = datetime.now(timezone.utc)
            self.db.commit()

            # Emit failure event via WebSocket
            emit_labeling_progress(
                labeling_job_id=labeling_job.id,
                event="labeling:failed",
                data={
                    "labeling_job_id": labeling_job.id,
                    "extraction_job_id": labeling_job.extraction_job_id,
                    "status": "failed",
                    "error_message": str(e),
                    "message": f"Labeling failed: {str(e)}"
                }
            )

            raise
        finally:
            # Always release HTTP client file descriptors, whether we succeeded or failed.
            if 'labeling_service' in locals() and hasattr(labeling_service, 'close'):
                try:
                    labeling_service.close()
                except Exception:
                    pass

    async def get_labeling_job(self, labeling_job_id: str) -> Optional[LabelingJob]:
        """
        Get a labeling job by ID.

        Args:
            labeling_job_id: ID of the labeling job

        Returns:
            LabelingJob or None if not found
        """
        result = await self.db.execute(
            select(LabelingJob).where(LabelingJob.id == labeling_job_id)
        )
        return result.scalar_one_or_none()

    async def list_labeling_jobs(
        self,
        extraction_job_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[List[LabelingJob], int]:
        """
        List labeling jobs with optional filtering.

        Args:
            extraction_job_id: Optional filter by extraction job ID
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            Tuple of (list of labeling jobs, total count)
        """
        from sqlalchemy import func

        # Build query
        query = select(LabelingJob).order_by(desc(LabelingJob.created_at))

        if extraction_job_id:
            query = query.where(LabelingJob.extraction_job_id == extraction_job_id)

        # Get total count
        count_query = select(func.count()).select_from(LabelingJob)
        if extraction_job_id:
            count_query = count_query.where(LabelingJob.extraction_job_id == extraction_job_id)

        count_result = await self.db.execute(count_query)
        total = count_result.scalar_one()

        # Get paginated results
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        jobs = result.scalars().all()

        return list(jobs), total

    async def cancel_labeling_job(self, labeling_job_id: str) -> bool:
        """
        Cancel a labeling job.

        Args:
            labeling_job_id: ID of the labeling job to cancel

        Returns:
            True if cancelled successfully

        Raises:
            ValueError: If job not found or not in cancellable state
        """
        result = await self.db.execute(
            select(LabelingJob).where(LabelingJob.id == labeling_job_id)
        )
        labeling_job = result.scalar_one_or_none()

        if not labeling_job:
            raise ValueError(f"Labeling job {labeling_job_id} not found")

        if labeling_job.status not in [LabelingStatus.QUEUED.value, LabelingStatus.LABELING.value]:
            raise ValueError(
                f"Cannot cancel labeling job {labeling_job_id} with status {labeling_job.status}"
            )

        # NO SECOND REVOKE HERE, AND EMPHATICALLY NOT A TERMINATING ONE.
        #
        # `request_cancel` below already issues a plain `revoke()` centrally,
        # for the one case revoke genuinely handles: a task that has not started
        # never will. This block used to issue its own with
        # `terminate=True, signal='SIGTERM'` on top of it, and that turned
        # cancel into a loop that restarted the very job it stopped.
        #
        # Measured, not reasoned: cancelling the 53k-feature L46 job on
        # 2026-09-10 wrote CANCELLED at 00:07:37 and the worker logged
        # `Connected to redis / mingle` at 00:07:44. SIGTERM reaches the solo
        # pool's MAIN process because solo has no child to signal, so the worker
        # dies mid-task; `task_acks_late` + `task_reject_on_worker_lost` then
        # requeue the message, and the replacement worker runs the job again.
        # The same job had already been resurrected once this way, by the deploy
        # at 10:57 the previous day, and had spent 13 GPU-hours re-labelling
        # 7,324 features it already held a verdict for.
        #
        # Stopping a running task is COOPERATIVE here. The row is the channel;
        # `_raise_if_cancelled` reads it once per batch.

        # Unload model from VRAM if using OpenAI-compatible endpoint (Ollama)
        if labeling_job.labeling_method == LabelingMethod.OPENAI_COMPATIBLE.value:
            await self._unload_ollama_model(
                labeling_job.openai_compatible_endpoint,
                labeling_job.openai_compatible_model
            )

        # THROUGH THE REGISTRY, so the word written here is by construction the
        # word `_raise_if_cancelled` reads. Writing the status inline worked
        # only because both sides happened to spell it the same way — which is
        # exactly what `saes.py` did not, where the endpoint wrote FAILED and a
        # checker looking for "cancelled" could never have seen it.
        from starlette.concurrency import run_in_threadpool

        from ..core.cancellation import request_cancel

        await run_in_threadpool(
            request_cancel, "labeling", labeling_job_id,
            reason="Cancelled by user",
            celery_task_id=getattr(labeling_job, "celery_task_id", None),
        )
        await self.db.refresh(labeling_job)

        logger.info(f"Cancelled labeling job {labeling_job_id}")
        return True

    async def _unload_ollama_model(self, endpoint: Optional[str], model_name: Optional[str]) -> None:
        """
        Unload model from Ollama VRAM by sending a request with keep_alive=0.

        Args:
            endpoint: Ollama endpoint URL
            model_name: Model name to unload
        """
        if not endpoint or not model_name:
            return

        try:
            import httpx
            # Extract base URL (remove /v1 or /api suffix if present)
            base_url = endpoint.rstrip('/').replace('/v1', '').replace('/api', '')
            unload_url = f"{base_url}/api/generate"

            async with httpx.AsyncClient(timeout=10.0) as client:
                # Send empty prompt with keep_alive=0 to unload model
                response = await client.post(
                    unload_url,
                    json={
                        "model": model_name,
                        "prompt": "",
                        "keep_alive": 0  # Unload immediately
                    }
                )
                if response.status_code == 200:
                    logger.info(f"Successfully unloaded model {model_name} from VRAM")
                else:
                    logger.warning(f"Failed to unload model {model_name}: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not unload model from VRAM: {e}")
            # Non-critical error, don't raise

    async def delete_labeling_job(self, labeling_job_id: str) -> bool:
        """
        Delete a labeling job.

        This does NOT delete the features or their labels, only the labeling job record.
        Feature labels will remain intact. If the job is active, it will be cancelled first.

        Args:
            labeling_job_id: ID of the labeling job to delete

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If job not found
        """
        from sqlalchemy import update

        result = await self.db.execute(
            select(LabelingJob).where(LabelingJob.id == labeling_job_id)
        )
        labeling_job = result.scalar_one_or_none()

        if not labeling_job:
            raise ValueError(f"Labeling job {labeling_job_id} not found")

        # THE SAME NON-TERMINATING REVOKE AS `cancel_labeling_job`, and for the
        # same reason — see the comment there. SIGTERM to a `--pool=solo`
        # worker has no pool child to land on.
        #
        # It matters MORE here, not less. This function then DELETES the row,
        # and the row is the only channel a running task has: the `labeling`
        # scope carries `missing_row="cancelled"` precisely so a vanished row
        # reads as a stop. Killing the worker on the way out replaces that
        # orderly stop with a dead pool and a task whose message may not have
        # been acked.
        if labeling_job.status in [LabelingStatus.QUEUED.value, LabelingStatus.LABELING.value]:
            if labeling_job.celery_task_id:
                from ..core.celery_app import celery_app
                logger.info(f"Auto-cancelling active job: revoking Celery task {labeling_job.celery_task_id}")
                celery_app.control.revoke(labeling_job.celery_task_id)

        # Clear labeling_job_id reference from features
        await self.db.execute(
            update(Feature).where(
                Feature.labeling_job_id == labeling_job_id
            ).values(labeling_job_id=None)
        )

        # Delete labeling job
        await self.db.delete(labeling_job)
        await self.db.commit()

        logger.info(f"Deleted labeling job {labeling_job_id}")
        return True
