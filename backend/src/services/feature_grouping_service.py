"""
Cross-feature grouping service (Feature 010).

Builds, for one extraction, a token→feature inverted index and splits each
shared-token bucket into context-similarity subgroups:

1. For every feature, read its top-N activating examples
   (``feature_activations`` is range-partitioned by feature_id — reads are
   always filtered by explicit feature-id batches).
2. Rank the feature's normalized prime tokens by activation-weighted share;
   keep ranks 1–3 with weight ≥ 0.2 as ``feature_token_index`` rows, each with
   a normalized ±window context bag.
3. Bucket features by rank-1 normalized token; within each bucket compute
   TF-IDF vectors over the context bags, threshold the cosine matrix, and take
   connected components as subgroups. Cohesion = mean pairwise cosine.

Mutable feature attributes (labels, stars) are never copied into the result
tables — consumers join them live from ``features``.
"""

import hashlib
import json
import logging
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from ..models.feature import Feature
from ..models.feature_activation import FeatureActivation
from ..models.feature_grouping import (
    FeatureGroup,
    FeatureGroupMember,
    FeatureGroupingRun,
    FeatureTokenIndex,
    GroupingRunStatus,
)
from ..utils.token_normalization import normalize_bag, normalize_token

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: Dict[str, Any] = {
    "context_window": 5,
    "similarity_threshold": 0.35,
    "top_examples": 10,
    "min_group_size": 2,
}

# Keep a feature's token in the index when it is one of the top 3 prime tokens
# and carries at least this activation-weighted share of the examples.
MAX_TOKEN_RANK = 3
MIN_TOKEN_WEIGHT = 0.2

FEATURE_BATCH_SIZE = 200
SNIPPET_MAX_CHARS = 160


def params_hash(params: Dict[str, Any]) -> str:
    """Stable hash of grouping params for idempotency checks."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class FeatureGroupingService:
    """Builds the grouping index for one extraction. Sync-session (Celery)."""

    def compute(
        self,
        db: Session,
        extraction_id: str,
        params: Optional[Dict[str, Any]] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_check_for: Optional[Callable[[str], Callable[[], bool]]] = None,
    ) -> FeatureGroupingRun:
        """Run the full pipeline; returns the completed run."""
        merged = {**DEFAULT_PARAMS, **(params or {})}
        run = FeatureGroupingRun(
            id=str(uuid.uuid4()),
            extraction_id=extraction_id,
            status=GroupingRunStatus.COMPUTING.value,
            params=merged,
            params_hash=params_hash(merged),
        )
        db.add(run)
        db.commit()

        def report(progress: float, stage: str) -> None:
            if progress_cb:
                progress_cb(progress, stage)

        try:
            features = (
                db.query(Feature.id, Feature.neuron_index)
                .filter(Feature.extraction_job_id == extraction_id)
                .order_by(Feature.neuron_index)
                .all()
            )
            run.feature_count = len(features)
            db.commit()
            if not features:
                self._finalize(db, run, group_count=0)
                report(1.0, "completed")
                return run

            report(0.0, "indexing")
            # A FACTORY, NOT A CHECKER. The run id does not exist until the
            # row above is created, so the caller cannot build a checker in
            # advance — and passing the scope NAME into this service instead
            # would put cancellation vocabulary somewhere with no business
            # knowing it.
            index_rows = self._build_index(
                db, run, features, merged, report,
                cancel_check=cancel_check_for(run.id) if cancel_check_for else None,
            )

            report(0.85, "grouping")
            group_count = self._build_groups(db, run, index_rows, merged)

            self._replace_previous_runs(db, run)
            self._finalize(db, run, group_count=group_count)
            report(1.0, "completed")
            logger.info(
                f"Feature grouping completed for extraction {extraction_id}: "
                f"{run.feature_count} features, {group_count} groups"
            )
            return run
        except Exception as e:
            db.rollback()
            run.status = GroupingRunStatus.FAILED.value
            run.error_message = str(e)[:500]
            run.completed_at = datetime.now(timezone.utc)
            db.commit()
            raise

    # ------------------------------------------------------------------ index

    def _build_index(
        self,
        db: Session,
        run: FeatureGroupingRun,
        features: List[Tuple[str, int]],
        params: Dict[str, Any],
        report: Callable[[float, str], None],
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> List[FeatureTokenIndex]:
        """Build and persist the inverted index; returns rank-1 rows in memory."""
        window = params["context_window"]
        top_examples = params["top_examples"]
        all_rank1: List[FeatureTokenIndex] = []
        total = len(features)

        for batch_start in range(0, total, FEATURE_BATCH_SIZE):
            # THE CHECKPOINT. One batch of features is the finest boundary this
            # pass can abandon at: the rank-1 index is accumulated in memory
            # and written once at the end, so stopping here leaves nothing
            # half-written.
            if cancel_check is not None and cancel_check():
                from ..core.cancellation import OperatorCancelled

                raise OperatorCancelled(
                    "feature_grouping", run.id,
                    detail=f"stopped at feature {batch_start} of {total}",
                )
            batch = features[batch_start : batch_start + FEATURE_BATCH_SIZE]
            batch_ids = [f.id for f in batch]
            neuron_by_id = {f.id: f.neuron_index for f in batch}

            # Partitioned table: always filter by explicit feature ids.
            activations = (
                db.query(
                    FeatureActivation.feature_id,
                    FeatureActivation.prime_token,
                    FeatureActivation.prefix_tokens,
                    FeatureActivation.suffix_tokens,
                    FeatureActivation.max_activation,
                )
                .filter(FeatureActivation.feature_id.in_(batch_ids))
                .order_by(FeatureActivation.max_activation.desc())
                .all()
            )
            examples_by_feature: Dict[str, list] = defaultdict(list)
            for act in activations:
                bucket = examples_by_feature[act.feature_id]
                if len(bucket) < top_examples:
                    bucket.append(act)

            rows: List[FeatureTokenIndex] = []
            for feature_id in batch_ids:
                examples = examples_by_feature.get(feature_id)
                if not examples:
                    continue
                rows.extend(
                    self._index_rows_for_feature(
                        run, feature_id, neuron_by_id[feature_id], examples, window
                    )
                )
            db.bulk_save_objects(rows, return_defaults=False)
            db.commit()
            all_rank1.extend(r for r in rows if r.token_rank == 1)

            done = min(batch_start + FEATURE_BATCH_SIZE, total)
            report(0.85 * done / total, f"indexing ({done}/{total} features)")

        return all_rank1

    def _index_rows_for_feature(
        self,
        run: FeatureGroupingRun,
        feature_id: str,
        neuron_index: int,
        examples: list,
        window: int,
    ) -> List[FeatureTokenIndex]:
        """Rank normalized prime tokens by activation weight; emit index rows."""
        weights: Counter = Counter()
        raw_forms: Dict[str, Counter] = defaultdict(Counter)
        context_bags: Dict[str, List[str]] = defaultdict(list)
        best_example: Dict[str, Any] = {}
        total_weight = 0.0

        for ex in examples:
            norm = normalize_token(ex.prime_token or "")
            weight = max(float(ex.max_activation or 0.0), 1e-6)
            total_weight += weight
            if norm is None:
                continue
            weights[norm] += weight
            raw_forms[norm][(ex.prime_token or "").strip() or norm] += 1
            prefix = (ex.prefix_tokens or [])[-window:]
            suffix = (ex.suffix_tokens or [])[:window]
            context_bags[norm].extend(normalize_bag(list(prefix) + list(suffix)))
            prev = best_example.get(norm)
            if prev is None or float(ex.max_activation or 0) > float(prev.max_activation or 0):
                best_example[norm] = ex

        if total_weight <= 0 or not weights:
            return []

        rows = []
        for rank, (norm, weight) in enumerate(weights.most_common(MAX_TOKEN_RANK), start=1):
            share = weight / total_weight
            if rank > 1 and share < MIN_TOKEN_WEIGHT:
                continue
            best = best_example[norm]
            rows.append(
                FeatureTokenIndex(
                    run_id=run.id,
                    extraction_id=run.extraction_id,
                    feature_id=feature_id,
                    neuron_index=neuron_index,
                    raw_token=raw_forms[norm].most_common(1)[0][0],
                    normalized_token=norm,
                    token_rank=rank,
                    weight=round(share, 4),
                    context_tokens=context_bags[norm][:60],
                )
            )
            # Stash snippet source on the row object for group building
            rows[-1]._snippet = self._snippet(best, window)  # type: ignore[attr-defined]
        return rows

    @staticmethod
    def _snippet(example: Any, window: int) -> str:
        prefix = " ".join((example.prefix_tokens or [])[-window:])
        suffix = " ".join((example.suffix_tokens or [])[:window])
        text = f"{prefix} *{(example.prime_token or '').strip()}* {suffix}".strip()
        return text[:SNIPPET_MAX_CHARS]

    # ----------------------------------------------------------------- groups

    def _build_groups(
        self,
        db: Session,
        run: FeatureGroupingRun,
        rank1_rows: List[FeatureTokenIndex],
        params: Dict[str, Any],
    ) -> int:
        """Bucket rank-1 rows by token; split buckets by context similarity."""
        threshold = params["similarity_threshold"]
        min_size = params["min_group_size"]

        buckets: Dict[str, List[FeatureTokenIndex]] = defaultdict(list)
        for row in rank1_rows:
            buckets[row.normalized_token].append(row)

        group_count = 0
        groups: List[FeatureGroup] = []
        members: List[FeatureGroupMember] = []

        for token, rows in buckets.items():
            if len(rows) < min_size:
                continue
            components, sims = self._context_components(rows, threshold)
            for component in components:
                if len(component) < min_size:
                    continue
                comp_rows = [rows[i] for i in component]
                cohesion, member_sims = self._cohesion(sims, component)
                display = Counter(r.raw_token for r in comp_rows).most_common(1)[0][0]
                group = FeatureGroup(
                    id=str(uuid.uuid4()),
                    run_id=run.id,
                    extraction_id=run.extraction_id,
                    normalized_token=token,
                    display_token=display,
                    member_count=len(comp_rows),
                    cohesion=round(cohesion, 4),
                )
                groups.append(group)
                for row, sim in zip(comp_rows, member_sims):
                    members.append(
                        FeatureGroupMember(
                            group_id=group.id,
                            feature_id=row.feature_id,
                            similarity=round(float(sim), 4),
                            context_snippet=getattr(row, "_snippet", None),
                        )
                    )
                group_count += 1

        db.bulk_save_objects(groups)
        db.bulk_save_objects(members)
        db.commit()
        return group_count

    @staticmethod
    def _context_components(
        rows: List[FeatureTokenIndex], threshold: float
    ) -> Tuple[List[List[int]], np.ndarray]:
        """Split a token bucket into context-similarity connected components."""
        from scipy.sparse.csgraph import connected_components
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        bags = [row.context_tokens or [] for row in rows]
        if all(not bag for bag in bags):
            # No context anywhere — keep the bucket as one group.
            n = len(rows)
            return [list(range(n))], np.ones((n, n))

        # Bags are pre-tokenized: identity analyzer, no lowercasing/re-splitting.
        vectorizer = TfidfVectorizer(analyzer=lambda bag: bag, lowercase=False)
        try:
            tfidf = vectorizer.fit_transform(bags)
        except ValueError:
            # Empty vocabulary (all bags empty after filtering)
            n = len(rows)
            return [list(range(n))], np.ones((n, n))

        sims = cosine_similarity(tfidf)
        adjacency = (sims >= threshold).astype(int)
        n_components, labels = connected_components(adjacency, directed=False)
        components = [
            [i for i, lbl in enumerate(labels) if lbl == comp]
            for comp in range(n_components)
        ]
        return components, sims

    @staticmethod
    def _cohesion(sims: np.ndarray, component: List[int]) -> Tuple[float, List[float]]:
        """Mean pairwise cosine within the component + per-member centroid sim."""
        if len(component) == 1:
            return 1.0, [1.0]
        sub = sims[np.ix_(component, component)]
        upper = sub[np.triu_indices(len(component), k=1)]
        cohesion = float(upper.mean()) if upper.size else 1.0
        # Member similarity: mean similarity to the other members (centroid proxy)
        member_sims = [
            float((sub[i].sum() - 1.0) / (len(component) - 1)) for i in range(len(component))
        ]
        return cohesion, member_sims

    # ------------------------------------------------------------------ misc

    @staticmethod
    def _replace_previous_runs(db: Session, run: FeatureGroupingRun) -> None:
        """Delete older runs for the extraction (cascade removes their rows)."""
        old_runs = (
            db.query(FeatureGroupingRun)
            .filter(
                FeatureGroupingRun.extraction_id == run.extraction_id,
                FeatureGroupingRun.id != run.id,
            )
            .all()
        )
        for old in old_runs:
            db.delete(old)
        if old_runs:
            db.commit()

    @staticmethod
    def _finalize(db: Session, run: FeatureGroupingRun, group_count: int) -> None:
        run.status = GroupingRunStatus.COMPLETED.value
        run.group_count = group_count
        run.completed_at = datetime.now(timezone.utc)
        db.commit()


def get_completed_run(db: Session, extraction_id: str) -> Optional[FeatureGroupingRun]:
    """Latest completed run for an extraction, if any."""
    return (
        db.query(FeatureGroupingRun)
        .filter(
            FeatureGroupingRun.extraction_id == extraction_id,
            FeatureGroupingRun.status == GroupingRunStatus.COMPLETED.value,
        )
        .order_by(FeatureGroupingRun.created_at.desc())
        .first()
    )
