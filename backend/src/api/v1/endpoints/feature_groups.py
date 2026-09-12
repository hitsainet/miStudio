"""
Cross-feature grouping API endpoints (Feature 010).

Serves BOTH the frontend Feature Groups view and the MCP server's `groups`
tools — one source of truth (BR-8.6). Labels/stars are joined live from
``features``; the persisted index holds only immutable data.
"""

import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.deps import get_db
from src.models.extraction_job import ExtractionJob
from src.models.feature import Feature
from src.models.feature_analysis_cache import AnalysisType, FeatureAnalysisCache
from src.models.feature_grouping import (
    FeatureGroup,
    FeatureGroupMember,
    FeatureGroupingRun,
    FeatureTokenIndex,
    GroupingRunStatus,
)
from src.schemas.feature_group import (
    ByTokenFeatureOut,
    ByTokenResponse,
    ComputeGroupsRequest,
    ComputeGroupsResponse,
    FeatureGroupDetailResponse,
    FeatureGroupListResponse,
    FeatureGroupMemberOut,
    FeatureGroupSummary,
    GroupingStatusResponse,
    RelatedFeatureOut,
    RelatedFeaturesResponse,
)
from src.services.feature_grouping_service import params_hash
from src.services.task_queue_service import TaskQueueService
from src.utils.token_normalization import normalize_token
from src.workers.feature_grouping_tasks import compute_feature_groups_task

logger = logging.getLogger(__name__)

router = APIRouter()


async def _get_extraction_or_404(db: AsyncSession, extraction_id: str) -> ExtractionJob:
    result = await db.execute(select(ExtractionJob).where(ExtractionJob.id == extraction_id))
    extraction = result.scalar_one_or_none()
    if not extraction:
        raise HTTPException(status_code=404, detail=f"Extraction {extraction_id} not found")
    return extraction


async def _latest_run(db: AsyncSession, extraction_id: str, completed_only: bool = False):
    query = (
        select(FeatureGroupingRun)
        .where(FeatureGroupingRun.extraction_id == extraction_id)
        .order_by(FeatureGroupingRun.created_at.desc())
        .limit(1)
    )
    if completed_only:
        query = query.where(FeatureGroupingRun.status == GroupingRunStatus.COMPLETED.value)
    result = await db.execute(query)
    return result.scalar_one_or_none()


@router.post(
    "/extractions/{extraction_id}/feature-groups/compute",
    response_model=ComputeGroupsResponse,
    status_code=202,
)
async def compute_feature_groups(
    extraction_id: str,
    request: ComputeGroupsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Start the grouping precompute job for an extraction (202).

    Idempotent: a completed run with identical params short-circuits unless
    ``force`` is set. Returns 409 while a run is already computing.
    """
    await _get_extraction_or_404(db, extraction_id)

    latest = await _latest_run(db, extraction_id)
    requested_hash = params_hash({**request.params.model_dump()})

    if latest and latest.status in (GroupingRunStatus.PENDING.value, GroupingRunStatus.COMPUTING.value):
        raise HTTPException(
            status_code=409,
            detail={"code": "ALREADY_COMPUTING", "run_id": latest.id,
                    "message": "A grouping run is already in progress for this extraction"},
        )

    if (
        latest
        and latest.status == GroupingRunStatus.COMPLETED.value
        and latest.params_hash == requested_hash
        and not request.force
    ):
        return ComputeGroupsResponse(
            run_id=latest.id,
            status="completed",
            message="Index already computed with identical params (use force=true to recompute)",
        )

    task = compute_feature_groups_task.delay(extraction_id, request.params.model_dump())
    await TaskQueueService.create_task_entry(
        db,
        task_type="feature_grouping",
        entity_id=extraction_id,
        entity_type="extraction",
        task_id=task.id,
        retry_params={"params": request.params.model_dump()},
    )

    # The run row is created by the worker; report queued state now.
    return ComputeGroupsResponse(
        task_id=task.id,
        run_id="pending",
        status="queued",
        message="Grouping job queued",
    )


@router.get(
    "/extractions/{extraction_id}/feature-groups/status",
    response_model=GroupingStatusResponse,
)
async def get_grouping_status(extraction_id: str, db: AsyncSession = Depends(get_db)):
    """Current grouping-index state for an extraction."""
    await _get_extraction_or_404(db, extraction_id)
    run = await _latest_run(db, extraction_id)
    if not run:
        return GroupingStatusResponse(status="none")
    return GroupingStatusResponse(
        status=run.status,  # type: ignore[arg-type]
        run_id=run.id,
        params=run.params,
        feature_count=run.feature_count,
        group_count=run.group_count,
        error_message=run.error_message,
        computed_at=run.completed_at,
    )


@router.get(
    "/extractions/{extraction_id}/feature-groups",
    response_model=FeatureGroupListResponse,
)
async def list_feature_groups(
    extraction_id: str,
    token: Optional[str] = Query(None, description="Exact normalized-token match"),
    search: Optional[str] = Query(None, description="Substring match on the token"),
    min_group_size: int = Query(2, ge=1),
    sort_by: Literal["size", "cohesion", "token"] = Query("size"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Paginated feature groups for the latest completed run."""
    await _get_extraction_or_404(db, extraction_id)
    run = await _latest_run(db, extraction_id, completed_only=True)
    if not run:
        return FeatureGroupListResponse(groups=[], total=0, limit=limit, offset=offset, index_status="none")

    query = select(FeatureGroup).where(
        FeatureGroup.run_id == run.id,
        FeatureGroup.member_count >= min_group_size,
    )
    if token:
        normalized = normalize_token(token) or token.lower()
        query = query.where(FeatureGroup.normalized_token == normalized)
    if search:
        query = query.where(FeatureGroup.normalized_token.ilike(f"%{search}%"))

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar() or 0

    order = {
        "size": FeatureGroup.member_count.desc(),
        "cohesion": FeatureGroup.cohesion.desc(),
        "token": FeatureGroup.normalized_token.asc(),
    }[sort_by]
    result = await db.execute(query.order_by(order).limit(limit).offset(offset))
    groups = result.scalars().all()

    # Up to 3 member labels per group for scanning (single query)
    sample_labels: dict[str, list[str]] = {g.id: [] for g in groups}
    if groups:
        members_result = await db.execute(
            select(FeatureGroupMember.group_id, Feature.name)
            .join(Feature, Feature.id == FeatureGroupMember.feature_id)
            .where(FeatureGroupMember.group_id.in_(list(sample_labels.keys())))
            .order_by(FeatureGroupMember.similarity.desc())
        )
        for group_id, name in members_result.all():
            bucket = sample_labels[group_id]
            if len(bucket) < 3 and name:
                bucket.append(name)

    return FeatureGroupListResponse(
        groups=[
            FeatureGroupSummary(
                group_id=g.id,
                normalized_token=g.normalized_token,
                display_token=g.display_token,
                member_count=g.member_count,
                cohesion=g.cohesion,
                sample_labels=sample_labels.get(g.id, []),
            )
            for g in groups
        ],
        total=total,
        limit=limit,
        offset=offset,
        index_status="completed",
    )


@router.get(
    "/extractions/{extraction_id}/feature-groups/{group_id}",
    response_model=FeatureGroupDetailResponse,
)
async def get_feature_group_members(
    extraction_id: str,
    group_id: str,
    category: Optional[str] = Query(None),
    has_label: Optional[bool] = Query(None, description="True → exclude auto-labeled features"),
    star_color: Optional[str] = Query(None),
    is_favorite: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Group members with label fields joined live from features."""
    group_result = await db.execute(
        select(FeatureGroup).where(FeatureGroup.id == group_id, FeatureGroup.extraction_id == extraction_id)
    )
    group = group_result.scalar_one_or_none()
    if not group:
        raise HTTPException(status_code=404, detail=f"Feature group {group_id} not found")

    query = (
        select(FeatureGroupMember, Feature)
        .join(Feature, Feature.id == FeatureGroupMember.feature_id)
        .where(FeatureGroupMember.group_id == group_id)
    )
    if category:
        query = query.where(Feature.category == category)
    if has_label is True:
        query = query.where(Feature.label_source != "auto")
    elif has_label is False:
        query = query.where(Feature.label_source == "auto")
    if star_color:
        query = query.where(Feature.star_color == star_color)
    if is_favorite is not None:
        query = query.where(Feature.is_favorite == is_favorite)

    result = await db.execute(query.order_by(FeatureGroupMember.similarity.desc()))
    rows = result.all()

    members = [
        FeatureGroupMemberOut(
            feature_id=feature.id,
            neuron_index=feature.neuron_index,
            name=feature.name,
            category=feature.category,
            label_source=str(feature.label_source) if feature.label_source else None,
            star_color=feature.star_color,
            is_favorite=feature.is_favorite,
            max_activation=feature.max_activation,
            activation_frequency=feature.activation_frequency,
            similarity=member.similarity,
            context_snippet=member.context_snippet,
        )
        for member, feature in rows
    ]
    return FeatureGroupDetailResponse(
        group_id=group.id,
        normalized_token=group.normalized_token,
        display_token=group.display_token,
        cohesion=group.cohesion,
        member_count=group.member_count,
        members=members,
    )


@router.get(
    "/extractions/{extraction_id}/features/by-token",
    response_model=ByTokenResponse,
)
async def find_features_by_token(
    extraction_id: str,
    token: str = Query(..., min_length=1),
    match: Literal["exact", "normalized", "prefix"] = Query("normalized"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Features whose top activating tokens match, from the inverted index."""
    await _get_extraction_or_404(db, extraction_id)
    run = await _latest_run(db, extraction_id, completed_only=True)
    if not run:
        raise HTTPException(
            status_code=409,
            detail={"code": "NO_INDEX", "message": "Grouping index not computed — POST .../feature-groups/compute first"},
        )

    query = (
        select(FeatureTokenIndex, Feature)
        .join(Feature, Feature.id == FeatureTokenIndex.feature_id)
        .where(FeatureTokenIndex.run_id == run.id)
    )
    if match == "exact":
        query = query.where(FeatureTokenIndex.raw_token == token)
    elif match == "normalized":
        normalized = normalize_token(token) or token.lower()
        query = query.where(FeatureTokenIndex.normalized_token == normalized)
    else:  # prefix
        normalized = normalize_token(token) or token.lower()
        query = query.where(FeatureTokenIndex.normalized_token.like(f"{normalized}%"))

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar() or 0
    result = await db.execute(
        query.order_by(FeatureTokenIndex.token_rank.asc(), FeatureTokenIndex.weight.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = result.all()

    return ByTokenResponse(
        features=[
            ByTokenFeatureOut(
                feature_id=feature.id,
                neuron_index=feature.neuron_index,
                name=feature.name,
                category=feature.category,
                raw_token=idx.raw_token,
                normalized_token=idx.normalized_token,
                token_rank=idx.token_rank,
                weight=idx.weight,
            )
            for idx, feature in rows
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/features/{feature_id}/related", response_model=RelatedFeaturesResponse)
async def find_related_features(
    feature_id: str,
    min_similarity: float = Query(0.2, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Features related to a seed feature via shared tokens, context, and cached correlations."""
    feature_result = await db.execute(select(Feature).where(Feature.id == feature_id))
    seed = feature_result.scalar_one_or_none()
    if not seed:
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")

    scores: dict[str, dict] = {}

    # 1) Shared normalized tokens + context Jaccard via the inverted index
    run = await _latest_run(db, seed.extraction_job_id, completed_only=True)
    if run:
        seed_rows_result = await db.execute(
            select(FeatureTokenIndex).where(
                FeatureTokenIndex.run_id == run.id,
                FeatureTokenIndex.feature_id == feature_id,
            )
        )
        seed_rows = seed_rows_result.scalars().all()
        seed_tokens = {r.normalized_token for r in seed_rows}
        seed_context = set()
        for r in seed_rows:
            seed_context.update(r.context_tokens or [])

        if seed_tokens:
            candidates_result = await db.execute(
                select(FeatureTokenIndex).where(
                    FeatureTokenIndex.run_id == run.id,
                    FeatureTokenIndex.normalized_token.in_(seed_tokens),
                    FeatureTokenIndex.feature_id != feature_id,
                )
            )
            for row in candidates_result.scalars().all():
                entry = scores.setdefault(
                    row.feature_id, {"score": 0.0, "link_types": set()}
                )
                entry["score"] += row.weight * 0.5
                entry["link_types"].add("shared_token")
                if seed_context and row.context_tokens:
                    ctx = set(row.context_tokens)
                    jaccard = len(seed_context & ctx) / max(len(seed_context | ctx), 1)
                    if jaccard >= min_similarity:
                        entry["score"] += jaccard * 0.5
                        entry["link_types"].add("context")

    # 2) Cached correlation analysis (never computed inline here)
    cache_result = await db.execute(
        select(FeatureAnalysisCache).where(
            FeatureAnalysisCache.feature_id == feature_id,
            FeatureAnalysisCache.analysis_type == AnalysisType.CORRELATIONS,
        )
    )
    cache = cache_result.scalar_one_or_none()
    if cache and isinstance(cache.result, dict):
        for corr in cache.result.get("correlations", [])[:50]:
            related_id = corr.get("feature_id")
            corr_score = float(corr.get("correlation", corr.get("score", 0)) or 0)
            if not related_id or related_id == feature_id:
                continue
            entry = scores.setdefault(related_id, {"score": 0.0, "link_types": set()})
            entry["score"] += corr_score * 0.3
            entry["link_types"].add("correlation")

    if not scores:
        return RelatedFeaturesResponse(seed_feature_id=feature_id, related=[])

    ranked = sorted(scores.items(), key=lambda kv: kv[1]["score"], reverse=True)[:limit]
    features_result = await db.execute(
        select(Feature).where(Feature.id.in_([fid for fid, _ in ranked]))
    )
    feature_map = {f.id: f for f in features_result.scalars().all()}

    related = []
    for fid, entry in ranked:
        f = feature_map.get(fid)
        related.append(
            RelatedFeatureOut(
                feature_id=fid,
                neuron_index=f.neuron_index if f else None,
                name=f.name if f else None,
                category=f.category if f else None,
                score=round(entry["score"], 4),
                link_types=sorted(entry["link_types"]),
            )
        )
    return RelatedFeaturesResponse(seed_feature_id=feature_id, related=related)


@router.post("/feature-groups/runs/{run_id}/cancel")
async def cancel_feature_grouping(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Stop a running feature-grouping precompute at its next feature.

    STARTABLE AND NOT STOPPABLE UNTIL NOW. The TF-IDF pass runs over every
    feature in an extraction, which on a wide dictionary is a long CPU job
    holding the single worker, and its only end was the pod restarting.
    """
    from starlette.concurrency import run_in_threadpool
    from sqlalchemy import select

    from ....core.cancellation import request_cancel
    from ....models.feature_grouping import FeatureGroupingRun

    result = await db.execute(
        select(FeatureGroupingRun).where(FeatureGroupingRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    outcome = await run_in_threadpool(
        request_cancel, "feature_grouping", run_id,
        reason="cancelled by operator",
    )
    if not outcome.requested:
        raise HTTPException(status_code=409, detail=outcome.detail)
    return {"id": run_id, "status": "cancelled", "message": outcome.detail}
