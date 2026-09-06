"""Cross-feature grouping tools (category: groups) — Feature 010's new capability."""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def compute_feature_groups(
        extraction_id: Annotated[str, Field(description="Extraction job id (extr_xxxxxxxx) from list_extractions — features belong to an EXTRACTION/SAE, not a training")],
        similarity_threshold: Annotated[float, Field(description="Cosine similarity floor for grouping; higher = tighter, smaller groups")] = 0.35,
        min_group_size: Annotated[int, Field(description="Ignore groups smaller than this")] = 2,
        force: Annotated[bool, Field(description="Proceed even when a prior result exists")] = False,
    ) -> Any:
        """Start the grouping precompute job for an extraction (background job).

        Builds a token→feature index and splits shared-token buckets into
        context-similarity subgroups. Poll get_task_status or
        get_feature_groups afterwards. Idempotent unless force=true.
        Returns 409 detail if a run is already computing.
        """
        return await client.post(
            f"/extractions/{extraction_id}/feature-groups/compute",
            json_body={
                "params": {
                    "similarity_threshold": similarity_threshold,
                    "min_group_size": min_group_size,
                },
                "force": force,
            },
        )

    @mcp.tool()
    async def get_grouping_status(extraction_id: Annotated[str, Field(description="Extraction job id (extr_xxxxxxxx) from list_extractions — features belong to an EXTRACTION/SAE, not a training")]) -> Any:
        """State of the grouping index: none | pending | computing | completed | failed,
        with progress, params, and counts."""
        return await client.get(f"/extractions/{extraction_id}/feature-groups/status")

    @mcp.tool()
    async def get_feature_groups(
        extraction_id: Annotated[str, Field(description="Extraction job id (extr_xxxxxxxx) from list_extractions — features belong to an EXTRACTION/SAE, not a training")],
        token: Annotated[Optional[str], Field(description="Top activating token to match")] = None,
        search: Annotated[Optional[str], Field(description="Free-text filter over labels and descriptions")] = None,
        min_group_size: Annotated[int, Field(description="Ignore groups smaller than this")] = 2,
        sort_by: Annotated[str, Field(description="Sort field — see this tool's description for the accepted values")] = "size",
        limit: Annotated[int, Field(description="Max rows to return")] = 50,
        offset: Annotated[int, Field(description="Rows to skip, for paging")] = 0,
    ) -> Any:
        """List feature groups (features sharing a top activating token with similar
        context). token = exact normalized match; search = substring.
        sort_by: size | cohesion | token. Paginated, limit ≤ 100."""
        return await client.get(
            f"/extractions/{extraction_id}/feature-groups",
            token=token,
            search=search,
            min_group_size=min_group_size,
            sort_by=sort_by,
            limit=min(limit, 100),
            offset=offset,
        )

    @mcp.tool()
    async def get_feature_group_members(
        extraction_id: Annotated[str, Field(description="Extraction job id (extr_xxxxxxxx) from list_extractions — features belong to an EXTRACTION/SAE, not a training")],
        group_id: Annotated[str, Field(description="Feature-group id from get_feature_groups (requires the grouping index)")],
        category: Annotated[Optional[str], Field(description="Filter by label category")] = None,
        has_label: Annotated[Optional[bool], Field(description="Filter to features that do (or do not) have a label")] = None,
        star_color: Annotated[Optional[str], Field(description="'yellow' (starred) | 'purple' (in flight) | 'aqua' (completed, protected from bulk overwrite)")] = None,
    ) -> Any:
        """Members of one group with current labels, stars, stats, and a context
        snippet each (labels are live — never stale)."""
        return await client.get(
            f"/extractions/{extraction_id}/feature-groups/{group_id}",
            category=category,
            has_label=has_label,
            star_color=star_color,
        )

    @mcp.tool()
    async def find_features_by_token(
        extraction_id: Annotated[str, Field(description="Extraction job id (extr_xxxxxxxx) from list_extractions — features belong to an EXTRACTION/SAE, not a training")],
        token: Annotated[str, Field(description="Top activating token to match")],
        match: Annotated[str, Field(description="'exact' (raw surface form) | 'normalized' (case/whitespace folded)")] = "normalized",
        limit: Annotated[int, Field(description="Max rows to return")] = 50,
        offset: Annotated[int, Field(description="Rows to skip, for paging")] = 0,
    ) -> Any:
        """Features whose top activating token matches. match: exact (raw surface
        form) | normalized (case/BPE-marker-insensitive) | prefix. Requires the
        grouping index (compute_feature_groups first; 409 NO_INDEX otherwise)."""
        return await client.get(
            f"/extractions/{extraction_id}/features/by-token",
            token=token,
            match=match,
            limit=min(limit, 100),
            offset=offset,
        )

    @mcp.tool()
    async def find_related_features(
        feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")], min_similarity: Annotated[float, Field(description="Cosine similarity floor for relatedness; higher = fewer, closer matches")] = 0.2, limit: Annotated[int, Field(description="Max rows to return")] = 50
    ) -> Any:
        """Features related to a seed feature via shared tokens, context overlap,
        and cached correlations. Each result carries link_types explaining the
        connection."""
        return await client.get(
            f"/features/{feature_id}/related",
            min_similarity=min_similarity,
            limit=min(limit, 100),
        )
