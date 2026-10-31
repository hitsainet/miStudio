"""Feature query tools (category: read) — search, detail, per-feature analysis."""

from typing import Annotated, Any, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def search_features(
        extraction_id: Annotated[str, Field(description="Extraction job id (extr_xxxxxxxx) from list_extractions — features belong to an EXTRACTION/SAE, not a training")],
        search: Annotated[Optional[str], Field(description="Free-text filter over labels and descriptions")] = None,
        category: Annotated[Optional[str], Field(description="Filter by label category")] = None,
        is_favorite: Annotated[Optional[bool], Field(description="Filter to starred features only")] = None,
        sort_by: Annotated[str, Field(description="Sort field — see this tool's description for the accepted values")] = "activation_freq",
        sort_order: Annotated[str, Field(description="'asc' | 'desc'")] = "desc",
        limit: Annotated[int, Field(description="Max rows to return")] = 50,
        offset: Annotated[int, Field(description="Rows to skip, for paging")] = 0,
    ) -> Any:
        """Search/filter features within an extraction.

        search matches name/description. sort_by: activation_freq | max_activation |
        feature_id | name | category. Results are paginated (limit ≤ 100) —
        never request the full feature set; narrow with filters instead.
        """
        return await client.get(
            f"/extractions/{extraction_id}/features",
            search=search,
            category=category,
            is_favorite=is_favorite,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=min(limit, 100),
            offset=offset,
        )

    @mcp.tool()
    async def get_feature(feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")]) -> Any:
        """Full feature detail: label, category, description, notes, statistics, star state."""
        return await client.get(f"/features/{feature_id}")

    @mcp.tool()
    async def get_feature_examples(feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")], limit: Annotated[int, Field(description="Max rows to return")] = 20) -> Any:
        """Top activating examples with per-token activations — the primary
        evidence for what a feature detects. limit ≤ 100."""
        return await client.get(f"/features/{feature_id}/examples", limit=min(limit, 100))

    @mcp.tool()
    async def get_feature_token_analysis(feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")]) -> Any:
        """Aggregated token statistics for a feature (ranked token frequencies)."""
        return await client.get(f"/features/{feature_id}/token-analysis")

    @mcp.tool()
    async def get_feature_logit_lens(feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")]) -> Any:
        """Logit-lens: vocabulary tokens this feature promotes/suppresses when active."""
        return await client.get(f"/features/{feature_id}/logit-lens")

    @mcp.tool()
    async def get_feature_correlations(feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")]) -> Any:
        """Features correlated with this one (token overlap + activation-stat similarity).
        May take a few seconds on first call; results are cached server-side."""
        return await client.get(f"/features/{feature_id}/correlations")

    @mcp.tool()
    async def get_feature_ablation(feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")]) -> Any:
        """STATISTICAL-ESTIMATE ablation impact for a feature (method=
        "statistical_estimate" in the response) — scored from activation
        frequency/magnitude/consistency, NOT a model forward pass. This is a
        heuristic, NOT causal evidence. For a real causal measurement (suppress
        the feature, run the model, measure the effect vs a null), use the
        circuit validation tier (validate_circuit_edges — Feature 017, rung 2)."""
        return await client.get(f"/features/{feature_id}/ablation")

    @mcp.tool()
    async def get_feature_nlp_analysis(feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")]) -> Any:
        """Stored NLP analysis (prime-token distribution, POS tags, context patterns,
        semantic clusters), if it has been computed for this feature."""
        return await client.get(f"/features/{feature_id}/nlp-analysis")

    @mcp.tool()
    async def get_feature_by_index(
        feature_idx: Annotated[int, Field(description="Feature INDEX within the SAE (0..n_features-1) — the number steering and circuits use", ge=0)],
        sae_id: Annotated[Optional[str], Field(description="miStudio SAE id (sae_xxxxxxxx). Give exactly one of sae_id or training_id")] = None,
        training_id: Annotated[Optional[str], Field(description="SAE training id (train_xxxxxxxx). Give exactly one of sae_id or training_id")] = None,
    ) -> Any:
        """Resolve a feature INDEX to its feature row id, so the index a steering result,
        circuit or allocation names can be fed to get_feature / get_feature_examples.

        Indexes and row ids are different namespaces: an index is a position in one SAE, a row
        id names one extracted feature. Feature rows exist only once features were EXTRACTED
        for that SAE or training — a 404 here usually means no extraction yet, not a bad index.
        """
        if (sae_id is None) == (training_id is None):
            return {"error": "exactly_one_owner",
                    "message": "give exactly one of sae_id or training_id"}
        if sae_id is not None:
            return await client.get(f"/saes/{sae_id}/features/by-index/{feature_idx}")
        return await client.get(f"/trainings/{training_id}/features/by-index/{feature_idx}")
