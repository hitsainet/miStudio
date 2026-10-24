"""Discovery tools (category: read) — extractions and trainings."""

from typing import Annotated, Any, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def list_extractions(
        status_filter: Annotated[Optional[str], Field(description="Comma-separated statuses, e.g. 'completed,failed'")] = None, limit: Annotated[int, Field(description="Max rows to return")] = 50, offset: Annotated[int, Field(description="Rows to skip, for paging")] = 0
    ) -> Any:
        """List feature-extraction jobs (the entry point for feature analysis).

        Each extraction harvested features from one SAE. Use its id with
        search_features, feature-group tools, and by-token queries.
        status_filter: comma-separated statuses (e.g. 'completed'). limit ≤ 100.
        """
        return await client.get(
            "/extractions", status_filter=status_filter, limit=min(limit, 100), offset=offset
        )

    @mcp.tool()
    async def get_extraction_summary(extraction_id: Annotated[str, Field(description="Extraction job id (extr_xxxxxxxx) from list_extractions — features belong to an EXTRACTION/SAE, not a training")]) -> Any:
        """Get one extraction job's detail: SAE, status, feature counts, config."""
        return await client.get(f"/extractions/{extraction_id}")

    @mcp.tool()
    async def list_trainings(limit: Annotated[int, Field(description="Max rows to return")] = 50, offset: Annotated[int, Field(description="Rows to skip, for paging")] = 0) -> Any:
        """List SAE training runs (id, model, status, hyperparameters, metrics).

        `/trainings` pages by page number, not offset (review R2D-2: the offset this tool
        used to send was ignored, so every page was page 1). The offset is rounded down to
        the start of its page and the rows before it on that page are dropped, so an offset
        that is not a multiple of limit returns fewer than limit rows.
        """
        limit = max(1, min(limit, 100))
        page_index, rows_before_offset = divmod(max(0, offset), limit)
        result = await client.get("/trainings", limit=limit, page=page_index + 1)
        if rows_before_offset and isinstance(result, dict) and isinstance(result.get("data"), list):
            result = {**result, "data": result["data"][rows_before_offset:]}
        return result

    # ── 2026-10-19 (MCP gap closure): the ids every other tool asks for ───────────────────
    #
    # `register_probe_dataset` takes a dataset id and steering/circuit tools take an SAE id, and
    # no tool could list either: an agent had to be handed them.

    @mcp.tool()
    async def get_training(
        training_id: Annotated[str, Field(description="SAE training id (train_xxxxxxxx) from list_trainings")],
    ) -> Any:
        """One SAE training run: model, dataset, hyperparameters, status, progress and final
        metrics."""
        return await client.get(f"/trainings/{training_id}")

    @mcp.tool()
    async def list_datasets(
        search: Annotated[Optional[str], Field(description="Match name or HuggingFace repo id")] = None,
        status: Annotated[Optional[str], Field(description="downloading | processing | ready | error")] = None,
        page: Annotated[int, Field(description="Page number, from 1", ge=1)] = 1,
        limit: Annotated[int, Field(description="Rows per page (1-100)", ge=1, le=100)] = 50,
    ) -> Any:
        """List miStudio's downloaded datasets. Their `id` (a UUID) is what
        `register_probe_dataset` takes as `dataset_id`."""
        return await client.get("/datasets", search=search, status=status, page=page,
                                limit=limit)

    @mcp.tool()
    async def get_dataset(
        dataset_id: Annotated[str, Field(description="Dataset id (UUID) from list_datasets")],
    ) -> Any:
        """One dataset: source, revision, splits, size and status. Check the columns and splits
        here before registering a probe dataset view over it."""
        return await client.get(f"/datasets/{dataset_id}")

    @mcp.tool()
    async def list_saes(
        search: Annotated[Optional[str], Field(description="Match name, description or HuggingFace repo")] = None,
        source: Annotated[Optional[str], Field(description="huggingface | local | trained")] = None,
        model_name: Annotated[Optional[str], Field(description="Only SAEs for this model name")] = None,
        limit: Annotated[int, Field(description="Rows to return (1-100)", ge=1, le=100)] = 50,
        skip: Annotated[int, Field(description="Rows to skip, for paging", ge=0)] = 0,
    ) -> Any:
        """List miStudio's SAEs (sae_xxxxxxxx) — the id steering, circuit and feature-index
        tools take. NOTE: miLLM's SAE ids are a DIFFERENT namespace."""
        return await client.get("/saes", search=search, source=source, model_name=model_name,
                                limit=limit, skip=skip)

    @mcp.tool()
    async def get_sae(
        sae_id: Annotated[str, Field(description="miStudio SAE id (sae_xxxxxxxx)")],
    ) -> Any:
        """One SAE: model, layer, hook, width, architecture, normalisation and where it came
        from (trained, downloaded, or published to HuggingFace)."""
        return await client.get(f"/saes/{sae_id}")
