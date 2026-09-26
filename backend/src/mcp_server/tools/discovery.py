"""Discovery tools (category: read) — extractions and trainings."""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
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
