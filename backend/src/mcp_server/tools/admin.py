"""DESTRUCTIVE admin tools (category: admin) — disabled by default (BR-5.3).

Enable by adding 'admin' to MCP_TOOL_CATEGORIES. Every tool here permanently
deletes data.
"""

from typing import Annotated, Any

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def delete_experiment(experiment_id: Annotated[str, Field(description="Saved experiment id from list_experiments")]) -> Any:
        """DESTRUCTIVE: permanently delete a saved steering experiment."""
        return await client.delete(f"/steering/experiments/{experiment_id}")

    @mcp.tool()
    async def delete_extraction(extraction_id: Annotated[str, Field(description="Extraction job id (extr_xxxxxxxx) from list_extractions — features belong to an EXTRACTION/SAE, not a training")]) -> Any:
        """DESTRUCTIVE: permanently delete an extraction job AND every feature,
        label, and activation example derived from it. Irreversible."""
        return await client.delete(f"/extractions/{extraction_id}")
