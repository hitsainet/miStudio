"""Saved-experiment tools (category: experiments) — persist steering evidence."""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def save_experiment(
        name: Annotated[str, Field(description="Human-readable name")],
        comparison_id: Annotated[str, Field(description="Comparison id lifted out of a get_steering_result payload")],
        result: Annotated[dict[str, Any], Field(description="The full payload returned by get_steering_result")],
        description: Annotated[Optional[str], Field(description="Longer free-text description")] = None,
        tags: Annotated[Optional[list[str]], Field(description="Free-form tags for filtering")] = None,
    ) -> Any:
        """Save a steering result as a persistent experiment record.

        comparison_id: unique id of the run being saved (e.g. the sweep_id or
        comparison_id from the steering result) — duplicates are rejected (409).
        result: the full steering result payload plus your findings/conclusion.
        Reference the returned experiment id in feature notes as evidence:
        '[MCP <date>] evidence: experiment <id> — <summary>'."""
        return await client.post(
            "/steering/experiments",
            json_body={
                "name": name,
                "comparison_id": comparison_id,
                "result": result,
                "description": description,
                "tags": tags,
            },
        )

    @mcp.tool()
    async def list_experiments(limit: Annotated[int, Field(description="Max rows to return")] = 50, offset: Annotated[int, Field(description="Rows to skip, for paging")] = 0) -> Any:
        """List saved steering experiments (paginated, limit ≤ 100)."""
        return await client.get("/steering/experiments", limit=min(limit, 100), offset=offset)

    @mcp.tool()
    async def get_experiment(experiment_id: Annotated[str, Field(description="Saved experiment id from list_experiments")]) -> Any:
        """Get one saved experiment with its full stored data."""
        return await client.get(f"/steering/experiments/{experiment_id}")
