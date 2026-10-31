"""Runtime environment settings, read-only (category: read; Feature 036, ENH-003 D5).

D5: every MCP server gets a READ tool for its app's runtime settings; miStudio has NO write tool in
v1, because its approval flow is caller-executed and a settings write needs a backend executor. The
operator changes these in Settings → Environment. The listing reaches the agent exactly as the
backend serves it: secrets are already reduced to "set / not set" there.
"""

from typing import Any

from mcp.server.mcpserver import MCPServer

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def get_environment_settings() -> Any:
        """miStudio's runtime settings and where each value comes from. Every setting the
        Settings → Environment tab shows: the value in force, the environment's value, whether
        an operator override is set, when a change takes effect, and which settings need a pod
        restart. Read-only for agents (no MCP write path in v1); secrets appear only as set or
        not set."""
        return await client.get("/environment")
