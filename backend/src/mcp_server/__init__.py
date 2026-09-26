"""
miStudio MCP server (Feature 010).

Exposes the post-extraction workflow (feature analysis, cross-feature
grouping, steering, labeling write-back) as MCP tools for agentic clients.
Runs as a separate process (``python -m src.mcp_server``) and talks to the
backend exclusively through the ``/api/v1`` REST API.
"""
