"""
Entry point: ``python -m src.mcp_server`` (streamable HTTP) or ``--stdio``.
"""

import argparse
import logging
import sys

from .config import MCPSettings
from .server import BearerAuthMiddleware, build_server, wrap_tool_with_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("mcp_server")


def main() -> None:
    parser = argparse.ArgumentParser(description="miStudio MCP server")
    parser.add_argument("--stdio", action="store_true", help="Run on stdio (local dev)")
    args = parser.parse_args()

    settings = MCPSettings()
    mcp, client = build_server(settings, stdio=args.stdio)
    wrap_tool_with_audit(mcp)

    if args.stdio:
        logger.info("Starting miStudio MCP server on stdio")
        mcp.run(transport="stdio")
        return

    import uvicorn

    app = mcp.streamable_http_app()
    if settings.auth_token:
        app.add_middleware(BearerAuthMiddleware, token=settings.auth_token)
    else:
        logger.warning("Running WITHOUT authentication (MCP_ALLOW_ANONYMOUS) — local dev only")

    logger.info(
        f"Starting miStudio MCP server on {settings.host}:{settings.port} "
        f"(backend: {settings.api_url})"
    )
    try:
        uvicorn.run(app, host=settings.host, port=settings.port,
                    log_level="info")
    finally:
        # 009 R3: close_backend_clients existed with no production caller —
        # close the backend HTTP clients + gate once the server stops.
        import asyncio

        close = getattr(mcp, "close_backend_clients", None)
        if close is not None:
            asyncio.run(close())


if __name__ == "__main__":
    sys.exit(main())
