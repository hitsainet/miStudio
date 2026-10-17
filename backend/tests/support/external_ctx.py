"""The `ExternalContext` tests hand an external tool module's `register` (Feature 034, TD8).

Every external module registers as `register(mcp, client, gate, ctx)`; `ctx` is REQUIRED on the
real signature, so a test that registers one must build a context too — a stand-in signature that
let it be omitted would be more forgiving than production.
"""

from typing import Any
from unittest.mock import MagicMock

from src.mcp_server.config import MCPSettings
from src.mcp_server.context import ExternalContext


def make_ctx(mistudio: Any = None, *, identity: str = "agent:mistudio-mcp",
             **settings: Any) -> ExternalContext:
    """A context whose settings take `settings` as `MCPSettings` fields."""
    return ExternalContext(
        settings=MCPSettings(allow_anonymous=True, **settings),
        mistudio=mistudio if mistudio is not None else MagicMock(),
        identity=identity,
    )
