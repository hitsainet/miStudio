"""The MCP server must import and build in a process that has no backend settings.

`mistudio-mcp` runs `python -m src.mcp_server` in a pod with `MCP_AUTH_TOKEN` and the product
URLs, and no `DATABASE_URL`, `SECRET_KEY` or Celery settings. On 2026-10-23 two tool modules
imported `HEX64` from `src.schemas.probe_definition`; the `src.schemas` package `__init__` loads the
models and `src.core.database`, so the new pod crash-looped on `Settings` validation while every
test passed in a process that had the backend's environment. `test_mcp_tools_stay_on_http.py`
reads import STATEMENTS and allowed `src.schemas`; this test checks what actually LOADS.
"""

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]

#: Backend packages that need the database settings, directly or through their imports.
FORBIDDEN = ("src.core.config", "src.core.database", "src.models", "src.services", "src.workers")

PROBE = """
import sys
from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_server
build_server(MCPSettings())
loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'src' and m.startswith(%r))
print('LOADED=' + ','.join(loaded))
""" % (FORBIDDEN,)


def _clean_env() -> dict[str, str]:
    """The pod's kind of environment: no backend settings at all."""
    keep = {"PATH", "HOME", "LANG", "PYTHONPATH", "VIRTUAL_ENV", "TMPDIR"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env["MCP_AUTH_TOKEN"] = "x" * 32
    env["PYTHONPATH"] = str(BACKEND)
    return env


def test_the_mcp_server_builds_without_backend_settings():
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=BACKEND, env=_clean_env(),
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        "the MCP server could not be built without the backend's settings — the pod would "
        f"crash-loop:\n{proc.stderr[-3000:]}"
    )
    line = next(l for l in proc.stdout.splitlines() if l.startswith("LOADED="))
    loaded = [m for m in line.removeprefix("LOADED=").split(",") if m]
    assert not loaded, f"building the MCP server loaded backend modules it cannot run: {loaded}"


def test_the_shared_patterns_equal_their_backend_source():
    from src.mcp_server import patterns
    from src.schemas.probe_definition import HEX64

    assert patterns.HEX64 == HEX64
