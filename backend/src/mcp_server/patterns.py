"""Argument patterns the MCP tools share, with no import of the backend application.

The MCP server runs in its own pod with no database settings, so a tool may not import anything
that loads `src.core` or `src.models`. Importing one constant from `src.schemas` did exactly that
(the package `__init__` loads the models and the database layer), and the pod crash-looped on
2026-10-23. Values here are pinned equal to their backend source by
`tests/unit/test_mcp_server_starts_without_the_backend.py`.
"""

#: 64 lowercase hex characters: a sha256 digest (same value as `src.schemas.probe_definition.HEX64`).
HEX64 = r"^[0-9a-f]{64}$"

#: A miWorkbench probe id: `pm_` and lowercase hex (`src.models.probe_monitor._pm_id` makes `pm_` + 12 hex). Checked
#: before an id goes into a URL path, so `../` or a query string never reaches miWorkbench. Pinned against the generator
#: by `tests/unit/test_probe_definition_handoff.py`, so a change of id format cannot start refusing real ids.
PROBE_MONITOR_ID = r"^pm_[0-9a-f]+$"
