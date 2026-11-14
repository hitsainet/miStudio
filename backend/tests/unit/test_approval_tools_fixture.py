"""The frontend's approval-tool fixture is generated from the backend's authorities (042 FR-042.28; FTASKS 12.23).

`frontend/src/components/approvals/approvalTools.fixture.json` is what `approvalLabels.test.ts` requires a label
for: `approvable` = the live `APPROVABLE_TOOLS` keys of the approval route, `history` = the tool names stored rows
can still carry (the four `millm_*` tools e008 drained — read from the migration module, which writes them out
forever — plus every `history_tool` record of the legacy-identifier export). Regenerate with
`python -m tests.unit.test_approval_tools_fixture` (from `backend/`) when either authority changes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
FIXTURE = BACKEND.parent / "frontend" / "src" / "components" / "approvals" / "approvalTools.fixture.json"


def expected() -> dict[str, list[str]]:
    from src.api.v1.endpoints.mcp_approvals import APPROVABLE_TOOLS
    from tests.support import legacy_export

    path = BACKEND / "alembic" / "versions" / "e008_millm_approvals_moved.py"
    spec = importlib.util.spec_from_file_location("e008_moved", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    history = set(module.MOVED_TOOLS) | {r["old"] for r in legacy_export.load().history_tools()}
    return {"approvable": sorted(APPROVABLE_TOOLS), "history": sorted(history)}


def test_fixture_equals_the_live_authorities():
    data = expected()
    assert data["approvable"] and data["history"], "an authority yielded nothing"
    assert json.loads(FIXTURE.read_text()) == data, "regenerate: python -m tests.unit.test_approval_tools_fixture"


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(expected(), indent=2) + "\n")
    print(f"wrote {FIXTURE}")
