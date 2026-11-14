"""`decide_in` comes from the product row's `approval_owner` (042 FR-042.16, TPD-8; FTID §3.10; FTASKS 12.7 – 12.8,
control K-12.34).

`ProductSpec.approval_owner` was read by nothing before the P3 cut (FTDD §17 C1). It is wired now: a relayed pending
answer carries `decide_in: <row name>` only for a row whose owner is `"product"` (the product's own queue decides).
Through the LIVE registry and the real `MiServerClient` on a fake miServer (the `test_millm_approvals_relay.py`
harness), every relayed tool's pending answer says `decide_in: "miserver"` — the row's name, not a literal.
"""
from __future__ import annotations

import pytest

from tests.unit.test_millm_approvals_relay import _ACTION_FOR, RELAYED, _MiLLM, _pending, live  # noqa: F401


@pytest.mark.parametrize("tool", sorted(RELAYED))
def test_decide_in_from_approval_owner(live, tool):  # noqa: F811
    from src.mcp_server.products import MISERVER

    assert MISERVER.approval_owner == "product"
    args, (method, path, _body, _params) = RELAYED[tool]
    call, _ = live(_MiLLM())
    result = call(tool, args)
    assert result == {**_pending(_ACTION_FOR[(method, path)]), "decide_in": MISERVER.name}
    assert result["decide_in"] == "miserver"
