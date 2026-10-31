"""Operator code rules in a rules world (Feature 040 S4): a rule set whose MATCH is one code rule, its document,
and the operator's approval through the real routes (FTDD Appendix C.5, C.6).

`code_set(seed, key)` — one ``code_rule`` signal (its stub params count occurrences of the unmasked value, so the
stub runner reproduces the masking set's MATCH on `rules_world`'s rows) and one `fired` output. No vetoes, no cases.
`approve(client, doc)` files the request (`POST /code-rules/{sha}/approval-requests`, the stub runs its one case twice)
and approves it as the operator (`POST /mcp/approvals/{id}/approve`, the RECORD branch).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from tests.support import rule_set_fixtures as rf
from tests.support import stub_code_runner as stub
from tests.support.random_behaviour import rng_for

API = "/api/v1"


def code_set(seed: str, behaviour_key: str, *, params: Optional[Dict[str, Any]] = None,
             function: str = "count_hits") -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    """`(rule set, code document, signal id, match expression)`."""
    rng = rng_for(seed + ":code")
    doc = stub.document(function, statement=f"fixture {rf.ident(rng, 'd')}")
    sid, expr, col = rf.ident(rng, "k"), rf.ident(rng, "e"), rf.ident(rng, "c")
    signal = stub.code_signal(sid, doc, params if params is not None else {"needle": rf.UNMASKED})
    rule_set = rf.document(behaviour_key, rng, signals=[signal], expressions={expr: {"signal": sid}},
                           outputs=[rf.fired(col, expr)])
    return rule_set, doc, sid, expr


def case_for(doc: Dict[str, Any], *, text: str = "x x", needle: str = "x", count: int = 2) -> Dict[str, Any]:
    return {"id": "c1", "function": doc["functions"][0]["name"], "params": {"needle": needle}, "text": text,
            "expect": {"count": count}}


async def file_request(client, doc: Dict[str, Any], *, headers: Optional[Dict[str, str]] = None,
                       cases: Optional[list] = None) -> Any:
    from src.rule_runner import contract

    sha = contract.code_sha256(doc)
    return await client.post(f"{API}/code-rules/{sha}/approval-requests",
                             json={"document": doc, "cases": cases or [case_for(doc)]}, headers=headers or {})


async def approve(client, doc: Dict[str, Any]) -> Tuple[str, str]:
    from src.rule_runner import contract

    filed = await file_request(client, doc, headers={"X-MiStudio-Agent": "mcp"})
    assert filed.status_code == 202, filed.text
    request_id = filed.json()["approval_request_id"]
    done = await client.post(f"{API}/mcp/approvals/{request_id}/approve")
    assert done.status_code == 200, done.text
    return contract.code_sha256(doc), request_id
