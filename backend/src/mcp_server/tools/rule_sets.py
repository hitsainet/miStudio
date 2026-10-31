"""Rule-set and rules ∪ probe tools (category: probe_monitors) — Feature 040 S3, FR-32, FR-37, FR-39; FTDD §6.7.

Pure HTTP through the miStudio client: this module imports nothing from the backend, because the `mistudio-mcp`
pod has no database (miStudio memory `mcp-server-is-a-separate-deployment`). Every tool is CPU work — none calls
an approval gate. Results and refusals are data: a `code`, the reasons as fields, and `next` calls.
Descriptions use `<behaviour-key>` placeholders only.

Feature 040 S4 (FTDD Appendix C.5, C.6): `validate_rule_set` and `evaluate_rule_signals` forward inline
`code_rules` documents, and `request_code_rule_approval` FILES a request for the operator to approve one code hash
(its own cases run in the isolated runner first). No tool approves, revokes or re-approves: those routes refuse an
agent (`X-MiStudio-Agent`, 403) and have no tool by design (`test_code_rules_api.py` derives that from the live
registry).
"""

from typing import Annotated, Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import MiStudioClient
from ..config import MCPSettings

RULE_SET = Annotated[Optional[Dict[str, Any]], Field(description="A `midataworks.rule-set/v1` document, inline (instead of behaviour_key)")]
BEHAVIOUR = Annotated[Optional[str], Field(description="Use this behaviour's rule set, e.g. '<behaviour-key>' (instead of rule_set)")]
VERSION = Annotated[Optional[int], Field(description="The behaviour version whose rule set is used; default the current one", ge=1)]
CODE_RULES = Annotated[Optional[List[Dict[str, Any]]], Field(description="Operator code the set names, inline: [{document: <midataworks.runner/v1 code-rule document>}]. Stored by hash; it runs only through the isolated runner and only once the operator approved that hash")]


def _choice(behaviour_key: Optional[str], version: Optional[int], rule_set: Optional[Dict[str, Any]],
            code_rules: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if code_rules is not None:
        body["code_rules"] = list(code_rules)
    if behaviour_key is not None:
        body["behaviour_key"] = behaviour_key
    if version is not None:
        body["version"] = version
    if rule_set is not None:
        body["rule_set"] = rule_set
    return body


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:  # noqa: ARG001
    @mcp.tool()
    async def validate_rule_set(
        behaviour_key: BEHAVIOUR = None,
        version: VERSION = None,
        rule_set: RULE_SET = None,
        code_rules: CODE_RULES = None,
    ) -> Any:
        """Check a rule set with miStudio's own engine: the vendored rule-set schema, every pattern compiled
        under the memory cap, and every case the set carries. Answers `valid: true` with the engine digest,
        the match expressions and whether it has veto signals — or `valid: false` with the `code` and, for
        failing cases, `failing_cases` (index, id, and the column that disagreed). Operator code
        (`code_rule`) is accepted: with no runner its expressions are listed `pending_expressions`; with one,
        each code hash needs its document and the operator's approval (`code_rule_not_approved` names
        `request_code_rule_approval`). Stores nothing but inline code documents."""
        return await client.post("/probe-monitors/rule-sets/validate",
                                 json_body=_choice(behaviour_key, version, rule_set, code_rules))

    @mcp.tool()
    async def evaluate_rule_signals(
        behaviour_key: BEHAVIOUR = None,
        version: VERSION = None,
        rule_set: RULE_SET = None,
        code_rules: CODE_RULES = None,
        text: Annotated[Optional[str], Field(description="One text (instead of messages)")] = None,
        messages: Annotated[Optional[List[Dict[str, Any]]], Field(description="A conversation [{role, content}] (instead of text)")] = None,
        scope: Annotated[Optional[str], Field(description="Which messages the rules read: all, content, user, assistant, last_assistant, input, last_user, or the window prompt/response; default all")] = None,
    ) -> Any:
        """Try a rule set on ONE text or conversation, synchronously: `rule_match`, `rule_veto` (a veto holds only
        when no match fires), `unknown` with its reason (`rule_budget_exceeded`, or `runner_unavailable` for
        operator code no runner can run, or a runner reason such as `rule_timeout`), the signals that fired with
        their spans `{message_index, start, end}` in characters — an operator code rule that fired carries its
        `count` and `spans: null` — and both digests. Nothing is stored or logged."""
        body = _choice(behaviour_key, version, rule_set, code_rules)
        if text is not None:
            body["text"] = text
        if messages is not None:
            body["messages"] = list(messages)
        if scope is not None:
            body["scope"] = scope
        return await client.post("/probe-monitors/rule-signals/evaluate", json_body=body)

    @mcp.tool()
    async def evaluate_detector_with_rules(
        probe_id: Annotated[str, Field(description="Probe id (pm_…) of a behaviour")],
        view_ids: Annotated[List[str], Field(description="Views the probe was evaluated on (pmd_…)", min_length=1)],
        version: VERSION = None,
        calibration: Annotated[bool, Field(description="Also cut the union bar on the probe's calibration negatives (needs recorded calibration positions)")] = True,
    ) -> Any:
        """Judge the behaviour's rule set and this probe TOGETHER on the views: rules alone, probe alone and
        rules or probe (a row is flagged when a rule matches, or the probe fires and no veto holds), overall and
        per declared group, with unknown rows counted and producer rule columns compared (never preferred).
        With `calibration`, the union bar that spends the false-positive budget as a union is reported (never
        written). A CPU background task, no approval: returns `evaluation_id` and `task_queue_id` — poll
        get_detector_rules_evaluation or get_task_status. A duplicate request returns the existing one."""
        body: Dict[str, Any] = {"view_ids": list(view_ids), "calibration": calibration}
        if version is not None:
            body["version"] = version
        return await client.post(f"/probe-monitors/probes/{probe_id}/rules-evaluations", json_body=body)

    @mcp.tool()
    async def get_detector_rules_evaluation(
        evaluation_id: Annotated[str, Field(description="Rules evaluation id (pre_…)")],
    ) -> Any:
        """One rules evaluation: its status and progress while it runs; once completed its stored result,
        byte-for-byte, with `stale` computed on read (which digest moved — the rule set, the engine's
        vendored vectors, the probe's decision — and the rerun `next`)."""
        return await client.get(f"/probe-monitors/rules-evaluations/{evaluation_id}")

    @mcp.tool()
    async def list_detector_rules_evaluations(
        key: Annotated[str, Field(description="Behaviour key, e.g. '<behaviour-key>'")],
        limit: Annotated[Optional[int], Field(description="Rows per page", ge=1)] = None,
        cursor: Annotated[Optional[str], Field(description="next_cursor of the previous page")] = None,
    ) -> Any:
        """A behaviour's rules evaluations, newest first, paged (`next_cursor`, `total`)."""
        return await client.get(f"/probe-monitors/behaviours/{key}/rules-evaluations", limit=limit, cursor=cursor)

    @mcp.tool()
    async def cancel_detector_rules_evaluation(
        evaluation_id: Annotated[str, Field(description="Rules evaluation id (pre_…)")],
    ) -> Any:
        """Stop a pending or running rules evaluation. The worker stops between chunks of rows and leaves a
        `cancelled` record; nothing partial is published."""
        return await client.post(f"/probe-monitors/rules-evaluations/{evaluation_id}/cancel")

    @mcp.tool()
    async def request_code_rule_approval(
        code_sha256: Annotated[str, Field(description="The code rule's hash (the code_sha256 a code_rule signal names)", pattern="^[0-9a-f]{64}$")],
        cases: Annotated[List[Dict[str, Any]], Field(description="The code rule's own cases: [{id, function, params, text, expect: {count} | {reason}}]", min_length=1)],
        document: Annotated[Optional[Dict[str, Any]], Field(description="The midataworks.runner/v1 code-rule document, when miStudio does not hold it yet (must hash to code_sha256)")] = None,
    ) -> Any:
        """Ask the operator to approve ONE operator code rule (by its hash) to run in miStudio's isolated runner.
        Its own cases are run first in the runner at the contract's floor limits, each on two fresh instances; a
        failing or non-deterministic case is refused at once and nothing is filed. Otherwise answers
        `{approval_request_id, status: "pending_approval"}`: the operator reads the source and decides in
        miStudio's approvals banner — you cannot approve it. Already approved: `status: "approved"`. Poll
        get_approval_status, then retry the call that was refused `code_rule_not_approved`."""
        body: Dict[str, Any] = {"cases": list(cases)}
        if document is not None:
            body["document"] = document
        return await client.post(f"/code-rules/{code_sha256}/approval-requests", json_body=body)
