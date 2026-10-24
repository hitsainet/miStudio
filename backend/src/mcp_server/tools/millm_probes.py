"""
miLLM probe-monitor tools (category: millm_probes) — Feature 24 / 033 phase 7.

A probe monitor is a linear detector trained in miStudio: a vector, a threshold, and a
record of how well it actually worked. These tools import one into a miLLM deployment,
arm it against live traffic, and read what it reported.

**Contract:** miLLM `docs/mcp-contract.md` v1.6, §4d and §4d-bis. These tools and
miLLM's `/api/probes/*` routes are a CO-RELEASE; neither half is shippable alone.

Three rules run through this module, and each exists because the alternative reads as a
claim nobody measured:

1. **`rung_language` is returned verbatim, never composed here from `rung`.** miStudio
   owns the vocabulary, miLLM mirrors it, and a third rendering in this layer would be
   free to drift — with the most likely drift being a detector's language rising above
   its evidence. No number→phrase map exists in this file, by construction.

2. **A probe records; it does not act.** Nothing here stops, re-routes or alters a
   generation. A verdict is evidence for whoever reads it, not a gate, and not a cause —
   a probe detects, it does not explain.

3. **Arming below rung 2 needs an explicit acknowledgement**, and that acknowledgement
   is stored separately from the one inside the definition. The agent that exported a
   weak probe and the one arming it against live traffic are not necessarily the same,
   and only the second is choosing to monitor with it.
"""

import inspect
import logging
import time
from typing import Annotated, Any, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from ..patterns import HEX64
from ..client import BackendError
from ..context import ExternalContext
from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient
from ..millm_send import send, sent_headers

logger = logging.getLogger(__name__)

#: miLLM's own per-request cap (`PROBE_SCORE_MAX_INPUTS`, contract §4 `millm_probes` v1.10).
MAX_PROBE_INPUTS = 64


class ProbeScoreInput(BaseModel):
    """One input to score: exactly one of `token_ids` (with optional `prompt_tokens`), `messages`
    or `text` (miLLM FR-27.4a). Checked in the tool, before the gate."""

    model_config = ConfigDict(extra="forbid")

    token_ids: Optional[list[int]] = Field(default=None, min_length=1)
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    messages: Optional[list[dict[str, str]]] = Field(default=None, min_length=1)
    text: Optional[str] = Field(default=None, min_length=1)


SCORE_PROBES_DESCRIPTION = inspect.cleandoc("""
    Score stored inputs with imported probes — armed or not — and get each probe's verdict.

    **It records nothing and changes nothing armed**: no probe event is written, no runtime
    state is touched, no parity report changes. Use it to label or audit data offline with
    exactly the code live serving uses.

    Each input is EXACTLY ONE of `token_ids` (authoritative — what the probe was trained and
    calibrated on; `prompt_tokens` marks where the reply starts), `messages` (rendered as a live
    chat request is), or `text` (refused by miLLM until its render is verified). At most
    {MAX_PROBE_INPUTS} inputs per call; for more, batch them (`/api/probes/score` is a batch
    endpoint).

    `probe_ids` are miLLM's probe ids, not miStudio's `pm_…` ids. Results per input, probe and
    window carry `score`, `threshold`, `verdict`, `rung` and `rung_language` VERBATIM.
    `verdict: null` means the probe SAID NOTHING (no threshold, or nothing scored) — it is never
    false. A per-input `error` and the top-level `skipped` list are returned as they come, so
    one bad input does not hide the others. Probe scoring is unsteered by construction. From
    miLLM Feature 038 each result is also a `mi.probe.result/v1` object (`result_id`, `outcome`,
    `reason`, `decision`, detector identity, `origin: stateless`, `persistence: not_persisted`),
    passed through unchanged.

    `export_results=true` asks miLLM to EXPORT these per-input results to its telemetry backend.
    Only sent when true. miLLM exports only when its deployment also permits it, and then answers
    `export: {requested, permitted, records_queued}` — `permitted: false` is a normal success
    (scoring never fails over telemetry), it just means nothing left the box. A miLLM without
    Feature 038's export phase refuses the field outright (a 422 naming `export_results`) and
    scores nothing: leave it unset against such a miLLM.

    Never loads a model; a 503 returns `retry_after_s`. Shapes: `{"result", "millm_headers"}`;
    `{"refused": <code>, "next", "retry_after_s"}`; `{"unavailable": "millm", "reason"}`;
    `{"error": …}` (your arguments).
""").replace("{MAX_PROBE_INPUTS}", str(MAX_PROBE_INPUTS))


#: The fields a contract `WindowDecision` carries — what may be sent for each window.
CONTRACT_WINDOW_FIELDS = ("threshold", "target_fpr", "realised_fpr", "n_negatives", "scope", "length_bands")


def contract_windows(windows: dict) -> dict:
    """A proposal's `window_decisions` reduced to the contract's fields, per window.

    ⚠ THE PROPOSAL CARRIES miSTUDIO'S BOOKKEEPING — `scores_path`, `lengths_path` — which was
    forwarded verbatim and stored in miLLM's copy of the definition: host paths in another
    service's document, and fields the contract's `WindowDecision` (`extra="forbid"`) refuses.
    Pure dict filtering, so this module stays HTTP-only (`test_mcp_tools_stay_on_http.py`).
    """
    out: dict = {}
    for window, entry in (windows or {}).items():
        if not isinstance(entry, dict):
            continue
        kept = {key: entry[key] for key in CONTRACT_WINDOW_FIELDS if entry.get(key) is not None}
        if kept.get("threshold") is not None:
            out[str(window)] = kept
    return out


#: The one identity algorithm whose detector digest this tool compares with miStudio's (D-T10, R-1):
#: a digest defined differently must not refuse every push.
IDENTITY_ALGORITHM = "mistudio.probe-identity/1"

#: miLLM's OpenAPI document, per process and per miLLM URL: `{url: (fetched_at, fields)}` (D3 (a)).
_OPENAPI_CACHE: dict[str, tuple[float, frozenset]] = {}


def _strip_nulls(value: Any) -> Any:
    """A JSON value with null-valued keys removed at every depth: "absent" and "null" compare equal
    (the contract hashes them equal too, FR-1 A3)."""
    if isinstance(value, dict):
        return {k: _strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_nulls(v) for v in value]
    return value


def decision_differences(agent: dict, decision: dict) -> list[dict]:
    """Field by field, the agent's bar against miStudio's decision IN FORCE (OQ-1 (a)). Floats are
    compared exactly — both sides come from the same JSON. Windows are compared after
    `contract_windows` (bookkeeping stripped); null and absent are equal."""
    out = []
    for field in ("threshold", "target_fpr", "realised_fpr", "threshold_source"):
        if agent.get(field) != decision.get(field):
            out.append({"field": field, "yours": agent.get(field), "in_force": decision.get(field)})
    windows = contract_windows(agent["windows"]) if agent.get("windows") is not None else None
    if _strip_nulls(windows or None) != _strip_nulls(decision.get("windows") or None):
        out.append({"field": "windows", "yours": windows, "in_force": decision.get("windows")})
    if _strip_nulls(agent.get("length_bands") or None) != _strip_nulls(decision.get("length_bands") or None):
        out.append({"field": "length_bands", "yours": agent.get("length_bands"),
                    "in_force": decision.get("length_bands")})
    return out


async def advertised_recalibrate_fields(millm: Any, ttl_seconds: int) -> frozenset:
    """The properties miLLM's `ProbeRecalibrationRequest` advertises, from its OpenAPI document.

    Cached per process and per miLLM URL for `ttl_seconds` (0 = never cached). A failed read is not
    cached and advertises nothing — the push then carries only `calibration_id`, which every miLLM
    accepts.
    """
    key = str(getattr(millm, "base_url", None) or id(millm))
    now = time.monotonic()
    cached = _OPENAPI_CACHE.get(key)
    if ttl_seconds > 0 and cached is not None and now - cached[0] < ttl_seconds:
        return cached[1]
    try:
        document = await millm.get("/openapi.json")
    except BackendError:
        return frozenset()
    schema = (((document or {}).get("components") or {}).get("schemas") or {}).get(
        "ProbeRecalibrationRequest") or {}
    fields = frozenset((schema.get("properties") or {}).keys())
    if ttl_seconds > 0:
        _OPENAPI_CACHE[key] = (now, fields)
    return fields


def _next(tool: str, arguments: dict, why: str, needs: Optional[list] = None) -> dict:
    """One `next` entry, the shape `services/next_calls.py` defines (built here: tools stay HTTP-only)."""
    call = {"tool": tool, "arguments": arguments, "why": why}
    if needs:
        call["needs"] = needs
    return call


def register(
    mcp: MCPServer, millm: MiLLMClient, gate: HealthGate, ctx: ExternalContext
) -> None:
    @mcp.tool()
    # NO @gated: the decorator runs the gate before the body, and an agent debugging its
    # own payload must not be told "millm is down". A gate failure is about the SERVER;
    # an argument error is about the CALL, and conflating them sends the agent to fix the
    # wrong thing (the millm_clusters / millm_circuits precedent).
    async def millm_import_probe(
        definition: Annotated[Optional[dict], Field(description="A `mistudio.probe-definition/v1` document, inline. Mutually exclusive with repo_id")] = None,
        repo_id: Annotated[Optional[str], Field(description="A HuggingFace repo tagged `mistudio-probe-definition`, e.g. 'mistudio/probes-lfm2'. Requires filename")] = None,
        filename: Annotated[Optional[str], Field(description="The `.probe.json` inside that repo")] = None,
        revision: Annotated[Optional[str], Field(description="Hub revision; defaults to the repo's main")] = None,
        on_conflict: Annotated[Optional[str], Field(description="'rename' (default — keep both) | 'fail' (refuse if the name exists). There is NO 'replace'")] = None,
    ) -> Any:
        """Import a probe definition, from a document or from HuggingFace.

        Importing does NOT arm. A probe sits idle until `millm_arm_probe`, which runs
        four gates: the armed-probe limit, model identity, the evidence rung, and parity
        against the scores miStudio recorded.

        `on_conflict` is 'rename' or 'fail'. **There is deliberately no 'replace':**
        overwriting a definition in place while its probe is armed would change the
        detector underneath a running monitor while every event before and after kept
        the same `probe_id` — the history would then describe two different detectors as
        one. To replace a probe, disarm it, delete it, and import the new one.
        """
        if on_conflict not in (None, "rename", "fail"):
            return {"error": "`on_conflict` must be 'rename' or 'fail'. There is no "
                             "'replace' — disarm, delete, then import"}
        if definition is not None and repo_id is not None:
            return {"error": "pass EITHER `definition` or `repo_id`+`filename`, not both"}
        if definition is None and repo_id is None:
            return {"error": "pass either `definition` (the document) or "
                             "`repo_id`+`filename` (a HuggingFace import)"}
        if definition is not None and not isinstance(definition, dict):
            return {"error": "`definition` must be the probe document object"}
        if repo_id is not None and not filename:
            return {"error": "`filename` is required with `repo_id` — a repo publishes "
                             "one definition per layer, and importing the wrong one "
                             "passes every gate except parity"}

        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}

        if definition is not None:
            return await millm.post(
                "/api/probes/import", json_body=definition,
                on_conflict=on_conflict or "rename",
            )
        return await millm.post(
            "/api/probes/hub/import",
            json_body={
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "on_conflict": on_conflict or "rename",
            },
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_list_probes(
        armed: Annotated[Optional[bool], Field(description="True for only the armed probes, False for only the idle ones, omit for all")] = None,
    ) -> Any:
        """Imported probes, with their evidence and their parity report.

        Each row carries `rung` AND `rung_language` — surface the language verbatim;
        do not compose your own phrase from the number. `next_step` says what would
        raise a probe's rung.

        Two fields are `null` for a reason and must not be read as zero or false:
        `threshold` is null when no threshold was placed (the probe **ranks** without
        deciding), and `parity` is null when parity has not been checked here yet —
        which is not the same as having passed.

        `label_mapping` is what the probe was FITTED on — `{raw_label: "positive" |
        "negative" | "excluded"}` from its training view, in the corpus's own strings.
        Read both sides before describing what a probe detects: a probe is a boundary,
        and the same positive label fitted against a different negative one is a
        different detector. `concept` is miStudio's own one-line statement of the
        positive side. Both are absent against a miLLM older than 2026-10-01.
        """
        return await millm.get("/api/probes", armed=armed)

    @mcp.tool()
    # NO @gated — validation before the gate, as above.
    async def millm_arm_probe(
        probe_id: Annotated[str, Field(description="miLLM probe id from millm_list_probes")],
        acknowledge_below_rung2: Annotated[bool, Field(description="Required to arm a probe below rung 2. Asserts that YOU are choosing to monitor live traffic with a detector that has not been shown to work on unseen tasks")] = False,
        reason: Annotated[str, Field(description="Why you are arming it. Recorded with the acknowledgement")] = "",
        windows: Annotated[Optional[list[str]], Field(description="Which slices of a request to report, from 'all', 'prompt', 'response', 'last_user'. Omit for 'all', 'prompt' and 'response', plus 'last_user' when the probe carries its own bar for it; a bar pushed to an ARMED probe for a window it was not armed with is read only after re-arming. 'last_user' reads the newest user message alone; 'prompt' reads everything before the reply (system text and every earlier turn included) and is independent of how long the reply was; 'response' reads the model's own output and is UNTRAINED for probes fitted on user text, so its verdicts come back flagged provisional")] = None,
    ) -> Any:
        """Arm a probe. Four gates run, in the order they are cheapest to fail.

        1. **Limit** — `PROBE_LIMIT` when the deployment's armed cap is reached.
        2. **Identity** — `PROBE_MODEL_MISMATCH` when the loaded model is not the one
           this probe was fitted on. `details.mismatches` names EVERY differing field,
           so you can tell "wrong model loaded" from "wrong probe imported". This gate
           cannot be waived: a probe's weights are a direction in one specific model's
           residual space, and read in another's they produce numbers that are
           plausible, stable, well-behaved and about nothing.
        3. **Evidence rung** — `UNVALIDATED_PROBE` below rung 2 unless
           `acknowledge_below_rung2=True`. The refusal carries `rung`, `rung_language`
           and `next_step`.
        4. **Parity** — `PROBE_PARITY_FAILED` when this build does not reproduce the
           scores miStudio recorded for the definition's test vectors. Note that
           `details.max_abs_diff` may be **null**, meaning no vector could be compared
           at all — which is not "zero off".

        ⚠ `UNVALIDATED_PROBE` and `PROBE_MODEL_MISMATCH` are not interchangeable. The
        first is resolved by asserting intent; the second can never be resolved by
        retrying.

        Arming has a throughput cost: while any probe is armed the deployment may run
        requests serially. Read `millm_probe_status` for what that is costing.

        ⚠ **`windows` IS NOT THE PROBE'S SCOPE AND DOES NOT CHANGE IT.** `scope` is the
        probe's identity — what it was trained on, what its threshold was cut under, and
        the only thing gate 4 can verify. `windows` chooses which slices of a request the
        same weights are READ over, because the prompt says something about the USER and
        the response says something about the MODEL, and a mean over both answers neither.
        A probe whose CONTRACT scope is `prompt` or `response` is still refused here.

        Each window produces its own verdict and its own event, so a request yields up to
        four. `last_user` reads the newest user message alone — use it on any client that resends
        the conversation, where `prompt` keeps firing on an earlier turn that matched. A window is returned **provisional** when it has no threshold of its own (the
        threshold is the (1 - target_fpr) quantile of negatives aggregated under ONE window, so
        over a different one the same number no longer names the same false-positive rate), and
        `response` is provisional whatever bar it carries, because the weights never saw a model
        reply. Treat a provisional verdict as a ranking, not a rate.
        """
        if not isinstance(probe_id, str) or not probe_id.strip():
            return {"error": "`probe_id` is required — get one from millm_list_probes"}
        ok, reason_unavailable = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason_unavailable}
        return await millm.post(
            f"/api/probes/{probe_id}/arm",
            json_body={
                "acknowledge_below_rung2": acknowledge_below_rung2,
                "reason": reason,
                # `None` is sent through deliberately: miLLM reads null as "the default set"
                # and an explicit `[]` as "the probe's own scope alone". Omitting the key
                # entirely would work too, but sending it makes the payload one shape, which
                # is what the reachability assertion pins.
                "windows": windows,
            },
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_disarm_probe(
        probe_id: Annotated[str, Field(description="miLLM probe id from millm_list_probes")],
    ) -> Any:
        """Disarm a probe: it stops scoring and its layer hook is removed.

        Non-destructive — the definition and its recorded events stay. Disarming is
        also the first step of replacing a probe (disarm → delete → import), since
        there is no in-place overwrite.
        """
        return await millm.post(f"/api/probes/{probe_id}/disarm")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_recalibrate_probe(
        probe_id: Annotated[str, Field(description="miLLM probe id from millm_list_probes")],
        mistudio_probe_id: Annotated[str, Field(description="The pm_… id the cut was made for. Compared against the stored definition's provenance.probe_id and REFUSED on a mismatch")],
        threshold: Annotated[Optional[float], Field(description="The re-cut bar. `null` means the probe ranks but does not decide — not a bar of zero")] = None,
        target_fpr: Annotated[Optional[float], Field(description="The false-positive budget it was cut at. Required whenever `threshold` is given")] = None,
        realised_fpr: Annotated[Optional[float], Field(description="What it actually spends, which is usually not the target")] = None,
        threshold_source: Annotated[Optional[str], Field(description="Which negatives it was cut from, e.g. calibration_set. Required whenever `threshold` is given")] = None,
        windows: Annotated[Optional[dict], Field(description="Per-window bars, straight from the proposal's `window_decisions`. Omit if there are none")] = None,
        length_bands: Annotated[Optional[list], Field(description="Per-length bars, straight from the proposal's `length_bands`. Must tile every length from 0 with an open-ended final band")] = None,
        calibration_id: Annotated[Optional[str], Field(description="The producer's id for this cut. Filled with miStudio's decision digest; a different non-null value is refused")] = None,
        reason: Annotated[str, Field(description="Why the bar moved. Recorded in the probe's threshold_history")] = "",
        decision_digest: Annotated[Optional[str], Field(pattern=HEX64, description="The decision digest you hold (from the re-cut's `current`). Checked against miStudio's; filled from it when omitted")] = None,
        decision_revision: Annotated[Optional[int], Field(ge=1, description="The decision revision you hold. Checked against miStudio's; filled from it when omitted")] = None,
    ) -> Any:
        """Push an already re-cut decision bar onto an imported, possibly ARMED miLLM probe.

        TWO STEPS, DELIBERATELY. Call `recalibrate_probe_monitor(<pm_id>, target_fpr)` FIRST: it
        re-cuts the bar in miStudio from the negatives already on disk — no GPU — and returns what
        the candidate bar would do to every evaluation set, plus a `transfer_proposed.caution`.
        Read that, then pass its `proposed` block here.

        ⚠ IT IS TWO TOOLS BECAUSE THIS SERVER CANNOT DO BOTH HALVES. The MCP server is a separate
        deployment from the miStudio backend: it has no database credentials, so a tool that
        imported `SyncSessionLocal` would be registered, documented, unit-tested and dead on
        arrival — this estate's signature defect, which the first version of this tool reproduced.
        Every other tool here reaches its service over HTTP, and so does this one.

        ⚠ AND IT IS TWO STEPS BECAUSE A BAR SHOULD NOT MOVE WITHOUT SOMEBODY READING THE
        CONSEQUENCE. A re-cut is still a quantile of the SAME calibration corpus and does not make
        a threshold transfer between distributions: on the first shipped probe, five evaluation
        sets' own 1% thresholds spanned 24 points. Lowering a bar until a probe fires on the
        traffic in front of you is not calibration.

        ⚠ THIS IS THE ONLY WAY TO CHANGE AN IMPORTED PROBE'S BAR WITHOUT DESTROYING ITS HISTORY.
        The alternative is disarm → delete → import, which assigns a new `probe_id`, renames the
        row, leaves a monitoring gap and CASCADE-DELETES every recorded verdict.

        It moves the BAR, never the detector. `on_conflict=replace` is still refused on import;
        miLLM's route cannot carry a detector field at all, refuses a cut it cannot match to the
        stored `provenance.probe_id`, and refuses a threshold with no budget and no named source.

        The response says whether the LIVE runtime was refreshed (`registry_updated`) and reports a
        row claiming armed with no live entry (`stale_armed`) rather than quietly fixing it — that
        probe is not scoring at all and needs re-arming.

        IDENTITY CHECKS (Feature 039), in order, and NOTHING is posted on any refusal:
        1. miStudio's decision in force is read (`GET …/probes/{pm}/decision`). miStudio unreachable →
           `{"unavailable": "mistudio"}`; a bar miStudio cannot write as a decision →
           `decision_not_formable` with `next`.
        2. Your values must EQUAL that decision, field by field (`threshold`, `target_fpr`,
           `realised_fpr`, `threshold_source`, `windows`, `length_bands`) — else
           `decision_not_current`, naming each field with yours and the one in force, and a `next`
           re-cut. The bar is moved to the decision miStudio holds, never to numbers typed here.
        3. `decision_digest`, `decision_revision` or a non-null `calibration_id` that disagree with
           miStudio's → `decision_identity_mismatch`; omitted ones are filled from miStudio, and
           `calibration_id` is always the decision digest.
        4. miLLM's copy is read: when it states a detector digest under `mistudio.probe-identity/1`
           that differs from miStudio's last built one → `detector_mismatch` (the copy is another
           detector — re-import it). A miLLM that states none, or another algorithm, is pushed to with
           `detector_unverified: true` and the reason.
        5. miLLM's OpenAPI says whether it takes `decision_digest` / `decision_revision` as fields
           (cached `MCP_MILLM_OPENAPI_TTL_SECONDS`); they are sent only when advertised.
        6. ONE POST, carrying miStudio's WHOLE decision (calibration included) — miLLM refuses a digest
           that does not name the decision it stores (`PROBE_IDENTITY_MISMATCH`).
        """
        agent = {"threshold": threshold, "target_fpr": target_fpr, "realised_fpr": realised_fpr,
                 "threshold_source": threshold_source, "windows": windows,
                 "length_bands": length_bands}

        # ── 1. miStudio's record of the decision in force ─────────────────────────────────
        try:
            current = await ctx.mistudio.get(f"/probe-monitors/probes/{mistudio_probe_id}/decision")
        except BackendError as exc:
            return {"unavailable": "mistudio",
                    "reason": f"miStudio could not be asked for probe {mistudio_probe_id}'s decision "
                              f"({exc.status_code}: {exc.detail}); nothing was pushed"}
        decision = (current or {}).get("decision")
        if decision is None:
            return {"refused": "decision_not_formable",
                    "reason": (current or {}).get("detail") or "miStudio cannot write this probe's "
                              "bar as a contract decision",
                    "next": [_next("build_probe_definition", {"probe_id": mistudio_probe_id},
                                   "The build names what makes the decision unformable (usually a "
                                   "calibration dataset with no HuggingFace id).")]}

        # ── 2. the agent's bar must BE the decision in force ──────────────────────────────
        differences = decision_differences(agent, decision)
        if differences:
            return {"refused": "decision_not_current", "differences": differences,
                    "reason": "these values are not the decision miStudio holds now; push the "
                              "decision in force, or re-cut it first",
                    "next": [_next("recalibrate_probe_monitor",
                                   {"probe_id": mistudio_probe_id,
                                    "target_fpr": decision.get("target_fpr") or target_fpr or 0.01,
                                    "preview": False},
                                   "Commit the bar you meant in miStudio first; its result's `next` "
                                   "carries the push with the committed values.")]}

        # ── 3. the identity the agent holds must agree; then it is filled from miStudio ────
        digest = current.get("decision_digest")
        revision = current.get("decision_revision")
        mismatched = [
            {"field": field, "yours": yours, "in_force": theirs}
            for field, yours, theirs in (("decision_digest", decision_digest, digest),
                                         ("decision_revision", decision_revision, revision),
                                         ("calibration_id", calibration_id, digest))
            if yours is not None and yours != theirs
        ]
        if mismatched:
            return {"refused": "decision_identity_mismatch", "mismatched": mismatched,
                    "reason": "the identity you passed does not name miStudio's decision in force"}

        # ── 4. the detector miLLM holds must be the one miStudio built ─────────────────────
        copy = await millm.get(f"/api/probes/{probe_id}")
        theirs = (copy or {}).get("detector_digest")
        algorithm = (copy or {}).get("identity_algorithm")
        ours = current.get("detector_digest")
        unverified: Optional[str] = None
        if theirs and algorithm == IDENTITY_ALGORITHM and ours:
            if theirs != ours:
                logger.info("detector_mismatch probe=%s mistudio_probe=%s millm=%s mistudio=%s",
                            probe_id, mistudio_probe_id, theirs, ours)
                return {"refused": "detector_mismatch",
                        "detector_digest": {"millm": theirs, "mistudio": ours},
                        "reason": "miLLM's copy is a different detector from the one miStudio last "
                                  "built; moving its bar would put this decision on another detector",
                        "next": [_next("export_probe_definition", {"probe_id": mistudio_probe_id},
                                       "Fetch the definition miStudio built."),
                                 _next("millm_import_probe", {},
                                       "Import it into miLLM as a new copy (the old copy keeps its "
                                       "history).", needs=["definition"])]}
        elif not ours:
            unverified = "miStudio has no built detector digest for this probe (never built)"
        elif not theirs:
            unverified = "miLLM does not report a detector digest for this copy (it predates 038)"
        else:
            unverified = (f"miLLM reports a detector digest under {algorithm!r}, not "
                          f"{IDENTITY_ALGORITHM!r}, so the two cannot be compared")

        # ── 5. which identity fields miLLM's request takes ─────────────────────────────────
        fields = await advertised_recalibrate_fields(
            millm, int(getattr(ctx.settings, "millm_openapi_ttl_seconds", 60)))
        body: dict[str, Any] = {
            # ⚠ THE WHOLE DECISION MIStUDIO HOLDS, `calibration` INCLUDED — not one rebuilt from the
            # agent's arguments. miLLM digests the decision it stores and refuses a sent digest that
            # names another (`PROBE_IDENTITY_MISMATCH`, 038 R-039); a block without `calibration`
            # would be refused for every calibration-set probe.
            "decision": decision,
            "mistudio_probe_id": mistudio_probe_id,
            "calibration_id": digest,
            "reason": reason,
        }
        if "decision_digest" in fields:
            body["decision_digest"] = digest
        if "decision_revision" in fields:
            body["decision_revision"] = revision

        # ── 6. one POST ─────────────────────────────────────────────────────────────────────
        result = await millm.post(f"/api/probes/{probe_id}/recalibrate", json_body=body)
        if unverified is not None:
            if not isinstance(result, dict):
                result = {"result": result}
            result = {**result, "detector_unverified": True, "detector_unverified_reason": unverified}
        return result

    @mcp.tool(description=SCORE_PROBES_DESCRIPTION)
    async def millm_score_probes(
        inputs: Annotated[list[ProbeScoreInput], Field(description="What to score. Each is EXACTLY ONE of token_ids (optionally with prompt_tokens), messages, or text", min_length=1)],
        probe_ids: Annotated[Optional[list[str]], Field(description="miLLM probe ids (from millm_list_probes), NOT miStudio's pm_… ids. Omit to score with every imported probe", min_length=1)] = None,
        windows: Annotated[Optional[list[str]], Field(description="Which windows to report; omit for every window the probe has")] = None,
        return_token_ids: Annotated[bool, Field(description="Return the token ids each input was scored on")] = False,
        export_results: Annotated[bool, Field(description="Ask miLLM to export the per-input results to its telemetry backend (only if the deployment permits it; the response's `export` block says). Sent only when true")] = False,
        lease_id: Annotated[Optional[str], Field(description="Your lease id, if you hold miLLM's lease; sent as X-miLLM-Lease")] = None,
    ) -> Any:
        """Score stored inputs with imported probes — armed or not — and record nothing."""
        # Arguments BEFORE the gate (FR-14 / BR-011): an agent fixing its payload must not be
        # told "millm is down".
        if len(inputs) > MAX_PROBE_INPUTS:
            return {"error": "over_synchronous_cap",
                    "message": f"{len(inputs)} inputs is over this tool's cap of "
                               f"{MAX_PROBE_INPUTS}; split the work, or score them as a batch "
                               "with millm_upload_batch_file and millm_submit_batch "
                               "(endpoint /api/probes/score)."}
        for index, item in enumerate(inputs):
            kinds = [k for k in ("token_ids", "messages", "text") if getattr(item, k) is not None]
            if len(kinds) != 1:
                return {"error": "invalid_input",
                        "message": f"inputs[{index}] must carry exactly one of token_ids, "
                                   f"messages or text; it carries {kinds or 'none'}"}
            if item.prompt_tokens is not None and kinds != ["token_ids"]:
                return {"error": "invalid_input",
                        "message": f"inputs[{index}]: prompt_tokens goes only with token_ids "
                                   "(for messages it is derived by miLLM)"}
        body: dict[str, Any] = {
            "inputs": [item.model_dump(exclude_none=True) for item in inputs],
        }
        if probe_ids is not None:
            body["probe_ids"] = list(probe_ids)
        if windows is not None:
            body["windows"] = list(windows)
        if return_token_ids:
            body["return_token_ids"] = True
        if export_results:
            # Only when true (038 FR-38.17): an explicit `false` still earns an `export` summary
            # from miLLM, and an older miLLM refuses the field outright under strict mode.
            body["export_results"] = True
        ok, reason_unavailable = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason_unavailable}
        # miLLM's body VERBATIM — the verdict, `null` and `rung_language` included. This tool
        # computes nothing from score and threshold (P-03: the boundary lives in miLLM).
        return await send(gate, millm.post(
            "/api/probes/score", json_body=body, headers=sent_headers(lease_id), full=True,
        ))

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_probe_status() -> Any:
        """What every armed probe is doing, and why any of them is NOT scoring.

        ⚠ **Read `paused_reason` and `paused_reasons` before concluding anything from a
        quiet probe.** An armed probe that is not scoring always says why; silence would
        otherwise read as "nothing detected", which is a claim the probe never made.
        Requests go unscored when they are batched, under speculative decoding, or
        served by continuous batching.

        Also carries the last request's measured overhead and the warning threshold, and
        whether continuous batching is currently disabled — arming costs throughput, and
        on a busy deployment that is a capacity decision rather than a detail.
        """
        return await millm.get("/api/probes/status")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_probe_events(
        probe_id: Annotated[Optional[str], Field(description="Only this probe's verdicts")] = None,
        request_id: Annotated[Optional[str], Field(description="The /v1 completion id: every verdict of one response")] = None,
        result_id: Annotated[Optional[str], Field(description="One result's id (a UUID) from millm_probe_verdicts, a score result, an event or an exported record — exactly one event", max_length=36)] = None,
        trace_id: Annotated[Optional[str], Field(description="A W3C trace id (32 lowercase hex) from a telemetry backend — that request's events", pattern=r"^[0-9a-f]{32}$")] = None,
        # Bounds are miLLM's own `Query(ge=1, le=500)` on GET /api/probes/events (review R1,
        # 038 S3-M): out of range used to travel and come back as miLLM's 422.
        limit: Annotated[int, Field(description="Max rows to return (1-500)", ge=1, le=500)] = 50,
    ) -> Any:
        """Verdicts, newest first.

        Filter by `probe_id`, by `request_id` (the /v1 completion id), by `result_id` (exactly one
        event: the id in `millm_probe_verdicts`, a score result or an exported record) or by
        `trace_id` (the request a telemetry backend's alert names) — the join from anywhere a
        result travelled back to the stored event.

        Each row is a `mi.probe.result/v1` object (miLLM Feature 038), passed through unchanged:
        `outcome` is `completed`, `skipped`, `failed` or `interrupted`, and any outcome but
        `completed` carries a `reason` code from miLLM's published list (e.g. `judge_traffic`,
        `batched_request`, `continuous_batching`, `engine_unsupported`, `no_scored_tokens`,
        `encoder_failed`, `non_finite_score`, `client_disconnected`, `cancelled`). Both lists are
        open within v1: read an unknown outcome or code as "other", never as a no.

        ⚠ **These rows carry NO prompt or context text.** The decoded window around a
        firing position is user content and is served only per event, by
        `millm_get_probe_event(event_id)` — one deliberate request per conversation, never a
        feed. Fetch it only when you need to read what the user actually wrote.

        Three things a row does not say:

        * **`verdict: null`** — no threshold was placed. The probe ranked without
          deciding. It is not a "no".
        * **`scored: false`** — this request was not scored, and `not_scored_reason`
          always says why (`batched_request`, `speculative_decoding`,
          `continuous_batching`, `role_mask_unreliable`). Also not a "no".
        * **A high score is not a cause.** A probe detects; it does not explain, and
          nothing here has shown the detected thing caused the output.

        `rung` on a row is the rung **as of that observation**, not the probe's rung
        today — re-evaluating a probe does not retroactively change what an old verdict
        was worth.
        """
        return await millm.get(
            "/api/probes/events",
            probe_id=probe_id, request_id=request_id, result_id=result_id, trace_id=trace_id,
            limit=limit,
        )

    # ── 2026-10-19 (MCP gap closure): read one, delete, parity, the Hub, clearing events ────
    #
    # Operator decision 2026-10-19: the single-event detail routes ARE exposed to agents
    # (`millm_get_probe_event` here, `millm_get_sensing_event` in millm_sensing). They return
    # decoded USER PROMPT CONTENT, and their descriptions say so plainly. They stay per-event:
    # the list tools still carry no context text, so reading a conversation is one explicit call.

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_probe_event(
        event_id: Annotated[int, Field(description="Probe event id from millm_probe_events")],
    ) -> Any:
        """⚠ RETURNS DECODED USER PROMPT CONTENT. One probe verdict WITH its context window:
        `context_text` is the decoded text around the firing position — what the user (and the
        model) actually wrote — and `context_token_ids` the tokens behind it.

        Fetch one event at a time, only when you need to read the conversation behind a verdict;
        treat the text as user data (do not copy it into logs, reports or other systems without
        reason). The summary fields are the same as in millm_probe_events, including the
        `mi.probe.result/v1` contract fields passed through unchanged (`result_id`, `trace_id`,
        `outcome`, `reason`, `decision`, detector identity, `persistence`): `verdict: null` is
        silence, not "no"; a high score is not a cause."""
        return await millm.get(f"/api/probes/events/{event_id}")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_probe(
        probe_id: Annotated[str, Field(description="miLLM probe id from millm_list_probes (not miStudio's pm_… id)")],
    ) -> Any:
        """One imported probe: its summary (rung, rung_language VERBATIM, threshold, armed
        state, parity report) plus the full stored `definition` document."""
        return await millm.get(f"/api/probes/{probe_id}")

    @mcp.tool()
    async def millm_delete_probe(
        probe_id: Annotated[str, Field(description="miLLM probe id to delete")],
        acknowledge_events_deleted: Annotated[bool, Field(description="Deleting a probe deletes EVERY verdict it recorded. Required, and irreversible")] = False,
    ) -> Any:
        """Delete an imported probe AND its recorded events. Refused by miLLM while the probe is
        ARMED — disarm it first (`millm_disarm_probe`).

        Refuses unless `acknowledge_events_deleted=True`: the probe's verdict history goes with
        it, and nothing else holds it. This is also the middle step of replacing a probe
        (disarm → delete → import), which is why there is no 'replace' on import."""
        if not acknowledge_events_deleted:
            return {"refused": "delete_not_acknowledged", "probe_id": probe_id,
                    "message": "Deleting a probe deletes every verdict it recorded. Re-call "
                               "with acknowledge_events_deleted=true if that is intended."}
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await millm.delete(f"/api/probes/{probe_id}")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_probe_parity(
        probe_id: Annotated[str, Field(description="miLLM probe id from millm_list_probes")],
    ) -> Any:
        """Re-run PARITY without arming, and store the report: score the definition's own test
        vectors on the model miLLM has loaded and compare with the scores miStudio recorded.

        Use it after a model reload — the same definition against a differently-loaded model
        (another precision or quantization) is exactly what parity exists to catch. Runs one
        forward pass per vector in an admission slot, UNSTEERED. The report names the loaded
        precision so a failure names its likely cause. Arming runs parity too; this lets you ask
        first."""
        return await millm.post(f"/api/probes/{probe_id}/parity")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_probe_hub_search(
        q: Annotated[Optional[str], Field(description="Free-text query")] = None,
        base_model: Annotated[Optional[str], Field(description="Only repos for this base model id")] = None,
        limit: Annotated[int, Field(description="Max repos (1-50)", ge=1, le=50)] = 30,
    ) -> Any:
        """Search HuggingFace for repos tagged `mistudio-probe-definition`. Then list a repo's
        files with `millm_probe_hub_definitions` and import one with `millm_import_probe`
        (`repo_id` + `filename`)."""
        return await millm.get("/api/probes/hub/search", q=q, base_model=base_model, limit=limit)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_probe_hub_definitions(
        repo_id: Annotated[str, Field(description="HuggingFace repo id, owner/name")],
        revision: Annotated[Optional[str], Field(description="Hub revision; defaults to main")] = None,
    ) -> Any:
        """The probe definitions a Hub repo publishes — `manifest.jsonl` preferred, loose
        `*.probe.json` files otherwise. A repo usually holds one definition per layer: pick the
        `filename` for the layer you mean, or parity is the only gate that will notice."""
        return await millm.get(f"/api/probes/hub/{repo_id}/definitions", revision=revision)

    @mcp.tool()
    async def millm_clear_probe_events(
        probe_id: Annotated[Optional[str], Field(description="Clear only this probe's events")] = None,
        all_probes: Annotated[bool, Field(description="Clear EVERY probe's events. Required when probe_id is omitted")] = False,
    ) -> Any:
        """Delete recorded probe verdicts — one probe's, or (explicitly) all of them. Returns
        `{removed}`. Irreversible.

        The scope is REQUIRED: an unscoped call is refused here rather than read as "delete
        everything" (the `millm_circuit_sensing_clear` rule)."""
        if probe_id is None and not all_probes:
            return {"error": "scope_required",
                    "message": "pass probe_id, or all_probes=true to clear every probe's events"}
        if probe_id is not None and all_probes:
            return {"error": "ambiguous_scope",
                    "message": "pass probe_id OR all_probes=true, not both"}
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await millm.delete("/api/probes/events", probe_id=probe_id)
