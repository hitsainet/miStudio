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

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient


def register(mcp: FastMCP, millm: MiLLMClient, gate: HealthGate) -> None:
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
        windows: Annotated[Optional[list[str]], Field(description="Which slices of a request to report, from 'all', 'prompt', 'response'. Omit for all three. 'prompt' reads the user's words alone and is independent of how long the model's reply was; 'response' reads the model's own output and is UNTRAINED for probes fitted on user text, so its verdicts come back flagged provisional")] = None,
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
        three. A window is returned **provisional** when it has no threshold of its own (the
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
        calibration_id: Annotated[Optional[str], Field(description="The producer's id for this cut, recorded against the revision")] = None,
        reason: Annotated[str, Field(description="Why the bar moved. Recorded in the probe's threshold_history")] = "",
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
        """
        decision: dict[str, Any] = {
            "threshold": threshold,
            "target_fpr": target_fpr,
            "realised_fpr": realised_fpr,
            "threshold_source": threshold_source,
        }
        # Omitted rather than sent as null: miLLM's parsers treat an absent block and an empty one
        # differently, and sending `None` for windows a probe HAS would read as "it has none".
        if windows is not None:
            decision["windows"] = windows
        if length_bands is not None:
            decision["length_bands"] = length_bands
        return await millm.post(
            f"/api/probes/{probe_id}/recalibrate",
            json_body={
                "decision": decision,
                "mistudio_probe_id": mistudio_probe_id,
                "calibration_id": calibration_id,
                "reason": reason,
            },
        )

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
        request_id: Annotated[Optional[str], Field(description="The /v1 completion id — the only link between a response and its verdict")] = None,
        limit: Annotated[int, Field(description="Max rows to return")] = 50,
    ) -> Any:
        """Verdicts, newest first.

        ⚠ **These rows carry NO prompt or context text.** The decoded window around a
        firing position is user content, and it is served only by the single-event detail
        route, which no tool in this category consumes — so a reviewer asks for one
        conversation rather than an agent receiving a feed of them.

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
            probe_id=probe_id, request_id=request_id, limit=limit,
        )
