"""Self-describing workflow guidance for agents (`mistudio_howto`).

WHY THIS EXISTS
---------------
An agent (me, 2026-07-21) ran the full circuit refinement loop — find labelled
features, author a definition, sweep strengths, serve it through miLLM — and
spent most of that session REDISCOVERING things this server already knew:

  * `kind` is ``mistudio.circuit-definition``, with NO ``/v1`` suffix. Guessed
    wrong on the first attempt; only a schema validator caught it.
  * Circuit members nest under ``member.feature.strength``. Read flat, the
    strength silently reads as ``None`` — which looks like missing data, not a
    shape error.
  * Features key on ``external_sae_id``, not ``training_id``. Querying by
    training returned "0 labelled features" for a corpus where every single
    feature is labelled.
  * A hand-authored circuit is rung 0, and miLLM REFUSES to activate it
    without ``acknowledge_unvalidated=true``.
  * Steering strength must be a fraction of a REAL ``max_activation``. A
    fixture with placeholder ``max_activation: 10.0`` on every member shipped
    at ~50x the usable ceiling and emitted pure token soup in production.

None of that is discoverable from tool signatures. Individually each tool is
well documented; what was missing is the ORDER, the SHAPES that cross tool
boundaries, and the failure modes that look like something else. This tool
carries that, so the next agent reads instead of excavating.

CONTENT RULES
-------------
Every claim here is either verified against the code or carries its evidence.
Numbers cite where they were measured. Nothing is aspirational: if a workflow
is not implemented, this says so rather than describing the intent.
"""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings

# ── Topic content ──────────────────────────────────────────────────────────
#
# Kept as data rather than prose in a docstring so `topics` can be enumerated
# and each entry returned whole. An agent reading one topic gets everything it
# needs for that task without paging through the rest.

_OVERVIEW = """\
miStudio and miLLM are two planes of one system.

  miStudio  DISCOVERS and CALIBRATES. It mines candidate circuits, runs real
            causal interventions to grade them, and sweeps steering strengths
            against generated text. It runs the model to LEARN.

  miLLM     SERVES. It imports a finished definition and applies it to live
            traffic behind an OpenAI-compatible API, so any client that speaks
            that protocol gets the steered model with no special support. It
            runs the model to SERVE.

The boundary is a document, not a code dependency: `mistudio.circuit-definition`
(and `mistudio.cluster-definition` for the single-layer case). Neither repo
imports the other. miStudio emits; miLLM consumes and re-exports losslessly —
miLLM never authors or re-grades.

WHAT CROSSES: a map (which layers, which features, what strength) PLUS an
evidence grade (the rung) for each edge.

Pick your topic:
  tool_map            the surface grouped by what you are trying to do
  tools               EVERY registered tool, derived live (complete index)
  discovery_pipeline  the REAL way to produce a circuit — mine, attribute,
                      validate, promote. Every stage is an MCP tool.
  circuit_authoring   write a definition by hand, correct field shapes
  finding_features    locate real labelled features to build from
  strength_calibration  choose strengths that steer without destroying output
  evidence_ladder     what the rungs mean and where they gate
  serving             import, activate, verify through the OpenAI API
  troubleshooting     symptom -> cause, for the failures that mislead
"""

_CIRCUIT_AUTHORING = """\
AUTHORING A CIRCUIT DEFINITION BY HAND

Required top-level fields: `name`, `saes`, `members`. Everything else optional.

THE FOUR THINGS THAT BITE (all cost real debugging time):

1. `kind` is "mistudio.circuit-definition" — NO "/v1" suffix, even though the
   schema file is named circuit-definition-v1.json and the docs say "v1".

2. Members NEST. Strength lives at `member.feature.strength`, not
   `member.strength`. Read flat and you get None, which looks like absent data
   rather than a wrong shape.

3. `edges` carry NO strength and do NOT affect steering. They are evidence
   metadata only. ALL steering comes from `members`. An edge list is how you
   record what connects to what and how well-founded that claim is.

4. Set `rung: 0` on hand-authored edges. Rung >= 2 asserts a causal validation
   you have not run, and the copy audit forbids the word "causal" below rung 2.

MINIMAL VALID DOCUMENT:

{
  "schema_version": "1",
  "kind": "mistudio.circuit-definition",
  "name": "example-l12-l13",
  "saes": [
    {"layer": 12, "mistudio_sae_id": "<sae id as miLLM knows it>"},
    {"layer": 13, "mistudio_sae_id": "<sae id as miLLM knows it>"}
  ],
  "members": [
    {"layer": 12, "member_kind": "feature_ref",
     "feature": {"feature_idx": 3121, "strength": 1.02,
                 "label": "casual_gaming", "max_activation": 6.81}},
    {"layer": 13, "member_kind": "feature_ref",
     "feature": {"feature_idx": 8154, "strength": 1.53,
                 "label": "casino_game", "max_activation": 10.2}}
  ],
  "edges": [
    {"up":   {"kind": "feature", "layer": 12, "feature_idx": 3121},
     "down": {"kind": "feature", "layer": 13, "feature_idx": 8154},
     "type": "computed", "rung": 0}
  ],
  "budget": {"intensity": 1.0, "intensity_range": [0.0, 2.0]}
}

The `mistudio_sae_id` must be the id MILLM uses (list them with
`millm_status` or miLLM's /api/saes), not miStudio's internal `sae_xxxx` row
id. They are different namespaces for the same SAE.

`max_activation` MUST be the real measured value for that feature — see the
strength_calibration topic for why a placeholder here is dangerous.
"""

_FINDING_FEATURES = """\
FINDING REAL LABELLED FEATURES

A circuit is only meaningful if its feature indices point at features you have
actually inspected. Arbitrary indices produce arbitrary steering.

THE KEY FACT that costs time otherwise: features key on the SAE, not the
training. `search_features` / `get_feature` are the tools; if you go to the
database directly, features join on `external_sae_id`. Querying by
`training_id` can return zero rows for a corpus where every feature is
labelled — the trainings that produced these SAEs are not the join key.

WORKFLOW:
  1. `list_extractions`      — which analyses exist
  2. `search_features`       — by token or label text
  3. `get_feature`           — label, description, max_activation,
                               activation_frequency, interpretability_score
  4. `get_feature_examples`  — what actually makes it fire. DO THIS. A label
                               is a summary; the examples are the evidence.

WHAT TO SELECT FOR:
  * A real `max_activation` — you need it to set strength (see calibration).
  * `activation_frequency` — RELATIVE TO THE SAE, not an absolute band.
    MEASURED on this deployment (2026-07-21, LFM2.5 residual SAEs): medians
    run 0.35-0.73 and the 5th percentile is still ~0.20-0.50. This field is
    NOT per-token sparsity; it is computed over evaluation samples, so a
    "0.7" feature is TYPICAL here, not near-ubiquitous.
    An earlier version of this guidance said to prefer ~0.001-0.05, which
    would reject essentially every feature in this deployment. Compare a
    candidate against its own SAE's distribution instead of a fixed number.
  * A layer where an SAE is ATTACHED in miLLM, or the circuit cannot serve
    fully and will degrade to a per-layer slice.

CHECK WHAT THE CAPTURE CORPUS IS before mining for a THEME. This deployment's
captures run over OpenWebText-2M — general web text. Features for a specific
theme (humor, legal register, code) exist and are labelled, but they fire
SPARSELY there, so themed edges can fail `s_min` support for a reason that has
nothing to do with whether the circuit is real. A null result on a themed
search is evidence about the CORPUS at least as much as about the model.

For a multi-layer circuit, pick features whose meanings plausibly relate
across layers (upstream earlier, downstream later). If you want that relation
ESTABLISHED rather than assumed, that is what discovery + validation are for —
this manual path asserts nothing, which is why it is rung 0.
"""

_STRENGTH_CALIBRATION = """\
CHOOSING STRENGTHS — the failure mode that looks like a broken model

Steering is ADDITIVE on the residual stream:

    modified = original + SUM(strength * W_dec[feature_idx])

Every member on every layer adds to the same sum. Overdrive it and the
residual is dominated by decoder vectors; the model emits whatever tokens sit
nearest that direction — high-norm multilingual fragments, not degraded
English. It looks like the model is broken. It is not.

MEASURED ENVELOPE (2026-07-21, LFM2.5-1.2B-Instruct, 2 layers, 1 edge; see
0xcc/docs/Archive/circuit-strength-calibration.md for the raw data):

    total 0.34 .. 3.07   fluent
    total 4.25           COLLAPSED to repeated determiners ("The. An. In.")
    total 8.50           12 characters of output
    total 17+            empty output

The cliff is SHARP — one step past it, output is unusable with no warning.

RULE OF THUMB: start at <= 0.15 * max_activation per member, on 2 layers, and
sweep upward while reading the generated text.

THE PER-MEMBER RULE DOES NOT BOUND THE SUM. Measured 2026-07-21 on a 4-member
3-layer circuit (crc_124fd83d1f2a): compute_cluster_allocation proposed total
3.50, EVERY member satisfied <= 0.15 x max_activation (1.1-5.9% in fact), and
3.50 collapsed generation outright — perplexity 695.9 vs 7.6 baseline. The
usable ceiling for that circuit was total ~1.4, roughly 0.4x the ~3 that a
2-layer measurement suggested. More layers means a tighter total budget, not a
looser one.

TRUST THE SWEEP OVER THE FORMULA. compute_cluster_allocation gives a principled
STARTING POINT and surfaces hazards (it flagged 'cancellation' on that circuit,
which was consistent with the tight envelope) — but it does not predict the
collapse point. Sweep and read the text; then set `intensity_range` so the dial
CANNOT reach the degradation point you observed.

WHY `max_activation` MUST BE REAL: expressing strength as a fraction of it is
what keeps members comparable. A fixture with a placeholder
`max_activation: 10.0` on every member shipped at 5 layers x 40/35/30/25/20 =
150 total — about 50x the measured ceiling — and produced pure token soup for
every caller until it was deactivated.

CAVEAT ON THE NUMBERS: one model, one prompt, one feature pair. Indicative,
not a law. Re-measure for your model. The METHOD generalises; the constant
may not.

HOW TO SWEEP: `steer_sweep` runs multiple strengths in miStudio and returns
generations to compare. If you sweep by hand through miLLM instead, deactivate
and delete each trial circuit — leaving them active contends for layers.

JUDGING OUTPUT: do not score coherence by character set. An ASCII-word-ratio
metric rated "The.InIn. An. The." at 1.00 — perfect — because the failure mode
is REPETITION, not foreign characters. Type-token ratio (unique words / total
words) separates the cases; below ~0.4 with short output means collapse.
"""

_EVIDENCE_LADDER = """\
THE EVIDENCE LADDER — what a rung means and where it bites

  0 MINED                  "associated"                    statistics only
  1 ATTRIBUTION_SUPPORTED  "suggested (attribution-supported)"  gradients agree
  2 CAUSALLY_VALIDATED     "causally validated (edge)"     real intervention
  3 FAITHFULNESS_TESTED    "faithfulness-tested (circuit)" circuit-level

A CIRCUIT's rung is the MINIMUM over its edges — one unvalidated edge caps the
whole circuit. Failures are recorded in `tested_and_failed` WITHOUT demotion: a
failed rung-2 test does not erase a real rung-0 association.

RUNG 3 IS NOT AN EDGE RUNG. Faithfulness ablates the whole member set at once,
so it is a CIRCUIT-level result carried in `circuit.faithfulness` +
`faithfulness_status`, shown separately. A faithfulness-tested circuit still
DISPLAYS at most rung 2. This is intentional, not a gap.

WHERE IT GATES — the two planes differ, and this surprises people:

  miStudio: BADGE, NOT GATE. Promotion, export and steering never require any
            rung. You can promote and export a rung-0 circuit. Honesty is
            enforced by LANGUAGE instead — the word "causal" is forbidden below
            rung 2 by a build-failing copy audit.

  miLLM:    GATE. Activating a circuit below rung 2 is REFUSED as
            UNVALIDATED_CIRCUIT unless you pass
            `acknowledge_unvalidated=true`. The acknowledgement does NOT
            persist — the intensity dial re-applies the same gate.

So: the discovery plane records evidence and refuses to gate; the serving
plane forces a decision. A hand-authored circuit is rung 0 by construction and
WILL be refused on first activation. That is the system working.

CLIMBING THE LADDER requires an INTERVENTION, and interventions only happen in
miStudio. miLLM's edge sensing observes members co-firing on live traffic, but
that is explicitly never evidence about a rung — observing an edge fire says
two features fired in order, not that one caused the other.
"""

_SERVING = """\
SERVING A CIRCUIT THROUGH miLLM

Once steering is active it is an AMBIENT property of the model: any client
that speaks the OpenAI API gets the steered model with no special parameters.
That is the point of the split.

SEQUENCE:
  1. `millm_import_circuit(definition=<the document>)`
        Import does NOT activate. Returns the assigned circuit id, the
        computed rung, and `serveable`.
  2. `millm_activate_circuit(circuit_id=..., acknowledge_unvalidated=true)`
        Required for any rung-0/1 circuit (see evidence_ladder). Check
        `serving_mode` in the response:
          "full"           = every referenced SAE bound. The whole circuit.
          "slice_fallback" = NOT the whole circuit — a per-layer projection,
                             because some SAE was unavailable. The rung header
                             is suppressed and the intensity dial is recorded
                             but NOT applied.
  3. Generate normally against the OpenAI endpoint.
  4. `millm_set_circuit_intensity` to dial without re-importing. Range is the
     circuit's authored `intensity_range`, not a fixed 0-2.
  5. `millm_deactivate_circuit` when done. `millm_delete_circuit` refuses on a
     serving circuit unless `acknowledge_serving=true`.

THE HONESTY HEADER: responses carry
    x-millm-circuit-rung: 2; language="causally validated (edge)"
It is SUPPRESSED in four cases — no circuit genuinely steering, a layer
composed by two circuits, slice_fallback, or a mid-generation apply failure.
Absence means "no single rung describes what you got", never "rung 0".

CONTENTION: the unit is the LAYER, not the feature, because steering is
additive on the residual stream. Two circuits on layer 12 are writing to the
same place regardless of which features they name.
  * CONTENTION (same layer, different features) is overridable with
    `allow_layer_overlap=true` — IF `CIRCUIT_ALLOW_CONCURRENT` is enabled in
    the deployment, which is checked FIRST and cannot be overridden.
  * COLLISION (same layer AND same feature) is NEVER overridable. Retrying
    with the override will be refused again.
  * Composition has a measured cost: two steered layers at strength 5
    destroyed generation on LFM2.5-1.2B-Instruct. One model, one fixture —
    indicative, not exhaustive.
"""

_TROUBLESHOOTING = """\
TROUBLESHOOTING — symptoms that mislead

"The model outputs token soup / foreign fragments / repeated determiners."
    Almost certainly steering strength, not a broken model. PROVE IT: deactivate
    the circuit and generate again. If output is fluent, strength is the cause.
    See strength_calibration. This exact symptom traced to a circuit running
    ~50x the usable ceiling.

"Activation was refused with UNVALIDATED_CIRCUIT."
    Working as designed. The circuit is below rung 2. Pass
    `acknowledge_unvalidated=true` if you accept steering on unvalidated
    evidence. You will need it again for the intensity dial.

"The intensity dial returns AMBIGUOUS_ACTIVE_CIRCUIT."
    More than one circuit is serving, so "the active circuit" is not one thing.
    Name the circuit explicitly.

"serving_mode came back slice_fallback."
    Some referenced SAE was not bound. You are serving a PER-LAYER PROJECTION,
    not the circuit. The dial is recorded but not applied, and the rung header
    is suppressed. Check which SAEs are attached in miLLM.

"The rung header is missing."
    Four causes: a composed layer, slice_fallback, no intervention ran, or a
    mid-generation apply failure. It is suppressed rather than wrong.

"Sensing is enabled but there are no events."
    Expected for arbitrary feature indices — the edges must actually co-fire on
    your traffic. `millm_circuit_sensing_status` reports how many edges are
    sensable. If `requests_sensed == 0`, no request reached sensing at all —
    that is a wiring fault, not quiet traffic.

"A feature query returned zero labelled features."
    Check the join key. Features key on the SAE (`external_sae_id`), not the
    training. Querying by training_id can return nothing for a fully-labelled
    corpus.

"A millm_* tool returned {"unavailable": "millm", ...}."
    The serving runtime is unreachable. Report the reason; do not retry in a
    loop. Distinguish this from {"error": ...}, which means YOUR call was
    malformed, and from a coded refusal, which means the operation was
    declined on its merits. Three different responses are required.
"""

_DISCOVERY_PIPELINE = """\
THE REAL DISCOVERY PIPELINE — fully available over MCP

This is the intended way to produce a circuit. Hand-authoring (see
circuit_authoring) asserts nothing and stays at rung 0; this pipeline EARNS
each rung. Every stage below is an MCP tool.

  1. `start_circuit_capture`      GPU. Two-phase: runs a probe and stops at a
                                  cost estimate; call again to confirm. Writes
                                  a sparse per-layer firing store plus a
                                  train/heldout document split.
     `list_circuit_captures`      progress / status

  2. `run_circuit_discovery`      NO GPU — pure statistics over the store.
                                  PMI/lift over a within-document circular-
                                  shift null, Benjamini-Hochberg FDR, then
                                  held-out replication on the excluded docs.
                                  -> RUNG 0 (mined)
     `get_discovery_results`      candidates + the statistical report

  3. `run_attribution_pass`       GPU. Gradient attribution through the SAE
                                  codes; checks sign agreement and magnitude.
                                  -> RUNG 1 (attribution-supported)

  4. `validate_circuit_edges`     GPU. REAL INTERVENTION: ablates the upstream
                                  feature and measures the downstream effect
                                  vs a support-matched non-edge null.
                                  -> RUNG 2 (causally validated)
     `list_validation_manifests` / `get_validation_manifest` / `reproduce_validation`
                                  the audit trail, and re-running it

  5. `create_circuit`             build the circuit from selected candidates.
                                  Circuit rung = MIN over its edges.

  6. `run_circuit_faithfulness`   GPU. Circuit-level necessity/sufficiency by
                                  ablating the whole member set.
                                  -> recorded in `faithfulness`, NOT as an
                                     edge rung (see evidence_ladder)

  7. `promote_circuit`            a badge, not a gate — never required
     `export_circuit_definition`  the portable document for miLLM
     `export_circuit_slices`      per-layer cluster projections for
                                  single-SAE consumers

SEQUENCING: the stages gate each other. Attribution-ordered validation is
refused unless attribution completed; faithfulness needs a discovery run id
because its prompts come from the capture store. GPU stages take a single-GPU
advisory lock — one at a time.

WHEN TO HAND-AUTHOR INSTEAD: quick experiments, reproducing a known circuit,
or when you already know which features you want and only need to tune
strengths. Be honest about the rung — a hand-authored circuit is rung 0 and
miLLM will require `acknowledge_unvalidated=true`.
"""

_TOOL_MAP = """\
THE SURFACE BY INTENT — what you are trying to do

Grouped for orientation. This is PROSE and may lag the code; for the
COMPLETE list derived from the live registry, use topic='tools'.

Written because a coverage audit found the server instructions named 17 of 92
tools, with FOUR ENTIRE CATEGORIES unmentioned. Every tool below has a good
docstring; what was missing was knowing they exist.

FEATURE ANALYSIS (`read`, `labeling`)
  list_extractions, list_trainings, get_extraction_summary
  search_features, get_feature, get_feature_examples
  get_feature_token_analysis    which tokens drive it
  get_feature_logit_lens        what it pushes toward in output space
  get_feature_correlations      features that co-vary with it
  get_feature_nlp_analysis      linguistic summary (only if precomputed)
  get_feature_ablation      statistical impact estimate — NO model inference,
                            NOT a causal claim
  update_feature_label      writes carry label_source='mcp_agent'
  run_enhanced_labeling     two-pass LLM labeling (background job)
  get_enhanced_label

GROUPING — features that fire together (`groups`)
  compute_feature_groups    background precompute; get_grouping_status polls it
  get_feature_groups        groups sharing a top token with similar context
  get_feature_group_members
  find_features_by_token    exact | normalized matching
  find_related_features     via shared tokens, context overlap, correlations
  This is the SINGLE-LAYER cousin of circuit discovery. Groups become cluster
  profiles; circuits span layers.

CLUSTER PROFILES — tuned single-layer artifacts (`profiles`)
  list_cluster_profiles, get_cluster_profile, save_cluster_profile
  export_cluster_definition   portable `mistudio.cluster-definition/v1`
  compute_cluster_allocation  budget/strength allocation across members

STEERING — try it, measure it (`steering`)
  enter_steering_mode        OPTIONAL — every steer_* submit auto-starts the
                             worker. Calling it first only pays the ~10s cold
                             start up front instead of inside your first task.
  exit_steering_mode         MANDATORY when done. Nothing reaps the worker; it
                             holds VRAM indefinitely otherwise.
  get_steering_mode, steering_status, get_approval_status
  steer_sweep         SAME feature at MULTIPLE strengths — the calibration
                      workhorse (see strength_calibration)
  steer_compare       steered vs unsteered, side by side
  steer_combined      multiple features at once
  cancel_steering_task, get_steering_result

EXPERIMENTS — keep what you found (`experiments`)
  save_experiment, list_experiments, get_experiment

CIRCUITS — multi-layer, evidence-graded (`circuits`, 19 tools)
  See the discovery_pipeline topic. Also: get_circuit, list_circuits,
  update_circuit, import_circuit_definition, export_circuit_slices,
  get_validation_manifest / list_validation_manifests / reproduce_validation.

miLLM SERVING (`millm_*`, present only when MILLM_API_URL is set)
  millm_status                    what is steering RIGHT NOW, one call

  circuits   millm_list_circuits, millm_circuit_status, millm_import_circuit,
             millm_activate_circuit, millm_deactivate_circuit,
             millm_set_circuit_intensity, millm_export_circuit,
             millm_delete_circuit
  claims     millm_circuit_claims, millm_release_circuit_claims
             (layer ownership; the recovery path for a stuck claim)
  circuit    millm_circuit_sensing_status, millm_circuit_sensing_events,
   sensing   millm_circuit_sensing_event, millm_circuit_sensing_enable,
             millm_circuit_sensing_disable, millm_circuit_sensing_clear
             (edge co-firing on LIVE traffic; _clear is irreversible and
             requires an explicit scope)
  clusters   millm_list_clusters, millm_import_cluster,
             millm_activate_cluster, millm_deactivate_cluster,
             millm_export_cluster, millm_hub_search
  profiles   millm_list_profiles, millm_activate_profile,
             millm_deactivate_profile
             (activate REPLACES live steering — check millm_status first)
  cluster    millm_sensing_status, millm_sensing_events,
   sensing   millm_sensing_enable, millm_sensing_disable,
             millm_sensing_config   (config WRITES a quorum; status reads)
  dial       millm_set_intensity

JOBS / ADMIN
  get_task_status                 poll ANY background job
  delete_experiment, delete_extraction   DESTRUCTIVE — deleting an extraction
                                  takes every feature, label and activation
                                  with it

CATEGORIES ARE GATED. `MCP_TOOL_CATEGORIES` decides what is registered; the
millm_* ones need MILLM_API_URL. If a tool you expect is absent, it was not
enabled — that is configuration, not a missing feature.
"""

TOPICS: dict[str, str] = {
    "overview": _OVERVIEW,
    "tool_map": _TOOL_MAP,
    "discovery_pipeline": _DISCOVERY_PIPELINE,
    "circuit_authoring": _CIRCUIT_AUTHORING,
    "finding_features": _FINDING_FEATURES,
    "strength_calibration": _STRENGTH_CALIBRATION,
    "evidence_ladder": _EVIDENCE_LADDER,
    "serving": _SERVING,
    "troubleshooting": _TROUBLESHOOTING,
}


# ── Generated coverage ─────────────────────────────────────────────────────



def _run_sync(coro):
    """Await `coro` whether or not a loop is already running.

    `_all_tools()` is called from an async tool AND from sync tests, so it
    cannot assume either context.
    """
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Inside a loop: run it on a worker thread with its own loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _all_tools() -> dict[str, list[tuple[str, str]]]:
    """Every tool the SERVER ACTUALLY REGISTERS, with its one-line summary.

    Derived by BUILDING THE SERVER and reading `list_tools()` — the same call
    an agent makes. Everything else is a proxy for that and can disagree with
    it.

    An earlier version scanned module SOURCE for `@mcp.tool()` decorators and
    called that "the live registry" in its own docstring. An adversarial pass
    defeated it five ways, all invisible: a tool registered by a helper or a
    loop (`mcp.tool()(fn)`) rather than a literal decorator; a module imported
    but absent from CATEGORY_MODULES — the ORIGINAL F20 defect verbatim; an
    aliased decorator. Each left a real tool out of the index while every guard
    stayed green, because the scanner and the guards shared the same blind
    spot and the contract generator laundered it on regeneration.

    A source scanner over a hand-maintained module list is the exact failure
    mode this index exists to end. The registry is the only authority on what
    is registered, so ask it.

    Category attribution still comes from the module maps: `list_tools()`
    returns a flat list with no category, so the server is built once per
    category. A tool in no map cannot be attributed — and cannot be reached by
    an agent either, which is the point.
    """
    import asyncio
    import os

    from ..config import MCPSettings
    from ..server import build_server

    from . import CATEGORY_MODULES, MILLM_CATEGORY_MODULES

    # MIS-E2E-115: do NOT write to os.environ.
    #
    # This used to `os.environ.setdefault("MILLM_API_URL", "http://millm.invalid")`
    # so the millm_* categories would register long enough to be listed. But
    # `setdefault` is a permanent process mutation: after any agent called
    # `mistudio_howto` — a read-only documentation tool — the unauthenticated
    # `/health` endpoint, which re-reads the setting per request, advertised
    # `http://millm.invalid` as the configured miLLM URL for the life of the
    # process. A docs lookup silently changed what the server reported about
    # its own configuration.
    #
    # The placeholder now rides on the throwaway settings object below, so it
    # exists only for this enumeration.

    out: dict[str, list[tuple[str, str]]] = {}
    for category in {**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}:
        mcp, _client = build_server(
            MCPSettings(tool_categories=category, allow_anonymous=True,
                        millm_api_url_override="http://millm.invalid"),
            stdio=True,
        )
        # `mistudio_howto` is itself an async tool, so this runs INSIDE a
        # running loop and `asyncio.run` raises there. A blanket `except:
        # continue` hid that and returned an EMPTY index — a silent blanking
        # of the completeness guarantee, which is the exact failure this whole
        # effort exists to prevent. Never swallow here.
        tools = _run_sync(mcp.list_tools())
        entries: list[tuple[str, str]] = []
        for tool in tools:
            summary = " ".join((tool.description or "").split())
            cut = summary.find(". ")
            if 0 < cut < 160:
                summary = summary[: cut + 1]
            entries.append((tool.name, summary[:170]))
        if entries:
            out[category] = sorted(entries)
    return out


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def mistudio_howto(topic: Annotated[Optional[str], Field(description="Guidance topic. Omit for the overview and the topic list")] = None) -> Any:
        """START HERE for circuit/steering work — workflow guidance an agent
        cannot infer from tool signatures.

        Explains the miStudio (discover/calibrate) → miLLM (serve) split, the
        exact document shapes that cross between them, how to pick real
        features, how to choose steering strengths that do not destroy
        generation, what the evidence rungs gate, and which symptoms mislead.

        Call with no argument for the overview and the topic list. Topics:
        overview, tool_map, discovery_pipeline, circuit_authoring, finding_features,
        strength_calibration, evidence_ladder, serving, troubleshooting.

        Written because an agent ran the full loop and lost most of a session
        rediscovering things this server already knew — a `/v1` suffix that
        does not belong, a nested member shape that reads as missing data, and
        a strength scale ~50x off, which shipped token soup to production.
        """
        # `tools` is DERIVED, not prose — it lives beside the written topics
        # so an agent discovers it the same way, but its content comes from
        # the live registry and cannot go stale.
        index = _all_tools()
        total = sum(len(v) for v in index.values())
        available = sorted([*TOPICS, "tools"])

        if topic is None:
            return {
                "topics": available,
                "tool_count": total,
                "categories": sorted(index),
                "overview": TOPICS["overview"],
            }

        key = topic.strip().lower().replace("-", "_").replace(" ", "_")

        if key == "tools":
            return {
                "topic": "tools",
                "tool_count": total,
                "note": (
                    "Derived from the live registry — complete by construction. "
                    "Categories not enabled in MCP_TOOL_CATEGORIES are listed "
                    "here but will not appear in your tool list; that is "
                    "configuration, not a missing feature."
                ),
                "tools": {
                    cat: [{"name": n, "summary": d} for n, d in entries]
                    for cat, entries in sorted(index.items())
                },
            }

        if key not in TOPICS:
            return {
                "error": (
                    f"unknown topic {topic!r}. Available: "
                    f"{', '.join(available)}. Call with no argument for "
                    "the overview."
                ),
                "topics": available,
            }
        return {"topic": key, "guidance": TOPICS[key]}
