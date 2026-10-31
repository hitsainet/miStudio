"""`next` — the tool calls an agent can run AS THEY STAND after a result (Feature 039 D-T1, BR-030).

A result that says "the definition is stale" and stops there leaves an agent to guess which tool
fixes it and with what arguments. A `next` list names them: each entry is `{tool, arguments, why}`,
where `tool` is a name in the LIVE MCP registry and `arguments` validate against that tool's input
schema (`tests/support/next_calls.py` checks both against the registry the server builds, so a
renamed tool or argument fails a test rather than an agent). `why` is one sentence for a person and
names no behaviour.

⚠ AN ARGUMENT ONLY THE AGENT CAN SUPPLY IS NAMED, NEVER INVENTED. A probe below rung 2 cannot be
built without `acknowledge_below_rung2_reason` — the agent's OWN reason for exporting weak evidence.
Writing a placeholder reason into `arguments` would make the call succeed on an acknowledgement
nobody gave, so the entry lists the argument under `needs` and leaves it out of `arguments` (review
R1, 2026-10-23: the first `after_build_race` omitted it and said nothing, so for such a probe the
call "as it stands" was refused).

The shape is shared by ENH-006's miStudio features: 039 defines it, 040 and 041 reuse it.
Constructors take ids and values, never a session.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TypedDict

try:  # Python 3.11+
    from typing import NotRequired
except ImportError:  # pragma: no cover
    from typing_extensions import NotRequired


class NextCall(TypedDict):
    tool: str
    arguments: Dict[str, Any]
    why: str
    #: Arguments the call requires that only the agent can supply (never filled in for it).
    needs: NotRequired[List[str]]


#: The evidence rung below which `build_probe_definition` needs the agent's acknowledgement.
ACKNOWLEDGEMENT_RUNG = 2


def build_probe_definition(probe_id: str, why: str, *, rung: Optional[int] = None) -> NextCall:
    call: NextCall = {"tool": "build_probe_definition", "arguments": {"probe_id": probe_id},
                      "why": why}
    if rung is not None and rung < ACKNOWLEDGEMENT_RUNG:
        call["needs"] = ["acknowledge_below_rung2_reason"]
        call["why"] = (
            why + f" The probe is at rung {rung}, below 2, so the build needs your own "
            "acknowledge_below_rung2_reason (at least 10 characters) or it is refused."
        )
    return call


def after_build_race(probe_id: str, *, rung: Optional[int] = None) -> List[NextCall]:
    """A build whose bar or rung moved while it ran: rebuild against the bar that is current now."""
    return [
        build_probe_definition(
            probe_id,
            "The probe's bar or rung changed while this definition was being built, so the file "
            "states the old one; a new build states the current one.",
            rung=rung,
        )
    ]


def recalibrate_preview(probe_id: str, target_fpr: float, why: str) -> NextCall:
    return {
        "tool": "recalibrate_probe_monitor",
        "arguments": {"probe_id": probe_id, "target_fpr": target_fpr, "preview": True},
        "why": why,
    }


def recalibrate_commit(probe_id: str, target_fpr: float, why: str) -> NextCall:
    return {
        "tool": "recalibrate_probe_monitor",
        "arguments": {"probe_id": probe_id, "target_fpr": target_fpr, "preview": False},
        "why": why,
    }


def publish_probe_definition(probe_id: str, published: Sequence[Dict[str, Any]]) -> Optional[NextCall]:
    """Re-publish to the repository the probe was last published to, or `None` when it never was.

    `private` is passed only when the entry recorded it: the tool defaults to private, and a value
    this code guessed would be the one way to make a private probe public by accident.
    """
    entries = [entry for entry in (published or []) if isinstance(entry, dict) and entry.get("repo_id")]
    if not entries:
        return None
    last = entries[-1]
    arguments: Dict[str, Any] = {"probe_id": probe_id, "repo_id": last["repo_id"]}
    if "private" in last and last["private"] is not None:
        arguments["private"] = bool(last["private"])
    return {
        "tool": "publish_probe_definition",
        "arguments": arguments,
        "why": "A copy already published states the old decision and cannot be changed in place; "
               "publishing the rebuilt definition adds one that states the new decision.",
    }


def millm_recalibrate_probe(
    millm_probe_id: str, mistudio_probe_id: str, decision: Dict[str, Any],
    *, decision_digest: Optional[str], decision_revision: Optional[int],
) -> NextCall:
    """Push the committed decision onto one imported miLLM copy, with the bar values it was cut at
    (the contract form miStudio stores, so the tool's equality check against the current bar passes
    as it stands)."""
    arguments: Dict[str, Any] = {
        "probe_id": millm_probe_id,
        "mistudio_probe_id": mistudio_probe_id,
        "threshold": decision.get("threshold"),
        "target_fpr": decision.get("target_fpr"),
        "realised_fpr": decision.get("realised_fpr"),
        "threshold_source": decision.get("threshold_source"),
    }
    if decision.get("windows") is not None:
        arguments["windows"] = decision["windows"]
    if decision.get("length_bands") is not None:
        arguments["length_bands"] = decision["length_bands"]
    if decision_digest is not None:
        arguments["decision_digest"] = decision_digest
    if decision_revision is not None:
        arguments["decision_revision"] = decision_revision
    return {
        "tool": "millm_recalibrate_probe",
        "arguments": arguments,
        "why": "This miLLM copy still serves the previous decision; this pushes the new bar onto it "
               "without re-importing (its recorded verdicts are kept).",
    }


def millm_list_probes(reason: str) -> NextCall:
    return {
        "tool": "millm_list_probes",
        "arguments": {},
        "why": "miLLM could not be asked which copies of this probe it holds "
               f"({reason}); list them to find any that still serve the previous decision.",
    }


def after_recut(
    probe: Any,
    *,
    committed: bool,
    had_definition: bool,
    decision: Optional[Dict[str, Any]],
    decision_digest: Optional[str],
    decision_revision: Optional[int],
    millm_copies: Any,
) -> List[NextCall]:
    """The calls that bring every copy of a re-cut probe up to the committed decision (FTDD §5.3).

    1. `build_probe_definition` when the probe had a definition or published copies;
    2. `publish_probe_definition` to the last repository when it was published;
    3. one `millm_recalibrate_probe` per imported miLLM copy (`millm_copies` a list of ids);
    4. `millm_list_probes` instead, when miLLM was unreachable (`{"unknown": reason}`); nothing when
       miLLM is not configured (`{"not_configured": reason}`) — a standalone miStudio is complete.
    """
    if not committed:
        return []
    out: List[NextCall] = []
    published = list(getattr(probe, "published", None) or [])
    if had_definition or published:
        out.append(build_probe_definition(
            probe.id,
            "The cached definition stated the previous decision and was invalidated; a new build "
            "states the committed one.",
            rung=getattr(probe, "rung", None),
        ))
    publish = publish_probe_definition(probe.id, published)
    if publish is not None:
        out.append(publish)
    if isinstance(millm_copies, list):
        if decision is not None:
            out += [
                millm_recalibrate_probe(copy, probe.id, decision, decision_digest=decision_digest,
                                        decision_revision=decision_revision)
                for copy in millm_copies
            ]
    elif isinstance(millm_copies, dict) and "unknown" in millm_copies:
        out.append(millm_list_probes(str(millm_copies["unknown"])))
    return out
