"""Score one input with a whole family (Feature 041 S3; FR-22; T18; FTDD §6.6).

ONE model load per input (`_load_model_for_run` of the first member's run — members share one model and
its quantization), then `score_one(…, loaded=…)` per member: the same scoring path, bands and render notes
as the single-probe score, so a member's family score equals its own. Each named candidate of a blend
report is applied with the report's RECORDED parameters (standardisation, weights, k, threshold) by the
engine's own `apply` / `fires_of` and each member's verdict at the decision the report recorded — the same
functions the report used.

The rule-set verdict comes from 040's single-text rule evaluator (`rules_evaluation.evaluate_text_sync`, the
function the synchronous `evaluate_rule_signals` tool calls) on the rule set the named report recorded, else the
behaviour's current version's: `rule_match`, `rule_veto`, `unknown` and its reason (operator code with no runner
reads PENDING, never false). A `rules_or_*` candidate fires as `rule_match OR (<name> fires AND NOT rule_veto)`
(040's `detectors`), its `<name>` applied at the union bar the report cut. No rule set: `rules` says `no_rule_set`.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

TRACE_FIELDS = ("tokens", "token_scores")


def score_family(
    db: Any,
    member_ids: Sequence[str],
    *,
    text: Optional[str] = None,
    messages: Optional[Sequence[Dict[str, str]]] = None,
    report: Optional[Dict[str, Any]] = None,
    resolved: Optional[Dict[str, Any]] = None,
    candidates: Optional[Sequence[str]] = None,
    include_traces: bool = False,
    loader: Optional[Callable[[Any], Any]] = None,
    behaviour_key: Optional[str] = None,
) -> Dict[str, Any]:
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorRun
    from . import probe_monitor_run
    from .probe_blend import candidates as cand
    from .probe_blend import engine

    first = db.query(ProbeMonitor).filter(ProbeMonitor.id == member_ids[0]).first()
    if first is None:
        raise ValueError(f"probe {member_ids[0]} not found")
    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == first.run_id).first()
    loaded = (loader or probe_monitor_run._load_model_for_run)(run)          # ⚠ ONCE per input
    members: List[Dict[str, Any]] = []
    for probe_id in member_ids:
        scored = probe_monitor_run.score_one(db, probe_id, text=text, messages=messages, loaded=loaded)
        if not include_traces:
            scored = {k: v for k, v in scored.items() if k not in TRACE_FIELDS}
        members.append(scored)
    scope = str((run.config or {}).get("scope", "all"))
    out: Dict[str, Any] = {
        "members": members,
        "rules": rule_verdict(db, behaviour_key or first.behaviour, resolved, text=text, messages=messages,
                              scope=scope),
        "candidates": [],
    }
    if report is not None and candidates:
        recorded = [m["probe_id"] for m in report["members"]]
        by_id = {m["probe_id"]: m for m in members}
        missing = [p for p in recorded if p not in by_id]
        if missing:
            raise ValueError(f"the report's members {missing} were not scored")
        scores = np.array([[float(by_id[p]["aggregate"]) for p in recorded]], dtype=np.float64)
        n_scored = np.array([[int(by_id[p]["n_scored"]) for p in recorded]], dtype=np.int64)
        decisions = [m["decision"] for m in (resolved or {}).get("members", [])]
        aligned = engine.Aligned(view="input", positions=np.array([0]), labels=np.array([0]), scores=scores,
                                 n_scored=n_scored)
        n_fired = engine.n_fired_of(decisions, aligned) if decisions else np.array([0])
        for name in candidates:
            entry = next((c for c in report["candidates"] if c["name"] == name), None)
            if entry is None:
                raise ValueError(f"{name!r} is not a candidate of report {report['report_id']}")
            candidate = cand.parse_name(name, len(recorded), True)
            base = cand.base_of(candidate)
            # A score candidate's recorded params carry its standardisation (mean, std) with its weights.
            value = engine.apply(base, entry["params"], scores, n_fired)
            fires = engine.fires_of(base, value, entry["calibration"]["threshold"])
            if candidate.rules:
                rules = out["rules"]
                if not rules.get("available"):
                    raise ValueError(f"{name!r} needs the rule set's verdict, which reads {rules.get('reason')!r}")
                verdict = engine.RulesAt(match=np.array([rules["rule_match"]]), veto=np.array([rules["rule_veto"]]),
                                         unknown=np.array([1 if rules["unknown"] else 0], dtype=np.uint8))
                fires = engine.union_fires(verdict, fires)
            out["candidates"].append({"name": name, "value": float(value[0]), "fires": bool(fires[0])})
    return out


def rule_verdict(db: Any, behaviour_key: Optional[str], resolved: Optional[Dict[str, Any]], *,
                 text: Optional[str], messages: Optional[Sequence[Dict[str, str]]], scope: str) -> Dict[str, Any]:
    """The input's rule-set verdict through 040's single-text evaluator: the rule set the named report recorded
    (its candidates were cut against it), else the behaviour's current version's."""
    from ..models.probe_behaviour import ProbeBehaviour, ProbeBehaviourVersion
    from .detector_next import DetectorRefused
    from .rules_evaluation import evaluate_text_sync

    if resolved is not None:
        digest = ((resolved.get("version") or {}).get("rule_set_digest"))
    else:
        head = db.query(ProbeBehaviour).filter(ProbeBehaviour.key == behaviour_key).first() if behaviour_key else None
        version = (db.query(ProbeBehaviourVersion).filter_by(key=head.key, version=head.current_version).first()
                   if head is not None else None)
        digest = version.rule_set_digest if version is not None else None
    if not digest:
        return {"available": False, "reason": "no_rule_set"}
    try:
        verdict = evaluate_text_sync(db, digest, text=text, messages=list(messages) if messages is not None else None,
                                     scope=scope)
    except DetectorRefused as refusal:          # the set cannot be used now (FR-29) or no runner can run its code
        return {"available": False, "reason": refusal.code, "rule_set_digest": digest}
    return {"available": True, **verdict}
