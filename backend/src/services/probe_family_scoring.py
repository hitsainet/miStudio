"""Score one input with a whole family (Feature 041 S3; FR-22; T18; FTDD §6.6).

ONE model load per input (`_load_model_for_run` of the first member's run — members share one model and
its quantization), then `score_one(…, loaded=…)` per member: the same scoring path, bands and render notes
as the single-probe score, so a member's family score equals its own. Each named candidate of a blend
report is applied with the report's RECORDED parameters (standardisation, weights, k, threshold) by the
engine's own `apply` / `fires_of` and each member's verdict at the decision the report recorded — the same
functions the report used.

The rule-set verdicts (rules ∪ blend) need 040 S3's single-text rule evaluator, which is not on this
deployment yet: `rules` says `engine_not_available` instead of guessing.
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
    out: Dict[str, Any] = {
        "members": members,
        "rules": {"available": False, "reason": "engine_not_available",
                  "detail": "rule-set verdicts need the behaviour version's rule engine (Feature 040 S3)"},
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
            candidate = cand.parse_name(name, len(recorded), False)
            # A score candidate's recorded params carry its standardisation (mean, std) with its weights.
            value = engine.apply(candidate, entry["params"], scores, n_fired)
            fires = engine.fires_of(candidate, value, entry["calibration"]["threshold"])
            out["candidates"].append({"name": name, "value": float(value[0]), "fires": bool(fires[0])})
    return out
