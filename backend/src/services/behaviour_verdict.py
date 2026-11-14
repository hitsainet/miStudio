"""The behaviour verdict: one served answer for a probe stack (ENH-006 G19; operator decisions G19-1 … G19-6,
2026-11-08; design `miSuite/specs/ENH-006-dlp-datasets/G19_behaviour_verdict.md`).

G19-1 REVISES D9: miServer serves an ADDITIVE behaviour-level verdict built from a definition miWorkbench exports;
the member results stay as they are, so an application can still blend for itself. miWorkbench owns the
definition (`docs/schemas/behaviour-verdict-v1.json`, `mistudio.behaviour-verdict/v1`) and the shared cases
(`docs/schemas/behaviour-verdict-cases.json`); miServer vendors both byte-identically and runs every case through
its own combiner.

**The combiner (`combine`), per behaviour and window** — 040's union and its precedence, so what is served is
what 041 measured:

1. **Rule match** (the rule set's positive after the vetoes: D53's explicit `match`, G19-4) → `fires: true`,
   `decided_by: rule_match`. A match beats a veto (040 FR-27).
2. **Rule veto** (a veto removed a span and nothing matched) suppresses the members INSIDE the veto's scope
   (G19-3: declared per member, default every member).
3. **Members**: the candidate over the members not suppressed — a verdict kind counts members whose OWN served
   verdict fired (`n >= k`, no bar of its own); a score kind standardises each member's score with the recorded
   mean and standard deviation, applies the recorded weights and compares `>=` one global bar. Each through the
   blend engine's own `apply` / `fires_of`, then 040's `union_fires` — never a copy.
4. **Unknown rules** (timed out, errored, pending operator code, or no rule side at all for a `rules_or_*`
   candidate) are no match and no veto: the members decide (040 FR-35); `rules.unknown` records it.
5. **Missing members** (not armed, not scored, or no decision) → G19-5: `fires: true` only when a rule or a
   present member PROVES it; otherwise `fires: null` with `decided_by: members_missing` — never `false`.

A score kind needs every member in the veto's scope (a blended value cannot drop one member): `export` refuses
a partial scope there (`veto_scope_needs_verdict_kind`).

⚠ NO TEXT. A definition carries ids, digests, numbers and expression names; a served verdict carries no text and
no span.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .detector_identity import jcs_bytes, sha256_hex
from .detector_next import DetectorRefused
from .probe_blend import candidates as cand
from .probe_blend import engine
from .probe_blend import report_format

SCHEMA = "mistudio.behaviour-verdict/v1"
CASES_FORMAT = "mistudio.behaviour-verdict-cases/1"
#: A combiner vector's blended value agrees within this (summation order differs between NumPy and a consumer's
#: plain loop); `fires` must agree exactly.
VALUE_TOLERANCE = 1e-9

#: The fixed sentence every definition carries.
STATEMENT = (
    "A behaviour verdict has no rung of its own; each member probe is judged by its own rung and its own "
    "decision. miServer serves this verdict beside the member results, which are unchanged (G19-1 revises D9)."
)

DECIDED_BY = ("rule_match", "members", "vetoed", "members_missing")


class VerdictRefused(DetectorRefused):
    """An export refused; the body is data (`code`, the reasons as fields, `next`)."""


# ── identity ─────────────────────────────────────────────────────────────────────────────────


def members_digest(detector_digests: Sequence[str]) -> str:
    """sha256 of the JCS (RFC 8785) of the SORTED detector digests — order-free, so both repos compute it alike
    from the definition's own `members` (miServer's `set` block digests miServer ids; this one digests detectors)."""
    return sha256_hex(jcs_bytes(sorted(str(d) for d in detector_digests)))


def definition_bytes(definition: Dict[str, Any]) -> bytes:
    """The definition's canonical bytes: 040's rounding, then plain JCS — two exports of one report are equal."""
    return report_format.canonical_bytes(definition)


def definition_sha256(definition: Dict[str, Any]) -> str:
    return hashlib.sha256(definition_bytes(definition)).hexdigest()


# ── the combiner ─────────────────────────────────────────────────────────────────────────────


def candidate_of(definition: Dict[str, Any]) -> cand.Candidate:
    c = definition["candidate"]
    kind = cand.REGISTRY[c["kind"]]
    return cand.Candidate(name=c["name"], kind=kind.name, produces=kind.produces, k=c.get("k"),
                          rules=bool(c.get("rules")))


def combine(definition: Dict[str, Any], members_in: Dict[str, Optional[Dict[str, Any]]],
            rules_in: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One behaviour verdict from one window's member inputs and rule inputs (see the module docstring).

    `members_in`: detector digest → `{"fires": bool | None, "score": float | None}`, or None / absent for a
    member that is not armed or produced nothing. `rules_in`: `{"match", "veto", "unknown"}` or None."""
    candidate = candidate_of(definition)
    base = cand.base_of(candidate)
    members = definition["members"]
    digests = [m["detector_digest"] for m in members]
    scoped = [bool(m.get("veto_applies", True)) for m in members]
    full_scope = all(scoped)

    unknown = candidate.rules and (rules_in is None or bool(rules_in.get("unknown")))
    match = bool(candidate.rules and not unknown and rules_in.get("match"))
    veto = bool(candidate.rules and not unknown and rules_in.get("veto") and not match)

    present = [members_in.get(d) for d in digests]
    needed = "fires" if base.produces == "verdict" else "score"
    missing = [d for d, p in zip(digests, present) if p is None or p.get(needed) is None]
    fired = [d for d, p in zip(digests, present) if p is not None and p.get("fires") is True]
    vetoed = [d for d, s in zip(digests, scoped) if veto and s and d in fired]

    # 040's union: with every member in scope it applies the veto to the whole blend; a partial scope has
    # already suppressed the in-scope members, so the union gets no veto of its own.
    rules_at = engine.RulesAt(match=np.array([match]), veto=np.array([veto and full_scope]),
                              unknown=np.array([1 if unknown else 0], dtype=np.uint8))
    value: Optional[float] = None
    if base.produces == "verdict":
        counted = [d for d in fired if full_scope or d not in vetoed]
        n = np.array([float(len(counted))])
        value = float(engine.apply(base, {}, np.zeros((1, len(digests))), n)[0])
        base_fires = engine.fires_of(base, np.array([value]), None)
    elif missing:
        base_fires = np.array([False])
    else:
        scores = np.array([[float(p["score"]) for p in present]], dtype=np.float64)
        value = float(engine.apply(base, definition["candidate"]["params"], scores, np.zeros(1))[0])
        base_fires = engine.fires_of(base, np.array([value]), definition["candidate"]["threshold"])
    final = bool(engine.union_fires(rules_at, base_fires)[0]) if candidate.rules else bool(base_fires[0])

    if final:
        fires: Optional[bool] = True
        decided_by = "rule_match" if match else "members"
    elif missing:
        fires, decided_by = None, "members_missing"            # G19-5: never a false clean
    elif veto:
        fires, decided_by = False, "vetoed"
    else:
        fires, decided_by = False, "members"
    return {
        "fires": fires,
        "decided_by": decided_by,
        "members_fired": fired,
        "members_vetoed": vetoed,
        "members_missing": missing,
        "rules": ({"match": None if unknown else match, "veto": None if unknown else veto, "unknown": bool(unknown)}
                  if candidate.rules else None),
        "value": value,
    }


def agrees(got: Dict[str, Any], expect: Dict[str, Any]) -> bool:
    """A consumer's verdict agrees with an expected one: every field exact, the value within `VALUE_TOLERANCE`."""
    for key in ("fires", "decided_by", "members_fired", "members_vetoed", "members_missing", "rules"):
        if got.get(key) != expect.get(key):
            return False
    a, b = got.get("value"), expect.get("value")
    if (a is None) != (b is None):
        return False
    return a is None or abs(float(a) - float(b)) <= VALUE_TOLERANCE


# ── combiner vectors (layer 3 of parity) ─────────────────────────────────────────────────────


def _member_input(produces: str, fires: Optional[bool], score: Optional[float]) -> Dict[str, Any]:
    return {"fires": fires, "score": score}


def vector_inputs(definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic member and rule inputs that exercise every branch the definition can take: nothing fires,
    each member alone, every member, the first member missing (alone and beside a firing member), and — for a
    `rules_or_*` candidate — a match, a veto over each member, and an unknown rule. Score kinds use scores at the
    recorded mean and three standard deviations either side of it."""
    candidate = candidate_of(definition)
    base = cand.base_of(candidate)
    digests = [m["detector_digest"] for m in definition["members"]]
    n = len(digests)
    params = definition["candidate"].get("params") or {}
    mean = params.get("mean") or [0.0] * n
    std = params.get("std") or [1.0] * n

    def member_set(firing: Sequence[int], missing: Sequence[int] = ()) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for j, d in enumerate(digests):
            if j in missing:
                out[d] = None
                continue
            on = j in firing
            score = float(mean[j]) + (3.0 if on else -3.0) * float(std[j]) if base.produces == "score" else None
            out[d] = _member_input(base.produces, on, score)
        return out

    shapes: List[Dict[str, Any]] = [{"id": "none_fire", "members": member_set([])},
                                    {"id": "all_fire", "members": member_set(range(n))}]
    shapes += [{"id": f"only_{j}", "members": member_set([j])} for j in range(n)]
    shapes += [{"id": "first_missing_none_fire", "members": member_set([], missing=[0])}]
    if n > 1:
        shapes += [{"id": "first_missing_second_fires", "members": member_set([1], missing=[0])}]
    vectors: List[Dict[str, Any]] = []
    if not candidate.rules:
        for s in shapes:
            vectors.append({"id": s["id"], "members": s["members"], "rules": None})
        return vectors
    clean = {"match": False, "veto": False, "unknown": False}
    for s in shapes:
        vectors.append({"id": s["id"], "members": s["members"], "rules": dict(clean)})
    vectors.append({"id": "match_none_fire", "members": member_set([]), "rules": {**clean, "match": True}})
    vectors.append({"id": "match_and_veto", "members": member_set(range(n)), "rules": {**clean, "match": True, "veto": True}})
    for j in range(n):
        vectors.append({"id": f"veto_only_{j}", "members": member_set([j]), "rules": {**clean, "veto": True}})
    vectors.append({"id": "unknown_all_fire", "members": member_set(range(n)), "rules": {**clean, "unknown": True}})
    vectors.append({"id": "unknown_none_fire", "members": member_set([]), "rules": {**clean, "unknown": True}})
    return vectors


def combiner_vectors(definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each input with the verdict THIS combiner gives on the definition's own (rounded) fields — a consumer
    reproduces every expectation from the file alone (suite memory `a-contract-must-be-reproducible-from-its-own-fields`)."""
    from .probe_blend.report_format import round_floats

    rounded = round_floats(definition)
    out = []
    for v in vector_inputs(rounded):
        out.append({**v, "expect": combine(rounded, v["members"], v["rules"])})
    return out


# ── the export (sliced from a stored 041 report; nothing recomputed) ─────────────────────────

WINDOWS = ("all", "prompt", "response", "last_user")
#: The candidate parameters a consumer applies, by base kind; everything else in the report's params
#: (C, max_iter, the rule identity) is evidence, not an input.
SCORE_PARAMS = {"stack": ("mean", "std", "weights", "intercept"), "max": ("mean", "std"),
                "best_single": ("mean", "std", "member_index")}


def build_definition(report: Dict[str, Any], entry: Dict[str, Any], *, report_sha256: str,
                     member_meta: Dict[str, Dict[str, Any]], match: Optional[Dict[str, Any]],
                     windows: Sequence[str], acknowledge_reason: Optional[str], actor: str) -> Dict[str, Any]:
    """PURE: the definition for one candidate of one stored blend report. Every number is the report's
    RECORDED value (041 FR-21's rule for the suggested blend): the candidate's params and bar, the
    calibration's realised rate. G19-6: over the target on the calibration negatives → refused unless
    acknowledged with a reason, which the definition then records."""
    from . import detector_next as nx

    candidate = cand.parse_name(entry["name"], len(report["members"]), True)
    base = cand.base_of(candidate)
    calibration = entry.get("calibration") or {}
    if calibration.get("refused"):
        raise VerdictRefused(409, "candidate_refused", f"{entry['name']} was refused at calibration "
                             f"({calibration['refused']}); it has no bar to serve",
                             {"candidate": entry["name"], "refused": calibration["refused"]})
    recorded = dict(entry.get("params") or {})
    scope = recorded.get("veto_scope")
    if scope is not None and base.produces != "verdict":
        raise VerdictRefused(422, "veto_scope_needs_verdict_kind", "a veto scoped to some members needs a "
                             "verdict-kind candidate", {"candidate": entry["name"], "veto_scope": scope})
    unbound = [m["probe_id"] for m in report["members"] if not m.get("detector_digest")]
    if unbound:
        # miLLM binds members by detector digest; a member whose definition was never built has none.
        raise VerdictRefused(409, "member_identity_missing", f"members {unbound} have no detector digest (no built "
                             "definition); miServer binds a verdict's members by it", {"probe_ids": unbound},
                             next=[nx.next_build_member(pid, None) for pid in unbound])
    members = []
    for m in report["members"]:
        meta = member_meta.get(m["probe_id"]) or {}
        members.append({
            "detector_digest": m["detector_digest"],
            "decision_digest": m.get("decision_digest"),
            "decision_revision": m.get("decision_revision"),
            "probe_id": m["probe_id"],
            "name": meta.get("name"),
            "role": meta.get("role"),
            "forms": [m["sub_question"]] if m.get("sub_question") else [],
            "veto_applies": candidate.rules and (scope is None or m["probe_id"] in set(scope)),
        })
    params = {k: recorded[k] for k in SCORE_PARAMS.get(base.kind, ()) if k in recorded}
    target = float(report["behaviour"]["target_fpr"])
    realised = calibration.get("realised_fpr")
    if realised is None:
        raise VerdictRefused(409, "union_not_measured", f"{entry['name']} has no realised false-positive rate on "
                             "the calibration negatives", {"candidate": entry["name"]})
    over = float(realised) > target + 1e-12
    acknowledged = None
    if over:
        if not acknowledge_reason or not str(acknowledge_reason).strip():
            raise VerdictRefused(
                409, "union_over_budget",
                f"{entry['name']} fires on {float(realised):.6f} of the calibration negatives, over the target "
                f"{target}; export it only with acknowledge_over_budget naming why",
                {"candidate": entry["name"], "realised_union_fpr": float(realised), "target_fpr": target,
                 "n_negatives": calibration.get("n_negatives")},
                next=[nx.next_export_behaviour_verdict(report["behaviour"]["key"], report["model_id"],
                                                       report["report_id"], entry["name"])])
        acknowledged = {"reason": str(acknowledge_reason).strip(), "by": actor}
    rule_set = (report["behaviour"].get("rule_set") or {}) if candidate.rules else {}
    if candidate.rules and not rule_set.get("digest"):
        raise VerdictRefused(409, "no_rule_set", f"{entry['name']} names the rules but the report read none",
                             {"candidate": entry["name"]})
    definition: Dict[str, Any] = {
        "schema": SCHEMA,
        "behaviour": {"key": report["behaviour"]["key"], "version": report["behaviour"]["version"]},
        "model": {"id": report["model_id"], "template_sha256": None},
        "members": members,
        "members_digest": members_digest([m["detector_digest"] for m in members]),
        "candidate": {"name": entry["name"], "kind": base.kind, "rules": candidate.rules,
                      "k": base.k if base.produces == "verdict" else None,
                      "params": params, "threshold": calibration.get("threshold") if base.produces == "score" else None},
        "rules": ({"rule_set_digest": rule_set["digest"], "engine_digest": rule_set.get("engine_digest"),
                   "match": match, "veto_scope": "all" if scope is None else "per_member"}
                  if candidate.rules else None),
        "windows": list(windows),
        "target_fpr": target,
        "calibration": {"report_id": report["report_id"], "report_sha256": report_sha256,
                        "view_id": (report.get("views") or {}).get("calibration", {}).get("id"),
                        "n_negatives": calibration.get("n_negatives"),
                        "realised_union_fpr": float(realised), "over_budget": over, "acknowledged": acknowledged},
        "end_to_end_vectors": [],
        "end_to_end_status": "not_generated",
        "statement": STATEMENT,
    }
    definition["combiner_vectors"] = combiner_vectors(definition)
    return report_format.round_floats(definition)


async def export(db: Any, key: str, model: str, report_id: str, candidate: str, *, windows: Optional[Sequence[str]],
                 acknowledge_over_budget: Optional[str], actor: str) -> Dict[str, Any]:
    """The definition of one candidate of a done, CURRENT report (`blend_report_stale` otherwise), its canonical
    bytes' sha256 and the bytes as text. Nothing is stored and nothing GPU-bound runs."""
    from sqlalchemy import select

    from ..models.probe_behaviour import ProbeBehaviourProbeRole
    from . import detector_next as nx
    from .probe_blend import service as blends

    windows = list(windows or ["all"])
    bad = [w for w in windows if w not in WINDOWS] or ([] if len(set(windows)) == len(windows) else windows)
    if bad or not windows:
        raise VerdictRefused(422, "invalid_field", f"windows must be distinct names from {list(WINDOWS)}",
                             {"field": "windows", "got": windows})
    row = await blends.get_row(db, report_id)
    if row.behaviour_key != key or row.model_id != model:
        raise VerdictRefused(422, "report_not_of_behaviour", f"report {report_id} is of {row.behaviour_key!r} on "
                             f"{row.model_id!r}, not {key!r} on {model!r}",
                             {"report_behaviour": row.behaviour_key, "report_model": row.model_id})
    raw = blends.report_bytes(row)
    reasons = await blends.stale_reasons(db, row)
    if reasons:
        raise VerdictRefused(409, "blend_report_stale", f"report {report_id} is stale", {"stale_reasons": reasons},
                             next=[nx.next_rerun_report(report_id)])
    report = json.loads(raw)
    entry = blends.candidate_of(report, candidate)
    ids = [m["probe_id"] for m in report["members"]]
    # A probe row has no name column; the definition's `name` stays null (display only, "not recorded").
    roles = {pid: role for pid, role in (await db.execute(
        select(ProbeBehaviourProbeRole.probe_id, ProbeBehaviourProbeRole.role)
        .where(ProbeBehaviourProbeRole.probe_id.in_(ids), ProbeBehaviourProbeRole.key == key))).all()}
    meta = {pid: {"name": None, "role": roles.get(pid, "candidate")} for pid in ids}
    match = None
    if cand.is_rules_name(candidate):
        match = await match_of_report(db, report)
    definition = build_definition(report, entry, report_sha256=hashlib.sha256(raw).hexdigest(), member_meta=meta,
                                  match=match, windows=windows, acknowledge_reason=acknowledge_over_budget,
                                  actor=actor)
    body = definition_bytes(definition)
    return {"definition": json.loads(body), "definition_sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body)}


async def match_of_report(db: Any, report: Dict[str, Any]) -> Dict[str, Any]:
    """The rule set's positive as miWorkbench's evaluator reads it (`compose`'s `match_source` and expressions):
    D53's explicit `match`, else the inference 040 used offline — so the served match is the measured one."""
    from .. import rule_signals
    from . import rule_set_store

    digest = ((report["behaviour"].get("rule_set") or {}).get("digest"))
    stored = await rule_set_store.get_by_digest(db, digest) if digest else None
    if stored is None:
        raise VerdictRefused(409, "rule_set_missing", f"the report's rule set {digest} is not stored here",
                             {"rule_set_digest": digest})
    compiled = rule_signals.load(rule_set_store.document_of(stored)[0], rule_set_store.engine_limits())
    return {"source": compiled.match_source, "any_of": list(compiled.match_expressions)}
