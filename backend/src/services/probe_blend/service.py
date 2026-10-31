"""Blend evaluation: start, resolution, refusals, admission, staleness, rerun, prune (Feature 041 S2;
FR-6, FR-7, FR-14, FR-16, FR-17, FR-19, FR-21; FTDD §6.3, FTID §3.7 – §3.8).

The service does the reads, the refusals and the assembly; the maths is `engine` (pure), the I/O and the
status the task (`workers/probe_blend_tasks.py`). ⚠ EVERY REFUSAL HAPPENS AT THE CALL, BEFORE A ROW EXISTS
(PT-5): a missing evaluation, misaligned members, an overlapping `blend_fit` view — nothing is queued and
nothing GPU-bound is started; `next` names the exact calls (approval-gated where they are GPU work).

`resolved` fixes every input at start (members with their decision inputs and digests, view ids by role,
evaluation ids, candidates, folds, seed, sample size): a member re-cut while the report runs does not move
it — the report finishes and reads `stale` at once (FR-17).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ...models.probe_behaviour import ProbeBehaviour, ProbeBehaviourVersion, ProbeBehaviourViewLink
from ...models.probe_blend import (
    ProbeBlendReport,
    ProbeBlendReportMember,
    ProbeSuggestedBlend,
    ProbeSuggestedBlendHistory,
)
from ...models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorEvaluation, ProbeMonitorRun
from .. import detector_next as nx
from ..detector_next import DetectorRefused
from . import candidates as cand
from . import engine
from . import report_format

ACTIVE = ("queued", "running")


class BlendRefused(DetectorRefused):
    """A blend request refused; the body is data (`code`, the reasons as fields, `next`)."""


def blends_root() -> Path:
    from ...core.config import settings

    return Path(settings.data_dir) / "probe_monitors" / "blends"


def probes_root() -> Path:
    from ...core.config import settings

    return Path(settings.data_dir) / "probe_monitors"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _effective():
    from ...core.runtime_settings import effective

    return effective()


# ── views ────────────────────────────────────────────────────────────────────────────────────


async def candidate_views(db: AsyncSession, key: str, dataset_version: Optional[str]) -> List[Tuple[ProbeMonitorDataset, str]]:
    """`(view, role)` for every unarchived view carrying the key or linked to it (a link's `use` overrides
    the view's role, 037), restricted to `dataset_version` (`dataset_version->>'version_id'`, OI-12)."""
    links = {lid: use for lid, use in (await db.execute(
        select(ProbeBehaviourViewLink.dataset_id, ProbeBehaviourViewLink.use).where(ProbeBehaviourViewLink.key == key)
    )).all()}
    query = select(ProbeMonitorDataset).where(
        or_(ProbeMonitorDataset.behaviour == key, ProbeMonitorDataset.id.in_(list(links) or [""])),
        ProbeMonitorDataset.archived_at.is_(None),
    )
    if dataset_version is not None:
        query = query.where(ProbeMonitorDataset.dataset_version["version_id"].astext == dataset_version)
    views = (await db.execute(query.order_by(ProbeMonitorDataset.created_at, ProbeMonitorDataset.id))).scalars().all()
    return [(view, links.get(view.id) or view.role) for view in views]


def held_out(view: ProbeMonitorDataset) -> bool:
    return view.distribution == "out_of_distribution"


async def resolve_views(db: AsyncSession, key: str, dataset_version: Optional[str]) -> Dict[str, Any]:
    """FR-7: exactly one `blend_fit`, the `eval` views as tests, exactly one `calibration`."""
    found = await candidate_views(db, key, dataset_version)
    by_role: Dict[str, List[ProbeMonitorDataset]] = {}
    for view, role in found:
        by_role.setdefault(role, []).append(view)
    for role in ("blend_fit", "calibration"):
        views = by_role.get(role, [])
        if not views:
            raise BlendRefused(422, "view_missing", f"no unarchived {role} view of {key!r}"
                               + (f" from dataset version {dataset_version!r}" if dataset_version else ""),
                               {"role": role, "dataset_version": dataset_version},
                               next=[nx.next_register_view(key, role)])
        if len(views) > 1:
            raise BlendRefused(422, "view_ambiguous", f"{len(views)} {role} views of {key!r}; name one",
                               {"role": role, "candidates": [v.id for v in views]})
    tests = by_role.get("eval", [])
    if not tests:
        raise BlendRefused(422, "view_missing", f"no unarchived evaluation (test) view of {key!r}",
                           {"role": "eval", "dataset_version": dataset_version},
                           next=[nx.next_register_view(key, "eval")])
    return {"blend_fit": by_role["blend_fit"][0], "tests": tests, "calibration": by_role["calibration"][0]}


def refuse_member_training_tests(test_ids: List[str], members: List[Dict[str, Any]]) -> None:
    """D46 (operator, 2026-10-24): a TEST view that is any member probe's own TRAINING view (the view the
    member's run trained on) is refused by name, as FR-14 refuses the blend_fit view: a member judged on the
    rows it was trained on reads better than it is, and so does every blend of it. Another probe's training
    view (not a member) is not refused."""
    trained_by: Dict[str, List[str]] = {}
    for member in members:
        if member.get("training_view"):
            trained_by.setdefault(member["training_view"], []).append(member["probe_id"])
    hits = [t for t in test_ids if t in trained_by]
    if hits:
        first = hits[0]
        raise BlendRefused(422, "test_view_is_member_training_view",
                           f"test view {first} is the training view of member probe(s) {trained_by[first]}; a "
                           "member judged on the rows it was trained on reads better than it is",
                           {"view": first, "probe_ids": sorted(trained_by[first]),
                            "test_views": hits, "members": {t: sorted(trained_by[t]) for t in hits}})


async def named_views(db: AsyncSession, views: Dict[str, Any],
                      members: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """FR-30: explicitly named views bypass resolution; each must exist and carry its role."""
    ids = [views["blend_fit"], *views["tests"], views["calibration"]]
    # FR-14 (Review R1): naming the blend_fit view among the tests is the overlap FR-14 refuses by name; the
    # role check below would otherwise answer it first as a role mismatch (`view_missing`).
    if views["blend_fit"] in views["tests"]:
        raise BlendRefused(422, "blend_fit_overlap",
                           f"the blend_fit view {views['blend_fit']} is also named as a test view; a blend fitted "
                           "on rows it is judged on reads better than it is",
                           {"view": views["blend_fit"], "training_views": [], "test_views": [views["blend_fit"]]})
    # D46: likewise before the role check — a member's training view named as a test reads its own code.
    refuse_member_training_tests(list(views["tests"]), members or [])
    rows ={v.id: v for v in (await db.execute(select(ProbeMonitorDataset).where(ProbeMonitorDataset.id.in_(ids)))).scalars()}
    missing = [i for i in ids if i not in rows]
    if missing:
        raise BlendRefused(422, "view_missing", f"no view {missing}", {"views": missing})
    for vid, role in [(views["blend_fit"], "blend_fit"), *[(t, "eval") for t in views["tests"]],
                      (views["calibration"], "calibration")]:
        if rows[vid].role != role:
            raise BlendRefused(422, "view_missing", f"view {vid} has role {rows[vid].role}, not {role}",
                               {"view": vid, "role": rows[vid].role, "wanted": role})
    return {"blend_fit": rows[views["blend_fit"]], "tests": [rows[t] for t in views["tests"]],
            "calibration": rows[views["calibration"]]}


# ── evaluations ──────────────────────────────────────────────────────────────────────────────


async def choose_evaluations(db: AsyncSession, members: Sequence[Dict[str, Any]], views: Dict[str, Any],
                             sample_rows: int) -> Dict[str, Dict[str, str]]:
    """T5: per (member, view) the newest `completed` evaluation with row positions — on the calibration
    view also `refused` (no positives, by construction) with `metrics.sample.rows` equal to the request's
    sample size. Missing → `member_scores_missing` naming EXACTLY the evaluations to run (never started)."""
    probe_ids = [m["probe_id"] for m in members]
    fit_and_tests = [views["blend_fit"].id, *[v.id for v in views["tests"]]]
    calibration = views["calibration"].id
    rows = (await db.execute(
        select(ProbeMonitorEvaluation).where(ProbeMonitorEvaluation.probe_id.in_(probe_ids),
                                             ProbeMonitorEvaluation.dataset_id.in_([*fit_and_tests, calibration]))
        .order_by(ProbeMonitorEvaluation.created_at.desc(), ProbeMonitorEvaluation.id.desc())
    )).scalars().all()
    chosen: Dict[str, Dict[str, str]] = {p: {} for p in probe_ids}
    without_positions: List[Dict[str, str]] = []
    other_sizes: List[Dict[str, Any]] = []
    for row in rows:
        picked = chosen[row.probe_id]
        if row.dataset_id in picked:
            continue
        if row.dataset_id == calibration:
            if row.status not in ("completed", "refused"):
                continue
            size = ((row.metrics or {}).get("sample") or {}).get("rows")
            if size != sample_rows:
                other_sizes.append({"probe_id": row.probe_id, "evaluation_id": row.id, "sample_rows": size})
                continue
            if not row.positions_path:
                without_positions.append({"probe_id": row.probe_id, "evaluation_id": row.id})
                continue
            picked[row.dataset_id] = row.id
        else:
            if row.status != "completed" or not row.positions_path:
                continue
            picked[row.dataset_id] = row.id
    missing_fit: Dict[str, List[str]] = {}
    missing_cal: List[str] = []
    for probe_id, picked in chosen.items():
        absent = [v for v in fit_and_tests if v not in picked]
        if absent:
            missing_fit[probe_id] = absent
        if calibration not in picked:
            missing_cal.append(probe_id)
    if without_positions and set(missing_cal) <= {w["probe_id"] for w in without_positions}:
        raise BlendRefused(
            409, "calibration_positions_not_recorded",
            "a member's calibration evaluation at this sample size has no row positions; re-evaluate it",
            {"evaluations": without_positions, "view": calibration, "sample_rows": sample_rows},
            next=[nx.next_evaluate_sampled(w["probe_id"], [calibration], sample_rows) for w in without_positions],
        )
    if missing_cal and not missing_fit and other_sizes and set(missing_cal) <= {o["probe_id"] for o in other_sizes}:
        raise BlendRefused(
            409, "calibration_samples_differ",
            f"members' calibration evaluations used other sample sizes than {sample_rows}; evaluate each at "
            "that size so every member scores the same rows",
            {"view": calibration, "sample_rows": sample_rows, "found": other_sizes},
            next=[nx.next_evaluate_sampled(p, [calibration], sample_rows) for p in missing_cal],
        )
    if missing_fit or missing_cal:
        calls = [nx.next_evaluate_sampled(p, views_, None) for p, views_ in missing_fit.items()]
        calls += [nx.next_evaluate_sampled(p, [calibration], sample_rows) for p in missing_cal]
        raise BlendRefused(
            409, "member_scores_missing",
            "some members have no stored per-row scores on a view the blend needs; run the evaluations "
            "below (each waits for the operator's approval), then ask again",
            {"missing": {**{p: v for p, v in missing_fit.items()},
                         **{p: missing_fit.get(p, []) + [calibration] for p in missing_cal}},
             "sample_rows": sample_rows},
            next=calls,
        )
    return chosen


def load_member_arrays(evaluation: ProbeMonitorEvaluation, root: Path) -> engine.MemberArrays:
    from ..probe_evaluation_scores import load_evaluation_arrays

    loaded = load_evaluation_arrays(evaluation, root=root)
    return engine.MemberArrays(scores=np.asarray(loaded.arrays["score"]), positions=np.asarray(loaded.arrays["position"]),
                               labels=np.asarray(loaded.arrays["label"]), n_scored=np.asarray(loaded.arrays["n_scored"]))


def _engine_refusal(exc: engine.EngineRefused, members: Sequence[str], key: str, model: str) -> BlendRefused:
    status = 422 if exc.code == "member_scores_constant" else 409
    nxt = []
    if exc.code == "member_scores_constant":
        rest = [m for m in members if m != exc.details["member"]]
        nxt = [nx.next_evaluate_blends(key, model, rest)]
    return BlendRefused(status, exc.code, str(exc), dict(getattr(exc, "details", {})), next=nxt)


# ── start ────────────────────────────────────────────────────────────────────────────────────


async def start(db: AsyncSession, body: Dict[str, Any], actor: str) -> ProbeBlendReport:
    """FR-6: resolve, refuse, admit, insert (`queued`), commit. The ROUTE dispatches after the commit."""
    from .. import behaviour_catalogue as catalogue
    from .. import probe_family
    from ...core.config import settings as static
    from ...schemas.probe_monitor import DEFAULT_CALIBRATION_MAX_ROWS
    from ..probe_dataset_lifecycle import refuse_archived_inputs

    from ...core.runtime_settings import effective

    _settings = effective()
    key = body["behaviour_key"]
    head = await catalogue.get_head(db, key)
    if head.archived_at is not None:
        raise BlendRefused(409, "behaviour_archived", f"{key!r} is archived", {"key": key},
                           next=[nx.next_unarchive_behaviour(key)])
    version = (await db.execute(select(ProbeBehaviourVersion).where(
        ProbeBehaviourVersion.key == key, ProbeBehaviourVersion.version == head.current_version))).scalar_one()
    resolved_family = await probe_family.resolve_members(db, key, model=body.get("model"),
                                                         member_ids=body.get("member_ids"))
    model = resolved_family["model"]
    members = resolved_family["members"]
    if body.get("member_ids") is None:
        members = [m for m in members if m["probe_id"] in resolved_family["default_blend_set"]]
    cap = int(_settings.blend_max_members)
    if len(members) > cap:
        raise BlendRefused(422, "cap_exceeded", f"{len(members)} members; blend_max_members allows {cap}",
                           {"setting": "blend_max_members", "value": cap, "got": len(members)})
    if len(members) < 2:
        raise BlendRefused(422, "members_too_few", f"a blend needs at least two members; the family has {len(members)}",
                           {"members": [m["probe_id"] for m in members]},
                           next=[nx.next_get_family(key, model)] if model else [])
    for field in ("precision", "render_form", "scope"):
        values = sorted({json.dumps(m[field]) for m in members})
        if len(values) > 1:
            raise BlendRefused(422, "members_incompatible",
                               f"members differ in {field.replace('_', ' ')}: {[json.loads(v) for v in values]}",
                               {"field": field, "values": {m["probe_id"]: m[field] for m in members}})

    if body.get("views"):
        views = await named_views(db, body["views"], members)
        dataset_version = None
    else:
        dataset_version = body.get("dataset_version")
        views = await resolve_views(db, key, dataset_version)
    view_ids = [views["blend_fit"].id, *[v.id for v in views["tests"]], views["calibration"].id]
    try:
        await refuse_archived_inputs(db, view_ids)
    except Exception as exc:  # HTTPException 409 probe_dataset_archived → the 041 code
        detail = getattr(exc, "detail", {}) or {}
        archived = detail.get("archived") if isinstance(detail, dict) else None
        if not archived:
            raise
        raise BlendRefused(409, "view_archived", detail.get("message", "a view is archived"), {"archived": archived},
                           next=[nx.next_unarchive_view(v) for v in archived]) from exc
    fit_id = views["blend_fit"].id
    trained_on = {m["training_view"] for m in members}
    if fit_id in trained_on or fit_id in {v.id for v in views["tests"]}:
        raise BlendRefused(422, "blend_fit_overlap",
                           f"the blend_fit view {fit_id} is a member's training view or a test view; a blend fitted "
                           "on rows it is judged on reads better than it is",
                           {"view": fit_id, "training_views": sorted(trained_on & {fit_id}),
                            "test_views": [v.id for v in views["tests"] if v.id == fit_id]})
    # D46: the derived path (and the named one again, harmlessly) — before any evaluation work starts.
    refuse_member_training_tests([v.id for v in views["tests"]], members)

    sample_rows = int(body.get("calibration_sample_rows") or version.calibration_floor or DEFAULT_CALIBRATION_MAX_ROWS)
    chosen = await choose_evaluations(db, members, views, sample_rows)

    # Alignment at the call (PT-5): positions and labels only, in a thread.
    ids = sorted({e for picked in chosen.values() for e in picked.values()})
    evaluations = {e.id: e for e in (await db.execute(
        select(ProbeMonitorEvaluation).where(ProbeMonitorEvaluation.id.in_(ids)))).scalars()}
    member_ids = [m["probe_id"] for m in members]
    root = probes_root()

    def _check() -> None:
        for view in view_ids:
            arrays = {p: load_member_arrays(evaluations[chosen[p][view]], root) for p in member_ids}
            aligned = engine.align(view, member_ids, arrays, float(_settings.blend_min_aligned_fraction))
            if view == fit_id:
                engine.standardise_fit(aligned.scores, member_ids)

    try:
        await run_in_threadpool(_check)
    except engine.EngineRefused as exc:
        raise _engine_refusal(exc, member_ids, key, model) from exc

    has_rule_set = version.rule_set_digest is not None
    flags = [] if has_rule_set else ["no_rule_set"]
    try:
        requested = body.get("candidates")
        rules_named = [n for n in (requested or []) if cand.is_rules_name(n)]
        if rules_named and not has_rule_set:
            cand.parse_name(rules_named[0], len(members), True)          # an unknown name still reads unknown
            raise BlendRefused(422, "no_rule_set",
                               f"version {version.version} of {key!r} names no rule set, so there is no rules ∪ "
                               "candidate; attach one with a new version, or ask for probe-only candidates",
                               {"candidates": rules_named, "behaviour_key": key, "version": version.version},
                               next=[nx.next_attach_rule_set(key, version.version)])
        chosen_candidates = cand.resolve(requested, len(members), has_rule_set)
    except cand.CandidateUnknown as exc:
        raise BlendRefused(422, "candidate_unknown", str(exc), {"candidate": exc.name, "vocabulary": exc.vocabulary}) from exc
    max_c = int(_settings.blend_max_candidates)
    if len(chosen_candidates) > max_c:
        raise BlendRefused(422, "cap_exceeded", f"{len(chosen_candidates)} candidates; blend_max_candidates allows {max_c}",
                           {"setting": "blend_max_candidates", "value": max_c, "got": len(chosen_candidates)})
    k = int(body.get("folds") or _settings.blend_folds_default)
    if not (int(_settings.blend_folds_min) <= k <= int(_settings.blend_folds_max)):
        raise BlendRefused(422, "cap_exceeded", f"{k} folds is outside blend_folds_min..blend_folds_max "
                           f"({_settings.blend_folds_min}..{_settings.blend_folds_max})",
                           {"setting": "blend_folds_max" if k > int(_settings.blend_folds_max) else "blend_folds_min",
                            "value": int(_settings.blend_folds_max) if k > int(_settings.blend_folds_max) else int(_settings.blend_folds_min),
                            "got": k})
    group_column = body.get("group_column")
    declared = list(views["blend_fit"].group_columns or [])
    if group_column is None:
        group_column = next((c for c in (version.held_out_columns or []) if c in declared), None)
    elif group_column not in declared:
        raise BlendRefused(422, "column_not_declared", f"column {group_column!r} is not declared on the blend_fit view",
                           {"column": group_column, "view": fit_id, "declared": declared},
                           next=[nx.next_update_columns(fit_id, declared + [group_column])])
    target_fpr = body.get("target_fpr") or version.target_fpr
    if target_fpr is None:
        raise BlendRefused(422, "invalid_field", "no target false-positive rate: name target_fpr, or set one on the behaviour",
                           {"field": "target_fpr", "rule": "required"}, next=[nx.next_list_versions(key)])

    # The blend criteria of the CURRENT version (FR-12): `blend_alone` judged on the probe-only candidates,
    # `rules_or_blend` on the rules ∪ ones.
    criteria = [{"id": c["id"], "detector": c["detector"], "subset": c.get("subset", "all"),
                 "held_out_test": c.get("held_out_test"), "min_recall": c["min_recall"],
                 "target_fpr": c.get("target_fpr") if c.get("target_fpr") is not None else version.target_fpr}
                for c in (version.success_criteria or []) if c.get("detector") in BLEND_FORMS]
    # 040's rule verdicts, BY POSITION, from the cache a completed rules evaluation filled: REQUIRED (refused at the
    # call, PT-5) for a rules ∪ candidate; read when present for a rule subset (else that criterion says so).
    rules = None
    rules_candidates = [c.name for c in chosen_candidates if c.rules]
    if has_rule_set and (rules_candidates or any(_rule_subset(c) for c in criteria)):
        rules = await resolve_rules(db, version, members, view_ids, rules_candidates)
    resolved = {
        "members": [{**{f: m[f] for f in ("probe_id", "run_id", "layer", "rule", "sub_question", "precision",
                                          "render_form", "scope", "rung", "threshold", "detector_digest",
                                          "decision_digest", "training_view")},
                     # The decision IN FORCE at start (039 `current_revision`), which `stale_from` compares.
                     "decision_revision": m["current_decision_revision"],
                     "decision": m["decision"]} for m in members],
        "model_id": model,
        "views": {"blend_fit": fit_id, "tests": [{"id": v.id, "held_out": held_out(v)} for v in views["tests"]],
                  "calibration": {"id": views["calibration"].id, "sample_rows": sample_rows}},
        "evaluations": chosen,
        "candidates": [c.name for c in chosen_candidates],
        "folds": k, "group_column": group_column, "target_fpr": float(target_fpr), "seed": int(body.get("seed") or 0),
        "min_rows_per_class": int(_settings.blend_cv_min_rows_per_class),
        "min_aligned_fraction": float(_settings.blend_min_aligned_fraction),
        "min_subgroup_rows": int(_settings.breakdown_min_subgroup_rows),
        "floor": int(version.calibration_floor or _settings.calibration_floor_negatives),
        "version": {"number": version.version, "rule_set_digest": version.rule_set_digest,
                    "success_criteria_digest": _digest(version.success_criteria),
                    "sub_questions_digest": _digest(version.sub_questions),
                    "held_out_columns": list(version.held_out_columns or [])},
        "criteria": criteria,
        "rules": rules,
        "flags": flags,
        "dataset_version": dataset_version,
    }
    row = ProbeBlendReport(behaviour_key=key, behaviour_version=version.version, model_id=model,
                           dataset_version=dataset_version, status="queued", request=_json(body),
                           resolved=resolved, engine_version=report_format.ENGINE_VERSION, created_by=actor)
    await admit(db)
    db.add(row)
    await db.flush()
    db.add_all([ProbeBlendReportMember(report_id=row.id, probe_id=p) for p in member_ids])
    await db.commit()
    await db.refresh(row)
    _ = static
    return row


BLEND_FORMS = ("blend_alone", "rules_or_blend")


def _rule_subset(criterion: Dict[str, Any]) -> bool:
    from ..detector_success_report import RULE_SUBSETS

    return isinstance(criterion.get("subset"), str) and criterion["subset"] in RULE_SUBSETS


async def resolve_rules(db: AsyncSession, version: ProbeBehaviourVersion, members: Sequence[Dict[str, Any]],
                        view_ids: Sequence[str], rules_candidates: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The version's rule set, the engine its verdicts are cached under, and each view's cache reading — fixed at
    start like every other input. FR-29: the set is validated at this use. A rules ∪ candidate needs operator code
    to have a runner (D23; refused with 040's `runner_unavailable`, probe-only candidates still run) and every code
    rule approved now (040's refusals), and every view's verdicts cached (`rule_verdicts_missing`, naming the rules
    evaluation that caches them). Without a rules ∪ candidate a missing view returns None: the rule subsets then
    read `missing_evaluation` on the criterion."""
    from ...models.rules_evaluation import RuleVerdictCache
    from .. import code_rule_runner, code_rules, rule_set_store
    from .. import rules_evaluation as rules_eval
    from ..detector_success_report import reading_digest

    settings = _effective()
    document, digest = await rules_eval._version_document(db, version)
    limits = rule_set_store.engine_limits(settings)
    rule_set_store.validate_for_use(document, limits=limits)
    shas = code_rules.referenced_hashes(document)
    if shas and rules_candidates:
        if not code_rule_runner.enabled(settings):
            raise BlendRefused(409, "runner_unavailable",
                               f"the rule set of version {version.version} runs operator code and this deployment has "
                               "no code-rule runner, so a rules ∪ candidate would read every such rule as pending; "
                               "ask for probe-only candidates", {"code_sha256": shas, "candidates": list(rules_candidates)})
        await rules_eval._require_code_rules(db, document, rules_eval.runner_identity_for(document, settings), limits)
    engine_digest = rules_eval.engine_for(document, limits, settings)
    scope = str(members[0]["scope"])
    views = {v.id: v for v in (await db.execute(select(ProbeMonitorDataset).where(
        ProbeMonitorDataset.id.in_(list(view_ids))))).scalars()}
    readings: Dict[str, str] = {}
    missing: List[str] = []
    for view_id in view_ids:
        reading = reading_digest(views[view_id], scope)
        cached = (await db.execute(select(RuleVerdictCache).where(
            RuleVerdictCache.view_id == view_id, RuleVerdictCache.rule_set_digest == digest,
            RuleVerdictCache.engine_digest == engine_digest,
            RuleVerdictCache.reading_digest == reading))).scalar_one_or_none()
        if cached is None or not Path(cached.path).is_dir():
            missing.append(view_id)
        readings[view_id] = reading
    if missing and not rules_candidates:
        return None
    if missing:
        raise BlendRefused(409, "rule_verdicts_missing",
                           f"the rule set's verdicts are not cached for {len(missing)} of the views; run the rules "
                           "evaluation below (a CPU task), then ask again — or ask for probe-only candidates",
                           {"views": missing, "rule_set_digest": digest, "engine_digest": engine_digest,
                            "candidates": list(rules_candidates)},
                           next=[nx.next_rules_evaluation(members[0]["probe_id"], missing, version.version)])
    return {"rule_set_digest": digest, "engine_digest": engine_digest, "scope": scope, "readings": readings,
            "unknown_share_max": float(settings.rule_unknown_share_max)}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _json(body: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(body, default=str))


async def admit(db: AsyncSession) -> None:
    """T26: under ONE advisory lock, count the active reports and refuse at `blend_max_concurrent` (429).
    The caller inserts in the same transaction, so two starts cannot both pass the count. On SQLite the lock
    is skipped (the race test runs on Postgres, where it means something — TI10)."""
    from ...core.runtime_settings import effective

    cap = int(effective().blend_max_concurrent)
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        await db.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 41))"), {"k": "probe_blend_reports"})
    active = (await db.execute(select(func.count()).select_from(ProbeBlendReport)
                               .where(ProbeBlendReport.status.in_(ACTIVE)))).scalar_one()
    if active >= cap:
        raise BlendRefused(429, "cap_exceeded",
                           f"{active} blend evaluations are queued or running; blend_max_concurrent allows {cap}",
                           {"setting": "blend_max_concurrent", "value": cap, "active": int(active)})


# ── reading ──────────────────────────────────────────────────────────────────────────────────


async def get_row(db: AsyncSession, report_id: str) -> ProbeBlendReport:
    row = await db.get(ProbeBlendReport, report_id)
    if row is None:
        raise BlendRefused(404, "not_found", f"no blend report {report_id}", {"report_id": report_id})
    return row


async def stale_reasons(db: AsyncSession, row: ProbeBlendReport) -> List[Dict[str, Any]]:
    """T30, computed ON READ: what moved since the report recorded its inputs. A member gone (run deleted),
    retired, moved to another key, or with another decision (digest or revision); the behaviour's current
    version differing in rule set, success criteria or sub-questions. Reads the rows, then the pure
    `stale_from` decides."""
    facts = await stale_facts(db, [row])
    return stale_from(row, facts)


async def stale_facts(db: AsyncSession, rows: Sequence[ProbeBlendReport]) -> Dict[str, Any]:
    """Everything `stale_from` needs for MANY reports, in four statements (the checklist pages call it)."""
    from ...models.probe_behaviour import ProbeBehaviourProbeRole

    ids = sorted({m["probe_id"] for row in rows for m in (row.resolved or {}).get("members", [])}) or [""]
    keys = sorted({row.behaviour_key for row in rows}) or [""]
    probes = {p.id: p for p in (await db.execute(select(ProbeMonitor).where(ProbeMonitor.id.in_(ids)))).scalars()}
    roles = {(pid, key): role for pid, key, role in (await db.execute(
        select(ProbeBehaviourProbeRole.probe_id, ProbeBehaviourProbeRole.key, ProbeBehaviourProbeRole.role)
        .where(ProbeBehaviourProbeRole.probe_id.in_(ids)))).all()}
    heads = {h.key: h for h in (await db.execute(select(ProbeBehaviour).where(ProbeBehaviour.key.in_(keys)))).scalars()}
    pairs = [(k, h.current_version) for k, h in heads.items()]
    versions = {}
    if pairs:
        from sqlalchemy import tuple_

        versions = {v.key: v for v in (await db.execute(select(ProbeBehaviourVersion).where(
            tuple_(ProbeBehaviourVersion.key, ProbeBehaviourVersion.version).in_(pairs)))).scalars()}
    # 040 FR-38: the engine the rules ∪ candidates' verdicts were cached under, as it reads NOW (one statement).
    engines: Dict[str, str] = {}
    digests = sorted({((row.resolved or {}).get("rules") or {}).get("rule_set_digest") for row in rows} - {None})
    if digests:
        from ...models.rule_set import RuleSet
        from .. import rule_set_store
        from ..rules_evaluation import engine_for

        settings = _effective()
        limits = rule_set_store.engine_limits(settings)
        for stored in (await db.execute(select(RuleSet).where(RuleSet.digest.in_(digests)))).scalars():
            try:
                engines[stored.digest] = engine_for(rule_set_store.document_of(stored)[0], limits, settings)
            except DetectorRefused as refusal:      # a code-rule limit below the floor (as 040's success report)
                from ... import rule_signals

                engines[stored.digest] = rule_signals.engine_digest(limits, {"available": False, "refused": refusal.code})
    return {"probes": probes, "roles": roles, "heads": heads, "versions": versions, "engines": engines}


def stale_from(row: ProbeBlendReport, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """PURE: the stale reasons of one report from the rows `stale_facts` read."""
    from ...schemas.probe_identity import current_revision

    resolved = row.resolved or {}
    reasons: List[Dict[str, Any]] = []
    for member in resolved.get("members", []):
        probe_id = member["probe_id"]
        probe = facts["probes"].get(probe_id)
        if probe is None:
            reasons.append({"code": "member_gone", "probe_id": probe_id})
            continue
        if probe.behaviour != row.behaviour_key:
            reasons.append({"code": "member_moved", "probe_id": probe_id, "key": probe.behaviour})
        if facts["roles"].get((probe_id, row.behaviour_key)) == "retired":
            reasons.append({"code": "member_retired", "probe_id": probe_id})
        revision = current_revision(probe.calibration_history)
        if probe.decision_digest != member.get("decision_digest") or revision != member.get("decision_revision"):
            reasons.append({"code": "decision_changed", "probe_id": probe_id,
                            "recorded": {"decision_digest": member.get("decision_digest"),
                                         "decision_revision": member.get("decision_revision")},
                            "current": {"decision_digest": probe.decision_digest, "decision_revision": revision}})
    head = facts["heads"].get(row.behaviour_key)
    if head is not None and head.current_version != row.behaviour_version:
        version = facts["versions"].get(row.behaviour_key)
        was = resolved.get("version") or {}
        moved = []
        if version is not None:
            if version.rule_set_digest != was.get("rule_set_digest"):
                moved.append("rule_set")
            if _digest(version.success_criteria) != was.get("success_criteria_digest"):
                moved.append("success_criteria")
            if _digest(version.sub_questions) != was.get("sub_questions_digest"):
                moved.append("sub_questions")
        if moved:
            reasons.append({"code": "version_changed", "recorded_version": row.behaviour_version,
                            "current_version": head.current_version, "fields": moved})
    rules = resolved.get("rules") or {}
    engine_now = (facts.get("engines") or {}).get(rules.get("rule_set_digest"))
    if engine_now is not None and engine_now != rules.get("engine_digest"):
        reasons.append({"code": "rules_engine_changed", "recorded": rules.get("engine_digest"), "current": engine_now})
    return reasons


async def envelope(db: AsyncSession, row: ProbeBlendReport, *, include_report: bool = True) -> Dict[str, Any]:
    """The report's row as the API serves it, `stale` computed on read (never stored)."""
    reasons = await stale_reasons(db, row)
    out: Dict[str, Any] = {
        "report_id": row.id, "behaviour_key": row.behaviour_key, "behaviour_version": row.behaviour_version,
        "model_id": row.model_id, "dataset_version": row.dataset_version, "status": row.status,
        "failure": ({"code": row.failure_code, "message": row.failure_message} if row.status == "failed" else None),
        "stale": bool(reasons), "stale_reasons": reasons,
        "next": [nx.next_rerun_report(row.id)] if reasons else [],
        "resolved": {k: row.resolved.get(k) for k in ("views", "candidates", "folds", "group_column", "target_fpr",
                                                       "seed", "flags")},
        "members": [m["probe_id"] for m in row.resolved.get("members", [])],
        "report_sha256": row.report_sha256, "report_bytes": row.report_bytes,
        "rows_pruned_at": row.rows_pruned_at.isoformat() if row.rows_pruned_at else None,
        "rerun_of": row.rerun_of, "rerun_by": row.rerun_by,
        "has_rung": False,
        "created_by": row.created_by,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }
    if include_report and row.status == "done" and row.report_path:
        report = json.loads(Path(row.report_path).read_bytes())
        out["ranking"] = report.get("ranking")
        out["report"] = report
    return out


def report_bytes(row: ProbeBlendReport) -> bytes:
    if row.status != "done" or not row.report_path:
        raise BlendRefused(409, "report_not_done", f"blend report {row.id} is {row.status}", {"status": row.status},
                           next=[nx.next_get_report(row.id)])
    return Path(row.report_path).read_bytes()


def candidate_of(report: Dict[str, Any], name: str) -> Dict[str, Any]:
    for entry in report.get("candidates", []):
        if entry["name"] == name:
            return entry
    raise BlendRefused(422, "candidate_unknown", f"{name!r} is not a candidate of this report",
                       {"candidate": name, "vocabulary": [c["name"] for c in report.get("candidates", [])]})


def report_view_ids(report: Dict[str, Any]) -> List[str]:
    views = report.get("views") or {}
    return [views["blend_fit"], *[t["id"] for t in views.get("tests", [])], views["calibration"]["id"]]


def refuse_view_not_in_report(report: Dict[str, Any], view_id: str) -> None:
    """Review R1: a view the report does not have is the CALLER's mistake (422), not pruned rows (409) — and is
    refused before a file name is built from the caller's string."""
    views = report_view_ids(report)
    if view_id not in views:
        raise BlendRefused(422, "view_not_in_report", f"view {view_id!r} is not one of report "
                           f"{report.get('report_id')}'s views", {"view_id": view_id, "views": views})


def candidate_rows(row: ProbeBlendReport, candidate: Dict[str, Any], view_id: str) -> Dict[str, np.ndarray]:
    """One candidate's per-row values on one view, its positions and labels, from the report's files."""
    if row.rows_pruned_at is not None:
        raise BlendRefused(409, "rows_pruned", f"blend report {row.id}'s per-row values were pruned",
                           {"rows_pruned_at": row.rows_pruned_at.isoformat()})
    directory = Path(row.rows_dir)
    # Verdict kinds share one `nfired` array; a rules ∪ candidate's final verdict has its own `c<NN>` file.
    values_file = (f"nfired__{view_id}.npy" if candidate["produces"] == "verdict"
                   and not cand.is_rules_name(candidate["name"]) else f"c{candidate['index']:02d}__{view_id}.npy")
    try:
        return {"positions": np.load(directory / f"positions__{view_id}.npy", mmap_mode="r"),
                "labels": np.load(directory / f"labels__{view_id}.npy", mmap_mode="r"),
                "values": np.load(directory / values_file, mmap_mode="r"),
                # The version's rule match on the aligned rows, when the report read rule verdicts.
                "rule_match": (np.load(directory / f"rules__{view_id}.npy", mmap_mode="r").astype(bool)
                               if (directory / f"rules__{view_id}.npy").exists() else None)}
    except FileNotFoundError as exc:
        raise BlendRefused(409, "rows_pruned", f"blend report {row.id} has no per-row values for view {view_id}",
                           {"view_id": view_id}) from exc


# ── suggested blend ──────────────────────────────────────────────────────────────────────────


def suggested_record(report: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """The record: the candidate SLICED out of the stored report — no recomputation (FR-21)."""
    return {
        "format": report_format.SUGGESTED_FORMAT,
        "report_id": report["report_id"],
        "behaviour": report["behaviour"],
        "model_id": report["model_id"],
        "members": [{"probe_id": m["probe_id"], "name": m.get("name"), "detector_digest": m["detector_digest"],
                     "decision_digest": m["decision_digest"], "decision_revision": m["decision_revision"]}
                    for m in report["members"]],
        "candidate": {"name": candidate["name"], "kind": candidate["kind"], "params": candidate["params"],
                      "calibration": candidate["calibration"], "tests": {v: {k: f[k] for k in f if k != "breakdown"}
                                                                         for v, f in candidate["tests"].items()}},
        "target_fpr": report["behaviour"]["target_fpr"],
        "advisory": True,
        "statement": report_format.STATEMENT,
    }


async def mark_suggested(db: AsyncSession, key: str, model: str, report_id: str, candidate: str, actor: str,
                         reason: Optional[str]) -> Tuple[bytes, str]:
    from .. import behaviour_catalogue as catalogue

    await catalogue.get_head(db, key)
    row = await get_row(db, report_id)
    if row.behaviour_key != key or row.model_id != model:
        raise BlendRefused(422, "report_not_of_behaviour", f"report {report_id} is of {row.behaviour_key!r} on "
                           f"{row.model_id!r}, not {key!r} on {model!r}",
                           {"report_behaviour": row.behaviour_key, "report_model": row.model_id})
    if row.status != "done":
        raise BlendRefused(409, "report_not_done", f"report {report_id} is {row.status}", {"status": row.status},
                           next=[nx.next_get_report(report_id)])
    reasons = await stale_reasons(db, row)
    if reasons:
        raise BlendRefused(409, "blend_report_stale", f"report {report_id} is stale", {"stale_reasons": reasons},
                           next=[nx.next_rerun_report(report_id)])
    report = json.loads(report_bytes(row))
    entry = candidate_of(report, candidate)
    record = report_format.canonical_bytes(suggested_record(report, entry))
    digest = hashlib.sha256(record).hexdigest()
    history = ProbeSuggestedBlendHistory(behaviour_key=key, model_id=model, report_id=report_id, candidate=candidate,
                                         record=record, record_sha256=digest, actor=actor, reason=reason)
    db.add(history)
    await db.flush()
    current = await db.get(ProbeSuggestedBlend, (key, model))
    if current is None:
        db.add(ProbeSuggestedBlend(behaviour_key=key, model_id=model, record=record, record_sha256=digest,
                                   report_id=report_id, candidate=candidate, history_id=history.id, updated_at=_now()))
    else:
        current.record, current.record_sha256, current.report_id = record, digest, report_id
        current.candidate, current.history_id, current.updated_at = candidate, history.id, _now()
    catalogue._history(db, key=key, subject_kind="blend", subject_id=model[:64], field="suggested", old=None,
                       new={"report_id": report_id, "candidate": candidate, "record_sha256": digest},
                       actor=actor, reason=reason)
    await db.commit()
    return record, history.id


async def suggested(db: AsyncSession, key: str, model: str) -> Tuple[ProbeSuggestedBlend, List[Dict[str, Any]]]:
    current = await db.get(ProbeSuggestedBlend, (key, model))
    if current is None:
        raise BlendRefused(404, "no_suggested_blend", f"no suggested blend for {key!r} on {model!r}",
                           {"key": key, "model": model}, next=[nx.next_list_reports(key, model)])
    row = await db.get(ProbeBlendReport, current.report_id)
    reasons = await stale_reasons(db, row) if row is not None else [{"code": "report_gone"}]
    return current, reasons


async def suggested_history(db: AsyncSession, key: str, model: str, offset: int, limit: int) -> Dict[str, Any]:
    rows = (await db.execute(
        select(ProbeSuggestedBlendHistory).where(ProbeSuggestedBlendHistory.behaviour_key == key,
                                                 ProbeSuggestedBlendHistory.model_id == model)
        .order_by(ProbeSuggestedBlendHistory.created_at.desc(), ProbeSuggestedBlendHistory.id.desc())
        .offset(offset).limit(limit + 1))).scalars().all()
    page = rows[:limit]
    return {"key": key, "model": model, "entries": [
        {"id": h.id, "report_id": h.report_id, "candidate": h.candidate, "record_sha256": h.record_sha256,
         "actor": h.actor, "reason": h.reason, "created_at": h.created_at.isoformat() if h.created_at else None}
        for h in page], "offset": offset, "limit": limit, "next_offset": offset + limit if len(rows) > limit else None}


# ── prune, rerun, cancel ─────────────────────────────────────────────────────────────────────


async def prune_rows(db: AsyncSession, report_id: str, actor: str) -> ProbeBlendReport:
    """FR-19: delete the per-row `.npy` files, keep `report.json`; refused behind the current suggestion."""
    row = await get_row(db, report_id)
    in_use = (await db.execute(select(ProbeSuggestedBlend).where(ProbeSuggestedBlend.report_id == report_id))).scalars().first()
    if in_use is not None:
        raise BlendRefused(409, "report_in_use", f"report {report_id} backs the current suggested blend of "
                           f"{in_use.behaviour_key!r} on {in_use.model_id!r}",
                           {"behaviour_key": in_use.behaviour_key, "model_id": in_use.model_id})
    if row.rows_pruned_at is None and row.rows_dir:
        for path in Path(row.rows_dir).glob("*.npy"):
            path.unlink(missing_ok=True)
    if row.rows_pruned_at is None:
        row.rows_pruned_at, row.rows_pruned_by = _now(), actor
        await db.commit()
    return row


async def rerun(db: AsyncSession, report_id: str, actor: str) -> ProbeBlendReport:
    """FR-17 / P4: the same request against today's members and version; linked both ways."""
    old = await get_row(db, report_id)
    request = dict(old.request or {})
    named = request.get("member_ids")
    if named:
        from ...models.probe_behaviour import ProbeBehaviourProbeRole

        present = {p for p in (await db.execute(select(ProbeMonitor.id).where(ProbeMonitor.id.in_(named)))).scalars()}
        retired = {p for p in (await db.execute(select(ProbeBehaviourProbeRole.probe_id).where(
            ProbeBehaviourProbeRole.probe_id.in_(named), ProbeBehaviourProbeRole.role == "retired"))).scalars()}
        gone = [p for p in named if p not in present or p in retired]
        if gone:
            raise BlendRefused(409, "member_retired_or_deleted", f"the report named members that are now retired or "
                               f"deleted: {gone}", {"members": gone},
                               next=[nx.next_evaluate_blends(old.behaviour_key, old.model_id, None)])
    new = await start(db, request, actor)
    new.rerun_of = old.id
    old.rerun_by = new.id
    await db.commit()
    return new


def remove_staging(report_id: str) -> None:
    shutil.rmtree(blends_root() / f"{report_id}.staging", ignore_errors=True)
