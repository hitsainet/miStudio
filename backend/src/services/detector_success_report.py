"""Does this behaviour's detector meet its own bar? (Feature 040 FR-19, FR-22 – FR-26; FTID §3.5).

One request judges every `success_criteria` entry of ONE behaviour version, for ONE model and ONE dataset
identity, at the decision a deployment applies, and stores the answer as an immutable report.

Order of work: version (default current) → model (TI-12: the only model with a probe of the behaviour,
else `ambiguous_model`) → dataset identity (040-T20: a manifest's `version_id` + canonical digest, else a
`view_set` of sorted view ids, row counts and snapshot digests) → probe (040-T21: the behaviour's primary
when it matches; else the only matching run's selected probe; else `ambiguous_probe`) → per criterion:
test views, the held-out overlap check (FR-23), figures at the RECORDED decision, a verdict → the overall
verdict (FR-24) → identity → `INSERT … ON CONFLICT (identity_sha256) DO NOTHING` → read back.

⚠ EVERY VALUE THE VERDICT TURNS ON IS READ FROM THE CRITERION: `min_recall`, the target FPR (the
criterion's, else the version's), `held_out_test`, the subset. Nothing here names a behaviour or a column.

Verdict reasons are DATA (FR-40): `few_negatives`, `held_out_overlap`, `held_out_unchecked`,
`target_fpr_mismatch`, `missing_evaluation`, `no_blend_report`, `engine_not_available`,
`in_distribution` (a label, not a failure). Before S3 every `rules_*` form and `rule_*` subset reads
`not_assessable` / `engine_not_available`; `blend_alone` reads `no_blend_report` until 041.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from ..models.detector_report import DetectorSuccessReport
from ..models.probe_behaviour import ProbeBehaviour, ProbeBehaviourVersion
from ..models.probe_monitor import (
    ProbeMonitor,
    ProbeMonitorDataset,
    ProbeMonitorEvaluation,
    ProbeMonitorRun,
)
from . import behaviour_catalogue as catalogue
from . import view_columns
from .behaviour_detector_definition import effective_target_fpr
from .calibration_size import calibration_size
from .detector_identity import REPORT_ALGORITHM, identity, jcs_bytes, round6, sha256_hex
from .detector_next import (
    DetectorRefused,
    next_evaluate,
    next_recalibrate_preview,
    next_success_report,
    next_update_columns,
)
from .probe_breakdown import evaluation_window, recorded_decisions
from .probe_monitor_metrics import wilson_interval

RULE_FORMS = {"rules_or_blend", "rules_alone"}
RULE_SUBSETS = {"rule_invisible_positives", "rule_visible_positives"}
VERDICTS = ("met", "not_met", "not_assessable", "no_bar_set")


# ── resolution ────────────────────────────────────────────────────────────────────────────────


async def resolve_version(db: Any, key: str, number: Optional[int]) -> Tuple[ProbeBehaviour, ProbeBehaviourVersion]:
    head = await catalogue.get_head(db, key)
    wanted = number if number is not None else head.current_version
    version = (await db.execute(select(ProbeBehaviourVersion).where(
        ProbeBehaviourVersion.key == key, ProbeBehaviourVersion.version == wanted))).scalar_one_or_none()
    if version is None:
        raise DetectorRefused(404, "version_not_found", f"{key!r} has no version {wanted}",
                              {"version": wanted, "current_version": head.current_version})
    return head, version


async def _probes_of(db: Any, key: str) -> List[Tuple[ProbeMonitor, ProbeMonitorRun]]:
    rows = (await db.execute(
        select(ProbeMonitor, ProbeMonitorRun)
        .join(ProbeMonitorRun, ProbeMonitorRun.id == ProbeMonitor.run_id)
        .where(ProbeMonitor.behaviour == key)
        .order_by(ProbeMonitor.created_at, ProbeMonitor.id)
    )).all()
    return [(p, r) for p, r in rows]


async def resolve_model(db: Any, key: str, model_id: Optional[str]) -> str:
    """TI-12: the named model; else the only model with a probe of the behaviour; else refused."""
    if model_id:
        return model_id
    models = sorted({run.model_id for _p, run in await _probes_of(db, key)})
    if len(models) == 1:
        return models[0]
    raise DetectorRefused(
        409 if models else 404, "ambiguous_model" if models else "no_probe",
        f"{len(models)} models have a probe of {key!r}; name one" if models
        else f"no probe of {key!r} exists yet; train one first",
        {"candidates": models},
        next=[next_success_report(key, model_id=m) for m in models],
    )


async def _views(db: Any, ids: Sequence[str]) -> List[ProbeMonitorDataset]:
    if not ids:
        return []
    rows = (await db.execute(select(ProbeMonitorDataset).where(ProbeMonitorDataset.id.in_(list(ids))))).scalars().all()
    by_id = {v.id: v for v in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise DetectorRefused(404, "view_not_found", f"view(s) {missing} not found", {"view_ids": missing})
    return [by_id[i] for i in ids]


def view_set_identity(views: Sequence[ProbeMonitorDataset]) -> Dict[str, Any]:
    return {"kind": "view_set", "views": [
        {"id": v.id, "rows": int((v.counts or {}).get("total") or sum(int((v.counts or {}).get(k) or 0)
                                  for k in ("positive", "negative", "excluded"))),
         "snapshots": {c: e["sha256"] for c, e in sorted((v.column_snapshots or {}).items())}}
        for v in sorted(views, key=lambda v: v.id)]}


async def resolve_identity(db: Any, dataset_version: Optional[str], view_ids: Optional[Sequence[str]],
                           default_views: Sequence[str]) -> Tuple[Dict[str, Any], List[ProbeMonitorDataset]]:
    """040-T20. A manifest version (every view registered with that `version_id`), explicit views, or the
    judged probe's own evaluation views."""
    if dataset_version:
        rows = (await db.execute(select(ProbeMonitorDataset))).scalars().all()
        views = [v for v in rows if (v.dataset_version or {}).get("version_id") == dataset_version]
        if not views:
            raise DetectorRefused(404, "dataset_version_unknown",
                                  f"no registered view comes from dataset version {dataset_version}",
                                  {"dataset_version": dataset_version})
        digests = sorted({(v.dataset_version or {}).get("manifest_sha256") for v in views} - {None})
        return {"kind": "manifest", "version_id": dataset_version, "manifest_sha256": digests}, views
    views = await _views(db, list(dict.fromkeys(view_ids or default_views)))
    return view_set_identity(views), views


def _training_identity(view: Optional[ProbeMonitorDataset]) -> Optional[str]:
    return (getattr(view, "dataset_version", None) or {}).get("version_id") if view is not None else None


async def resolve_probe(db: Any, key: str, model_id: str, probe_id: Optional[str],
                        dataset_version: Optional[str]) -> Tuple[ProbeMonitor, ProbeMonitorRun, str]:
    """040-T21. Returns (probe, run, how it was chosen)."""
    pairs = [(p, r) for p, r in await _probes_of(db, key) if r.model_id == model_id]
    if probe_id:
        for p, r in pairs:
            if p.id == probe_id:
                return p, r, "named"
        raise DetectorRefused(404, "probe_not_found",
                              f"probe {probe_id} is not a probe of {key!r} on model {model_id}",
                              {"probe_id": probe_id})

    async def matches(run: ProbeMonitorRun) -> bool:
        if not dataset_version:
            return True
        view = (await db.execute(select(ProbeMonitorDataset).where(
            ProbeMonitorDataset.id == run.train_dataset_id))).scalar_one_or_none()
        return _training_identity(view) == dataset_version

    primary = await catalogue.primary_probe(db, key)
    for p, r in pairs:
        if primary is not None and p.id == primary.id and await matches(r):
            return p, r, "primary"
    runs = {}
    for p, r in pairs:
        if r.status == "completed" and await matches(r):
            runs.setdefault(r.id, (r, []))[1].append(p)
    selected = [(next((p for p in ps if p.selected), None), r) for r, ps in runs.values()]
    selected = [(p, r) for p, r in selected if p is not None]
    if len(selected) == 1:
        return selected[0][0], selected[0][1], "only_run_selected"
    raise DetectorRefused(
        409 if selected else 404, "ambiguous_probe" if selected else "no_probe",
        f"{len(selected)} runs of {key!r} on {model_id} each select a probe; name one (or make one primary)"
        if selected else f"no completed run of {key!r} on {model_id} selects a probe",
        {"candidates": [p.id for p, _ in selected]},
        next=[next_success_report(key, model_id=model_id, probe_id=p.id) for p, _ in selected],
    )


# ── per-criterion figures ─────────────────────────────────────────────────────────────────────


async def _evaluation(db: Any, probe_id: str, view_id: str) -> Optional[ProbeMonitorEvaluation]:
    rows = (await db.execute(select(ProbeMonitorEvaluation).where(
        ProbeMonitorEvaluation.probe_id == probe_id, ProbeMonitorEvaluation.dataset_id == view_id,
        ProbeMonitorEvaluation.positions_path.isnot(None),
    ).order_by(ProbeMonitorEvaluation.created_at.desc(), ProbeMonitorEvaluation.id.desc()))).scalars().all()
    return rows[0] if rows else None


def _figures(probe: ProbeMonitor, run: ProbeMonitorRun, evaluation: ProbeMonitorEvaluation, view: ProbeMonitorDataset,
             subset: Any, root: Path) -> Dict[str, Any]:
    """Counts at the recorded decision for one evaluation, optionally on a `{column, value}` subset."""
    from .probe_evaluation_scores import load_evaluation_arrays
    from .probe_breakdown import _value_key

    loaded = load_evaluation_arrays(evaluation, root=root)
    scores = np.asarray(loaded.arrays["score"], dtype=np.float64)
    positions = np.asarray(loaded.arrays["position"], dtype=np.int64)
    labels = np.asarray(loaded.arrays["label"])
    run_scope = (run.config or {}).get("scope", "all")
    window = evaluation_window(run_scope)
    decisions = recorded_decisions(probe, run_scope, loaded.arrays["n_scored"])
    fires = decisions.fires(scores)
    mask = np.ones(scores.shape[0], dtype=bool)
    if isinstance(subset, dict):
        snap = view_columns.load(view.column_snapshots[subset["column"]], subset["column"])
        wanted = [i for i, v in enumerate(snap.values) if _value_key(v) == _value_key(subset["value"])]
        mask &= np.isin(np.asarray(snap.codes)[positions], wanted)
    pos = (labels == 1) & mask
    neg = (labels == 0) & mask
    return {"positives": int(pos.sum()), "flagged_positives": int((pos & fires).sum()),
            "negatives": int(neg.sum()), "flagged_negatives": int((neg & fires).sum()),
            "rule_counts": decisions.rule_counts(), "window": window}


def overlap(criterion_column_views: Sequence[ProbeMonitorDataset], training_views: Sequence[ProbeMonitorDataset],
            columns: Sequence[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """FR-23: (first overlap found, first column that could not be checked). Values are compared from
    the snapshots of each DECLARED held-out column — never a fixed column name."""
    for column in columns:
        for test in criterion_column_views:
            for train in training_views:
                if column not in (test.column_snapshots or {}) or column not in (train.column_snapshots or {}):
                    missing = test if column not in (test.column_snapshots or {}) else train
                    return None, {"column": column, "view_id": missing.id}
                test_values = set(json.dumps(v) for v in view_columns.load(test.column_snapshots[column], column).values)
                train_values = set(json.dumps(v) for v in view_columns.load(train.column_snapshots[column], column).values)
                shared = sorted(test_values & train_values)
                if shared:
                    return {"column": column, "value": json.loads(shared[0]), "test_view_id": test.id,
                            "training_view_id": train.id}, None
    return None, None


def overall(verdicts: Sequence[str]) -> str:
    """FR-24's truth table."""
    if not verdicts:
        return "no_bar_set"
    if all(v == "met" for v in verdicts):
        return "met"
    if "not_met" in verdicts:
        return "not_met"
    return "not_assessable"


# ── building one report ───────────────────────────────────────────────────────────────────────


def _reason(code: str, **details: Any) -> Dict[str, Any]:
    return {"code": code, **details}


async def _judge_criterion(db: Any, criterion: Dict[str, Any], version: ProbeBehaviourVersion,
                           probe: ProbeMonitor, run: ProbeMonitorRun, views: Sequence[ProbeMonitorDataset],
                           floor_setting: int, root: Path, cited: Dict[str, List[str]]) -> Dict[str, Any]:
    form, subset = criterion["detector"], criterion["subset"]
    target = effective_target_fpr(criterion, version.target_fpr)
    min_recall = float(criterion["min_recall"])
    held = criterion.get("held_out_test")
    entry: Dict[str, Any] = {
        "id": criterion["id"], "detector": form, "subset": subset, "held_out_test": held,
        "in_distribution": held is None, "target_fpr": target, "min_recall": min_recall,
        "verdict": None, "reasons": [], "next": [],
    }
    # A blend is 041's, the rule engine S3's: until then these read not assessable, saying which —
    # `no_blend_report` first for a blend criterion (FTDD §15: the pilot's criterion (ii)).
    pending = []
    if form == "blend_alone":
        # Feature 041 FR-20: judged on the named (or suggested) blend candidate by the SAME function the blend
        # report used; without one it reads `no_blend_report` (and `blend_report_stale` when the suggestion's
        # report moved).
        blend = cited.get("blend_context")
        if blend and not blend.get("stale"):
            entry.update(blend_figures(blend, criterion, version))
            return entry
        pending.append(_reason("blend_report_stale", report_id=blend["report_id"]) if blend
                       else _reason("no_blend_report"))
        if blend:
            from .detector_next import next_rerun_report

            entry["next"] = [next_rerun_report(blend["report_id"])]
    if form in RULE_FORMS or (isinstance(subset, str) and subset in RULE_SUBSETS):
        pending.append(_reason("engine_not_available"))
    if pending:
        entry.update(verdict="not_assessable", reasons=pending)
        return entry

    judged, judged_run = probe, run
    if form == "named_probe":
        pair = (await db.execute(select(ProbeMonitor, ProbeMonitorRun)
                                 .join(ProbeMonitorRun, ProbeMonitorRun.id == ProbeMonitor.run_id)
                                 .where(ProbeMonitor.id == criterion["probe_id"]))).first()
        if pair is None:
            entry.update(verdict="not_assessable", reasons=[_reason("missing_probe", probe_id=criterion["probe_id"])])
            return entry
        judged, judged_run = pair
    entry["probe_id"] = judged.id
    cited["probes"].append(judged.id)

    # Test views: held-out criteria are judged on OUT-OF-DISTRIBUTION evaluation views only (FR-23);
    # a criterion without `held_out_test` on the in-distribution ones, labelled so (FR-5).
    if held:
        tests = [v for v in views if v.role == "eval" and v.distribution == "out_of_distribution"]
    else:
        tests = [v for v in views if v.role == "eval" and v.distribution != "out_of_distribution"]
    entry["test_view_ids"] = [v.id for v in tests]
    reasons: List[Dict[str, Any]] = []
    nexts: List[Any] = []
    if not tests:
        reasons.append(_reason("missing_evaluation", missing="test_view",
                               wanted="out_of_distribution" if held else "in_distribution"))
    if held and tests:
        training = [v for v in await _views(db, [judged_run.train_dataset_id])]
        found, unchecked = overlap(tests, training, list(version.held_out_columns or []))
        if found:
            reasons.append(_reason("held_out_overlap", **found))
        elif unchecked:
            reasons.append(_reason("held_out_unchecked", **unchecked))
            view = next(v for v in [*tests, *training] if v.id == unchecked["view_id"])
            nexts.append(next_update_columns(view.id, list(view.group_columns or []) + [unchecked["column"]]))
    if isinstance(subset, dict):
        for view in tests:
            if subset["column"] not in (view.column_snapshots or {}):
                reasons.append(_reason("column_not_declared", column=subset["column"], view_id=view.id))
                nexts.append(next_update_columns(view.id, list(view.group_columns or []) + [subset["column"]]))
    totals = {"positives": 0, "flagged_positives": 0, "negatives": 0, "flagged_negatives": 0}
    rule_counts: Dict[str, int] = {}
    evaluated: List[str] = []
    for view in tests:
        evaluation = await _evaluation(db, judged.id, view.id)
        if evaluation is None:
            reasons.append(_reason("missing_evaluation", view_id=view.id))
            nexts.append(next_evaluate(judged.id, view.id))
            continue
        if any(r["code"] == "column_not_declared" and r.get("view_id") == view.id for r in reasons):
            continue
        figures = _figures(judged, judged_run, evaluation, view, subset, root)
        evaluated.append(evaluation.id)
        for k in totals:
            totals[k] += figures[k]
        for rule, n in figures["rule_counts"].items():
            rule_counts[rule] = rule_counts.get(rule, 0) + n
    cited["evaluations"] += evaluated
    entry["evaluation_ids"] = evaluated
    p, hits = totals["positives"], totals["flagged_positives"]
    recall = hits / p if p else None
    lo, hi = wilson_interval(hits, p) if p else (None, None)
    n, fp = totals["negatives"], totals["flagged_negatives"]
    entry["figures"] = {
        **totals, "recall": round6(recall) if recall is not None else None,
        "recall_ci": [round6(lo), round6(hi)] if p else None,
        "test_fpr": round6(fp / n) if n else None, "rule_counts": rule_counts,
    }
    size = calibration_size(judged, version.calibration_floor, floor_setting)
    entry["calibration"] = {**size, "decision_revision": _revision(judged), "realised_fpr": judged.realised_fpr}
    if size["few_negatives"]:
        reasons.append(_reason("few_negatives", n_negatives=size["n_negatives"], floor=size["floor"],
                               floor_source=size["floor_source"]))
    if target is not None and (judged.target_fpr is None or abs(float(judged.target_fpr) - float(target)) > 1e-12):
        reasons.append(_reason("target_fpr_mismatch", criterion_target_fpr=target, probe_target_fpr=judged.target_fpr))
        nexts.append(next_recalibrate_preview(judged.id, float(target)))
    if p == 0 and tests and not any(r["code"] == "missing_evaluation" for r in reasons):
        reasons.append(_reason("missing_evaluation", missing="positives"))
    if reasons:
        verdict = "not_assessable"
    else:
        # ⚠ THE POINT ESTIMATE DECIDES (FR-22); the lower bound is reported beside it, never instead.
        verdict = "met" if recall >= min_recall else "not_met"
    entry["met_at_lower_bound"] = (lo >= min_recall) if (lo is not None and not reasons) else None
    entry.update(verdict=verdict, reasons=reasons, next=nexts)
    return entry


async def blend_context(db: Any, key: str, model_id: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Feature 041 FR-20: the blend candidate `blend_alone` criteria are judged on — named in the request, else
    the model's current suggested blend, else none. A NAMED stale report is refused (409
    `blend_report_stale`); a stale SUGGESTION reads not assessable on the criterion, naming it."""
    import json as _json

    from ..models.probe_blend import ProbeSuggestedBlend
    from .probe_blend import service as blends

    report_id, candidate = request.get("blend_report_id"), request.get("candidate")
    if bool(report_id) != bool(candidate):
        raise DetectorRefused(422, "invalid_field", "name both blend_report_id and candidate, or neither",
                              {"field": "candidate" if report_id else "blend_report_id", "rule": "both_or_neither"})
    named = bool(report_id)
    if not named:
        current = await db.get(ProbeSuggestedBlend, (key, model_id))
        if current is None:
            return None
        report_id, candidate = current.report_id, current.candidate
    row = await blends.get_row(db, report_id)
    if row.behaviour_key != key or row.model_id != model_id:
        raise DetectorRefused(422, "report_not_of_behaviour", f"report {report_id} is not of {key!r} on {model_id!r}",
                              {"report_behaviour": row.behaviour_key, "report_model": row.model_id})
    report = _json.loads(blends.report_bytes(row))
    entry = blends.candidate_of(report, candidate)
    reasons = await blends.stale_reasons(db, row)
    if reasons and named:
        from .detector_next import next_rerun_report

        raise DetectorRefused(409, "blend_report_stale", f"blend report {report_id} is stale",
                              {"stale_reasons": reasons}, next=[next_rerun_report(report_id)])
    return {"report_id": report_id, "candidate": candidate, "report_sha256": row.report_sha256, "row": row,
            "report": report, "entry": entry, "stale": bool(reasons)}


def blend_figures(blend: Dict[str, Any], criterion: Dict[str, Any], version: ProbeBehaviourVersion) -> Dict[str, Any]:
    """One `blend_alone` criterion on the blend candidate, from the report's FILES through the engine's
    `blend_criterion_verdict` — the function the report itself called for its `criteria`."""
    from types import SimpleNamespace

    from .probe_blend import candidates as cand
    from .probe_blend import engine
    from .probe_blend import service as blends

    report, entry, row = blend["report"], blend["entry"], blend["row"]
    tests = report["views"]["tests"]
    values: Dict[str, Any] = {}
    aligned: Dict[str, Any] = {}
    for test in tests:
        loaded = blends.candidate_rows(row, entry, test["id"])
        values[test["id"]] = np.asarray(loaded["values"], dtype=np.float64)
        aligned[test["id"]] = SimpleNamespace(positions=np.asarray(loaded["positions"]),
                                              labels=np.asarray(loaded["labels"]))
    criterion_in = {**criterion, "target_fpr": effective_target_fpr(criterion, version.target_fpr)}
    candidate = cand.parse_name(entry["name"], len(report["members"]), False)
    judged = engine.blend_criterion_verdict(criterion_in, candidate, tests, blend.get("snapshots") or {}, values,
                                            aligned, entry["calibration"])
    return {"verdict": judged["verdict"], "reasons": judged["reasons"], "figures": judged["figures"],
            "test_view_ids": judged["test_view_ids"],
            "blend": {"report_id": blend["report_id"], "candidate": blend["candidate"],
                      "report_sha256": blend["report_sha256"]}}


def _revision(probe: ProbeMonitor) -> int:
    from ..schemas.probe_identity import current_revision

    return current_revision(getattr(probe, "calibration_history", None))


async def build(db: Any, key: str, request: Dict[str, Any], actor: str, *, root: Path,
                settings: Any = None) -> Dict[str, Any]:
    """Compute, store (one row per identity) and return the envelope `{id, …, report, stale}`."""
    if settings is None:
        from ..core.runtime_settings import effective

        settings = effective()
    head, version = await resolve_version(db, key, request.get("version"))
    model_id = await resolve_model(db, key, request.get("model_id"))
    probe, run, chosen_by = await resolve_probe(db, key, model_id, request.get("probe_id"), request.get("dataset_version"))
    default_views = list(dict.fromkeys([*(run.eval_dataset_ids or [])]))
    dataset_identity, views = await resolve_identity(db, request.get("dataset_version"), request.get("view_ids"),
                                                     default_views)
    if request.get("dataset_version"):
        trained_on = _training_identity((await _views(db, [run.train_dataset_id]))[0])
        if trained_on != request["dataset_version"]:
            raise DetectorRefused(
                409, "dataset_identity_mismatch",
                f"probe {probe.id} was trained on dataset version {trained_on}, not {request['dataset_version']}",
                {"difference": {"probe_trained_on": trained_on, "requested": request["dataset_version"]}})
    floor_setting = int(settings.calibration_floor_negatives)
    cited: Dict[str, Any] = {"probes": [probe.id], "evaluations": [], "views": [v.id for v in views]}
    blend = await blend_context(db, key, model_id, request) if any(
        c.get("detector") == "blend_alone" for c in (version.success_criteria or [])) or request.get("blend_report_id") \
        else None
    if blend is not None:
        blend["snapshots"] = await _blend_snapshots(db, blend["report"])
        cited["blend_context"] = blend
    criteria = [await _judge_criterion(db, c, version, probe, run, views, floor_setting, root, cited)
                for c in (version.success_criteria or [])]
    from .probe_decision import identity_fields

    ident = identity_fields(probe)
    body = {
        "format": "mistudio.detector-success-report/1",
        "behaviour_key": key,
        "behaviour_version": version.version,
        "model_id": model_id,
        "dataset_identity": dataset_identity,
        "probe": {"id": probe.id, "chosen_by": chosen_by, "run_id": run.id, "decision_revision": _revision(probe),
                  "detector_digest": ident.get("detector_digest"), "decision_digest": ident.get("decision_digest")},
        "criteria": criteria,
        "overall_verdict": overall([c["verdict"] for c in criteria]),
        "rule_set_digest": version.rule_set_digest,
        "report_algorithm": REPORT_ALGORITHM,
        "settings": {"calibration_floor_negatives": floor_setting},
    }
    if not criteria:
        # FR-22: no bar → figures shown for the selected probe on the identity's views, no verdict invented.
        body["figures"] = []
        for view in views:
            evaluation = await _evaluation(db, probe.id, view.id)
            if evaluation is not None and view.role == "eval":
                body["figures"].append({"view_id": view.id, "evaluation_id": evaluation.id,
                                        "distribution": view.distribution,
                                        **{k: v for k, v in _figures(probe, run, evaluation, view, "all", root).items()}})
                cited["evaluations"].append(evaluation.id)
    inputs = {"report_algorithm": REPORT_ALGORITHM, "behaviour_key": key, "behaviour_version": version.version,
              "model_id": model_id, "dataset_identity": dataset_identity, "probe": body["probe"],
              "cited": {k: sorted(set(v)) for k, v in cited.items() if k != "blend_context"},
              "rule_set_digest": version.rule_set_digest,
              # Feature 041 FR-20: a different blend (or its bytes) is a different report.
              "blend": ({k: blend[k] for k in ("report_id", "candidate", "report_sha256")} if blend else None),
              "settings": body["settings"], "request_probe": request.get("probe_id")}
    digest = identity(inputs)
    raw = jcs_bytes(body)
    cap = int(settings.success_report_max_bytes)
    if len(raw) > cap:
        raise DetectorRefused(422, "over_cap", f"the report is {len(raw)} bytes; success_report_max_bytes allows {cap}",
                              {"setting": "success_report_max_bytes", "value": cap, "got": len(raw)})
    await db.execute(insert(DetectorSuccessReport).values(
        identity_sha256=digest, behaviour_key=key, behaviour_version=version.version, model_id=model_id,
        dataset_identity=dataset_identity, report_algorithm=REPORT_ALGORITHM, inputs=inputs,
        body=raw.decode("utf-8"), body_sha256=sha256_hex(raw), overall_verdict=body["overall_verdict"],
        created_by=actor, id=f"dsr_{digest[:12]}",
    ).on_conflict_do_nothing(index_elements=["identity_sha256"]))
    await db.commit()
    row = (await db.execute(select(DetectorSuccessReport).where(
        DetectorSuccessReport.identity_sha256 == digest))).scalar_one()
    return await envelope(db, row)


async def _blend_snapshots(db: Any, report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """`{test view: {column: (codes by position, values)}}` for the subsets a blend criterion names."""
    ids = [t["id"] for t in report["views"]["tests"]]
    out: Dict[str, Dict[str, Any]] = {}
    for view in await _views(db, ids):
        out[view.id] = {}
        for column, entry in (view.column_snapshots or {}).items():
            snap = view_columns.load(entry, column)
            out[view.id][column] = (np.asarray(snap.codes), list(snap.values))
    return out


async def _missing(db: Any, model: Any, ids: Sequence[str]) -> List[str]:
    if not ids:
        return []
    present = set((await db.execute(select(model.id).where(model.id.in_(list(ids))))).scalars().all())
    return [i for i in ids if i not in present]


async def staleness(db: Any, row: DetectorSuccessReport) -> Dict[str, Any]:
    """040-T19: computed on READ, never written back. Names each input that moved and the rerun."""
    moved: List[Dict[str, Any]] = []
    if row.report_algorithm != REPORT_ALGORITHM:
        moved.append({"input": "report_algorithm", "was": row.report_algorithm, "now": REPORT_ALGORITHM})
    head = (await db.execute(select(ProbeBehaviour).where(ProbeBehaviour.key == row.behaviour_key))).scalar_one_or_none()
    if head is not None and head.current_version != row.behaviour_version:
        moved.append({"input": "behaviour_version", "was": row.behaviour_version, "now": head.current_version})
        current = (await db.execute(select(ProbeBehaviourVersion).where(
            ProbeBehaviourVersion.key == row.behaviour_key,
            ProbeBehaviourVersion.version == head.current_version))).scalar_one_or_none()
        if current is not None and current.rule_set_digest != (row.inputs or {}).get("rule_set_digest"):
            moved.append({"input": "rule_set_digest", "was": (row.inputs or {}).get("rule_set_digest"),
                          "now": current.rule_set_digest})
    cited = (row.inputs or {}).get("cited", {})
    removed = {
        "probes": await _missing(db, ProbeMonitor, cited.get("probes", [])),
        "evaluations": await _missing(db, ProbeMonitorEvaluation, cited.get("evaluations", [])),
        "views": await _missing(db, ProbeMonitorDataset, cited.get("views", [])),
    }
    removed = {k: v for k, v in removed.items() if v}
    stale = bool(moved or removed)
    return {
        "stale": stale, "moved": moved, "inputs_removed": removed,
        "next": [next_success_report(row.behaviour_key, model_id=row.model_id)] if stale else [],
    }


async def envelope(db: Any, row: DetectorSuccessReport) -> Dict[str, Any]:
    return {"id": row.id, "created_at": row.created_at.isoformat() if row.created_at else None,
            "created_by": row.created_by, "body_sha256": row.body_sha256,
            "report": json.loads(row.body), "stale": await staleness(db, row)}
