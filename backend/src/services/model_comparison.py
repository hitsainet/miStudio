"""One behaviour's detectors on several models, side by side (Feature 040 FR-26; FTDD §5.2).

The models are the definition's `models` (each a miStudio model id or a Hugging Face repo id resolved to
one), or, when the definition names none, every model with a probe of the behaviour. Each row is that
model's stored success report (`detector_success_report.build`, so the same inputs return the same
report) with its criteria verdicts in adjacent columns; the blend column reads `not_assessable`,
`no_blend_report`, until 041.

⚠ DIFFERENT DATASETS ARE NOT COMPARED (FR-26): when the models' probes were trained on different
dataset versions the comparison is refused `dataset_identity_mismatch`, naming each model's version.
A model whose report cannot be built is a row with its refusal (code, message, `next`), never a gap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import or_, select

from ..models.model import Model
from . import detector_success_report as reports
from .detector_next import DetectorRefused


async def _model_ids(db: Any, key: str, version: Any) -> List[str]:
    declared = [m["model_id"] for m in (version.models or [])]
    if not declared:
        return sorted({run.model_id for _p, run in await reports._probes_of(db, key)})
    out: List[str] = []
    for ref in declared:
        row = (await db.execute(select(Model.id).where(or_(Model.id == ref, Model.repo_id == ref)).limit(1))).first()
        out.append(row[0] if row is not None else ref)
    return list(dict.fromkeys(out))


async def compare(db: Any, key: str, *, version: Optional[int], dataset_version: Optional[str],
                  view_ids: Optional[List[str]], limit: Optional[int], offset: int, actor: str, root: Path,
                  cursor_of: Callable[[int], str], settings: Any = None) -> Dict[str, Any]:
    if settings is None:
        from ..core.runtime_settings import effective

        settings = effective()
    _head, row = await reports.resolve_version(db, key, version)
    models = await _model_ids(db, key, row)
    cap = int(settings.behaviour_max_models)
    size = limit if limit is not None else cap
    if size > cap:
        raise DetectorRefused(422, "over_cap", f"a page of {size} models is more than behaviour_max_models ({cap})",
                              {"setting": "behaviour_max_models", "value": cap, "field": "limit", "got": size})
    page = models[offset: offset + size]
    # ⚠ THE IDENTITY CHECK COMES FIRST, so a refused comparison stores no reports.
    resolved: Dict[str, Any] = {}
    trained_on: Dict[str, Optional[str]] = {}
    for model_id in page:
        try:
            probe, run, _ = await reports.resolve_probe(db, key, model_id, None, dataset_version)
        except DetectorRefused as refusal:
            resolved[model_id] = refusal
            continue
        train_view = (await reports._views(db, [run.train_dataset_id]))[0]
        trained_on[model_id] = reports._training_identity(train_view)
        resolved[model_id] = probe
    if len(set(trained_on.values())) > 1:
        raise DetectorRefused(
            409, "dataset_identity_mismatch",
            "these models' probes were trained on different dataset versions, so their figures are not "
            "comparable; name one dataset_version",
            {"difference": trained_on})
    rows: List[Dict[str, Any]] = []
    for model_id in page:
        if isinstance(resolved[model_id], DetectorRefused):
            rows.append({"model_id": model_id, "refused": resolved[model_id].as_detail()})
            continue
        request = {"version": row.version, "model_id": model_id, "dataset_version": dataset_version,
                   "view_ids": view_ids}
        try:
            built = await reports.build(db, key, request, actor, root=root, settings=settings)
        except DetectorRefused as refusal:
            rows.append({"model_id": model_id, "refused": refusal.as_detail()})
            continue
        body = built["report"]
        rows.append({
            "model_id": model_id,
            "report_id": built["id"],
            "probe_id": body["probe"]["id"],
            "overall_verdict": body["overall_verdict"],
            "criteria": [{"id": c["id"], "verdict": c["verdict"], "recall": (c.get("figures") or {}).get("recall"),
                          "reasons": [r["code"] for r in c["reasons"]]} for c in body["criteria"]],
            "blend": {"verdict": "not_assessable", "reasons": ["no_blend_report"]},
            "stale": built["stale"]["stale"],
        })
    return {
        "key": key, "behaviour_version": row.version, "dataset_version": dataset_version,
        "models": rows, "total": len(models),
        "next_cursor": cursor_of(offset + size) if offset + size < len(models) else None,
    }
