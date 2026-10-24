"""The detector data a behaviour version carries: validation, carry-over and diff (Feature 040 FR-1 – FR-5, FR-7).

⚠ CONSISTENT, NOT PREMATURE (FR-2). On write only INTERNAL consistency is checked: unique keys and ids,
a `held_out_test` the version declares, bounds, target inheritance, `named_probe` ⇔ `probe_id`, list caps.
Things that may simply not exist YET — a model not downloaded, a column no view carries — are accepted
and returned as `warnings`, each with a `next`; evaluation checks them hard (FR-22).

⚠ EVERY REFUSAL NAMES THE FIELD AND THE RULE: ``invalid_field`` with ``field`` (a path such as
``success_criteria[2].held_out_test``) and ``rule``; a list past its cap is ``over_cap`` with the runtime
key and its value (FR-7). A framework 422 would name neither.

⚠ CAPS COME FROM `Limits`, BUILT ONCE PER REQUEST FROM `runtime_settings.effective()` (FR-7, FR-11): no
module-level settings read and no literal, so an operator's change applies at the next request and a
mutation that reads a literal is caught by the per-cap test.

⚠ CARRY-OVER IS FROM THE NAMED BASE VERSION'S ROW (FR-3), never from whatever is current: the caller
passes the row it loaded for `base_version`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .detector_identity import jcs_bytes
from .detector_next import DetectorRefused, next_download_model, next_find_views, next_list_models

#: The six FR-1 fields as stored on `probe_behaviour_versions` (the request's `rule_set` is stored as
#: its digest).
STORED_FIELDS = (
    "sub_questions", "success_criteria", "held_out_columns", "models", "calibration_floor",
    "rule_set_digest",
)


@dataclass(frozen=True)
class Limits:
    max_sub_questions: int
    max_success_criteria: int
    max_held_out_columns: int
    max_models: int
    rule_set_max_bytes: int

    @classmethod
    def from_settings(cls, settings: Any = None) -> "Limits":
        if settings is None:
            from ..core.runtime_settings import effective

            settings = effective()
        return cls(
            max_sub_questions=settings.behaviour_max_sub_questions,
            max_success_criteria=settings.behaviour_max_success_criteria,
            max_held_out_columns=settings.behaviour_max_held_out_columns,
            max_models=settings.behaviour_max_models,
            rule_set_max_bytes=settings.rule_set_max_bytes,
        )


#: (request field, `Limits` attribute, runtime key)
CAPS = (
    ("sub_questions", "max_sub_questions", "behaviour_max_sub_questions"),
    ("success_criteria", "max_success_criteria", "behaviour_max_success_criteria"),
    ("held_out_columns", "max_held_out_columns", "behaviour_max_held_out_columns"),
    ("models", "max_models", "behaviour_max_models"),
)


def invalid(field: str, rule: str, message: str, **details: Any) -> DetectorRefused:
    return DetectorRefused(422, "invalid_field", message, {"field": field, "rule": rule, **details})


def over_cap(setting: str, value: int, field: str, got: int) -> DetectorRefused:
    return DetectorRefused(
        422, "over_cap",
        f"{field} has {got} entries; the runtime setting {setting} allows at most {value}. Nothing is "
        f"truncated: send fewer, or ask the operator to raise {setting}.",
        {"setting": setting, "value": value, "field": field, "got": got},
    )


def _dump(items: Optional[Sequence[Any]]) -> Optional[List[Any]]:
    """JSON values of a pydantic list, omitting unset optionals (absent stays absent)."""
    if items is None:
        return None
    out = []
    for item in items:
        if hasattr(item, "model_dump"):
            item = item.model_dump(mode="json", exclude_none=True)
        elif isinstance(item, dict):
            item = {k: v for k, v in item.items() if v is not None}
        out.append(item)
    return out


def detector_values(body: Any) -> Dict[str, Any]:
    """The four list fields and the floor of a request body, as JSON values (rule set handled apart)."""
    get = body.get if isinstance(body, Mapping) else (lambda name: getattr(body, name, None))
    return {
        "sub_questions": _dump(get("sub_questions")),
        "success_criteria": _dump(get("success_criteria")),
        "held_out_columns": list(get("held_out_columns")) if get("held_out_columns") is not None else None,
        "models": _dump(get("models")),
        "calibration_floor": get("calibration_floor"),
    }


def _duplicates(values: Iterable[Any]) -> Optional[int]:
    seen: Set[Any] = set()
    for index, value in enumerate(values):
        if value in seen:
            return index
        seen.add(value)
    return None


def validate(fields: Mapping[str, Any], version_target_fpr: Optional[float], limits: Limits) -> None:
    """FR-2's internal-consistency rules on the stored JSON values; raises `DetectorRefused`."""
    for field, attribute, setting in CAPS:
        values = fields.get(field)
        cap = getattr(limits, attribute)
        if values is not None and len(values) > cap:
            raise over_cap(setting, cap, field, len(values))

    subs = fields.get("sub_questions") or []
    dup = _duplicates(s["key"] for s in subs)
    if dup is not None:
        raise invalid(f"sub_questions[{dup}].key", "duplicate_key",
                      f"sub-question key {subs[dup]['key']!r} appears twice; keys are unique in a version")

    held_out = fields.get("held_out_columns")
    dup = _duplicates(held_out or [])
    if dup is not None:
        raise invalid(f"held_out_columns[{dup}]", "duplicate_column",
                      f"held-out column {held_out[dup]!r} appears twice")
    for index, column in enumerate(held_out or []):
        if not isinstance(column, str) or not column.strip() or len(column) > 255:
            raise invalid(f"held_out_columns[{index}]", "invalid_column",
                          "a held-out column is a column name of 1 to 255 characters")

    models = fields.get("models") or []
    dup = _duplicates(m["model_id"] for m in models)
    if dup is not None:
        raise invalid(f"models[{dup}].model_id", "duplicate_model",
                      f"model {models[dup]['model_id']!r} appears twice")

    floor = fields.get("calibration_floor")
    if floor is not None and (isinstance(floor, bool) or not isinstance(floor, int) or floor < 1):
        raise invalid("calibration_floor", "out_of_range", "calibration_floor is a whole number of at least 1")

    criteria = fields.get("success_criteria") or []
    dup = _duplicates(c["id"] for c in criteria)
    if dup is not None:
        raise invalid(f"success_criteria[{dup}].id", "duplicate_id",
                      f"criterion id {criteria[dup]['id']!r} appears twice; ids are unique in a version")
    declared = list(held_out or [])
    for index, criterion in enumerate(criteria):
        path = f"success_criteria[{index}]"
        test = criterion.get("held_out_test")
        if test is not None and test not in declared:
            raise invalid(f"{path}.held_out_test", "not_declared",
                          f"held_out_test {test!r} is not one of this version's held_out_columns "
                          f"({declared}); declare it there or judge the criterion in distribution",
                          declared=declared)
        fpr = criterion.get("target_fpr")
        if fpr is not None and not (0.0 < float(fpr) < 1.0):
            raise invalid(f"{path}.target_fpr", "out_of_range", "a target false-positive rate is in (0, 1)")
        # ⚠ INHERITANCE (FR-2, 040-D22): a criterion without its own target uses the version's.
        if fpr is None and version_target_fpr is None:
            raise invalid(f"{path}.target_fpr", "target_fpr_missing",
                          f"criterion {criterion['id']!r} sets no target_fpr and the version sets none to "
                          f"inherit; set one of {path}.target_fpr or target_fpr",
                          fields=[f"{path}.target_fpr", "target_fpr"])
        recall = criterion.get("min_recall")
        if recall is None or not (0.0 <= float(recall) <= 1.0):
            raise invalid(f"{path}.min_recall", "out_of_range", "min_recall is in [0, 1]")
        named = criterion.get("detector") == "named_probe"
        if named and not criterion.get("probe_id"):
            raise invalid(f"{path}.probe_id", "required_by_named_probe",
                          "a named_probe criterion names the probe it judges (probe_id)")
        if not named and criterion.get("probe_id"):
            raise invalid(f"{path}.probe_id", "only_with_named_probe",
                          f"probe_id is read only by the named_probe form, not {criterion.get('detector')!r}")


def effective_target_fpr(criterion: Mapping[str, Any], version_target_fpr: Optional[float]) -> Optional[float]:
    return criterion.get("target_fpr") if criterion.get("target_fpr") is not None else version_target_fpr


def columns_named(fields: Mapping[str, Any]) -> List[str]:
    """Held-out columns and `{column, value}` subset columns, in declaration order, once each."""
    out: List[str] = []
    for column in fields.get("held_out_columns") or []:
        if column not in out:
            out.append(column)
    for criterion in fields.get("success_criteria") or []:
        subset = criterion.get("subset")
        if isinstance(subset, dict) and subset.get("column") not in out:
            out.append(subset["column"])
    return out


async def warnings(db: Any, fields: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Things that may not exist YET (FR-2): returned, never refused, each with a `next`."""
    from sqlalchemy import or_, select

    from ..models.model import Model
    from ..models.probe_monitor import ProbeMonitorDataset

    out: List[Dict[str, Any]] = []
    for index, ref in enumerate(fields.get("models") or []):
        model_id = ref["model_id"]
        found = (await db.execute(
            select(Model.id).where(or_(Model.id == model_id, Model.repo_id == model_id)).limit(1)
        )).first()
        if found is None:
            out.append({
                "code": "model_not_downloaded",
                "field": f"models[{index}].model_id",
                "message": f"no model {model_id!r} is on this deployment yet; the version is saved and "
                           f"the model is checked when the behaviour is evaluated",
                "next": [next_download_model(model_id) if "/" in model_id else next_list_models()],
            })
    columns = columns_named(fields)
    if columns:
        declared: Set[str] = set()
        group_columns = getattr(ProbeMonitorDataset, "group_columns", None)
        if group_columns is not None:  # S2 adds view columns; until then no view declares any
            for value in (await db.execute(select(group_columns).where(group_columns.isnot(None)))).scalars():
                declared.update(value or [])
        for column in columns:
            if column not in declared:
                out.append({
                    "code": "column_not_on_any_view",
                    "field": "held_out_columns" if column in (fields.get("held_out_columns") or []) else "success_criteria",
                    "column": column,
                    "message": f"no registered view declares column {column!r} yet; criteria that read "
                               f"it are judged once a view does",
                    "next": [next_find_views()],
                })
    return out


def carry_over(base: Any, change: Mapping[str, Any], fields_set: Iterable[str], names: Sequence[str]) -> Dict[str, Any]:
    """Every field in `names` the request did not SET is copied from `base` (the named base version's
    row); a field it set — null included, which removes it (FR-4) — is taken from `change`."""
    sent = set(fields_set)
    return {name: (change.get(name) if name in sent else getattr(base, name)) for name in names}


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return jcs_bytes(a) == jcs_bytes(b)


def field_diff(before: Mapping[str, Any], after: Mapping[str, Any], names: Sequence[str]) -> List[Dict[str, Any]]:
    """`[{path, before, after}]` for each field whose value changed. JSON values are compared by their
    RFC 8785 bytes, so key order never reads as a change."""
    return [
        {"path": name, "before": before.get(name), "after": after.get(name)}
        for name in names
        if not _same(before.get(name), after.get(name))
    ]
