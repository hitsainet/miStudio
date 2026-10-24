"""The report object (Feature 041 FR-12; FTDD §6.4, FTID §7).

EVERY top-level key is present (`null` when absent), so a reader never mistakes a missing key for a
missing feature. No row text anywhere: ids, positions (in the `.npy` files, not here), counts and the group
VALUES used for subgroups (the privacy class of 040's breakdown). `has_rung: false` and the fixed
`statement` say what a blend is not.
"""
from __future__ import annotations

from typing import Any, Dict

from . import report_format


def report(row: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    resolved = row.resolved
    version = resolved.get("version") or {}
    views = resolved["views"]
    members = []
    for m in resolved["members"]:
        members.append({
            "probe_id": m["probe_id"], "run_id": m["run_id"], "name": None, "layer": m["layer"], "rule": m["rule"],
            "sub_question": m["sub_question"], "precision": m["precision"], "render_form": m["render_form"],
            "scope": m["scope"], "rung": m["rung"], "threshold": m["threshold"],
            "decision_revision": m["decision_revision"], "detector_digest": m["detector_digest"],
            "decision_digest": m["decision_digest"],
            "evaluation_ids": dict(resolved["evaluations"][m["probe_id"]]),
            "rows": {v: result["alignment"][v]["members"][m["probe_id"]] for v in result["alignment"]},
        })
    flags = list(resolved.get("flags") or []) + [f for f in result["folds"]["flags"] if f not in (resolved.get("flags") or [])]
    return {
        "format": report_format.FORMAT,
        "engine_version": report_format.ENGINE_VERSION,
        "report_id": row.id,
        "behaviour": {
            "key": row.behaviour_key, "version": row.behaviour_version, "target_fpr": resolved["target_fpr"],
            "rule_set": ({"digest": version.get("rule_set_digest")} if version.get("rule_set_digest") else None),
            "success_criteria_digest": version.get("success_criteria_digest"),
            "held_out_columns": version.get("held_out_columns") or [],
        },
        "model_id": row.model_id,
        "dataset_version": row.dataset_version,
        "views": {"blend_fit": views["blend_fit"], "tests": views["tests"],
                  "calibration": {"id": views["calibration"]["id"], "sample_rows": views["calibration"]["sample_rows"]}},
        "members": members,
        "alignment": {v: {"rows": a["rows"]} for v, a in result["alignment"].items()},
        "standardisation": result["standardisation"],
        "folds": {**result["folds"], "group_column": resolved.get("group_column")},
        "best_single": result["best_single"],
        "candidates": [
            {k: c[k] for k in ("index", "name", "kind", "produces", "params", "calibration", "cv", "tests", "criteria")}
            for c in result["candidates"]
        ],
        "members_own_figures": result["members_own_figures"],
        "ranking": result["ranking"],
        "flags": flags,
        "statement": report_format.STATEMENT,
        "has_rung": False,
    }
