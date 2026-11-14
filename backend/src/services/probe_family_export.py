"""Family export: all or nothing, untouched bytes (Feature 041 S3; FR-23; T13; FTDD §6.7).

* `blockers` lists EVERY reason the family cannot be exported — a member with no current built
  definition, a built file gone from disk, members on different models, two members whose definition
  names collide, a build below rung 2 without its acknowledgement — each with its `next`. One blocker is
  enough to refuse the whole export (`family_export_blocked`), and all of them are named at once, so an
  agent fixes them in one pass rather than one refusal at a time.
* `archive` is a ZIP of each member's built file, STORED (no compression), entries sorted by probe id, a
  fixed 1980-01-01 timestamp and fixed permissions — so the archive's bytes depend only on the files.
* `index` is what an agent gets instead (MCP carries no binary): per member the single export's route, the
  file's sha256 and size, and the archive's route.

Entry names are probe ids (`<probe_id>.probe.json`, the single export's filename), never caller strings.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .detector_next import DetectorRefused, next_build_member

ZIP_TIME = (1980, 1, 1, 0, 0, 0)
EXPORT_ROUTE = "/api/v1/probe-monitors/probes/{probe_id}/export"


class FamilyExportBlocked(DetectorRefused):
    pass


def _path(probe: Any) -> Path:
    from ..core.config import settings

    return settings.resolve_data_path(probe.definition_path)


def blockers(members: Sequence[Any], runs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every blocker, in member order. `members` are probes; `runs` maps run id → run."""
    from .probe_decision import identity_fields

    out: List[Dict[str, Any]] = []
    models = sorted({runs[p.run_id].model_id for p in members if p.run_id in runs})
    if len(models) > 1:
        out.append({"probe_id": None, "reason": "models_differ", "models": models, "next": []})
    names: Dict[str, List[str]] = {}
    for probe in members:
        if not probe.definition_path or identity_fields(probe)["identity_current"] is False:
            out.append({"probe_id": probe.id, "reason": "no_current_definition",
                        "next": [next_build_member(probe.id, probe.rung)]})
            continue
        path = _path(probe)
        if not path.is_file():
            out.append({"probe_id": probe.id, "reason": "file_missing",
                        "next": [next_build_member(probe.id, probe.rung)]})
            continue
        try:
            document = json.loads(path.read_bytes())
        except ValueError:
            out.append({"probe_id": probe.id, "reason": "file_unreadable", "next": [next_build_member(probe.id, probe.rung)]})
            continue
        evidence = document.get("evidence") or {}
        if (evidence.get("rung") or 0) < 2 and not evidence.get("acknowledgement"):
            out.append({"probe_id": probe.id, "reason": "below_rung2_unacknowledged",
                        "next": [next_build_member(probe.id, probe.rung)]})
        names.setdefault(str(document.get("name") or probe.id), []).append(probe.id)
    for name, ids in names.items():
        if len(ids) > 1:
            out.append({"probe_id": None, "reason": "member_filename_collision", "name": name, "members": ids,
                        "next": []})
    return out


def refuse_if_blocked(members: Sequence[Any], runs: Dict[str, Any]) -> None:
    found = blockers(members, runs)
    if found:
        raise FamilyExportBlocked(
            409, "family_export_blocked",
            f"{len(found)} blocker(s) stop the family export; nothing is exported until every one is cleared",
            {"blockers": found},
            next=[n for b in found for n in b["next"]],
        )


def files(members: Sequence[Any]) -> List[Dict[str, Any]]:
    """Each member's untouched bytes, read once, in probe-id order."""
    out = []
    for probe in sorted(members, key=lambda p: p.id):
        payload = _path(probe).read_bytes()
        out.append({"probe_id": probe.id, "filename": f"{probe.id}.probe.json", "bytes": payload,
                    "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)})
    return out


def archive(entries: Sequence[Dict[str, Any]]) -> bytes:
    """ZIP_STORED, sorted, fixed timestamp and permissions: the bytes depend only on the files (T13)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
        for entry in sorted(entries, key=lambda e: e["probe_id"]):
            info = zipfile.ZipInfo(entry["filename"], date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, entry["bytes"])
    return buffer.getvalue()


def index(entries: Sequence[Dict[str, Any]], archive_route: str) -> Dict[str, Any]:
    return {
        "members": [{"probe_id": e["probe_id"], "filename": e["filename"], "sha256": e["sha256"], "size": e["size"],
                     "export_route": EXPORT_ROUTE.format(probe_id=e["probe_id"])} for e in entries],
        "archive_route": archive_route,
    }
