"""The disagreement export (038 FR-11, FR-11b; D33, D57).

Every line is a `DisagreementRow` (`mistudio.spot-check-disagreement/v1`) whose `messages` is role/content
chat. The cross-repo test runs miDataworks' OWN upload detection (`services/sources/detection.py`) over an
exported sample with the column pins it derives from OUR published schema, and requires `role_content` and
recognition as a spot-check export. It skips loudly without the sibling and fails under
MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1. Control C-MESSAGES: `messages` renamed in the model → red.
"""
from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

from src.schemas.spot_check import DisagreementRow
from src.services import probe_spot_check as spot
from tests.support.sibling_repos import midataworks_repo

PUBLISHED = pathlib.Path(__file__).resolve().parents[3] / "docs" / "schemas" / "spot-check-disagreement-v1.json"
CHECK = SimpleNamespace(id="psc_1", behaviour="pii", prompt_key="pii-presence/v1", prompt_hash="a" * 64)


def _item(fired, positive, **over):
    base = dict(not_recorded=False, parse_failed=False, probe_fired=fired, judge_positive=positive,
                judged_text=[{"role": "user", "content": "my name is Ada"},
                             {"role": "assistant", "content": "hello"}],
                session_id="ps_1", request_id="chatcmpl-1", turn_index=0, event_id=7, stratum="fired",
                probe_id="pm_1", millm_probe_id="pr_1", probe_score=0.9, threshold=0.5, judge_value="no",
                judge_score=None, window="all", judge_saw="newest_turn")
    base.update(over)
    return SimpleNamespace(**base)


def _lines():
    items = [_item(True, False), _item(True, True), _item(False, True, request_id="chatcmpl-2"),
             _item(True, None, parse_failed=True), _item(True, False, judged_text=None)]
    return list(spot.export_jsonl(CHECK, items))


def test_lines_are_role_content():
    lines = _lines()
    assert len(lines) == 2, "only judged disagreements with their text are exported"
    for line in lines:
        row = json.loads(line)
        DisagreementRow.model_validate(row)
        assert all(set(m) == {"role", "content"} for m in row["messages"])
        assert row["source"] == "millm-session" and row["format"] == "mistudio.spot-check-disagreement/v1"
        assert row["prompt_hash"] == "a" * 64 and row["behaviour"] == "pii"


def _midataworks_detection():
    root = midataworks_repo() / "backend" / "src" / "services" / "sources"
    if not (root / "detection.py").is_file():
        message = (f"miDataworks' upload detection is not at {root} (set MIDATAWORKS_REPO). Skipping is "
                   f"expected on the public mirror; MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1 makes it fail.")
        if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    # A namespace for miDataworks' package so its `src` never shadows ours; its __init__ is not run.
    package = types.ModuleType("_mdw_sources")
    package.__path__ = [str(root)]
    sys.modules["_mdw_sources"] = package
    try:
        return importlib.import_module("_mdw_sources.detection"), importlib.import_module("_mdw_sources.spot_check")
    finally:
        for name in [n for n in sys.modules if n.startswith("_mdw_sources")]:
            sys.modules.pop(name, None)


def test_midataworks_detects_the_export_as_chat_and_as_a_spot_check():
    detection, spot_check = _midataworks_detection()
    pins = spot_check.pinned_columns(json.loads(PUBLISHED.read_text()))
    assert pins and "session_id" in pins and "messages" not in pins
    rows = [json.loads(line) for line in _lines()]
    columns = [(name, "string") for name in rows[0]]
    detected = detection.detect(columns, rows, spot_columns=pins)
    assert detected.chat_format == "role_content"
    assert detected.provenance == spot_check.PROVENANCE
