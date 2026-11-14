"""The pilot's rule sets (Feature 040 S3; FR-47 (b); §15; FTASKS 23.6).

The public pack `tests/fixtures/behaviour_packs/dlp-identifiers.json` carries one rule set per identifier type,
built from validator-library primitives only, with cases on the published test or reference values and the
constructed unissuable values the vendored miDataworks vectors already carry (D28/D30). Pinned here:

* every pilot case passes on miStudio's engine — zero mismatches (the vendored vectors for the four validators
  are `test_rule_signals_conformance.py`'s);
* each type's check-digit or range primitive has a case that turns red when the primitive's final step is dropped
  (controls P1 – P4 in the controls record);
* the masked row above the probe's bar is VETOED in the rules ∪ probe evaluation, through the real task.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.rule_signals as rule_signals
from src.core.config import settings
from tests.support import rules_world as rw

PACK = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "behaviour_packs" / "dlp-identifiers.json")
                  .read_text())
SETS = {b["key"]: b["rule_set"]["document"] for b in PACK["behaviours"]}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def test_the_pack_carries_four_rule_sets():
    assert len(SETS) == 4
    for key, doc in SETS.items():
        assert doc["behaviour_key"] == key


@pytest.mark.parametrize("key", sorted(SETS))
def test_every_pilot_case_passes(key):
    compiled = rule_signals.load(SETS[key], rule_signals.Limits())
    report = rule_signals.run_cases(compiled)
    assert report.ok, [f.report() for f in report.failures]
    assert report.passed == len(SETS[key]["cases"]) and not report.unverified


@pytest.mark.parametrize("key", sorted(SETS))
def test_the_cases_cover_valid_masked_and_refused(key):
    cases = SETS[key]["cases"]
    outcomes = {(c["expect"]["outputs"]["disclosed"], c["expect"]["vetoed"]) for c in cases}
    assert outcomes == {(True, False), (False, True), (False, False)}
    kinds = {c["source"]["kind"] for c in cases if c["expect"]["outputs"]["disclosed"]}
    assert kinds <= {"published_test_value", "published_reference_value"}, "a valid case must be a published value (D28)"


async def test_the_masked_row_above_the_bar_is_vetoed(client, async_session, data_dir, monkeypatch, request):
    text = {"U": "my number is 078-05-1120 thanks", "M": "my number is XXX-XX-1120 thanks",
            "B": "old XXX-XX-1120 new 078-05-1120", "P": "nothing to see in these plain words"}
    world = await rw.build(client, async_session, data_dir, request.node.nodeid, key="ssn-disclosure",
                           rule_set=SETS["ssn-disclosure"], text=text)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert out["outcome"]["status"] == "completed", out
    view = (await client.get(f"{rw.API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    # p3 (masked, score above the bar) is vetoed: union recall 4/6, not the probe's 3 + the rules' 3 unvetoed
    det = view["detectors"]
    assert (det["probe_alone"]["positives"]["flagged"], det["rules_or_probe"]["positives"]["flagged"]) == (3, 4)
    assert det["rules_or_probe"]["negatives"]["flagged"] == 2  # n2 (masked, above the bar) vetoed too
