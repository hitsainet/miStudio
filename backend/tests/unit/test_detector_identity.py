"""Report identity (Feature 040 S2 FR-25; 040-T17; FTASKS 11.1)."""
from __future__ import annotations

import copy

import pytest

from src.services.detector_identity import REPORT_ALGORITHM, identity, jcs_bytes, round6

BASE = {"report_algorithm": REPORT_ALGORITHM, "behaviour_key": "k-a", "behaviour_version": 2, "model_id": "m_1",
        "dataset_identity": {"kind": "view_set", "views": [{"id": "pmd_1", "rows": 10, "snapshots": {"c": "a" * 64}}]},
        "probe": {"id": "pm_1", "decision_revision": 3}, "cited": {"evaluations": ["pme_1"]},
        "rule_set_digest": None, "settings": {"calibration_floor_negatives": 2000}}


def test_identical_inputs_identical_digest_whatever_the_key_order():
    reordered = dict(reversed(list(copy.deepcopy(BASE).items())))
    assert identity(BASE) == identity(reordered)


@pytest.mark.parametrize("path,value", [
    (("report_algorithm",), REPORT_ALGORITHM + 1), (("behaviour_version",), 3), (("model_id",), "m_2"),
    (("dataset_identity", "views", 0, "rows"), 11), (("probe", "decision_revision"), 4),
    (("cited", "evaluations", 0), "pme_2"), (("rule_set_digest",), "b" * 64),
    (("settings", "calibration_floor_negatives"), 5000),
])
def test_each_input_class_moves_the_digest(path, value):
    moved = copy.deepcopy(BASE)
    target = moved
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    assert identity(moved) != identity(BASE)


def test_rfc8785_numbers_and_rounding():
    assert jcs_bytes({"a": 1.0, "b": 1e21}) == b'{"a":1,"b":1e+21}'
    assert round6(0.1234567) == 0.123457 and round6(-0.0000001) == 0.0 and round6(None) is None
