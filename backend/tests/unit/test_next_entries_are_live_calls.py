"""Every `next` Feature 041 can write names a live tool with arguments it accepts (FR-26; FTASKS 7.4).

The constructors are the only place 041 writes a tool name in `services/` (TI-17); each is checked here
against the REAL server's registry and the tool's input schema, derived from `detector_next.CONSTRUCTORS`
(no hand list). The refusals produced through the routes are checked call by call in
`test_blend_refusals.py`, `test_probe_family.py` and `test_behaviour_checklist_family.py`.
"""
from __future__ import annotations

import pytest

from src.services import detector_next
from tests.support.next_calls import assert_next_calls_are_live, live_input_schemas

#: The tools 041 points agents at (its own and the ones it reuses).
TOOLS_041 = {"get_probe_family", "start_probe_training_run", "register_probe_dataset", "build_probe_definition",
             "evaluate_probe_monitor", "unarchive_behaviour", "unarchive_probe_dataset", "evaluate_probe_blends",
             "rerun_blend_report", "get_blend_report", "list_blend_reports"}


def test_every_041_tool_is_reached_by_a_constructor():
    assert TOOLS_041 <= detector_next.next_tools()


@pytest.mark.parametrize("name", sorted(n for n, c in detector_next.CONSTRUCTORS.items() if c.tool in TOOLS_041))
def test_each_constructor_is_live(name):
    constructor = detector_next.CONSTRUCTORS[name]
    assert constructor.tool in live_input_schemas()
    assert_next_calls_are_live([constructor.example()])
