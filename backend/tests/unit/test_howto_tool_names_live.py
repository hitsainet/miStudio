"""The `behaviour_detectors` how-to names 041's tools, and only live ones (Feature 041 FR-26; FTASKS 2.12, 9.8).

040's `test_howto_behaviour_detectors.py` checks every back-quoted name in the topic against the LIVE
registry; this pins that the topic actually TEACHES 041's steps (a topic that never mentions them would
pass that check), with each 041 tool named also present in the live registry.
"""
from __future__ import annotations

import re

from src.mcp_server.tools import howto
from tests.support.next_calls import live_input_schemas

#: The 041 tools the topic must name, by slice.
TAUGHT = {"get_probe_family", "start_probe_training_run", "register_probe_dataset", "get_behaviour_checklist",
          # S2
          "evaluate_probe_blends", "get_blend_report", "evaluate_probe_monitor", "rerun_blend_report",
          "get_blend_report_scores", "set_suggested_blend", "get_suggested_blend", "get_detector_success_report"}


def named(text: str) -> set:
    return set(re.findall(r"`([a-z][a-z0-9_]+)`", text))


def test_the_topic_names_the_family_tools_and_they_are_live():
    names = named(howto.TOPICS["behaviour_detectors"])
    assert TAUGHT <= names, sorted(TAUGHT - names)
    live = set(live_input_schemas())
    assert TAUGHT <= live, sorted(TAUGHT - live)


def test_the_topic_names_no_behaviour():
    """Placeholders only: every back-quoted name in the 041 steps is a tool or a field, never a key."""
    text = howto.TOPICS["behaviour_detectors"]
    tail = text[text.index("get_probe_family") - 40:]
    live = set(live_input_schemas())
    fields = {"model", "sub_question", "next", "missing", "family_trained", "blend_report", "family_exported",
              # Feature 040 S3 steps 10-11 follow 041's: field and value names, not tools or keys
              # (the same set test_howto_behaviour_detectors.py exempts)
              "rule_set", "code_rule", "runner_unavailable", "text_column", "rule_columns",
              "rules_alone", "rules_or_blend", "recut_windows"}
    assert named(tail) <= live | fields, sorted(named(tail) - live - fields)
