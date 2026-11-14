"""Text reduction keyed on miStudio's internal scopes (Feature 040 FR-33; FTDD §6.2; FTID D-4, TI-13; FTASKS 17.5).

Pinned: the scope set is the render module's (`SCOPE_ROLES`), so a scope added there is red here until the
reducer learns it; each scope's message selection; contents joined by one newline; non-string content
skipped and empty content kept; spans mapped back to `(message_index, offset)`, split at a join.
"""
from __future__ import annotations

import pytest

from src.rule_signals import reduce as r

CONVERSATION = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"},
    {"role": "assistant", "content": "reply"},
]


def test_the_scopes_are_the_render_modules():
    from src.services.probe_monitor_render import SCOPE_ROLES

    assert set(r.SCOPES) == set(SCOPE_ROLES)


@pytest.mark.parametrize("scope,text", [
    ("all", "sys\nu1\na1\nu2\nreply"),
    ("content", "sys\nu1\na1\nu2\nreply"),
    ("user", "u1\nu2"),
    ("assistant", "a1\nreply"),
    ("last_assistant", "reply"),
    ("input", "sys\nu1\na1\nu2"),
    ("last_user", "u2"),
])
def test_each_scope_selects_its_messages(scope, text):
    assert r.reduce(CONVERSATION, scope).text == text


def test_a_user_ended_conversation_has_no_reply_and_all_of_it_is_input():
    convo = CONVERSATION[:-1]
    assert r.reduce(convo, "last_assistant").missing == "no_scored_tokens"
    assert r.reduce(convo, "input").text == "sys\nu1\na1\nu2"


def test_plain_prose_is_one_user_message():
    msgs = r.as_messages("a row of prose")
    assert msgs == [{"role": "user", "content": "a row of prose"}]
    for scope in ("all", "content", "user", "input", "last_user"):
        assert r.reduce(msgs, scope).text == "a row of prose"
    assert r.reduce(msgs, "last_user").missing is None


def test_null_content_is_skipped_and_empty_content_kept():
    msgs = [{"role": "user", "content": ""}, {"role": "assistant", "content": None}, {"role": "user", "content": "x"}]
    assert r.reduce(msgs, "all").text == "\nx"


def test_no_user_turn_is_named():
    assert r.reduce([{"role": "system", "content": "s"}], "last_user").missing == "no_user_turn"


def test_spans_map_back_to_messages_and_split_at_a_join():
    reduced = r.reduce(CONVERSATION, "user")  # "u1\nu2"
    assert reduced.locate(0, 2) == [{"message_index": 1, "start": 0, "end": 2}]
    assert reduced.locate(3, 5) == [{"message_index": 3, "start": 0, "end": 2}]
    assert reduced.locate(1, 4) == [{"message_index": 1, "start": 1, "end": 2}, {"message_index": 3, "start": 0, "end": 1}]


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError):
        r.reduce(CONVERSATION, "bogus")
