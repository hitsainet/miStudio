"""Feature 040 FR-9 (FTASKS 1.5): the default judge prompt's behaviour key is DATA, read from its seed file.

`stakes_spec()` held the key as a literal — a second copy of the seed file's value and a behaviour named
in source. It now reads the seed's `behaviour`, so the key a judge run records (`metrics.behaviour`) is
unchanged byte for byte, and follows the file if the file is the one that changes.
"""
from __future__ import annotations

import json

from src.models.probe_judge_prompt import ProbeJudgePrompt
from src.services import probe_monitor_judge as judge
from src.services.judge_prompt_library import is_default_prompt, summary
from src.services.judge_prompt_seeds import (
    BUILTIN_SEED_DIR,
    DEFAULT_PROMPT_NAME,
    DEFAULT_PROMPT_VERSION,
    builtin_document,
)
from tests.support.random_behaviour import rng_for, slug


def _seed_value() -> str:
    path = BUILTIN_SEED_DIR / f"{DEFAULT_PROMPT_NAME}.v{DEFAULT_PROMPT_VERSION}.json"
    return json.loads(path.read_bytes())["behaviour"]


def test_recorded_key_is_the_seed_files_byte_for_byte():
    spec = judge.stakes_spec()
    assert spec.behaviour == _seed_value()
    assert spec.behaviour.encode("utf-8") == _seed_value().encode("utf-8")


def test_the_key_is_read_not_held(monkeypatch, request):
    """A literal would survive a changed seed; the read follows it."""
    other = slug(rng_for(request.node.nodeid))
    real = builtin_document

    def patched(name, version):
        doc = dict(real(name, version))
        doc["behaviour"] = other
        return doc

    monkeypatch.setattr(judge, "builtin_document", patched)
    assert judge.stakes_spec().behaviour == other


def test_the_default_row_is_marked_for_the_ui(request):
    rng = rng_for(request.node.nodeid)
    default = ProbeJudgePrompt(
        id="jp_x", name=DEFAULT_PROMPT_NAME, version=DEFAULT_PROMPT_VERSION, title="t", behaviour=slug(rng),
        system_prompt="s", user_template="{text}", answer_form={"kind": "binary"}, max_interaction_chars=10,
        content_hash="a" * 64, source="seed", created_by="seed",
    )
    other = ProbeJudgePrompt(
        id="jp_y", name=slug(rng), version=DEFAULT_PROMPT_VERSION, title="t", behaviour=slug(rng),
        system_prompt="s", user_template="{text}", answer_form={"kind": "binary"}, max_interaction_chars=10,
        content_hash="a" * 64, source="seed", created_by="seed",
    )
    assert summary(default)["is_default"] is True
    assert summary(other)["is_default"] is False
    assert is_default_prompt(DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_VERSION + 1) is False
