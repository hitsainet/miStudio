"""The 040 test helpers do what the tests built on them assume (FTASKS 0.5, 1.1)."""
from __future__ import annotations

import re

import pytest
from sqlalchemy import text

from src.schemas.behaviour import KEY_PATTERN
from tests.support.behaviour_tokens import Matcher, derive_tokens
from tests.support.query_counter import count_queries
from tests.support.random_behaviour import behaviour_body, rng_for, slug


def test_slugs_match_the_key_pattern(request):
    rng = rng_for(request.node.nodeid)
    for _ in range(500):
        assert re.fullmatch(KEY_PATTERN, slug(rng))


def test_slugs_reproduce_from_the_seed(request):
    a = [slug(rng_for(request.node.nodeid)) for _ in range(3)]
    b = [slug(rng_for(request.node.nodeid)) for _ in range(3)]
    assert a == b
    assert slug(rng_for("x")) != slug(rng_for("y"))


def test_random_slugs_are_never_a_derived_token(request):
    tokens = set(derive_tokens())
    rng = rng_for(request.node.nodeid)
    assert not {slug(rng) for _ in range(2000)} & tokens


def test_behaviour_body_overrides(request):
    body = behaviour_body(rng_for(request.node.nodeid), target_fpr=0.02)
    assert body["target_fpr"] == 0.02 and re.fullmatch(KEY_PATTERN, body["key"])


def test_slug_boundary_matching(request):
    rng = rng_for(request.node.nodeid)
    key, other = slug(rng, words=1), slug(rng, words=1)
    phrase = f"{slug(rng, words=1)}-{slug(rng, words=1)}"
    m = Matcher([key, phrase])
    assert m.find(f"{key}-{other}") == []           # a key inside a longer slug
    assert m.find(f"{other}-{key}") == []
    assert m.find(f"{key}9") == []
    assert m.find(f"the {key} rule") == [key]       # prose
    assert m.find(f"'{key.upper()}'") == [key]      # case-insensitive
    assert m.find(f"a {phrase} interaction") == [phrase]
    assert m.find(f"{key}_count") == [key]          # `_` is not in the slug alphabet


def test_labels_match_as_phrases(request):
    rng = rng_for(request.node.nodeid)
    a, b = slug(rng, words=1), slug(rng, words=1)
    m = Matcher([f"{a} {b}"])
    assert m.find(f"x {a}\n   {b.upper()} y") == [f"{a} {b}"]
    assert m.find(f"{a}{b}") == []


def test_seed_prompt_names_are_not_tokens():
    """TI-15: a judge prompt's `name` names an instrument, not a behaviour."""
    import json

    from tests.support.behaviour_tokens import BACKEND

    tokens = set(derive_tokens())
    for path in (BACKEND / "src/data/judge_prompt_seeds").glob("*.json"):
        doc = json.loads(path.read_text())
        assert doc["name"] not in tokens
        # …and the key does not match inside the prompt name.
        assert Matcher([doc["behaviour"]]).find(doc["name"]) == [] or doc["behaviour"] == doc["name"]


@pytest.mark.asyncio
async def test_query_counter_counts_statements(async_engine):
    async with async_engine.connect() as conn:
        with count_queries(async_engine) as counter:
            await conn.execute(text("SELECT 1"))
            await conn.execute(text("SELECT 2"))
    assert counter.count == 2
