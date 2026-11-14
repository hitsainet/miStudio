"""The candidate vocabulary and the verdict kinds (Feature 041 FR-8; FTASKS 6.1, 6.4).

The expected vocabulary is DERIVED FROM `REGISTRY` here, never typed out, so a kind added to the registry
is covered and a kind dropped from the default list is caught. Verdicts are checked against hand counts on
five members.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.services.probe_blend import candidates as cand
from src.services.probe_blend import engine
from tests.support.blend_fixtures import bar


def expected_names(n: int, rules: bool) -> list:
    names = []
    for kind in cand.REGISTRY.values():
        names += [f"{kind.name}:{k}" for k in range(2, n)] if kind.parametrised else [kind.name]
    return names + ([cand.RULES_PREFIX + x for x in names] if rules else [])


@pytest.mark.parametrize("n", [2, 3, 5, 9])
@pytest.mark.parametrize("rules", [False, True])
def test_vocabulary_complete_from_registry(n, rules):
    assert cand.vocabulary(n, rules) == expected_names(n, rules)
    assert cand.resolve(None, n, rules) == [cand.parse_name(x, n, rules) for x in expected_names(n, rules)]


def test_unknown_name_lists_vocabulary():
    with pytest.raises(cand.CandidateUnknown) as caught:
        cand.parse_name("weighted_vote", 4, False)
    assert caught.value.vocabulary == expected_names(4, False)
    with pytest.raises(cand.CandidateUnknown):
        cand.parse_name("k_of_n:4", 4, False)               # k must be in 2..n-1
    with pytest.raises(cand.CandidateUnknown):
        cand.parse_name("rules_or_stack", 4, False)         # no rule set, no wrapper


def test_kind_names_expand_in_the_fixed_order():
    got = [c.name for c in cand.resolve(["k_of_n", "stack"], 5, False)]
    assert got == ["stack", "k_of_n:2", "k_of_n:3", "k_of_n:4"]


def test_any_all_k_parameters():
    assert cand.parse_name("any_of", 5).k == 1 and cand.parse_name("all_of", 5).k == 5
    assert cand.parse_name("k_of_n:3", 5).k == 3 and cand.parse_name("stack", 5).k is None


def test_or_and_k_of_n_equal_hand_verdicts():
    """Five members with known bars; per row the hand count of members at or above their bar decides."""
    rng = np.random.default_rng(7)
    rows, m = 200, 5
    scores = rng.random((rows, m))
    bars = [0.3, 0.5, 0.5, 0.7, 0.9]
    aligned = engine.Aligned(view="v", positions=np.arange(rows), labels=np.zeros(rows, np.int64), scores=scores,
                             n_scored=np.full((rows, m), 10))
    n_fired = engine.n_fired_of([bar(b) for b in bars], aligned)
    hand = (scores >= np.asarray(bars)).sum(axis=1)
    np.testing.assert_array_equal(n_fired, hand)
    for name, k in (("any_of", 1), ("all_of", 5), ("k_of_n:2", 2), ("k_of_n:3", 3), ("k_of_n:4", 4)):
        c = cand.parse_name(name, m)
        np.testing.assert_array_equal(engine.fires_of(c, n_fired, None), hand >= k)
    # rows sitting exactly at k fire (>=, not >)
    c3 = cand.parse_name("k_of_n:3", m)
    assert (hand == 3).any() and engine.fires_of(c3, n_fired, None)[hand == 3].all()
