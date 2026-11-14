"""Gemma Scope names are parsed in linear time, with the regexes' exact answers.

CodeQL py/polynomial-redos #59/#60 (security alerts 2026-10-17). `gemma_scope_hook_kind` ran
two regexes over a user-supplied repository id or file path. Both backtracked quadratically
on a name repeating the prefix: `--gemma-scope-` x 4000 took 0.76 s and doubling the input
quadrupled it. The replacement is a segment parser.

Two properties, each tested against the retired regexes kept here as the ORACLE:
1. SAME ANSWER on every input — hand cases plus a seeded fuzz over the alphabet the
   regexes care about (`/`, `-`, `--`, `_all`, `-canonical`, the kind names).
2. LINEAR — the time ratio between an input and one 8x longer. Linear gives ~8x, the old
   quadratic ~64x. A ratio, not a wall-clock bound, so a slow CI machine cannot fail it.
"""

from __future__ import annotations

import random
import re
import time

import pytest

from src.services.sae_manager_service import gemma_scope_hook_kind

_OLD_1 = re.compile(
    r"(?:^|/|--)gemma-scope-[a-z0-9.\-]*?-(res|mlp|att|transcoders)(?:-canonical)?(?:/|$)"
)
_OLD_2 = re.compile(
    r"(?:^|/|--)gemma-scope-2-[a-z0-9.\-]+/(?:.*/)?(resid_post|mlp_out|attn_out|transcoder|clt|crosscoder)(?:_all)?(?:/|$)"
)
_OLD_1_HOOK = {"res": "residual", "mlp": "mlp", "att": "attention", "transcoders": "transcoder"}
_OLD_2_HOOK = {
    "resid_post": "residual", "mlp_out": "mlp", "attn_out": "attention",
    "transcoder": "transcoder", "clt": "transcoder", "crosscoder": "crosscoder",
}


def oracle(origin):
    """The implementation as it stood before the fix, verbatim."""
    if not origin:
        return None
    text = str(origin).replace("\\", "/").lower()
    match = _OLD_2.search(text)
    if match:
        return _OLD_2_HOOK[match.group(1)]
    match = _OLD_1.search(text)
    if not match:
        return None
    if match.group(1) == "res" and re.search(r"(?:^|/)embedding(?:/|$)", text[match.end() - 1:]):
        return "embedding"
    return _OLD_1_HOOK[match.group(1)]


HAND_CASES = [
    "google/gemma-scope-2b-pt-res",
    "google/gemma-scope-2b-pt-res-canonical",
    "google/gemma-scope-2b-pt-mlp",
    "google/gemma-scope-9b-it-att/layer_3",
    "google/gemma-scope-2b-pt-transcoders",
    "google/gemma-scope-2b-pt-res/embedding/width_4k",
    "google/gemma-scope-2b-pt-res/layer_0/embedding",
    "google/gemma-scope-2b-pt-mlp/embedding",
    "/data/hf/models--google--gemma-scope-2b-pt-res/snapshots/abc/layer_5",
    "google/gemma-scope-2-4b-pt/resid_post/layer_9",
    "google/gemma-scope-2-4b-pt/resid_post_all/layer_9",
    "google/gemma-scope-2-4b-pt/x/mlp_out/y/attn_out",
    "google/gemma-scope-2-4b-pt/clt",
    "google/gemma-scope-2-4b-pt",
    "models--google--gemma-scope-2-270m-pt/snapshots/h/transcoder/l",
    "google/gemma-scope-2-4b-pt-res",
    "GOOGLE\\Gemma-Scope-2B-PT-RES",
    "google/gemma-scope--res",
    "x--y--gemma-scope-a-res",
    "gemma-scope-a_b-res",
    "gemma-scope-a_b--gemma-scope-c-res",
    "gemma-scope-2-/resid_post",
    "my-gemma-scope-2b-pt-res",
    "",
    None,
    "neuronpedia/gpt2-small-res-jb",
]

ALPHABET = ["/", "-", "--", "a", "2b", ".", "_", "x", "gemma-scope-", "gemma-scope-2-",
            "res", "mlp", "att", "transcoders", "-canonical", "resid_post", "mlp_out",
            "attn_out", "transcoder", "clt", "crosscoder", "_all", "embedding", "models--"]


@pytest.mark.parametrize("origin", HAND_CASES)
def test_hand_cases_match_the_regexes(origin):
    assert gemma_scope_hook_kind(origin) == oracle(origin)


def test_a_seeded_fuzz_matches_the_regexes():
    rng = random.Random(20261017)
    hits = 0

    def noise(k):
        return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, k)))

    for i in range(20000):
        if i % 2:
            origin = noise(14)
        else:
            # Shaped like the names that match, then perturbed — so the fuzz lives near
            # the boundary where the two implementations could disagree.
            name = rng.choice(["gemma-scope-", "gemma-scope-2-"]) + noise(3)
            kind = rng.choice(["res", "mlp", "att", "transcoders", "resid_post", "mlp_out",
                               "clt", "crosscoder", "x"])
            tail = rng.choice(["", "-canonical", "_all", "/embedding", "/" + noise(2)])
            sep = rng.choice(["/", "--", "-", "/"])
            origin = noise(3) + sep + name + rng.choice(["-", "/", "/" + noise(2) + "/"]) + kind + tail + noise(2)
        expected = oracle(origin)
        hits += expected is not None
        assert gemma_scope_hook_kind(origin) == expected, origin
    # The fuzz must actually exercise the matching branches, not only the misses.
    assert hits > 800, hits


def _best_of(fn, arg, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(arg)
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize(
    "unit",
    ["--gemma-scope-", "--gemma-scope-2-", "/gemma-scope-2-a/", "gemma-scope-a-"],
)
def test_parsing_time_grows_linearly(unit):
    small, large = unit * 1000, unit * 8000
    _best_of(gemma_scope_hook_kind, small, 1)  # warm
    ratio = _best_of(gemma_scope_hook_kind, large) / max(_best_of(gemma_scope_hook_kind, small), 1e-6)
    assert ratio < 24, f"8x the input took {ratio:.1f}x the time — superlinear (quadratic is ~64x)"
