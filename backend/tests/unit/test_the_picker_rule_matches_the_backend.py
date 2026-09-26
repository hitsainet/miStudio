"""The steering picker's ported copy of the SAE rule must match the rule itself (review R3-B).

The picker used to offer every READY SAE, so choosing a Gemma Scope MLP or attention SAE, or
one recording no layer, produced a 422 only after the user had browsed it, picked features and
pressed Generate. `frontend/src/utils/saeSteerability.ts` therefore ports
``sae_hook_support``'s rule so the panel stops OFFERING what will be refused.

Two copies of one rule drift. This asserts they do not — and it lives HERE, beside the
authority, rather than in the frontend suite, for two reasons found the hard way:

* Vite will not read a file outside its project root. A `?raw` import of the backend rule
  from the vitest side failed to load, and the entire test file stopped running — 39 tests
  silently gone, the suite still reporting "passed" for everything else. A guard that can
  vanish like that is worse than none.
* `frontend/tsconfig.test.json` deliberately carries no ``@types/node`` and its error count is
  ratcheted DOWN-only, so `node:fs` / `node:path` / `process` added three errors to a gate
  whose whole purpose is that the count cannot grow back.

FAILS CLOSED: a missing file, a pattern that matches nothing, or a list that parses empty is a
failure, never a skip. A source scrape that quietly matches nothing is this repo's recurring
way of passing for the wrong reason.

MUTATION CONTROL (recorded in the WS-A review notes):
  TS1 a token changed in the TypeScript copy ('mlp' -> 'mlpx') -> test_the_token_sets_match
"""

import re
from pathlib import Path

import pytest

from src.services import sae_hook_support as backend_rule

PICKER = (
    Path(__file__).resolve().parents[3] / "frontend" / "src" / "utils" / "saeSteerability.ts"
)

#: (the exported TypeScript name, the frozenset it must equal). The Python names are private
#: by convention; reading them here is deliberate — the guard must compare the values the
#: refusals actually use, not a restatement of them.
TOKEN_SETS = [
    ("MLP_TOKENS", backend_rule._MLP_TOKENS),
    ("ATTENTION_TOKENS", backend_rule._ATTENTION_TOKENS),
    ("EMBEDDING_TOKENS", backend_rule._EMBEDDING_TOKENS),
    ("RESIDUAL_TOKENS", backend_rule._RESIDUAL_TOKENS),
]


def _picker_source() -> str:
    if not PICKER.is_file():
        pytest.fail(
            f"{PICKER} is missing: the picker's copy of the rule cannot be checked, and an "
            "unchecked copy is exactly what this guard exists to prevent"
        )
    return PICKER.read_text()


def _ported_tokens(source: str, name: str) -> set:
    match = re.search(rf"export const {name}\s*=\s*\[([^\]]*)\]", source)
    if not match:
        pytest.fail(
            f"{name} was not found in {PICKER}: this guard cannot pass by finding nothing"
        )
    tokens = set(re.findall(r"'([^']+)'", match.group(1)))
    if not tokens:
        pytest.fail(f"{name} in {PICKER} parsed to an empty list")
    return tokens


def test_the_guard_reads_a_real_file():
    """A scrape over an empty string would agree with anything."""
    source = _picker_source()
    assert len(source) > 500
    assert "export function classifyHook" in source


@pytest.mark.parametrize("name, authoritative", TOKEN_SETS)
def test_the_token_sets_match(name, authoritative):
    """Diverge and the panel offers an SAE the backend refuses, or hides one it accepts —
    and both files look right read on their own."""
    assert _ported_tokens(_picker_source(), name) == set(authoritative), (
        f"{name} has drifted: the authority is src/services/sae_hook_support.py"
    )


def test_the_picker_refuses_the_same_two_things_the_backend_does():
    """The kinds and the missing layer, named in both places."""
    source = _picker_source()
    for kind in (
        backend_rule.KIND_MLP,
        backend_rule.KIND_ATTENTION,
        backend_rule.KIND_EMBEDDING,
        backend_rule.KIND_RESID_PRE,
        backend_rule.KIND_RESID_MID,
    ):
        assert f"'{kind}'" in source, f"the picker no longer knows the kind {kind!r}"
    # And the layer half, which is the R3B-11 refusal the picker mirrors.
    assert "no layer recorded" in source
    assert hasattr(backend_rule, "unrecorded_layer_reason"), (
        "the backend no longer refuses a missing layer, so the picker should stop disabling "
        "those SAEs"
    )
