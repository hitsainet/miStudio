"""Random behaviour keys and bodies for tests (Feature 040, FPRD §3; FTID §8.2).

⚠ NO TEST NAMES A BEHAVIOUR. A suite whose fixtures all say one behaviour's key can stay green over code
that only works for that key (the generality failure 040 exists to prevent). Keys, names, sub-question
keys, criterion ids and column names come from here, seeded from the test's node id so a failure
reproduces with the same values (`rng_for(request.node.nodeid)`).

Only the pilot pack and the second-behaviour fixture under `tests/fixtures/behaviour_packs/` are fixed.
"""
from __future__ import annotations

import hashlib
import random
import re
import string
from typing import Any, Dict, Optional

from src.schemas.behaviour import KEY_PATTERN

_ALPHABET = string.ascii_lowercase + string.digits


def rng_for(seed: str) -> random.Random:
    """A generator seeded from a string (a test's node id), stable across processes."""
    return random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big"))


_DEFAULT = random.Random()


def slug(rng: Optional[random.Random] = None, *, words: int = 2, prefix: str = "") -> str:
    """A key matching `KEY_PATTERN`: `<prefix>` + `words` random 5-letter parts joined by hyphens."""
    rng = rng or _DEFAULT
    parts = ["".join(rng.choice(_ALPHABET) for _ in range(5)) for _ in range(words)]
    # A part must start with a letter so the key never starts with a digit-only word that reads as a
    # number in a column name; KEY_PATTERN itself allows a digit.
    parts = [rng.choice(string.ascii_lowercase) + p[1:] for p in parts]
    value = (prefix + "-".join(parts))[:64].strip("-")
    assert re.fullmatch(KEY_PATTERN, value), value
    return value


def words(rng: Optional[random.Random] = None, n: int = 3) -> str:
    rng = rng or _DEFAULT
    return " ".join(slug(rng, words=1) for _ in range(n)).capitalize()


def behaviour_body(rng: Optional[random.Random] = None, **overrides: Any) -> Dict[str, Any]:
    """A `BehaviourCreate` body with a random key and random prose; `overrides` replace fields."""
    rng = rng or _DEFAULT
    body: Dict[str, Any] = {
        "key": slug(rng),
        "name": words(rng),
        "description": None,
        "positive_meaning": words(rng, 6),
        "negative_meaning": words(rng, 6),
        "target_fpr": None,
    }
    body.update(overrides)
    return body
