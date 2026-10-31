"""THE validator registry of miStudio's rule engine (Feature 040 FR-27; the producer's FR-016.1/016.2).

``VALIDATORS`` is keyed by the exact ``(id, version)`` and mirrors miDataworks' registered ids and versions
one for one. A rule names an exact version and :func:`resolve` returns exactly that version, never "the
latest"; a version this engine does not know refuses the rule set at load, naming it (FTDD §6.1).
``test_rule_signals_conformance.py`` fails on a registered validator without shared vectors.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple

from .algorithms import GENERIC_VALIDATORS
from .base import Candidate, Validator, ValidatorDecl, ValidatorUnknown, Verdict
from .formats import FORMAT_VALIDATORS
from .masked import MASKED

__all__ = ["VALIDATORS", "Candidate", "Validator", "ValidatorDecl", "ValidatorUnknown", "Verdict",
           "registered", "resolve", "superseded_by"]

_ALL: Tuple[Validator, ...] = (*FORMAT_VALIDATORS, *GENERIC_VALIDATORS, MASKED)
VALIDATORS: Dict[Tuple[str, int], Validator] = {(v.decl.id, v.decl.version): v for v in _ALL}


def registered() -> FrozenSet[str]:
    """Every registered ``id@version``."""
    return frozenset(f"{i}@{v}" for i, v in VALIDATORS)


def resolve(validator_id: str, version: int) -> Validator:
    found = VALIDATORS.get((validator_id, int(version)))
    if found is None:
        raise ValidatorUnknown(validator_id, int(version))
    return found


def superseded_by(validator_id: str, version: int) -> Optional[int]:
    later = [v for (i, v) in VALIDATORS if i == validator_id and v > version]
    return min(later) if later else None
