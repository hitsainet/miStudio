"""The blend candidate vocabulary (Feature 041 FR-8; FTID §3.6).

One `CandidateKind` per name. ⚠ THE REGISTRY IS THE VOCABULARY: `expand`, `parse_name`, the refusal that
lists the vocabulary and the completeness test all read `REGISTRY`, so no second list can drift from it.

* score kinds (`stack`, `max`, `best_single`) produce a per-row value that one global threshold cuts
  (calibrated on the calibration negatives at the target false-positive rate, OQ-3 (a));
* verdict kinds (`any_of`, `all_of`, `k_of_n:k`) count how many members fire at their OWN recorded
  decision (040's `deployed_decision`) and fire at `n_fired >= k` (any_of: k = 1; all_of: k = n);
* `rules_or_<name>` wraps any of them with the behaviour version's rule set (rules ∪ candidate), which
  needs 040 S3's rule engine and union calibration.

Order is FIXED — registry order, then k ascending, then the rules wrappers — so candidate indices (the
`c<NN>` file names) are stable between two evaluations of the same request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

RULES_PREFIX = "rules_or_"


@dataclass(frozen=True)
class CandidateKind:
    name: str
    produces: Literal["score", "verdict"]
    #: `k_of_n` takes `:k`.
    parametrised: bool = False


REGISTRY: Dict[str, CandidateKind] = {
    kind.name: kind
    for kind in (
        CandidateKind("stack", "score"),
        CandidateKind("max", "score"),
        CandidateKind("any_of", "verdict"),
        CandidateKind("all_of", "verdict"),
        CandidateKind("k_of_n", "verdict", parametrised=True),
        CandidateKind("best_single", "score"),
    )
}

#: The default request: every kind.
DEFAULT_KINDS: Tuple[str, ...] = tuple(REGISTRY)


class CandidateUnknown(ValueError):
    """A name outside the vocabulary (`candidate_unknown`)."""

    def __init__(self, name: str, vocabulary: List[str]):
        super().__init__(f"{name!r} is not a blend candidate; the vocabulary is {vocabulary}")
        self.name = name
        self.vocabulary = vocabulary


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    produces: str
    k: Optional[int]
    rules: bool


def vocabulary(n_members: int, has_rule_set: bool) -> List[str]:
    """Every concrete name the registry can produce for `n_members`, in the fixed order."""
    return expand(DEFAULT_KINDS, n_members, has_rule_set)


def expand(kinds: Sequence[str], n_members: int, has_rule_set: bool) -> List[str]:
    """The concrete candidate names for `kinds` (registry kind names), fixed order."""
    unknown = [k for k in kinds if k not in REGISTRY]
    if unknown:
        raise CandidateUnknown(unknown[0], sorted(REGISTRY))
    names: List[str] = []
    for kind in REGISTRY.values():
        if kind.name not in kinds:
            continue
        if kind.parametrised:
            names += [f"{kind.name}:{k}" for k in range(2, n_members)]
        else:
            names.append(kind.name)
    if has_rule_set:
        names += [RULES_PREFIX + name for name in list(names)]
    return names


def parse_name(name: str, n_members: int, has_rule_set: bool = True) -> Candidate:
    """`stack`, `k_of_n:3`, `rules_or_any_of`, … → a `Candidate`; anything else → `CandidateUnknown`."""
    rules = name.startswith(RULES_PREFIX)
    base = name[len(RULES_PREFIX):] if rules else name
    kind_name, _, param = base.partition(":")
    kind = REGISTRY.get(kind_name)
    allowed = vocabulary(n_members, has_rule_set)
    if kind is None or name not in allowed:
        raise CandidateUnknown(name, allowed)
    k: Optional[int] = None
    if kind.parametrised:
        k = int(param)
    elif kind.name == "any_of":
        k = 1
    elif kind.name == "all_of":
        k = n_members
    return Candidate(name=name, kind=kind.name, produces=kind.produces, k=k, rules=rules)


def resolve(requested: Optional[Sequence[str]], n_members: int, has_rule_set: bool) -> List[Candidate]:
    """The request's candidates (names or kind names), expanded and parsed, in the fixed order."""
    if requested is None:
        names = vocabulary(n_members, has_rule_set)
    else:
        wanted: List[str] = []
        for item in requested:
            if item in REGISTRY:
                wanted += expand([item], n_members, False)
            else:
                wanted.append(item)
        allowed = vocabulary(n_members, has_rule_set)
        for name in wanted:
            parse_name(name, n_members, has_rule_set)       # refuses outside the vocabulary
        names = [name for name in allowed if name in set(wanted)]
    return [parse_name(name, n_members, has_rule_set) for name in names]
