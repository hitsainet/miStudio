"""Generate falsifiable correctness probes for circuit calibration (IDL-37).

The correctness cliff — the strength above which the model produces confidently
FALSE output — is invisible to perplexity/theme metrics. Finding it needs a
probe whose right answer is falsifiable, judged by an LLM.

The load-bearing design choice: probes are on NEUTRAL factual topics the circuit
should NOT influence, NOT on the circuit's own concept. A humor circuit's probe
must be a plain factual question ("What is the capital of France?"), because
degradation then shows up as the humor tint corrupting an unrelated fact — the
exact empirical signature ("an Irish wedding honors the deceased"). A probe
*about* humor has no falsifiable answer, so the judge could never detect the
cliff. Because they are generated (not authored) the resulting band is
`provisional`.

The LLM call is injected so this module is unit-testable without a live
endpoint; production passes an `EnhancedLabelingService._call_llm`-shaped callable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)

#: A deterministic fallback used when the LLM is unavailable or returns garbage.
#: These are neutral by construction — no circuit concept should touch basic
#: geography/arithmetic — so calibration still runs (provisional) offline.
_FALLBACK_PROBES: List[Dict[str, str]] = [
    {"prompt": "What is the capital of France?", "expected": "Paris"},
    {"prompt": "What is 12 multiplied by 8?", "expected": "96"},
    {"prompt": "How many days are there in a week?", "expected": "Seven (7)"},
]


def _concept(member_labels: List[str]) -> str:
    """A short human phrase for the circuit's concept, from its member labels."""
    labels = [str(x).strip() for x in member_labels if str(x).strip()]
    # De-dup preserving order; cap so the prompt stays small.
    seen, out = set(), []
    for lb in labels:
        key = lb.lower()
        if key not in seen:
            seen.add(key)
            out.append(lb)
    return ", ".join(out[:8]) or "an unknown concept"


def _build_prompt(concept: str, n: int) -> str:
    return (
        f"A model is being steered toward the concept: {concept}.\n"
        f"Write {n} general-knowledge questions whose subject is COMPLETELY "
        f"UNRELATED to that concept — plain facts (geography, arithmetic, "
        f"science, history) with a single verifiable correct answer.\n"
        f"The questions must NOT be about {concept} in any way: we use them to "
        f"detect when steering corrupts unrelated facts.\n"
        f'Return ONLY a JSON array: [{{"prompt": "...", "expected": "..."}}]. '
        f"Keep each expected answer to a few words."
    )


def _extract_array(raw: str):
    """Find the first JSON ARRAY OF OBJECTS in the text, tolerant of stray
    brackets in the surrounding prose (a greedy [...] span breaks on '[1]'
    footnotes or trailing '[end]' notes; a naive first-array match would return
    the footnote '[1]'). Scans each '[', raw_decodes, and prefers an array that
    actually contains dicts — falling back to any list only if none qualifies.
    """
    decoder = json.JSONDecoder()
    first_list = None
    for i, ch in enumerate(raw):
        if ch != "[":
            continue
        try:
            obj, _end = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            if any(isinstance(x, dict) for x in obj):
                return obj                      # the real probe array
            if first_list is None:
                first_list = obj                # remember, but keep looking
    if first_list is not None:
        return first_list
    raise ValueError("no JSON array in probe-generator response")


def _parse(raw: str) -> List[Dict[str, str]]:
    """Extract the JSON array from an LLM response; tolerant of code fences and
    stray brackets in surrounding prose."""
    data = _extract_array(raw)
    probes: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).strip()
        expected = str(item.get("expected", "")).strip()
        if prompt and expected:
            probes.append({"prompt": prompt, "expected": expected})
    if not probes:
        raise ValueError("probe-generator response had no usable {prompt, expected} items")
    return probes


#: Generic words that carry no concept signal. A label token in this set is NOT
#: used to flag a probe on-concept — otherwise a neutral geography/science probe
#: that happens to contain "language"/"state"/"time" would be wrongly dropped,
#: and if a common word recurred it could nuke EVERY probe and silently defeat
#: the LLM generator. Distinctive short words ("pun", "war", "wit") are NOT here,
#: so they still match — which the old len>=4 gate wrongly skipped.
_STOPWORDS = frozenset("""
a an the of to in on at for and or but with without within into onto from by as
is are was were be been being do does did has have had will would can could may
might must should this that these those it its their his her they them we you i
about above after again against all am any because before below between both
each few more most other some such only own same so than too very s t just
language languages word words phrase phrases text texts term terms tone style
reference references context contexts pattern patterns kind type form forms
sense meaning related informal formal general specific common usage use used
character characters situation situations thing things part parts group groups
feature features concept concepts topic topics subject content example examples
""".split())

#: A label token must be at least this long to be considered a concept signal —
#: but ONLY as a floor below which even non-stopwords are too generic to trust.
#: Kept at 3 so genuine short concepts ("pun", "gag", "war", "wit") still count.
_MIN_CONCEPT_LEN = 3


def _concept_tokens(member_labels: List[str]) -> set:
    """The distinctive tokens that identify the circuit's concept — label words
    that are neither stopwords nor too short."""
    toks = set()
    for label in member_labels:
        for tok in re.split(r"[^a-z0-9]+", str(label).lower()):
            if len(tok) >= _MIN_CONCEPT_LEN and tok not in _STOPWORDS:
                toks.add(tok)
    return toks


def _looks_on_concept(probe: Dict[str, str], member_labels: List[str]) -> bool:
    """True if the probe mentions a DISTINCTIVE concept token (whole word) —
    which would make it non-falsifiable for cliff detection.

    Whole-word match against the concept tokens only (stopwords excluded), so a
    neutral probe containing a generic label word is NOT dropped, while a probe
    genuinely about the concept (including short concepts like "pun") IS.
    """
    concept = _concept_tokens(member_labels)
    if not concept:
        return False
    hay = (probe.get("prompt", "") + " " + probe.get("expected", "")).lower()
    hay_tokens = set(re.split(r"[^a-z0-9]+", hay))
    return bool(concept & hay_tokens)


def generate_probes(
    member_labels: List[str],
    *,
    llm_call: Callable[[str, int], str] | None = None,
    n: int = 3,
) -> List[Dict[str, str]]:
    """Return n neutral-topic falsifiable probes for the circuit.

    `llm_call(prompt, max_tokens) -> str` is injected (production:
    `EnhancedLabelingService._call_llm`). On any failure — no LLM, bad JSON, or
    every generated probe landing ON the concept — falls back to the neutral
    static set so calibration still runs. The band is provisional regardless.
    """
    concept = _concept(member_labels)
    if llm_call is None:
        logger.info("No LLM for probe generation; using neutral fallback probes.")
        return _FALLBACK_PROBES[:n]
    try:
        raw = llm_call(_build_prompt(concept, n), 600)
        probes = _parse(raw)
    except Exception:
        logger.exception("Probe generation failed; using neutral fallback probes.")
        return _FALLBACK_PROBES[:n]

    # Drop any probe that drifted ONTO the concept — those can't detect the
    # cliff. If that empties the set, fall back rather than calibrate blind.
    neutral = [p for p in probes if not _looks_on_concept(p, member_labels)]
    if not neutral:
        logger.warning(
            "Every generated probe was on-concept for %r; using neutral "
            "fallback probes instead.", concept)
        return _FALLBACK_PROBES[:n]
    return neutral[:n]
