"""Turning dataset rows into labelled examples (032 FR-1, FR-3, FR-5).

Four decisions live here, each extracted as a pure function so it can be tested
exhaustively and asserted by AST at its call site:

  `parse_input`      plain text vs native messages vs a JSON-STRING of messages
  `map_label`        a raw column value → positive / negative / excluded / unmapped
  `filter_rows`      which row INDICES survive a keyword filter
  `split_rows`       train / validation, without splitting a contrastive pair

⚠ THE FILTER RETURNS INDICES, AND THAT SIGNATURE IS THE GUARANTEE (BR-003).
`filter_rows` cannot express a label because it does not return rows or labels — it
returns the positions that survive. A filter that could say "rows containing
'urgent' are positive" would produce an AUROC measuring the keyword, and it would
look like a result. `test_probe_monitor_inputs` mutates the function to assign a
label and requires the suite to go red.

⚠ AN UNMAPPED VALUE IS A COUNTED REFUSAL, NEVER A SILENT DROP. `map_label` returns
`None` for a value the mapping does not name, and `build_examples` counts those
rows as `unparseable` rather than omitting them. A row that disappears between the
input and the counts is how an AUROC over 80 rows gets reported as one over 100 —
the same class of error as the padding that filled 48% of every SAE batch here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.conversation_formats import (
    ConversationFormat,
    detect_conversation_format,
    extract_messages_from_conversation,
)

Message = Dict[str, str]
#: What `parse_input` reports about HOW it read a value, so a run can log the mix
#: rather than assume one. A corpus that is 90% plain text and 10% JSON-string chat
#: is a corpus where a template change affects a tenth of the rows.
#: "plain" | "messages" | "json_messages" | "messages_roles_guessed" | "unparseable"
#:
#: ⚠ `messages_roles_guessed` IS NOT A COSMETIC DISTINCTION. A bare list of strings
#: (`["...", "..."]`, the estate's `SIMPLE_LIST` format) carries no roles at all, so
#: they are assigned BY POSITION — first turn user, second assistant, alternating. If
#: the source does not actually alternate, an `assistant`-scoped probe reads the
#: user's text under the wrong name, which is precisely the failure the role mask
#: exists to prevent. Such a row is usable (refusing it would discard a whole column
#: for a knowable reason), but only under scope `all`, and the run reports how many
#: rows are in that state.
InputKind = str


@dataclass
class ParsedInput:
    messages: List[Message]
    kind: InputKind

    @property
    def ok(self) -> bool:
        return self.kind != "unparseable" and bool(self.messages)

    @property
    def roles_are_known(self) -> bool:
        """False when the roles were assigned by position rather than stated.

        A row whose roles are guessed may only be scored under scope `all`; the
        render pass marks it `role_mask_unreliable` for the same reason it marks a
        template that rewrites earlier turns.
        """
        return self.kind != "messages_roles_guessed"


@dataclass
class Example:
    """One labelled example, ready for rendering."""

    index: int                 # position in the source dataset, for reproducibility
    messages: List[Message]
    label: int                 # 1 positive, 0 negative
    pair_id: Optional[str] = None
    kind: InputKind = "plain"


@dataclass
class BuildCounts:
    """Exactly the five numbers `probe_monitor_datasets.counts` stores."""

    positive: int = 0
    negative: int = 0
    excluded: int = 0
    filtered_out: int = 0
    unparseable: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "excluded": self.excluded,
            "filtered_out": self.filtered_out,
            "unparseable": self.unparseable,
        }

    @property
    def total(self) -> int:
        return (
            self.positive + self.negative + self.excluded
            + self.filtered_out + self.unparseable
        )


@dataclass
class BuildResult:
    examples: List[Example]
    counts: BuildCounts
    kinds: Dict[str, int] = field(default_factory=dict)


def parse_input(value: Any) -> ParsedInput:
    """Read one input cell as a conversation.

    Three shapes appear in real probe datasets and the third is the trap:

      1. a list of dicts — native messages
      2. a plain string — one user turn
      3. a STRING CONTAINING JSON — `'[{"role": "user", ...}]'`, which is what a
         CSV round trip or a `datasets` column typed as string produces

    Missing the third case does not fail. It renders the literal characters
    `[{"role": "user"...` as a user message, so the probe trains on JSON syntax and
    the run completes with plausible-looking numbers. That is the same silent-wrong
    shape as tokenizing a `conversations` column as its repr, which happened here on
    2026-09-12 and reported READY at "99.8% real tokens".
    """
    if value is None:
        return ParsedInput([], "unparseable")

    # Native messages (or another recognised conversation layout, e.g. ShareGPT).
    if isinstance(value, (list, tuple)):
        messages, guessed = _messages_from_sequence(list(value))
        if not messages:
            return ParsedInput([], "unparseable")
        return ParsedInput(messages, "messages_roles_guessed" if guessed else "messages")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ParsedInput([], "unparseable")
        # A JSON array of message dicts, arriving as a string.
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except (ValueError, TypeError):
                decoded = None
            if isinstance(decoded, list):
                messages, guessed = _messages_from_sequence(decoded)
                if messages:
                    return ParsedInput(
                        messages,
                        "messages_roles_guessed" if guessed else "json_messages",
                    )
                # A JSON list that is NOT messages (e.g. a list of numbers) is not
                # silently re-read as prose: `'[1, 2, 3]'` as a user turn is a
                # probe trained on digits and brackets.
                return ParsedInput([], "unparseable")
        return ParsedInput([{"role": "user", "content": value}], "plain")

    # Anything else (a dict, a number) is refused rather than str()-ed. `str(dict)`
    # is how a raw list-of-dicts became 490 blocks of `<|im_end|>` here.
    return ParsedInput([], "unparseable")


def _messages_from_sequence(items: Sequence[Any]) -> Tuple[List[Message], bool]:
    """(messages, roles_were_guessed). Empty list means the value is not a conversation.

    The flag is what separates a `SIMPLE_LIST` — a bare list of strings whose roles
    are assigned by position — from a layout that STATES its roles. Both are usable;
    only one supports a role-scoped probe.
    """
    if not items:
        return [], False
    # `detect_conversation_format` takes a list of COLUMN VALUES (each value being
    # one conversation) plus the column name, and returns a ConversationColumnInfo —
    # not a bare format. One conversation is therefore a one-element sample.
    info = detect_conversation_format([list(items)], "input")
    if info.format in (ConversationFormat.UNKNOWN, ConversationFormat.NOT_CONVERSATION):
        return [], False
    messages = extract_messages_from_conversation(list(items), info.format)
    cleaned = [
        {"role": str(m["role"]), "content": str(m["content"])}
        for m in messages
        if m.get("role") and m.get("content") is not None
    ]
    return cleaned, info.format == ConversationFormat.SIMPLE_LIST


def map_label(value: Any, mapping: Mapping[str, str]) -> Optional[str]:
    """A raw column value → "positive" | "negative" | "excluded", or None if unmapped.

    Compared as STRINGS because a label column arrives as `True`, `1`, `"1"` or
    `"high"` depending on the source and the Arrow dtype, and a mapping written in
    the UI is always strings. `None` (an unmapped value) is a counted refusal, not a
    drop — see the module docstring.
    """
    if value is None:
        return mapping.get("None") or mapping.get("null")
    key = str(value)
    if key in mapping:
        return mapping[key]
    # Booleans round-trip through several spellings; try the canonical ones before
    # giving up, so a mapping of {"true": ...} matches a numpy bool_.
    lowered = key.lower()
    if lowered in mapping:
        return mapping[lowered]
    if isinstance(value, bool) or lowered in ("true", "false"):
        alternative = "1" if lowered == "true" else "0"
        if alternative in mapping:
            return mapping[alternative]
    return None


def filter_rows(values: Sequence[str], spec: Optional[Mapping[str, Any]]) -> List[int]:
    """The INDICES that survive a keyword filter. It cannot label; see the module docstring.

    `mode="any"` keeps a row containing at least one term, `"all"` keeps a row
    containing every term. With no spec every index survives, which is what makes
    the filter genuinely optional rather than a default narrowing nobody chose.
    """
    if not spec:
        return list(range(len(values)))
    terms = [t for t in (spec.get("terms") or []) if str(t).strip()]
    if not terms:
        return list(range(len(values)))
    case_sensitive = bool(spec.get("case_sensitive", False))
    mode = spec.get("mode", "any")
    if mode not in ("any", "all"):
        raise ValueError(f"unknown filter mode {mode!r}; use 'any' or 'all'")

    if not case_sensitive:
        terms = [t.lower() for t in terms]

    kept: List[int] = []
    for i, raw in enumerate(values):
        haystack = raw if case_sensitive else raw.lower()
        hits = (term in haystack for term in terms)
        if (all(hits) if mode == "all" else any(hits)):
            kept.append(i)
    return kept


def build_examples(
    inputs: Sequence[Any],
    labels: Sequence[Any],
    label_mapping: Mapping[str, str],
    *,
    keyword_filter: Optional[Mapping[str, Any]] = None,
    pair_values: Optional[Sequence[Any]] = None,
) -> BuildResult:
    """Rows → labelled examples, with every discarded row accounted for.

    ORDER MATTERS AND IS DELIBERATE: parse, then map, then filter. Filtering first
    would make `filtered_out` include rows that were never parseable, so the counts
    would hide a broken column behind a narrow filter.
    """
    if not (len(inputs) == len(labels)):
        raise ValueError(f"{len(inputs)} inputs against {len(labels)} labels")
    if pair_values is not None and len(pair_values) != len(inputs):
        raise ValueError(f"{len(pair_values)} pair values against {len(inputs)} inputs")

    counts = BuildCounts()
    kinds: Dict[str, int] = {}

    # Parse and map first, keeping the surviving positions so the filter can be
    # applied over the same index space.
    staged: List[Tuple[int, ParsedInput, str]] = []
    for i, (raw_input, raw_label) in enumerate(zip(inputs, labels)):
        target = map_label(raw_label, label_mapping)
        if target is None:
            counts.unparseable += 1
            kinds["unmapped_label"] = kinds.get("unmapped_label", 0) + 1
            continue
        if target == "excluded":
            counts.excluded += 1
            continue
        parsed = parse_input(raw_input)
        kinds[parsed.kind] = kinds.get(parsed.kind, 0) + 1
        if not parsed.ok:
            counts.unparseable += 1
            continue
        if not parsed.roles_are_known:
            kinds["roles_guessed"] = kinds.get("roles_guessed", 0) + 1
        staged.append((i, parsed, target))

    # The filter reads the RENDERED-FREE text of the surviving rows only.
    texts = [" ".join(m["content"] for m in parsed.messages) for _, parsed, _ in staged]
    kept = set(filter_rows(texts, keyword_filter))
    examples: List[Example] = []
    for position, (index, parsed, target) in enumerate(staged):
        if position not in kept:
            counts.filtered_out += 1
            continue
        if target == "positive":
            counts.positive += 1
        else:
            counts.negative += 1
        pair_id = None
        if pair_values is not None and pair_values[index] is not None:
            pair_id = str(pair_values[index])
        examples.append(
            Example(
                index=index,
                messages=parsed.messages,
                label=1 if target == "positive" else 0,
                pair_id=pair_id,
                kind=parsed.kind,
            )
        )
    return BuildResult(examples=examples, counts=counts, kinds=kinds)


def split_rows(
    examples: Sequence[Example], *, val_fraction: float, seed: int
) -> Tuple[List[Example], List[Example]]:
    """Train / validation, keeping every contrastive PAIR on one side.

    ⚠ A PAIR SPLIT ACROSS THE BOUNDARY LEAKS, AND LEAKS UPWARD. Contrastive data
    ships two near-identical rows differing in the thing being detected; with one in
    train and one in validation the validation AUROC measures recall of a memorised
    passage. That inflates the number the LAYER SELECTION reads, so the leak does not
    merely flatter the report — it chooses the layer.

    Assignment is by a HASH of the group key, not by shuffling: a hash is stable
    under row reordering, under a re-download that changes row order, and across
    processes, so the same (seed, key) always lands on the same side. A shuffle with
    a seeded RNG is reproducible only if the input order is too, and this estate has
    already shipped one unseeded shuffle that made a template "never the only
    variable".
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    train: List[Example] = []
    validation: List[Example] = []
    for example in examples:
        # A row without a pair is its own group, keyed on its source index so two
        # unpaired rows never share a group by accident.
        key = example.pair_id if example.pair_id is not None else f"__row_{example.index}"
        digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
        # 8 hex characters = 32 bits: enough resolution that the realised fraction
        # tracks the requested one, and stable across Python versions (`hash()` is
        # salted per process and must never be used for this).
        position = int(digest[:8], 16) / 0xFFFFFFFF
        (validation if position < val_fraction else train).append(example)
    return train, validation


def class_counts(examples: Sequence[Example]) -> Tuple[int, int]:
    """(positives, negatives) — what a 422 has to name."""
    positive = sum(1 for e in examples if e.label == 1)
    return positive, len(examples) - positive
