"""A seeded generator of NONSENSE TEXT for the rung-2 nonsense-text check (ENH-006 D84, 2026-10-29).

Why: the everyday-text check (D82) passed the layer-0 `mean` probe of `pmr_bbe9a3608802` at 0.35% false
alarms on OpenWebText, while that probe fired on 280/280 of miLLM's benchmark chats — random common words
strung into sentences. Ordinary prose and text with no meaning are different failure surfaces, so rung 2
also needs few false alarms on a nonsense-text set. This module makes one without a GPU and without a
download: every row is generated from `random.Random(seed)`, so the same `(rows, seed, prose)` gives the
same rows, byte for byte.

THE KINDS (column `kind`):

* ``word_salad``        — common words in random order, sentence-cased and punctuated (miLLM's benchmark
                          `make_prompt`, with a larger list);
* ``random_characters`` — letters, digits-free punctuation and spaces in random runs;
* ``repeated_token``    — one token repeated: a word, a punctuation mark or a single letter;
* ``shuffled_prose``    — sentences of REAL prose from a local corpus, shuffled across documents. Only when a
                          prose corpus is given; otherwise the kind is skipped and `skipped` says why;
* ``garbled_code``      — keywords, identifiers, operators and brackets of several languages in random order;
* ``digit_noise``       — groups of digits with separators, NEVER SSN-shaped (below);
* ``mixed_scripts``     — characters drawn from Latin, Greek, Cyrillic, Hebrew, Arabic, Devanagari, Thai, CJK,
                          Hangul and emoji in one string.

Lengths run from one or two tokens to about 4,000 (log-uniform, so short rows are as common as long ones).
Token counts are approximate: a word is ~1.35 tokens, a random character ~0.5-1, a CJK character 1-2.

⚠ NEVER SSN-SHAPED. The probes this check exists for detect US Social Security numbers, so a nonsense row
that contains one would be a POSITIVE read as a negative. Two guarantees, the second a backstop to the first:

1. by construction in `digit_noise`, the only kind that emits digits on purpose: digit groups are at most 8
   digits long and every separator is at least 4 non-digit characters. Any SSN shape needs 9 digits within
   a window whose gaps are at most 3 non-digits, so it would have to sit inside ONE group — and no group
   has 9 digits;
2. a final pass over EVERY row of EVERY kind (`defuse_ssn_shapes`) that breaks any run of 9 or more digits
   and any 3-2-4 digit grouping with 0-3 non-digit characters between the groups (`SSN_SHAPED` — broader
   than the formats a detector reads, on purpose), replacing one digit with a letter until none is left.
   Prose and code can carry numbers; this pass is what makes the guarantee hold for them too. `\\d` is
   Unicode-aware, so digits of other scripts count.

Tests: `test_nonsense_text.py` (determinism per seed, the guarantee over many seeds, the defuser on crafted
inputs, the kinds and lengths). Negative control: make `defuse_ssn_shapes` a no-op or let `digit_noise` emit
a 3-2-4 grouping -> red.
"""
from __future__ import annotations

import math
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

GENERATOR_ID = "mistudio.nonsense-text/v1"

KINDS = (
    "word_salad",
    "random_characters",
    "repeated_token",
    "shuffled_prose",
    "garbled_code",
    "digit_noise",
    "mixed_scripts",
)

#: Lengths in TOKENS (approximate): log-uniform between these.
MIN_TOKENS = 1
MAX_TOKENS = 4000
TOKENS_PER_WORD = 1.35

#: miLLM's benchmark list (`tests/hardware/otel_acceptance.py` WORDS) plus more common words.
COMMON_WORDS = (
    "river", "garden", "morning", "quiet", "window", "lantern", "harbor", "meadow", "pencil",
    "orchard", "valley", "station", "kitchen", "library", "bridge", "canvas", "thunder", "ladder",
    "planet", "market", "pocket", "forest", "candle", "basket", "mirror", "island", "village",
    "summer", "autumn", "winter", "spring", "teacher", "painter", "farmer", "sailor", "doctor",
    "walks", "builds", "carries", "watches", "follows", "gathers", "paints", "reads", "writes",
    "slowly", "gently", "often", "always", "under", "beside", "across", "toward", "through",
    "bright", "simple", "careful", "patient", "golden", "silver", "narrow", "distant", "early",
    "the", "a", "of", "and", "to", "in", "is", "it", "that", "was", "for", "on", "are", "with", "as",
    "his", "they", "at", "be", "this", "from", "have", "or", "by", "one", "had", "not", "but", "what",
    "all", "were", "when", "we", "there", "can", "an", "your", "which", "their", "said", "if", "do",
    "will", "each", "about", "how", "up", "out", "them", "then", "she", "many", "some", "so", "these",
    "would", "other", "into", "has", "more", "her", "two", "like", "him", "see", "time", "could", "no",
    "make", "than", "first", "been", "its", "who", "now", "people", "my", "made", "over", "did", "down",
    "only", "way", "find", "use", "may", "water", "long", "little", "very", "after", "words", "called",
    "just", "where", "most", "know", "get", "through", "back", "much", "before", "go", "good", "new",
    "write", "our", "used", "me", "man", "too", "any", "day", "same", "right", "look", "think", "also",
    "around", "another", "came", "come", "work", "three", "word", "must", "because", "does", "part",
    "even", "place", "well", "such", "here", "take", "why", "things", "help", "put", "years", "different",
    "away", "again", "off", "went", "old", "number", "great", "tell", "men", "say", "small", "every",
    "found", "still", "between", "name", "should", "home", "big", "give", "air", "line", "set", "own",
    "under", "read", "last", "never", "us", "left", "end", "along", "while", "might", "next", "sound",
    "below", "saw", "something", "thought", "both", "few", "those", "show", "large", "often", "together",
)

CODE_TOKENS = (
    "def", "return", "class", "import", "from", "if", "else", "elif", "for", "while", "in", "not", "and",
    "or", "None", "True", "False", "lambda", "yield", "async", "await", "try", "except", "finally", "with",
    "function", "const", "let", "var", "=>", "this", "new", "typeof", "null", "undefined", "public",
    "private", "static", "void", "int", "char", "struct", "#include", "fn", "mut", "impl", "match",
    "SELECT", "FROM", "WHERE", "JOIN", "GROUP", "BY", "{", "}", "(", ")", "[", "]", ";", ":", ",", ".",
    "=", "==", "!=", "<=", ">=", "+=", "->", "::", "&&", "||", "!", "?", "*", "/", "%", "<", ">", "&",
    "|", "^", "~", "@", "#", "$", "'", '"', "`", "\\n", "foo", "bar", "baz", "tmp", "val", "x", "y",
    "i", "j", "k", "result", "data", "self", "args", "kwargs", "obj", "node", "item", "buf", "len",
    "print", "console.log", "printf", "malloc", "free", "map", "filter", "reduce", "push", "pop",
    "0", "1", "2", "42", "0x1F", "3.14", "1e9", "-1",
)

#: Inclusive codepoint ranges, one per script; all are printable letters/marks/symbols.
SCRIPT_RANGES = (
    (0x0041, 0x005A), (0x0061, 0x007A),       # Latin
    (0x00C0, 0x00FF),                         # Latin-1 letters
    (0x0391, 0x03A9), (0x03B1, 0x03C9),       # Greek
    (0x0410, 0x044F),                         # Cyrillic
    (0x05D0, 0x05EA),                         # Hebrew
    (0x0627, 0x064A),                         # Arabic letters
    (0x0905, 0x0939),                         # Devanagari
    (0x0E01, 0x0E2E),                         # Thai
    (0x3041, 0x3096),                         # Hiragana
    (0x4E00, 0x4FFF),                         # CJK
    (0xAC00, 0xAD00),                         # Hangul
    (0x1F600, 0x1F64F),                       # emoji
)

#: Punctuation without digits; used by random characters and separators.
PUNCTUATION = "".join(c for c in string.punctuation)
#: Every one at least 4 non-digit characters: wider than the 0-3 gap `SSN_SHAPED` bridges (module note).
DIGIT_SEPARATORS = ("    ", " -- ", " :: ", " // ", ", , ", " | | ", "\n\n\n\n", " ... ", "____", " ; ; ")

#: The shapes a nonsense row must never carry (see the module note). Unicode `\d`.
SSN_SHAPED = re.compile(r"\d{9}|\d{3}\D{0,3}\d{2}\D{0,3}\d{4}")


def ssn_shaped(text: str) -> bool:
    """True when `text` holds a run of 9+ digits or a 3-2-4 digit grouping (0-3 non-digits between)."""
    return SSN_SHAPED.search(text) is not None


def defuse_ssn_shapes(text: str) -> str:
    """`text` with every SSN shape broken: one digit of each match replaced by a letter, until none is left.

    The replaced digit is the FIRST digit of the match's last group (the 9th digit of a run, or the first of
    the 4-digit group), so a 9-digit run becomes 8 + letter + rest and a 3-2-4 grouping loses its 4-group.
    Each pass removes one digit, so it terminates.
    """
    while True:
        match = SSN_SHAPED.search(text)
        if match is None:
            return text
        span = match.group(0)
        if len(span) >= 9 and span[:9].isdigit():
            offset = match.start() + 8
        else:
            # The last 4 characters of a 3-2-4 match are its 4-digit group.
            offset = match.end() - 4
        text = text[:offset] + "x" + text[offset + 1:]


@dataclass
class NonsenseSet:
    """The generated rows, with what each kind contributed and any kind that was skipped and why."""

    texts: List[str]
    kinds: List[str]
    seed: int
    counts: Dict[str, int] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)

    def manifest(self) -> Dict[str, object]:
        return {"generator": GENERATOR_ID, "seed": self.seed, "rows": len(self.texts),
                "kinds": dict(self.counts), "skipped": dict(self.skipped)}


def _target_tokens(rng: random.Random) -> int:
    return max(MIN_TOKENS, int(round(math.exp(rng.uniform(math.log(MIN_TOKENS + 1), math.log(MAX_TOKENS))))))


def _word_salad(rng: random.Random, tokens: int) -> str:
    n = max(1, int(tokens / TOKENS_PER_WORD))
    words = [rng.choice(COMMON_WORDS) for _ in range(n)]
    out: List[str] = []
    i = 0
    while i < len(words):
        size = rng.randint(3, 16)
        chunk = words[i:i + size]
        out.append(" ".join(chunk).capitalize() + rng.choice(".....!?;"))
        i += size
    return " ".join(out)


def _random_characters(rng: random.Random, tokens: int) -> str:
    alphabet = string.ascii_letters + PUNCTUATION + "     "
    return "".join(rng.choice(alphabet) for _ in range(max(1, tokens * 2)))


def _repeated_token(rng: random.Random, tokens: int) -> str:
    unit = rng.choice([rng.choice(COMMON_WORDS), rng.choice(PUNCTUATION), rng.choice(string.ascii_letters),
                       rng.choice(("ok", "lol", "ha", "na", "um", "the", "and", "...", "--", "xx"))])
    sep = rng.choice((" ", "", " ", "\n", ", "))
    return sep.join([unit] * max(1, tokens))


def _garbled_code(rng: random.Random, tokens: int) -> str:
    parts: List[str] = []
    for _ in range(max(1, tokens // 2)):
        parts.append(rng.choice(CODE_TOKENS))
        parts.append(rng.choice((" ", " ", " ", "", "\n", "\n    ", "\t")))
    return "".join(parts).strip() or rng.choice(CODE_TOKENS)


def _digit_noise(rng: random.Random, tokens: int) -> str:
    """Digit groups of 1-8 digits, separated by 4+ non-digits. Not SSN-shaped by construction (module note)."""
    parts: List[str] = []
    budget = max(1, tokens)
    while budget > 0:
        size = rng.randint(1, 8)
        parts.append("".join(rng.choice(string.digits) for _ in range(size)))
        parts.append(rng.choice(DIGIT_SEPARATORS))
        budget -= max(1, size // 2) + 1
    return "".join(parts).strip() or "0"


def _mixed_scripts(rng: random.Random, tokens: int) -> str:
    chars: List[str] = []
    for _ in range(max(1, int(tokens * 0.7))):
        lo, hi = rng.choice(SCRIPT_RANGES)
        chars.append(chr(rng.randint(lo, hi)))
        if rng.random() < 0.15:
            chars.append(" ")
    return "".join(chars).strip() or "ж"


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def prose_sentences(documents: Sequence[str], *, limit: int = 200_000) -> List[str]:
    """Sentences of `documents` (split on . ! ? followed by space), stripped, at most `limit`."""
    out: List[str] = []
    for document in documents:
        if not isinstance(document, str):
            continue
        for sentence in _SENTENCE_END.split(document):
            sentence = " ".join(sentence.split())
            if len(sentence) >= 8:
                out.append(sentence)
                if len(out) >= limit:
                    return out
    return out


def _shuffled_prose(rng: random.Random, tokens: int, sentences: Sequence[str]) -> str:
    out: List[str] = []
    budget = max(1, int(tokens / TOKENS_PER_WORD))
    while budget > 0:
        sentence = rng.choice(sentences)
        out.append(sentence)
        budget -= max(1, len(sentence.split()))
    return " ".join(out)


def generate(rows: int, seed: int, *, prose: Optional[Sequence[str]] = None) -> NonsenseSet:
    """`rows` nonsense rows from `seed`: every kind in near-equal shares, in a seeded random order.

    `prose`: documents of a real prose corpus for `shuffled_prose`; without it (or with fewer than 20
    sentences in it) that kind is skipped and `skipped` says so. Deterministic: the same arguments give the
    same rows.
    """
    if rows < 1:
        raise ValueError("rows must be at least 1")
    rng = random.Random(f"{GENERATOR_ID}:{int(seed)}")
    skipped: Dict[str, str] = {}
    sentences: List[str] = []
    if prose is None:
        skipped["shuffled_prose"] = "no prose corpus was given (prose_dataset_id), so no shuffled real sentences"
    else:
        sentences = prose_sentences(prose)
        if len(sentences) < 20:
            skipped["shuffled_prose"] = (f"the prose corpus gave {len(sentences)} sentences, fewer than the 20 "
                                         f"shuffling needs")
            sentences = []
    kinds = [k for k in KINDS if k not in skipped]
    order = [kinds[i % len(kinds)] for i in range(rows)]
    rng.shuffle(order)

    texts: List[str] = []
    counts: Dict[str, int] = {k: 0 for k in kinds}
    for kind in order:
        tokens = _target_tokens(rng)
        if kind == "word_salad":
            text = _word_salad(rng, tokens)
        elif kind == "random_characters":
            text = _random_characters(rng, tokens)
        elif kind == "repeated_token":
            text = _repeated_token(rng, tokens)
        elif kind == "shuffled_prose":
            text = _shuffled_prose(rng, tokens, sentences)
        elif kind == "garbled_code":
            text = _garbled_code(rng, tokens)
        elif kind == "digit_noise":
            text = _digit_noise(rng, tokens)
        else:
            text = _mixed_scripts(rng, tokens)
        texts.append(defuse_ssn_shapes(text))
        counts[kind] += 1
    return NonsenseSet(texts=texts, kinds=order, seed=int(seed), counts=counts, skipped=skipped)


def write_dataset(generated: NonsenseSet, directory: "Path") -> int:
    """Write `generated` where a download writes (`save_to_disk`): columns `text` and `kind`. Returns bytes.

    The same on-disk format as a downloaded dataset, so `register_probe_dataset` / `open_split` read it like
    any other. `directory` must not exist yet.
    """
    from datasets import Dataset

    data = Dataset.from_dict({"text": list(generated.texts), "kind": list(generated.kinds)})
    data.save_to_disk(str(directory))
    return int(sum(f.stat().st_size for f in Path(directory).rglob("*") if f.is_file()))
