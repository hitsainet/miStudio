"""
Token normalization for cross-feature grouping.

Tokenizer surface forms differ across models (SentencePiece ``▁``, GPT-2 BPE
``Ġ``, WordPiece ``##``) and across positions (case, surrounding punctuation).
Grouping features "by top activating token" requires a canonical form so that
``▁Love``, ``Ġlove`` and ``love`` land in the same bucket.
"""

import unicodedata

# Leading sub-word markers by tokenizer family
_BPE_LEADING = ("▁", "Ġ")

# Punctuation stripped only from the token's edges (interior kept: "don't")
_EDGE_PUNCT = "\"'`.,;:!?()[]{}<>-—–…·*_~^\\/|"


def normalize_token(raw: str) -> str | None:
    """Normalize a tokenizer surface form for cross-feature matching.

    Returns ``None`` when nothing meaningful remains (pure punctuation or
    marker-only tokens), which excludes the token from grouping.
    """
    if raw is None:
        return None
    t = raw
    for marker in _BPE_LEADING:
        if t.startswith(marker):
            t = t[len(marker):]
            break
    if t.startswith("##"):
        t = t[2:]
    if t.endswith("##"):
        t = t[:-2]
    t = unicodedata.normalize("NFKC", t).lower().strip()
    t = t.strip(_EDGE_PUNCT).strip()
    return t or None


def normalize_bag(tokens: list[str] | None) -> list[str]:
    """Normalize a list of context tokens, dropping empties. Order preserved."""
    if not tokens:
        return []
    out = []
    for tok in tokens:
        norm = normalize_token(tok)
        if norm:
            out.append(norm)
    return out
