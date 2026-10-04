"""Refuse a tokenization that produced a degenerate corpus.

WHY THIS EXISTS. `teknium/OpenHermes-2.5` tokenized to 490 blocks in which
**every single token was `<|im_end|>`** — `unique_tokens_used: 1` across
1,001,551 source documents. The row went READY, the worker logged
"99.8% real tokens", and the statistics block reported a tidy
`length_distribution` with all 490 rows in the `1000+` bucket.

Every one of those numbers was true. The attention mask counts EOS separators as
real tokens, so occupancy really was 99.8%; the blocks really were full; the
documents really were all processed. Nothing in the success path is capable of
noticing that the corpus carries no information.

That is the failure mode this repo keeps recording: not a crash, not an absence,
but a plausible-looking number over data that is silently wrong. A corpus of one
repeated token would have trained an SAE to convergence with an excellent FVU,
and the first sign of trouble would have been uninterpretable features weeks
later.

The check is deliberately crude. It is not trying to assess corpus quality —
only to catch the case where tokenization produced something no one could have
wanted, and to say so instead of reporting success.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["DegenerateCorpusError", "check_token_diversity", "MIN_UNIQUE_TOKENS"]

#: Below this many distinct token ids, a corpus of any real size is broken.
#: Natural text in any language clears this within a few hundred tokens; the
#: observed failure produced exactly 1. Set low on purpose — this is a
#: catastrophe detector, not a quality bar.
MIN_UNIQUE_TOKENS = 16

#: Corpora smaller than this are exempt: a deliberate tiny fixture or a
#: single-document smoke test can legitimately carry very few distinct tokens.
MIN_TOKENS_TO_JUDGE = 10_000


class DegenerateCorpusError(ValueError):
    """Tokenization completed and produced something unusable."""


def check_token_diversity(
    unique_tokens: Optional[int],
    total_tokens: Optional[int],
    *,
    context: str = "",
) -> None:
    """Raise when a corpus large enough to judge carries almost no distinct tokens.

    Silent on unknown inputs. A statistics block that failed to compute is a
    different problem, and turning "I don't know" into "this is broken" would
    fail runs for the wrong reason.
    """
    if unique_tokens is None or total_tokens is None:
        return
    if total_tokens < MIN_TOKENS_TO_JUDGE:
        return
    if unique_tokens >= MIN_UNIQUE_TOKENS:
        return

    where = f" ({context})" if context else ""
    raise DegenerateCorpusError(
        f"Tokenization produced a degenerate corpus{where}: {total_tokens:,} "
        f"tokens using only {unique_tokens} distinct token id"
        f"{'' if unique_tokens == 1 else 's'}. That is not text. The usual "
        f"cause is a text_column that does not hold text — e.g. naming a raw "
        f"conversation column, whose rendered output lives in a different "
        f"column, so the tokenizer received structured data and emitted "
        f"separators only. Nothing downstream can detect this: occupancy, "
        f"block counts and document counts all look correct."
    )
