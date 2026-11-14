"""Which corpora a feature extraction reads, and how much of each.

WHY THIS EXISTS. SAE feature extraction took ONE dataset. The SAEs on this
estate were trained on a five-corpus mixture at 35/30/15/10/10, so every
feature's top-k activating examples — the examples labeling reads, and the only
evidence anyone ever sees for what a feature means — came from one corpus while
the dictionary learned from five.

That is not a presentation problem. `activation_frequency` is
`feature_activation_counts / len(dataset)`, and it feeds a dead-neuron gate. A
feature firing on 40% of code rows, extracted against OpenWebText alone, scores
~0 and is DELETED along with its examples. Extracted on the mixture it trained
on it scores 0.15 x 0.40 = 0.06, comfortably above the 0.001 gate. Single-corpus
extraction was quietly destroying the domain-specific features the estate exists
to find.

WHAT THIS MODULE IS NOT. It does not re-implement the mixture arithmetic.
`dataset_mixture.allocate_tokens` already caps each source at what it holds,
redistributes the shortfall, and meets the budget exactly; it is unit-agnostic
integer arithmetic. This module supplies the two things extraction needs that
training does not: an equal-shares default, and a span layout that lets a batch
be attributed to exactly one corpus.

ROWS, NOT TOKENS. Training weights tokens because the optimiser consumes tokens.
Extraction weights ROWS because a top-k slot is a row — "30% of the example
slots should come from code" is a statement about rows. Note the honest caveat:
a 17-token headline in a 2048 window gives a feature 17 chances to fire, not
2048, so 30% of rows is not 30% of firing opportunities. The caller records the
realised token split beside the row split so the divergence is measured rather
than argued about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .dataset_mixture import allocate_tokens, describe_mixture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixtureSpan:
    """One corpus's contiguous rows in the concatenated dataset.

    `start`/`stop` are GLOBAL indices, i.e. offsets into the dataset produced by
    `concatenate_datasets`, not into the corpus's own file. That is deliberate:
    it makes `sample_index` unique across corpora within a run for free, and it
    is what lets a batch be attributed to a corpus by a range check.
    """

    dataset_id: str
    label: str
    start: int
    stop: int

    @property
    def rows(self) -> int:
        return max(0, self.stop - self.start)


def extraction_quotas(
    available: Sequence[int],
    total: int,
    weights: Optional[Sequence[float]] = None,
) -> List[int]:
    """How many ROWS to read from each corpus.

    EQUAL SHARES WHEN NO WEIGHTS ARE GIVEN, which is a deliberate departure from
    `allocate_tokens`' availability-proportional default, and the same departure
    `holdout_evaluation.holdout_quotas` makes for the same reason: the examples
    are a statement about every source. A corpus that happens to be ten times
    larger should not own ten times the example slots merely for being large —
    OpenWebText has 548,716 blocks against codeparrot's 55,926, and proportional
    shares would give code 1.8% of the evidence for a dictionary that devoted
    15% of its training to it.

    Passing `None` through to `allocate_tokens` would get availability, so the
    equal case is expressed explicitly as `[1.0] * n`. Everything else — capping
    at what a corpus holds, redistributing the shortfall, meeting the budget
    exactly — is `allocate_tokens` unchanged.
    """
    n = len(available)
    if n == 0:
        return []
    shares = list(weights) if weights is not None else [1.0] * n
    return allocate_tokens(available, total, shares)


def plan_spans(
    labels: Sequence[str],
    dataset_ids: Sequence[str],
    available: Sequence[int],
    total: int,
    weights: Optional[Sequence[float]] = None,
    start_sample: int = 0,
) -> List[MixtureSpan]:
    """Lay the allocated rows out end to end, in the order given.

    `start_sample` is applied PER CORPUS — each corpus contributes its own slice
    beginning at that offset — because the tokenized blocks are already
    permuted or measured unordered, so a prefix of each corpus is already a
    representative sample of it.

    A corpus allocated zero rows still gets a span, an empty one, so the returned
    list stays positionally aligned with `dataset_ids` and the per-corpus
    statistics have an entry for every corpus the operator selected.
    """
    if not (len(labels) == len(dataset_ids) == len(available)):
        raise ValueError(
            f"plan_spans got {len(labels)} labels, {len(dataset_ids)} dataset ids "
            f"and {len(available)} row counts; they must correspond one-to-one"
        )

    reachable = [max(0, int(a) - max(0, int(start_sample))) for a in available]
    quotas = extraction_quotas(reachable, total, weights)

    spans: List[MixtureSpan] = []
    cursor = 0
    # strict: a short `quotas` would silently drop trailing corpora, and
    # positional alignment with `dataset_ids` is the contract the per-corpus
    # statistics depend on. Raise rather than truncate.
    for label, dataset_id, quota in zip(labels, dataset_ids, quotas, strict=True):
        spans.append(
            MixtureSpan(dataset_id=dataset_id, label=label, start=cursor, stop=cursor + quota)
        )
        cursor += quota
    return spans


def batch_spans(spans: Sequence[MixtureSpan], batch_size: int) -> List[Tuple[int, int]]:
    """Contiguous batches that NEVER straddle a corpus boundary.

    This is a correctness requirement, not tidiness. `batch_process_features`
    returns `fired_counts` summed over the whole call, so a batch spanning two
    corpora cannot be attributed to either — the per-corpus frequencies would be
    unrecoverable, and those are the numbers that say whether a feature fires on
    code or on chat.

    With a single span this is exactly `range(0, len(dataset), batch_size)`
    clipped at the end, i.e. the loop this replaces.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    out: List[Tuple[int, int]] = []
    for span in spans:
        for start in range(span.start, span.stop, batch_size):
            out.append((start, min(start + batch_size, span.stop)))
    return out


def span_of(spans: Sequence[MixtureSpan], index: int) -> int:
    """Which corpus a global row index belongs to; -1 if none.

    Used to attribute a batch's counts. Because `batch_spans` never straddles,
    the batch's first index decides the whole batch.
    """
    for i, span in enumerate(spans):
        if span.start <= index < span.stop:
            return i
    return -1


def calibration_indices(spans: Sequence[MixtureSpan], batch_size: int) -> List[int]:
    """Rows to calibrate the JumpReLU thresholds on — the head of EVERY corpus.

    The threshold calibration sets `sae.activation.log_threshold` from the z
    values it observes. Calibrating on `dataset[:batch_size]` samples whichever
    corpus sorts first, so under a mixture a code feature would have its firing
    threshold fixed from chat activations and would then read as dead on code.
    That is a bug the mixture creates, not a pre-existing one.

    Each corpus contributes in proportion to its span, with at least one row
    each so no corpus is silently unrepresented. Head-of-corpus rather than a
    random draw: the blocks are already permuted or measured unordered, so a
    prefix of each corpus is already representative and re-shuffling would buy
    nothing.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    populated = [s for s in spans if s.rows > 0]
    if not populated:
        return []

    total_rows = sum(s.rows for s in populated)
    budget = min(batch_size, total_rows)

    take = [max(1, int(budget * s.rows / total_rows)) for s in populated]
    # `max(1, ...)` and rounding can overshoot the budget; trim from the largest
    # contributors first so every corpus keeps its row.
    while sum(take) > budget:
        i = max(range(len(take)), key=lambda j: take[j])
        if take[i] <= 1:
            break
        take[i] -= 1

    indices: List[int] = []
    for span, n in zip(populated, take, strict=True):
        indices.extend(range(span.start, min(span.start + n, span.stop)))
    return indices


def describe_spans(spans: Sequence[MixtureSpan]) -> str:
    """One checkable line: what this run will actually read, by corpus."""
    return describe_mixture([s.label for s in spans], [s.rows for s in spans])
