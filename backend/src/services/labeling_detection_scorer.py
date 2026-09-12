"""Automated detection scoring for feature labels.

Given ONLY a label, can a judge pick out which passages activate the feature?
That question turns label quality into a number, which is what lets prompt
templates be ranked instead of read.

Three design rules, each earned:

**The ruler is pinned.** DETECTION_PROMPT_V1 is a module constant, never a
`labeling_prompt_templates` row. The template under test varies; the instrument
measuring it must not, or scores from different days are not comparable. Every
result records the prompt version that produced it.

**Negatives come from other features, and the claim is bounded.**
`feature_activations` stores only each feature's top-K ACTIVATING passages — on
the L46 extraction every feature has exactly 100 rows and the smallest stored
activation is 0.67, strictly positive. So the "bottom-K" of a feature's own
examples are its weakest POSITIVES, and using them as negatives would punish a
correct label. Negatives are therefore drawn from other features. There is no
encode-on-text service, so we cannot certify a passage as non-activating; the
only defensible claim is that it falls below the target's stored-example
threshold, and that threshold travels with the result as `negative_ceiling`.

**A judge that fails the gate yields no score.** Reporting a weak judge's failure
as a low score would send a user rewriting prompts that were already good. This
mirrors the stance circuit calibration already takes with `judge_unreliable`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from .detection_metrics import confusion, is_degenerate, panel_score

logger = logging.getLogger(__name__)

# ── the pinned instrument ────────────────────────────────────────────────────

DETECTION_PROMPT_VERSION = "detection/v1"

# Asks for an OBJECT, not a bare array. The transport sets
# response_format={"type":"json_object"} and the labeling path prepends a
# directive demanding a reply that starts with "{" — a template asking for
# "[0, 1, 1, 0]" contradicts both, which is one of the three reasons the seeded
# eleutherai_detection template never worked.
#
# The true mix is 50/50 but the prompt never says so: disclosing it anchors the
# judge to a 50% output rate, which inflates balanced accuracy AND hides the
# "says 1 to everything" failure that balanced accuracy exists to expose.
DETECTION_PROMPT_V1 = (
    "A feature inside a language model has been described as follows.\n\n"
    "DESCRIPTION: {explanation}\n\n"
    "Below are {n} numbered text passages. For each one, decide whether that "
    "feature is active in it, judging ONLY by the description above.\n\n"
    "{passages}\n\n"
    "Reply with a JSON object and nothing else, containing exactly {n} entries "
    "in the order shown, each 1 (the feature is active) or 0 (it is not):\n"
    '{{"labels": [...]}}'
)

DEFAULT_BATCH_SIZE = 10
DEFAULT_N_POSITIVE = 10
DEFAULT_N_HARD_NEGATIVE = 5
DEFAULT_N_EASY_NEGATIVE = 5
MAX_PASSAGE_TOKENS = 48

# Gate thresholds.
LITERAL_ORACLE_MIN_BA = 0.75      # below: the judge cannot do the task at all
NULL_CONTROL_MAX_BA = 0.60        # above: the harness is leaking the answer
MAX_DEGENERATE_RATE = 0.30        # above: the judge answers all-1 or all-0
MAX_PARSE_FAILURE_RATE = 0.10     # above: replies are unusable

# Per-FEATURE floors, applied when scoring the panel (the gate thresholds above
# apply to the controls only). Without these a feature whose batches mostly
# failed still contributed a number to the panel mean: two of three batches
# unparseable and the third answered all-1 yields balanced accuracy 0.5, which
# is indistinguishable from a genuinely vague label. That is the exact
# "plausible-but-meaningless score" this module exists to refuse.
MAX_FEATURE_PARSE_FAILURE_RATE = 0.50
MIN_ITEMS_PER_CLASS = 3
"""Both classes need this many scored items.

A pure count floor certified nothing: 3 positives and 1 negative passes at 4
items, but TNR is then estimated from a SINGLE item, so it is quantised to {0,1}
and balanced accuracy swings by 0.5 on that one judgement. `panel_score` takes an
unweighted mean and cannot see n, so that feature carried the same weight as a
20-item one.
"""


class DetectionScoringError(Exception):
    """Raised when a scoring run cannot proceed at all."""


# ── deterministic shuffling ──────────────────────────────────────────────────

def make_rng(panel_id: str, feature_id: str) -> random.Random:
    """A reproducible per-(panel, feature) RNG.

    Deliberately NOT seeded from `hash()`: Python salts str hashing per process
    (PYTHONHASHSEED), so a hash()-seeded shuffle differs between two Celery
    workers and the run stops being reproducible. blake2b is stable everywhere.
    """
    digest = hashlib.blake2b(
        f"{panel_id}:{feature_id}".encode("utf-8"), digest_size=8
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def panel_id_for(extraction_job_id: str, feature_ids: Sequence[str]) -> str:
    """Content-addressed panel identity.

    Equal ids PROVE an identical, order-independent, extraction-bound feature
    set, so a comparison can refuse a mismatch instead of trusting a join.
    """
    payload = f"{extraction_job_id}|{','.join(sorted(feature_ids))}"
    return "pnl_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── passage rendering ────────────────────────────────────────────────────────

def assemble_items(
    positives: Sequence[Dict[str, Any]],
    hard_negatives: Sequence[Dict[str, Any]],
    easy_negatives: Sequence[Dict[str, Any]],
    *,
    panel_id: str,
    feature_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Build one feature's scoring items: every batch mixed, no batch guessable.

    Two properties are required, and the obvious implementations give one each.

    A plain shuffle leaves a mixed batch merely LIKELY — measured at 1.66% of
    seeds for a 10/1/1 ratio and 9.4% for 20/2/2. A single-class batch is
    answered uniformly by a CORRECT judge, gets scored as degenerate, and
    because the seed is a pure function of (panel_id, feature_id) that feature
    fails identically forever.

    Deterministic interleaving fixes that and introduces something worse: the
    class pattern becomes a pure function of (n_pos, n_neg), so at the module
    defaults EVERY batch of EVERY feature of EVERY trial has ground truth
    [1,0,1,0,1,0,1,0,1,0]. A judge with any alternation bias then scores 1.0 on
    every label under every template, and two templates that genuinely differ
    are reported indistinguishable.

    So: stratify ACROSS batches, shuffle WITHIN each batch. Class balance is
    structural; order is seeded-random and differs per feature.
    """
    if not positives or not (hard_negatives or easy_negatives):
        raise DetectionScoringError(
            f"cannot assemble items for {feature_id}: "
            f"{len(positives)} positive(s) and "
            f"{len(hard_negatives) + len(easy_negatives)} negative(s); "
            f"both classes are required for a balanced-accuracy score"
        )

    rng = make_rng(panel_id, feature_id)
    pos = [{**p, "label": 1, "kind": "positive"} for p in positives]
    neg = ([{**n, "label": 0, "kind": "hard_negative"} for n in hard_negatives]
           + [{**n, "label": 0, "kind": "easy_negative"} for n in easy_negatives])
    rng.shuffle(pos)
    rng.shuffle(neg)

    total = len(pos) + len(neg)
    n_batches = max(1, (total + batch_size - 1) // batch_size)

    # Buckets must match how the CONSUMER slices, or the guarantee is void:
    # equal-sized buckets over 12 items with batch_size 10 gave 6+6, while
    # score_feature sliced 10+2 and the 2-item tail came out single-class.
    sizes = [batch_size] * (n_batches - 1) + [total - batch_size * (n_batches - 1)]
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(n_batches)]

    # Deal the MINORITY class first, round-robin. Filling proportionally instead
    # rounds a small bucket's minority share to zero and hands back the
    # single-class batch this function exists to prevent.
    minority, majority = (neg, pos) if len(neg) <= len(pos) else (pos, neg)
    b = 0
    for item in minority:
        for _ in range(n_batches):
            if len(buckets[b]) < sizes[b]:
                buckets[b].append(item)
                b = (b + 1) % n_batches
                break
            b = (b + 1) % n_batches
    b = 0
    for item in majority:
        for _ in range(n_batches):
            if len(buckets[b]) < sizes[b]:
                buckets[b].append(item)
                b = (b + 1) % n_batches
                break
            b = (b + 1) % n_batches

    items: List[Dict[str, Any]] = []
    for bucket in buckets:
        rng.shuffle(bucket)          # order inside a batch is unguessable
        items.extend(bucket)
    return items


def negative_ceiling(target_examples: Sequence[Dict[str, Any]]) -> Optional[float]:
    """The strongest claim available about a negative, as a number.

    We cannot certify a passage as non-activating — there is no encode-on-text
    service, so nothing evaluates the target feature on a donor's passage. What
    IS knowable is the target's own weakest stored activation: every negative is
    drawn from outside that stored set, so the honest claim is "below this".

    Returned so it travels with the result. The module docstring promised this
    value and did not produce it, which left the validity argument resting on a
    number nobody could see.
    """
    acts = [
        e["max_activation"] for e in target_examples
        if e.get("max_activation") is not None
    ]
    return min(acts) if acts else None


def render_passage(row: Dict[str, Any], *, max_tokens: int = MAX_PASSAGE_TOKENS) -> str:
    """Render one stored example as plain text, leaking nothing.

    Positives and negatives MUST go through this same function. The labeling
    formatter is unusable here: it wraps the prime token in `<<>>` markers and
    its caller prefixes an activation value, either of which is a total answer
    leak — the judge would score 1.0 against any label whatsoever.

    Truncation is symmetric for the same reason: if positives were consistently
    longer than negatives, length alone would separate the classes and every
    template would score well.
    """
    prefix = row.get("prefix_tokens") or []
    suffix = row.get("suffix_tokens") or []
    prime = row.get("prime_token") or ""

    def _clean(tok: str) -> str:
        return (
            str(tok)
            .replace("▁", " ")   # sentencepiece
            .replace("Ġ", " ")        # byte-level BPE
            .replace("##", "")        # wordpiece
        )

    tokens = [_clean(t) for t in prefix] + [_clean(prime)] + [_clean(t) for t in suffix]
    if len(tokens) > max_tokens:
        # Keep the prime token centred so both classes are cropped the same way.
        centre = len(prefix)
        half = max_tokens // 2
        lo = max(0, centre - half)
        tokens = tokens[lo:lo + max_tokens]

    # Plain join, nothing added. There is deliberately NO marker-stripping pass
    # here: a scrubber would mangle legitimate corpus text (`**bold**`, `[[wiki]]`)
    # and, worse, it would MASK a leak introduced upstream — a mutation that wraps
    # the prime token in `<<>>` would be silently cleaned up and no test would
    # notice. The invariant is "this function adds nothing", pinned by an exact
    # equality test, not "we clean up afterwards".
    return re.sub(r"\s+", " ", "".join(tokens)).strip()


# ── negative sampling ────────────────────────────────────────────────────────

_HARD_NEGATIVES_SQL = text("""
    WITH target AS (
        SELECT normalized_token
        FROM feature_token_index
        WHERE feature_id = :feature_id AND token_rank = 1
        LIMIT 1
    ),
    donors AS (
        SELECT fti.feature_id
        FROM feature_token_index fti, target t
        WHERE fti.normalized_token = t.normalized_token
          AND fti.token_rank = 1
          AND fti.extraction_id = :extraction_id
          AND fti.feature_id <> :feature_id
        -- ORDER BY is load-bearing, not tidiness: LIMIT without it lets
        -- PostgreSQL return a different donor subset after a plan change or a
        -- vacuum, so two trials over the same panel would draw different
        -- negatives and stop being paired.
        ORDER BY fti.feature_id
        LIMIT :donor_limit
    ),
    ranked AS (
        SELECT fa.feature_id, fa.sample_index, fa.max_activation,
               fa.prefix_tokens, fa.prime_token, fa.suffix_tokens,
               ROW_NUMBER() OVER (
                   PARTITION BY fa.feature_id ORDER BY fa.max_activation DESC, fa.id
               ) AS per_donor_rank
        FROM feature_activations fa
        JOIN donors d ON d.feature_id = fa.feature_id
        WHERE fa.sample_index <> ALL(:exclude_samples)
    )
    -- Round-robin: take each donor's best passage before any donor's second.
    -- Ordering the union by max_activation alone let the highest-activating
    -- donors monopolise the draw — measured on L46, 5 negatives came from only
    -- 2 of 10 available donors, so "hard negatives" tested a far narrower slice
    -- than the item count implied.
    SELECT feature_id, sample_index, max_activation,
           prefix_tokens, prime_token, suffix_tokens
    FROM ranked
    -- Tiebreak rank-1 rows by a per-target hash, not by activation.
    -- `ORDER BY per_donor_rank, max_activation DESC` fixed the monopolisation
    -- but replaced it with a DETERMINISTIC bias: with 20 donors and 5 slots it
    -- drew the same 5 highest-activating donors on every feature, forever, and
    -- the other 15 were never sampled. Hashing on the salt spreads the draw
    -- across donors while staying reproducible for a given (feature, panel).
    ORDER BY per_donor_rank,
             md5(feature_id || '\x1f' || :salt || '\x1f' || sample_index::text)
    LIMIT :limit
""")

_EASY_NEGATIVES_SQL = text("""
    SELECT fa.feature_id, fa.sample_index, fa.max_activation,
           fa.prefix_tokens, fa.prime_token, fa.suffix_tokens
    FROM feature_activations fa
    JOIN features f ON f.id = fa.feature_id
    WHERE f.extraction_job_id = :extraction_id
      AND fa.feature_id <> :feature_id
      AND fa.sample_index <> ALL(:exclude_samples)
    -- The separators make the sort key unambiguous UNCONDITIONALLY.
    --
    -- In practice `:salt` is non-empty (it defaults to the target feature id)
    -- and already sits between the two variable-length fields, so it prevents
    -- the boundary collision on its own — md5('a1'||''||'23') equalling
    -- md5('a12'||''||'3') is only reachable with an EMPTY salt. That makes this
    -- a guard against a caller, not against today's call site, and the test for
    -- it must pass an empty salt or it proves nothing.
    ORDER BY md5(fa.feature_id || '\x1f' || :salt || '\x1f' || fa.sample_index::text)
    LIMIT :limit
""")


def sample_negatives(
    db,
    *,
    feature_id: str,
    extraction_id: str,
    exclude_samples: Sequence[int],
    n_hard: int = DEFAULT_N_HARD_NEGATIVE,
    n_easy: int = DEFAULT_N_EASY_NEGATIVE,
    salt: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Draw negatives from OTHER features in the same extraction.

    `exclude_samples` is not optional hygiene — without it a "negative" can be
    literally one of the target's own top-100 passages, arriving through a donor
    that happens to share the sample. That single filter is the difference
    between a metric and a random number generator.

    Hard negatives share the target's rank-1 token, so they separate "fires on
    the word running" from "fires on physical locomotion". Easy negatives do not.
    Scored separately: the gap between them says how much of a label's apparent
    quality is just naming the token.
    """
    # A single NULL in the bound array makes `<> ALL(...)` reject every row —
    # zero negatives, every score None, the feature silently unscored and the
    # panel quietly smaller while still reporting success. Postgres keeps all
    # rows for an EMPTY array, so no sentinel is needed; only the None filter is.
    exclude = [s for s in exclude_samples if s is not None]

    hard = [
        dict(r._mapping) for r in db.execute(
            _HARD_NEGATIVES_SQL,
            {
                "feature_id": feature_id,
                "extraction_id": extraction_id,
                "exclude_samples": exclude,
                "donor_limit": max(n_hard * 4, 20),
                "limit": n_hard,
                "salt": salt or feature_id,
            },
        )
    ]
    easy = [
        dict(r._mapping) for r in db.execute(
            _EASY_NEGATIVES_SQL,
            {
                "feature_id": feature_id,
                "extraction_id": extraction_id,
                "exclude_samples": exclude,
                "salt": salt or feature_id,
                "limit": n_easy,
            },
        )
    ]
    return hard, easy


# ── judge reply parsing ──────────────────────────────────────────────────────

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_detection_vector(raw: str, expected_len: int) -> Optional[List[int]]:
    """Parse a judge reply into a binary vector, or return None.

    Tolerant of shape, intolerant of ambiguity. A reply whose length differs
    from `expected_len` is REJECTED rather than padded or truncated: a
    misaligned vector silently scrambles ground-truth alignment and yields a
    plausible ~0.5, which reads as "mediocre label" when the real problem is the
    judge. None means "no evidence" and the feature is skipped, never imputed.
    """
    if not raw:
        return None
    text_ = _FENCE.sub("", raw.strip())

    candidate: Any = None
    try:
        parsed = json.loads(text_)
        if isinstance(parsed, dict):
            for key in ("labels", "answers", "predictions", "result"):
                if key in parsed:
                    candidate = parsed[key]
                    break
        elif isinstance(parsed, list):
            candidate = parsed
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\[[\s,01truefalse]*\]", text_, re.IGNORECASE)
        if m:
            try:
                candidate = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None

    if not isinstance(candidate, list):
        return None

    out: List[int] = []
    for v in candidate:
        if isinstance(v, bool):
            out.append(1 if v else 0)
        elif isinstance(v, (int, float)) and v in (0, 1):
            out.append(int(v))
        elif isinstance(v, str) and v.strip() in ("0", "1"):
            out.append(int(v.strip()))
        else:
            return None

    if len(out) != expected_len:
        logger.warning(
            "detection reply had %d entries, expected %d — rejecting rather than "
            "aligning; a misaligned vector scores at chance and looks like a bad label",
            len(out), expected_len,
        )
        return None
    return out


# ── scoring ──────────────────────────────────────────────────────────────────

JudgeFn = Callable[[str], str]
"""Takes a rendered prompt, returns the raw model reply. Injected so the whole
gate/metric/refusal surface is testable without a model."""


def build_detection_prompt(explanation: str, passages: Sequence[str]) -> str:
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(passages, start=1))
    return DETECTION_PROMPT_V1.format(
        explanation=explanation.strip(), n=len(passages), passages=numbered
    )


def score_feature(
    explanation: str,
    items: Sequence[Dict[str, Any]],
    judge: JudgeFn,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    negative_ceiling_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Score one feature's label. `items` come from `assemble_items`.

    `negative_ceiling_value` is echoed into the result so the bound on what a
    negative actually IS travels with the number it produced. Without it the
    module's validity argument lives only in a docstring: a reader gets a
    balanced accuracy with no way to know that "negative" here means "below this
    feature's weakest stored activation" rather than "verified inactive".
    """
    preds: List[int] = []
    truth: List[int] = []
    kinds: List[str] = []
    parse_failures = 0
    degenerate_batches = 0
    batches = 0

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        batches += 1
        reply = judge(build_detection_prompt(explanation, [b["text"] for b in batch]))
        vector = parse_detection_vector(reply, len(batch))
        if vector is None:
            parse_failures += 1
            continue
        batch_truth = [int(b["label"]) for b in batch]
        if is_degenerate(vector, batch_truth):
            degenerate_batches += 1
        preds.extend(vector)
        truth.extend(int(b["label"]) for b in batch)
        kinds.extend(str(b["kind"]) for b in batch)

    if not preds:
        return {
            "balanced_accuracy": None, "ba_hard": None, "ba_easy": None,
            "parse_failure_rate": 1.0 if batches else None,
            "degenerate_rate": None, "confusion": None,
            "negative_ceiling": negative_ceiling_value,
            "batches": batches, "failed_batches": parse_failures,
            "degenerate_batches": degenerate_batches,
            "reason": "no batch produced a usable reply",
        }

    parse_rate = parse_failures / batches if batches else 0.0
    n_pos = sum(1 for t in truth if t == 1)
    n_neg = len(truth) - n_pos
    thin_class = min(n_pos, n_neg) < MIN_ITEMS_PER_CLASS
    # No separate total-count floor: `min(n_pos, n_neg) < 3` already implies
    # fewer than 4 scored items, so a count floor could never be the decisive
    # condition and only made the reason strings harder to follow.
    if parse_rate > MAX_FEATURE_PARSE_FAILURE_RATE or thin_class:
        return {
            "balanced_accuracy": None, "ba_hard": None, "ba_easy": None,
            "parse_failure_rate": parse_rate,
            "degenerate_rate": degenerate_batches / batches if batches else None,
            "batches": batches, "failed_batches": parse_failures,
            "degenerate_batches": degenerate_batches,
            "confusion": None,
            "negative_ceiling": negative_ceiling_value,
            "reason": (
                f"only {len(preds)} item(s) scored across {batches} batch(es) "
                f"({parse_rate:.0%} unparseable, {n_pos} positive / {n_neg} "
                f"negative); too little to score this feature"
                if parse_rate > MAX_FEATURE_PARSE_FAILURE_RATE
                else
                f"only {n_pos} positive and {n_neg} negative item(s) survived; "
                f"balanced accuracy needs at least {MIN_ITEMS_PER_CLASS} of each "
                f"or a single judgement swings it by half"
            ),
        }

    overall = confusion(preds, truth)
    if overall.balanced_accuracy is None:
        # Reached when the surviving items are all one class. Every other
        # refusal here carries a reason; this one returned None/None and the
        # feature vanished from the panel with no explanation available anywhere.
        return {
            "balanced_accuracy": None, "ba_hard": None, "ba_easy": None,
            "parse_failure_rate": parse_rate,
            "degenerate_rate": degenerate_batches / batches if batches else None,
            "batches": batches, "failed_batches": parse_failures,
            "degenerate_batches": degenerate_batches,
            "confusion": overall.to_dict(),
            "negative_ceiling": negative_ceiling_value,
            "reason": (
                f"the {len(preds)} surviving item(s) are all one class "
                f"({n_pos} positive / {n_neg} negative); balanced accuracy is "
                f"undefined without both"
            ),
        }

    def _subset(kind_filter) -> Optional[float]:
        idx = [i for i, k in enumerate(kinds) if k == "positive" or kind_filter(k)]
        if not idx:
            return None
        return confusion([preds[i] for i in idx], [truth[i] for i in idx]).balanced_accuracy

    return {
        "balanced_accuracy": overall.balanced_accuracy,
        "ba_hard": _subset(lambda k: k == "hard_negative"),
        "ba_easy": _subset(lambda k: k == "easy_negative"),
        "parse_failure_rate": parse_failures / batches if batches else None,
        "degenerate_rate": degenerate_batches / batches if batches else None,
        "batches": batches, "failed_batches": parse_failures,
        "degenerate_batches": degenerate_batches,
        "confusion": overall.to_dict(),
        "negative_ceiling": negative_ceiling_value,
        "reason": None,
    }


def run_gate(
    controls: Sequence[Dict[str, Any]],
    judge: JudgeFn,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Decide whether this judge may grade explanations at all.

    Runs BEFORE scoring, so a failed gate costs a handful of calls rather than
    the whole panel.

    Control A (literal oracle, ceiling): the explanation names a token the
    passages literally contain. A judge that cannot find that cannot grade
    anything subtler. Scored WITHOUT hard negatives — those contain the token by
    construction, so the literal explanation is genuinely wrong on them and
    including them would fail a perfectly good judge.

    A control's `items` MUST come from `assemble_items`. Grouped items make the
    early batches single-class, a constant answer legitimately fits them, and the
    degeneracy check silently measures nothing — verified: a judge answering
    all-1 on 40% of literal batches reported degenerate_rate 0.0 on grouped
    items and 0.4 on the same items assembled.

    Control B (mismatched label, floor): another feature's explanation. Expected
    ~chance. Scoring well means positives are separable from negatives WITHOUT
    the label — a rendering leak, a length artefact, or negatives that aren't
    hard. Nothing else in this module catches those, and each one inflates every
    template equally, which looks like success.
    """
    if not controls:
        raise DetectionScoringError(
            "run_gate called with no controls; it would otherwise report "
            "`control_unscorable` for a control that does not exist, having "
            "never called the judge at all"
        )

    literal_scores: List[float] = []
    null_scores: List[float] = []
    parse_failures = 0
    degenerate = 0
    degenerate_batches_seen = 0
    total_batches = 0

    for ctl in controls:
        easy_only = [i for i in ctl["items"] if i["kind"] != "hard_negative"]
        lit = score_feature(ctl["literal_explanation"], easy_only, judge,
                            batch_size=batch_size)
        nul = score_feature(ctl["mismatched_explanation"], ctl["items"], judge,
                            batch_size=batch_size)
        # Degeneracy is counted from the LITERAL control ONLY.
        #
        # The null control deliberately supplies a label describing nothing in
        # the passages, so the CORRECT answer to it is all-zero — and counting
        # that as degeneracy failed capable judges. Verified: a judge that aced
        # the literal oracle (BA 1.0) and correctly scored chance on the null
        # control (0.5) was rejected as `judge_degenerate` at rate 0.5. Only the
        # literal control asks the judge to discriminate, so only it can show a
        # refusal to.
        degenerate += lit.get("degenerate_batches") or 0
        # Denominator is PARSED batches only. A batch that produced no vector
        # cannot be degenerate, so counting it as evidence of non-degeneracy
        # dilutes the rate — the identical dilution the parse-rate fix three
        # lines below was written to remove. Measured: 3 degenerate of 7 parsed
        # (0.43, over the 0.30 threshold) was reported as 3/10 = 0.30, and the
        # strict `>` then let it through.
        degenerate_batches_seen += (
            (lit.get("batches") or 0) - (lit.get("failed_batches") or 0)
        )

        for r in (lit, nul):
            # Pool BATCH COUNTS, not rates. Summing per-run rates and dividing by
            # the number of runs gave every run equal weight regardless of size —
            # and the literal control runs on fewer items than the null control,
            # so a single failure in the smaller run was inflated. A run with no
            # items also incremented the denominator while contributing nothing,
            # silently diluting the rate toward zero.
            parse_failures += r.get("failed_batches") or 0
            total_batches += r.get("batches") or 0
        if lit["balanced_accuracy"] is not None:
            literal_scores.append(lit["balanced_accuracy"])
        if nul["balanced_accuracy"] is not None:
            null_scores.append(nul["balanced_accuracy"])

    lit_ba = sum(literal_scores) / len(literal_scores) if literal_scores else None
    null_ba = sum(null_scores) / len(null_scores) if null_scores else None
    parse_rate = parse_failures / total_batches if total_batches else None
    degen_rate = (
        degenerate / degenerate_batches_seen if degenerate_batches_seen else None
    )

    failures: List[str] = []
    if lit_ba is None:
        # score_feature already produced a precise reason; use it instead of
        # guessing. A literal control with too few easy negatives is a CONTROL
        # construction problem, not a judge problem, and saying otherwise sends
        # the operator to debug the wrong thing.
        if parse_rate is not None and parse_rate > MAX_PARSE_FAILURE_RATE:
            failures.append("judge_unparseable")
        else:
            failures.append("control_unscorable")
    elif lit_ba < LITERAL_ORACLE_MIN_BA:
        failures.append("judge_unreliable")
    if null_ba is not None and null_ba > NULL_CONTROL_MAX_BA:
        failures.append("harness_leakage")
    if degen_rate is not None and degen_rate > MAX_DEGENERATE_RATE:
        failures.append("judge_degenerate")
    if parse_rate is not None and parse_rate > MAX_PARSE_FAILURE_RATE:
        failures.append("judge_unparseable")

    reasons = {
        "judge_unreliable": (
            f"the judge scored {lit_ba:.2f} balanced accuracy on an explanation that "
            f"literally named a token in the passages; it cannot grade explanations. "
            f"Use a stronger judge model." if lit_ba is not None else ""
        ),
        "harness_leakage": (
            f"a MISMATCHED explanation still scored {null_ba:.2f}; positives are "
            f"distinguishable from negatives without reference to the label. Check "
            f"passage rendering, length, or truncation." if null_ba is not None else ""
        ),
        "judge_degenerate": "the judge answered all-1 or all-0 for most batches",
        "judge_unparseable": "too many replies could not be parsed into a vector",
        "control_unscorable": (
            "the literal control could not be scored, and the judge's replies "
            "parsed fine — the control itself is too thin. It is scored without "
            "hard negatives (they contain the token by construction), so it needs "
            f"at least {MIN_ITEMS_PER_CLASS} EASY negatives. Widen the easy-negative "
            "draw for the control features."
        ),
    }

    return {
        "passed": not failures,
        # Reported affirmatively, so it must not be True for a judge that
        # produced nothing at all — a consumer checking this field rather than
        # `passed` would otherwise proceed on silence.
        # `judge_degenerate` belongs here: a judge that answered uniformly on
        # most of the batches where it said anything refused to discriminate,
        # which is exactly what "unreliable" means to a consumer reading this
        # affirmative field instead of `passed`. `harness_leakage` is excluded
        # deliberately — it indicts the harness, not the judge.
        "judge_reliable": not (
            {"judge_unreliable", "judge_unparseable", "judge_degenerate"}
            & set(failures)
        ) and lit_ba is not None,
        "literal_control_ba": lit_ba,
        "null_control_ba": null_ba,
        "parse_failure_rate": parse_rate,
        "degenerate_rate": degen_rate,
        "prompt_version": DETECTION_PROMPT_VERSION,
        "failures": sorted(set(failures)),
        "reason": "; ".join(reasons[f] for f in sorted(set(failures)) if reasons.get(f)) or None,
        "thresholds": {
            "literal_oracle_min_ba": LITERAL_ORACLE_MIN_BA,
            "null_control_max_ba": NULL_CONTROL_MAX_BA,
            "max_degenerate_rate": MAX_DEGENERATE_RATE,
            "max_parse_failure_rate": MAX_PARSE_FAILURE_RATE,
        },
    }


def score_panel(
    features: Sequence[Dict[str, Any]],
    judge: JudgeFn,
    *,
    gate: Optional[Dict[str, Any]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Score every feature in a panel, refusing outright if the gate failed.

    A failed gate yields no scores at all. Reporting them would attribute the
    judge's incapacity to the template, which is exactly the mistake that sends
    someone off rewriting prompts that were already good.
    """
    # An ABSENT gate is not a passing gate. Defaulting to "score anyway" meant
    # the plain call score_panel(features, judge) skipped the whole sanity check
    # and returned scored=True from an unvetted judge — the one outcome this
    # module exists to prevent, reachable by simply forgetting an argument.
    if gate is not None:
        # Shape and provenance are checked UNCONDITIONALLY. Checking them only
        # when `passed` was truthy left two holes: a failed gate from an older
        # ruler was echoed inside a result stamped with the CURRENT ruler (one
        # document asserting two rulers), and `passed` was never required to be
        # a bool — a JSON round-trip turning it into the string "false" is
        # truthy in Python, so a gate carrying
        # `failures: ["judge_unreliable", "harness_leakage"]` authorised scoring
        # and the result echoed the failure list it had just ignored.
        missing = {"literal_control_ba", "null_control_ba", "prompt_version",
                   "failures", "passed"} - set(gate)
        if missing:
            raise DetectionScoringError(
                f"gate is missing {sorted(missing)}; a bare {{'passed': True}} "
                f"authorises scoring without any control having run"
            )
        if not isinstance(gate["passed"], bool):
            raise DetectionScoringError(
                f"gate['passed'] is {type(gate['passed']).__name__}, not bool; "
                f"a truthy non-boolean (the string 'false', say) would authorise "
                f"scoring on a judge that failed every control"
            )
        if gate["passed"] and gate.get("failures"):
            raise DetectionScoringError(
                f"gate says passed=True but lists failures {gate['failures']}; "
                f"refusing to score on a self-contradictory gate"
            )
        if gate["prompt_version"] != DETECTION_PROMPT_VERSION:
            raise DetectionScoringError(
                f"gate was measured under ruler {gate['prompt_version']!r} but "
                f"scoring runs under {DETECTION_PROMPT_VERSION!r}; scores from "
                f"different rulers are not comparable"
            )

    if gate is None or not gate.get("passed", False):
        return {
            "scored": False,
            "balanced_accuracy_mean": None,
            "per_feature": {},
            "gate": gate,
            "features_scored": 0,
            "features_total": len(features),
            "ci": None,
            "prompt_version": DETECTION_PROMPT_VERSION,
            "reason": (
                "no judge sanity gate was supplied; an ungated judge cannot "
                "produce a comparable score"
                if gate is None
                else (gate.get("reason") or "judge failed the sanity gate")
            ),
        }

    per_feature: Dict[str, Any] = {}
    ba_by_feature: Dict[str, Optional[float]] = {}
    for f in features:
        result = score_feature(
            f["explanation"], f["items"], judge, batch_size=batch_size,
            negative_ceiling_value=f.get("negative_ceiling"),
        )
        per_feature[f["feature_id"]] = result
        ba_by_feature[f["feature_id"]] = result["balanced_accuracy"]

    agg = panel_score(ba_by_feature)
    return {
        **agg,
        "per_feature": per_feature,
        "gate": gate,
        "prompt_version": DETECTION_PROMPT_VERSION,
    }
