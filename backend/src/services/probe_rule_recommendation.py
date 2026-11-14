"""Which pooling rule fits a probe's training view, measured from where its evidence sits (ENH-006 G39).

A run's `rules` default to `mean, max, attention` (`schemas/probe_monitor.py`, `ProbeRunConfig.rules`).
Nothing said which shape fits a LOCAL pattern (an SSN in a long text) and which a WHOLE-TEXT behaviour,
so the D128 stack was filed without `rolling_mean_max`. This module measures the shape and says so.

WHERE THE EVIDENCE COMES FROM (operator, 2026-11-09). A view may carry an optional SPAN COLUMN: per row,
the character offsets (Unicode code points, half-open ``[start, end)``) of the evidence in the row's
input text. miDataworks fills it from a rule set's span output (the SSN pack's ``ssn_spans``) when it
sends a view; anyone registering by hand may name one. It is read ONCE, at registration, into a
profile stored on the view (`evidence_spans`): numbers only, never text. A view without one has no
profile, `recommend` answers that no evidence span is recorded and recommends nothing, and the submit
refusal therefore never fires on a guess.

What the profile measures, over the rows the view keeps (after its label mapping and keyword filter):

* **evidence share** (`ratio`): per positive row with a span, the characters the spans cover over the
  row's characters. The pooling question in one number: `mean` divides the evidence by the row.
* **position**: where the first span starts, as a fraction of the row.
* **length spread**: the 90th over the 10th percentile of row length, over every kept row.
* **surface-only hard negatives**: NEGATIVE rows that carry a span. The span output marks the surface
  form (an SSN-shaped, valid number), and the label says the row is not the behaviour, so the
  difference lies in the context around the span, not in the span.

THE RULE (`recommend`), each threshold with its reason:

* `MIN_SPAN_ROWS` = 20 positive rows with a span, and `MIN_COVERAGE` = 0.2 of the positives. Below
  either the median describes a handful of rows, or a small minority of the positives (a rendering
  the rule set cannot see, e.g. spelled-out digits, carries no span); a refusal resting on that
  would be a guess, so nothing is recommended and the numbers are reported.
* `LOCAL_RATIO` = 0.2: when the median positive's evidence covers at most a fifth of the row, the
  `mean` of the token scores is at least four-fifths non-evidence, and a row twice as long halves the
  evidence's weight in it. Recommend `max` and `rolling_mean_max`: both read the strongest place and
  ignore the rest of the row. (`rolling_mean_max` grows with row length, `ProbeRunConfig.rules`'
  measured table, so it is recommended BESIDE `max`, never as the only rule, and the length-spread
  caution below says to read it by length band.)
* `WHOLE_TEXT_RATIO` = 0.6: when the evidence covers most of the row, the behaviour is a property of
  the whole text; `mean` is the least length-sensitive rule measured (0.231–0.464 separation units
  across length quartiles against 0.518–1.836 for `max`), so recommend `mean`.
* Between the two the shape is mixed; recommend `mean` AND `max`, so a run trains both shapes and the
  selection compares them.
* Rolling window: the 75th percentile of the evidence's width in the MODEL's tokens, at least
  `MIN_WINDOW` = 4 (a shorter window is a noisy `max`), at most `MAX_WINDOW` = 128. A window narrower
  than the evidence sees part of it; three quarters of the spans fit whole in this one. Tokens need a
  tokenizer, so the window is given only when a model is named (`span_tokens`); the character width
  is always reported.
* `HARD_NEGATIVE_SHARE` = 0.05: when at least one negative in twenty carries the surface form, a
  window holding only the span cannot tell those rows apart, so a second, wider window
  (`CONTEXT_WINDOW_FACTOR` = 4 x the first, capped at `MAX_WINDOW`) is recommended to take in the
  context that decides.
* `LENGTH_SPREAD_CAUTION` = 4.0: rows whose 90th-percentile length is four times the 10th are where
  `max` and `rolling_mean_max` drift with length (the measured table again); the recommendation says
  to read recall by length band. A caution, never a refusal.

The refusal (`rule_choice_refusal`) fires only when a recommendation names rules and the run omits
one of them without `acknowledge_rule_choice`; the reason is then recorded on the run.

After training (`constant_floor`, `dilution`): the share of rows whose score sits at the run's
minimum (the shape miWorkbench memory `max-pooling-floored-by-the-template` describes: 271 of 1,443 rows
at exactly 17.65 when the template prefix floored a `max` probe), and the evidence's tokens over the
row's scored tokens.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Span = Tuple[int, int]

MIN_SPAN_ROWS = 20
MIN_COVERAGE = 0.2
LOCAL_RATIO = 0.2
WHOLE_TEXT_RATIO = 0.6
MIN_WINDOW = 4
MAX_WINDOW = 128
HARD_NEGATIVE_SHARE = 0.05
CONTEXT_WINDOW_FACTOR = 4
LENGTH_SPREAD_CAUTION = 4.0
#: A score within this of the minimum (relative to the scale) counts as AT the minimum.
FLOOR_TOLERANCE = 1e-6
#: A floor holding at least this share of the rows is called out (`floored: true`).
FLOOR_SHARE_CALLED = 0.05

THRESHOLDS: Dict[str, float] = {
    "min_span_rows": MIN_SPAN_ROWS,
    "min_coverage": MIN_COVERAGE,
    "local_ratio": LOCAL_RATIO,
    "whole_text_ratio": WHOLE_TEXT_RATIO,
    "min_window": MIN_WINDOW,
    "max_window": MAX_WINDOW,
    "hard_negative_share": HARD_NEGATIVE_SHARE,
    "context_window_factor": CONTEXT_WINDOW_FACTOR,
    "length_spread_caution": LENGTH_SPREAD_CAUTION,
}

NO_SPAN_REASON = (
    "no evidence span is recorded on this view: it was registered without a span_column (miDataworks "
    "sends one from a rule set's span output; by hand, name the column holding [[start, end], ...] "
    "character offsets of the evidence). Nothing is recommended, and no run is refused on a guess"
)


class SpanColumnInvalid(ValueError):
    """A span cell that cannot be read as offsets into its row's text. `detail` is JSON-safe."""

    def __init__(self, code: str, message: str, detail: Mapping[str, Any]):
        super().__init__(message)
        self.code = code
        self.detail = dict(detail)


# ── reading a span cell ──────────────────────────────────────────────────────────────────────────


def parse_span_cell(cell: Any) -> Optional[List[Span]]:
    """A cell as a list of ``(start, end)``; None for no span (null, empty, ``[]``).

    Accepts the rule-set output's canonical JSON text (``"[[17,28]]"``) and a native list of pairs.
    Raises `ValueError` naming what is wrong for anything else: a span column that cannot be read is
    refused, never read as "no span".
    """
    if cell is None:
        return None
    value = cell
    if isinstance(cell, (bytes, bytearray)):
        value = cell.decode("utf-8")
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not JSON: {exc.msg}") from exc
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"expected a list of [start, end] pairs, got {type(value).__name__}")
    out: List[Span] = []
    for pair in value:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in pair)):
            raise ValueError(f"expected [start, end] integer pairs, got {pair!r}")
        out.append((int(pair[0]), int(pair[1])))
    return out or None


def check_spans(spans: Sequence[Span], text_length: int) -> Optional[str]:
    """Why these spans cannot index a text of `text_length` code points, or None when they can."""
    for start, end in spans:
        if not 0 <= start < end <= text_length:
            return f"span [{start}, {end}) is not inside a text of {text_length} characters"
    return None


def covered(spans: Sequence[Span]) -> int:
    """Characters the spans cover, overlaps counted once."""
    total, reach = 0, None
    for start, end in sorted(spans):
        if reach is None or start >= reach:
            total += end - start
            reach = end
        elif end > reach:
            total += end - reach
            reach = end
    return total


# ── the profile, at registration ─────────────────────────────────────────────────────────────────


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    value = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return round(float(value), 6)


def _summary(values: Sequence[float], qs: Iterable[Tuple[str, float]]) -> Dict[str, Optional[float]]:
    return {name: _quantile(values, q) for name, q in qs}


MEDIAN_P75_P90 = (("median", 0.5), ("p75", 0.75), ("p90", 0.9))
P10_MEDIAN_P90 = (("p10", 0.1), ("median", 0.5), ("p90", 0.9))


def profile_spans(
    column: str,
    rows: Sequence[Tuple[int, int, Any, Any]],
) -> Dict[str, Any]:
    """The evidence profile of a view: `rows` are ``(split position, label 1|0, text, span cell)`` for
    every row the view keeps. Pure; raises `SpanColumnInvalid` on a cell that cannot index its text.

    A row whose text is not a string (a conversation) may carry no span: offsets index text, and a
    conversation's characters are not the ones the rule set read.
    """
    positives = negatives = 0
    pos_with = neg_with = 0
    span_chars: List[float] = []
    ratios: List[float] = []
    starts: List[float] = []
    per_row: List[float] = []
    lengths: List[float] = []
    for position, label, text, cell in rows:
        try:
            spans = parse_span_cell(cell)
        except ValueError as exc:
            raise SpanColumnInvalid(
                "span_unreadable", f"span column {column!r}, row {position}: {exc}",
                {"column": column, "row": int(position)},
            ) from exc
        is_text = isinstance(text, str)
        if is_text:
            lengths.append(float(len(text)))
        if label == 1:
            positives += 1
        else:
            negatives += 1
        if not spans:
            continue
        if not is_text:
            raise SpanColumnInvalid(
                "span_needs_text",
                f"span column {column!r}, row {position}: the input is not plain text, so character "
                f"offsets cannot index it",
                {"column": column, "row": int(position)},
            )
        why = check_spans(spans, len(text))
        if why is not None:
            raise SpanColumnInvalid(
                "span_out_of_range", f"span column {column!r}, row {position}: {why}",
                {"column": column, "row": int(position), "text_length": len(text),
                 "spans": [list(s) for s in spans]},
            )
        if label != 1:
            neg_with += 1
            continue
        pos_with += 1
        width = covered(spans)
        span_chars.append(float(width))
        ratios.append(width / len(text))
        starts.append(min(s for s, _ in spans) / len(text))
        per_row.append(float(len(spans)))
    p10 = _quantile(lengths, 0.1)
    p90 = _quantile(lengths, 0.9)
    return {
        "column": column,
        "unit": "code_points",
        "positives": positives,
        "positives_with_span": pos_with,
        "negatives": negatives,
        "negatives_with_span": neg_with,
        "span_chars": _summary(span_chars, MEDIAN_P75_P90),
        "ratio": _summary(ratios, MEDIAN_P75_P90),
        "position": _summary(starts, P10_MEDIAN_P90),
        "spans_per_row": {"median": _quantile(per_row, 0.5), "max": max(per_row) if per_row else None},
        "row_chars": {"p10": p10, "median": _quantile(lengths, 0.5), "p90": p90},
        "length_spread": round(p90 / p10, 4) if p10 and p90 else None,
    }


# ── the recommendation ───────────────────────────────────────────────────────────────────────────


def _window(span_tokens: Optional[Mapping[str, Any]]) -> Optional[int]:
    p75 = (span_tokens or {}).get("p75")
    if p75 is None:
        return None
    return int(min(MAX_WINDOW, max(MIN_WINDOW, math.ceil(float(p75)))))


def recommend(
    profile: Optional[Mapping[str, Any]],
    *,
    span_tokens: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The rules (and window) a run on this view should train, with the evidence that says so.

    `profile` is the view's `evidence_spans`; `span_tokens` (optional) the evidence's width in one
    model's tokens, ``{n, median, p75, p90}``. `rules` is empty whenever nothing can be said, and then
    nothing is refused.
    """
    base: Dict[str, Any] = {"thresholds": dict(THRESHOLDS)}
    if not profile:
        return {**base, "recorded": False, "shape": None, "rules": [], "rolling_windows": None,
                "reason": NO_SPAN_REASON, "evidence": None, "cautions": []}
    positives = int(profile.get("positives") or 0)
    with_span = int(profile.get("positives_with_span") or 0)
    negatives = int(profile.get("negatives") or 0)
    hard = int(profile.get("negatives_with_span") or 0)
    coverage = with_span / positives if positives else 0.0
    ratio = (profile.get("ratio") or {}).get("median")
    hard_share = hard / negatives if negatives else 0.0
    evidence = {
        "span_column": profile.get("column"),
        "positives": positives,
        "positives_with_span": with_span,
        "span_coverage": round(coverage, 4),
        "evidence_share_median": ratio,
        "evidence_share_p90": (profile.get("ratio") or {}).get("p90"),
        "span_chars": profile.get("span_chars"),
        "span_tokens": dict(span_tokens) if span_tokens else None,
        "position": profile.get("position"),
        "row_chars": profile.get("row_chars"),
        "length_spread": profile.get("length_spread"),
        "negatives": negatives,
        "surface_only_negatives": hard,
        "surface_only_negative_share": round(hard_share, 4),
    }
    out = {**base, "recorded": True, "evidence": evidence}
    if with_span < MIN_SPAN_ROWS or coverage < MIN_COVERAGE or ratio is None:
        return {**out, "shape": "unmeasured", "rules": [], "rolling_windows": None, "cautions": [],
                "reason": (
                    f"{with_span} of {positives} positive rows carry a span; at least {MIN_SPAN_ROWS} rows and "
                    f"{MIN_COVERAGE:.0%} of the positives are needed before the share of the row the evidence "
                    f"covers says anything. Nothing is recommended, and no run is refused"
                )}
    cautions: List[str] = []
    spread = profile.get("length_spread")
    if ratio <= LOCAL_RATIO:
        shape, rules = "local", ["max", "rolling_mean_max"]
        reason = (
            f"the evidence covers a median {ratio:.1%} of a positive row (at most {LOCAL_RATIO:.0%} is local): "
            f"`mean` would average it with {1 - ratio:.0%} of the row that is not evidence, and a longer row "
            f"dilutes it further. `max` and `rolling_mean_max` read the strongest place in the row"
        )
    elif ratio >= WHOLE_TEXT_RATIO:
        shape, rules = "whole_text", ["mean"]
        reason = (
            f"the evidence covers a median {ratio:.1%} of a positive row (at least {WHOLE_TEXT_RATIO:.0%} is the "
            f"whole text): the behaviour is a property of the text, and `mean` is the least length-sensitive rule"
        )
    else:
        shape, rules = "mixed", ["mean", "max"]
        reason = (
            f"the evidence covers a median {ratio:.1%} of a positive row, between local ({LOCAL_RATIO:.0%}) and "
            f"whole-text ({WHOLE_TEXT_RATIO:.0%}): train both shapes and let the selection compare them"
        )
    windows: Optional[List[int]] = None
    if "rolling_mean_max" in rules:
        first = _window(span_tokens)
        if first is None:
            cautions.append(
                "the rolling window is sized in the model's tokens: name a model (model_id) to have it measured; "
                f"the evidence spans a median {(profile.get('span_chars') or {}).get('median')} characters"
            )
        else:
            windows = [first]
            if hard_share >= HARD_NEGATIVE_SHARE:
                wider = min(MAX_WINDOW, first * CONTEXT_WINDOW_FACTOR)
                if wider > first:
                    windows.append(wider)
    if hard_share >= HARD_NEGATIVE_SHARE:
        cautions.append(
            f"{hard} of {negatives} negatives ({hard_share:.1%}) carry the same surface form as the evidence: "
            "what separates them is the context around the span"
            + (", so a wider window is recommended beside the span-sized one" if windows and len(windows) > 1
               else "")
        )
    if spread is not None and spread >= LENGTH_SPREAD_CAUTION and shape != "whole_text":
        cautions.append(
            f"row length spreads {spread:.1f}x (90th over 10th percentile): `max` and `rolling_mean_max` grow "
            "with length, so read recall by length band before trusting one bar"
        )
    return {**out, "shape": shape, "rules": rules, "rolling_windows": windows, "reason": reason,
            "cautions": cautions}


def rule_choice_refusal(
    recommendation: Mapping[str, Any],
    rules: Sequence[str],
    acknowledgement: Optional[str],
) -> Optional[Dict[str, Any]]:
    """The refusal a run omitting a recommended rule gets, or None (nothing recommended, nothing
    omitted, or the omission acknowledged with a reason)."""
    missing = [r for r in recommendation.get("rules") or [] if r not in set(rules)]
    if not missing or (acknowledgement and acknowledgement.strip()):
        return None
    return {
        "code": "rule_choice_omits_recommendation",
        "message": (
            f"this run trains {list(rules)} and omits {missing}, which the training view's evidence "
            f"recommends: {recommendation.get('reason')}. Add {missing} to config.rules"
            + (f" (and config.rolling_windows {recommendation['rolling_windows']})"
               if "rolling_mean_max" in missing and recommendation.get("rolling_windows") else "")
            + ", or pass config.acknowledge_rule_choice with the reason; the reason is recorded on the run"
        ),
        "missing": missing,
        "recommendation": dict(recommendation),
    }


def run_record(recommendation: Mapping[str, Any], rules: Sequence[str],
               acknowledgement: Optional[str]) -> Dict[str, Any]:
    """What a submitted run records in `config.rule_recommendation`."""
    missing = [r for r in recommendation.get("rules") or [] if r not in set(rules)]
    return {
        "shape": recommendation.get("shape"),
        "rules": list(recommendation.get("rules") or []),
        "rolling_windows": recommendation.get("rolling_windows"),
        "missing": missing,
        "acknowledged": bool(missing and acknowledgement),
        "evidence": recommendation.get("evidence"),
    }


# ── the evidence in tokens ───────────────────────────────────────────────────────────────────────


def span_token_counts(tokenizer: Any, items: Sequence[Tuple[str, Sequence[Span]]]) -> List[int]:
    """Each item's evidence in the tokenizer's tokens: every span tokenised alone, summed. Out of
    context a span can merge with its neighbour differently, by about a token; that is the precision
    a window size needs."""
    from .probe_monitor_render import text_token_count

    out: List[int] = []
    for text, spans in items:
        out.append(sum(text_token_count(tokenizer, text[s:e]) for s, e in spans))
    return out


def token_summary(counts: Sequence[float]) -> Optional[Dict[str, Any]]:
    if not counts:
        return None
    return {"n": len(counts), **_summary(counts, MEDIAN_P75_P90)}


# ── after training ───────────────────────────────────────────────────────────────────────────────


def constant_floor(scores: Sequence[float], labels: Optional[Sequence[int]] = None) -> Optional[Dict[str, Any]]:
    """The share of rows whose score sits at the minimum. A probe whose minimum holds many rows is
    floored by something every row shares (the template prefix, a constant window), and its content
    tokens got no gradient there. With `labels`, also the positives sitting on it: evidence the probe
    did not see."""
    pairs = [(float(s), (labels[i] if labels is not None else None)) for i, s in enumerate(scores)
             if s is not None and math.isfinite(float(s))]
    if not pairs:
        return None
    low = min(v for v, _ in pairs)
    tolerance = FLOOR_TOLERANCE * max(1.0, abs(low))
    at = [label for v, label in pairs if v - low <= tolerance]
    share = len(at) / len(pairs)
    out = {"min_score": round(low, 6), "rows_at_min": len(at), "rows": len(pairs), "share": round(share, 6),
           "floored": len(at) > 1 and share >= FLOOR_SHARE_CALLED}
    if labels is not None:
        out["positives_at_min"] = sum(1 for label in at if label == 1)
    return out


def dilution(span_tokens: Sequence[int], scored_widths: Sequence[int]) -> Optional[Dict[str, Any]]:
    """The evidence's tokens over the row's scored tokens, per positive row with a span."""
    ratios = [min(1.0, s / w) for s, w in zip(span_tokens, scored_widths, strict=True) if w > 0]
    if not ratios:
        return None
    return {"n": len(ratios), **_summary(ratios, P10_MEDIAN_P90)}


# ── reading a view's split (the only functions here that touch a file) ──────────────────────────


def read_split_column(path: Any, split: Optional[str], column: str) -> List[Any]:
    """Every cell of `column` in the split, by split position. Refuses an absent column by name."""
    from .probe_monitor_service import open_split

    data = open_split(path, split)
    names = list(getattr(data, "column_names", []) or [])
    if column not in names:
        raise SpanColumnInvalid(
            "column_absent", f"span column {column!r} is not a column of the split; it has {names}",
            {"columns": [column]},
        )
    return list(data[column])


def view_rows(examples: Sequence[Any], inputs: Sequence[Any], cells: Sequence[Any]) -> List[Tuple[int, int, Any, Any]]:
    """`profile_spans`' rows from a built view: the text only for a plain-text row (a conversation's
    characters are not the ones a rule set read), the span cell at the example's split position."""
    return [
        (int(e.index), int(e.label), inputs[e.index] if getattr(e, "kind", "plain") == "plain" else None,
         cells[e.index])
        for e in examples
    ]


def positive_evidence(examples: Sequence[Any], inputs: Sequence[Any], cells: Sequence[Any]) -> Dict[int, Tuple[str, List[Span]]]:
    """Split position -> (text, spans) for every positive plain-text row with a readable span."""
    out: Dict[int, Tuple[str, List[Span]]] = {}
    for position, label, text, cell in view_rows(examples, inputs, cells):
        if label != 1 or not isinstance(text, str):
            continue
        try:
            spans = parse_span_cell(cell)
        except ValueError:
            continue
        if spans and check_spans(spans, len(text)) is None:
            out[position] = (text, spans)
    return out


def evidence_of_view(view: Any, dataset_path: Any, examples: Sequence[Any]) -> Dict[int, Tuple[str, List[Span]]]:
    """`positive_evidence` for a registered view, reading its input and span columns from the split."""
    inputs = read_split_column(dataset_path, view.split, view.input_column)
    cells = read_split_column(dataset_path, view.split, view.span_column)
    return positive_evidence(examples, inputs, cells)


def measure_span_tokens(view: Any, dataset_path: Any, examples: Sequence[Any], tokenizer: Any) -> Optional[Dict[str, Any]]:
    """The evidence's width in `tokenizer`'s tokens over the view's positive rows, or None without spans."""
    if not getattr(view, "span_column", None):
        return None
    evidence = evidence_of_view(view, dataset_path, examples)
    return token_summary(span_token_counts(tokenizer, list(evidence.values())))


def built_examples_of(session: Any, view: Any) -> Tuple[Any, List[Any]]:
    """The view's dataset path and its examples, as registration built them (map and filter; no render)."""
    from ..models.dataset import Dataset
    from .probe_monitor_service import (
        ResolvedColumns,
        build_view,
        load_columns,
        resolve_dataset_path,
    )

    dataset = session.query(Dataset).filter(Dataset.id == view.dataset_id).first()
    if dataset is None:
        raise ValueError(f"dataset {view.dataset_id} is gone")
    path = resolve_dataset_path(dataset.raw_path)
    inputs, labels, pairs, _total = load_columns(
        path, ResolvedColumns(input_column=view.input_column, label_column=view.label_column,
                              pair_column=view.pair_column), split=view.split,
    )
    built = build_view(inputs, labels, dict(view.label_mapping or {}), keyword_filter=view.keyword_filter,
                       pair_values=pairs, role=view.role)
    return path, list(built.examples)


def recommendation_for(view_id: str, model_id: Optional[str]) -> Dict[str, Any]:
    """`GET /probe-monitors/datasets/{id}/rule-recommendation`: the view's recommendation, with the
    window measured in `model_id`'s tokens when one is named (CPU: tokenizer only). Blocking; the
    route runs it in a thread. A window that cannot be measured is said, never guessed."""
    from types import SimpleNamespace

    from ..core.database import SyncSessionLocal
    from ..models.probe_monitor import ProbeMonitorDataset

    session = SyncSessionLocal()
    try:
        view = session.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == view_id).first()
        if view is None:
            raise LookupError(view_id)
        span_tokens = None
        unmeasured: Optional[str] = None
        if model_id and view.evidence_spans:
            try:
                from . import probe_monitor_run as run_module

                path, examples = built_examples_of(session, view)
                tokenizer = run_module._load_tokenizer_for_run(SimpleNamespace(model_id=model_id))
                span_tokens = measure_span_tokens(view, path, examples, tokenizer)
            except Exception as exc:  # noqa: BLE001 - the rules stand without the window; say why
                unmeasured = f"the window could not be measured in {model_id}'s tokens: {type(exc).__name__}: {exc}"
        answer = recommend(view.evidence_spans, span_tokens=span_tokens)
        if unmeasured:
            answer["cautions"] = [*answer.get("cautions", []), unmeasured]
        return {"view_id": view.id, "model_id": model_id, **answer}
    finally:
        session.close()


def run_diagnostics(
    *,
    view: Any,
    dataset_path: Any,
    examples: Sequence[Any],
    scored_widths: Sequence[int],
    tokenizer: Any,
    probes: Sequence[Sequence[Any]],
    score: Any,
) -> Dict[str, Any]:
    """After training (ENH-006 G39): the evidence's dilution over the training view's scored tokens,
    and each probe's constant floor on the validation rows. `probes` are the run's ``(probe, head,
    trained, val_rows, val_labels)`` entries; `score(head, rule, rule_params, rows)` is the run's own
    aggregation. A part that cannot be measured says why and never fails the run."""
    out: Dict[str, Any] = {"dilution": None, "probes": []}
    if not getattr(view, "span_column", None):
        out["dilution_reason"] = (
            "the training view records no span column, so the evidence's share of the scored tokens is unknown"
        )
    else:
        try:
            evidence = evidence_of_view(view, dataset_path, examples)
            where = {int(e.index): i for i, e in enumerate(examples)}
            items = [(where[p], text, spans) for p, (text, spans) in evidence.items() if p in where]
            counts = span_token_counts(tokenizer, [(text, spans) for _, text, spans in items])
            out["dilution"] = dilution(counts, [int(scored_widths[i]) for i, _, _ in items])
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not cost a trained run
            out["dilution_reason"] = f"{type(exc).__name__}: {exc}"
    for entry in probes:
        probe, head, _trained, val_rows, val_labels = entry[:5]
        row: Dict[str, Any] = {"probe_id": probe.id, "rule": probe.rule, "layer": probe.layer,
                               "window": (probe.rule_params or {}).get("window"),
                               "variant": getattr(probe, "variant", "dense")}
        try:
            row["constant_floor"] = constant_floor(
                score(head, probe.rule, dict(probe.rule_params or {}), val_rows), list(val_labels)
            )
        except Exception as exc:  # noqa: BLE001 - as above
            row["constant_floor"] = None
            row["reason"] = f"{type(exc).__name__}: {exc}"
        out["probes"].append(row)
    out["floored"] = [r["probe_id"] for r in out["probes"] if (r.get("constant_floor") or {}).get("floored")]
    return out
