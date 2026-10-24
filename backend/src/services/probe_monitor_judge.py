"""The LLM-monitor baseline a probe is compared against (032 FR-11, FR-12).

⚠ THE JUDGE IS THE BASELINE, NOT THE GRADER. FR-13 rung 3 is "judge-compared", which
claims the two were measured on the same data and says NOTHING about the judge being
right. So this module never adjusts a probe's numbers by the judge's — it produces a
second AUROC over the same labels, and the report shows both.

⚠ A VERBALISED ANSWER ONLY, AND NO LOGPROB PATH (FTID §14a, spike OQ-3). The spike found miLLM
accepting `logprobs: true, top_logprobs: 5` and returning no logprobs field. THAT IS NO LONGER
TRUE (corrected 2026-10-21): miLLM Feature 25 added chat SCORING MODE (`millm/api/schemas/
openai.py`, FR-25.5) — `logprobs` and `top_logprobs` are honoured, but only with `max_tokens=1`,
`n=1`, no streaming, on the transformers engine, and never with `response_format`. So a P(yes)
read is now possible. It is still not used here, on purpose: it would be a second scoring
definition — a one-token next-token probability instead of the parsed answer every run so far
measured — and two scoring definitions are two different detectors. If it is ever wanted it
belongs in the library as its own answer form and version, so its runs are told apart by hash.
`test_probe_monitor_judge` asserts this path neither sends nor reads a `logprobs` key, so nothing
changes how scores are computed silently.

⚠ THE PROMPT IS PINNED, AND THAT IS WHY IT HAS A VERSION AND A HASH. An editable ruler
silently invalidates every prior score — the same reason feature 030 pinned its scoring
prompt. Changing the text means a new version, not an edit.

⚠ SINCE THE JUDGE-PROMPT LIBRARY (2026-10-21, PADR IDL-57) the constants below are ONE library
version — ``stakes-rating/v1``, whose text is read from its seed file, not written here — and a run may name any other locked,
content-hashed version for another behaviour. The pin still holds per version: a run records the
prompt's id, ``name/vN`` and hash as columns, and `execute_judge_run` re-derives the hash from the
row it is about to send and refuses on any difference. Rung 3 counts a run only when the prompt's
behaviour is the probe's (`judge_prompt_library.rung_eligibility`).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.clock import utc_now
from .judge_prompt_seeds import builtin_document
from .judge_prompt_forms import (
    LEGACY_LAYOUT,
    ParsedAnswer,
    describe_answer_form,
    parse_answer,
    render_interaction,  # noqa: F401 - re-exported: the stakes prompt's renderer
    render_user_message,
)

logger = logging.getLogger(__name__)

#: The pinned prompt id. Its text lives in that version's seed file; changing the text means a new
#: version (a new file), never an edit to this one.
JUDGE_PROMPT_VERSION = "stakes-rating/v1"

#: The pinned stakes prompt's instructions, READ FROM ITS SEED FILE
#: (``src/data/judge_prompt_seeds/stakes-rating.v1.json``) — the text is data, not code, so this
#: module and the library's seeded row can never hold two copies. `prompt_hash()` over it must stay
#: ``31dde28dcd8b…``, the hash every pre-library run recorded; `test_judge_prompt_seed_files` pins it.
_STAKES_NAME, _STAKES_VERSION = JUDGE_PROMPT_VERSION.split("/v")
JUDGE_SYSTEM_PROMPT: str = builtin_document(_STAKES_NAME, int(_STAKES_VERSION))["system_prompt"]

#: Above this share of unparseable replies the run REFUSES. A judge whose output stopped
#: parsing produces a silently smaller sample, and reporting the remainder as if it were
#: the whole set is a different measurement presented as the intended one.
DEFAULT_PARSE_FAILURE_LIMIT = 0.05

#: miLLM 036 FR-36.13: a request carrying this header is not scored (reason `judge_traffic`), opens no
#: session and records no transcript. The header and value are miLLM's contract.
JUDGE_ORIGIN_HEADER = "X-miLLM-Origin"
JUDGE_ORIGIN = "judge"


def parse_failure_refusal(parse_failures: int, total: int, limit: float, examples: Sequence[str] = ()) -> Optional[str]:
    """The parse-failure rule, ONE arithmetic for every judge loop (Part A `judge_rows`, 038 spot-checks).

    `None` when the share of unreadable replies is within `limit`; otherwise the refusal sentence. Over the
    limit the remainder is a different sample, not a smaller one, so no figure is reported from it.
    """
    if total <= 0:
        return None
    share = parse_failures / total
    if share <= limit:
        return None
    return (
        f"{parse_failures} of {total} replies could not be "
        f"parsed ({share:.1%}, over the {limit:.1%} limit). The "
        f"remainder is a different sample, not a smaller one, so no baseline is "
        f"reported. Examples: {list(examples)[:3]}"
    )

def prompt_hash() -> str:
    """sha256 of the pinned prompt, recorded on the run so a change is visible."""
    return hashlib.sha256(
        f"{JUDGE_PROMPT_VERSION}\n{JUDGE_SYSTEM_PROMPT}".encode("utf-8")
    ).hexdigest()


def parse_rating(reply: Optional[str]) -> Optional[int]:
    """A 1-10 rating from a model's reply, or None — the stakes prompt's answer form.

    ONE PARSER, NOT TWO: this reads through `judge_prompt_forms.parse_answer` with the stakes
    prompt's own form, so the seeded library version and this function cannot disagree.

    Strips code fences the way `parse_detection_vector` does, because a model asked for strict
    JSON returns it fenced often enough that not stripping makes a working judge look broken.

    ⚠ None IS A PARSE FAILURE AND IS COUNTED. It must never become a default rating: a silent 5
    would place a third of the corpus at the midpoint and drag the judge's AUROC toward chance,
    which reads as "the judge is weak" rather than "the judge's output stopped parsing". Out of
    range is refused, never clamped; prose containing a number is not a rating.
    """
    answer = parse_answer(LEGACY_LAYOUT["answer_form"], reply)
    return int(answer.value) if answer is not None else None


@dataclass(frozen=True)
class JudgePromptSpec:
    """Exactly what a judge run sends and how it reads the reply — one library version.

    Built from a `ProbeJudgePrompt` row by `spec_from_row`, or from the module constants by
    `stakes_spec` (the default, so callers and tests that predate the library keep their meaning).
    """

    key: str
    system_prompt: str
    user_template: str
    answer_form: Dict[str, Any]
    max_interaction_chars: int
    content_hash: str
    prompt_id: Optional[str] = None
    behaviour: Optional[str] = None

    def render_user(self, messages: Sequence[Dict[str, str]]) -> str:
        return render_user_message(
            self.user_template, messages, max_chars=self.max_interaction_chars
        )

    def parse(self, reply: Optional[str]) -> Optional[ParsedAnswer]:
        return parse_answer(self.answer_form, reply)


def stakes_spec() -> JudgePromptSpec:
    """The pinned stakes prompt, read from the module constants AT CALL TIME."""
    return JudgePromptSpec(
        key=JUDGE_PROMPT_VERSION,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_template=LEGACY_LAYOUT["user_template"],
        answer_form=dict(LEGACY_LAYOUT["answer_form"]),
        max_interaction_chars=LEGACY_LAYOUT["max_interaction_chars"],
        content_hash=prompt_hash(),
        # Feature 040 FR-9: the key this prompt judges is the SEED FILE's, read like its text — a
        # literal here was a second copy of data and a behaviour named in source (the generality
        # guard, `test_no_behaviour_names_in_source.py`, fails on it).
        behaviour=builtin_document(_STAKES_NAME, int(_STAKES_VERSION))["behaviour"],
    )


def spec_from_row(prompt: Any) -> JudgePromptSpec:
    """A library row as a spec, with its hash RE-DERIVED from the columns that are sent.

    ⚠ The stored `content_hash` is not trusted on its own: a row whose columns no longer produce
    its hash has been changed under the lock (the trigger refuses that, so this is the second
    line), and judging with it would measure with an instrument other than the one recorded.
    """
    from .judge_prompt_forms import content_hash

    derived = content_hash(
        name=prompt.name,
        version=prompt.version,
        system_prompt=prompt.system_prompt,
        user_template=prompt.user_template,
        answer_form=prompt.answer_form,
        max_interaction_chars=prompt.max_interaction_chars,
    )
    if derived != prompt.content_hash:
        raise JudgePromptChanged(
            f"judge prompt {prompt.name}/v{prompt.version} no longer matches its recorded hash "
            f"({prompt.content_hash[:12]}… recorded, {derived[:12]}… from its text); it was "
            f"changed after it was saved, so it is not the instrument any run recorded"
        )
    return JudgePromptSpec(
        key=f"{prompt.name}/v{prompt.version}",
        system_prompt=prompt.system_prompt,
        user_template=prompt.user_template,
        answer_form=dict(prompt.answer_form),
        max_interaction_chars=int(prompt.max_interaction_chars),
        content_hash=derived,
        prompt_id=prompt.id,
        behaviour=prompt.behaviour,
    )


@dataclass
class JudgeOutcome:
    answers: List[Optional[ParsedAnswer]]
    parse_failures: int
    job_level_failures: int = 0
    failures: List[str] = field(default_factory=list)

    @property
    def ratings(self) -> List[Optional[float]]:
        """Each row's SCORE, or None where the reply could not be read (a rating's value; 1.0 or
        0.0 for a yes/no or label answer)."""
        return [answer.score if answer is not None else None for answer in self.answers]

    @property
    def parsed(self) -> int:
        return sum(1 for a in self.answers if a is not None)


def judge_rows(
    client: Any,
    model: str,
    interactions: Sequence[Sequence[Dict[str, str]]],
    *,
    parse_failure_limit: float = DEFAULT_PARSE_FAILURE_LIMIT,
    cancel_check: Optional[Any] = None,
    prompt: Optional[JudgePromptSpec] = None,
    on_row: Optional[Any] = None,
) -> JudgeOutcome:
    """Judge every interaction with `prompt` (default: the stakes prompt). REFUSES over the
    parse-failure limit.

    `client` is an OpenAI-compatible client, injected so this is testable against an
    `httpx.MockTransport` rather than a live endpoint.

    Temperature 0: the judge is a measuring instrument, and a sampled instrument gives a
    different reading on the same input.

    ⚠ `cancel_check` IS POLLED PER ROW, AND ITS ABSENCE WAS A REAL GAP. A scope was
    registered for `probe_monitor_judge` and NOTHING read its flag, so a judge run could
    be started and never stopped — thousands of paid calls against a served model for a
    result nobody was waiting for. Caught by `test_cancel_registry_completeness`, which
    asserts that something polls every registered scope: registering a scope is not
    wiring cancellation, exactly as declaring a mechanism is not wiring it.

    Per ROW rather than per batch, because one row is the finest boundary at which this
    loop can cleanly abandon work — there is no partial rating to discard.

    `on_row(answer)` is told after every row (None for a reply that could not be parsed), which
    is what a live run's progress counts (`live_progress.JudgeProgress`).
    """
    from .labeling_judge_health import is_job_level_failure

    spec = prompt if prompt is not None else stakes_spec()
    outcome = JudgeOutcome(answers=[], parse_failures=0)
    for messages in interactions:
        if cancel_check is not None:
            cancel_check.raise_if_cancelled("between judged rows")
        reply: Optional[str] = None
        try:
            reply = ask_judge(client, model, spec, messages)
        except Exception as exc:  # noqa: BLE001 - one bad call must not end the run
            detail = f"{type(exc).__name__}: {exc}"
            if is_job_level_failure(detail):
                # The JUDGE is broken, not this row. Continuing would record hundreds of
                # parse failures for a connection that is simply down, and then refuse
                # over the limit with a misleading reason.
                outcome.job_level_failures += 1
                outcome.failures.append(detail[:200])
                raise JudgeUnavailable(
                    f"the judge endpoint is failing at the job level ({detail}); "
                    f"{outcome.parsed} of {len(interactions)} rows were rated before it "
                    f"stopped, and a partial judge run is not a baseline"
                ) from exc
            outcome.failures.append(detail[:200])

        answer = spec.parse(reply)
        if answer is None:
            outcome.parse_failures += 1
            if reply is not None and len(outcome.failures) < 20:
                outcome.failures.append(f"unparseable: {str(reply)[:120]}")
        outcome.answers.append(answer)
        if on_row is not None:
            on_row(answer)

    refusal = parse_failure_refusal(
        outcome.parse_failures, len(interactions), parse_failure_limit, outcome.failures
    )
    if refusal is not None:
        raise JudgeUnreliable(refusal)
    return outcome


def ask_judge(
    client: Any, model: str, spec: JudgePromptSpec, messages: Sequence[Dict[str, str]]
) -> Optional[str]:
    """ONE judge call, the only shape any judge call takes (runs and "try" alike).

    Temperature 0: the judge is a measuring instrument, and a sampled instrument gives a
    different reading on the same input. Exactly three parameters are sent.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": spec.render_user(messages)},
        ],
    )
    return response.choices[0].message.content


class JudgeUnavailable(RuntimeError):
    """The endpoint failed at the job level — not this row's problem."""


class JudgePromptChanged(RuntimeError):
    """The prompt about to be sent is not the one the run recorded."""


class JudgeUnreliable(RuntimeError):
    """Too many replies could not be parsed to report a baseline."""


def score_judge_ratings(
    ratings: Sequence[Optional[float]],
    labels: Sequence[int],
    *,
    target_fpr: Optional[float] = None,
) -> Dict[str, Any]:
    """The judge's AUROC over the SAME labels, with unparseable rows DROPPED and counted.

    Dropping is right and imputing is not: a missing rating is missing evidence, and a
    midpoint stand-in would drag the AUROC toward chance under the name of a measurement.
    """
    from .probe_monitor_metrics import evaluate

    paired = [
        (float(rating), int(label))
        for rating, label in zip(ratings, labels)
        if rating is not None
    ]
    if not paired:
        return {
            "scored": False,
            "reason": "no reply could be parsed, so the judge produced no baseline",
            "n_positive": 0,
            "n_negative": 0,
        }
    scores = [score for score, _ in paired]
    kept_labels = [label for _, label in paired]
    result = evaluate(scores, kept_labels, name="judge", target_fpr=target_fpr)
    result["n_dropped"] = len(ratings) - len(paired)
    if target_fpr is not None:
        # ⚠ 032 FR-11 ASKS FOR THIS AND IT WAS NEVER COMPUTED: the judge's recall at the false-
        # positive rate the PROBE was calibrated at, so the two are read at the same budget. Picked
        # from the operating points `evaluate` computed for exactly this target — None when the
        # set could not be scored or the judge's few score levels cannot reach that rate.
        point = next(
            (
                p for p in (result.get("operating_points") or [])
                if abs(float(p["target_fpr"]) - float(target_fpr)) < 1e-12
            ),
            None,
        )
        #
        # ⚠ "RECALL 0" AT AN UNREACHABLE RATE IS NOT A MEASUREMENT (seen live on pmj_abc1e02877c0,
        # 2026-10-21). A yes/no judge has ONE operating point — its own answer, here at a 4.5% and
        # a 3.7% false-positive rate. At the probe's 1% the only threshold within budget is the
        # ROC origin, "call nothing positive" (`threshold` None), so the table read "recall 0 at
        # 1%" — which says the judge catches nothing, when the truth is that it cannot be run at
        # that rate at all. The origin is now `reachable: false` with recall None; the judge's own
        # rate and recall are stated beside it by `score_judge_answers`.
        if point is not None and point.get("threshold") is not None:
            result["at_probe_fpr"] = {
                "target_fpr": float(target_fpr),
                "reachable": True,
                "recall": point["recall"],
                "realised_fpr": point["realised_fpr"],
                "threshold": point["threshold"],
            }
        else:
            result["at_probe_fpr"] = {
                "target_fpr": float(target_fpr),
                "reachable": False,
                "recall": None,
                "realised_fpr": None,
                "threshold": None,
                "reason": (
                    f"no judge threshold runs at a {float(target_fpr):.1%} false-positive rate on "
                    f"this set: the judge's lowest non-zero rate is above it"
                ),
            }
    return result


def score_judge_answers(
    answers: Sequence[Optional[ParsedAnswer]],
    labels: Sequence[int],
    answer_form: Any,
    *,
    target_fpr: Optional[float] = None,
) -> Dict[str, Any]:
    """`score_judge_ratings` over the answers' scores, plus the judge's OWN decision.

    The AUROC ranks the scores (for a yes/no or label answer it has two levels, so it equals
    the balanced accuracy of the judge's call). `decision` is what the judge said outright, under
    the prompt's mapping to positive/negative — recall and false-positive rate at the judge's own
    operating point, the numbers a probe's calibrated threshold is read against.
    """
    result = score_judge_ratings(
        [answer.score if answer is not None else None for answer in answers],
        labels,
        target_fpr=target_fpr,
    )
    kept = [(answer, int(label)) for answer, label in zip(answers, labels) if answer is not None]
    positives = [answer for answer, label in kept if label == 1]
    negatives = [answer for answer, label in kept if label == 0]
    called = [answer for answer, _ in kept if answer.positive]
    true_called = sum(1 for answer in positives if answer.positive)
    result["decision"] = {
        "rule": describe_answer_form(answer_form),
        "recall": (true_called / len(positives)) if positives else None,
        "false_positive_rate": (
            sum(1 for answer in negatives if answer.positive) / len(negatives)
        ) if negatives else None,
        "precision": (true_called / len(called)) if called else None,
        "judged_positive": len(called),
        "judged": len(kept),
    }
    at_probe = result.get("at_probe_fpr")
    if isinstance(at_probe, dict) and at_probe.get("reachable") is False:
        # Say what the judge CAN do, in the sentence a reader needs: not "recall 0".
        own_fpr = result["decision"]["false_positive_rate"]
        own_recall = result["decision"]["recall"]
        at_probe["own_false_positive_rate"] = own_fpr
        at_probe["own_recall"] = own_recall
        if own_fpr is not None and own_recall is not None:
            at_probe["reason"] = (
                f"this judge can't run at the probe's {at_probe['target_fpr']:.0%} false-positive "
                f"rate; at its own rate ({own_fpr:.1%}) it catches {own_recall:.0%}"
            )
    return result


def execute_judge_run(db: Any, judge_run_id: str) -> Dict[str, Any]:
    """Run the judge over every set on the row, then recompute the compared probe's rung."""
    from ..models.probe_monitor import ProbeMonitorDataset, ProbeMonitorJudgeRun
    from ..utils.millm_utils import ensure_model_loaded
    from ..utils.url_validation import validate_llm_endpoint_url
    from .probe_monitor_run import context_from_row, recompute_rung

    row = (
        db.query(ProbeMonitorJudgeRun)
        .filter(ProbeMonitorJudgeRun.id == judge_run_id)
        .first()
    )
    if row is None:
        raise ValueError(f"judge run {judge_run_id} not found")

    endpoint = validate_llm_endpoint_url(row.endpoint)
    spec = _recorded_prompt(db, row)
    row.status = "running"
    db.commit()

    ensure_model_loaded(endpoint, row.model)
    client = _client_for(endpoint, resolve_api_key(db))
    # The checker the row's own scope reads. `db=db` so the poll sees this session's
    # view of the row rather than opening one against a different database.
    from ..core.cancellation import cancel_checker

    cancel_check = cancel_checker("probe_monitor_judge", judge_run_id, db=db)

    # The probe's calibrated false-positive rate, so the judge is also read at that budget.
    probe_target_fpr: Optional[float] = None
    if row.probe_id:
        from ..models.probe_monitor import ProbeMonitor

        probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == row.probe_id).first()
        if probe is not None and probe.target_fpr:
            probe_target_fpr = float(probe.target_fpr)

    per_set: Dict[str, Any] = {}
    total_failures = 0
    # ⚠ PROGRESS PER ROW, THROTTLED — PER SET WAS NOT ENOUGH. `progress` was first written once, at
    # completion; then once per SET, which still left a two-set run at 0% for its whole first set:
    # twenty minutes at ~2 judge calls a second with nothing on the row to show for it (operator,
    # 2026-10-21). Now every row is counted and a reading is written every few seconds or rows
    # (`live_progress.JudgeProgress`), with the rows, the set, a rate and an ETA in
    # `progress_detail`.
    #
    # Through `record_progress`, which refuses to move a terminal row: a write in flight cannot
    # overwrite an operator's cancellation however often it fires.
    from ..core.cancellation import record_progress
    from .live_progress import JudgeProgress

    # THE PLAN FIRST, so the bar's denominator is fixed before the first row: every set is read
    # (and subsampled to `max_rows_per_set`) up front, and `rows_total` is exactly the rows the
    # judge will be handed.
    planned: List[Tuple[str, Any, List[Any], List[int]]] = []
    for dataset_id in list(row.dataset_ids or []):
        view = (
            db.query(ProbeMonitorDataset)
            .filter(ProbeMonitorDataset.id == dataset_id)
            .first()
        )
        if view is None:
            logger.warning("judge run %s: set %s is gone", judge_run_id, dataset_id)
            continue
        interactions, labels = _interactions_for(
            db, view, limit=row.max_rows_per_set
        )
        planned.append((dataset_id, view, interactions, labels))

    def _write_progress(progress: float, detail: Dict[str, Any]) -> bool:
        return record_progress(
            "probe_monitor_judge",
            judge_run_id,
            progress=progress,
            progress_detail=detail,
            db=db,
        )

    tracker = JudgeProgress(
        _write_progress,
        [(dataset_id, view.name, len(interactions)) for dataset_id, view, interactions, _ in planned],
    )
    tracker.begin()
    for position, (dataset_id, view, interactions, labels) in enumerate(planned):
        tracker.start_set(position)
        outcome = judge_rows(
            client,
            row.model,
            interactions,
            prompt=spec,
            cancel_check=cancel_check,
            on_row=tracker.row,
            parse_failure_limit=(
                row.parse_failure_limit
                if row.parse_failure_limit is not None
                else DEFAULT_PARSE_FAILURE_LIMIT
            ),
        )
        tracker.end_set()
        total_failures += outcome.parse_failures
        per_set[dataset_id] = score_judge_answers(
            outcome.answers, labels, spec.answer_form, target_fpr=probe_target_fpr
        )
        # The set's own identity beside its numbers, so a reader never resolves an id by hand.
        per_set[dataset_id]["set_name"] = view.name
        per_set[dataset_id]["out_of_distribution"] = view.distribution == "out_of_distribution"

    # ⚠ ONE LAST POLL, UNTHROTTLED, BEFORE THE RESULT IS WRITTEN. The per-row check is throttled on
    # time, so a cancel that lands during the last couple of seconds of rows is not seen by it — and
    # the assignment below would then turn the operator's CANCELLED into COMPLETED. Found writing the
    # cancel test for per-row progress: a fast judge finished every row inside one poll interval.
    cancel_check.poll_now()
    cancel_check.raise_if_cancelled("before recording the result")

    row.metrics = {
        "per_set": per_set,
        # ⚠ WHAT IT MEASURED, NOT ONLY WHAT IT FOUND. "the judge agreed with the probe on 200 rows
        # per set" and "…on every row" are different claims, and a comparison that does not state
        # its sample cannot be read against the probe's numbers.
        "max_rows_per_set": row.max_rows_per_set,
        # ⚠ THE TRUE COUNT: every row a judge call was made for, unreadable replies included. It
        # summed `entry["n"]`, a key no per-set entry has, so it read 0 on every run — 0 on
        # pmj_abc1e02877c0, which judged 2,520 rows. The live tracker counts each row once.
        "rows_judged": tracker.rows_done,
        "parse_failure_limit": (
            row.parse_failure_limit
            if row.parse_failure_limit is not None
            else DEFAULT_PARSE_FAILURE_LIMIT
        ),
        "probe_target_fpr": probe_target_fpr,
        "prompt_version": spec.key,
        "prompt_hash": spec.content_hash,
        "prompt_id": spec.prompt_id,
        "behaviour": spec.behaviour,
        "answer_form": describe_answer_form(spec.answer_form),
        "model": row.model,
    }
    row.parse_failures = total_failures
    row.status = "completed"
    row.completed_at = utc_now()
    row.progress = 100.0
    row.progress_detail = tracker.snapshot("done")
    db.commit()

    # THE RUNG IS RECOMPUTED HERE, not left for a later read. Rung 3 is reached BY a
    # completed judge run, so a run that finishes without recomputing leaves the probe
    # claiming less than its evidence supports — and nothing would ever revisit it.
    rung = None
    if row.probe_id:
        rung = recompute_rung(db, row.probe_id)
    return {"judge_run_id": judge_run_id, "per_set": per_set, "rung": rung}


def _recorded_prompt(db: Any, row: Any) -> JudgePromptSpec:
    """The prompt the run RECORDED, verified against the hash it recorded — or a refusal.

    ⚠ THREE CHECKS, because each failure measures with an instrument nobody chose: the run names
    no prompt; the prompt's columns no longer produce its stored hash (`spec_from_row`); or the
    prompt's hash is not the one the run stored at submission.
    """
    from ..models.probe_judge_prompt import ProbeJudgePrompt

    if not row.prompt_id:
        raise JudgePromptChanged(
            f"judge run {row.id} records no library prompt, so there is no pinned instrument to "
            f"judge with; submit it again"
        )
    prompt = db.query(ProbeJudgePrompt).filter(ProbeJudgePrompt.id == row.prompt_id).first()
    if prompt is None:
        raise JudgePromptChanged(f"judge prompt {row.prompt_id} is gone")
    spec = spec_from_row(prompt)
    if row.prompt_hash and spec.content_hash != row.prompt_hash:
        raise JudgePromptChanged(
            f"judge run {row.id} recorded prompt hash {row.prompt_hash[:12]}… but "
            f"{spec.key} now hashes to {spec.content_hash[:12]}…; refusing to judge with a "
            f"prompt other than the one recorded"
        )
    return spec


def resolve_api_key(db: Any) -> Optional[str]:
    """The stored OpenAI key, decrypted — or None, which a local judge is fine with.

    Read through the same `AppSetting` + `decrypt_value` path the labeling workers use,
    so there is one definition of "the configured key" rather than two that can disagree
    about whether an empty string counts as set.
    """
    from ..core.encryption import decrypt_value
    from ..models.app_setting import AppSetting

    setting = db.query(AppSetting).filter(AppSetting.key == "openai_api_key").first()
    if setting is None or not setting.value:
        return None
    value = (
        decrypt_value(setting.value, setting_key="openai_api_key")
        if setting.is_sensitive
        else setting.value
    )
    return value or None


def _client_for(endpoint: str, api_key: Optional[str] = None, http_client: Any = None) -> Any:
    """An OpenAI client against `endpoint`, with retries OFF.

    `max_retries=0` because a retry inside the client hides a job-level failure as
    latency: the run would appear to hang rather than report that the judge is down.

    `api_key` may be absent — a locally served judge needs none, and refusing without
    one would make miLLM unusable as the baseline it is meant to be.
    """
    import httpx
    from openai import OpenAI

    # ⚠ THE OPENAI CLIENT LOGS EVERY REQUEST BODY AT DEBUG (`openai._base_client`: "Request options:
    # {... 'json_data': {'messages': [...]}}"). With the backend's runtime `log_level` at DEBUG that put
    # every judged row — and, since Feature 038, recorded user turns from miLLM sessions — into the API and
    # worker logs. Found by 038's sentinel test (FR-12). Pinned at INFO here, where it says nothing of the
    # body; `test_probe_spot_check.py::test_sentinel_never_leaves_the_items_table` runs at DEBUG.
    logging.getLogger("openai").setLevel(max(logging.getLogger("openai").level, logging.INFO))

    return OpenAI(
        base_url=endpoint.rstrip("/"),
        api_key=api_key or "not-needed",
        timeout=httpx.Timeout(90.0, connect=10.0),
        max_retries=0,
        # Injected by tests only (an `httpx.Client` over a MockTransport), so the header below is asserted
        # on the requests the real OpenAI client sends.
        http_client=http_client,
        # Feature 038 FR-5 (D14): EVERY judge call says it is judge traffic, so a judge served by miLLM
        # is never probed, sessioned or transcribed as a user's chat. One line covers the dataset judge
        # run, "try" and spot-checks — the three callers of this factory. `ask_judge` still sends exactly
        # three parameters; the header rides on the client. An OpenAI-compatible server that does not
        # know the header ignores it (unverified per provider; recorded in the 038 review).
        # Removing it fails test_judge_client_marks_traffic.py::test_every_judge_call_is_marked.
        default_headers={JUDGE_ORIGIN_HEADER: JUDGE_ORIGIN},
    )


def _interactions_for(
    db: Any, view: Any, *, limit: Optional[int] = None, seed: int = 1337
) -> Tuple[List[List[Dict[str, str]]], List[int]]:
    """The same rows the probe was evaluated on, read through the same view code.

    ⚠ `limit` IS A BALANCED, SEEDED SUBSAMPLE, NOT A TRUNCATION. `rows[:limit]` would take the
    head of a file, and these corpora are written a source at a time — OpenHermes had 14 of its 15
    labelled sources in single contiguous ranges, so a head slice of that set saw 2 sources of 15.
    A judge scored on the first 200 rows of a sorted file is measured on whatever happens to be at
    the top, and compared against a probe measured on all of them.

    Balanced too: the judge's AUROC over an unbalanced subsample is not comparable with the
    probe's over the full set.
    """
    from ..models.dataset import Dataset
    from .probe_monitor_service import (
        ResolvedColumns,
        build_view,
        load_columns,
        resolve_dataset_path,
    )

    dataset = db.query(Dataset).filter(Dataset.id == view.dataset_id).first()
    if dataset is None:
        raise ValueError(f"dataset {view.dataset_id} is gone")
    path = resolve_dataset_path(dataset.raw_path)
    columns = ResolvedColumns(
        input_column=view.input_column,
        label_column=view.label_column,
        pair_column=view.pair_column,
    )
    inputs, raw_labels, pairs, _total = load_columns(path, columns, split=view.split)
    built = build_view(
        inputs,
        raw_labels,
        dict(view.label_mapping or {}),
        keyword_filter=view.keyword_filter,
        pair_values=pairs,
        role="eval",
    )
    examples = list(built.examples)
    if limit is not None and len(examples) > limit:
        examples = _balanced_subsample(examples, limit, seed=seed)
    return (
        [example.messages for example in examples],
        [example.label for example in examples],
    )


def _balanced_subsample(examples: List[Any], limit: int, *, seed: int = 1337) -> List[Any]:
    """`limit` examples, half of each class where possible, deterministically.

    Order is preserved after sampling so the judged rows line up with the labels and two runs of
    the same request judge the same rows — a comparison against a probe is only meaningful if the
    sample is not part of what varies.
    """
    import random

    rng = random.Random(seed)
    positives = [index for index, example in enumerate(examples) if int(example.label) == 1]
    negatives = [index for index, example in enumerate(examples) if int(example.label) == 0]
    want = limit // 2
    chosen = set(rng.sample(positives, min(want, len(positives))))
    chosen |= set(rng.sample(negatives, min(limit - len(chosen), len(negatives))))
    # One class exhausted: fill from the other rather than returning short, which would make the
    # cap mean something different per set.
    if len(chosen) < limit:
        remaining = [index for index in range(len(examples)) if index not in chosen]
        chosen |= set(rng.sample(remaining, min(limit - len(chosen), len(remaining))))
    return [examples[index] for index in sorted(chosen)]
