"""Session spot-checks (Feature 038; ENH-005 D14, D28–D35, D39, D51–D54; PADR IDL-59).

1. `start` checks the request (miLLM serves the session routes; the behaviour is catalogued and not
   archived; the prompt judges THIS behaviour and is not archived; the size; one running check per
   behaviour; the behaviour's primary probe is armed in miLLM), writes the run and dispatches it.
2. `execute` (the Celery body) asks miLLM ONCE for a seeded, stratified sample of recorded turns,
   stratified on the primary probe (D53), reads each turn's newest-turn transcript, reads every compared
   probe's verdict on that turn, sends the judge the window each probe scored (D29), parses the reply with
   the prompt's own answer form (D2), and writes one item per (turn, probe) AS IT GOES — so a cancelled
   or failed run keeps its partial items (D-T2). Part A's parse-failure rule applies through the shared
   `probe_monitor_judge.parse_failure_refusal`.
3. `report` is a pure function of the items: per compared probe, overall and per stratum, the 2 × 2 table,
   agreement with a seeded bootstrap confidence interval, and Cohen's kappa — under the fixed statement
   that the judge is a second opinion, not ground truth (D32).
4. `export_jsonl` streams the disagreements as `DisagreementRow` lines (D33, D57).

⚠ TEXT NEVER LEAVES THE ITEMS TABLE (D35, FR-12). Nothing here logs a transcript, a judged excerpt or a
judge reply; the Celery task takes and returns ids and counts only. `test_spot_check_privacy.py` runs a
check whose turn carries a sentinel and looks for it in logs, task arguments and results.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ..core.clock import utc_now
from ..models.probe_behaviour import ProbeBehaviour
from ..models.probe_judge_prompt import ProbeJudgePrompt
from ..models.probe_monitor import ProbeMonitor
from ..models.probe_spot_check import ProbeSpotCheck, ProbeSpotCheckItem
from ..schemas.spot_check import MAX_SIZE, DisagreementRow
from . import behaviour_catalogue as catalogue
from . import millm_read
from .judge_prompt_forms import prompt_key

logger = logging.getLogger(__name__)

STRATA = ("fired", "near_threshold", "not_fired")
ACTIVE = ("pending", "running")
EXCERPT_DAYS = 7
REPLY_CHARS = 2000
BOOTSTRAP_RESAMPLES = 1000

#: D32. Always in the report, first. Removing it fails `test_spot_check_report.py::test_statement_is_present`.
STATEMENT = (
    "The judge is a second opinion, not ground truth. Disagreement means one of the two is wrong, not which."
)
#: FR-10: the strata are not proportional to traffic.
OVERALL_NOTE = (
    "Overall figures reflect the sampling design (equal quotas of fired, near-threshold and not-fired "
    "turns), not the mix of real traffic."
)


class SpotCheckRefused(Exception):
    def __init__(self, status: int, code: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}

    def as_detail(self) -> Dict[str, Any]:
        detail: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            detail["details"] = self.details
        return detail


# ── the judged text (D29, D51; FR-4) ─────────────────────────────────────────────────────────


def messages_for_window(window: str, turn: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """What the judge is shown for a probe that scored `window`, and the `judge_saw` label.

    * `last_user` → the user's message: exactly what the probe scored (`same_window`);
    * `response` → the user's message AND the reply: the probe scored the reply, and a reply alone is
      not judgeable (FPRD FR-4 planner choice; `user_and_reply`);
    * `prompt` / `all` → the newest recorded turn. The probe saw the whole resent conversation; miLLM
      records only the newest turn (D5), so these are labelled `newest_turn` and reported apart (D51).
    """
    user = {"role": "user", "content": str(turn.get("user_text") or "")}
    reply = {"role": "assistant", "content": str(turn.get("reply_text") or "")}
    if window == "last_user":
        return [user], "same_window"
    if window == "response":
        return [user, reply], "user_and_reply"
    if window == "prompt":
        return [user], "newest_turn"
    return [user, reply], "newest_turn"


def split_strata(size: int) -> Dict[str, int]:
    """`size` over the three strata as evenly as possible; the remainder goes to `fired` first."""
    base, extra = divmod(size, len(STRATA))
    return {name: base + (1 if i < extra else 0) for i, name in enumerate(STRATA)}


# ── start ────────────────────────────────────────────────────────────────────────────────────


async def start(db: AsyncSession, body: Dict[str, Any], actor: str) -> ProbeSpotCheck:
    """Validate, write the run, and return it (the caller dispatches after commit)."""
    from .judge_prompt_library import lock
    from .probe_monitor_judge import JudgePromptChanged, spec_from_row

    availability = await run_in_threadpool(millm_read.spot_check_availability)
    if not availability["available"]:
        raise SpotCheckRefused(503, "millm_unavailable", availability["reason"])

    key = body["behaviour"]
    head = (await db.execute(select(ProbeBehaviour).where(ProbeBehaviour.key == key))).scalar_one_or_none()
    if head is None:
        raise SpotCheckRefused(422, "behaviour_not_catalogued",
                               f"{key!r} is not catalogued; catalogue it on the Behaviours tab first")
    if head.archived_at is not None:
        raise SpotCheckRefused(422, "behaviour_archived", f"{key!r} is archived; unarchive it to check it")

    prompt = (
        await db.execute(select(ProbeJudgePrompt).where(ProbeJudgePrompt.id == body["prompt_id"]))
    ).scalar_one_or_none()
    if prompt is None:
        raise SpotCheckRefused(404, "prompt_not_found", f"judge prompt {body['prompt_id']} not found")
    if prompt.archived_at is not None:
        raise SpotCheckRefused(422, "prompt_archived",
                               f"{prompt_key(prompt.name, prompt.version)} is archived; choose another version")
    # FR-7: a spot-check is ABOUT the behaviour, so a judge of another behaviour is refused outright.
    if prompt.behaviour != key:
        raise SpotCheckRefused(
            422, "prompt_behaviour_mismatch",
            f"{prompt_key(prompt.name, prompt.version)} judges {prompt.behaviour!r}, not {key!r}; "
            f"choose a prompt for {key!r}",
            {"prompt_behaviour": prompt.behaviour, "behaviour": key},
        )

    strata = body.get("strata") or split_strata(int(body.get("size") or 100))
    total = sum(int(v) for v in strata.values())
    if total < 1 or total > MAX_SIZE:
        raise SpotCheckRefused(422, "size", f"a spot-check samples 1 to {MAX_SIZE} turns; {total} were asked for")

    running = (
        await db.execute(
            select(ProbeSpotCheck.id).where(ProbeSpotCheck.behaviour == key, ProbeSpotCheck.status.in_(ACTIVE))
        )
    ).first()
    if running is not None:
        raise SpotCheckRefused(409, "check_running",
                               f"spot-check {running[0]} of {key!r} is still running; one at a time per behaviour")

    primary = await catalogue.primary_probe(db, key)
    if primary is None:
        raise SpotCheckRefused(422, "no_primary_probe",
                               f"{key!r} has no primary probe; the sample is drawn on the primary (D53)")
    armed = await run_in_threadpool(millm_read.armed_probes)
    if isinstance(armed, millm_read.Unknown):
        raise SpotCheckRefused(503, "millm_unavailable", armed.reason)
    behaviour_probes = set(
        (await db.execute(select(ProbeMonitor.id).where(ProbeMonitor.behaviour == key))).scalars().all()
    )
    compared = [
        {"mistudio_probe_id": str(p["source_probe_id"]), "millm_probe_id": str(p.get("id"))}
        for p in armed
        if p.get("source_probe_id") in behaviour_probes
    ]
    primary_entry = next((c for c in compared if c["mistudio_probe_id"] == primary.id), None)
    if primary_entry is None:
        raise SpotCheckRefused(
            422, "primary_not_armed",
            f"the primary probe {primary.id} is not armed in miLLM, so there are no verdicts to sample",
        )

    try:
        spec = spec_from_row(prompt)
    except JudgePromptChanged as exc:
        raise SpotCheckRefused(409, "prompt_changed", str(exc)) from exc

    now = utc_now()
    seed = body.get("seed")
    if seed is None:
        seed = random.SystemRandom().randrange(2**31 - 1)
    current = await catalogue.current_version(db, head)
    row = ProbeSpotCheck(
        behaviour=key,
        behaviour_version=current.version,
        prompt_id=prompt.id,
        prompt_key=spec.key,
        prompt_hash=spec.content_hash,
        endpoint=body["endpoint"],
        model=body["model"],
        sample_spec={
            "seed": int(seed),
            "strata": {name: int(strata.get(name, 0)) for name in STRATA},
            "window": body.get("window") or "all",
            "near_band": float(body.get("near_band") or 0.1),
            "since": (body.get("since") or (now - timedelta(days=EXCERPT_DAYS))).isoformat(),
            "until": body["until"].isoformat() if body.get("until") else None,
            "primary": primary_entry,
            "compared": compared,
        },
        status="pending",
        parse_failure_limit=float(body.get("parse_failure_limit") or 0.05),
        excerpts_expire_at=now + timedelta(days=EXCERPT_DAYS),
        created_by=actor,
    )
    db.add(row)
    lock(prompt)  # the library's rule: a version a run used never changes (IDL-57)
    await db.commit()
    await db.refresh(row)
    return row


def sample_request(row: ProbeSpotCheck) -> Dict[str, Any]:
    """The ONE body sent to miLLM's SAMPLE route (036 FR-36.12)."""
    spec = row.sample_spec or {}
    return {
        "probe_id": spec["primary"]["millm_probe_id"],
        "window": spec.get("window") or "all",
        "strata": spec["strata"],
        "near_band": spec.get("near_band", 0.1),
        "since": spec.get("since"),
        "until": spec.get("until"),
        "seed": spec["seed"],
        "require_transcript": True,
    }


# ── execute (the Celery body) ────────────────────────────────────────────────────────────────


def _pick_event(events: List[Dict[str, Any]], millm_probe_id: str, window: str) -> Optional[Dict[str, Any]]:
    mine = [e for e in events if str(e.get("probe_id")) == millm_probe_id and e.get("scored", True)]
    for event in mine:
        if event.get("window", "all") == window:
            return event
    return mine[0] if mine else None


def execute(
    db: Any,
    spot_check_id: str,
    *,
    reader_factory: Callable[[], Any] = None,
    judge_client: Any = None,
) -> Dict[str, Any]:
    """Run one spot-check. Returns ids and counts only (never text)."""
    from ..core.cancellation import OperatorCancelled, cancel_checker
    from .labeling_judge_health import is_job_level_failure
    from .probe_monitor_judge import (
        JudgePromptChanged,
        JudgeUnavailable,
        _client_for,
        ask_judge,
        parse_failure_refusal,
        resolve_api_key,
        spec_from_row,
    )

    row = db.query(ProbeSpotCheck).filter(ProbeSpotCheck.id == spot_check_id).first()
    if row is None:
        raise LookupError(f"spot-check {spot_check_id} not found")
    prompt = db.query(ProbeJudgePrompt).filter(ProbeJudgePrompt.id == row.prompt_id).first()
    spec = spec_from_row(prompt)
    if spec.content_hash != row.prompt_hash:
        raise JudgePromptChanged(
            f"spot-check {row.id} recorded prompt hash {row.prompt_hash[:12]}… but {spec.key} now hashes "
            f"to {spec.content_hash[:12]}…; refusing to judge with a prompt other than the one recorded"
        )

    cancel = cancel_checker("probe_spot_check", spot_check_id, db=db)
    row.status = "running"
    row.progress = 0.0
    db.commit()

    compared = (row.sample_spec or {}).get("compared") or []
    primary_millm = row.sample_spec["primary"]["millm_probe_id"]
    reader = (reader_factory or millm_read.SessionReader)()
    client = judge_client or _client_for(row.endpoint, resolve_api_key(db))
    replies: Dict[str, Optional[str]] = {}
    outcome = "completed"
    try:
        sampled = reader.sample(sample_request(row))
        turns = sampled["turns"]
        shortfall = {
            name: {
                "requested": int((sampled["strata"].get(name) or {}).get("requested", row.sample_spec["strata"][name])),
                "returned": int((sampled["strata"].get(name) or {}).get(
                    "returned", sum(1 for t in turns if t.get("stratum") == name))),
                "short": int((sampled["strata"].get(name) or {}).get("short", 0)),
            }
            for name in STRATA
        }
        row.sample_returned = {
            "turns": [
                {k: t.get(k) for k in ("session_id", "request_id", "turn_index", "event_id", "stratum")}
                for t in turns
            ],
            "strata": shortfall,
        }
        db.commit()

        for index, turn in enumerate(turns):
            # Polled at EVERY turn boundary (one indexed read): a turn is the finest unit a check can
            # abandon, and a throttled poll would let a cancelled check spend more judge calls on a shared
            # miLLM. Removing these two lines fails test_probe_spot_check_cancellation.py.
            cancel.poll_now()
            cancel.raise_if_cancelled("between sampled turns")
            transcript = reader.turn_text(str(turn["session_id"]), turn.get("turn_index"))
            if len(compared) > 1:
                events = reader.events_for_request(str(turn["request_id"]))
            else:
                events = []
            for probe in compared:
                if probe["millm_probe_id"] == primary_millm:
                    event = {
                        "id": turn.get("event_id"), "score": turn.get("score"),
                        "threshold": turn.get("threshold"), "verdict": turn.get("verdict"),
                        "threshold_revision": turn.get("threshold_revision"),
                        "window": turn.get("window") or row.sample_spec.get("window") or "all",
                    }
                else:
                    event = _pick_event(events, probe["millm_probe_id"], row.sample_spec.get("window") or "all")
                    if event is None:
                        continue
                window = str(event.get("window") or "all")
                item = ProbeSpotCheckItem(
                    spot_check_id=row.id,
                    session_id=str(turn["session_id"]),
                    request_id=str(turn["request_id"]),
                    turn_index=turn.get("turn_index"),
                    event_id=event.get("id"),
                    stratum=str(turn.get("stratum") or "not_fired"),
                    probe_id=probe["mistudio_probe_id"],
                    millm_probe_id=probe["millm_probe_id"],
                    window=window,
                    probe_score=event.get("score"),
                    threshold=event.get("threshold"),
                    threshold_revision=event.get("threshold_revision"),
                    probe_fired=bool(event.get("verdict")),
                    judge_saw=messages_for_window(window, transcript or {})[1],
                )
                if transcript is None:
                    item.not_recorded = True
                    db.add(item)
                    continue
                messages, _saw = messages_for_window(window, transcript)
                cache_key = json.dumps(messages, sort_keys=True)
                if cache_key not in replies:
                    try:
                        replies[cache_key] = ask_judge(client, row.model, spec, messages)
                    except Exception as exc:  # noqa: BLE001 - classified below
                        detail = f"{type(exc).__name__}: {exc}"
                        if is_job_level_failure(detail):
                            raise JudgeUnavailable(f"the judge endpoint is failing ({detail[:200]})") from exc
                        replies[cache_key] = None
                reply = replies[cache_key]
                answer = spec.parse(reply)
                item.judged_text = messages
                item.judge_reply = (reply or "")[:REPLY_CHARS] if reply is not None else None
                if answer is None:
                    item.parse_failed = True
                    row.parse_failures = (row.parse_failures or 0) + 1
                else:
                    item.judge_value = str(answer.value)[:64]
                    item.judge_score = answer.score
                    item.judge_positive = bool(answer.positive)
                db.add(item)
            row.progress = (index + 1) / max(len(turns), 1)
            db.commit()

        judged = db.query(ProbeSpotCheckItem).filter(
            ProbeSpotCheckItem.spot_check_id == row.id, ProbeSpotCheckItem.not_recorded.is_(False)
        ).count()
        refusal = parse_failure_refusal(row.parse_failures or 0, judged, row.parse_failure_limit)
        if refusal is not None:
            outcome = "failed"
            row.error_message = f"JudgeUnreliable: {refusal}"[:2000]
    except OperatorCancelled:
        outcome = "cancelled"
        row.partial = True
    except (JudgeUnavailable, millm_read.MiLLMUnavailable) as exc:
        outcome = "failed"
        row.partial = True
        row.error_message = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()

    items = db.query(ProbeSpotCheckItem).filter(ProbeSpotCheckItem.spot_check_id == row.id).all()
    row.metrics = report(items, row.sample_returned, seed=int(row.sample_spec["seed"]), partial=row.partial)
    row.status = outcome
    row.completed_at = utc_now()
    db.commit()
    logger.info("spot-check %s %s: %d items", row.id, outcome, len(items))
    return {"spot_check_id": row.id, "status": outcome, "items": len(items)}


# ── report (pure) ────────────────────────────────────────────────────────────────────────────


def _stats(pairs: List[Tuple[bool, bool]], seed: int) -> Dict[str, Any]:
    from .probe_monitor_metrics import agreement, agreement_ci, agreement_table, cohen_kappa

    return {
        "n": len(pairs),
        **agreement_table(pairs),
        "agreement": agreement(pairs),
        "agreement_ci": agreement_ci(pairs, resamples=BOOTSTRAP_RESAMPLES, seed=seed),
        "kappa": cohen_kappa(pairs),
    }


def report(items: Iterable[Any], sample_returned: Optional[Dict[str, Any]], *, seed: int, partial: bool = False) -> Dict[str, Any]:
    """The disagreement report (FR-10, D32), a pure function of the items."""
    by_probe: Dict[str, List[Any]] = {}
    for item in items:
        by_probe.setdefault(item.probe_id, []).append(item)
    probes: Dict[str, Any] = {}
    for probe_id, rows in sorted(by_probe.items()):
        judged = [r for r in rows if not r.not_recorded and not r.parse_failed and r.judge_positive is not None]
        pairs = [(bool(r.probe_fired), bool(r.judge_positive)) for r in judged]
        saw: Dict[str, int] = {}
        for r in rows:
            saw[r.judge_saw] = saw.get(r.judge_saw, 0) + 1
        probes[probe_id] = {
            "millm_probe_id": rows[0].millm_probe_id,
            "newest_turn_only": all(r.judge_saw == "newest_turn" for r in rows),
            "overall": _stats(pairs, seed),
            "strata": {
                name: _stats([(bool(r.probe_fired), bool(r.judge_positive)) for r in judged if r.stratum == name], seed)
                for name in STRATA
            },
            "n_parse_failed": sum(1 for r in rows if r.parse_failed),
            "n_not_recorded": sum(1 for r in rows if r.not_recorded),
            "judge_saw": saw,
            "threshold_revisions": sorted({r.threshold_revision for r in rows if r.threshold_revision is not None}),
            "disagreements": sum(1 for r in judged if bool(r.probe_fired) != bool(r.judge_positive)),
        }
    return {
        "statement": STATEMENT,
        "overall_note": OVERALL_NOTE,
        "partial": bool(partial),
        "shortfall": (sample_returned or {}).get("strata") or {},
        # D51: probes whose window the judge could not see whole are reported apart.
        "probes": {k: v for k, v in probes.items() if not v["newest_turn_only"]},
        "newest_turn_probes": {k: v for k, v in probes.items() if v["newest_turn_only"]},
    }


# ── export (D33, D57) ────────────────────────────────────────────────────────────────────────


def disagreement_rows(check: Any, items: Iterable[Any]) -> Iterator[DisagreementRow]:
    for item in items:
        if item.not_recorded or item.parse_failed or item.judge_positive is None or not item.judged_text:
            continue
        if bool(item.probe_fired) == bool(item.judge_positive):
            continue
        yield DisagreementRow(
            messages=item.judged_text,
            behaviour=check.behaviour,
            spot_check_id=check.id,
            session_id=item.session_id,
            request_id=item.request_id,
            turn_index=item.turn_index,
            event_id=item.event_id,
            stratum=item.stratum,
            probe_id=item.probe_id,
            millm_probe_id=item.millm_probe_id,
            probe_score=item.probe_score,
            threshold=item.threshold,
            probe_fired=bool(item.probe_fired),
            judge_positive=bool(item.judge_positive),
            judge_value=item.judge_value,
            judge_score=item.judge_score,
            prompt_key=check.prompt_key,
            prompt_hash=check.prompt_hash,
            window=item.window,
            judge_saw=item.judge_saw,
        )


def export_jsonl(check: Any, items: Iterable[Any]) -> Iterator[str]:
    for row in disagreement_rows(check, items):
        yield row.model_dump_json() + "\n"


# ── retention (D39) ──────────────────────────────────────────────────────────────────────────


def expire_excerpts(db: Any) -> int:
    """Null the judged text and replies of every unpinned check past `excerpts_expire_at`."""
    now = utc_now()
    ids = [
        r[0]
        for r in db.query(ProbeSpotCheck.id)
        .filter(ProbeSpotCheck.pinned.is_(False), ProbeSpotCheck.excerpts_expire_at < now)
        .all()
    ]
    if not ids:
        return 0
    n = (
        db.query(ProbeSpotCheckItem)
        .filter(ProbeSpotCheckItem.spot_check_id.in_(ids), ProbeSpotCheckItem.judged_text.isnot(None))
        .update({"judged_text": None, "judge_reply": None}, synchronize_session=False)
    )
    db.commit()
    return int(n)


# ── serialisers ──────────────────────────────────────────────────────────────────────────────


def check_out(row: ProbeSpotCheck) -> Dict[str, Any]:
    iso = lambda v: v.isoformat() if v is not None else None  # noqa: E731
    return {
        "id": row.id, "behaviour": row.behaviour, "behaviour_version": row.behaviour_version,
        "prompt_id": row.prompt_id, "prompt_key": row.prompt_key, "prompt_hash": row.prompt_hash,
        "endpoint": row.endpoint, "model": row.model, "sample_spec": row.sample_spec,
        "sample_returned": row.sample_returned, "status": row.status, "progress": row.progress,
        "error_message": row.error_message, "partial": row.partial, "parse_failures": row.parse_failures,
        "parse_failure_limit": row.parse_failure_limit, "metrics": row.metrics, "pinned": row.pinned,
        "excerpts_expire_at": iso(row.excerpts_expire_at), "created_by": row.created_by,
        "created_at": iso(row.created_at), "completed_at": iso(row.completed_at),
    }


def item_out(item: ProbeSpotCheckItem) -> Dict[str, Any]:
    return {
        "id": item.id, "session_id": item.session_id, "request_id": item.request_id,
        "turn_index": item.turn_index, "event_id": item.event_id, "stratum": item.stratum,
        "probe_id": item.probe_id, "millm_probe_id": item.millm_probe_id, "window": item.window,
        "judge_saw": item.judge_saw, "probe_score": item.probe_score, "threshold": item.threshold,
        "threshold_revision": item.threshold_revision, "probe_fired": item.probe_fired,
        "not_recorded": item.not_recorded, "judged_text": item.judged_text, "judge_reply": item.judge_reply,
        "judge_value": item.judge_value, "judge_score": item.judge_score,
        "judge_positive": item.judge_positive, "parse_failed": item.parse_failed,
        "disagrees": (not item.not_recorded and not item.parse_failed and item.judge_positive is not None
                      and bool(item.probe_fired) != bool(item.judge_positive)),
    }


async def count_items(db: AsyncSession, check_id: str) -> int:
    return int((await db.execute(
        select(func.count()).select_from(ProbeSpotCheckItem).where(ProbeSpotCheckItem.spot_check_id == check_id)
    )).scalar_one())
