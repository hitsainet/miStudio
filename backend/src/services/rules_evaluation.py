"""Rules ∪ probe: one behaviour's rule set and one probe, judged on the same rows (Feature 040 S3; FR-29, FR-32 – FR-38;
FTDD §6.4, §6.5; FTID §3.6, §7.4).

Three surfaces share this module:

* `validate_request` / `evaluate_one` — the synchronous single-text tools (FR-32). Nothing stored or logged.
* `request` / `read` / `list_for` / `cancel` — the evaluation's API half: identity, the partial-unique insert
  (a duplicate returns the existing row, 040-T29), a `task_queue` row, dispatch AFTER commit (FTID §3.6).
* `run` — the task's body (`workers/rules_evaluation_tasks.py`), on the `rules` queue: per view, rule verdicts
  from the cache or the engine; the three detectors per row; producer agreement; the unknown share; the union
  calibration; the body written ONCE.

**Three detectors per row (FR-33).** ``rules = rule_match``; ``probe = fires`` (the row's decision of FR-15,
`probe_breakdown.recorded_decisions`); ``union = rule_match OR (probe AND NOT rule_veto)``. A row whose rule
verdict is ``unknown`` (over the text budget, or depending on operator code no runner can run) has no match
and no veto, so the probe decides (FR-35). A behaviour with no rule set, or a set with no vetoes, is valid:
reason ``no_rule_set`` / ``no_veto_signals``, and union equals probe by construction.

**Producer columns are compared, never preferred (FR-34).** A view that declares producer `rule_columns`
gets agreement counts per disagreement kind; miStudio's verdict is the one used — except FR-31's single
exception: where miStudio's verdict is unknown because operator code is pending, a producer verdict for
EXACTLY this rule-set digest is used and labelled ``producer_computed``.

**The union spends the FPR budget as a union (FR-36; §6.5).** On the probe's calibration negatives aligned BY
POSITION with rule verdicts on the calibration view: vetoed negatives are removed, the rules' own false
positives are charged first, and the probe's bar is cut on what is left with `calibrate`'s order statistic.
Reported, never written (OQ-040-4 (a)); its `next` is a recalibration PREVIEW at the probe-alone FPR.

⚠ NO ROW TEXT LEAVES THIS MODULE. Bodies hold counts, rates, ids, digests and positions; logs hold ids,
counts and digests (FTID §13). The text is read, evaluated and dropped, chunk by chunk.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .detector_identity import identity, jcs_bytes, round6, sha256_hex
from .detector_next import (
    DetectorRefused,
    next_evaluate,
    next_get_rules_evaluation,
    next_recalibrate_preview,
    next_recut_for_positions,
    next_rules_evaluation,
)

logger = logging.getLogger(__name__)

#: Bumped when any figure of a stored result would be computed differently: every older result reads stale.
RESULT_ALGORITHM = 1
RESULT_FORMAT = "mistudio.rules-evaluation/1"
CACHE_DIR = "rule_verdicts"
#: Rows read and evaluated between cancellation checks and heartbeats (TI-19).
CHUNK_ROWS = 1000
#: Unknown-verdict codes stored per row in the cache.
UNKNOWN_NONE, UNKNOWN_BUDGET, UNKNOWN_PENDING = 0, 1, 2
UNKNOWN_REASONS = {UNKNOWN_BUDGET: "rule_budget_exceeded", UNKNOWN_PENDING: "runner_unavailable"}
#: S4 (FTDD C.3): a code rule that failed in the runner makes its row unknown WITH the contract reason (3 – 10).
CODE_REASON_CODES = {reason: 3 + i for i, reason in enumerate((
    "rule_timeout", "rule_resource_limit", "rule_worker_unavailable", "rule_error", "rule_load_refused",
    "rule_output_invalid", "rule_input_too_large", "rule_not_approved"))}
ALL_UNKNOWN_REASONS = {**UNKNOWN_REASONS, **{code: reason for reason, code in CODE_REASON_CODES.items()}}
#: ⚠ NEVER CACHED (040-A6): a host fault must not freeze an outage into a verdict, and an approval can change.
UNCACHEABLE_REASONS = frozenset({"rule_worker_unavailable", "rule_timeout", "rule_not_approved"})
#: The settings a result reads, recorded in its inputs (TI-22) — the identity moves with each.
SETTINGS_READ = ("rule_text_max_code_points", "rule_pattern_max_mem_bytes", "rule_set_max_signals",
                 "rule_set_max_cases", "rule_set_max_bytes", "rule_signals_max_messages",
                 "rule_unknown_share_max", "breakdown_min_subgroup_rows", "rules_evaluation_max_bytes")


def _effective() -> Any:
    from ..core.runtime_settings import effective

    return effective()


def _limits(values: Dict[str, Any]) -> Any:
    from ..rule_signals import Limits

    return Limits(max_mem=int(values["rule_pattern_max_mem_bytes"]),
                  text_max_code_points=int(values["rule_text_max_code_points"]),
                  max_signals=int(values["rule_set_max_signals"]), max_cases=int(values["rule_set_max_cases"]),
                  max_bytes=int(values["rule_set_max_bytes"]), max_messages=int(values["rule_signals_max_messages"]))


def _settings_read(settings: Any = None) -> Dict[str, Any]:
    """Every setting a result reads, by name (explicit reads: `test_settings_readers.py` finds each)."""
    settings = settings if settings is not None else _effective()
    out = {
        "rule_text_max_code_points": settings.rule_text_max_code_points,
        "rule_pattern_max_mem_bytes": settings.rule_pattern_max_mem_bytes,
        "rule_set_max_signals": settings.rule_set_max_signals,
        "rule_set_max_cases": settings.rule_set_max_cases,
        "rule_set_max_bytes": settings.rule_set_max_bytes,
        "rule_signals_max_messages": settings.rule_signals_max_messages,
        "rule_unknown_share_max": settings.rule_unknown_share_max,
        "breakdown_min_subgroup_rows": settings.breakdown_min_subgroup_rows,
        "rules_evaluation_max_bytes": settings.rules_evaluation_max_bytes,
    }
    assert tuple(out) == SETTINGS_READ
    return out


# ── operator code (S4; FTDD Appendix C.3, C.6) ──────────────────────────────────────────────────


def runner_identity_for(document: Optional[Dict[str, Any]], settings: Any = None) -> Optional[Dict[str, Any]]:
    """The runner identity a set's verdicts depend on: None for a set with no code rule (its digest is S3's), else
    `code_rule_runner.identity` — ``{"available": False}`` with no runner (Review R1-F8)."""
    from . import code_rule_runner, code_rules

    if not document or not code_rules.referenced_hashes(document):
        return None
    return code_rule_runner.identity(settings if settings is not None else _effective())


def engine_for(document: Optional[Dict[str, Any]], limits: Any, settings: Any = None) -> str:
    """THE engine digest of a set: the cache key, the stored result's `engine_digest`, and its staleness."""
    from .. import rule_signals

    return rule_signals.engine_digest(limits, runner_identity_for(document, settings))


async def code_context(db: Any, body: Dict[str, Any], document: Dict[str, Any], settings: Any, *,
                       actor: str, lane: str = "interactive") -> Tuple[Any, Dict[str, Any]]:
    """`(runner, report)` for a synchronous call over ``document`` (C.5, C.6). Stores inline ``code_rules``
    documents. With no runner (or code rules off) the runner is `UnavailableRunner` and operator code is pending
    (D41). With one, every referenced hash needs its document (`code_rule_document_missing`), the set's code
    signals must fit their documents (`check_code_rules`) and every hash must be APPROVED, read fresh now
    (`code_rule_not_approved`, `next` = request_code_rule_approval)."""
    from .. import rule_signals
    from ..rule_runner import contract as runner_contract
    from . import code_rule_runner, code_rules

    shas = code_rules.referenced_hashes(document)
    inline = code_rules.inline_documents(body.get("code_rules"), document)
    if inline:
        await code_rules.store(db, inline, actor)
        await db.commit()
    if not shas:
        return rule_signals.UnavailableRunner(), {}
    if not code_rule_runner.enabled(settings):
        return rule_signals.UnavailableRunner(), {"runner": "unavailable", "code_sha256": shas}
    documents = {**(await code_rules.documents_for(db, shas)), **inline}
    code_rules.require_documents(shas, documents)
    try:
        rule_signals.check_code_rules(rule_signals.load(document, rule_set_store_limits(settings)),
                                      {sha: doc for sha, (doc, _c) in documents.items()},
                                      runner_format=runner_contract.FORMAT)
    except rule_signals.RuleSetRefused as exc:
        raise DetectorRefused(422, exc.code, f"the rule set cannot be used: {exc.message}",
                              {"field": "rule_set", **exc.details}) from exc
    found = await code_rules.states(db, shas)
    code_rules.require_approved(found)
    runner = code_rule_runner.runner_for(settings, documents={sha: c for sha, (_d, c) in documents.items()},
                                         approved=None, lane=lane)
    return runner, {"runner": "available", "code_rules": {sha: found[sha] for sha in shas}}


def rule_set_store_limits(settings: Any) -> Any:
    from . import rule_set_store

    return rule_set_store.engine_limits(settings)


# ── the rule set a request names ──────────────────────────────────────────────────────────────────


async def _version(db: Any, key: str, number: Optional[int]) -> Tuple[Any, Any]:
    from .detector_success_report import resolve_version

    return await resolve_version(db, key, number)


async def _version_document(db: Any, version: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    from . import rule_set_store

    if version.rule_set_digest is None:
        return None, None
    row = await rule_set_store.get_by_digest(db, version.rule_set_digest)
    if row is None:
        raise DetectorRefused(500, "rule_set_missing",
                              f"rule set {version.rule_set_digest[:12]}… named by {version.key} v{version.version} "
                              f"is not stored", {"rule_set_digest": version.rule_set_digest})
    return rule_set_store.document_of(row)[0], version.rule_set_digest


async def _choice(db: Any, body: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """`(document, digest, source)` for a `{behaviour_key, version?} | {rule_set}` body."""
    from ..rule_signals import canonical

    if body.get("rule_set") is not None:
        document = body["rule_set"]
        try:
            return document, canonical.identity(document), {"kind": "inline"}
        except canonical.CanonicalRefused as exc:
            raise DetectorRefused(422, "invalid_field", f"the rule set cannot be canonicalised: {exc}",
                                  {"field": "rule_set", "rule": "not_canonicalisable"}) from exc
    head, version = await _version(db, body["behaviour_key"], body.get("version"))
    document, digest = await _version_document(db, version)
    if document is None:
        raise DetectorRefused(409, "no_rule_set",
                              f"version {version.version} of {head.key!r} names no rule set; attach one with a new "
                              f"version, or send a rule set inline",
                              {"behaviour_key": head.key, "version": version.version})
    return document, digest, {"kind": "behaviour", "behaviour_key": head.key, "version": version.version}


async def validate_request(db: Any, body: Dict[str, Any]) -> Dict[str, Any]:
    """FR-32 `validate_rule_set`: `{valid, …}`. A set that fails is an ANSWER (`valid: false`), not an error."""
    from .. import rule_signals
    from . import rule_set_store

    document, digest, source = await _choice(db, body)
    settings = _effective()
    limits = rule_set_store.engine_limits(settings)
    out: Dict[str, Any] = {"rule_set_digest": digest, "source": source,
                           "engine_version": rule_signals.ENGINE_VERSION,
                           "engine_digest": engine_for(document, limits, settings)}
    try:
        record = rule_set_store.validate_for_use(document, limits=limits)
        _runner, code = await code_context(db, body, document, settings, actor=_actor(body))
    except DetectorRefused as refusal:
        details = {k: v for k, v in refusal.details.items() if k != "field"}
        answer = {**out, "valid": False, "code": refusal.code, "message": refusal.message, **details}
        if refusal.next:
            answer["next"] = refusal.next
        return answer
    record = {k: v for k, v in record.items() if k not in ("status", "engine_digest")}
    return {**out, "valid": True, **record, **code}


def _actor(body: Dict[str, Any]) -> str:
    return str(body.get("_actor") or "operator")


async def evaluate_one(db: Any, body: Dict[str, Any]) -> Dict[str, Any]:
    """FR-32 `evaluate_rule_signals`: one text or conversation, synchronously. Caps: the input's total code
    points (`rule_text_max_code_points`) and its message count (`rule_signals_max_messages`). Nothing is
    stored or logged."""
    from .. import rule_signals
    from ..rule_signals import reduce as reducer
    from . import rule_set_store

    settings = _effective()
    messages = body.get("messages")
    if messages is not None and len(messages) > int(settings.rule_signals_max_messages):
        raise DetectorRefused(422, "over_cap",
                              f"{len(messages)} messages; the runtime setting rule_signals_max_messages allows "
                              f"{settings.rule_signals_max_messages}",
                              {"setting": "rule_signals_max_messages", "value": settings.rule_signals_max_messages,
                               "got": len(messages)})
    total = len(body["text"]) if body.get("text") is not None else sum(
        len(m.get("content") or "") for m in messages or [])
    if total > int(settings.rule_text_max_code_points):
        raise DetectorRefused(422, "over_cap",
                              f"the input is {total} characters; the runtime setting rule_text_max_code_points "
                              f"allows {settings.rule_text_max_code_points}",
                              {"setting": "rule_text_max_code_points", "value": settings.rule_text_max_code_points,
                               "got": total})
    scope = body.get("scope") or "all"
    scope = reducer.CONTRACT_WINDOW_SCOPES.get(scope, scope)
    if scope not in reducer.SCOPES:
        raise DetectorRefused(422, "invalid_field",
                              f"scope {body.get('scope')!r} is not one of {sorted(set(reducer.SCOPES) | set(reducer.CONTRACT_WINDOW_SCOPES))}",
                              {"field": "scope", "rule": "unknown_scope"})
    document, digest, source = await _choice(db, body)
    limits = rule_set_store.engine_limits(settings)
    rule_set_store.validate_for_use(document, limits=limits)  # FR-29: at every use
    runner, code = await code_context(db, body, document, settings, actor=_actor(body))
    compiled = rule_signals.load(document, limits)
    if getattr(runner, "available", False):
        from starlette.concurrency import run_in_threadpool

        # The runner client is synchronous HTTP: off the event loop.
        verdict = await run_in_threadpool(evaluate_input, compiled, limits, text=body.get("text"),
                                          messages=messages, scope=scope, runner=runner)
    else:
        verdict = evaluate_input(compiled, limits, text=body.get("text"), messages=messages, scope=scope,
                                 runner=runner)
    out = verdict.as_dict()
    out.update({"rule_set_digest": digest, "engine_digest": engine_for(document, limits, settings), "scope": scope,
                "source": source, "has_vetoes": compiled.has_vetoes,
                "match_expressions": list(compiled.match_expressions), **code})
    notes = [] if compiled.has_vetoes else ["no_veto_signals"]
    if compiled.match_veto_conditions:
        notes.append("match_includes_veto_condition")
    if notes:
        out["notes"] = notes
    return out


def evaluate_input(compiled: Any, limits: Any, *, text: Optional[str], messages: Optional[List[Dict[str, Any]]],
                   scope: str, runner: Any) -> Any:
    """THE single-text rule evaluation (FR-32): one text or conversation through the engine. `evaluate_one` (the
    synchronous tools) and Feature 041's family scoring (`probe_family_scoring.score_family`) both call it."""
    from .. import rule_signals

    try:
        return rule_signals.evaluate(compiled, limits, text=text, messages=messages, scope=scope, runner=runner)
    except rule_signals.RuleSignalError as exc:
        raise DetectorRefused(422, "rule_signal_error",
                              f"signal {exc.signal_id!r} ({exc.validator}) failed on this input",
                              {"signal": exc.signal_id, "validator": exc.validator}) from None


def stored_rule_set_sync(db: Any, digest: str) -> Dict[str, Any]:
    """The stored document of a rule-set digest (a worker's sync session)."""
    from ..models.rule_set import RuleSet
    from . import rule_set_store

    row = db.query(RuleSet).filter(RuleSet.digest == digest).first()
    if row is None:
        raise DetectorRefused(500, "rule_set_missing", f"rule set {digest[:12]}… is not stored",
                              {"rule_set_digest": digest})
    return rule_set_store.document_of(row)[0]


def interactive_runner_sync(db: Any, document: Dict[str, Any], settings: Any) -> Any:
    """The runner one synchronous evaluation in a worker uses: none needed for a set without operator code; with
    code and no runner, `UnavailableRunner` (the rows read PENDING, never false — D41); with a runner, every
    approval read FRESH per call (C.6), so a revoked rule reads `rule_not_approved`, never a verdict."""
    from .. import rule_signals
    from . import code_rule_runner, code_rules

    shas = code_rules.referenced_hashes(document)
    if not shas or not code_rule_runner.enabled(settings):
        return rule_signals.UnavailableRunner()
    documents = code_rules.documents_for_sync(db, shas)
    return code_rule_runner.runner_for(settings, documents={sha: c for sha, (_d, c) in documents.items()},
                                       approved=lambda sha: code_rules.approved_now(db, sha), lane="interactive")


def evaluate_text_sync(db: Any, rule_set_digest: str, *, text: Optional[str],
                       messages: Optional[List[Dict[str, Any]]], scope: str, settings: Any = None) -> Dict[str, Any]:
    """One input against a STORED rule set from a worker (sync session): `{rule_match, rule_veto, unknown, reason,
    rule_set_digest, engine_digest}` — counts and booleans only, never the text. FR-29: validated at every use."""
    from .. import rule_signals
    from ..rule_signals import reduce as reducer
    from . import rule_set_store

    settings = settings if settings is not None else _effective()
    document = stored_rule_set_sync(db, rule_set_digest)
    limits = rule_set_store.engine_limits(settings)
    rule_set_store.validate_for_use(document, limits=limits)
    compiled = rule_signals.load(document, limits)
    scope = reducer.CONTRACT_WINDOW_SCOPES.get(scope, scope)
    verdict = evaluate_input(compiled, limits, text=text, messages=list(messages) if messages is not None else None,
                             scope=scope, runner=interactive_runner_sync(db, document, settings))
    return {"rule_match": bool(verdict.rule_match), "rule_veto": bool(verdict.rule_veto),
            "unknown": bool(verdict.unknown), "reason": verdict.reason, "rule_set_digest": rule_set_digest,
            "engine_digest": engine_for(document, limits, settings), "has_vetoes": bool(compiled.has_vetoes)}


# ── reading a view's text (FTDD §4.5: live, with the drift check) ───────────────────────────────


def split_rows(view: Any) -> int:
    """The split's row count at registration: every row lands in exactly one of `counts`' five buckets."""
    counts = view.counts or {}
    return int(sum(int(counts.get(k) or 0) for k in ("positive", "negative", "excluded", "filtered_out", "unparseable")))


def open_view_split(db_dataset: Any, view: Any) -> Any:
    """The view's split, refused `split_changed` when it no longer matches what was registered."""
    from .probe_monitor_service import open_split, resolve_dataset_path

    data = open_split(resolve_dataset_path(db_dataset.raw_path), view.split)
    expected = split_rows(view)
    if expected and len(data) != expected:
        raise DetectorRefused(409, "split_changed",
                              f"view {view.id}'s split has {len(data)} rows; it had {expected} when registered",
                              {"view_id": view.id, "rows_now": len(data), "rows_registered": expected})
    commit = (view.dataset_version or {}).get("commit")
    revision = getattr(db_dataset, "revision", None)
    if commit and revision and commit != revision:
        raise DetectorRefused(409, "split_changed",
                              f"view {view.id} was registered at commit {commit}; the download is at {revision}",
                              {"view_id": view.id, "commit_registered": commit, "commit_now": revision})
    return data


def reading_of(view: Any, scope: str) -> Dict[str, Any]:
    """What the rules read on this view: a declared `text_column` whole, else the probe's span of the input."""
    if view.text_column:
        return {"column": view.text_column, "scope": "all", "source": "text_column"}
    return {"column": view.input_column, "scope": scope, "source": "input_column"}


def _messages(value: Any) -> List[Dict[str, Any]]:
    from .probe_monitor_inputs import parse_input

    return list(parse_input(value).messages)


# ── the verdict cache (040-T13, TI-17) ───────────────────────────────────────────────────────────


def cache_root() -> Path:
    from ..core.config import settings

    return Path(settings.data_dir) / CACHE_DIR


@dataclass
class ViewVerdicts:
    rule_match: np.ndarray
    rule_veto: np.ndarray
    unknown: np.ndarray
    n_rows: int
    cached: bool
    #: S4: per code-rule signal id, `{"fired": n, "reasons": {reason: n}}` over the view's rows.
    code: Optional[Dict[str, Dict[str, Any]]] = None
    #: False when a row carries a host fault or `rule_not_approved` (040-A6): such verdicts are never cached.
    cacheable: bool = True


def view_cache_dir(view_id: str) -> Path:
    """The view's verdict-cache directory; refuses (ValueError) an id that would escape `cache_root()`."""
    from ..utils.file_utils import resolve_under

    return resolve_under(cache_root(), view_id)


def _cache_dir(view_id: str, set_digest: str, engine: str, reading: str) -> Path:
    return view_cache_dir(view_id) / f"{set_digest[:24]}__{engine[:24]}__{reading[:24]}"


def _load_cached(path: Path, n_rows: int) -> Optional[ViewVerdicts]:
    try:
        arrays = [np.load(path / f"{name}.npy", allow_pickle=False) for name in ("rule_match", "rule_veto", "unknown")]
    except (OSError, ValueError):
        return None
    if any(int(a.shape[0]) != n_rows for a in arrays):
        return None
    code = None
    if (path / "code_rules.json").is_file():
        try:
            code = json.loads((path / "code_rules.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return ViewVerdicts(arrays[0].astype(bool), arrays[1].astype(bool), arrays[2].astype(np.uint8), n_rows, True,
                        code=code)


def _unknown_code(reason: Optional[str]) -> int:
    if reason == "rule_budget_exceeded":
        return UNKNOWN_BUDGET
    return CODE_REASON_CODES.get(str(reason), UNKNOWN_PENDING)


def compute_view_verdicts(view: Any, data: Any, compiled: Any, limits: Any, scope: str, *,
                          should_stop: Callable[[], None], beat: Callable[[int, int], None],
                          runner: Any = None, batch_texts: int = 256) -> ViewVerdicts:
    """Every split row through the engine, chunk by chunk (TI-19). The text is never kept.

    S4 (C.3): `rule_signals.evaluate_many` per chunk, so each code-rule signal reaches the runner once per batch of
    up to ``batch_texts`` texts; a row whose code rule failed is unknown WITH the contract reason."""
    from .. import rule_signals

    reading = reading_of(view, scope)
    column = data[reading["column"]] if reading["column"] in getattr(data, "column_names", []) else None
    if column is None:
        raise DetectorRefused(409, "column_absent",
                              f"column {reading['column']!r} is not in view {view.id}'s split",
                              {"view_id": view.id, "column": reading["column"]})
    n = len(data)
    match = np.zeros(n, dtype=np.uint8)
    veto = np.zeros(n, dtype=np.uint8)
    unknown = np.zeros(n, dtype=np.uint8)
    code: Dict[str, Dict[str, Any]] = {s["id"]: {"code_sha256": s["code_sha256"], "fired": 0, "reasons": {}}
                                       for s in compiled.signals if s["kind"] == "code_rule"}
    transient = False
    for start in range(0, n, CHUNK_ROWS):
        should_stop()
        end = min(n, start + CHUNK_ROWS)
        inputs = [_messages(value) for value in data[reading["column"]][start:end]]
        verdicts = rule_signals.evaluate_many(compiled, limits, inputs, scope=reading["scope"], runner=runner,
                                              spans=False, batch_texts=batch_texts)
        for i, verdict in enumerate(verdicts, start=start):
            for sid, result in verdict.code.items():
                if "reason" in result:
                    # ⚠ Review R2 (S4): a host fault on ANY code rule of a row makes the view uncacheable — even when
                    # the row's verdict names another code rule's (cacheable) reason first: the per-code-rule tally
                    # would otherwise freeze the outage into the cache.
                    transient = transient or result["reason"] in UNCACHEABLE_REASONS
                    reasons = code[sid]["reasons"]
                    reasons[result["reason"]] = reasons.get(result["reason"], 0) + 1
                elif int(result.get("count", 0)) >= 1:
                    code[sid]["fired"] += 1
            if verdict.unknown:
                unknown[i] = _unknown_code(verdict.reason)
            else:
                match[i] = verdict.rule_match
                veto[i] = verdict.rule_veto
        beat(end, n)
    faults = {CODE_REASON_CODES[r] for r in UNCACHEABLE_REASONS}
    cacheable = not transient and not bool(np.isin(unknown, list(faults)).any())
    return ViewVerdicts(match.astype(bool), veto.astype(bool), unknown, n, False, code=code or None,
                        cacheable=cacheable)


def publish_cache(path: Path, verdicts: ViewVerdicts) -> int:
    """Write to a temporary directory and rename into place (atomic on one filesystem). Returns bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        np.save(tmp / "rule_match.npy", verdicts.rule_match.astype(np.uint8), allow_pickle=False)
        np.save(tmp / "rule_veto.npy", verdicts.rule_veto.astype(np.uint8), allow_pickle=False)
        np.save(tmp / "unknown.npy", verdicts.unknown.astype(np.uint8), allow_pickle=False)
        if verdicts.code:
            (tmp / "code_rules.json").write_text(json.dumps(verdicts.code, sort_keys=True), encoding="utf-8")
        size = sum(f.stat().st_size for f in tmp.iterdir())
        if path.exists():
            shutil.rmtree(tmp)
            return size
        os.rename(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return size


def view_verdicts(db: Any, view: Any, data: Any, compiled: Any, limits: Any, scope: str, *, set_digest: str,
                  engine: str, evaluation_id: str, should_stop: Callable[[], None],
                  beat: Callable[[int, int], None], runner: Any = None, batch_texts: int = 256) -> ViewVerdicts:
    """From the cache when (view, set, engine, reading) is there; else computed, published, indexed — unless a row
    carries a host fault or `rule_not_approved`: then NOTHING is cached (040-A6), so an outage or a revocation is
    never frozen into a verdict."""
    from sqlalchemy.dialects.postgresql import insert

    from ..core.clock import utc_now
    from ..models.rules_evaluation import RuleVerdictCache

    reading = sha256_hex(jcs_bytes(reading_of(view, scope)))
    key = dict(view_id=view.id, rule_set_digest=set_digest, engine_digest=engine, reading_digest=reading)
    row = db.query(RuleVerdictCache).filter_by(**key).first()
    path = _cache_dir(view.id, set_digest, engine, reading)
    if row is not None:
        cached = _load_cached(Path(row.path), len(data))
        if cached is not None:
            row.last_used_at, row.in_use_by = utc_now(), evaluation_id
            db.commit()
            return cached
    verdicts = compute_view_verdicts(view, data, compiled, limits, scope, should_stop=should_stop, beat=beat,
                                     runner=runner, batch_texts=batch_texts)
    if not verdicts.cacheable:
        return verdicts
    size = publish_cache(path, verdicts)
    db.execute(insert(RuleVerdictCache).values(
        **key, n_rows=verdicts.n_rows, bytes=size, path=str(path), created_at=utc_now(), last_used_at=utc_now(),
        in_use_by=evaluation_id,
    ).on_conflict_do_nothing())
    db.commit()
    return verdicts


def prune_cache(db: Any, *, max_bytes: int, max_age_days: int, now: Any = None) -> Dict[str, int]:
    """LRU + age pruning that NEVER removes an entry `in_use_by` a running evaluation (040-T36)."""
    from datetime import timedelta

    from ..core.clock import utc_now
    from ..models.rules_evaluation import ProbeRulesEvaluation, RuleVerdictCache

    now = now or utc_now()
    running = {r[0] for r in db.query(ProbeRulesEvaluation.id).filter(
        ProbeRulesEvaluation.status.in_(("pending", "running"))).all()}
    rows = db.query(RuleVerdictCache).order_by(RuleVerdictCache.last_used_at.asc()).all()
    total = sum(int(r.bytes) for r in rows)
    removed = {"age": 0, "size": 0}
    for row in rows:
        if row.in_use_by in running:
            continue
        last = row.last_used_at if row.last_used_at.tzinfo else row.last_used_at.replace(tzinfo=now.tzinfo)
        reason = "age" if now - last > timedelta(days=max_age_days) else ("size" if total > max_bytes else None)
        if reason is None:
            continue
        shutil.rmtree(row.path, ignore_errors=True)
        total -= int(row.bytes)
        db.delete(row)
        removed[reason] += 1
    db.commit()
    return removed


def purge_view(view_id: str) -> Optional[str]:
    """FTASKS 20.3: a purged view's cached verdicts go with it (its index rows are swept as missing files)."""
    path = view_cache_dir(view_id)
    if not path.exists():
        return None
    shutil.rmtree(path, ignore_errors=True)
    return str(path)


# ── figures ──────────────────────────────────────────────────────────────────────────────────────


def detectors(match: np.ndarray, veto: np.ndarray, unknown: np.ndarray, fires: np.ndarray) -> Dict[str, np.ndarray]:
    """FR-33's three detectors. ⚠ An unknown verdict is NO match and NO veto: the probe decides (FR-35)."""
    known = unknown == UNKNOWN_NONE
    rule_match = match & known
    rule_veto = veto & known
    return {"rules_alone": rule_match, "probe_alone": fires,
            "rules_or_probe": rule_match | (fires & ~rule_veto)}


def rates(flagged: np.ndarray, labels: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    from .probe_monitor_metrics import wilson_interval

    mask = np.ones(labels.shape[0], dtype=bool) if mask is None else mask
    pos, neg = (labels == 1) & mask, (labels == 0) & mask
    p, n = int(pos.sum()), int(neg.sum())
    tp, fp = int((flagged & pos).sum()), int((flagged & neg).sum())
    r_lo, r_hi = wilson_interval(tp, p) if p else (None, None)
    f_lo, f_hi = wilson_interval(fp, n) if n else (None, None)
    return {"positives": {"n": p, "flagged": tp, "recall": round6(tp / p) if p else None,
                          "ci": [round6(r_lo), round6(r_hi)] if p else None},
            "negatives": {"n": n, "flagged": fp, "fpr": round6(fp / n) if n else None,
                          "ci": [round6(f_lo), round6(f_hi)] if n else None}}


def agreement(mine: np.ndarray, theirs: np.ndarray) -> Dict[str, int]:
    return {"both": int((mine & theirs).sum()), "neither": int((~mine & ~theirs).sum()),
            "mistudio_only": int((mine & ~theirs).sum()), "producer_only": int((~mine & theirs).sum())}


def _bool_snapshot(entry: Dict[str, Any], column: str) -> np.ndarray:
    from . import view_columns

    snap = view_columns.load(entry, column)
    truthy = np.array([v is True or v == 1 or v == "true" for v in snap.values], dtype=bool)
    return truthy[np.asarray(snap.codes)]


# ── union calibration (FR-36; FTDD §6.5) ─────────────────────────────────────────────────────────


def _cut(scores: np.ndarray, allowed: int) -> Optional[float]:
    """The order-statistic bar admitting `allowed` of `scores` (memory `calibrate-is-an-order-statistic`):
    `calibrate` itself, with a target whose floor is exactly `allowed`."""
    from .probe_monitor_trainer import calibrate

    m = int(scores.shape[0])
    if allowed <= 0 or m == 0:
        return None
    if allowed >= m:
        return float(np.min(scores))
    return calibrate(scores.tolist(), target_fpr=(allowed + 0.5) / m, source="rules_union").threshold


def union_cut(scores: np.ndarray, match: np.ndarray, veto: np.ndarray, target_fpr: float) -> Dict[str, Any]:
    """One union bar on one set of negatives (global, or one band)."""
    n = int(scores.shape[0])
    budget = int(math.floor(target_fpr * n + 1e-12))
    rule_fp = int(match.sum())
    remaining = ~match & ~veto
    allowed = budget - rule_fp
    bar = _cut(scores[remaining], allowed)
    probe_fp = int((scores[remaining] >= bar).sum()) if bar is not None else 0
    probe_alone = int((scores >= bar).sum()) / n if (bar is not None and n) else 0.0
    return {"threshold": bar, "n_negatives": n, "budget": budget, "rule_false_positives": rule_fp,
            "vetoed": int((veto & ~match).sum()), "probe_false_positives_allowed": max(allowed, 0),
            "probe_false_positives": probe_fp,
            "realised_union_fpr": round6((rule_fp + probe_fp) / n) if n else None,
            "rule_share": round6(rule_fp / n) if n else None, "probe_alone_fpr_at_bar": round6(probe_alone)}


def union_rows(match: np.ndarray, veto: np.ndarray, unknown: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`(rule_match, rule_veto)` as the union spends them, for rows already aligned BY POSITION: an unknown verdict
    is no match and no veto (FR-35), and a matched row is the rules' own flag, never a veto. Feature 041's blend
    candidates (`rules_or_*`) call this and `union_cut`, so a blend's union is cut exactly as a probe's."""
    det = detectors(match, veto, unknown, np.zeros(np.asarray(match).shape[0], dtype=bool))
    rule_match = det["rules_alone"]
    return rule_match, np.asarray(veto, dtype=bool) & (np.asarray(unknown) == UNKNOWN_NONE) & ~rule_match


def refuse_rules_over_budget(match: np.ndarray, target_fpr: float) -> None:
    """FR-36: the rules alone flagging the target share of the negatives (or more) leave the detector nothing to
    spend — `rules_spend_budget`."""
    n = int(np.asarray(match).shape[0])
    rule_fpr = int(np.asarray(match).sum()) / n if n else 0.0
    if rule_fpr >= target_fpr:
        raise DetectorRefused(
            422, "rules_spend_budget",
            f"the rules alone flag {rule_fpr:.6f} of the calibration negatives, which is not below the target "
            f"{target_fpr}; the probe has nothing left to spend",
            {"rule_fpr": round6(rule_fpr), "target_fpr": target_fpr})


def union_calibration(probe: Any, run_scope: str, scores: np.ndarray, positions: np.ndarray,
                      lengths: Optional[np.ndarray], verdicts: ViewVerdicts, target_fpr: float) -> Dict[str, Any]:
    """FR-36. Negatives aligned with the calibration view's rule verdicts BY POSITION (FR-20)."""
    from .probe_breakdown import recorded_decisions
    from .probe_monitor_metrics import band_for_length, usable_bands

    match, veto = union_rows(verdicts.rule_match[positions], verdicts.rule_veto[positions],
                             verdicts.unknown[positions])
    n = int(scores.shape[0])
    refuse_rules_over_budget(match, target_fpr)
    out: Dict[str, Any] = {"target_fpr": target_fpr, "aligned_by": "position", "global": union_cut(scores, match, veto, target_fpr)}
    bands = usable_bands(probe.length_bands) if probe.length_bands else None
    if bands and lengths is not None:
        # A band affords the quantile with at least 1/target negatives — `length_band_decisions`' own rule.
        minimum = int(math.ceil(1.0 / target_fpr))
        out["bands"] = []
        for band in bands:
            in_band = np.array([band_for_length(bands, int(x)) is band for x in lengths], dtype=bool)
            remaining = int((in_band & ~match & ~veto).sum())
            entry = {"min_tokens": band.get("min_tokens"), "max_tokens": band.get("max_tokens")}
            if remaining < minimum:
                entry.update(threshold=out["global"]["threshold"], fallback="global_union_bar",
                             reason="thin_band", n_negatives=int(in_band.sum()))
            else:
                entry.update(union_cut(scores[in_band], match[in_band], veto[in_band], target_fpr))
            out["bands"].append(entry)
    out["windows"] = "not_re_cut"
    # 040-T27: the union's realised FPR at the RECORDED decision, beside the cut.
    n_scored = lengths if lengths is not None else np.zeros(n, dtype=np.int64)
    fires = recorded_decisions(probe, run_scope, n_scored).fires(scores)
    union_recorded = match | (fires & ~veto)
    out["at_recorded_decision"] = {"realised_union_fpr": round6(int(union_recorded.sum()) / n) if n else None,
                                   "probe_alone_fpr": round6(int(fires.sum()) / n) if n else None,
                                   "over_budget": bool(n and int(union_recorded.sum()) / n > target_fpr)}
    alone = out["global"]["probe_alone_fpr_at_bar"]
    out["next"] = [next_recalibrate_preview(probe.id, float(alone))] if alone and 0 < alone < 1 else []
    return out


# ── request / read / list / cancel ───────────────────────────────────────────────────────────────


def _row_out(row: Any) -> Dict[str, Any]:
    return {"evaluation_id": row.id, "status": row.status, "probe_id": row.probe_id, "view_ids": row.view_ids,
            "behaviour_key": row.behaviour_key, "behaviour_version": row.behaviour_version,
            "rule_set_digest": row.rule_set_digest, "engine_digest": row.engine_digest,
            "calibration": row.calibration, "task_queue_id": row.task_queue_id, "progress": row.progress,
            "error": row.error, "created_by": row.created_by,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "body_sha256": row.body_sha256}


async def _probe_and_run(db: Any, probe_id: str) -> Tuple[Any, Any]:
    from sqlalchemy import select

    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorRun

    probe = (await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))).scalar_one_or_none()
    if probe is None:
        raise DetectorRefused(404, "probe_not_found", f"probe {probe_id} not found", {"probe_id": probe_id})
    run = (await db.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == probe.run_id))).scalar_one_or_none()
    if run is None:
        raise DetectorRefused(409, "run_gone", f"probe {probe_id}'s run is gone", {"probe_id": probe_id})
    return probe, run


def _dispatch(evaluation_id: str) -> str:
    """Enqueue the task (routed to `rules` by its full name). Patched in tests."""
    from ..workers.rules_evaluation_tasks import run_rules_evaluation

    return run_rules_evaluation.delay(evaluation_id).id


async def request(db: Any, probe_id: str, body: Dict[str, Any], actor: str) -> Tuple[Dict[str, Any], int]:
    """FTID §3.6. Returns `(envelope, status)`: 202 for a new or in-flight row, 200 for a completed duplicate."""
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert

    from .. import rule_signals
    from ..core.clock import utc_now
    from ..models.rules_evaluation import ProbeRulesEvaluation
    from . import rule_set_store
    from .detector_success_report import _evaluation, _revision, _views
    from .probe_dataset_lifecycle import refuse_archived_inputs
    from .task_queue_service import TaskQueueService

    probe, run = await _probe_and_run(db, probe_id)
    if not probe.behaviour:
        raise DetectorRefused(409, "no_behaviour",
                              f"probe {probe_id} names no behaviour; a rules evaluation judges a behaviour's rules "
                              f"with its probe", {"probe_id": probe_id})
    view_ids = list(dict.fromkeys(body["view_ids"]))
    views = await _views(db, view_ids)
    await refuse_archived_inputs(db, view_ids)
    head, version = await _version(db, probe.behaviour, body.get("version") or getattr(probe, "behaviour_version", None))
    settings = _effective()
    values = _settings_read(settings)
    limits = _limits(values)
    document, set_digest = await _version_document(db, version)
    validation = rule_set_store.validate_for_use(document, limits=limits) if document is not None else None
    evaluations = {}
    for view in views:
        evaluation = await _evaluation(db, probe.id, view.id)
        if evaluation is None:
            raise DetectorRefused(409, "missing_evaluation",
                                  f"probe {probe_id} has no evaluation of view {view.id} with recorded positions; "
                                  f"evaluate it first", {"view_id": view.id}, next=[next_evaluate(probe.id, view.id)])
        evaluations[view.id] = evaluation.id
    calibration_view = None
    if body.get("calibration", True):
        calibration_view = getattr(run, "calibration_dataset_id", None) or probe.calibration_dataset_id
        if not calibration_view:
            raise DetectorRefused(409, "no_calibration_view",
                                  f"probe {probe_id} has no calibration view, so no union bar can be cut; ask "
                                  f"without calibration", {"probe_id": probe_id})
        if not probe.calibration_positions_path:
            target = float(probe.target_fpr or version.target_fpr or 0.01)
            raise DetectorRefused(
                409, "calibration_positions_not_recorded",
                f"probe {probe_id}'s calibration negatives carry no recorded positions, so they cannot be aligned "
                f"with rule verdicts (FR-20). The GPU re-cut writes them.",
                {"probe_id": probe_id}, next=[next_recut_for_positions(probe.id, target)])
    rows = sum(split_rows(v) for v in views)
    if rows > int(settings.rules_evaluation_max_rows):
        raise DetectorRefused(422, "over_cap",
                              f"{rows} rows over the views; rules_evaluation_max_rows allows "
                              f"{settings.rules_evaluation_max_rows}",
                              {"setting": "rules_evaluation_max_rows", "value": settings.rules_evaluation_max_rows,
                               "got": rows})
    from .probe_decision import identity_fields

    # S4 (C.3, C.6; Review R1-F8): a set with operator code is computed under the runner's identity, and with a
    # runner every hash needs its document and the operator's approval, read fresh now.
    runner_id = runner_identity_for(document, settings)
    code_approvals = await _require_code_rules(db, document, runner_id, limits)
    engine = rule_signals.engine_digest(limits, runner_id)
    inputs = {
        "result_algorithm": RESULT_ALGORITHM, "probe_id": probe.id, "decision_revision": _revision(probe),
        "decision_digest": identity_fields(probe).get("decision_digest"),
        "behaviour_key": head.key, "behaviour_version": version.version, "rule_set_digest": set_digest,
        "engine_digest": engine, "vectors": rule_signals.vector_digests(), "view_ids": sorted(view_ids),
        "evaluations": evaluations, "calibration_view": calibration_view,
        "calibration": bool(body.get("calibration", True)), "settings": values,
        "views": [{"id": v.id, "rows": split_rows(v), "rule_columns": v.rule_columns, "text_column": v.text_column,
                   "snapshots": {c: e["sha256"] for c, e in sorted((v.column_snapshots or {}).items())}}
                  for v in sorted(views, key=lambda v: v.id)],
        "has_vetoes": bool(validation and validation["has_vetoes"]),
    }
    if runner_id is not None:  # a set with no code rule keeps S3's identity exactly
        inputs["runner"] = runner_id
        inputs["code_rules"] = code_approvals
    digest = identity(inputs)
    inserted = (await db.execute(insert(ProbeRulesEvaluation).values(
        id=f"pre_{uuid.uuid4().hex[:12]}", identity_sha256=digest, probe_id=probe.id, view_ids=view_ids,
        behaviour_key=head.key, behaviour_version=version.version, rule_set_digest=set_digest, engine_digest=engine,
        calibration=bool(body.get("calibration", True)), status="pending", inputs=inputs, created_by=actor,
    ).on_conflict_do_nothing(index_elements=["identity_sha256"],
                             index_where=ProbeRulesEvaluation.status.in_(("pending", "running", "completed")))
        .returning(ProbeRulesEvaluation.id))).scalar_one_or_none()
    await db.commit()
    if inserted is None:
        existing = (await db.execute(select(ProbeRulesEvaluation).where(
            ProbeRulesEvaluation.identity_sha256 == digest,
            ProbeRulesEvaluation.status.in_(("pending", "running", "completed"))))).scalar_one()
        return {**_row_out(existing), "duplicate": True,
                "next": [next_get_rules_evaluation(existing.id)]}, 200 if existing.status == "completed" else 202
    entry = await TaskQueueService.create_task_entry(db, task_type="rules_evaluation", entity_id=inserted,
                                                     entity_type="probe_rules_evaluation")
    row = (await db.execute(select(ProbeRulesEvaluation).where(ProbeRulesEvaluation.id == inserted))).scalar_one()
    row.task_queue_id = entry.id
    await db.commit()
    # ⚠ DISPATCH AFTER COMMIT (FTID §3.6): the task refuses without its row.
    try:
        celery_id = _dispatch(inserted)
    except Exception:  # noqa: BLE001 - a broker outage must not leave a `pending` ghost (Review R1, S3)
        # The partial-unique identity covers `pending`, and the reaper reaps `running` only: a row left pending
        # here would answer every identical request as an in-flight duplicate that never runs.
        logger.exception("could not dispatch rules evaluation %s", inserted)
        row.status, row.completed_at = "failed", utc_now()
        row.error = {"code": "broker_unavailable", "message": "the rules evaluation could not be queued (the task "
                                                              "broker is unavailable); request it again"}
        entry.status, entry.completed_at = "failed", utc_now()
        await db.commit()
        raise DetectorRefused(503, "broker_unavailable", "the rules evaluation could not be queued (the task broker "
                              "is unavailable); request it again", {"evaluation_id": inserted})
    entry.task_id = celery_id
    await db.commit()
    await db.refresh(row)
    return {**_row_out(row), "duplicate": False, "next": [next_get_rules_evaluation(row.id)]}, 202


async def _require_code_rules(db: Any, document: Optional[Dict[str, Any]], runner_id: Optional[Dict[str, Any]],
                              limits: Any) -> Optional[Dict[str, Any]]:
    """With a runner: every referenced hash has its document, fits it, and is APPROVED now → ``{sha: approval row
    id}``. Without one (or no code rule): None — operator code is pending (D41)."""
    if runner_id is None or runner_id.get("available") is False:
        return None
    from .. import rule_signals
    from ..rule_runner import contract as runner_contract
    from . import code_rules

    shas = code_rules.referenced_hashes(document or {})
    documents = await code_rules.documents_for(db, shas)
    code_rules.require_documents(shas, documents)
    try:
        rule_signals.check_code_rules(rule_signals.load(document, limits),
                                      {sha: doc for sha, (doc, _c) in documents.items()},
                                      runner_format=runner_contract.FORMAT)
    except rule_signals.RuleSetRefused as exc:
        raise DetectorRefused(422, exc.code, f"the rule set cannot be used: {exc.message}", dict(exc.details)) from exc
    found = await code_rules.states(db, shas)
    code_rules.require_approved(found)
    return {sha: found[sha]["approval_row_id"] for sha in shas}


async def _row(db: Any, evaluation_id: str) -> Any:
    from sqlalchemy import select

    from ..models.rules_evaluation import ProbeRulesEvaluation

    row = (await db.execute(select(ProbeRulesEvaluation).where(ProbeRulesEvaluation.id == evaluation_id))).scalar_one_or_none()
    if row is None:
        raise DetectorRefused(404, "not_found", f"rules evaluation {evaluation_id} not found",
                              {"evaluation_id": evaluation_id})
    return row


async def staleness(db: Any, row: Any) -> Dict[str, Any]:
    """FR-38, computed on READ: which input moved since the result was computed; nothing is rewritten."""
    from sqlalchemy import select

    from .. import rule_signals
    from ..models.probe_behaviour import ProbeBehaviour, ProbeBehaviourVersion
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorEvaluation
    from .detector_success_report import _revision

    inputs = row.inputs or {}
    moved: List[Dict[str, Any]] = []
    if inputs.get("result_algorithm") != RESULT_ALGORITHM:
        moved.append({"input": "result_algorithm", "was": inputs.get("result_algorithm"), "now": RESULT_ALGORITHM})
    runner_now = None
    if inputs.get("runner") is not None:
        from . import code_rule_runner

        try:
            runner_now = code_rule_runner.identity(_effective())
        except DetectorRefused as refusal:  # a limit below the floor: the runner cannot run at all now
            runner_now = {"available": False, "refused": refusal.code}
    engine_now = rule_signals.engine_digest(_limits(inputs.get("settings") or _settings_read()), runner_now)
    if engine_now != row.engine_digest:
        moved.append({"input": "engine_digest", "was": row.engine_digest, "now": engine_now,
                      "vectors_now": rule_signals.vector_digests()})
    head = (await db.execute(select(ProbeBehaviour).where(ProbeBehaviour.key == row.behaviour_key))).scalar_one_or_none()
    if head is not None and head.current_version != row.behaviour_version:
        current = (await db.execute(select(ProbeBehaviourVersion).where(
            ProbeBehaviourVersion.key == row.behaviour_key,
            ProbeBehaviourVersion.version == head.current_version))).scalar_one_or_none()
        if current is not None and current.rule_set_digest != row.rule_set_digest:
            moved.append({"input": "rule_set_digest", "was": row.rule_set_digest, "now": current.rule_set_digest})
    probe = (await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == row.probe_id))).scalar_one_or_none()
    if probe is None:
        moved.append({"input": "probe", "was": row.probe_id, "now": None})
    elif _revision(probe) != inputs.get("decision_revision"):
        moved.append({"input": "decision_revision", "was": inputs.get("decision_revision"), "now": _revision(probe)})
    cited = list((inputs.get("evaluations") or {}).values())
    if cited:
        present = set((await db.execute(select(ProbeMonitorEvaluation.id).where(
            ProbeMonitorEvaluation.id.in_(cited)))).scalars().all())
        gone = [e for e in cited if e not in present]
        if gone:
            moved.append({"input": "evaluations", "was": gone, "now": None})
    stale = bool(moved)
    return {"stale": stale, "moved": moved,
            "next": [next_rules_evaluation(row.probe_id, list(row.view_ids), None)] if stale else []}


async def read(db: Any, evaluation_id: str) -> Dict[str, Any]:
    row = await _row(db, evaluation_id)
    out = _row_out(row)
    out["result"] = json.loads(row.body) if row.body is not None else None
    out["stale"] = await staleness(db, row) if row.status == "completed" else {"stale": False, "moved": [], "next": []}
    return out


async def list_for(db: Any, key: str, *, limit: Optional[int], cursor: Optional[str]) -> Dict[str, Any]:
    from sqlalchemy import func, select

    from ..models.rules_evaluation import ProbeRulesEvaluation

    settings = _effective()
    page_max, page_default = int(settings.report_list_page_max), int(settings.report_list_page_default)
    size = limit or page_default
    if size > page_max:
        raise DetectorRefused(422, "over_cap", f"limit {size}; report_list_page_max allows {page_max}",
                              {"setting": "report_list_page_max", "value": page_max, "got": size})
    offset = 0
    if cursor:
        try:
            offset = int(json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))["o"])
        except Exception as exc:  # noqa: BLE001
            raise DetectorRefused(422, "invalid_field", "malformed cursor", {"field": "cursor", "rule": "malformed"}) from exc
    base = select(ProbeRulesEvaluation).where(ProbeRulesEvaluation.behaviour_key == key)
    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    rows = (await db.execute(base.order_by(ProbeRulesEvaluation.created_at.desc(), ProbeRulesEvaluation.id.desc())
                             .offset(offset).limit(size))).scalars().all()
    nxt = offset + len(rows)
    return {"behaviour_key": key, "rules_evaluations": [_row_out(r) for r in rows], "total": total,
            "next_cursor": base64.urlsafe_b64encode(json.dumps({"o": nxt}).encode()).decode().rstrip("=")
            if nxt < total else None}


async def cancel(db: Any, evaluation_id: str) -> Dict[str, Any]:
    """FR-37: the cancellation scope `probe_rules_evaluation`; the worker stops between chunks of rows."""
    from starlette.concurrency import run_in_threadpool

    from ..core.cancellation import request_cancel

    row = await _row(db, evaluation_id)
    celery_id = None
    if row.task_queue_id:
        from ..models.task_queue import TaskQueue

        entry = await db.get(TaskQueue, row.task_queue_id)
        celery_id = entry.task_id if entry is not None else None
    outcome = await run_in_threadpool(request_cancel, "probe_rules_evaluation", evaluation_id,
                                      reason="cancelled by request", celery_task_id=celery_id)
    return {"evaluation_id": evaluation_id, "accepted": outcome.requested, "was_running": outcome.was_running,
            "message": outcome.detail, "prior_status": outcome.prior_status}


# ── the task body ────────────────────────────────────────────────────────────────────────────────


def _snapshot_codes(view: Any, column: str) -> Tuple[np.ndarray, List[Any]]:
    from . import view_columns

    snap = view_columns.load(view.column_snapshots[column], column)
    return np.asarray(snap.codes), list(snap.values)


def _groups(view: Any, positions: np.ndarray, labels: np.ndarray, dets: Dict[str, np.ndarray],
            unknown: np.ndarray, min_rows: int) -> Dict[str, Any]:
    from .probe_breakdown import _value_key  # the breakdown's one value key

    out: Dict[str, Any] = {}
    for column in list(view.group_columns or []):
        if column not in (view.column_snapshots or {}):
            continue
        codes, values = _snapshot_codes(view, column)
        at = codes[positions]
        groups = []
        for code in sorted(set(at.tolist())):
            mask = at == code
            groups.append({"value": values[code], "rows": int(mask.sum()), "too_small": int(mask.sum()) < min_rows,
                           "unknown": int((unknown[mask] != UNKNOWN_NONE).sum()),
                           "detectors": {name: rates(flag, labels, mask) for name, flag in dets.items()}})
        groups.sort(key=lambda g: _value_key(g["value"]))
        out[column] = groups
    return out


def run(db: Any, evaluation_id: str, *, should_stop: Callable[[], None], beat: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    """Compute and return the body (the caller writes it once). Sync session (the worker's)."""
    from .. import rule_signals
    from ..core.config import settings as app_settings
    from ..models.dataset import Dataset
    from ..models.probe_behaviour import ProbeBehaviourVersion
    from ..models.probe_monitor import (
        ProbeMonitor,
        ProbeMonitorDataset,
        ProbeMonitorEvaluation,
        ProbeMonitorRun,
    )
    from ..models.rule_set import RuleSet
    from ..models.rules_evaluation import ProbeRulesEvaluation
    from . import rule_set_store
    from .probe_breakdown import evaluation_window, recorded_decisions
    from .probe_evaluation_scores import load_evaluation_arrays

    row = db.query(ProbeRulesEvaluation).filter(ProbeRulesEvaluation.id == evaluation_id).one()
    inputs = row.inputs
    values = inputs["settings"]
    limits = _limits(values)
    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == row.probe_id).one()
    run_row = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).one()
    scope = (run_row.config or {}).get("scope", "all")
    version = db.query(ProbeBehaviourVersion).filter_by(key=row.behaviour_key, version=row.behaviour_version).one()
    compiled = None
    reasons: List[str] = []
    if row.rule_set_digest is not None:
        stored = db.query(RuleSet).filter(RuleSet.digest == row.rule_set_digest).one()
        document = rule_set_store.document_of(stored)[0]
        rule_set_store.validate_for_use(document, limits=limits)  # FR-29: at every use
        compiled = rule_signals.load(document, limits)
        if not compiled.has_vetoes:
            reasons.append("no_veto_signals")
        if compiled.match_veto_conditions:
            reasons.append("match_includes_veto_condition")
    else:
        reasons.append("no_rule_set")
    # ⚠ THE DIGEST THE REQUEST RECORDED (with the runner's identity for a set with code rules, R1-F8): the cache key.
    engine = row.engine_digest
    runner, batch_texts = _task_runner(db, inputs, compiled)
    root = Path(app_settings.data_dir) / "probe_monitors"
    share_max = float(values["rule_unknown_share_max"])
    min_rows = int(values["breakdown_min_subgroup_rows"])
    views_out: List[Dict[str, Any]] = []
    totals_rows = sum(int(v["rows"]) for v in inputs["views"]) or 1
    done_rows = 0

    def verdicts_for(view: Any) -> ViewVerdicts:
        nonlocal done_rows
        dataset = db.query(Dataset).filter(Dataset.id == view.dataset_id).one()
        data = open_view_split(dataset, view)
        if compiled is None:
            n = len(data)
            return ViewVerdicts(np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, np.uint8), n, False)
        base = done_rows

        def progress(end: int, n: int) -> None:
            beat({"rows_done": base + end, "rows_total": totals_rows})

        out = view_verdicts(db, view, data, compiled, limits, scope, set_digest=row.rule_set_digest, engine=engine,
                            evaluation_id=evaluation_id, should_stop=should_stop, beat=progress, runner=runner,
                            batch_texts=batch_texts)
        done_rows += out.n_rows
        return out

    for view_id in row.view_ids:  # the request's order (the identity sorts them)
        view = db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == view_id).one()
        evaluation = db.query(ProbeMonitorEvaluation).filter(
            ProbeMonitorEvaluation.id == inputs["evaluations"][view_id]).one()
        arrays = load_evaluation_arrays(evaluation, root=root).arrays
        positions = np.asarray(arrays["position"], dtype=np.int64)
        labels = np.asarray(arrays["label"])
        fires = recorded_decisions(probe, scope, arrays["n_scored"]).fires(np.asarray(arrays["score"], dtype=np.float64))
        verdicts = verdicts_for(view)
        match, veto, unknown = verdicts.rule_match[positions], verdicts.rule_veto[positions], verdicts.unknown[positions]
        entry: Dict[str, Any] = {"view_id": view.id, "evaluation_id": evaluation.id, "rows": int(positions.shape[0]),
                                 "window": evaluation_window(scope), "reading": reading_of(view, scope)["source"],
                                 "cached": verdicts.cached}
        producer = view.rule_columns or None
        labels_used: List[str] = []
        if producer and compiled is not None:
            snaps = view.column_snapshots or {}
            p_match = _bool_snapshot(snaps[producer["rule_match"]], producer["rule_match"])[positions] \
                if producer.get("rule_match") in snaps else None
            p_veto = _bool_snapshot(snaps[producer["rule_veto"]], producer["rule_veto"])[positions] \
                if producer.get("rule_veto") in snaps else None
            known = unknown == UNKNOWN_NONE
            entry["agreement"] = {
                "rule_set_digest_matches": producer.get("rule_set_digest") == row.rule_set_digest,
                "rule_match": agreement(match & known, p_match) if p_match is not None else None,
                "rule_veto": agreement(veto & known, p_veto) if p_veto is not None else None,
                "compared_rows": int(known.sum()), "used": "mistudio",
            }
            # ⚠ FR-31's ONE EXCEPTION: pending operator code, and a producer verdict for EXACTLY this digest.
            pending = unknown == UNKNOWN_PENDING
            if pending.any() and producer.get("rule_set_digest") == row.rule_set_digest and p_match is not None:
                match = np.where(pending, p_match, match)
                veto = np.where(pending, p_veto if p_veto is not None else False, veto)
                unknown = np.where(pending, UNKNOWN_NONE, unknown).astype(np.uint8)
                labels_used.append("producer_computed")
                entry["agreement"]["used"] = "mistudio_with_producer_computed"
        dets = detectors(match, veto, unknown, fires)
        n_unknown = int((unknown != UNKNOWN_NONE).sum())
        share = n_unknown / positions.shape[0] if positions.shape[0] else 0.0
        by_reason = {UNKNOWN_REASONS[c]: int((unknown == c).sum()) for c in UNKNOWN_REASONS}
        by_reason.update({r: int((unknown == c).sum()) for r, c in CODE_REASON_CODES.items()
                          if int((unknown == c).sum())})
        entry["unknown"] = {"n": n_unknown, "share": round6(share), "by_reason": by_reason,
                            "over_share": share > share_max, "share_max": share_max}
        view_reasons = list(reasons) + labels_used
        if share > share_max:
            view_reasons.append("rule_unknown_share")
            if int((unknown == UNKNOWN_PENDING).sum()):
                view_reasons.append("runner_unavailable")
            view_reasons.extend(r for r in CODE_REASON_CODES if by_reason.get(r))
        if verdicts.code:
            entry["code_rules"] = _code_report(verdicts.code, inputs)
        entry["reasons"] = view_reasons
        entry["detectors"] = {name: rates(flag, labels) for name, flag in dets.items()}
        entry["groups"] = _groups(view, positions, labels, dets, unknown, min_rows)
        views_out.append(entry)
        should_stop()

    calibration_out = None
    if inputs.get("calibration"):
        view = db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == inputs["calibration_view"]).one()
        scores = np.load(probe.calibration_scores_path, allow_pickle=False).astype(np.float64).reshape(-1)
        positions = np.load(probe.calibration_positions_path, allow_pickle=False).astype(np.int64).reshape(-1)
        lengths = (np.load(probe.calibration_lengths_path, allow_pickle=False).reshape(-1)
                   if probe.calibration_lengths_path else None)
        if positions.shape[0] != scores.shape[0] or (lengths is not None and lengths.shape[0] != scores.shape[0]):
            raise DetectorRefused(500, "calibration_misaligned",
                                  "the probe's calibration negatives and positions differ in length",
                                  {"probe_id": probe.id})
        target = float(version.target_fpr or probe.target_fpr or 0.01)
        verdicts = verdicts_for(view)
        try:
            calibration_out = {"view_id": view.id, **union_calibration(probe, scope, scores, positions, lengths,
                                                                        verdicts, target)}
        except DetectorRefused as refusal:
            calibration_out = {"view_id": view.id, "refused": refusal.code, **refusal.details}
    rule_vetoes = bool(compiled is not None and compiled.has_vetoes)
    body = {
        "format": RESULT_FORMAT, "result_algorithm": RESULT_ALGORITHM, "evaluation_id": evaluation_id,
        "probe_id": probe.id, "behaviour_key": row.behaviour_key, "behaviour_version": row.behaviour_version,
        "rule_set_digest": row.rule_set_digest, "engine_digest": row.engine_digest,
        "engine_version": rule_signals.ENGINE_VERSION, "vectors": inputs.get("vectors"),
        "decision_revision": inputs.get("decision_revision"), "reasons": reasons,
        # TI-25: the expressions read as the rule MATCH (the set's `fired` outputs), recorded for audit.
        "match_expressions": list(compiled.match_expressions) if compiled is not None else [],
        "veto_note": ("an application vetoing per span may see a different figure: this veto is per row "
                      "(FR-33)") if rule_vetoes else None,
        "views": views_out, "calibration": calibration_out, "settings": values,
    }
    if inputs.get("runner") is not None:
        body["runner"] = inputs["runner"]
    return body


class _SwitchedOffRunner:
    """Code rules were switched off (or the runner URL removed) after the request: every text is a host fault —
    unknown, never cached, never "did not fire"."""

    available = True

    def run_batch(self, signal: Any, texts: Any) -> List[Dict[str, Any]]:
        return [{"reason": "rule_worker_unavailable"} for _ in texts]

    def identity(self) -> Dict[str, Any]:
        return {"available": False}


def _task_runner(db: Any, inputs: Dict[str, Any], compiled: Any) -> Tuple[Any, int]:
    """The runner a task uses, from what its REQUEST recorded (limits included), re-checked against the switch now."""
    from .. import rule_signals

    recorded = inputs.get("runner")
    if compiled is None or recorded is None or recorded.get("available") is False:
        return rule_signals.UnavailableRunner(), 256
    from . import code_rule_runner, code_rules

    settings = _effective()
    batch = int(settings.code_rule_batch_texts)
    if not code_rule_runner.enabled(settings):
        return _SwitchedOffRunner(), batch
    shas = sorted({s["code_sha256"] for s in compiled.signals if s["kind"] == "code_rule"})
    documents = code_rules.documents_for_sync(db, shas)
    # ⚠ THE FRESH APPROVAL READ PER CHUNK (C.6): a revocation mid-run stops the next chunk (`rule_not_approved`).
    return code_rule_runner.HttpCodeRuleRunner(
        str(settings.code_rule_runner_url), limits=recorded["limits"],
        documents={sha: c for sha, (_d, c) in documents.items()},
        approved=lambda sha: code_rules.approved_now(db, sha), lane="batch"), batch


def _code_report(code: Dict[str, Dict[str, Any]], inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """C.3: per code rule, what ran it and how often it failed — counts only."""
    runner_id = inputs.get("runner") or {}
    approvals = inputs.get("code_rules") or {}
    out = []
    for sid, tally in sorted(code.items()):
        sha = tally["code_sha256"]
        out.append({"signal_id": sid, "code_sha256": sha, "runner_contract": runner_id.get("format"),
                    "contract_version": runner_id.get("contract_version"),
                    "bundle_digest": runner_id.get("bundle_digest"), "approval_row_id": approvals.get(sha),
                    "rows_fired": int(tally.get("fired", 0)), "reasons": dict(sorted(tally.get("reasons", {}).items()))})
    return out


def finish(db: Any, evaluation_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Write the body ONCE (the ORM listener and the trigger refuse a rewrite). Refuses over the byte cap."""
    from ..core.clock import utc_now
    from ..models.rules_evaluation import ProbeRulesEvaluation

    raw = jcs_bytes(body)
    cap = int(body["settings"]["rules_evaluation_max_bytes"])
    row = db.query(ProbeRulesEvaluation).filter(ProbeRulesEvaluation.id == evaluation_id).populate_existing().one()
    if row.status != "running":
        return {"evaluation_id": evaluation_id, "status": row.status}
    if len(raw) > cap:
        row.status, row.completed_at = "failed", utc_now()
        row.error = {"code": "over_cap", "setting": "rules_evaluation_max_bytes", "value": cap, "got": len(raw)}
        db.commit()
        return {"evaluation_id": evaluation_id, "status": "failed"}
    row.body, row.body_sha256 = raw.decode("utf-8"), sha256_hex(raw)
    row.status, row.completed_at = "completed", utc_now()
    db.commit()
    return {"evaluation_id": evaluation_id, "status": "completed", "body_sha256": row.body_sha256}
