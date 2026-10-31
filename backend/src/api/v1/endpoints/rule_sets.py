"""REST surface of rule sets and rules evaluations (Feature 040 S3; FR-32, FR-37, FR-38; FTDD §5.2; PADR IDL-61).

Every route is CPU-only — none loads a model or takes a GPU lease (FTDD §5.5).

* `POST /rule-sets/validate`                     — check a set: schema, engine, its own cases (FR-32)
* `POST /rule-signals/evaluate`                  — try a set on ONE text or conversation, synchronously (FR-32)
* `POST /probes/{id}/rules-evaluations`          — rules ∪ probe over views, a background task (FR-33 – FR-37)
* `GET  /rules-evaluations/{rid}`                — one, with `stale` computed on read (FR-38)
* `GET  /behaviours/{key}/rules-evaluations`     — a behaviour's, newest first, paged
* `POST /rules-evaluations/{rid}/cancel`         — cooperative cancel (FR-37)

⚠ NOTHING HERE STORES OR LOGS A TEXT. `evaluate_rule_signals` returns spans of the caller's own text and
keeps nothing; the evaluation routes store counts, rates, ids and digests only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....services import rule_set_store, rules_evaluation
from ....services.detector_next import DetectorRefused
from ....services.probe_dataset_lifecycle import actor_of
from ._probe_ids import refuse_blend_report_ids

router = APIRouter(prefix="/probe-monitors", tags=["rule-sets"])

#: Message roles a conversation may carry (the reducer reads `role` and `content` only).
Role = Literal["system", "user", "assistant", "tool", "developer"]


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Role
    content: Optional[str] = None


class CodeRuleItem(BaseModel):
    """S4 (FTDD Appendix C.5): a `midataworks.runner/v1` code-rule document a `code_rule` signal of the set names by
    hash, sent inline. miStudio hashes it with the contract's rule and stores it once (content-addressed)."""

    model_config = ConfigDict(extra="forbid")

    document: Dict[str, Any]


class RuleSetChoice(BaseModel):
    """Which rule set: the behaviour's (its current or a named version), or one sent inline."""

    model_config = ConfigDict(extra="forbid")

    behaviour_key: Optional[str] = Field(default=None, min_length=1, max_length=64)
    version: Optional[int] = Field(default=None, ge=1)
    rule_set: Optional[Dict[str, Any]] = None
    #: S4: documents of the operator code the set names (inline). Running them needs the runner AND the operator's
    #: per-hash approval; without a runner they are stored and the code expressions read pending (D41).
    code_rules: Optional[List[CodeRuleItem]] = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _one_source(self):
        if (self.behaviour_key is None) == (self.rule_set is None):
            raise ValueError("send exactly one of behaviour_key (with an optional version) or rule_set")
        if self.version is not None and self.behaviour_key is None:
            raise ValueError("version needs behaviour_key")
        return self


class ValidateRequest(RuleSetChoice):
    pass


class EvaluateRequest(RuleSetChoice):
    text: Optional[str] = None
    messages: Optional[List[Message]] = Field(default=None, min_length=1)
    #: miStudio's internal scope (`all`, `content`, `user`, `assistant`, `last_assistant`, `input`,
    #: `last_user`) or a contract window (`prompt`, `response`); default `all`.
    scope: Optional[str] = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def _one_input(self):
        if (self.text is None) == (self.messages is None):
            raise ValueError("send exactly one of text or messages")
        return self


class RulesEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view_ids: List[str] = Field(min_length=1, max_length=64)
    version: Optional[int] = Field(default=None, ge=1)
    calibration: bool = True


def _http(refusal: Exception) -> HTTPException:
    if isinstance(refusal, DetectorRefused):
        return refusal.to_http()
    from ....services import behaviour_catalogue as catalogue

    if isinstance(refusal, catalogue.BehaviourRefused):
        return HTTPException(status_code=refusal.status, detail=refusal.as_detail())
    raise refusal


@router.post("/rule-sets/validate")
async def validate_rule_set(body: ValidateRequest, request: Request,
                            db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Check a rule set with miStudio's engine: the vendored schema, compilation under the memory cap, and
    every case it carries. Answers `{valid, …}` — a set that fails is `valid: false` with the code and the
    failing cases, not an HTTP error. Stores nothing but inline operator-code documents (S4: content-addressed;
    with a runner, each named hash must be approved — `code_rule_not_approved`)."""
    try:
        return await rules_evaluation.validate_request(db, {**body.model_dump(), "_actor": actor_of(request)})
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.post("/rule-signals/evaluate")
async def evaluate_rule_signals(body: EvaluateRequest, request: Request,
                                db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Try a rule set on one text or conversation: `rule_match`, `rule_veto`, `unknown` with its reason, the
    signals that fired with their spans `(message_index, start, end)` in code points, and both digests. A fired
    operator code rule (S4) carries its count and `spans: null`. Synchronous; nothing is stored or logged."""
    try:
        return await rules_evaluation.evaluate_one(db, {**body.model_dump(exclude_none=False),
                                                        "_actor": actor_of(request)})
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.post("/probes/{probe_id}/rules-evaluations", status_code=202,
             dependencies=[Depends(refuse_blend_report_ids)])
async def create_rules_evaluation(
    probe_id: str, body: RulesEvaluationRequest, request: Request, db: AsyncSession = Depends(get_db)
):
    """Start (or return) a rules ∪ probe evaluation of one probe on the named views. 202 with the task id
    while it runs; a duplicate request returns the existing row (200 when it is completed)."""
    from fastapi.responses import JSONResponse

    try:
        out, status = await rules_evaluation.request(db, probe_id, body.model_dump(), actor_of(request))
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    return JSONResponse(out, status_code=status)


@router.get("/rules-evaluations/{evaluation_id}")
async def get_rules_evaluation(evaluation_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """One rules evaluation: its status and, when completed, its stored result byte-for-byte, with `stale`
    computed on read (which input moved, and the rerun `next`)."""
    try:
        return await rules_evaluation.read(db, evaluation_id)
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.get("/behaviours/{key}/rules-evaluations")
async def list_rules_evaluations(
    key: str, limit: Optional[int] = Query(None, ge=1), cursor: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """A behaviour's rules evaluations, newest first, paged (`limit`, `cursor` → `next_cursor`, `total`)."""
    try:
        return await rules_evaluation.list_for(db, key, limit=limit, cursor=cursor)
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.post("/rules-evaluations/{evaluation_id}/cancel", status_code=202)
async def cancel_rules_evaluation(evaluation_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Ask a pending or running rules evaluation to stop; the worker stops between chunks of rows and leaves
    a `cancelled` record."""
    try:
        return await rules_evaluation.cancel(db, evaluation_id)
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


__all__ = ["router", "rule_set_store"]
