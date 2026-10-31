"""Refusals and `next` calls of the behaviour-detector feature (Feature 040 FR-40; FTID §3.1, TI-5, TI-6).

Two things live here, so a refusal and the call that fixes it are written in one place:

* `DetectorRefused` — a service exception with `status`, `code`, `message`, `details` and `next`, turned
  into ``HTTPException(status, detail={code, message, **details, next})`` by `to_http()`.
* one CONSTRUCTOR per kind of `next` entry. ⚠ THESE ARE THE ONLY PLACES A TOOL NAME IS WRITTEN IN
  `services/` FOR THIS FEATURE (TI-6). Each is registered in `CONSTRUCTORS` by the `@constructor`
  decorator with an example call, so `test_detector_next_is_live.py` checks every one against the LIVE MCP
  registry and the tool's input schema without a hand list (suite memory `a-guard-can-work-by-accident`),
  and the generality guard reads every constructor's output.

The entry shape is Feature 039's `next_calls.NextCall` — ``{tool, arguments, why}`` plus `needs` for an
argument only the agent can supply — reused, not redefined (FTID §3.1 wrote ``{tool, arguments, route}``
before 039 fixed the shared shape; 039's module says 040 and 041 reuse it). `why` names no behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException

from .next_calls import NextCall


class DetectorRefused(Exception):
    """A request the detector feature will not carry out; the body is data an agent can act on."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        next: Optional[List[NextCall]] = None,  # noqa: A002 - the wire name
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = dict(details or {})
        self.next = list(next or [])

    def as_detail(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.details, "next": self.next}

    def to_http(self) -> HTTPException:
        return HTTPException(status_code=self.status, detail=self.as_detail())


@dataclass(frozen=True)
class Constructor:
    tool: str
    build: Callable[..., NextCall]
    example_kwargs: Dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> NextCall:
        return self.build(*args, **kwargs)

    def example(self) -> NextCall:
        return self.build(**self.example_kwargs)


#: name → constructor. Filled by `@constructor`; read by the FR-40 test and the generality guard.
CONSTRUCTORS: Dict[str, Constructor] = {}


def constructor(tool: str, **example_kwargs: Any) -> Callable[[Callable[..., NextCall]], Constructor]:
    def register(build: Callable[..., NextCall]) -> Constructor:
        made = Constructor(tool=tool, build=build, example_kwargs=example_kwargs)
        CONSTRUCTORS[build.__name__] = made
        return made

    return register


def next_tools() -> frozenset:
    """Every tool a `next` entry of this feature can name (derived from the constructors)."""
    return frozenset(c.tool for c in CONSTRUCTORS.values())


def _call(tool: str, arguments: Dict[str, Any], why: str, needs: Optional[List[str]] = None) -> NextCall:
    entry: NextCall = {"tool": tool, "arguments": {k: v for k, v in arguments.items() if v is not None},
                       "why": why}
    if needs:
        entry["needs"] = list(needs)
    return entry


# ── constructors ──────────────────────────────────────────────────────────────────────────────


@constructor("download_model", repo_id="org/model")
def next_download_model(repo_id: str) -> NextCall:
    return _call(
        "download_model", {"repo_id": repo_id},
        "The definition names a model this deployment has not downloaded; evaluating the behaviour "
        "on it needs the model here.",
    )


@constructor("list_models")
def next_list_models() -> NextCall:
    return _call(
        "list_models", {},
        "The definition names a model id this deployment does not have; list the models here to find "
        "its id, or download it by its Hugging Face repository id.",
    )


@constructor("list_probe_datasets")
def next_find_views() -> NextCall:
    return _call(
        "list_probe_datasets", {},
        "No registered view carries this column yet; register or update a view that has it before "
        "a criterion can be judged on it.",
    )


@constructor("list_behaviour_versions", key="key-example")
def next_list_versions(key: str) -> NextCall:
    return _call(
        "list_behaviour_versions", {"key": key},
        "The version you started from is no longer the current one; read the versions, then send "
        "your change against the current version.",
    )


_INLINE_WHY = (
    "miDataworks could not supply the rule set; repeat this call with the set's document inline "
    "(rule_set = {document: <the midataworks.rule-set/v1 document>})."
)


@constructor("create_behaviour", arguments={"key": "key-example", "name": "n", "positive_meaning": "p"})
def next_create_with_inline_rule_set(arguments: Dict[str, Any]) -> NextCall:
    """The same `create_behaviour` call, with the rule set to be sent inline by the agent."""
    return _call("create_behaviour", {k: v for k, v in arguments.items() if k != "rule_set"},
                 _INLINE_WHY, needs=["rule_set"])


@constructor("new_behaviour_version", arguments={"key": "key-example", "base_version": 1})
def next_version_with_inline_rule_set(arguments: Dict[str, Any]) -> NextCall:
    """The same `new_behaviour_version` call, with the rule set to be sent inline by the agent."""
    return _call("new_behaviour_version", {k: v for k, v in arguments.items() if k != "rule_set"},
                 _INLINE_WHY, needs=["rule_set"])


@constructor("new_behaviour_version", key="key-example", base_version=1)
def next_attach_rule_set(key: str, base_version: int) -> NextCall:
    return _call(
        "new_behaviour_version", {"key": key, "base_version": base_version},
        "This version names no rule set, so nothing can be OR-ed with a blend: attach one as a change against "
        "the current version.",
        needs=["rule_set"],
    )


@constructor("new_behaviour_version", key="key-example", base_version=1)
def next_add_success_criteria(key: str, base_version: int) -> NextCall:
    return _call(
        "new_behaviour_version", {"key": key, "base_version": base_version},
        "Say what the detector must achieve: add success criteria (and the held-out columns they "
        "name) as a change against the current version.",
        needs=["success_criteria"],
    )


@constructor("list_behaviours")
def next_restart_listing(q: Optional[str] = None, status: Optional[str] = None) -> NextCall:
    return _call(
        "list_behaviours", {"q": q, "status": status},
        "A cursor is valid only with the filters it was issued for; list again from the first page.",
    )


# ── S2 constructors ───────────────────────────────────────────────────────────────────────────


@constructor("evaluate_probe_monitor", probe_id="pm_x", dataset_id="pmd_x")
def next_evaluate(probe_id: str, dataset_id: str) -> NextCall:
    """GPU work behind the existing approval gate; the agent sees `pending_approval` and repeats."""
    return _call(
        "evaluate_probe_monitor", {"probe_id": probe_id, "dataset_ids": [dataset_id]},
        "This probe has no evaluation with saved row positions on this view; evaluating it (approval-gated "
        "GPU work) produces one that can be broken down and judged.",
    )


@constructor("update_probe_dataset_columns", dataset_id="pmd_x", group_columns=["c"])
def next_update_columns(dataset_id: str, group_columns: List[str]) -> NextCall:
    return _call(
        "update_probe_dataset_columns", {"dataset_id": dataset_id, "group_columns": list(group_columns)},
        "Declare the column on this view (additive: every declared column plus the new one); it is read "
        "and snapshotted from the downloaded split.",
    )


@constructor("recalibrate_probe_monitor", probe_id="pm_x", target_fpr=0.01)
def next_recalibrate_preview(probe_id: str, target_fpr: float) -> NextCall:
    return _call(
        "recalibrate_probe_monitor", {"probe_id": probe_id, "target_fpr": target_fpr, "preview": True},
        "The probe's bar was cut at another false-positive rate than the criterion's; a preview shows the "
        "bar at the criterion's rate (nothing is written until you commit).",
    )


@constructor("get_detector_success_report", key="key-example", model_id="m_x", probe_id="pm_x")
def next_success_report(key: str, model_id: Optional[str] = None, probe_id: Optional[str] = None) -> NextCall:
    return _call(
        "get_detector_success_report", {"key": key, "model_id": model_id, "probe_id": probe_id},
        "Build the success report with this choice named.",
    )


@constructor("get_probe_breakdown", probe_id="pm_x", evaluation_id="pme_x", by_column="c")
def next_breakdown(probe_id: str, evaluation_id: str, by_column: str) -> NextCall:
    return _call(
        "get_probe_breakdown", {"probe_id": probe_id, "evaluation_id": evaluation_id, "by_column": by_column},
        "Break this evaluation down by a declared column at the decision in force.",
    )


# ── Feature 041 S1 constructors (families) ───────────────────────────────────────────────────


@constructor("get_probe_family", key="key-example", model="m_x")
def next_get_family(key: str, model: str) -> NextCall:
    return _call(
        "get_probe_family", {"key": key, "model": model},
        "The behaviour has probes on more than one model; read the family on one model at a time.",
    )


@constructor("start_probe_training_run", model_id="m_x", train_dataset_id="pmd_x", key="key-example")
def next_train_specialist(model_id: str, train_dataset_id: str, key: str) -> NextCall:
    """GPU work behind the existing approval gate; the agent sees `pending_approval` and repeats."""
    return _call(
        "start_probe_training_run",
        {"model_id": model_id, "train_dataset_id": train_dataset_id, "behaviour": key},
        "This sub-question has no member on the model; train one on its registered training view "
        "(approval-gated GPU work).",
    )


#: The `register_probe_dataset` arguments only the agent knows (which data, which columns, which labels).
REGISTER_VIEW_NEEDS = ["name", "dataset_id", "input_column", "label_column", "label_mapping"]


@constructor("register_probe_dataset", key="key-example", sub_question="sq-example", behaviour_version=1)
def next_register_specialist_view(key: str, sub_question: str, behaviour_version: Optional[int]) -> NextCall:
    return _call(
        "register_probe_dataset",
        {"behaviour": key, "sub_question": sub_question, "role": "train",
         "behaviour_version": behaviour_version},
        "No training view answers this sub-question yet; register one (your data, its columns and its "
        "label mapping), then train a member on it.",
        needs=list(REGISTER_VIEW_NEEDS),
    )


@constructor("build_probe_definition", probe_id="pm_x", rung=2)
def next_build_member(probe_id: str, rung: Optional[int]) -> NextCall:
    """A family member without a current built definition; the 039 constructor (it asks the agent for
    an acknowledgement below rung 2 rather than inventing one)."""
    from .next_calls import build_probe_definition

    return build_probe_definition(
        probe_id, "This family member has no current built definition; build it so the family can be "
        "exported.", rung=rung,
    )


# ── Feature 041 S2 constructors (blend evaluation) ───────────────────────────────────────────


@constructor("evaluate_probe_monitor", probe_id="pm_x", dataset_ids=["pmd_x"], sample_rows=2000)
def next_evaluate_sampled(probe_id: str, dataset_ids: List[str], sample_rows: Optional[int]) -> NextCall:
    """GPU work behind the existing approval gate. With `sample_rows` (a calibration view) every member
    evaluated at that size scores THE SAME rows (the sample's seed is the view and the size, T20)."""
    return _call(
        "evaluate_probe_monitor",
        {"probe_id": probe_id, "dataset_ids": list(dataset_ids), "sample_rows": sample_rows},
        "This member has no stored per-row scores on a view the blend needs; evaluate it (approval-gated GPU "
        "work)" + (" on the shared calibration sample of this size." if sample_rows else "."),
    )


@constructor("register_probe_dataset", key="key-example", role="blend_fit")
def next_register_view(key: str, role: str) -> NextCall:
    return _call(
        "register_probe_dataset", {"behaviour": key, "role": role},
        f"No unarchived {role} view of this behaviour exists; register one (your data, its columns and its "
        "label mapping).",
        needs=list(REGISTER_VIEW_NEEDS),
    )


@constructor("unarchive_behaviour", key="key-example")
def next_unarchive_behaviour(key: str) -> NextCall:
    return _call("unarchive_behaviour", {"key": key},
                 "The behaviour is archived; unarchive it before evaluating blends of its probes.")


@constructor("unarchive_probe_dataset", dataset_id="pmd_x")
def next_unarchive_view(dataset_id: str) -> NextCall:
    return _call("unarchive_probe_dataset", {"dataset_id": dataset_id},
                 "This view is archived; unarchive it to use it as a blend input.")


@constructor("evaluate_probe_blends", key="key-example", model="m_x", member_ids=["pm_a", "pm_b"])
def next_evaluate_blends(key: str, model: Optional[str], member_ids: Optional[List[str]]) -> NextCall:
    return _call(
        "evaluate_probe_blends",
        {"behaviour_key": key, "model": model, "member_ids": list(member_ids) if member_ids else None},
        "Evaluate the blends again with these members." if member_ids else
        "Evaluate the blends of the behaviour's default family.",
    )


@constructor("rerun_blend_report", report_id="pbr_x")
def next_rerun_report(report_id: str) -> NextCall:
    return _call("rerun_blend_report", {"report_id": report_id},
                 "An input this report recorded has moved; rerun it against today's members and definition.")


@constructor("get_blend_report", report_id="pbr_x")
def next_get_report(report_id: str) -> NextCall:
    return _call("get_blend_report", {"report_id": report_id},
                 "Read the blend report (its status, ranking and figures).")


@constructor("list_blend_reports", key="key-example", model="m_x")
def next_list_reports(key: str, model: Optional[str]) -> NextCall:
    return _call("list_blend_reports", {"behaviour_key": key, "model": model},
                 "List the behaviour's blend reports to choose one to mark as the suggested blend.")


# ── Feature 040 S3: rules ∪ probe ────────────────────────────────────────────────────────────────


@constructor("evaluate_detector_with_rules", probe_id="pm_x", view_ids=["pmd_x"], version=None)
def next_rules_evaluation(probe_id: str, view_ids: List[str], version: Optional[int]) -> NextCall:
    return _call(
        "evaluate_detector_with_rules", {"probe_id": probe_id, "view_ids": list(view_ids), "version": version},
        "Judge the behaviour's rule set and this probe together on these views (rules alone, probe alone and "
        "rules or probe). A CPU background task: poll it with get_detector_rules_evaluation or get_task_status.",
    )


@constructor("get_detector_rules_evaluation", evaluation_id="pre_x")
def next_get_rules_evaluation(evaluation_id: str) -> NextCall:
    return _call(
        "get_detector_rules_evaluation", {"evaluation_id": evaluation_id},
        "Read the rules evaluation: its status while it runs, its stored result once completed.",
    )


@constructor("recalibrate_probe_monitor", probe_id="pm_x", target_fpr=0.01)
def next_recut_for_positions(probe_id: str, target_fpr: float) -> NextCall:
    """The GPU re-cut arm, asked FOR the calibration positions (review R1-5): it refuses before the GPU when no
    pass of the arm can produce them (`positions_not_recoverable`)."""
    return _call(
        "recalibrate_probe_monitor",
        {"probe_id": probe_id, "target_fpr": target_fpr, "preview": False, "recut_windows": True,
         "require_positions": True},
        "The probe's calibration negatives carry no recorded positions, so rule verdicts cannot be aligned with "
        "them. The GPU re-cut of its windows writes the positions (it needs operator approval and keeps the "
        "target false-positive rate).",
    )


# ── Feature 040 S4: operator code rules ──────────────────────────────────────────────────────────


@constructor("request_code_rule_approval", code_sha256="a" * 64)
def next_request_code_rule_approval(code_sha256: str) -> NextCall:
    return _call(
        "request_code_rule_approval", {"code_sha256": code_sha256},
        "Operator code runs here only after the operator approves its exact hash. File the request with the "
        "document (if miStudio does not hold it yet) and its own cases; they are run in the isolated runner first, "
        "then the operator decides in miStudio's approvals banner.",
        needs=["cases"],
    )
