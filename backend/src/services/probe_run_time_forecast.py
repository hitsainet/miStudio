"""How long a probe training run will take, said BEFORE the operator approves it (ENH-006 G15).

A Llama run listed a 1,000,000-row everyday view among its evaluation sets. Every set was scored whole, the
ETA read 347 hours, and nobody knew until hours of GPU time had gone. The size guard (`eval_max_rows`) stops
that one shape; this module answers the general question — how long — before anything starts.

WHAT IT IS, AND WHAT IT IS NOT

A forecast from MEASUREMENTS. Every completed run records `environment.stage_seconds`. Each stage's time is
divided by the work that stage did, giving a measured rate; the new run's work in the same unit, times those
rates, is the forecast. Nothing here is an assumed throughput: with no comparable completed run there is no
number, and the answer says so (`forecast: False`).

THE WORK UNIT OF EACH STAGE, read from the code that does the work (`probe_monitor_run.execute_probe_run`):

    rendering         rows rendered. `_prepare_examples` maps and renders every row of the train view once.
    pooled_capture    tokens forwarded. `capture_pooled` runs ONE forward per batch with a hook on every swept
                      layer, so the forward dominates and the layer count does not multiply it. Counted as the
                      scored tokens plus the template-prefix positions the run skips (G38): under scope `all`
                      that is every token forwarded. The same two figures are on every run's `render` summary
                      and on the preflight's memory answer, so history and forecast count alike.
    layer_selection   swept layers x rows. `select_layers` fits a classifier per (layer, pooling) over every
                      row's pooled vector; the two poolings are a constant factor.
    token_capture     chosen layers x tokens forwarded. `capture_tokens` runs a forward per chosen layer.
    training          probes x scored tokens, on ONE device. `train_rule` runs epochs over the standardised
                      batches, whose size is the scored tokens; one probe per (rule, window) per chosen layer
                      (`RunContext.rule_windows`), plus one per SAE k when `sae_variant`. CPU and GPU rates are
                      kept apart (`environment.training_device`); a run without that record trained on the
                      CPU, which every run did before commit 72e317394 (2026-10-22).
    calibrating       chosen layers x calibration rows. `_calibrate_probes` runs one forward per chosen layer
                      over `calibration_max_rows` of the calibration view (2,000 for a config without it);
                      with no calibration view it runs no forward and is not timed.
    evaluating        chosen layers x evaluation rows. `_evaluate_probes_on_sets` runs one forward per (layer,
                      set) over each set, sampled to `eval_max_rows` when that is set.

    (`rung` takes well under a second and is not forecast.)

WHICH RUNS ARE COMPARABLE

The same model and the same scope (a scope changes what "scored tokens" counts). Of those, the most recent
`RUNS_PER_STAGE` with a measurement of each stage: the stages' code has changed under them (calibration fell
from 36 passes to one per layer, evaluation from one forward per probe to one per layer), so an old run's rate
describes code that no longer runs. When the request names a card by UUID and runs on that card exist, only
those are used; otherwise the cards the runs ran on are listed in the basis.

THE ANSWER is a range: each stage's work times the lowest, the median and the highest rate among its runs;
the totals are the sums. `basis` names the runs and the units; `stages` shows every rate. A stage with no
measured rate is listed in `missing_stages` and the total is then a lower bound, said as "at least".

`forecast_time` and everything it calls are pure: the backtest in `0xcc/reviews/` runs them on run records
read over MCP, and `time_forecast_for_request` runs them on the database's rows.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The stages forecast, in the order a run reaches them (`probe_monitor_run.STAGES` less `rung`).
FORECAST_STAGES: Tuple[str, ...] = (
    "rendering",
    "pooled_capture",
    "layer_selection",
    "token_capture",
    "training",
    "calibrating",
    "evaluating",
)

#: What each stage's work is counted in. Shown on the answer so a reader can check the arithmetic.
STAGE_UNITS: Dict[str, str] = {
    "rendering": "rows rendered",
    "pooled_capture": "tokens forwarded (scored tokens + template-prefix positions skipped)",
    "layer_selection": "swept layers x rows",
    "token_capture": "chosen layers x tokens forwarded",
    "training": "probes x scored tokens, on the device the batches go to",
    "calibrating": "chosen layers x calibration rows",
    "evaluating": "chosen layers x evaluation rows (each set capped at eval_max_rows)",
}

#: The most recent comparable runs whose rate is used, per stage. See the module docstring.
RUNS_PER_STAGE = 5

#: The calibration sample of a config written before `calibration_max_rows` (`probe_monitor_run`
#: `LEGACY_CALIBRATION_MAX_ROWS`; repeated here so this module imports nothing heavy).
LEGACY_CALIBRATION_MAX_ROWS = 2000

#: The window every windowed rule used before `rolling_windows` existed (`DEFAULT_WINDOW`): one probe.
_DEFAULT_WINDOWS = 1


@dataclass(frozen=True)
class WorkUnits:
    """One run's work, counted per stage (see the module docstring). `None` is "not known", never zero."""

    rows: Optional[int]
    scored_tokens: Optional[int]
    tokens: Optional[int]
    swept_layers: Optional[int]
    chosen_layers: Optional[int]
    probes: Optional[int]
    calibration_rows: Optional[int]
    eval_rows: Optional[int]
    eval_rows_by_set: Dict[str, Optional[int]] = field(default_factory=dict)

    def per_stage(self) -> Dict[str, Optional[int]]:
        def times(a: Optional[int], b: Optional[int]) -> Optional[int]:
            return None if a is None or b is None else int(a) * int(b)

        return {
            "rendering": self.rows,
            "pooled_capture": self.tokens,
            "layer_selection": times(self.swept_layers, self.rows),
            "token_capture": times(self.chosen_layers, self.tokens),
            "training": times(self.probes, self.scored_tokens),
            "calibrating": times(self.chosen_layers, self.calibration_rows),
            "evaluating": times(self.chosen_layers, self.eval_rows),
        }


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def windowed_rules() -> frozenset:
    """The rules that train one probe per rolling window (`ml.probe_monitor_model.WINDOWED_RULES`)."""
    from ..ml.probe_monitor_model import WINDOWED_RULES

    return WINDOWED_RULES


def probes_trained(config: Mapping[str, Any], chosen_layers: int) -> int:
    """Probes a run trains: one per (rule, window) per chosen layer, plus one per SAE k per rule per layer.

    Mirrors `RunContext.rule_windows` (a windowed rule trains one probe per entry of `rolling_windows`, which
    falls back to the single default window) and the SAE loop in `execute_probe_run`."""
    rules = list(config.get("rules") or ["mean"])
    windows = len(config.get("rolling_windows") or []) or _DEFAULT_WINDOWS
    windowed = windowed_rules()
    per_layer = sum(windows if rule in windowed else 1 for rule in rules)
    if config.get("sae_variant"):
        per_layer += len(rules) * len(config.get("sae_k") or [128])
    return per_layer * max(int(chosen_layers), 1)


def view_rows(counts: Mapping[str, Any]) -> Optional[int]:
    """The rows a view yields once mapped: positives plus negatives (`ProbeMonitorDataset.counts`)."""
    if not isinstance(counts, Mapping):
        return None
    positive, negative = _int(counts.get("positive")), _int(counts.get("negative"))
    if positive is None and negative is None:
        return None
    return (positive or 0) + (negative or 0)


def eval_rows_of(
    eval_ids: Sequence[str], rows_by_view: Mapping[str, Optional[int]], eval_max_rows: Any
) -> Tuple[Optional[int], Dict[str, Optional[int]]]:
    """Rows the evaluating stage scores: each set whole, or sampled to `eval_max_rows` when it is larger
    (`subsample_indices` takes everything at or under the cap). `None` when any set's size is unknown."""
    cap = _int(eval_max_rows)
    by_set: Dict[str, Optional[int]] = {}
    for view_id in eval_ids:
        rows = rows_by_view.get(view_id)
        by_set[view_id] = None if rows is None else (min(rows, cap) if cap else rows)
    if any(rows is None for rows in by_set.values()):
        return None, by_set
    return sum(by_set.values()), by_set  # type: ignore[arg-type]


def calibration_rows_of(
    calibration_id: Optional[str], rows_by_view: Mapping[str, Optional[int]], config: Mapping[str, Any]
) -> Optional[int]:
    """Rows the calibrating stage runs a forward over; 0 when there is no calibration view (no forward)."""
    if not calibration_id:
        return 0
    rows = rows_by_view.get(calibration_id)
    if rows is None:
        return None
    cap = _int(config.get("calibration_max_rows")) or LEGACY_CALIBRATION_MAX_ROWS
    return min(rows, cap)


def units_of_run(run: Mapping[str, Any], rows_by_view: Mapping[str, Optional[int]]) -> WorkUnits:
    """A COMPLETED run's work, from what it recorded (`environment.render`, `layers_swept`, the layer
    selection's `chosen`) and its config. The same quantities `units_of_request` counts for a new run."""
    environment = run.get("environment") or {}
    config = run.get("config") or {}
    render = environment.get("render") or {}
    rows = _int(render.get("rendered"))
    scored = _int(render.get("scored_tokens"))
    tokens = None if scored is None else scored + (_int(render.get("template_prefix_skipped")) or 0)
    swept = len(environment.get("layers_swept") or []) or None
    chosen_list = (run.get("layer_selection") or {}).get("chosen")
    chosen = len(chosen_list) if chosen_list else _int(config.get("top_n_layers")) or 1
    eval_rows, by_set = eval_rows_of(list(run.get("eval_dataset_ids") or []), rows_by_view,
                                     config.get("eval_max_rows"))
    return WorkUnits(
        rows=rows,
        scored_tokens=scored,
        tokens=tokens,
        swept_layers=swept,
        chosen_layers=chosen,
        probes=probes_trained(config, chosen),
        calibration_rows=calibration_rows_of(run.get("calibration_dataset_id"), rows_by_view, config),
        eval_rows=eval_rows,
        eval_rows_by_set=by_set,
    )


def units_of_request(
    request: Mapping[str, Any], memory: Mapping[str, Any], rows_by_view: Mapping[str, Optional[int]]
) -> WorkUnits:
    """A NEW run's work: the train view's rows and tokens as the preflight tokenised them (the memory
    answer's `rows`, `real_tokens`, `template_prefix_skipped`, `swept_layers`), the rest from the request."""
    config = request.get("config") or {}
    checked = bool(memory.get("checked"))
    rows = _int(memory.get("rows")) if checked else None
    scored = _int(memory.get("real_tokens")) if checked else None
    tokens = None if scored is None else scored + (_int(memory.get("template_prefix_skipped")) or 0)
    swept = _int(memory.get("swept_layers")) if checked else None
    top_n = _int(config.get("top_n_layers")) or 1
    chosen = min(top_n, swept) if swept else top_n
    eval_rows, by_set = eval_rows_of(list(request.get("eval_dataset_ids") or []), rows_by_view,
                                     config.get("eval_max_rows"))
    return WorkUnits(
        rows=rows,
        scored_tokens=scored,
        tokens=tokens,
        swept_layers=swept,
        chosen_layers=chosen,
        probes=probes_trained(config, chosen),
        calibration_rows=calibration_rows_of(request.get("calibration_dataset_id"), rows_by_view, config),
        eval_rows=eval_rows,
        eval_rows_by_set=by_set,
    )


def training_device_of_run(run: Mapping[str, Any]) -> Optional[str]:
    """'cpu' or 'gpu' for the device every chosen layer trained on; None when layers differed.

    A run without `environment.training_device` trained on the CPU: before commit 72e317394 (2026-10-22)
    `execute_probe_run` passed `torch.device("cpu")` to every `train_rule` call, and that commit added the
    record."""
    record = (run.get("environment") or {}).get("training_device")
    if not record:
        return "cpu"
    kinds = {"gpu" if str(entry.get("device", "")).startswith("cuda") else "cpu"
             for entry in record.values() if isinstance(entry, Mapping)}
    return kinds.pop() if len(kinds) == 1 else None


def card_free_bytes(runs: Iterable[Mapping[str, Any]]) -> List[int]:
    """The card's free memory each comparable run READ at its training stage (`training_device.free_bytes`)."""
    found: List[int] = []
    for run in runs:
        for entry in ((run.get("environment") or {}).get("training_device") or {}).values():
            value = _int(entry.get("free_bytes")) if isinstance(entry, Mapping) else None
            if value is not None:
                found.append(value)
    return found


def expected_training_device(
    batch_bytes: Optional[int],
    runs: Sequence[Mapping[str, Any]],
    *,
    storage_bytes: int = 4,
    host_room_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Where the new run's batches will go and at what width, by the run's own rule (`batch_holding`) applied
    to the free memory comparable runs measured on the card at that stage. Unknown when either figure is missing.

    `batch_bytes` is the batches at FLOAT32 (the memory answer's `float32_batches_bytes`): `batch_holding`
    decides the width. For a float32 model (`storage_bytes` 4) that is the only width. `host_room_bytes` is the
    memory answer's `batch_room_bytes`, which decides the width when the batches stay in host memory."""
    free = card_free_bytes(runs)
    if batch_bytes is None:
        return {"expected": None, "reason": "the memory forecast gave no batch size"}
    if not free:
        return {"expected": None, "reason": "no comparable run recorded the card's free memory at training"}
    from .probe_monitor_trainer import plan_holding

    median_free = int(statistics.median(free))
    holding = plan_holding(
        storage_bytes=int(storage_bytes), float32_bytes=int(batch_bytes), card_present=True,
        card_free=median_free, host_room=host_room_bytes,
    )
    on_card = holding.on_card
    return {
        "expected": "gpu" if on_card else "cpu",
        "reason": f"{holding.device_reason} (the median free memory {len(free)} comparable run(s) measured at training)",
        "batch_bytes": holding.batch_bytes,
        "free_bytes_measured": median_free,
        # ENH-006 G25: the width matters to the rate, so it is decided here as the run decides it.
        "element_bytes": holding.element_bytes,
        "held_reason": holding.width_reason,
    }


def held_width_of_run(run: Mapping[str, Any]) -> Optional[int]:
    """The bytes per slot every chosen layer's batches were held at (`environment.training_device.*.element_bytes`,
    recorded since ENH-006 G25); None when not recorded or when layers differed."""
    record = (run.get("environment") or {}).get("training_device") or {}
    widths = {entry.get("element_bytes") for entry in record.values() if isinstance(entry, Mapping)}
    if len(widths) != 1:
        return None
    width = widths.pop()
    return int(width) if isinstance(width, int) else None


def comparable_runs(
    history: Sequence[Mapping[str, Any]], *, model_id: str, scope: str, gpu: Optional[str]
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """Completed runs of this model at this scope with stage timings, newest first; narrowed to the
    named card when the request names one by UUID and runs on it exist."""
    runs = [
        run for run in history
        if run.get("model_id") == model_id
        and run.get("status", "completed") == "completed"
        and (run.get("environment") or {}).get("stage_seconds")
        and str((run.get("config") or {}).get("scope", "all")) == scope
    ]
    runs.sort(key=lambda run: str(run.get("completed_at") or run.get("created_at") or ""), reverse=True)
    card = {"pinned": False, "cards": sorted({str(r.get("gpu_uuid")) for r in runs if r.get("gpu_uuid")})}
    if gpu and str(gpu).startswith("GPU-"):
        on_card = [run for run in runs if run.get("gpu_uuid") == gpu]
        if on_card:
            runs = on_card
            card = {"pinned": True, "cards": [str(gpu)]}
        else:
            card["note"] = f"no comparable run ran on {gpu}; the rates are from the cards listed"
    return runs, card


def view_overlap(stage: str, run: Mapping[str, Any], request_views: Mapping[str, Any]) -> float:
    """How much of `stage`'s data `run` shares with the request: the evaluation sets in common over all
    the sets of either (Jaccard), or 1.0 for the same calibration view. 0.0 for every other stage.

    A row is the unit of both stages, and rows differ in length by set: an everyday-text row is many times
    a short probe row, and a 1,001,551-row calibration corpus is read whole before its sample is drawn.
    A run that scored the same sets measured the rate this run will see."""
    if stage == "evaluating":
        mine, theirs = set(request_views.get("eval") or ()), set(run.get("eval_dataset_ids") or ())
        union = mine | theirs
        return len(mine & theirs) / len(union) if union else 0.0
    if stage == "calibrating":
        wanted = request_views.get("calibration")
        return 1.0 if wanted and run.get("calibration_dataset_id") == wanted else 0.0
    return 0.0


def ordered_for(
    stage: str, runs: Sequence[Mapping[str, Any]], request_views: Mapping[str, Any]
) -> List[Mapping[str, Any]]:
    """`runs` (newest first) in the order `stage` takes its rates from: most shared data first, then newest.
    Python's sort is stable, so equal overlaps keep the recency order."""
    return sorted(runs, key=lambda run: -view_overlap(stage, run, request_views))


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 5400 else f"{seconds / 60:.0f} min"


def forecast_time(
    *,
    units: WorkUnits,
    history: Sequence[Mapping[str, Any]],
    rows_by_view: Mapping[str, Optional[int]],
    model_id: str,
    scope: str,
    gpu: Optional[str] = None,
    batch_bytes: Optional[int] = None,
    storage_bytes: int = 4,
    host_room_bytes: Optional[int] = None,
    eval_max_rows: Any = None,
    runs_per_stage: int = RUNS_PER_STAGE,
    eval_dataset_ids: Sequence[str] = (),
    calibration_dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The forecast. Pure: `history` is run records (the shape `GET /probe-monitors/runs/{id}` answers).

    `batch_bytes` is the batches at float32 and `storage_bytes` the model's stored width (see
    `expected_training_device`)."""
    request_views = {"eval": set(eval_dataset_ids), "calibration": calibration_dataset_id}
    eval_facts = {
        "eval_rows": units.eval_rows,
        "eval_rows_by_set": dict(units.eval_rows_by_set),
        "eval_max_rows": _int(eval_max_rows),
    }
    runs, card = comparable_runs(history, model_id=model_id, scope=scope, gpu=gpu)
    if not runs:
        reason = (
            f"no completed run of model {model_id} at scope {scope} records stage timings, so there is "
            f"nothing measured to forecast from"
        )
        return {
            "forecast": False,
            "reason": reason,
            "summary": f"No time forecast: {reason}. {_eval_sentence(eval_facts)}",
            **eval_facts,
        }

    device = expected_training_device(batch_bytes, runs, storage_bytes=storage_bytes,
                                      host_room_bytes=host_room_bytes)
    width = device.get("element_bytes")
    unknown_width_rates: List[str] = []
    new_units = units.per_stage()
    stages: Dict[str, Any] = {}
    caveats: List[str] = []
    missing: List[str] = []
    used: Dict[str, Mapping[str, Any]] = {}
    for stage in FORECAST_STAGES:
        work = new_units[stage]
        entry: Dict[str, Any] = {"unit": STAGE_UNITS[stage], "units": work}
        if work is None:
            entry["reason"] = "this run's work in this unit is not known (see `work`)"
            missing.append(stage)
            stages[stage] = entry
            continue
        if work == 0:
            entry["seconds"] = {"low": 0.0, "typical": 0.0, "high": 0.0}
            entry["reason"] = ("no calibration view, so no calibration forward" if stage == "calibrating"
                               else "no work in this unit")
            stages[stage] = entry
            continue
        rates = []
        for run in ordered_for(stage, runs, request_views):
            if stage == "training":
                ran_on = training_device_of_run(run)
                if ran_on is None or (device["expected"] is not None and ran_on != device["expected"]):
                    continue
                # ENH-006 G25: a 16-bit run's batches held raw train 2.3-6x slower per epoch than widened.
                held_at = held_width_of_run(run)
                if width is not None and held_at is not None and held_at != width:
                    continue
            seconds = (run.get("environment") or {}).get("stage_seconds", {}).get(stage)
            done = units_of_run(run, rows_by_view).per_stage()[stage]
            if not isinstance(seconds, (int, float)) or seconds <= 0 or not done:
                continue
            rates.append({"run_id": run.get("id"), "seconds_per_unit": float(seconds) / done,
                          "seconds": float(seconds), "units": done,
                          **({"device": training_device_of_run(run)} if stage == "training" else {})})
            used[str(run.get("id"))] = run
            if stage == "training" and int(storage_bytes) < 4 and held_width_of_run(run) is None:
                unknown_width_rates.append(str(run.get("id")))
            if len(rates) >= runs_per_stage:
                break
        if not rates:
            entry["reason"] = (
                "no comparable run measured this stage"
                + (f" on the {device['expected']}" if stage == "training" and device["expected"] else "")
            )
            missing.append(stage)
            stages[stage] = entry
            continue
        per_unit = sorted(rate["seconds_per_unit"] for rate in rates)
        entry["seconds"] = {
            "low": round(work * per_unit[0], 1),
            "typical": round(work * statistics.median(per_unit), 1),
            "high": round(work * per_unit[-1], 1),
        }
        entry["rates"] = rates
        # How far the measured runs disagree: the stage the operator should trust least.
        entry["spread"] = round(per_unit[-1] / per_unit[0], 2) if per_unit[0] > 0 else None
        if stage == "evaluating" and request_views["eval"]:
            seen = set().union(*(set(r.get("eval_dataset_ids") or ()) for r in runs
                                 if str(r.get("id")) in {str(x["run_id"]) for x in rates}))
            unseen = sorted(request_views["eval"] - seen)
            if unseen:
                caveats.append(
                    f"{len(unseen)} of {len(request_views['eval'])} evaluation set(s) were never scored by the "
                    f"runs the evaluating rate comes from ({', '.join(unseen)}); rows differ in length by set, so "
                    f"a set of longer rows takes longer per row than this rate says"
                )
        stages[stage] = entry

    if unknown_width_rates:
        caveats.append(
            f"the training rate comes in part from run(s) that did not record the width their batches were held at "
            f"({', '.join(unknown_width_rates)}); a 16-bit run's batches held as float16 train 2-6x slower per epoch "
            f"than widened to float32, so the training time may be off"
        )
    total = {
        key: round(sum(stage["seconds"][key] for stage in stages.values() if "seconds" in stage), 1)
        for key in ("low", "typical", "high")
    }
    basis_runs = [
        {"run_id": run.get("id"), "gpu_uuid": run.get("gpu_uuid"), "completed_at": run.get("completed_at"),
         "total_seconds": round(sum(v for k, v in (run.get("environment") or {}).get("stage_seconds", {}).items()
                                    if isinstance(v, (int, float))), 1)}
        for run in runs if str(run.get("id")) in used
    ]
    answer: Dict[str, Any] = {
        "forecast": True,
        "complete": not missing,
        "seconds": total,
        "hours": {key: round(value / 3600, 2) for key, value in total.items()},
        "missing_stages": missing,
        "least_certain": least_certain(stages),
        "caveats": caveats,
        "stages": stages,
        "training_device": device,
        "work": asdict(units),
        **eval_facts,
        "basis": {
            "model_id": model_id,
            "scope": scope,
            "runs": basis_runs,
            "card": card,
            "runs_per_stage": runs_per_stage,
            "units": dict(STAGE_UNITS),
            "method": (
                "each stage's work in its unit x the seconds per unit that stage took on the most recent "
                f"{runs_per_stage} comparable completed runs (same model, same scope); low, typical and high "
                "are the lowest, median and highest of those rates, and the totals are their sums"
            ),
        },
    }
    answer["summary"] = summary_line(answer)
    return answer


def least_certain(stages: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The stage whose measured rates disagree most, weighted by its time: where the range comes from."""
    spread = [
        (stage, entry["spread"], entry["seconds"]["high"] - entry["seconds"]["low"])
        for stage, entry in stages.items()
        if entry.get("spread") and "seconds" in entry
    ]
    if not spread:
        return None
    stage, ratio, width = max(spread, key=lambda item: item[2])
    return {"stage": stage, "spread": ratio, "range_seconds": round(width, 1)}


def _eval_sentence(eval_facts: Mapping[str, Any]) -> str:
    rows, cap = eval_facts.get("eval_rows"), eval_facts.get("eval_max_rows")
    sets = len(eval_facts.get("eval_rows_by_set") or {})
    if not sets:
        return "No evaluation sets named, so evaluating counts 0 rows."
    if rows is None:
        return f"The size of {sets} evaluation set(s) is not known."
    capped = f", each capped at eval_max_rows {cap:,}" if cap else ", every set whole (no eval_max_rows)"
    return f"Evaluation scores {rows:,} rows over {sets} set(s){capped}."


def summary_line(answer: Mapping[str, Any]) -> str:
    """The one line an operator reads before approving."""
    if not answer.get("forecast"):
        return str(answer.get("summary") or answer.get("reason") or "No time forecast.")
    seconds = answer["seconds"]
    runs = answer["basis"]["runs"]
    lead = "At least " if not answer["complete"] else "About "
    line = (
        f"{lead}{_hours(seconds['typical'])} ({_hours(seconds['low'])} to {_hours(seconds['high'])}) "
        f"by the measured rates of {len(runs)} completed run(s) of this model at scope {answer['basis']['scope']}"
        + (f" (newest {runs[0]['run_id']})" if runs else "")
        + ". "
        + _eval_sentence(answer)
    )
    device = (answer.get("training_device") or {}).get("expected")
    if device:
        width = (answer.get("training_device") or {}).get("element_bytes")
        line += f" Training expected on the {device.upper()}" + (
            ", with 16-bit batches held as float16 (float32 does not fit)." if width == 2 else ".")
    else:
        line += " Where training runs is not known, so its range spans CPU and GPU rates."
    worst = answer.get("least_certain")
    if worst and worst["spread"] and worst["spread"] >= 1.5:
        line += f" Least certain: {worst['stage']} (its measured rates differ {worst['spread']:.1f}-fold)."
    caveats = answer.get("caveats") or []
    if any("never scored" in caveat for caveat in caveats):
        line += " Evaluation sets not seen before: the evaluating time may be off."
    if any("did not record the width" in caveat for caveat in caveats):
        line += " Some training rates are from runs that did not record their batch width: the training time may be off."
    if answer["missing_stages"]:
        line += f" Not measured, so not counted: {', '.join(answer['missing_stages'])}."
    return line


def run_record(row: Any) -> Dict[str, Any]:
    """A `ProbeMonitorRun` row as the mapping `forecast_time` reads (the run API's field names)."""
    return {
        "id": row.id,
        "model_id": row.model_id,
        "status": row.status,
        "gpu_uuid": row.gpu_uuid,
        "config": dict(row.config or {}),
        "environment": dict(row.environment or {}),
        "layer_selection": dict(row.layer_selection or {}),
        "eval_dataset_ids": list(row.eval_dataset_ids or []),
        "calibration_dataset_id": row.calibration_dataset_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "completed_at": row.completed_at.isoformat() if getattr(row, "completed_at", None) else None,
    }


#: Completed runs read per forecast. Ample for `RUNS_PER_STAGE` per stage after the scope filter.
HISTORY_LIMIT = 50


def card_uuid(gpu: Any) -> Optional[str]:
    """The UUID of the card a request names (an index resolved as submit resolves it), else the request
    as given. Never raises: a card that cannot be resolved only means the rates are not narrowed to it."""
    if gpu is None:
        return None
    try:
        from .gpu_placement import resolve_request

        return resolve_request(gpu)
    except Exception:  # noqa: BLE001 - see the docstring
        return str(gpu)


def time_forecast_from_db(session: Any, request: Mapping[str, Any], memory: Mapping[str, Any]) -> Dict[str, Any]:
    """The forecast for a run request, from the database's completed runs and views."""
    from ..models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun

    config = request.get("config") or {}
    model_id = str(request.get("model_id"))
    rows = (
        session.query(ProbeMonitorRun)
        .filter(ProbeMonitorRun.model_id == model_id, ProbeMonitorRun.status == "completed")
        .order_by(ProbeMonitorRun.completed_at.desc().nulls_last(), ProbeMonitorRun.created_at.desc())
        .limit(HISTORY_LIMIT)
        .all()
    )
    history = [run_record(row) for row in rows]
    wanted = set(request.get("eval_dataset_ids") or [])
    if request.get("calibration_dataset_id"):
        wanted.add(request["calibration_dataset_id"])
    for run in history:
        wanted.update(run["eval_dataset_ids"])
        if run["calibration_dataset_id"]:
            wanted.add(run["calibration_dataset_id"])
    rows_by_view: Dict[str, Optional[int]] = {}
    if wanted:
        for view in session.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id.in_(sorted(wanted))).all():
            rows_by_view[view.id] = view_rows(view.counts or {})
    return forecast_time(
        units=units_of_request(request, memory, rows_by_view),
        history=history,
        rows_by_view=rows_by_view,
        model_id=model_id,
        scope=str(config.get("scope", "all")),
        gpu=card_uuid(request.get("gpu")),
        # The float32 figure and the stored width, so the card and the width are decided as the run decides them
        # (`batch_holding`); `batches_bytes` from a forecast without them is a float32 model's (ENH-006 G25).
        batch_bytes=_int(memory.get("float32_batches_bytes", memory.get("batches_bytes")))
        if memory.get("checked") else None,
        storage_bytes=_int(memory.get("storage_bytes")) or 4,
        host_room_bytes=_int(memory.get("batch_room_bytes")),
        eval_max_rows=config.get("eval_max_rows"),
        eval_dataset_ids=list(request.get("eval_dataset_ids") or []),
        calibration_dataset_id=request.get("calibration_dataset_id"),
    )


def time_forecast_for_request(request: Mapping[str, Any], memory: Mapping[str, Any]) -> Dict[str, Any]:
    """The `/runs/forecast` route's `time_forecast`. Blocking (database reads), so the route runs it in a
    thread. A failure is an answer with `forecast: False` and the reason: it must never cost the memory
    answer beside it."""
    from ..core.database import SyncSessionLocal

    session = SyncSessionLocal()
    try:
        return time_forecast_from_db(session, request, memory)
    except Exception as exc:  # noqa: BLE001 - a forecast that cannot run says so
        logger.warning("probe run time forecast could not run: %s", exc)
        return {"forecast": False, "reason": f"{type(exc).__name__}: {exc}",
                "summary": f"No time forecast: it could not run ({type(exc).__name__}: {exc})."}
    finally:
        session.close()

