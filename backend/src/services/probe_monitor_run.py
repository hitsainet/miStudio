"""The probe run, stage by stage (032 FR-5–FR-10, FR-12, FR-15).

`execute_probe_run` is the orchestration the Celery task calls. It is here rather than
in the task so the sequence can be driven by a test without Celery, and so the task
stays what it should be: a lease, a session, and an error handler.

THE STAGES ARE WRITTEN TO THE ROW AS THEY BEGIN. "Failed at token_capture" and "failed
at training" have different causes and different fixes, and a run that dies must be
diagnosable from its row rather than from a worker log that has since rotated.

⚠ THE CANCELLATION CHECK SITS AT EVERY STAGE BOUNDARY, and stage boundaries are the
finest points this pipeline can cleanly abandon work at. A check inside the capture
loop would also be honest, and the capture functions take a progress callback that a
caller can use for one — but the guarantee made here is the boundary, because that is
where the artifacts are consistent.

⚠ EVERY NUMBER THE RUN DEPENDS ON IS RECORDED IN `environment` (FR-15). Model revision,
dtype, template hash, seeds, per-stage wall times. A run nobody can reproduce is an
anecdote, and the reason this estate cannot compare its 21 older SAE trainings is
precisely that what fed them was not written down.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import random

import numpy as np

from ..core.cancellation import record_progress
from ..ml.probe_monitor_model import DEFAULT_WINDOW
from ..ml.native_dtype import load_dtype_record, storage_dtype_of_load
from .probe_monitor_metrics import length_band_decisions
from ..core.clock import utc_now
from ..core.config import settings

logger = logging.getLogger(__name__)

StageCallback = Callable[[str, float, Optional[str]], None]

#: Stage names in order, with the fraction of the run each has finished by. Declared
#: rather than computed so progress is monotone and means the same thing across runs.
STAGES: Tuple[Tuple[str, float], ...] = (
    ("rendering", 5.0),
    ("pooled_capture", 30.0),
    ("layer_selection", 40.0),
    ("token_capture", 60.0),
    ("training", 80.0),
    ("calibrating", 85.0),
    ("evaluating", 97.0),
    ("rung", 100.0),
)
STAGE_PROGRESS: Dict[str, float] = dict(STAGES)

#: How often a long stage writes a within-stage heartbeat. 60 s is the interval the
#: tokenization packing fix settled on after a live job was reaped; the reaper's window
#: is 90 minutes, so this leaves ninety heartbeats of margin.
HEARTBEAT_SECONDS: float = 60.0


def sub_band(name: str, index: int, count: int) -> Tuple[float, float]:
    """The `index`-th of `count` equal slices of `name`'s band.

    A stage that repeats work per probe needs its progress divided, not restarted. With one
    probe this is exactly `heartbeat_band(name)`, so the single-probe case is unchanged.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if not 0 <= index < count:
        raise ValueError(f"index {index} is outside 0..{count - 1}")
    low, high = heartbeat_band(name)
    width = (high - low) / count
    return low + index * width, low + (index + 1) * width


def heartbeat_band(name: str) -> Tuple[float, float]:
    """The progress range a within-stage heartbeat may report for `name`.

    ⚠ NOT A PAIR OF LITERALS, BECAUSE I GOT THEM WRONG. `stage()` writes its stage's
    value ON ENTRY, so `pooled_capture` is at 30.0 the moment it begins. I first gave it
    the band (5.0, 30.0) — reading STAGES as "the value this stage ends at" — which would
    have sent the progress bar from 30% straight back to 5% at the first heartbeat and
    then crawled back up. Derived from the tuple instead, so the two cannot disagree:
    a stage heartbeats from its OWN entry value toward the NEXT stage's.

    The last stage has no successor, so it tops out at 100.
    """
    names = [stage for stage, _ in STAGES]
    if name not in names:
        raise KeyError(f"{name} is not a stage; STAGES has {names}")
    index = names.index(name)
    low = STAGE_PROGRESS[name]
    high = STAGE_PROGRESS[names[index + 1]] if index + 1 < len(names) else 100.0
    return low, high


def artifact_dir_for(run_id: str) -> Path:
    """`<data_dir>/probe_monitors/<run_id>` — derived from the id, never from a request.

    A path taken from a request is a path-traversal sink, and this one is passed to
    `unlink` and `rmtree` when a run is deleted.
    """
    return Path(settings.data_dir) / "probe_monitors" / run_id


@dataclass
class RunContext:
    """Everything the stages share, assembled once."""

    run_id: str
    config: Dict[str, Any]
    artifact_dir: Path
    seed: int
    scope: str
    max_length: int
    target_fpr: float
    rules: List[str]
    #: One probe per entry, for rules that take a window. See `rule_windows`.
    rolling_windows: List[int]
    top_n_layers: int
    val_fraction: float

    def rule_windows(self, rule: str) -> List[Optional[int]]:
        """The windows to train `rule` at — `[None]` for every rule that has none.

        `None` means "call `train_rule` without a window and let it bind its own default",
        which is exactly what every call did before this existed. Keeping that as a distinct
        value rather than substituting `DEFAULT_WINDOW` means a rule that takes no window
        cannot acquire one by accident, and `rule_parameters` stays the single place that
        decides which rules record one.
        """
        from ..ml.probe_monitor_model import WINDOWED_RULES

        if rule not in WINDOWED_RULES:
            return [None]
        return list(self.rolling_windows) or [None]


def cancel_checker_for(run_id: str, db: Any = None) -> Any:
    """The throttled cancellation checker for this run.

    ⚠ `guard_allows` IS NOT THIS, AND THE FIRST VERSION CALLED IT. `guard_allows(kind,
    current_status, incoming_status)` is a pure predicate about whether a STATUS WRITE
    would be accepted onto a row in a given state — it does not read the cancel flag and
    it returns a bool. Calling it as `guard_allows(kind, run_id)` and discarding the
    result type-checks, runs, and checks NOTHING: the pipeline had a cancellation
    checkpoint at every stage boundary that could never fire.

    Caught by the end-to-end test asserting a requested cancel actually stops the run —
    the mechanism was declared and not wired, which is the failure this repo has
    recorded more than once. `cancel_checker` is the real checkpoint: it returns a
    callable that RAISES `OperatorCancelled`, throttled on time rather than call count.
    """
    from ..core.cancellation import cancel_checker

    return cancel_checker("probe_monitor_run", run_id, db=db)


def resolve_layers(
    config: Dict[str, Any], n_layers: int
) -> List[int]:
    """The layers to sweep: an explicit list, or a stride over the model's depth.

    ⚠ REFUSES AN OUT-OF-RANGE LAYER RATHER THAN CLAMPING IT. `register_hooks` warns and
    SKIPS a layer beyond the model, so a clamped or silently-dropped index produces a
    run that swept fewer layers than its config claims — and the config is what the
    report shows. A 422 naming the depth is the only honest answer.
    """
    explicit = config.get("layers")
    if explicit:
        out_of_range = [layer for layer in explicit if layer < 0 or layer >= n_layers]
        if out_of_range:
            raise ValueError(
                f"layers {sorted(out_of_range)} are outside this model's depth "
                f"(0..{n_layers - 1}); a skipped layer would make the run's config "
                f"claim a sweep it did not perform"
            )
        return sorted(set(int(layer) for layer in explicit))
    stride = int(config.get("stride") or 5)
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    # From the LAST layer backwards, so the deepest layer is always swept: a probe's
    # concept is usually late-layer, and a forward stride from 0 can miss the final
    # layer entirely for most strides.
    layers = sorted(range(n_layers - 1, -1, -stride))
    return layers or [n_layers - 1]


def context_from_row(row: Any) -> RunContext:
    config = dict(row.config or {})
    return RunContext(
        run_id=row.id,
        config=config,
        artifact_dir=artifact_dir_for(row.id),
        seed=int(config.get("seed", 1337)),
        scope=str(config.get("scope", "all")),
        max_length=int(config.get("max_length", 4096)),
        target_fpr=float(config.get("target_fpr", 0.01)),
        rules=list(config.get("rules") or ["mean"]),
        # ⚠ An OLD RUN ROW HAS NO `rolling_windows`. Falling back to the module default is
        # correct and not a fabrication: every probe trained before this field existed was
        # trained at `DEFAULT_WINDOW`, so this reproduces what actually happened.
        rolling_windows=list(config.get("rolling_windows") or [DEFAULT_WINDOW]),
        top_n_layers=int(config.get("top_n_layers", 1)),
        val_fraction=float(config.get("val_fraction", 0.15)),
    )


def persist_calibration_negatives(
    artifact_dir: Path, probe_id: str, scores: Sequence[float], *, name: str = "negatives"
) -> Optional[str]:
    """Save the negative scores a threshold was cut from. Returns the path, or None.

    ⚠ **THIS IS WHAT MAKES THE OPERATING POINT A DIAL RATHER THAN A RERUN.** A threshold is the
    `(1 - target_fpr)` quantile of exactly these numbers, so re-deriving it at another target is
    arithmetic over an array already computed. Without the file, answering "what would 5% look
    like on our traffic" meant a fresh ~70-minute run that re-captured activations to recompute a
    percentile — and that is why the operating point felt fixed at training time when it is the
    one part genuinely free to move.

    The evaluation stage has always persisted its score arrays (`probe.id__dataset_id.npy`) for
    the same reason. Calibration did not, and the asymmetry was invisible because nothing failed:
    the threshold was correct, just unrepeatable.

    ⚠ `name` EXISTS BECAUSE ONE ARRAY IS NOT THE WHOLE DIAL. The base `negatives` moves the
    GLOBAL threshold and nothing else. A probe also carries a bar per contract window and a bar
    per length band, and both are quantiles of DIFFERENT arrays — a window changes which tokens
    enter every row's aggregate, and a length band needs the per-row token counts. All of those
    arrays are in memory during the `calibrating` stage and were thrown away, so moving any bar
    but the global one still cost a GPU pass to recover a few kilobytes. Saved under
    `negatives`, `lengths`, and `negatives_window_<scope>`.

    Never raises. Losing the convenience must not cost a probe its threshold, so a failure is
    logged and returns None — which reads correctly downstream as "this probe predates the
    persistence, or its write failed; recomputing needs a new run".
    """
    import numpy as _np

    try:
        path = Path(artifact_dir) / f"{probe_id}__calibration_{name}.npy"
        _np.save(path, _np.asarray(list(scores), dtype=_np.float32))
        return str(path)
    except Exception as exc:
        logger.warning(
            "probe %s: could not persist calibration %s (%s); the threshold is "
            "unaffected, but re-deriving it at another target FPR will need a new run",
            probe_id, name, exc,
        )
        return None


def resolve_calibration_negatives(
    calibration_dataset_id: Optional[str],
    *,
    from_calibration_set: Callable[[], Tuple[List[float], int]],
    from_validation_negatives: Callable[[], List[float]],
) -> Tuple[List[float], str]:
    """Which negatives place the threshold, and the `threshold_source` that describes them.

    ⚠ **EXTRACTED BECAUSE THE INLINE VERSION COULD NOT BE TESTED, AND MY FIRST GUARD PROVED IT.**
    The branch lived in the calibrating stage, and the test asserting the calibration scorer was
    reachable read the stage's AST for a CALL. Replacing `if calibration_id:` with `if False:` —
    which restores the original defect exactly — left that call present inside a dead branch and
    the whole suite green. A test that reads source cannot tell a statement that RUNS from one
    that is merely there; this repo has recorded that failure repeatedly, and I reproduced it here
    within hours of fixing the same shape in miLLM's startup reconciliation.

    So the decision is a function with two injected producers. Its behaviour is exercised directly
    — which path ran, and what source was recorded — and neutering the branch now fails a
    behavioural assertion rather than hiding behind a scrape.

    ⚠ The two returned values MOVE TOGETHER BY CONSTRUCTION, which is the point. `threshold_source`
    is what the exported document tells a consumer the operating point came from, and
    `probe_definition_builder` gates `decision.calibration` on it. Returning the scores and the
    label from one place makes it impossible to read one distribution and claim the other — the
    defect this replaced, where a run named its calibration set in the export while calibrating on
    the training set's validation split.
    """
    if calibration_dataset_id:
        scores, n_rows = from_calibration_set()
        logger.info(
            "calibrating on set %s: %d of %d scored rows are negatives",
            calibration_dataset_id, len(scores), n_rows,
        )
        return scores, "calibration_set"
    return from_validation_negatives(), "validation_negatives"


def execute_probe_run(
    db: Any,
    run_id: str,
    *,
    on_stage: Optional[StageCallback] = None,
    model_loader: Optional[Callable[[Any], Tuple[Any, Any, str]]] = None,
) -> Dict[str, Any]:
    """The whole run. Returns a summary dict for the completion event.

    `model_loader` is injectable so the CPU integration test can drive the real
    sequence against a tiny random LM — the alternative is a test that exercises each
    stage in isolation and never runs the dispatch, which is exactly how feature 030's
    detection path shipped with three independent breaks behind green unit tests.
    """
    from ..models.probe_monitor import (
        ProbeMonitor,
        ProbeMonitorDataset,
        ProbeMonitorRun,
    )
    from .probe_monitor_capture import capture_pooled, capture_tokens, count_scored_tokens
    from .probe_monitor_inputs import class_counts, split_rows
    from .probe_monitor_render import render_all, template_hash
    from .probe_monitor_service import (
        ResolvedColumns,
        build_view,
        load_columns,
        resolve_dataset_path,
    )
    from .probe_monitor_trainer import (
        calibrate,
        head_from_trained,
        select_layers,
        train_rule,
    )

    row = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == run_id).first()
    if row is None:
        raise ValueError(f"probe monitor run {run_id} not found")
    context = context_from_row(row)
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    row.artifact_dir = str(context.artifact_dir)

    environment: Dict[str, Any] = dict(row.environment or {})
    timings: Dict[str, float] = {}

    check_cancelled = cancel_checker_for(run_id, db=db)

    def stage(name: str, message: Optional[str] = None) -> float:
        # ⚠ `raise_if_cancelled()`, NOT `check_cancelled()`. The checker's `__call__`
        # RETURNS a bool; only `raise_if_cancelled` raises. Calling it bare and
        # discarding the answer is the same "declared but not wired" defect one level
        # down from the `guard_allows` mix-up, and it type-checks just as cleanly.
        check_cancelled.raise_if_cancelled(f"before stage {name}")
        row.stage = name
        # ⚠ THE ENVIRONMENT IS PERSISTED AT EVERY BOUNDARY, NOT ONLY AT THE END. It used
        # to be assigned once, on the success path, so a run that failed at
        # `token_capture` stored NOTHING about how it had been configured — no swept
        # layers, no template hash, no seed, no stage timings, and no render summary.
        # FR-15 requires a run to be reproducible from what it stored, and the first two
        # Stage 1 acceptance failures were diagnosed from the worker log because the row
        # itself had nothing. A dict reassignment is what JSONB needs to notice the
        # change; mutating it in place does not mark the attribute dirty.
        row.environment = dict(environment)
        db.commit()
        progress = STAGE_PROGRESS[name]
        if on_stage is not None:
            on_stage(name, progress, message)
        timings[name] = time.time()
        return progress

    def heartbeat(
        name: str, span: Tuple[float, float], unit: str = "batches"
    ) -> Callable[[int, int], None]:
        """A within-stage progress writer, throttled on TIME (FR-14).

        ⚠ WITHOUT THIS A LONG STAGE IS INDISTINGUISHABLE FROM A DEAD WORKER. The stage
        helper writes once at each boundary, and the first Stage 1 acceptance run spent
        OVER FORTY MINUTES inside `pooled_capture` alone — a single stage longer than
        the reaper's 90-minute threshold is reachable on a larger set, and
        `cleanup_stuck_probe_monitor_runs`'s own docstring asserts the opposite ("tens
        of minutes rather than hours"). That reaper needs three conditions, so today
        `task_looks_alive` is the only thing standing between a live 3090 job and being
        reclaimed. This estate has already reaped a LIVE 5.8-hour packing job that
        reported only over the WebSocket.

        Throttled on TIME, not on batch count, for the reason the cancellation module
        records: `% 5` over attribution batches is up to twenty minutes, while `% 25`
        over training steps can be milliseconds. A batch here is seconds to minutes
        depending on the token budget, so counting batches cannot bound the gap.

        The write goes through `record_progress`, so it REFUSES to move a terminal row:
        an in-flight heartbeat must not overwrite a cancellation the endpoint has just
        written (the whole point of the cooperative design).
        """
        low, high = span
        state = {"last": 0.0}

        def report(done: int, total: int) -> None:
            now = time.time()
            if now - state["last"] < HEARTBEAT_SECONDS and done < total:
                return
            state["last"] = now
            fraction = (done / total) if total else 0.0
            record_progress(
                "probe_monitor_run",
                run_id,
                progress=round(low + (high - low) * fraction, 2),
                db=db,
            )
            if on_stage is not None:
                on_stage(name, low + (high - low) * fraction, f"{done}/{total} {unit}")

        return report

    def finish(name: str) -> None:
        started = timings.get(name)
        if started is not None:
            environment.setdefault("stage_seconds", {})[name] = round(time.time() - started, 3)
        # ⚠ NO COMMIT HERE, AND THAT IS MEASURED, NOT ASSUMED. It had one, and mutation
        # H10 — deleting it — left the suite green. The reason is that `stage()` commits
        # the environment on ENTRY, so the timing this line just recorded reaches the
        # database one line later, at the next boundary; the final stage's timing is
        # covered by the completion write. A second commit per stage was redundant, and a
        # line no test can miss is a line that should not be here.

    # ── the train view ────────────────────────────────────────────────────────
    train_view = (
        db.query(ProbeMonitorDataset)
        .filter(ProbeMonitorDataset.id == row.train_dataset_id)
        .first()
    )
    if train_view is None:
        raise ValueError(f"training dataset {row.train_dataset_id} not found")

    loader = model_loader or _load_model_for_run
    model, tokenizer, architecture = loader(row)
    # ⚠ THE RUN'S OWN BOOKKEEPING IS COMMITTED HERE, BEFORE ANYTHING READS THE ROW.
    #
    # `artifact_dir` above and the `gpu_request` / `gpu_uuid` the loader just set from its
    # placement are both assigned and not yet committed at this point, and the next thing
    # to touch the row is `stage()`'s cancel check. `CancelCheck._fetch` reads with
    # `.populate_existing()`, deliberately, so a long-lived task session can observe a
    # cancel written by the API process (MIS-E2E-057) — and `SyncSessionLocal` is
    # configured `autoflush=False`, so that read refreshes every attribute from the
    # database and REVERTS both assignments.
    #
    # MEASURED, not inferred. Of the four Stage 1 runs, the only one that kept its
    # `artifact_dir` and `gpu_uuid` is the one that died BEFORE reaching a stage; every
    # run that reached `stage()` lost both, including the run that completed. So
    # `pmr_f70190de92e6` finished, reached rung 2, and stored `artifact_dir = NULL` while
    # `/data/probe_monitors/pmr_f70190de92e6/tokens_layer11.f16` held 7.7 GB — and
    # `DELETE /runs/{id}` reclaims the directory by reading `artifact_dir`, so the capture
    # of every successful run was unreclaimable. Reproduced directly in the pod:
    # assign, `raise_if_cancelled()`, and the attribute reads back None.
    #
    # One commit, not two: nothing between the assignment and here touches this session,
    # so an earlier commit would be redundant — and a redundant line is one no test can
    # miss.
    db.commit()
    n_layers = _model_depth(model)
    layers = resolve_layers(context.config, n_layers)

    environment.update(
        {
            "n_layers": n_layers,
            "layers_swept": layers,
            "template_hash": template_hash(tokenizer),
            "seed": context.seed,
            "scope": context.scope,
            "max_length": context.max_length,
            "architecture": architecture,
            # The precision the model ACTUALLY loaded at, and why (`ml/native_dtype.py`). The
            # module docstring has promised "dtype" here since 032, and nothing wrote it — so
            # when the served model's precision differed (miLLM: bfloat16; this loader:
            # float16), nothing on the run could say so, and a probe failed parity before
            # anyone knew why. None means "not recorded" (a loader that stamps nothing), never
            # a guessed value. The definition builder copies `model_dtype` from HERE.
            **(load_dtype_record(model) or {"model_dtype": None}),
        }
    )

    # ── S1 render ─────────────────────────────────────────────────────────────
    stage("rendering")
    examples, rendered_examples, render_summary, labels = _prepare_examples(
        db, train_view, context, tokenizer, role="train"
    )
    environment["render"] = render_summary
    train_index, val_index = _split_indices(examples, context)
    finish("rendering")

    positives, negatives = class_counts([examples[i] for i in train_index])
    if min(positives, negatives) < 1:
        raise ValueError(
            f"the training split has {positives} positive and {negatives} negative rows; "
            f"a single-class split cannot train a probe. Lower val_fraction or check the "
            f"pair column, which keeps whole pairs on one side"
        )

    # ── S2 pooled capture ─────────────────────────────────────────────────────
    stage("pooled_capture", f"{len(layers)} layers in one pass")
    pooled = capture_pooled(
        model,
        rendered_examples,
        layers,
        scope=context.scope,
        architecture=architecture,
        pad_id=_pad_id(tokenizer),
        progress=heartbeat("pooled_capture", heartbeat_band("pooled_capture")),
    )
    finish("pooled_capture")

    # ── S3 layer selection ────────────────────────────────────────────────────
    stage("layer_selection")
    selection = select_layers(
        pooled, labels, train_index, val_index,
        top_n=context.top_n_layers, seed=context.seed,
    )
    row.layer_selection = selection.as_dict()
    db.commit()
    finish("layer_selection")

    # ── S4 token capture at the chosen layer(s) ───────────────────────────────
    stage("token_capture", f"layers {selection.chosen}")
    captures = {}
    # Stored at the dtype the load implies (`ml/native_dtype.py`): float16 for a 16-bit load —
    # exact for bfloat16 — and float32 for a float32 one. The suffix names what is inside.
    storage = storage_dtype_of_load(model)
    for layer in selection.chosen:
        captures[layer] = capture_tokens(
            model,
            rendered_examples,
            layer,
            context.artifact_dir / f"tokens_layer{layer}.{'f16' if storage == 'float16' else 'f32'}",
            dtype=storage,
            scope=context.scope,
            architecture=architecture,
            pad_id=_pad_id(tokenizer),
            progress=heartbeat("token_capture", heartbeat_band("token_capture")),
        )
    finish("token_capture")

    # ── S5 train each rule ────────────────────────────────────────────────────
    stage("training", f"{len(context.rules)} rules")
    #
    # ⚠ TRAINING WAS THE ONLY LONG STAGE WITH NO DATABASE HEARTBEAT, AND IT IS THE LONGEST.
    # `pooled_capture`, `token_capture`, `calibrating` and `evaluating` all heartbeat; this
    # one did not, so `probe_monitor_runs.updated_at` went stale for the WHOLE stage —
    # `_persist_probe` writes to `probe_monitors`, a different table, so a probe landing did
    # not refresh it either.
    #
    # Measured on run `pmr_54563518010f` (2026-10-01): the quiet timer climbed monotonically
    # through 737s -> 841s -> 963s -> 1084s across four probes landing, on a 24-probe run
    # pacing at ~3.5 min/probe. `cleanup_stuck_probe_monitor_runs` examines a row at 90
    # minutes of silence and reaps it on the following pass, so a run of this size was
    # arithmetically certain to be killed MID-TRAINING while burning 207% CPU.
    #
    # This is the exact failure `heartbeat`'s own docstring warns about eight lines up —
    # "this estate has already reaped a LIVE 5.8-hour packing job that reported only over
    # the WebSocket" — and the socket emits were indeed flowing the entire time.
    #
    total_probes = sum(
        len(context.rule_windows(rule)) for rule in context.rules
    ) * max(len(captures), 1)
    train_beat = heartbeat("training", heartbeat_band("training"), unit="probes")
    probes: List[Any] = []
    for layer, capture in captures.items():
        memmap = capture.open_memmap()
        rows_by_index = [
            np.asarray(memmap[int(capture.offsets[i]) : int(capture.offsets[i + 1])])
            for i in range(len(rendered_examples))
        ]
        train_rows = [rows_by_index[i] for i in train_index]
        val_rows = [rows_by_index[i] for i in val_index]
        train_labels = [labels[i] for i in train_index]
        val_labels = [labels[i] for i in val_index]

        # ⚠ BUILT ONCE PER LAYER, NOT ONCE PER RULE. The standardisation statistics and the
        # length-bucketed batches depend only on the TRAIN rows and the byte budget — nothing
        # that varies by rule — so a four-rule run was rebuilding ~16.6 GiB of standardised
        # activations four times from the same memmap, for four identical results.
        #
        # Statistics from TRAIN ONLY, which is the thing the hoist could quietly break: computing
        # them over train+validation leaks the validation distribution's scale into the model and
        # would still train, still converge and still look fine.
        # `test_training_refactors_change_nothing` requires bit-identical weights under reuse and
        # has a fixture that can tell the two statistic sets apart.
        # `torch` is imported inside this function further down (the calibrating stage), which
        # makes it a LOCAL name for the whole function — using it here, above that statement,
        # raises UnboundLocalError. Imported locally rather than relying on an import that has
        # not run yet.
        import torch as _torch

        from ..services.probe_monitor_trainer import _standardisation, _standardised_batches

        _device = _torch.device("cpu")
        _mean, _std = _standardisation(train_rows)
        _prebuilt = (
            _mean, _std,
            _standardised_batches(train_rows, _mean, _std, _device),
            _standardised_batches(val_rows, _mean, _std, _device) if val_rows else [],
        )

        for rule in context.rules:
            # One probe per (rule, window). `rule_windows` yields `[None]` for every rule
            # that takes no window, so this loop is a no-op widening for all of them and
            # the `window=None` call below is byte-for-byte the call that was here before.
            for window in context.rule_windows(rule):
                check_cancelled.raise_if_cancelled("between rules")
                trained = train_rule(
                    rule, train_rows, train_labels, val_rows, val_labels, seed=context.seed,
                    prebuilt=_prebuilt,
                    **({} if window is None else {"window": int(window)}),
                )
                head = head_from_trained(trained, layer)
                probe = _persist_probe(
                    db, row, trained, head, layer, context, capture, selection
                )
                probes.append((probe, head, trained, val_rows, val_labels))
                # AFTER the probe is persisted, so the heartbeat reports work that is
                # durable rather than work in flight. `record_progress` refuses to move a
                # terminal row, so this cannot overwrite a cancellation.
                train_beat(len(probes), total_probes)

            # The k-sparse SAE variant, one probe per k (FR-8, D15). Trained from the
            # SAME token capture, so the dense and sparse probes describe the same rows.
            if context.config.get("sae_variant"):
                sae_id = _sae_for_layer(db, row.model_id, layer)
                for k in context.config.get("sae_k") or [128]:
                    check_cancelled.raise_if_cancelled("between sae variants")
                    probes.append(
                        train_sae_variant(
                            db, row, context, capture, labels, train_index, val_index,
                            sae_id=sae_id, rule=rule, k=int(k),
                        )
                    )
    finish("training")

    # ── S6 calibrate ──────────────────────────────────────────────────────────
    stage("calibrating")
    from ..ml.probe_monitor_model import combine

    import torch

    calibration_id = getattr(row, "calibration_dataset_id", None)

    # ⚠ A HEARTBEAT AND SUB-PROGRESS, BECAUSE THIS STAGE GREW FOUR-FOLD.
    #
    # `calibrating` used to be one scoring pass and finished inside a minute on validation
    # negatives. Per-window thresholds made it `probes x (1 + windows)` passes over the whole
    # calibration corpus — MEASURED on `pmr_15c9e85a1d04`: one pass over 2000 OpenHermes
    # conversations took about seven minutes, and the stage sat at a fixed 85% for over half an
    # hour writing NOTHING to the row.
    #
    # Two separate failures in that sentence. Writing nothing for half an hour is how this estate
    # reaped a LIVE 5.8-hour packing job as "worker lost". And a fixed 85% over a half-hour stage
    # means the only way to answer "which pass is it on?" is to grep a worker log for a line that
    # happens to be emitted once per pass — which is what an operator actually had to do.
    #
    # The scorer already takes a progress callback, so the beat reaches BATCH level inside each
    # pass rather than only between them.
    #
    # ⚠ ONE FORWARD PER LAYER, NOT FOUR PER PROBE. The stage used to run, for every probe, a base
    # pass and then one pass per contract window — each re-reading the corpus, re-rendering the
    # same rows and running a full forward. `pmr_416ce6b1ee63` (9 probes, 3 layers) paid 36 of them
    # at ~482 s: 17,346 s. A window is a mask applied after the forward and probes on one layer
    # read the same activations, so the rows are prepared once and each layer is run once, scoring
    # every probe on it under every scope. Thresholds are bit-identical to the old passes; see
    # `_score_calibration_group`.
    _calibrate_probes(
        db, probes, calibration_id, context, model, tokenizer, architecture,
        beat=heartbeat("calibrating", heartbeat_band("calibrating"), unit="layer forwards"),
    )
    db.commit()
    finish("calibrating")

    # ── S7/S8 evaluation and rung ─────────────────────────────────────────────
    stage("evaluating", f"{len(row.eval_dataset_ids or [])} sets")
    evaluated = 0
    # ⚠ ONE SUB-BAND PER PROBE, BECAUSE THE BAR WENT BACKWARDS. Every probe gets its own
    # heartbeat closure, and each one started at the band's low end again — observed live
    # on run `pmr_413d1e0e03cf`: 99.2% → 100.0% → **97.7%** as the second probe began.
    # `test_progress_is_monotone` could not see it: the fixture's evaluation sets fit in a
    # single batch, so each probe emitted exactly one report at `done == total`, i.e. 100.0
    # every time — three identical values are sorted, and the rewind was invisible.
    # ⚠ GROUPED BY LAYER, SO ONE FORWARD SERVES EVERY PROBE THAT READS IT.
    #
    # This ran the model once per (probe, set). With `top_n_layers=1` every probe reads the SAME
    # layer, so the activations were identical and recomputed for each: measured at 1164 s for
    # one probe over five sets and 2388.6 s for two — linear in probes, for one layer's work.
    # A four-rule run paid ~80 minutes where ~20 would do, which is most of the reason nobody
    # runs the sweep the feature exists to support.
    #
    # Grouped rather than assumed: with `top_n_layers > 1` probes do NOT share a layer, and
    # scoring one against another layer's activations would give plausible numbers in the wrong
    # basis, invisible in every metric. `forward_scores_many` refuses a mixed group outright, so
    # this grouping is a requirement rather than a convention.
    by_layer: Dict[int, List[Any]] = {}
    for entry in probes:
        by_layer.setdefault(entry[0].layer, []).append(entry)

    for group_index, (group_layer, group) in enumerate(sorted(by_layer.items())):
        evaluated += _evaluate_probes_on_sets(
            db, group, context, model, tokenizer, architecture,
            list(row.eval_dataset_ids or []),
            progress=heartbeat(
                "evaluating", sub_band("evaluating", group_index, len(by_layer))
            ),
        )
        # ⚠ INSIDE THE GROUP LOOP, NOT AFTER IT. Promoting the rung only once every group had
        # finished would mean a run that fails on the second group leaves the first group's
        # probes at rung 0 with complete out-of-distribution evaluations beside them — which is
        # the exact defect `pm_8428015843e3` hit, and which grouping by layer almost
        # reintroduced. Caught by `test_recompute_rung_is_called_INSIDE_the_evaluation_loop`.
        for probe, *_rest in group:
            # ⚠ PROMOTED HERE, NOT ONLY IN THE "rung" STAGE BELOW, and that is a fix rather than a
            # belt-and-braces duplicate. The rung stage runs after EVERY probe, so a run that fails on
            # probe 2 left probe 1 at rung 0 with five complete out-of-distribution evaluations beside
            # it. That is exactly what happened to `pm_8428015843e3` on `pmr_5d81ad3b81f2`: the SAE
            # probe raised, the loop never reached the rung stage, and a fully-evaluated dense probe
            # kept a rung that said "trained".
            #
            # It is not a cosmetic staleness. The rung is what 033's export gate reads, so the probe
            # was refused for lacking evidence it had; and had it been exported with an
            # acknowledgement, the document would have stated "rung 0 — trained" above five
            # out-of-distribution AUROCs, contradicting itself in the field a consumer trusts most.
            #
            # `recompute_rung` is idempotent and cheap (it reads rows and commits), so the stage below
            # stays as the place that reports progress, and this call is the one that makes the value
            # true the moment it can be.
            recompute_rung(db, probe.id)
    finish("evaluating")

    stage("rung")
    for probe, *_ in probes:
        recompute_rung(db, probe.id)
    finish("rung")

    best = select_best_probe(db, [p for p, *_ in probes])
    if best is not None:
        best.selected = True
    row.status = "completed"
    row.progress = 100.0
    row.environment = environment
    row.completed_at = utc_now()
    db.commit()

    return {
        "run_id": run_id,
        "probes": [p.id for p, *_ in probes],
        "selected": best.id if best is not None else None,
        "layers_chosen": selection.chosen,
        "evaluations": evaluated,
        "stage_seconds": environment.get("stage_seconds", {}),
    }


def probe_selection_key(
    val_auroc: Optional[float], ood_aurocs: Sequence[Optional[float]]
) -> Tuple[float, float]:
    """`(mean OOD AUROC, val AUROC)` — the order probes are ranked in.

    ⚠ **SELECTION USED TO BE `max(val_auroc)` ALONE, AND THAT NUMBER CANNOT CHOOSE.** Every
    `mean` probe trained here scores 0.9982 on it; the layer sweep ties to within 0.0001 and
    the run's own report said so in as many words — "a near-tie, so the chosen layer is close
    to arbitrary". Worse, it understates what matters: `attention` scored 0.9590 in
    distribution against `mean`'s 0.9982, a gap of 0.04, while out of distribution the same
    two probes scored 0.6986 and 0.8841, a gap of 0.19. A saturated in-distribution metric
    was deciding which probe ships.

    The out-of-distribution evaluations are already computed and committed by the time the
    selection runs — this function simply reads them. `val_auroc` stays as the tie-break,
    because a probe with no OOD evaluation at all must still be rankable and must still lose
    to one that has them.

    Returns a tuple so `max` is total and stable: no OOD evidence sorts as 0.0, which is
    below any real AUROC including a chance one, and is NOT the same as scoring chance.
    """
    scored = [float(a) for a in ood_aurocs if a is not None]
    mean_ood = sum(scored) / len(scored) if scored else 0.0
    return (mean_ood, float(val_auroc or 0.0))


def select_best_probe(db: Any, probes: Sequence[Any]) -> Optional[Any]:
    """The probe a run marks `selected`, ranked by `probe_selection_key`."""
    from ..models.probe_monitor import ProbeMonitorDataset, ProbeMonitorEvaluation

    if not probes:
        return None
    ood = (
        db.query(ProbeMonitorEvaluation, ProbeMonitorDataset)
        .join(ProbeMonitorDataset, ProbeMonitorEvaluation.dataset_id == ProbeMonitorDataset.id)
        .filter(ProbeMonitorEvaluation.probe_id.in_([p.id for p in probes]))
        .filter(ProbeMonitorDataset.distribution == "out_of_distribution")
        .all()
    )
    by_probe: Dict[str, List[Optional[float]]] = {}
    for evaluation, _view in ood:
        # A REFUSED evaluation contributes nothing rather than a 0.5. "Could not be scored"
        # and "scored at chance" are different facts, and averaging the second in would
        # punish a probe for a set that was too small for anyone.
        if evaluation.status != "refused":
            by_probe.setdefault(evaluation.probe_id, []).append(
                (evaluation.metrics or {}).get("auroc")
            )
    return max(
        probes,
        key=lambda p: probe_selection_key(
            (p.val_metrics or {}).get("val_auroc"), by_probe.get(p.id, [])
        ),
    )


def _sae_for_layer(db: Any, model_id: str, layer: int) -> str:
    """The ready residual SAE at `(model, layer)`, or a 422-shaped refusal.

    Named in the error, because "no SAE" is unactionable while "no ready SAE for
    LFM2.5-1.2B at layer 12" tells an operator exactly what to train or import.
    """
    from ..models.external_sae import ExternalSAE

    row = (
        db.query(ExternalSAE)
        .filter(
            ExternalSAE.model_id == model_id,
            ExternalSAE.layer == layer,
        )
        .first()
    )
    if row is None:
        raise ValueError(
            f"the SAE variant was requested but there is no external SAE for model "
            f"{model_id} at layer {layer}; train or import one, or submit the run "
            f"without sae_variant"
        )
    return row.id


def _model_depth(model: Any) -> int:
    from ..ml.layer_discovery import discover_transformer_structure

    return int(discover_transformer_structure(model).num_layers)


def _pad_id(tokenizer: Any) -> int:
    for candidate in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if candidate is not None:
            return int(candidate)
    return 0


def resolve_weights_dir(
    model_id: str, file_path: Optional[str], quantized_path: Optional[str]
) -> str:
    """The weights directory to load from, or a refusal naming both candidates.

    ⚠ `quantized_path` IS A PHANTOM ON EVERY ROW THIS ESTATE HAS. `model_tasks` sets it
    for any format other than FP32 — `<models_dir>/quantized/<id>_<fmt>` — but nothing
    ever writes that directory: quantization is applied at LOAD time by bitsandbytes,
    from the raw checkpoint. So the column names a path that does not exist, and this
    loader preferred it unconditionally.

    The failure was not a missing-file error. `from_pretrained` treats a path it cannot
    find as a HUB REPO ID, so the first Stage 1 acceptance run died with

        OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
        '/data/models/quantized/m_40e78d80_FP16'

    which names neither the model nor the real problem, and sent the investigation into
    the capture path. Hence the refusal below: it names the model and both candidates,
    so the next reader is not debugging a hub error over a missing directory.

    Preferring the quantized directory only when it is really there is the
    `logit_lens_service` precedent. `extraction_service`, `analysis_service`,
    `circuit_capture_service` and `circuit_attribution_service` all read `file_path`
    alone, so today every path leads to the raw download either way — which is the
    HuggingFace cache root (`.../raw/<id>/models--<org>--<name>/snapshots/<sha>`), and
    `_load_model` resolves the snapshot inside it.
    """
    candidates = [("quantized_path", quantized_path), ("file_path", file_path)]
    for column, value in candidates:
        if not value:
            continue
        resolved = settings.resolve_data_path(value)
        if resolved.exists():
            return str(resolved)
        logger.info(
            "model %s: %s is %s, which does not exist on disk — skipping",
            model_id, column, value,
        )
    named = ", ".join(f"{column}={value!r}" for column, value in candidates if value)
    raise ValueError(
        f"model {model_id} has no usable weights directory on disk "
        f"({named or 'both file_path and quantized_path are unset'}); download or "
        f"re-download the model before training a probe"
    )



def probe_precision(db: Any, run: Any) -> Dict[str, Any]:
    """What precision a run TRAINED at, what its model loads at NOW, and whether that forbids reuse.

    ⚠ EVERY PATH THAT RE-LOADS A RUN'S MODEL MUST ASK THIS FIRST. Since 2026-10-03 models load at
    the checkpoint's own precision (`ml/native_dtype.py`) — bfloat16 for everything here — while
    every probe trained before that was fitted, evaluated and thresholded at float16. Re-evaluating
    such a probe would write bfloat16 AUROCs into its float16 evidence; re-cutting its windows
    would cut new bars from bfloat16 negatives under a float16 head and move them silently. The
    same holds for a later run if its model row's quantization changes.

    Answered WITHOUT loading the model (review round 1, M2): the recorded value is on the run and
    the load dtype is the resolver's answer over the snapshot's config.json — so a refusal costs
    no GPU. `loads_at` is None when the weights cannot be found; the load would fail anyway.
    """
    from ..ml.native_dtype import resolve_for_snapshot
    from ..models.model import Model
    from .activation_service import resolve_model_snapshot

    environment = getattr(run, "environment", None) or {}
    recorded = environment.get("model_dtype")
    recorded_quant = environment.get("quantization")
    loads_at: Optional[str] = None
    current_quant: Optional[str] = None
    model_row = db.query(Model).filter(Model.id == run.model_id).first()
    if model_row is not None:
        current_quant = str(getattr(model_row.quantization, "value", model_row.quantization))
        try:
            snapshot = resolve_model_snapshot(
                resolve_weights_dir(run.model_id, model_row.file_path, model_row.quantized_path)
            )
            loads_at = resolve_for_snapshot(model_row.quantization, snapshot).name
        except Exception as exc:  # noqa: BLE001 - advisory to an arithmetic re-cut; never fatal
            # Round 3, LOW-5: `resolve_model_snapshot` raises ActivationExtractionError (an
            # Exception, not ValueError/OSError) for a cache dir with no snapshot — which made a
            # zero-GPU recalibration preview fail on a field it only reports.
            logger.warning("probe_precision: cannot resolve %s's load dtype: %s", run.model_id, exc)
            loads_at = None
    refusal: Optional[str] = None
    if recorded is None:
        refusal = (
            f"run {run.id} predates precision recording: it trained at float16 (every load path "
            "hardcoded it before 2026-10-03), and its model now loads at "
            f"{loads_at or 'its checkpoint precision'} — work done on it today would describe a "
            "different distribution from the one its weights and threshold came from. Retrain it."
        )
    elif loads_at is not None and loads_at != recorded:
        refusal = (
            f"run {run.id} trained at {recorded}, but its model now loads at {loads_at} (the "
            "model row's quantization changed). Restore the quantization it trained under, or "
            "retrain."
        )
    elif recorded_quant is not None and current_quant is not None and recorded_quant != current_quant:
        # Same precision, different quantization: a Q4 and an FP16 load of one bfloat16
        # checkpoint both resolve to "bfloat16" and still read different activations.
        refusal = (
            f"run {run.id} trained on its model at {recorded_quant}, and the model row is now "
            f"{current_quant}: the same precision, different activations. Restore {recorded_quant} "
            "or retrain."
        )
    return {"recorded": recorded, "loads_at": loads_at, "refusal": refusal}


class ProbePrecisionRefused(ValueError):
    """Reloading this run's model would read a different precision from the one it trained at.

    Raised inside GPU tasks, which record it as the failure; the API refuses the same condition
    with a 409 before queueing (`endpoints.probe_monitors.refuse_precision_mismatch`)."""


def require_matching_precision(db: Any, run: Any) -> None:
    """Refuse — before any model load — work that would mix precisions into a probe's evidence."""
    refusal = probe_precision(db, run)["refusal"]
    if refusal is not None:
        raise ProbePrecisionRefused(refusal)

def _load_model_for_run(row: Any) -> Tuple[Any, Any, str]:
    """Load the run's model at the row's own quantization. Returns (model, tokenizer, arch).

    ⚠ THE ROW'S QUANTIZATION IS READ, NOT IGNORED. A model row configured Q4 loaded at
    native dtype produced activations agreeing with the served model at only ~0.93
    cosine per token here, and the mismatch is invisible in the numbers — the probe
    trains fine against a distribution nothing serves. The J-lens fitter shipped that
    bug once; reusing `ActivationService._load_model` is what keeps this path honest,
    because it is the same loader the SAE corpus is captured with.

    It requires CUDA, deliberately: a probe trained on CPU activations of a quantized
    model is not the probe that will be served. The integration test injects its own
    loader rather than relaxing this.
    """
    from ..core.database import SyncSessionLocal
    from ..models.model import Model
    from .activation_service import ActivationService
    from .gpu_placement import place_job

    session = SyncSessionLocal()
    try:
        model_row = session.query(Model).filter(Model.id == row.model_id).first()
        if model_row is None:
            raise ValueError(f"model {row.model_id} not found")
        local_path = resolve_weights_dir(
            row.model_id, model_row.file_path, model_row.quantized_path
        )
        quantization = model_row.quantization
        architecture = model_row.architecture or ""
    finally:
        session.close()

    # `place_job` under the task's `gpu_job` claim chooses the card the run was
    # submitted for and makes it CURRENT — bitsandbytes and any bare `torch.cuda`
    # call would otherwise land on index 0 whatever card was leased. With no CUDA,
    # AUTO returns the CPU, which is what lets the integration test run at all.
    placement = place_job(row.gpu_request or "auto")
    for column, value in placement.gpu_columns().items():
        setattr(row, column, value)
    service = ActivationService()
    model, tokenizer = service._load_model(local_path, quantization, placement)
    return model, tokenizer, architecture


def subsample_indices(total: int, limit: int, seed: int) -> Optional[List[int]]:
    """`limit` ascending indices drawn from `range(total)`, or `None` to take everything.

    ⚠ **A SEEDED RANDOM SAMPLE, NEVER THE FIRST N, AND THAT IS NOT FASTIDIOUSNESS.** The
    calibration corpus this exists for is OpenHermes-2.5, which is written **one source at a
    time**: 14 of its 15 labelled sources occupy a single contiguous row range. This estate
    has already shipped the head-N version of this mistake — an extraction read 7,000 of
    189,087 blocks and saw **2 of 15 sources**, never reaching the 18.2% that is
    glaive-code-assist nor the 49.6% unlabelled remainder. A threshold calibrated on the
    first 2,000 rows of a source-ordered corpus is calibrated on one source.

    Ascending rather than in draw order so the sample indexes the underlying columns in one
    forward pass, and reproducible from `seed` so two runs of the same config calibrate on the
    same rows.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if total <= limit:
        return None
    return sorted(random.Random(seed).sample(range(total), limit))


def _prepare_examples(
    db: Any, view: Any, context: RunContext, tokenizer: Any, *, role: str,
    max_rows: Optional[int] = None,
) -> Tuple[List[Any], List[Any], Dict[str, Any], List[int]]:
    """Load, map, filter and render one probe dataset.

    Returns `(examples, rendered, summary, labels)` — and the RENDERED OBJECTS ARE
    SEPARATE FROM THE SUMMARY on purpose.

    ⚠ THEY WERE ONCE THE SAME DICT, AND IT BROKE THE WHOLE RUN AT THE LAST COMMIT. The
    summary goes into `probe_monitor_runs.environment`, which is JSONB, so a
    `RenderedExample` in it raises `TypeError: Object of type RenderedExample is not
    JSON serializable` — at the FINAL commit, after every stage had succeeded and the
    artifacts were written. Caught by the end-to-end pipeline test; no unit test of a
    single stage could see it, because the failure is in what one stage hands the next.
    """
    from ..models.dataset import Dataset
    from .probe_monitor_render import render_all
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
    inputs, raw_labels, pairs, total = load_columns(path, columns, split=view.split)

    # ⚠ CAPPED BEFORE `build_view`, NOT AFTER. Sampling afterwards would parse every one of the
    # corpus's rows to throw almost all of them away — the calibration view this was built for
    # holds 1,001,551 conversations. `max_rows` is None for every other caller, so train and
    # eval are untouched.
    if max_rows is not None:
        keep = subsample_indices(len(inputs), max_rows, context.seed)
        if keep is not None:
            inputs = [inputs[i] for i in keep]
            raw_labels = [raw_labels[i] for i in keep]
            pairs = [pairs[i] for i in keep] if pairs else pairs
            total = len(inputs)

    built = build_view(
        inputs,
        raw_labels,
        dict(view.label_mapping or {}),
        keyword_filter=view.keyword_filter,
        pair_values=pairs,
        role=role,
    )
    conversations = [example.messages for example in built.examples]
    roles_known = [example.kind != "messages_roles_guessed" for example in built.examples]
    rendered, summary = render_all(
        tokenizer,
        conversations,
        scope=context.scope,
        max_length=context.max_length,
        roles_known=roles_known,
    )
    # Rows whose render FAILED are dropped here — and counted. Keeping the label
    # alongside a None example would misalign labels and scores by a few rows, which
    # reads as a slightly weak probe rather than as an error.
    kept = [
        (example, render)
        for example, render in zip(built.examples, rendered)
        if render is not None
    ]
    examples = [example for example, _ in kept]
    rendered_objects = [render for _, render in kept]
    # JSON-SAFE ONLY. Nothing here may be a live object: this dict is stored in JSONB.
    summary_payload = summary.as_dict()
    summary_payload["source_rows"] = int(total)
    summary_payload["counts"] = built.counts.as_dict()
    summary_payload["kinds"] = dict(built.kinds)
    return (
        examples,
        rendered_objects,
        summary_payload,
        [example.label for example in examples],
    )


def _split_indices(examples: Sequence[Any], context: RunContext) -> Tuple[List[int], List[int]]:
    from .probe_monitor_inputs import split_rows

    train, validation = split_rows(
        examples, val_fraction=context.val_fraction, seed=context.seed
    )
    positions = {id(example): index for index, example in enumerate(examples)}
    return (
        [positions[id(example)] for example in train],
        [positions[id(example)] for example in validation],
    )


def _aggregate_rows(head: Any, rule: str, rule_params: Dict[str, Any], rows: Sequence[Any]) -> List[float]:
    """Aggregate scores for a list of (t, d) token blocks, through the shared rule."""
    import torch

    from ..ml.probe_monitor_model import combine

    if not rows:
        return []
    widths = [int(np.asarray(r).shape[0]) for r in rows]
    width = max(widths)
    d_model = int(np.asarray(rows[0]).shape[1])
    packed = torch.zeros((len(rows), width, d_model), dtype=torch.float32)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, block in enumerate(rows):
        array = np.asarray(block, dtype=np.float32)
        packed[index, : array.shape[0]] = torch.from_numpy(array)
        mask[index, : array.shape[0]] = True
    scores = head.token_scores(packed)
    logits = head.attention_logits(packed) if rule == "attention" else None
    aggregates = combine(
        rule,
        scores,
        mask=mask,
        attention_logits=logits,
        **{k: v for k, v in rule_params.items() if k in ("tau", "window")},
    )
    return [float(v) for v in aggregates.detach().cpu().tolist()]


def _persist_probe(
    db: Any, run: Any, trained: Any, head: Any, layer: int,
    context: RunContext, capture: Any, selection: Any,
    *,
    variant: str = "dense",
    sae_id: Optional[str] = None,
    sae_feature_indices: Optional[List[int]] = None,
) -> Any:
    """Write the probe row and its weights. Tensors go to safetensors, not the DB."""
    from safetensors.torch import save_file

    from ..models.probe_monitor import ProbeMonitor
    from ..ml.probe_monitor_model import is_streamable

    probe = ProbeMonitor(
        run_id=run.id,
        layer=layer,
        rule=trained.rule,
        rule_params=dict(trained.rule_params),
        variant=variant,
        sae_id=sae_id,
        sae_feature_indices=sae_feature_indices,
        val_metrics={
            "val_auroc": trained.val_auroc,
            # ⚠ `val_auroc` IS SELECTION-MAXIMISED, NOT A HELD-OUT ESTIMATE, AND NOTHING SAID SO.
            #
            # One validation split does three jobs: it ranks the layers (7 layers x 2 poolings on
            # the reference recipe), it picks the epoch (the MAX over up to 400), and it is then
            # reported as the probe's validation AUROC. That is not leakage — the split is honest
            # and `select_layers` fits its standardisation on train — but each selection makes the
            # surviving number optimistic, and a reader has no way to tell it apart from a
            # genuinely held-out one.
            #
            # It matters because the two sit side by side. Stage 1 recorded 0.9982 in-distribution
            # against 0.8841 out of distribution, and the first is a max over selections while the
            # second is clean. Anyone reading them as the same kind of number concludes the probe
            # collapses out of distribution; what it actually shows is one measured estimate and
            # one upper bound.
            #
            # An INDEPENDENT in-distribution evaluation VIEW gives the honest number — a separate
            # view never touched by training, which the run scores like any other evaluation set —
            # and the evidence ladder already refuses to claim rung 1 without one.
            "val_auroc_is_selection_maximised": True,
            "val_auroc_note": (
                "maximum over the epochs and layers this same validation split selected; "
                "optimistic by an unmeasured amount. For a held-out in-distribution number, "
                "add an evaluation view marked in_distribution."
            ),
            "best_epoch": trained.best_epoch,
            "epochs_run": trained.epochs_run,
            # ⚠ THE CURVE IS STORED, AND NOT STORING IT COST A WRONG DIAGNOSIS.
            #
            # `best_epoch` and `epochs_run` say WHERE training stopped and nothing about
            # whether it had converged. The Stage 1 acceptance run's attention probe
            # stopped at epoch 88 with its best at 48, and from those two numbers alone the
            # obvious reading was "early stopping cut it off" — so the fix would have been
            # to loosen the budget. Re-running the same rule for 600 epochs with early
            # stopping disabled gained 0.006 of AUROC: it had converged, and the real
            # finding was that the rule overfits (a LOWER training loss than the `mean`
            # rule and a much worse validation AUROC). The loss curve says which of those
            # two it is at a glance, and it had to be regenerated from the saved capture in
            # a 26-minute experiment because the run discarded it.
            #
            # Cheap: bounded by `epochs`, so a few hundred entries of three small numbers.
            "history": [
                {
                    "epoch": entry["epoch"],
                    "loss": entry["loss"],
                    "val_auroc": entry["val_auroc"],
                }
                for entry in (trained.history or [])
            ],
        },
        streamable=is_streamable(trained.rule),
    )
    db.add(probe)
    db.flush()          # need the id for the filename
    weights_path = context.artifact_dir / f"{probe.id}.safetensors"
    tensors = {
        "weight": trained.weight.contiguous(),
        "mean": trained.mean.contiguous(),
        "std": trained.std.contiguous(),
    }
    if trained.attention_query is not None:
        tensors["attention_query"] = trained.attention_query.contiguous()
    save_file(tensors, str(weights_path), metadata={"bias": repr(trained.bias)})
    probe.weights_path = str(weights_path)
    probe.norm_path = str(weights_path)     # mean/std travel in the same file
    db.commit()
    return probe


def encode_sae_features(
    db: Any, sae_id: str, capture: Any, device: str = "cpu"
) -> List[np.ndarray]:
    """Per-row SAE feature activations over a token capture (FR-8, D15).

    ⚠ `encode_with_training_normalization`, NEVER a bare `encode()`. `encode()` does not
    normalise — `forward()` does, and then calls it — so a caller reaching for `encode`
    hands the dictionary RAW activations when it was trained on normalised ones. Every
    circuit discovered from a capture on this estate was once mined that way: the features
    fire, the numbers are plausible, and the basis is wrong. MIS-E2E-083.

    ⚠ AND A NON-RESIDUAL SAE IS REFUSED. A probe reads `resid_post`; an SAE trained on an
    MLP or attention output describes a different space, and encoding one against the
    other produces features that mean nothing in particular.
    """
    import torch

    from ..models.external_sae import ExternalSAE
    from ..ml.sparse_autoencoder import encode_with_training_normalization
    from .circuit_capture_service import _load_sae_sync
    from .sae_hook_support import refuse_non_residual

    sae_row = db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
    if sae_row is None:
        raise ValueError(
            f"SAE {sae_id} is gone, so the k-sparse variant cannot be built. The probe "
            f"row keeps its sae_id (SET NULL only on delete) precisely so this is "
            f"reportable rather than silent"
        )
    refuse_non_residual(getattr(sae_row, "hook_type", None), "a probe monitor")
    sae = _load_sae_sync(sae_row, device)

    memmap = capture.open_memmap()
    rows: List[np.ndarray] = []
    with torch.no_grad():
        for index in range(len(capture.offsets) - 1):
            start, end = int(capture.offsets[index]), int(capture.offsets[index + 1])
            if end <= start:
                rows.append(np.zeros((0, 1), dtype=np.float32))
                continue
            block = torch.tensor(
                np.asarray(memmap[start:end]), dtype=torch.float32, device=device
            )
            features = encode_with_training_normalization(sae, block)
            rows.append(features.detach().cpu().numpy())
    return rows


def _probe_device(model: Any) -> Any:
    """The device to encode on: the model's, so the dictionary sits beside the weights.

    Falls back to None (meaning "wherever the tensors already are") when the model has no
    parameters — the injected CPU loader in the pipeline test.
    """
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return None


def sae_encoder_for(
    db: Any,
    sae_id: str,
    feature_indices: Sequence[int],
    device: Any = None,
) -> Callable[[Any], Any]:
    """A `(batch, tokens, d_model) -> (batch, tokens, k)` transform for a k-sparse probe.

    ⚠ IT MUST MATCH `train_sae_variant` EXACTLY, and "exactly" includes the normalisation.
    Training does `encode_with_training_normalization(sae, block)` and then `[:, indices]`;
    anything else here is a different basis, which produces plausible features and is invisible
    in the numbers (MIS-E2E-083, and the reason that helper exists at all rather than a bare
    `encode()`).

    The encode runs on the SAE's device and returns k columns, so only `k` floats per token cross
    back — 128 instead of 2,048 on the reference model. The activation arrives on the CPU because
    `HookManager` stores it there; the result is returned on whatever device it came from, because
    the head and the mask are there.
    """
    import torch

    from ..models.external_sae import ExternalSAE
    from ..ml.sparse_autoencoder import encode_with_training_normalization
    from .circuit_capture_service import _load_sae_sync
    from .sae_hook_support import refuse_non_residual

    sae_row = db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
    if sae_row is None:
        raise ValueError(
            f"SAE {sae_id} is gone, so this k-sparse probe cannot be scored. The probe row keeps "
            f"its sae_id (SET NULL only on delete) precisely so this is reportable rather than "
            f"silent"
        )
    refuse_non_residual(getattr(sae_row, "hook_type", None), "a probe monitor")
    sae = _load_sae_sync(sae_row, device)
    columns = torch.as_tensor(list(feature_indices), dtype=torch.long)

    def encode(hidden: Any) -> Any:
        source = hidden.device
        flat = hidden.reshape(-1, hidden.shape[-1])
        with torch.no_grad():
            features = encode_with_training_normalization(sae, flat.to(device) if device else flat)
            selected = features.index_select(1, columns.to(features.device))
        return selected.reshape(hidden.shape[0], hidden.shape[1], -1).to(source)

    return encode


def train_sae_variant(
    db: Any,
    run: Any,
    context: RunContext,
    capture: Any,
    labels: Sequence[int],
    train_index: Sequence[int],
    val_index: Sequence[int],
    *,
    sae_id: str,
    rule: str,
    k: int,
) -> Any:
    """A k-sparse probe over the SAE basis, ranked on TRAIN ONLY.

    The ranking is `select_sae_features`, which scores a standardised class-mean
    difference over train rows. Ranking on validation rows would leak them into the
    model's STRUCTURE — which features exist at all — so the validation AUROC that then
    drives early stopping is no longer held out.
    """
    from .probe_monitor_trainer import head_from_trained, select_sae_features, train_rule

    features = encode_sae_features(db, sae_id, capture)
    train_rows = [features[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    indices = select_sae_features(train_rows, train_labels, k=k)

    sliced_train = [rows[:, indices] for rows in train_rows]
    sliced_val = [features[i][:, indices] for i in val_index]
    val_labels = [labels[i] for i in val_index]

    trained = train_rule(
        rule, sliced_train, train_labels, sliced_val, val_labels, seed=context.seed
    )
    head = head_from_trained(trained, capture.layer)
    probe = _persist_probe(
        db, run, trained, head, capture.layer, context, capture, None,
        variant="sae", sae_id=sae_id, sae_feature_indices=[int(i) for i in indices],
    )
    return probe, head, trained, sliced_val, val_labels


@dataclass(frozen=True)
class CalibrationScores:
    """One probe's negatives from a calibration set under ONE scope."""

    negatives: List[float]
    n_rows: int
    lengths: List[int]


def calibration_scopes(context: RunContext) -> List[str]:
    """The run's own scope first, then every contract window's internal scope, without repeats.

    The run's scope places the global bar and the length bands; each window's scope places that
    window's bar. When the run is scoped `all` the first window IS the base pass, so it is scored
    once and both readers take it — the duplicated pass this replaces.
    """
    from .probe_monitor_render import CONTRACT_WINDOW_SCOPES

    ordered: List[str] = []
    for scope in [context.scope, *CONTRACT_WINDOW_SCOPES.values()]:
        if scope not in ordered:
            ordered.append(scope)
    return ordered


def _prepare_calibration_rows(
    db: Any, dataset_id: str, context: RunContext, tokenizer: Any
) -> Tuple[List[Any], List[int]]:
    """The calibration set's rendered rows and labels, prepared ONCE for every probe and window.

    ⚠ **THE CALIBRATION SET WAS ACCEPTED, VALIDATED, STORED, NAMED IN THE EXPORTED DOCUMENT —
    AND ONCE NEVER USED TO CALIBRATE ANYTHING.** The calibrating stage scored the training set's
    own validation split and hardcoded `source="validation_negatives"`, so a run given a
    calibration set got a threshold from plain training prose while `decision.calibration` named
    the set it had not used (found 2026-09-28 on a probe firing at *"What is the capital of
    France?"*). A false-positive rate is a property of the negative distribution the monitor will
    actually see. This is the function that reads that distribution.

    ⚠ AND IT RAN 36 TIMES PER RUN. Every probe re-read the 1,001,551-row corpus and re-rendered
    the same 2,000 sampled rows once for its base pass and once per window. The sample is seeded
    and the render is deterministic, so once is the same rows.
    """
    from ..models.probe_monitor import ProbeMonitorDataset
    from ..schemas.probe_monitor import DEFAULT_CALIBRATION_MAX_ROWS

    view = (
        db.query(ProbeMonitorDataset)
        .filter(ProbeMonitorDataset.id == dataset_id)
        .first()
    )
    if view is None:
        raise ValueError(
            f"calibration set {dataset_id} is gone; refusing to fall back to validation "
            f"negatives silently, because the threshold would then describe a different "
            f"distribution from the one the run was configured with"
        )
    max_rows = int(context.config.get("calibration_max_rows") or DEFAULT_CALIBRATION_MAX_ROWS)
    _examples, rendered, _summary, labels = _prepare_examples(
        db, view, context, tokenizer, role="calibration", max_rows=max_rows
    )
    return rendered, labels


def _score_calibration_group(
    db: Any,
    rows: Tuple[List[Any], List[int]],
    group: Sequence[Tuple[Any, Any, Any]],
    scopes: Sequence[str],
    model: Any,
    tokenizer: Any,
    architecture: str,
    progress: Optional[Callable[[int, int], None]] = None,
    required_scopes: Sequence[str] = (),
) -> List[Dict[str, CalibrationScores]]:
    """Every probe in `group` (one layer) under every scope, from ONE forward over the rows.

    `group` is `(probe, head, trained)` triples and must share a layer — `forward_scores_by_scope`
    refuses otherwise. Returns one `{scope: CalibrationScores}` per probe, in group order. A scope
    that cannot be scored over these rows is ABSENT from every probe's dict, unless it is in
    `required_scopes`, in which case it raises — see `forward_scores_by_scope`.

    ⚠ BIT-IDENTICAL TO THE PASSES IT REPLACES, by construction rather than by luck: the rows, their
    order and the batch plan are those each old pass built, and a scope only changes the mask
    applied after the forward. `test_calibration_one_forward_per_layer.py` holds it to that against
    separate one-probe, one-scope calls, because bf16 is not batch-invariant and any drift in the
    batch plan would move every threshold.

    Scored through the same scorer evaluation uses, so the calibration aggregates come from the
    code path that produced every reported metric.
    """
    from .probe_monitor_capture import ScoreSpec, forward_scores_by_scope

    rendered, labels = rows
    specs = []
    for probe, head, trained in group:
        encoder = None
        if getattr(probe, "variant", "dense") == "sae":
            encoder = sae_encoder_for(
                db, probe.sae_id, probe.sae_feature_indices, device=_probe_device(model)
            )
        specs.append(
            ScoreSpec(head=head, rule=trained.rule, rule_params=trained.rule_params, encoder=encoder)
        )
    by_scope = forward_scores_by_scope(
        model,
        rendered,
        specs,
        scopes=scopes,
        architecture=architecture,
        pad_id=_pad_id(tokenizer),
        # Batch-level, so a long pass writes to the row throughout rather than only at its end —
        # the gap, not the total, is what the reaper reads.
        progress=progress,
        # Only aggregates and counts are used; 2,000 rows' token traces x scopes x probes would
        # hold hundreds of millions of Python floats.
        keep_token_scores=False,
        required_scopes=required_scopes,
    )
    results: List[Dict[str, CalibrationScores]] = []
    for spec_index in range(len(specs)):
        per_scope: Dict[str, CalibrationScores] = {}
        for scope in scopes:
            if scope not in by_scope:
                continue
            scored = by_scope[scope][spec_index]
            # ⚠ Filtered to label 0. A calibration view cannot map anything to `positive` — the
            # schema refuses it — but it CAN map values to `excluded`, and an excluded row is not
            # a negative.
            #
            # ⚠ THE TOKEN COUNT TRAVELS WITH THE SCORE, so a threshold that varies with length
            # needs no pass of its own.
            per_scope[scope] = CalibrationScores(
                negatives=[row.aggregate for row, label in zip(scored, labels) if label == 0],
                n_rows=len(scored),
                lengths=[row.n_scored for row, label in zip(scored, labels) if label == 0],
            )
        results.append(per_scope)
    return results


def _calibrate_probes(
    db: Any,
    probes: Sequence[Tuple[Any, Any, Any, Any, Any]],
    calibration_id: Optional[str],
    context: RunContext,
    model: Any,
    tokenizer: Any,
    architecture: str,
    beat: Optional[Callable[[int, int], None]] = None,
) -> None:
    """The calibrating stage: place every probe's global, per-length and per-window bars.

    `probes` is the training stage's `(probe, head, trained, val_rows, val_labels)` tuples. Writes
    onto each probe row; the caller commits. Extracted from `execute_probe_run` so the stage's
    DATA FLOW can be tested by running it, rather than by reading its source for names.
    """
    from .probe_monitor_trainer import calibrate

    scores_by_probe: Dict[Any, Dict[str, CalibrationScores]] = {}
    groups: Dict[int, List[Tuple[Any, Any, Any]]] = {}
    if calibration_id:
        for probe, head, trained, _val_rows, _val_labels in probes:
            groups.setdefault(head.layer, []).append((probe, head, trained))

    cal_units = max(1, len(groups))
    cal_state = {"done": 0}
    cal_beat = beat or (lambda done, total: None)

    def cal_progress(done_in_pass: int, total_in_pass: int) -> None:
        """Batch-level within one forward, mapped onto the stage's whole span."""
        fraction = (done_in_pass / total_in_pass) if total_in_pass else 0.0
        cal_beat(int(round((cal_state["done"] + fraction) * 1000)), cal_units * 1000)

    if groups:
        calibration_rows = _prepare_calibration_rows(db, calibration_id, context, tokenizer)
        scopes = calibration_scopes(context)
        for layer, group in groups.items():
            scored = _score_calibration_group(
                db, calibration_rows, group, scopes, model, tokenizer, architecture,
                progress=cal_progress,
                # The run's own scope places the global bar and the bands; it failing fails the
                # stage, as the base pass always did. A window that cannot be scored is absent.
                required_scopes=[context.scope],
            )
            for (probe, _head, _trained), per_scope in zip(group, scored):
                scores_by_probe[id(probe)] = per_scope
            cal_state["done"] = min(cal_state["done"] + 1, cal_units)
            cal_beat(cal_state["done"] * 1000, cal_units * 1000)
    else:
        # The validation-negatives path runs no forward; say so once rather than nothing at all.
        cal_beat(1000, 1000)

    for probe, head, trained, val_rows, val_labels in probes:
        # Filled only on the calibration-set path. Left empty on the validation-negatives
        # path, which is the honest state: those rows were aggregated from the capture memmap
        # and their lengths describe the training corpus, not the traffic a threshold is for.
        cal_lengths: List[int] = []
        probe_scores = scores_by_probe.get(id(probe))

        def _from_calibration_set() -> Tuple[List[float], int]:
            base = probe_scores[context.scope]
            cal_lengths[:] = base.lengths
            return base.negatives, base.n_rows

        negatives_scores, threshold_source = resolve_calibration_negatives(
            calibration_id,
            from_calibration_set=_from_calibration_set,
            from_validation_negatives=lambda: _aggregate_rows(
                head, trained.rule, trained.rule_params,
                [row_ for row_, label in zip(val_rows, val_labels) if label == 0],
            ),
        )
        if not negatives_scores:
            logger.warning(
                "probe %s has no negatives from %s, so no threshold is placed; the "
                "probe is still usable for ranking and the absence is recorded",
                probe.id, threshold_source,
            )
            continue

        # The negatives are kept so the operating point can be re-derived without a GPU.
        probe.calibration_scores_path = persist_calibration_negatives(
            context.artifact_dir, probe.id, negatives_scores
        )
        # ⚠ AND THE LENGTHS, OR `length_bands` IS THE ONE BAR THAT STILL NEEDS A GPU TO MOVE.
        # `length_band_decisions` refuses on a length/score mismatch and chooses its boundaries
        # as quantiles of the lengths actually observed, so they cannot be reconstructed from the
        # scores. Empty on the validation-negatives path, where there are no traffic lengths to
        # record — `None` then says "never attempted", matching `length_bands` itself.
        probe.calibration_lengths_path = (
            persist_calibration_negatives(
                context.artifact_dir, probe.id, cal_lengths, name="lengths"
            )
            if cal_lengths
            else None
        )
        calibration = calibrate(
            negatives_scores,
            target_fpr=context.target_fpr,
            source=threshold_source,
        )
        moved = probe.threshold != calibration.threshold
        probe.threshold = calibration.threshold
        probe.target_fpr = calibration.target_fpr
        probe.realised_fpr = calibration.realised_fpr
        probe.threshold_source = calibration.source
        if moved and getattr(probe, "definition_path", None):
            # The same reasoning as the rung: an exported definition carries this operating
            # point, and a consumer has no way to tell it has moved.
            from .probe_definition_builder import invalidate_definition

            invalidate_definition(db, probe.id, reason="threshold recalibrated")

        # ⚠ A THRESHOLD THAT VARIES WITH LENGTH, from the negatives already scored. Placed
        # BESIDE `probe.threshold`, never replacing it: a consumer that ignores the table
        # behaves exactly as before, which is what makes this additive.
        probe.length_bands = (
            length_band_decisions(
                negatives_scores,
                cal_lengths,
                target_fpr=context.target_fpr,
                global_threshold=calibration.threshold,
            )
            if cal_lengths
            else None
        )

        probe.window_decisions = _calibrate_windows(
            probe, context, threshold_source, probe_scores
        )


def _calibrate_windows(
    probe: Any,
    context: RunContext,
    source: str,
    scores_by_scope: Optional[Dict[str, CalibrationScores]],
) -> Optional[Dict[str, Any]]:
    """A threshold per CONTRACT window, cut from the same corpus under each window's own mask.

    `scores_by_scope` comes from `_score_calibration_group`, which scored every window off the
    probe's layer's ONE forward. This function used to run that forward itself, once per window,
    which with the base pass made four forwards per probe where one per LAYER suffices.

    ⚠ WHY THIS IS NOT ARITHMETIC ON THE NUMBERS WE ALREADY HAVE. `calibration_scores_path` keeps
    the negative AGGREGATES, which is enough to move the operating point to another `target_fpr`
    without a GPU — the quantile of an array already computed. It is not enough to change the
    WINDOW: a different window is a different set of tokens entering the mean, so every row's
    aggregate changes. That needs the model.

    ⚠ AND IT IS ONLY AVAILABLE FROM A CALIBRATION SET. The fallback path aggregates the
    validation split's rows, which were captured under the run's own scope and cannot be
    re-windowed without re-capturing. A run without a calibration set therefore gets no per-window
    thresholds, and `None` says so — rather than a dict that silently describes one window's
    quantile as three.

    Returns `{window: {threshold, target_fpr, realised_fpr, n_negatives, scope}}`, or `None`.
    """
    if scores_by_scope is None or source != "calibration_set":
        return None

    from .probe_monitor_render import CONTRACT_WINDOW_SCOPES

    decisions: Dict[str, Any] = {}
    for window, scope in CONTRACT_WINDOW_SCOPES.items():
        scored = scores_by_scope.get(scope)
        if scored is None:
            # Never silently borrowed from another scope: a window's bar describes the tokens
            # that window reads, and no other window's negatives do.
            logger.warning(
                "probe %s: window %r was not scored, so no threshold is placed for it",
                probe.id, window,
            )
            continue
        negatives, _lengths = scored.negatives, scored.lengths
        if not negatives:
            # A corpus with no assistant turns has nothing in the `response` window. That is a
            # fact about the corpus, not an error, and an absent entry states it.
            logger.info(
                "probe %s: window %r scored no negatives, so no threshold is placed for it",
                probe.id, window,
            )
            continue
        # ⚠ THE PER-ROW LENGTHS, FROM THE ONE PASS THAT *IS* THE BASE PASS.
        #
        # `length_bands` is cut from the lengths of the base pass, which runs at `context.scope`.
        # When a window's internal scope equals `context.scope` its scores ARE that pass's —
        # one array, scored once — so its lengths ARE that array, and keeping them makes a
        # per-length bar re-cuttable from disk.
        #
        # ⚠ THE SCOPE EQUALITY IS A CORRECTNESS GUARD, NOT AN OPTIMISATION. Lengths from another
        # window describe a different token span, so pairing them with the base pass's scores
        # would build a plausible band table whose boundaries were measured over spans nothing
        # was scored on. Persisting from the wrong pass is worse than persisting nothing: nothing
        # refuses, and a wrong table does not.
        #
        # Written HERE as well as in the calibrating stage so the GPU re-cut arm leaves a probe
        # FULLY re-cuttable. Before this it re-scored the three windows, persisted them, and was
        # then refused by its own length table — twenty minutes of GPU spent on a refusal, which
        # is what made the arm's "one payment per probe" claim false for every probe that has one.
        if scope == context.scope and _lengths:
            probe.calibration_lengths_path = persist_calibration_negatives(
                context.artifact_dir, probe.id, _lengths, name="lengths"
            )

        # ⚠ IMPORTED HERE. `calibrate` is imported inside `execute_probe_run`, not at module
        # scope, so a module-level helper that used it bare was a NameError waiting for the first
        # run with a calibration set — caught by `test_no_undefined_names`, which exists because
        # exactly this shape once broke the commit of every J-lens fit.
        from .probe_monitor_trainer import calibrate

        calibration = calibrate(
            negatives, target_fpr=context.target_fpr, source=source
        )
        decisions[window] = {
            "threshold": calibration.threshold,
            "target_fpr": calibration.target_fpr,
            "realised_fpr": calibration.realised_fpr,
            "n_negatives": len(negatives),
            # The INTERNAL scope each window was cut under, recorded so a later reader can tell
            # what `prompt` meant here without re-deriving the mapping.
            "scope": scope,
            # ⚠ THIS WINDOW'S OWN NEGATIVES, so its bar can move to another `target_fpr` without
            # the GPU pass that produced them. The docstring above is right that a different
            # WINDOW needs the model; a different TARGET on an existing window does not, and
            # until this was kept, the two were conflated and both cost a run.
            "scores_path": persist_calibration_negatives(
                context.artifact_dir, probe.id, negatives, name=f"negatives_window_{scope}"
            ),
        }
    return decisions or None


def _evaluate_probes_on_sets(
    db: Any, group: Sequence[Any], context: RunContext,
    model: Any, tokenizer: Any, architecture: str, dataset_ids: Sequence[str],
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Evaluate every probe in `group` — all of which read ONE layer — over `dataset_ids`.

    ⚠ ONE MODEL FORWARD PER (LAYER, SET), NOT PER (PROBE, SET). This took one probe and ran the
    model for each of its sets; the caller then looped probes. With `top_n_layers=1` every probe
    reads the same layer, so every forward after the first recomputed activations that already
    existed. Measured: 1164 s for one probe over five sets, 2388.6 s for two — linear in probes,
    for one layer's work. A four-rule run paid ~80 minutes where ~20 would do.

    The caller groups by layer and `forward_scores_many` refuses a mixed group, because scoring a
    probe against another layer's activations produces plausible numbers in the wrong basis and
    nothing downstream could tell.
    """
    from ..models.probe_monitor import ProbeMonitorDataset, ProbeMonitorEvaluation
    from .probe_monitor_capture import ScoreSpec, forward_scores_many
    from .probe_monitor_metrics import evaluate as evaluate_scores

    if not group:
        return 0

    # ⚠ THE ENCODER IS BUILT ONCE PER PROBE, not per set: loading a dictionary five times is five
    # identical reads of the same file.
    specs: List[ScoreSpec] = []
    for probe, head, trained, _v, _l in group:
        encoder = None
        if getattr(probe, "variant", "dense") == "sae":
            if not probe.sae_id or not probe.sae_feature_indices:
                raise ValueError(
                    f"probe {probe.id} is an SAE variant but carries "
                    f"sae_id={probe.sae_id!r} and {len(probe.sae_feature_indices or [])} feature "
                    f"indices; it cannot be scored in the basis it was trained in"
                )
            encoder = sae_encoder_for(
                db, probe.sae_id, probe.sae_feature_indices, device=_probe_device(model)
            )
        specs.append(
            ScoreSpec(
                head=head, rule=trained.rule,
                rule_params=trained.rule_params, encoder=encoder,
            )
        )

    written = 0
    for dataset_id in dataset_ids:
        view = (
            db.query(ProbeMonitorDataset)
            .filter(ProbeMonitorDataset.id == dataset_id)
            .first()
        )
        if view is None:
            logger.warning("evaluation set %s is gone; skipping", dataset_id)
            continue
        examples, rendered_examples, _summary, labels = _prepare_examples(
            db, view, context, tokenizer, role="eval"
        )
        per_probe = forward_scores_many(
            model,
            rendered_examples,
            specs,
            scope=context.scope,
            architecture=architecture,
            pad_id=_pad_id(tokenizer),
            progress=progress,
        )
        for (probe, _head, _trained, _v, _l), scored in zip(group, per_probe):
            aggregates = [r.aggregate for r in scored]
            lengths = [r.n_scored for r in scored]
            metrics = evaluate_scores(
                aggregates,
                labels,
                name=view.name,
                out_of_distribution=(view.distribution == "out_of_distribution"),
                lengths=lengths,
                target_fpr=probe.target_fpr or context.target_fpr,
            )
            scores_path = context.artifact_dir / f"{probe.id}__{dataset_id}.npy"
            np.save(scores_path, np.asarray(aggregates, dtype=np.float32))
            db.add(
                ProbeMonitorEvaluation(
                    probe_id=probe.id,
                    dataset_id=dataset_id,
                    status="completed" if metrics.get("scored") else "refused",
                    n_positive=metrics.get("n_positive"),
                    n_negative=metrics.get("n_negative"),
                    metrics=metrics,
                    scores_path=str(scores_path),
                )
            )
            written += 1
    db.commit()
    return written


def recompute_rung(db: Any, probe_id: str) -> Dict[str, Any]:
    """Recompute a probe's rung from the evaluations that EXIST (FR-13).

    Called after every evaluation and after every judge run, because a rung is a
    statement about evidence gathered — not a field set once at training time.
    """
    from ..models.probe_monitor import (
        ProbeMonitor,
        ProbeMonitorDataset,
        ProbeMonitorEvaluation,
        ProbeMonitorJudgeRun,
    )
    from .probe_monitor_rung import SetResult, compute

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
    if probe is None:
        raise ValueError(f"probe {probe_id} not found")

    evaluations = (
        db.query(ProbeMonitorEvaluation)
        .filter(ProbeMonitorEvaluation.probe_id == probe_id)
        .all()
    )
    judged_ids = set()
    for judge_run in (
        db.query(ProbeMonitorJudgeRun)
        .filter(
            ProbeMonitorJudgeRun.probe_id == probe_id,
            ProbeMonitorJudgeRun.status == "completed",
        )
        .all()
    ):
        judged_ids.update(judge_run.dataset_ids or [])

    results: List[SetResult] = []
    for evaluation in evaluations:
        view = (
            db.query(ProbeMonitorDataset)
            .filter(ProbeMonitorDataset.id == evaluation.dataset_id)
            .first()
        )
        metrics = evaluation.metrics or {}
        ci = metrics.get("ci") if metrics.get("scored") else None
        ci_low = ci.get("low") if isinstance(ci, dict) else None
        results.append(
            SetResult(
                # The DATASET ID, not the display name: `from_evaluations` refuses
                # duplicate names, and two views can legitimately share one.
                name=evaluation.dataset_id,
                out_of_distribution=bool(
                    view is not None and view.distribution == "out_of_distribution"
                ),
                ci_low=ci_low,
                judge_completed=evaluation.dataset_id in judged_ids,
            )
        )

    outcome = compute(results)
    before = int(probe.rung or 0)
    probe.rung = int(outcome.rung)
    probe.rung_reasons = list(outcome.reasons)
    db.commit()
    # ⚠ A CACHED 033 DEFINITION STATES THIS RUNG, so a change makes the document a lie that still
    # parses and still validates. `recompute_rung` runs after every evaluation AND after every
    # judge run, which is exactly when a stale export would start overstating its evidence.
    # Invalidated rather than rebuilt: a rebuild is a GPU job, and queueing one implicitly inside
    # an evaluation's commit path would do work nobody asked for.
    if int(outcome.rung) != before:
        from .probe_definition_builder import invalidate_definition

        invalidate_definition(
            db, probe_id, reason=f"rung changed {before} -> {int(outcome.rung)}"
        )
    return outcome.as_dict()


def evaluate_probe(db: Any, probe_id: str, dataset_ids: Sequence[str]) -> Dict[str, Any]:
    """Evaluate an existing probe on additional sets, then recompute its rung."""
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorRun

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
    if probe is None:
        raise ValueError(f"probe {probe_id} not found")
    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).first()
    if run is None:
        raise ValueError(f"probe {probe_id} has no run, so it cannot be reproduced")
    # An evaluation WRITES evidence the rung is computed from; at another precision it would be
    # another probe's evidence (review round 1, H1).
    require_matching_precision(db, run)
    context = context_from_row(run)
    model, tokenizer, architecture = _load_model_for_run(run)
    head, trained = load_probe(db, probe_id)
    # A group of one — the same path the run uses, so a re-evaluation and the original cannot
    # disagree about how a probe is scored.
    written = _evaluate_probes_on_sets(
        db, [(probe, head, trained, None, None)], context, model, tokenizer,
        architecture, dataset_ids,
    )
    rung = recompute_rung(db, probe_id)
    return {"probe_id": probe_id, "evaluations": written, "rung": rung}


def unrecuttable_kept_windows(
    previous: Optional[Dict[str, Any]], rescored: Optional[Dict[str, Any]]
) -> List[str]:
    """Windows a re-score could not reach that also have no stored negatives to move them with."""
    return sorted(
        window for window, entry in (previous or {}).items()
        if window not in (rescored or {}) and not (entry or {}).get("scores_path")
    )


def merge_rescored_windows(
    probe_id: str,
    previous: Optional[Dict[str, Any]],
    rescored: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """The window table after a GPU re-score: re-scored windows replace, the rest are KEPT.

    ⚠ A WINDOW THAT COULD NOT BE RE-SCORED KEEPS ITS EXISTING BAR (review round 2, M2). Replacing
    the table wholesale would delete it with only a log line — the harm the arm's own refusal
    exists to prevent, for one window instead of all of them. Its stored negatives (`scores_path`)
    still let `propose` move it to the new target. Nothing re-scored returns None, which the arm
    refuses rather than writing.
    """
    if not rescored:
        return None
    kept = {window: entry for window, entry in (previous or {}).items() if window not in rescored}
    if kept:
        logger.warning(
            "probe %s: windows %s could not be re-scored and keep their existing bars",
            probe_id, sorted(kept),
        )
    return {**kept, **rescored}


def recut_probe_windows_on_gpu(
    db: Any, probe_id: str, *, target_fpr: float, reason: str = ""
) -> Dict[str, Any]:
    """Re-score this probe's per-window negatives, then move every bar to `target_fpr`.

    ⚠ THE ONLY PART OF A RECALIBRATION THAT NEEDS THE MODEL, AND ONLY FOR PROBES CALIBRATED
    BEFORE THEIR WINDOW ARRAYS WERE KEPT. A different WINDOW is a different set of tokens
    entering every row's aggregate, so there is nothing on disk to re-quantile — that is the
    distinction `_calibrate_windows`' own docstring draws, and until the arrays were persisted
    it also swallowed the case that needs no GPU at all.

    ⚠ "ONE PAYMENT PER PROBE" WAS HALF TRUE WHEN FIRST WRITTEN, AND THAT HALF WAS THE EXPENSIVE
    ONE. The arm persisted each window's negatives and NOT the per-row lengths, so on a probe
    that also has a length table — which both production probes do — it re-scored three windows,
    saved them, and was then refused by `propose` over the lengths it had not kept. Twenty
    minutes of GPU for a refusal. `_calibrate_windows` now keeps the lengths from the pass whose
    scope equals the run's, and that pass IS the base pass, so after this runs once the same
    probe re-cuts in microseconds forever — global, per-window and per-length alike.

    Refuses, naming which, rather than returning an empty result: `window_decisions = None`
    already means "never attempted", and a silent `None` here would overwrite a real table with
    that claim.
    """
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorRun
    from .probe_recalibration import RecalibrationRefused, apply, propose

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
    if probe is None:
        raise RecalibrationRefused("probe_not_found", f"probe {probe_id} not found", status=404)
    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).first()
    if run is None:
        raise RecalibrationRefused(
            "run_gone",
            f"probe {probe_id}'s run row is gone, so its capture cannot be reproduced and its "
            f"per-window bars cannot be re-scored; the global bar can still be re-cut from the "
            f"stored negatives",
        )
    context = context_from_row(run)
    calibration_id = getattr(run, "calibration_dataset_id", None) or probe.calibration_dataset_id
    if not calibration_id:
        raise RecalibrationRefused(
            "no_calibration_view",
            f"probe {probe_id} has no calibration view, so there are no per-window negatives to "
            f"score; a run without one never had per-window bars",
        )
    # ⚠ REFUSE BEFORE THE GPU, NOT AFTER IT. The window loop persists the per-row lengths only
    # from the pass whose scope equals the run's, because only that pass is the base pass. A run
    # scoped `assistant` or `user` is matched by NO contract window (`all`, `input`,
    # `last_assistant`), so its lengths would never be written and `propose` would refuse this
    # probe's length table — after twenty minutes of scoring. A precondition that is checked
    # after the cost has been paid is not a precondition.
    if probe.length_bands and not probe.calibration_lengths_path:
        from .probe_monitor_render import CONTRACT_WINDOW_SCOPES

        if context.scope not in set(CONTRACT_WINDOW_SCOPES.values()):
            raise RecalibrationRefused(
                "lengths_not_recoverable",
                f"probe {probe_id} has a per-length table whose lengths were never persisted, "
                f"and its run was scoped {context.scope!r} — which no contract window "
                f"({', '.join(sorted(set(CONTRACT_WINDOW_SCOPES.values())))}) reproduces, so "
                f"re-scoring the windows would not recover them. Its global and per-window bars "
                f"can still be re-cut; the per-length table needs a new run",
            )

    # Re-cut bars are cut from negatives scored NOW; at another precision they would describe a
    # different distribution from the head they serve, and every later zero-GPU re-cut would read
    # them (review round 1, H1).
    precision = probe_precision(db, run)
    if precision["refusal"] is not None:
        raise RecalibrationRefused("precision_mismatch", precision["refusal"])
    model, tokenizer, architecture = _load_model_for_run(run)
    head, trained = load_probe(db, probe_id)
    from .probe_monitor_render import CONTRACT_WINDOW_SCOPES

    # Every window off ONE forward — this arm used to run one per window.
    [scores] = _score_calibration_group(
        db,
        _prepare_calibration_rows(db, calibration_id, context, tokenizer),
        [(probe, head, trained)],
        list(dict.fromkeys(CONTRACT_WINDOW_SCOPES.values())),
        model, tokenizer, architecture,
    )
    decisions = _calibrate_windows(
        probe, context, probe.threshold_source or "calibration_set", scores
    )
    unrecuttable = unrecuttable_kept_windows(probe.window_decisions, decisions)
    if decisions and unrecuttable:
        # ⚠ REFUSED BEFORE ANYTHING IS WRITTEN (review round 3, M-A). A kept window with no stored
        # negatives cannot be moved by `propose`, which would then refuse AFTER the re-scored
        # windows were committed at the run's old target — half a re-cut, no revision, and a
        # message telling the operator to run this same GPU job again, which would fail the same
        # way because an empty window is a fact about the data.
        raise RecalibrationRefused(
            "window_not_rescorable",
            f"probe {probe_id}: window(s) {', '.join(sorted(unrecuttable))} could not be re-scored "
            f"(no row has tokens in them) and have no stored negatives to re-cut from, so their "
            f"bars cannot move to the new target. Nothing was written. Their existing bars stand; "
            f"a new run is needed to re-cut them.",
        )
    decisions = merge_rescored_windows(probe_id, probe.window_decisions, decisions)
    if not decisions:
        raise RecalibrationRefused(
            "windows_scored_nothing",
            f"re-scoring probe {probe_id}'s windows produced no thresholds; its existing table "
            f"is left in place rather than replaced with a claim that none was attempted",
        )
    probe.window_decisions = decisions
    db.commit()

    # Now every array this probe needs is on disk, so the move itself is the cheap path.
    evaluations = (
        db.query(_evaluation_model()).filter(_evaluation_model().probe_id == probe_id).all()
    )
    proposal = propose(probe, evaluations, target_fpr=target_fpr)
    return apply(db, probe, proposal, reason=reason or "windows re-scored and bars re-cut")


def _evaluation_model() -> Any:
    """`ProbeMonitorEvaluation`, imported lazily to match this module's import discipline."""
    from ..models.probe_monitor import ProbeMonitorEvaluation

    return ProbeMonitorEvaluation


def load_probe(db: Any, probe_id: str) -> Tuple[Any, Any]:
    """Rebuild a `ProbeHead` and a rule descriptor from the stored weights.

    One loader, so a training path and a serving path cannot disagree about what a
    probe is. 033 reads the same file to build an exported definition.
    """
    from safetensors.torch import load_file

    from ..models.probe_monitor import ProbeMonitor
    from ..ml.probe_monitor_model import ProbeHead
    from .probe_monitor_trainer import TrainedRule

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
    if probe is None:
        raise ValueError(f"probe {probe_id} not found")
    if not probe.weights_path:
        raise ValueError(
            f"probe {probe_id} has no weights on disk; its run did not finish training"
        )
    path = settings.resolve_data_path(probe.weights_path)
    tensors = load_file(str(path))
    # The bias travels in the file's metadata, so a head is never assembled from two
    # sources that can disagree.
    import json

    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    bias = float(metadata.get("bias", "0.0"))
    head = ProbeHead(
        weight=tensors["weight"],
        bias=bias,
        mean=tensors.get("mean"),
        std=tensors.get("std"),
        attention_query=tensors.get("attention_query"),
        layer=probe.layer,
    )
    trained = TrainedRule(
        rule=probe.rule,
        weight=tensors["weight"],
        bias=bias,
        mean=tensors.get("mean"),
        std=tensors.get("std"),
        attention_query=tensors.get("attention_query"),
        rule_params=dict(probe.rule_params or {}),
        val_auroc=(probe.val_metrics or {}).get("val_auroc"),
        epochs_run=0,
        best_epoch=0,
    )
    return head, trained


def _display_tokens(tokenizer: Any, tokens: Sequence[str], mask: Sequence[bool]) -> List[Dict[str, Any]]:
    """Each token as the vocabulary spells it AND as a reader should see it.

    ⚠ `convert_ids_to_tokens` RETURNS THE VOCABULARY STRING, NOT TEXT. On a byte-level BPE
    tokenizer that means a space is `Ġ` and a newline is `Ċ`, so a scored trace rendered straight
    from it reads

        <|im_start|>ĠuserĊRetĠireĠonline

    which is not what the model was given — it is how the vocabulary encodes what the model was
    given. This estate has paid for that distinction before: `filter_stop_words` was inert for
    months because it stripped SentencePiece's `▁` and never byte-level `Ġ`.

    ⚠ THE DECODE IS THE TOKENIZER'S JOB, NOT THE FRONTEND'S. A table in the UI would have to know
    which family this model uses — byte-level BPE (`Ġ`) or SentencePiece (`▁`) — and this estate
    runs both. The tokenizer already knows, and it is right here.

    ⚠ AND SPECIAL TOKENS KEEP THEIR LITERAL SPELLING, deliberately. `<|im_start|>` decodes to
    nothing useful, and a probe that fires on chat-template scaffolding rather than on content is
    a FINDING — hiding those tokens would hide it. They are marked `special` so the UI can render
    them as scaffolding rather than as text.

    ⚠ `all_special_tokens` IS NOT THE LIST, AND MEASURING SAID SO. On this estate's LFM2.5-1.2B it
    holds three entries — `<|startoftext|>`, `<|im_end|>`, `<|pad|>` — and **omits `<|im_start|>`**,
    which is as much chat scaffolding as the one beside it. `get_added_vocab()` has 509 entries and
    contains both; on Llama-3.1-8B it has 256 against `all_special_tokens`' two. The union is used
    because neither list is guaranteed a superset of the other, and a template token rendered as
    ordinary text is the failure this flag exists to prevent.
    """
    specials = set(getattr(tokenizer, "all_special_tokens", None) or [])
    try:
        specials |= set(tokenizer.get_added_vocab() or {})
    except Exception:  # noqa: BLE001 - a tokenizer without added vocab keeps the smaller list
        pass
    out: List[Dict[str, Any]] = []
    for token, keep in zip(tokens, mask):
        special = token in specials
        if special:
            text = token
        else:
            try:
                text = tokenizer.convert_tokens_to_string([token])
            except Exception:  # noqa: BLE001 - an undecodable token shows its raw spelling
                text = token
        out.append(
            {"token": token, "text": text, "special": special, "scored": bool(keep)}
        )
    return out


def score_verdict(
    global_threshold: Optional[float],
    length_bands: Optional[Sequence[Dict[str, Any]]],
    aggregate: float,
    n_scored: int,
) -> Dict[str, Any]:
    """The verdict fields for one scored input: which bar applied, and whether it fires.

    Pure, so the decision can be tested without a model and the call asserted by walking the AST
    — this repo's recorded remedy after a guard left inline was defeated by `if False:`.

    `n_scored` keys the band because it is the quantity the bands were CUT on: the calibration
    stage records each row's scored-token count and `length_band_decisions` places boundaries on
    exactly that. Keying on the raw sequence length would put chat-template tokens into the count
    and shift an input across a boundary for reasons unrelated to its content.
    """
    from .probe_monitor_metrics import band_for_length, threshold_for_length

    band = band_for_length(length_bands, n_scored)
    threshold = threshold_for_length(length_bands, n_scored, global_threshold)
    return {
        "threshold": threshold,
        # The headline bar beside the applied one, so a reader can see a band moved it.
        "global_threshold": global_threshold,
        # WHICH band applied, or None when the probe has no table. `threshold_source` is
        # "global" for a band too thin to cut, which inherits the global bar.
        "threshold_band": (
            None if band is None else {
                "min_tokens": band.get("min_tokens"),
                "max_tokens": band.get("max_tokens"),
                "threshold_source": band.get("threshold_source"),
            }
        ),
        # Above the threshold, or None when no threshold was placed. NOT False — a probe with
        # no calibration has not said "no", it has said nothing.
        "fires": None if threshold is None else aggregate >= threshold,
    }


def score_one(
    db: Any,
    probe_id: str,
    *,
    text: Optional[str] = None,
    messages: Optional[Sequence[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Score one input and return the per-token trace (FR-12)."""
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorRun
    from .probe_monitor_capture import forward_scores
    from .probe_monitor_render import render_messages

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
    if probe is None:
        raise ValueError(f"probe {probe_id} not found")
    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).first()
    if run is None:
        raise ValueError(f"probe {probe_id} has no run")
    # NOT refused: an offline score writes nothing, and "try it on your own text" stays useful
    # for an older probe. But the result SAYS which precision it was scored at beside the one the
    # probe was fitted at, because the two differ for every probe trained before 2026-10-03.
    precision = probe_precision(db, run)
    context = context_from_row(run)
    model, tokenizer, architecture = _load_model_for_run(run)
    head, trained = load_probe(db, probe_id)

    conversation = (
        list(messages) if messages is not None else [{"role": "user", "content": text or ""}]
    )
    rendered = render_messages(tokenizer, conversation, max_length=context.max_length)
    # The same basis rule as evaluation: an SAE probe scores through its dictionary.
    encoder = None
    if getattr(probe, "variant", "dense") == "sae" and probe.sae_id and probe.sae_feature_indices:
        encoder = sae_encoder_for(
            db, probe.sae_id, probe.sae_feature_indices, device=_probe_device(model)
        )
    scored = forward_scores(
        model, [rendered], head, encoder=encoder,
        rule=trained.rule, rule_params=trained.rule_params,
        scope=context.scope, architecture=architecture, pad_id=_pad_id(tokenizer),
    )[0]
    mask = rendered.scored_mask(
        context.scope if rendered.role_mask_reliable else "all"
    )
    tokens = tokenizer.convert_ids_to_tokens(rendered.input_ids)

    # ⚠ THE BAR THIS LENGTH IS ACTUALLY JUDGED AGAINST, NOT THE PROBE'S HEADLINE NUMBER.
    #
    # This returned `probe.threshold` — the single global bar — for every input. But the
    # contract miStudio publishes opens "ONE CONSTANT THRESHOLD IS MISCALIBRATED AT EVERY LENGTH
    # BUT THE ONE IT WAS CUT AT", and miLLM's runtime applies the band for the request's length.
    # So this panel and the live monitor disagreed about which bar applied, in BOTH directions:
    # on `pm_f463a8a235ae` band 0-203 is 12.806 and band 204-339 is 26.602 against a global
    # 17.980, so the panel was too strict on short inputs and too lenient on medium ones. Found
    # 2026-10-03 from a pasted screenshot that showed "threshold 17.9802" for a ~50-token input.
    #
    # `n_scored` is the quantity the bands were CUT on (the calibration stage keys them on each
    # row's scored-token count), so it is the right key here too.
    return {
        "probe_id": probe_id,
        "aggregate": scored.aggregate,
        **score_verdict(probe.threshold, probe.length_bands, scored.aggregate, scored.n_scored),
        "tokens": _display_tokens(tokenizer, tokens, mask),
        "token_scores": scored.token_scores,
        "n_scored": scored.n_scored,
        "role_mask_reliable": rendered.role_mask_reliable,
        "truncated": rendered.truncated,
        # The probe's precision beside the one this score used. `fitted_at` None = not recorded
        # (every probe trained before 2026-10-03, which was float16).
        "precision": {
            "fitted_at": precision["recorded"],
            "scored_at": (load_dtype_record(model) or {}).get("model_dtype") or precision["loads_at"],
            "note": precision["refusal"],
        },
    }
