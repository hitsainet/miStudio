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

import numpy as np

from ..core.cancellation import record_progress
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
    top_n_layers: int
    val_fraction: float


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
        top_n_layers=int(config.get("top_n_layers", 1)),
        val_fraction=float(config.get("val_fraction", 0.15)),
    )


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

    def heartbeat(name: str, span: Tuple[float, float]) -> Callable[[int, int], None]:
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
                on_stage(name, low + (high - low) * fraction, f"{done}/{total} batches")

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
    for layer in selection.chosen:
        captures[layer] = capture_tokens(
            model,
            rendered_examples,
            layer,
            context.artifact_dir / f"tokens_layer{layer}.f16",
            scope=context.scope,
            architecture=architecture,
            pad_id=_pad_id(tokenizer),
            progress=heartbeat("token_capture", heartbeat_band("token_capture")),
        )
    finish("token_capture")

    # ── S5 train each rule ────────────────────────────────────────────────────
    stage("training", f"{len(context.rules)} rules")
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
        for rule in context.rules:
            check_cancelled.raise_if_cancelled("between rules")
            trained = train_rule(
                rule, train_rows, train_labels, val_rows, val_labels, seed=context.seed
            )
            head = head_from_trained(trained, layer)
            probe = _persist_probe(
                db, row, trained, head, layer, context, capture, selection
            )
            probes.append((probe, head, trained, val_rows, val_labels))

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

    for probe, head, trained, val_rows, val_labels in probes:
        negatives_scores = _aggregate_rows(
            head, trained.rule, trained.rule_params,
            [row_ for row_, label in zip(val_rows, val_labels) if label == 0],
        )
        if not negatives_scores:
            logger.warning(
                "probe %s has no validation negatives, so no threshold is placed; the "
                "probe is still usable for ranking and the absence is recorded",
                probe.id,
            )
            continue
        calibration = calibrate(
            negatives_scores,
            target_fpr=context.target_fpr,
            source="validation_negatives",
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
    for index, (probe, head, trained, _v, _l) in enumerate(probes):
        evaluated += _evaluate_probe_on_sets(
            db, probe, head, trained, context, model, tokenizer, architecture,
            list(row.eval_dataset_ids or []),
            progress=heartbeat("evaluating", sub_band("evaluating", index, len(probes))),
        )
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

    best = max(
        (p for p, *_ in probes),
        key=lambda p: (p.val_metrics or {}).get("val_auroc") or 0.0,
        default=None,
    )
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


def _prepare_examples(
    db: Any, view: Any, context: RunContext, tokenizer: Any, *, role: str
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


def _evaluate_probe_on_sets(
    db: Any, probe: Any, head: Any, trained: Any, context: RunContext,
    model: Any, tokenizer: Any, architecture: str, dataset_ids: Sequence[str],
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    from ..models.probe_monitor import ProbeMonitorDataset, ProbeMonitorEvaluation
    from .probe_monitor_capture import forward_scores
    from .probe_monitor_metrics import evaluate as evaluate_scores

    # ⚠ A k-SPARSE PROBE IS SCORED THROUGH ITS SAE. Built ONCE per probe rather than per set:
    # loading the dictionary five times would be five identical reads of the same file.
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
        scored = forward_scores(
            model,
            rendered_examples,
            head,
            rule=trained.rule,
            rule_params=trained.rule_params,
            scope=context.scope,
            architecture=architecture,
            pad_id=_pad_id(tokenizer),
            progress=progress,
            encoder=encoder,
        )
        aggregates = [row.aggregate for row in scored]
        lengths = [row.n_scored for row in scored]
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
        evaluation = ProbeMonitorEvaluation(
            probe_id=probe.id,
            dataset_id=dataset_id,
            status="completed" if metrics.get("scored") else "refused",
            n_positive=metrics.get("n_positive"),
            n_negative=metrics.get("n_negative"),
            metrics=metrics,
            scores_path=str(scores_path),
        )
        db.add(evaluation)
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
    context = context_from_row(run)
    model, tokenizer, architecture = _load_model_for_run(run)
    head, trained = load_probe(db, probe_id)
    written = _evaluate_probe_on_sets(
        db, probe, head, trained, context, model, tokenizer, architecture, dataset_ids
    )
    rung = recompute_rung(db, probe_id)
    return {"probe_id": probe_id, "evaluations": written, "rung": rung}


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
    return {
        "probe_id": probe_id,
        "aggregate": scored.aggregate,
        "threshold": probe.threshold,
        # Above the threshold, or None when no threshold was placed. NOT False — a
        # probe with no calibration has not said "no", it has said nothing.
        "fires": (
            None if probe.threshold is None else scored.aggregate >= probe.threshold
        ),
        "tokens": [
            {"token": token, "scored": bool(keep)}
            for token, keep in zip(tokens, mask)
        ],
        "token_scores": scored.token_scores,
        "n_scored": scored.n_scored,
        "role_mask_reliable": rendered.role_mask_reliable,
        "truncated": rendered.truncated,
    }
