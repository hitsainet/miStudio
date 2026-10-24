"""The post-run evaluation of a training's SAEs, on blocks the training never read.

SAE TRAINING REMEDIATION, ITEM 6 (2026-09-15). Evaluating train_6247e768 by hand
on held-out text showed healthy sparsity and a large cost to the model — +0.3
nats of cross-entropy per spliced layer, +1.45 with all three — and none of the
run's own numbers could have shown it:

* the spliced-CE step was SKIPPED on every cached-activation run ("no tokenized
  dataset in hand"), which is every run that matters;
* when it did run, its ablation baseline substituted ``b_dec`` — a vector in the
  SAE's NORMALISED space — as a raw activation;
* and it stored its numbers by overloading ``training_metrics`` columns.

This module is the replacement. After the community export, while the job still
holds its GPU lease, it loads the base model with the training's placement,
reads blocks each extraction never read, and records base / spliced /
mean-ablated / zero-ablated CE, loss recovered, KL, L0, centred FVU and the
all-layers-spliced CE in ``trainings.evaluation``. The same entry point runs
from ``POST /trainings/{id}/evaluate`` for a completed training.

WHICH BLOCKS ARE UNSEEN. Extraction reads the first ``max_samples`` rows of its
tokenization in stored order (``activation_service._load_dataset``), and the
metadata it writes records both. Rows at or above that bound were never
extracted, so no SAE trained on the extraction could have seen them — and
:func:`select_unseen_rows` can only draw from there. A training that extracts on
the fly reads rows from the whole tokenization, so no row is unseen unless the
caller names the rows it held out (``EvalSource.candidate_rows``); without them
the evaluation is SKIPPED, with the reason, rather than run on training data.

A FAILURE NEVER FAILS THE TRAINING. Everything here runs after the SAEs are
saved; :func:`run_evaluation` catches every exception, records
``status = "failed"`` with the reason, and returns.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from . import dataset_mixture
from .sae_evaluation import evaluate_spliced_layers

logger = logging.getLogger(__name__)

#: Bumped when the shape of the stored document changes.
EVALUATION_VERSION = 1

#: Default real-token budget, split across sources by the mixture. At 2,048-token
#: blocks that is 64 blocks — enough for CE to two decimals on these models.
DEFAULT_TOKEN_BUDGET = 131_072

#: Tokens per forward pass. Base log-probabilities are held for the batch while
#: every substitution runs (vocab x tokens x 4 bytes: 0.5 GB for LFM2.5 at 2,048).
DEFAULT_BATCH_TOKENS = 4_096

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
#: Stopped by the operator (or a lost GPU lease) between batches; the partial numbers are not recorded.
STATUS_CANCELLED = "cancelled"
ACTIVE_STATUSES = (STATUS_PENDING, STATUS_RUNNING)

#: An operator's Stop of a pending or running evaluation, written into its document
#: (review R3-A, R2D-1). A training is COMPLETED once its full-length export is saved,
#: so its row status can no longer carry a Stop to the evaluation that follows; the
#: Stop endpoint writes this key instead, and the evaluation reads it between batches.
STOP_REQUESTED_AT = "stop_requested_at"
STOP_REQUESTED_BY = "stop_requested_by"


class EvaluationStopped(Exception):
    """Raised between batches when the job has been told to stop (review R1-D R1D-7).

    An ``Exception``, not ``BaseException``: it is the evaluation's own control flow,
    caught in :func:`run_evaluation` and recorded as ``cancelled``, never allowed to
    reach the training task as a failure.
    """

#: Seconds between the record writes a running evaluation makes as it goes. The
#: janitor judges a running evaluation by the age of its last write, so a long
#: one must keep writing (the lesson of `long-phases-need-a-db-heartbeat`).
HEARTBEAT_SECONDS = 60

#: A RUNNING evaluation whose record is older than this — ten heartbeats — and
#: whose task no longer looks alive belongs to a worker that is gone.
ABANDONED_AFTER_SECONDS = 600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)


# ── the janitor ─────────────────────────────────────────────────────────────


def evaluation_looks_abandoned(
    document: Any,
    *,
    task_alive: Callable[[Optional[str], datetime], bool],
    now: Optional[datetime] = None,
) -> bool:
    """Whether a RUNNING evaluation belongs to a worker that is gone.

    WITHOUT THIS, A KILLED EVALUATION SAID "running" FOREVER (review R1-C,
    2026-09-15). Nothing reset it: the panel disabled its button and polled every
    ten seconds indefinitely, the endpoint refused a new run with 409, and the only
    way out was an API call with ``force=true`` that the UI never sends — the
    `nlp_status` shape exactly. A pod roll, an OOM kill or a lost lease all end
    that way, since an exception that is not an ``Exception`` never reaches
    :func:`run_evaluation`'s recording handler.

    ONLY RUNNING. A pending evaluation may be queued behind a days-long training on
    a single-GPU queue and has no clock of its own to be judged by; it is left to
    the operator, whom the panel offers a forced re-run once the request is old.

    Args:
        document: ``trainings.evaluation``.
        task_alive: ``(task_id, last_write) -> bool``; consulted only once the
            record is stale, so a live task is never condemned for quiet.
    """
    if not isinstance(document, Mapping) or document.get("status") != STATUS_RUNNING:
        return False
    stamp = _parse_time(document.get("updated_at")) or _parse_time(document.get("started_at"))
    if stamp is None:
        return False
    age = ((now or datetime.now(timezone.utc)) - stamp).total_seconds()
    if age <= ABANDONED_AFTER_SECONDS:
        return False
    return not task_alive(document.get("task_id"), stamp)


def _task_alive(task_id: Optional[str], last_write: datetime) -> bool:
    from types import SimpleNamespace

    from ..workers.task_heartbeat import task_looks_alive

    return task_looks_alive(task_id, SimpleNamespace(updated_at=last_write), started=True)


def reap_abandoned_evaluations(
    db: Any,
    *,
    now: Optional[datetime] = None,
    task_alive: Optional[Callable[[Optional[str], datetime], bool]] = None,
) -> List[str]:
    """Mark every abandoned running evaluation failed, with the reason. Returns the training ids.

    Called by the ``cleanup_stuck_trainings`` janitor. The training's own status is
    never touched: an evaluation's failure never fails its training.
    """
    from ..models.training import Training

    alive = task_alive or _task_alive
    moment = now or datetime.now(timezone.utc)
    rows = db.query(Training).filter(Training.evaluation["status"].astext == STATUS_RUNNING).all()
    reaped: List[str] = []
    for row in rows:
        document = dict(getattr(row, "evaluation", None) or {})
        if not evaluation_looks_abandoned(document, task_alive=alive, now=moment):
            continue
        last = document.get("updated_at") or document.get("started_at")
        document.update(
            status=STATUS_FAILED,
            reason=(
                f"the evaluation stopped reporting at {last} and its task is no longer running "
                "(the worker was restarted or killed); run it again"
            ),
            completed_at=moment.isoformat(),
            updated_at=moment.isoformat(),
        )
        row.evaluation = document
        db.commit()
        reaped.append(str(row.id))
        _emit(str(row.id), document)
    return reaped


# ── sources ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalSource:
    """One tokenization the training read from, and which of its rows it read.

    Exactly one of ``rows_read`` and ``candidate_rows`` says what is unseen:

    * ``rows_read`` is the extraction's ``max_samples`` as recorded (0 = every
      row) and ``rows_processed`` its ``num_samples_processed``; the leading rows
      they cover (``rows_read_by_extraction``) were read, and the rest were not;
    * ``candidate_rows`` lists the rows known never to have been read.

    With neither, nothing counts as unseen.
    """

    label: str
    dataset_path: str
    rows_read: Optional[int] = None
    rows_processed: int = 0
    candidate_rows: Optional[Tuple[int, ...]] = None
    #: The source's share of the training mixture, before normalisation.
    weight: float = 1.0


def rows_read_by_extraction(metadata: Mapping[str, Any], total_rows: int) -> int:
    """How many leading rows of its tokenization an extraction read.

    ``_load_dataset`` keeps ``range(max_samples)`` when ``max_samples > 0`` and the
    tokenization is longer, and everything otherwise. ``num_samples_processed`` is
    taken too, as a floor: if the two ever disagree, the larger bound is the safe
    one — it can only shrink the unseen pool, never let a read row into it.
    """
    max_samples = int(metadata.get("max_samples") or 0)
    read = total_rows if max_samples <= 0 else min(max_samples, total_rows)
    processed = int(metadata.get("num_samples_processed") or 0)
    return max(read, min(processed, total_rows))


def sources_from_extractions(
    extractions: Sequence[Any],
    resolve_path: Callable[[str], Path],
    dataset_weights: Optional[Sequence[float]] = None,
) -> List[EvalSource]:
    """One source per extraction, from the ``metadata.json`` the extraction wrote.

    ``extractions`` are rows with ``id`` and ``output_path``. Without explicit
    ``dataset_weights`` each source is weighted by the rows its extraction read —
    the share of the corpus the SAE trained on, which is what "proportional to
    the mixture" means for a run with no stated mixture.
    """
    sources = []
    weights = list(dataset_weights) if dataset_weights is not None else None
    if weights is not None and len(weights) != len(extractions):
        raise ValueError(
            f"dataset_weights has {len(weights)} entries for {len(extractions)} extractions"
        )
    for index, extraction in enumerate(extractions):
        metadata_path = Path(resolve_path(extraction.output_path)) / "metadata.json"
        with open(metadata_path) as handle:
            metadata = json.load(handle)
        dataset_path = metadata.get("dataset_path")
        if not dataset_path:
            raise ValueError(f"{metadata_path} records no dataset_path")
        max_samples = int(metadata.get("max_samples") or 0)
        processed = int(metadata.get("num_samples_processed") or 0)
        sources.append(EvalSource(
            label=str(extraction.id),
            dataset_path=str(resolve_path(dataset_path)),
            # As recorded; resolved against the tokenization's length when it is
            # opened (a max_samples of 0 means the extraction read every row).
            rows_read=max_samples,
            rows_processed=processed,
            weight=float(weights[index]) if weights is not None else float(max(processed, max_samples, 1)),
        ))
    return sources


@dataclass
class OpenedSource:
    source: EvalSource
    total_rows: int
    unseen: int
    rows: List[int] = field(default_factory=list)
    dataset: Any = None


def first_unseen_row(source: EvalSource, total_rows: int) -> Optional[int]:
    """The first row a bounded source's training never read; None for a candidate-row or unknown source."""
    if source.candidate_rows is not None or source.rows_read is None:
        return None
    return rows_read_by_extraction(
        {"max_samples": source.rows_read, "num_samples_processed": source.rows_processed}, total_rows
    )


def _unseen_count(source: EvalSource, total_rows: int) -> int:
    if source.candidate_rows is not None:
        return len(source.candidate_rows)
    first = first_unseen_row(source, total_rows)
    return 0 if first is None else max(0, total_rows - first)


def select_unseen_rows(
    sources: Sequence[EvalSource],
    total_rows: Sequence[int],
    seq_lens: Sequence[int],
    token_budget: int,
    seed: int,
) -> List[List[int]]:
    """For each source, the sorted rows to evaluate — every one a row the training never read.

    The token budget is split by ``dataset_mixture.allocate_tokens`` over each
    source's UNSEEN tokens with the sources' mixture weights, so a source that
    cannot meet its share gives what it has and the rest is redistributed. Rows
    within a source are a seeded sample, so the evaluation is reproducible.
    """
    if not (len(sources) == len(total_rows) == len(seq_lens)):
        raise ValueError("sources, total_rows and seq_lens must correspond")
    unseen = [_unseen_count(s, n) for s, n in zip(sources, total_rows)]
    unseen_tokens = [u * max(1, int(w)) for u, w in zip(unseen, seq_lens)]
    weights = [max(0.0, float(s.weight)) for s in sources]
    if sum(w for w, u in zip(weights, unseen) if u > 0) <= 0:
        weights = None
    alloc = dataset_mixture.allocate_tokens(unseen_tokens, max(0, int(token_budget)), weights)

    chosen: List[List[int]] = []
    for index, (source, n_total, n_unseen, tokens, width) in enumerate(
        zip(sources, total_rows, unseen, alloc, seq_lens)
    ):
        k = min(n_unseen, math.ceil(tokens / max(1, int(width)))) if tokens > 0 else 0
        if k <= 0:
            chosen.append([])
            continue
        rng = np.random.default_rng([int(seed), index])
        picks = rng.choice(n_unseen, size=k, replace=False)
        if source.candidate_rows is not None:
            rows = [int(source.candidate_rows[i]) for i in picks]
        else:
            first_unseen = first_unseen_row(source, int(n_total))
            rows = [first_unseen + int(i) for i in picks]
        chosen.append(sorted(rows))
    return chosen


def merge_shared_tokenizations(sources: Sequence[EvalSource]) -> List[EvalSource]:
    """One source per tokenization: a row any of the training's extractions read is read.

    TWO EXTRACTIONS OF ONE TOKENIZATION (review R1-C, 2026-09-15). Every extraction
    reads the LEADING rows of its tokenization, and nothing stops a training from
    reading two extractions of the same one — a 10,000-row and a 50,000-row
    extraction of one corpus. Judged one source at a time, rows 10,000-49,999 are
    unseen by the first although the SAE trained on them through the second, and
    the evaluation drew them as blocks "the training never read". Merged, the bound
    is the larger one, and an extraction that read every row (0) wins outright.

    Sources that name their candidate rows (the on-the-fly path) keep only the rows
    every one of them names. A tokenization read both ways cannot be decided and
    raises, which the caller records as a failure.
    """
    import os

    groups: Dict[str, List[EvalSource]] = {}
    for source in sources:
        groups.setdefault(os.path.normpath(str(source.dataset_path)), []).append(source)

    merged: List[EvalSource] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        label = "+".join(s.label for s in group)
        weight = float(sum(float(s.weight) for s in group))
        path = group[0].dataset_path
        named = [s for s in group if s.candidate_rows is not None]
        bounded = [s for s in group if s.candidate_rows is None]
        if named and bounded:
            raise ValueError(
                f"{path} is read both through an extraction and on the fly; which of its rows "
                "the training never read cannot be decided"
            )
        if named:
            common = set(named[0].candidate_rows)
            for s in named[1:]:
                common &= set(s.candidate_rows)
            merged.append(EvalSource(label=label, dataset_path=path,
                                     candidate_rows=tuple(sorted(common)), weight=weight))
            continue
        if any(s.rows_read is None for s in bounded):
            rows_read = None  # an unknown bound: nothing counts as unseen
        elif any(int(s.rows_read) <= 0 for s in bounded):
            rows_read = 0     # one extraction read every row
        else:
            rows_read = max(int(s.rows_read) for s in bounded)
        merged.append(EvalSource(
            label=label, dataset_path=path, rows_read=rows_read,
            rows_processed=max(int(s.rows_processed) for s in bounded), weight=weight,
        ))
    return merged


def _materialise(dataset: Any, rows: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """The rows' ``input_ids`` and ``attention_mask`` as CPU tensors, right-padded if ragged."""
    subset = dataset.select(list(rows))
    ids_rows = [list(r) for r in subset["input_ids"]]
    names = getattr(subset, "column_names", None) or []
    masks = (
        [list(m) for m in subset["attention_mask"]] if "attention_mask" in names
        else [[1] * len(r) for r in ids_rows]
    )
    width = max(len(r) for r in ids_rows)
    ids = torch.zeros(len(ids_rows), width, dtype=torch.long)
    mask = torch.zeros(len(ids_rows), width, dtype=torch.long)
    for i, (r, m) in enumerate(zip(ids_rows, masks)):
        ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, :len(m)] = torch.tensor(m, dtype=torch.long)
    return ids, mask


def _row_width(dataset: Any) -> int:
    """The first row's width: the blocks of one tokenization share it (packed, or padded to max_length)."""
    return max(1, len(dataset[0]["input_ids"]))


# ── SAEs from the export ────────────────────────────────────────────────────


#: Keys the community converter emits that some architectures do not have and
#: that carry nothing: the zero `decoder.bias` it adds for a bias-free decoder.
_BENIGN_EXTRA_KEYS = {"decoder.bias"}


def build_sae_for_training(hp: Mapping[str, Any], hidden_dim: int, latent_dim: int) -> torch.nn.Module:
    """The SAE module a training with these hyperparameters built, on CPU, untrained."""
    from ..core.framework_defaults import get_framework_defaults
    from ..ml.sparse_autoencoder import create_sae

    architecture_type = hp.get("architecture_type", "standard")
    if architecture_type == "standard":
        architecture_type = "standard_saelens"
    fw = get_framework_defaults(architecture_type)
    return create_sae(
        architecture_type=architecture_type,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        l1_alpha=(hp["l1_alpha"] if hp.get("l1_alpha") is not None else fw.get("default_l1_alpha", 5e-4)),
        ghost_gradient_penalty=hp.get("ghost_gradient_penalty", 0.0),
        normalize_activations=hp.get("normalize_activations", fw["normalize_activations"]),
        top_k_sparsity=hp.get("top_k_sparsity", None),
        top_k=hp.get("top_k"),
        aux_k=hp.get("aux_k"),
        aux_loss_alpha=hp.get("aux_loss_alpha"),
        initial_threshold=hp.get("initial_threshold", 0.5),
        bandwidth=hp.get("bandwidth", 0.01),
        ste_bandwidth=hp.get("ste_bandwidth", 0.5),
        sparsity_coeff=hp.get("sparsity_coeff"),
        normalize_decoder=hp.get("normalize_decoder", fw["normalize_decoder"]),
    )


def load_exported_sae(hp: Mapping[str, Any], sae_dir: Path) -> torch.nn.Module:
    """Rebuild one SAE from its Community Standard export, refusing any weight that does not fit.

    STRICT, WITH ONE NAMED EXCEPTION. A lenient load would leave a parameter at
    its random initialisation and evaluate a different SAE than the one that was
    trained — a number with no error attached. Missing keys raise; unexpected keys
    raise unless they are the zero ``decoder.bias`` the converter emits.
    """
    from ..ml.community_format import load_sae_community_format

    weights, _config, _sparsity = load_sae_community_format(Path(sae_dir), device="cpu")
    encoder = weights.get("W_enc", weights.get("encoder.weight"))
    if encoder is None:
        raise ValueError(f"{sae_dir}: no encoder weight in the export")
    latent_dim, hidden_dim = int(encoder.shape[0]), int(encoder.shape[1])
    model = build_sae_for_training(hp, hidden_dim, latent_dim)
    expected = model.state_dict()

    state = dict(weights)
    if "b_pre" in expected and "decoder_bias" in state and "b_pre" not in state:
        state["b_pre"] = state.pop("decoder_bias")  # TopK names its centring bias b_pre
    unexpected = set(state) - set(expected)
    if unexpected - _BENIGN_EXTRA_KEYS:
        raise ValueError(f"{sae_dir}: weights the architecture does not have: {sorted(unexpected)}")
    model.load_state_dict({k: v for k, v in state.items() if k in expected}, strict=True)
    return model.eval()


def exported_sae_dir(community_dir: Path, layer_idx: int, hook_type: str) -> Path:
    named = Path(community_dir) / f"layer_{layer_idx}_{hook_type}"
    return named if named.exists() else Path(community_dir) / f"layer_{layer_idx}"


# ── persistence ─────────────────────────────────────────────────────────────


def write_evaluation(get_db: Callable, training_id: str, document: Dict[str, Any]) -> bool:
    """Store the document on the training row. Returns False when the row is gone.

    NOT ``record_progress``: that guard refuses writes to a terminal row, and a
    completed training is terminal — which is exactly when a re-run writes here.
    """
    from ..models.training import Training

    with get_db() as db:
        row = db.query(Training).filter(Training.id == training_id).first()
        if row is None:
            logger.warning("Training %s is gone; its evaluation is not recorded", training_id)
            return False
        row.evaluation = carry_stop_request(getattr(row, "evaluation", None), document)
        db.commit()
    return True


def request_evaluation_stop(
    document: Any, *, requested_by: str, now: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """The evaluation document with an operator's Stop in it, or None when nothing is pending or running.

    Review R3-A, R2D-1: once a training's full-length export is saved the run is
    COMPLETED, and a Stop cancels only its evaluation. The endpoint stores what this
    returns; the evaluation honours it at its next check (``requested_stop_reason``).
    """
    if not isinstance(document, Mapping) or document.get("status") not in ACTIVE_STATUSES:
        return None
    return {**document, STOP_REQUESTED_AT: now or _now(), STOP_REQUESTED_BY: str(requested_by)}


def requested_stop_reason(document: Any) -> Optional[str]:
    """Why a pending or running evaluation must stop because an operator asked, or None."""
    if not isinstance(document, Mapping) or document.get("status") not in ACTIVE_STATUSES:
        return None
    if not document.get(STOP_REQUESTED_AT):
        return None
    return (
        f"the operator stopped the evaluation ({document.get(STOP_REQUESTED_BY) or 'stop'} "
        f"at {document[STOP_REQUESTED_AT]})"
    )


def carry_stop_request(stored: Any, document: Mapping[str, Any]) -> Dict[str, Any]:
    """``document``, keeping a Stop the stored record carries for the same pending or running evaluation.

    The evaluation rewrites its whole record as it goes (the heartbeat). Without the
    carry, a Stop the endpoint wrote between two of those writes was erased by the
    next one before the evaluation read it. Only onto a record that is itself still
    pending or running, and only for the same task: a finished record needs no Stop,
    and a re-run queued since is a different evaluation.
    """
    new = dict(document)
    if (
        isinstance(stored, Mapping)
        and requested_stop_reason(stored)
        and new.get("status") in ACTIVE_STATUSES
        and stored.get("task_id") == new.get("task_id")
        and not new.get(STOP_REQUESTED_AT)
    ):
        new[STOP_REQUESTED_AT] = stored[STOP_REQUESTED_AT]
        new[STOP_REQUESTED_BY] = stored.get(STOP_REQUESTED_BY)
    return new


def post_run_placeholder(
    *, task_id: Optional[str], not_run_reason: Optional[str] = None, now: Optional[str] = None
) -> Dict[str, Any]:
    """The post-run record a training writes in the same commit that marks it COMPLETED (review R3-A).

    ``pending`` when the evaluation is about to run: a Stop pressed before its first
    write then lands on a record the evaluation carries (``carry_stop_request``).
    ``cancelled`` with ``not_run_reason`` when a Stop or Pause landed while the last
    steps and the export ran, so the evaluation is not run at all.
    """
    stamp = now or _now()
    document: Dict[str, Any] = {
        "version": EVALUATION_VERSION,
        "trigger": "post_run",
        "hook_point": "resid_post",
        "task_id": task_id,
        "updated_at": stamp,
    }
    if not_run_reason is None:
        return {**document, "status": STATUS_PENDING, "requested_at": stamp}
    return {**document, "status": STATUS_CANCELLED, "completed_at": stamp, "reason": not_run_reason[:2000]}


def announce_evaluation(training_id: str, document: Dict[str, Any]) -> None:
    """Send a finished record on the event the UI listens for (``training:evaluation``)."""
    _emit(training_id, document)


def _emit(training_id: str, document: Dict[str, Any]) -> None:
    try:
        from ..workers.websocket_emitter import emit_training_progress

        emit_training_progress(
            training_id=training_id,
            event="training:evaluation",
            data={"training_id": training_id, "evaluation": document},
        )
    except Exception:  # noqa: BLE001 - notification is best-effort
        logger.warning("Could not emit the evaluation event for %s", training_id)


# ── the step ────────────────────────────────────────────────────────────────


def base_model_loader(
    *,
    model_fields: Optional[Mapping[str, Any]],
    placement: Any,
    sae_mb: float,
    loader: Callable[..., Any],
    resolve_path: Callable[[str], Path],
    keep: Any = None,
) -> Callable[[], Tuple[Any, Optional[Callable[[], None]]]]:
    """The ``load_base_model`` argument of :func:`run_evaluation`, for a placed job.

    ``keep`` is a model the caller already has (the on-the-fly training path),
    returned as-is with no release. Otherwise the model row's repo, quantization
    and files are loaded the way the training's own base-model load does: the
    placement's ``device_map`` — the placed card, or a split over its cards — and,
    for a split, ``max_memory`` with the SAEs' share kept free on their card.
    """
    def load():
        if keep is not None:
            return keep, None
        if not model_fields or not model_fields.get("repo_id"):
            raise ValueError("no base model on record for this training")
        from ..ml.model_loader import QuantizationFormat
        from .base_model_budget import budget_beside_sae

        file_path = model_fields.get("file_path")
        model_dir = resolve_path(file_path) if file_path else None
        model, _tokenizer, _config, _meta = loader(
            repo_id=model_fields["repo_id"],
            quant_format=QuantizationFormat(model_fields["quantization"]),
            cache_dir=model_dir,
            device_map=placement.device_map,
            max_memory=budget_beside_sae(placement, sae_mb),
            local_files_only=bool(model_dir and Path(model_dir).exists()),
        )
        model.eval()
        holder = [model]

        def release():
            holder.clear()
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return model, release

    return load


def _batches_over(
    opened: Sequence[OpenedSource], batch_tokens: int, device: torch.device
) -> Callable[[], Iterable[Tuple[torch.Tensor, torch.Tensor]]]:
    """A re-iterable batch source: every source's rows, materialised once, batched by width."""
    blocks = []
    for item in opened:
        if not item.rows:
            continue
        ids, mask = _materialise(item.dataset, item.rows)
        per_batch = max(1, int(batch_tokens) // max(1, ids.shape[1]))
        for start in range(0, ids.shape[0], per_batch):
            blocks.append((ids[start:start + per_batch], mask[start:start + per_batch]))

    def batches():
        for ids, mask in blocks:
            yield ids.to(device), mask.to(device)

    return batches


#: GPU bytes allowed for the two logit tensors a cross-entropy batch holds (see
#: ``sae_evaluation.ce_working_bytes``): a wide vocabulary gets a shorter batch.
LOGITS_BUDGET_BYTES = 2 * 1024 ** 3

#: Activations, SAE latents and allocator slack beside the cross-entropy arithmetic.
ACTIVATION_MARGIN_BYTES = 512 * 1024 ** 2

#: The block width assumed before a tokenization is opened; a batch holds at least one row.
ASSUMED_ROW_TOKENS = 2_048


def vocab_size_of(source: Any) -> Optional[int]:
    """The output vocabulary of a loaded model, or of a model row's ``architecture_config``."""
    if isinstance(source, Mapping):
        for config in (source, source.get("text_config")):
            if isinstance(config, Mapping):
                value = config.get("vocab_size")
                if isinstance(value, int) and value > 0:
                    return value
        return None
    head = getattr(source, "get_output_embeddings", None)
    try:
        weight = head().weight if callable(head) and head() is not None else None
    except Exception:  # noqa: BLE001 - fall back to the config
        weight = None
    if weight is not None and weight.dim() == 2:
        return int(weight.shape[0])
    config = getattr(source, "config", None)
    for cfg in (config, getattr(config, "text_config", None)):
        value = getattr(cfg, "vocab_size", None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def evaluation_batch_tokens(requested: int, vocab_size: Optional[int]) -> int:
    """Tokens a forward pass may take: ``requested``, shortened so two logit tensors fit the budget."""
    if not vocab_size:
        return max(1, int(requested))
    return max(1, min(int(requested), LOGITS_BUDGET_BYTES // (2 * 4 * int(vocab_size))))


def evaluation_working_mb(
    vocab_size: Optional[int], batch_tokens: int = DEFAULT_BATCH_TOKENS
) -> Optional[float]:
    """MB the evaluation holds beside the base model and SAE weights; None when the vocabulary is unknown."""
    from .sae_evaluation import ce_working_bytes

    if not vocab_size:
        return None
    tokens = max(evaluation_batch_tokens(batch_tokens, vocab_size), ASSUMED_ROW_TOKENS)
    return (ce_working_bytes(vocab_size, tokens, logit_bytes=4) + ACTIVATION_MARGIN_BYTES) / 1024 ** 2


def unspliceable_reason(architecture_type: Optional[str], hook_type: str) -> Optional[str]:
    """Why an SAE cannot be spliced into the residual stream, or None when it can.

    ONE RULE for the post-run step (which sees loaded modules) and the re-run task
    (which must decide before loading, from the hyperparameters). Only a
    whole-layer output can be substituted: a transcoder maps one layer to another,
    and an attention or MLP sub-output is not what the next layer reads.
    """
    from ..ml.forward_hooks import HookType

    if hook_type != HookType.RESIDUAL.value:
        return "only residual (resid_post) SAEs can be spliced"
    if str(architecture_type or "").lower() == "transcoder":
        return "a transcoder reconstructs a different layer's output"
    return None


def run_evaluation(
    *,
    get_db: Callable,
    training_id: str,
    hp: Mapping[str, Any],
    saes: Mapping[Tuple[int, str], torch.nn.Module],
    sources: Any,
    load_base_model: Callable[[], Any],
    trigger: str,
    token_budget: Optional[int] = None,
    batch_tokens: int = DEFAULT_BATCH_TOKENS,
    open_dataset: Optional[Callable[[str], Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    skipped_saes: Optional[Sequence[Mapping[str, Any]]] = None,
    should_stop: Optional[Callable[[], Optional[str]]] = None,
) -> Dict[str, Any]:
    """Evaluate the SAEs on unseen blocks and record the result. NEVER RAISES.

    Args:
        get_db: sync session factory (a task's ``get_db``).
        saes: ``(layer, hook_type)`` -> trained SAE, on the device it will run on.
        sources: where unseen blocks come from (see :class:`EvalSource`) — a sequence,
            or a callable returning one, so a failure building them is recorded too.
        load_base_model: called once, after every cheaper check has passed; returns
            ``(model, release)`` — the model on the job's placement and a callable
            that frees it (a no-op when the caller keeps the model).
        trigger: ``"post_run"`` or ``"rerun"``, recorded.
        token_budget: tokens to read; ``hp['evaluation_token_budget']`` or the default when None.
        open_dataset: path -> tokenized dataset (``datasets.load_from_disk`` by default).
        extra: recorded as-is beside the result (e.g. the Celery task id).

    Returns:
        The stored document.
    """
    budget = token_budget if token_budget is not None else hp.get("evaluation_token_budget")
    budget = DEFAULT_TOKEN_BUDGET if budget is None else int(budget)
    base = {
        "version": EVALUATION_VERSION,
        "trigger": trigger,
        "hook_point": "resid_post",
        **dict(extra or {}),
    }
    config = {"token_budget": budget, "batch_tokens": int(batch_tokens), "seed": int(hp.get("seed") or 0)}

    def record(status: str, **fields: Any) -> Dict[str, Any]:
        document = {**base, "status": status, "config": config, "updated_at": _now(), **fields}
        try:
            write_evaluation(get_db, training_id, document)
        except Exception:  # noqa: BLE001 - recording must not raise either
            logger.exception("Could not record the evaluation of %s", training_id)
        if status not in ACTIVE_STATUSES:
            _emit(training_id, document)
        return document

    def stop_reason() -> Optional[str]:
        """The caller's stop (a Stop on the row, a lost lease), then an operator's Stop on the record.

        THE RECORD, READ HERE, FOR BOTH TRIGGERS (review R3-A, R2D-1). A post-run
        evaluation now runs on a COMPLETED row, whose status carries no Stop; the
        Stop endpoint writes the request into this record instead.
        """
        if should_stop is not None:
            reason = should_stop()
            if reason:
                return reason
        from ..models.training import Training

        with get_db() as db:
            row = db.query(Training).filter(Training.id == training_id).first()
            return requested_stop_reason(getattr(row, "evaluation", None)) if row is not None else None

    # THE FLAG GOVERNS THE AUTOMATIC EVALUATION ONLY (user decision, 2026-09-16).
    # An explicit re-run — `trigger="rerun"`, from POST /trainings/{id}/evaluate — is
    # an operator asking for THIS evaluation now. Honouring the flag there accepted
    # the request with 202, took a GPU lease, released it a second later and recorded
    # "skipped" with a reason visible only inside the JSONB: a button that quietly did
    # nothing. Same shape as Finalize — a guard on the automatic path, an explicit
    # request that overrides it.
    if trigger == "post_run" and not hp.get("evaluate_ce_delta", True):
        return record(STATUS_SKIPPED, reason="disabled by the evaluate_ce_delta hyperparameter")
    if budget <= 0:
        return record(STATUS_SKIPPED, reason="evaluation_token_budget is 0")

    started_at = _now()
    record(STATUS_RUNNING, started_at=started_at)
    release = None
    model = None
    try:
        # Only a whole-layer output can be substituted; a transcoder maps one
        # layer to another and an attention/MLP sub-output is not what the next
        # layer reads, so splicing either would measure something else.
        from ..ml.forward_hooks import HookType

        # The caller may have decided some before loading them (the re-run task).
        evaluable, skipped = {}, [dict(entry) for entry in (skipped_saes or [])]
        for (layer_idx, hook_type), sae in sorted(saes.items()):
            reason = unspliceable_reason(
                "transcoder" if type(sae).__name__ == "Transcoder" else hp.get("architecture_type"),
                hook_type,
            )
            if reason:
                skipped.append({"layer": layer_idx, "hook_type": hook_type, "reason": reason})
            else:
                evaluable[int(layer_idx)] = sae
        if not evaluable:
            return record(STATUS_SKIPPED, started_at=started_at, skipped_saes=skipped,
                          reason="no SAE in this training can be spliced into the residual stream")

        if open_dataset is None:
            from datasets import load_from_disk as open_dataset  # noqa: N813

        if callable(sources):
            sources = sources()
        # Two extractions of one tokenization are one source: a row either read is read.
        sources = merge_shared_tokenizations(list(sources))
        opened: List[OpenedSource] = []
        for source in sources:
            dataset = open_dataset(source.dataset_path)
            total = len(dataset)
            opened.append(OpenedSource(source=source, total_rows=total,
                                       unseen=_unseen_count(source, total), dataset=dataset))
        widths = [_row_width(o.dataset) if o.total_rows else 1 for o in opened]
        selection = select_unseen_rows(
            [o.source for o in opened], [o.total_rows for o in opened], widths,
            budget, config["seed"],
        )
        for item, rows in zip(opened, selection):
            item.rows = rows
        source_report = [
            {
                "label": o.source.label,
                "dataset_path": o.source.dataset_path,
                "total_rows": o.total_rows,
                "rows_read_by_training": first_unseen_row(o.source, o.total_rows),
                "unseen_rows": o.unseen,
                "weight": o.source.weight,
                "blocks_evaluated": len(o.rows),
            }
            for o in opened
        ]
        if not any(o.rows for o in opened):
            return record(
                STATUS_SKIPPED, started_at=started_at, sources=source_report, skipped_saes=skipped,
                reason=(
                    "no block the training never read: every source's rows were all read by "
                    "the training (or, for a run that extracts on the fly, no held-out rows were named)"
                ),
            )

        import time

        # A fresh write before the slowest silent step, so the janitor's clock starts here.
        record(STATUS_RUNNING, started_at=started_at, sources=source_report,
               progress={"stage": "loading_model"})
        # A STOP BEFORE THE SLOWEST SILENT STEP (review R3-A): a Stop pressed while the
        # evaluation was pending, or while it chose its blocks, would otherwise wait for
        # a whole model load and a first batch.
        reason = stop_reason()
        if reason:
            raise EvaluationStopped(reason)
        model, release = load_base_model()
        from ..ml.layer_discovery import discover_transformer_structure
        from ..ml.model_devices import input_device

        structure = discover_transformer_structure(model)
        layer_modules = {L: structure.layers_module[L] for L in evaluable}
        # A WIDE VOCABULARY GETS A SHORTER BATCH: the batch holds two logit tensors.
        batch_tokens_used = evaluation_batch_tokens(batch_tokens, vocab_size_of(model))
        if batch_tokens_used != int(batch_tokens):
            logger.info("Evaluation of %s: %s tokens a batch (vocabulary %s)",
                        training_id, batch_tokens_used, vocab_size_of(model))
        config["batch_tokens"] = batch_tokens_used
        batches = _batches_over(opened, batch_tokens_used, input_device(model))
        n_batches = sum(
            math.ceil(len(o.rows) / max(1, batch_tokens_used // max(1, width)))
            for o, width in zip(opened, widths) if o.rows
        )

        clock = time.monotonic
        last_beat = [clock()]

        def heartbeat(stage: str, batches_done: int) -> None:
            """After every batch: stop if told to, then rewrite the record's clock at most once a HEARTBEAT_SECONDS.

            THE STOP IS CHECKED EVERY BATCH, NOT THROTTLED (review R1-D, R1D-7). A
            Stop during the evaluation used to let every remaining batch run: 35 of
            36 forwards after it, with the job holding its card throughout. One
            cross-entropy batch is 2 + 3 x layers forwards, which bounds the work
            done after a stop; the row read it costs is small beside that.
            """
            reason = stop_reason()
            if reason:
                raise EvaluationStopped(reason)
            if clock() - last_beat[0] < HEARTBEAT_SECONDS:
                return
            last_beat[0] = clock()
            record(STATUS_RUNNING, started_at=started_at, sources=source_report,
                   progress={"stage": stage, "batches_done": int(batches_done), "batches": n_batches})

        result = evaluate_spliced_layers(model, evaluable, layer_modules, batches, progress=heartbeat)
        for entry in result["layers"]:
            entry["hook_type"] = HookType.RESIDUAL.value
        logger.info(
            "Evaluation of %s on %s unseen tokens: base CE %.4f; %s", training_id, result["tokens"],
            result["ce_base"] or float("nan"),
            "; ".join(
                f"L{e['layer']} spliced {e['ce_spliced']} recovered(mean) {e['loss_recovered_vs_mean']}"
                for e in result["layers"]
            ),
        )
        return record(STATUS_COMPLETED, started_at=started_at, completed_at=_now(),
                      sources=source_report, skipped_saes=skipped, **result)
    except EvaluationStopped as stop:
        # Between batches, on the job's instruction (R1D-7). Not a failure, and the
        # partial sums are not a result: nothing but the reason is recorded.
        logger.info("Evaluation of %s stopped: %s", training_id, stop)
        return record(STATUS_CANCELLED, started_at=started_at, completed_at=_now(),
                      reason=f"stopped: {stop}"[:2000])
    except Exception as exc:  # noqa: BLE001 - an evaluation must never fail its training
        logger.exception("Evaluation of %s failed", training_id)
        return record(STATUS_FAILED, started_at=started_at, completed_at=_now(),
                      reason=f"{type(exc).__name__}: {exc}"[:2000])
    finally:
        # Drop this frame's reference first, or the release frees nothing.
        model = None
        if callable(release):
            try:
                release()
            except Exception:  # noqa: BLE001
                logger.warning("Could not release the evaluation's base model", exc_info=True)
