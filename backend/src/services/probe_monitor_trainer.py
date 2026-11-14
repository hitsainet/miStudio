"""Layer selection, rule training, the k-sparse SAE variant and calibration (FR-6–FR-8, FR-10).

Four stages, each a pure-ish function over arrays so it can be tested without a GPU:

  `select_layers`      sklearn LR per (layer, pooling) → validation AUROC → top-N
  `train_rule`         torch, AdamW, BCE on the AGGREGATE, early stop on val AUROC
  `select_sae_features` k-sparse feature choice, scored on TRAIN ONLY
  `calibrate`          the threshold at a target FPR, and what it really spends

⚠ SELECTION SCORES ON VALIDATION, FEATURE RANKING SCORES ON TRAIN. The two are not
inconsistent — they are avoiding different leaks. A layer chosen by its TRAINING AUROC
would pick whichever layer memorises best, so selection needs held-out data. An SAE
feature set ranked on VALIDATION rows leaks those rows into the model's structure, so
the validation AUROC that then decides early stopping is no longer held out. Each
stage must score on data the thing it is choosing has not seen.

⚠ TRAINING OPTIMISES THE AGGREGATE, NOT THE PER-TOKEN SCORE. A probe is judged by one
number per input, and the rule that produces it is part of the function. Fitting
per-token labels instead — every token of a positive row labelled positive — trains a
detector for "this token looks like it came from a risky document", which is a
different and easier problem, and then the aggregate is whatever it happens to be.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Pooling names used in the selection grid. They are the poolings `capture_pooled`
#: produces, and they mirror the `mean` and `last` rules.
POOLINGS = ("mean", "last")

#: FULL-BATCH STEPS, not passes over mini-batches. A probe is a d-dimensional linear
#: model over a few thousand rows, so one gradient per step over the whole set is both
#: cheaper and exactly reproducible — there is no batch order to seed.
#:
#: ⚠ THESE NUMBERS ARE MEASURED, NOT GUESSED, AND THE FIRST GUESS DID NOT TRAIN AT ALL.
#: 60 steps at lr 1e-3 from a zero init reached **0.51 AUROC on linearly separable
#: data** — the loop ran, early-stopped, and reported a number, which is the worst
#: possible failure mode for a trainer. `test_probe_monitor_trainer` now asserts a
#: separable problem is actually solved, so a configuration that cannot converge fails
#: the suite instead of shipping a probe that reads as "the concept is undetectable".
DEFAULT_EPOCHS = 400
DEFAULT_PATIENCE = 40
DEFAULT_LR = 0.05
DEFAULT_WEIGHT_DECAY = 1e-2


@dataclass
class LayerScore:
    layer: int
    pooling: str
    val_auroc: Optional[float]
    n_train: int
    n_val: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "pooling": self.pooling,
            "val_auroc": self.val_auroc,
            "n_train": self.n_train,
            "n_val": self.n_val,
        }


@dataclass
class LayerSelection:
    """The FULL grid plus the winners, because a near-tie must be visible."""

    grid: List[LayerScore]
    chosen: List[int]
    #: The runner-up's margin. A layer chosen by 0.002 of AUROC is an arbitrary
    #: choice, and a report that hides that invites over-reading the layer.
    margin: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "grid": [score.as_dict() for score in self.grid],
            "chosen": list(self.chosen),
            "margin": self.margin,
            "poolings": list(POOLINGS),
        }


def selection_workers(cells: int) -> int:
    """Threads for the layer-selection sweep: one per cell, capped by the cores this process may use."""
    import os

    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cores = os.cpu_count() or 1
    return max(1, min(cells, cores))


def select_layers(
    pooled: Any,
    labels: Sequence[int],
    train_index: Sequence[int],
    val_index: Sequence[int],
    *,
    top_n: int = 1,
    seed: int = 1337,
    workers: Optional[int] = None,
) -> LayerSelection:
    """Fit an L2 logistic regression per (layer, pooling) and rank by VALIDATION AUROC.

    sklearn on CPU, because this is a d-dimensional problem over a few thousand rows
    and a torch loop would be slower and less reproducible. Standardisation is fitted
    on TRAIN and applied to validation — fitting it on everything leaks the validation
    distribution's scale into the model.

    Returns the whole grid. A sweep that records only its argmax cannot be audited for
    a near-tie, and a near-tie is exactly when the chosen layer is arbitrary.

    ⚠ THE CELLS FIT IN PARALLEL, ONE BLAS THREAD EACH. Each (layer, pooling) fit is independent
    and deterministic, and one fit is a few thousand rows by d_model: too small for a
    multithreaded BLAS to use the cores, so the sequential sweep left most of them idle and was
    MEASURED at 301.7 s of a 1,408 s run (`pmr_2bbbf15c1a89`, 32 cells). `workers` threads with
    BLAS capped at one thread per worker run the same fits side by side; `map` keeps the grid in
    cell order, so the grid, the ranking and the chosen layers are identical to a sequential
    sweep (`workers=1` is one). Pinned by `test_layer_selection_parallel_is_identical`.
    """
    from concurrent.futures import ThreadPoolExecutor

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits

    y = np.asarray(labels)
    y_train = y[list(train_index)]
    y_val = y[list(val_index)]
    scorable = len(set(y_train.tolist())) > 1 and len(set(y_val.tolist())) > 1

    cells: List[Tuple[str, int, Any]] = []
    for pooling in POOLINGS:
        source = getattr(pooled, pooling)
        for layer in sorted(source):
            matrix = source[layer].numpy() if hasattr(source[layer], "numpy") else source[layer]
            cells.append((pooling, layer, matrix))

    def fit_cell(cell: Tuple[str, int, Any]) -> LayerScore:
        pooling, layer, matrix = cell
        x_train = np.asarray(matrix)[list(train_index)]
        x_val = np.asarray(matrix)[list(val_index)]
        score: Optional[float] = None
        if scorable:
            mu = x_train.mean(axis=0)
            sigma = x_train.std(axis=0)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)
            model = LogisticRegression(max_iter=2000, random_state=seed)
            model.fit((x_train - mu) / sigma, y_train)
            predictions = model.decision_function((x_val - mu) / sigma)
            score = float(roc_auc_score(y_val, predictions))
        return LayerScore(
            layer=layer,
            pooling=pooling,
            val_auroc=score,
            n_train=len(x_train),
            n_val=len(x_val),
        )

    if workers is None:
        workers = selection_workers(len(cells))
    workers = max(1, min(int(workers), len(cells) or 1))
    if workers == 1:
        grid = [fit_cell(cell) for cell in cells]
    else:
        with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
            grid = list(pool.map(fit_cell, cells))

    scored = [entry for entry in grid if entry.val_auroc is not None]
    if not scored:
        raise ValueError(
            "no (layer, pooling) cell could be scored: one of the splits has a single "
            "class, so there is no validation AUROC to rank by"
        )

    # Rank by the BEST pooling per layer: the layer is what gets captured at token
    # level, and a layer that wins under `last` is still that layer.
    best_per_layer: Dict[int, float] = {}
    for entry in scored:
        current = best_per_layer.get(entry.layer)
        if current is None or entry.val_auroc > current:
            best_per_layer[entry.layer] = float(entry.val_auroc)
    ordered = sorted(best_per_layer.items(), key=lambda pair: (-pair[1], pair[0]))
    chosen = [layer for layer, _ in ordered[: max(1, top_n)]]
    margin = None
    if len(ordered) > len(chosen):
        margin = float(ordered[len(chosen) - 1][1] - ordered[len(chosen)][1])
    return LayerSelection(grid=grid, chosen=chosen, margin=margin)


@dataclass
class TrainedRule:
    """What a rule's training produced."""

    rule: str
    weight: torch.Tensor
    bias: float
    mean: torch.Tensor
    std: torch.Tensor
    attention_query: Optional[torch.Tensor]
    rule_params: Dict[str, Any]
    val_auroc: Optional[float]
    epochs_run: int
    best_epoch: int
    history: List[Dict[str, Any]] = field(default_factory=list)


def _standardisation(
    rows: Sequence[np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and std over every TRAINING token.

    ⚠ A DEGENERATE CHANNEL GETS std 1.0, NOT eps. Dividing by a floor of 1e-6
    amplifies a constant channel's serving-time drift by a million — the defect fixed
    in `ProbeHead.standardise`, and the statistics must not reintroduce it from the
    other side by handing it a tiny std to clamp.

    ⚠ STREAMED IN TWO PASSES, NOT CONCATENATED. It used to build one fp32 array of every
    training token: 850,425 x 4096 x 4 B = 13.9 GB on the acceptance set, and the list
    comprehension feeding `np.concatenate` held a second full copy while the concatenation
    ran, so the peak was about 28 GB — on top of the 8 GB the rows already occupy and the
    14 GB the batches will. It is the same class of allocation as the 154 GB padded tensor
    this module was OOMKilled on, just smaller, and it buys nothing: the mean and variance
    over a set of rows do not need the rows in one buffer.

    TWO PASSES RATHER THAN sum/sumsq. `E[x^2] - E[x]^2` loses precision by catastrophic
    cancellation exactly when the mean is large relative to the spread, which is the
    normal condition for residual-stream channels. The second pass costs one more read of
    a memmap-backed array and is unconditionally stable.
    """
    if not rows:
        raise ValueError("no rows to compute standardisation from")
    d_model = int(np.asarray(rows[0]).shape[1])
    count = 0
    total = np.zeros(d_model, dtype=np.float64)
    for chunk in rows:
        array = np.asarray(chunk, dtype=np.float32)
        if array.shape[0] == 0:
            continue
        total += array.sum(axis=0, dtype=np.float64)
        count += int(array.shape[0])
    if count == 0:
        raise ValueError("every row is empty, so there is nothing to standardise against")
    mean = total / count

    squared = np.zeros(d_model, dtype=np.float64)
    for chunk in rows:
        array = np.asarray(chunk, dtype=np.float32)
        if array.shape[0] == 0:
            continue
        squared += ((array - mean) ** 2).sum(axis=0, dtype=np.float64)
    # Population std, matching `np.std`'s default — `ProbeHead` standardises with these
    # same numbers, so the convention has to be the one the head was fitted under.
    std = np.sqrt(squared / count)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


#: Bytes of padded fp32 activations per training batch.
#:
#: ⚠ IN BYTES, NOT SLOTS, AND THAT IS THE SECOND TIME THIS NUMBER WAS WRONG. The first
#: version of this fix used a slot budget of 32,000,000 and its own comment claimed that
#: was "512 MB at d=4096" — it is 524 GB (32e6 x 4096 x 4). At that budget all 6,800 rows
#: still landed in ONE batch, so the length bucketing never engaged and the forecast
#: still read 154 GB: a fix that changed nothing, caught by asserting the waste factor
#: rather than by reading the code. A byte budget cannot be wrong by a factor of d_model,
#: because d_model is in the conversion.
DEFAULT_BATCH_BYTES: int = 512 * 1024 ** 2

def host_ram_usable_bytes(available_bytes: Optional[int] = None, reserved_bytes: int = 0) -> int:
    """What this host can give the standardised batches right now.

    ⚠ THIS WAS A FIXED 48 GiB (`MAX_BATCH_BYTES`), written for a 124 GB node shared with a 16 GB model.
    A constant is wrong in both directions: on a node with about 90 GB available it refused what would
    fit, and on a busy or smaller one it allowed what would not. Capacity is provisioned, so the bound
    is what is free when asked.

    The rule `gpu_job_claim.host_ram_refusal` already applies to a job starting beside another: the
    kernel's MemAvailable, less `ResourceConfig.RAM_SAFETY_MARGIN` of it, less what other live jobs have
    reserved and not yet allocated. MemAvailable counts reclaimable page cache as available, so the
    token memmaps the batches are read from do not need room of their own.
    """
    from .resource_config import ResourceConfig

    if available_bytes is None:
        import psutil

        available_bytes = psutil.virtual_memory().available
    return int(int(available_bytes) * (1 - ResourceConfig.RAM_SAFETY_MARGIN)) - int(reserved_bytes)


def _ram_ceiling(device: torch.device, max_bytes: Optional[int]) -> Optional[int]:
    """The limit a build is held to. An explicit one wins; batches bound for a card are not going to
    live in host memory, and `training_device` has already judged the card."""
    if max_bytes is not None:
        return max_bytes
    return None if device.type == "cuda" else host_ram_usable_bytes()


#: Card memory kept free beyond the resident batches: the per-token scores, their gradients and
#: the allocator's own slack. Plus a quarter of the batches again, because `aggregate` builds
#: token-sized intermediates (`x @ query`, the combine's masked scores) per batch.
DEVICE_HEADROOM_BYTES: int = 2 * 1024 ** 3


def attention_query_init(d_model: int, device: torch.device, seed: int) -> torch.Tensor:
    """The attention rule's starting query: a small draw from a CPU generator seeded with `seed`.

    ⚠ ZERO WAS TRIED AND IT DID NOT TRAIN. With the query at zero (2026-10-22, `147ae17d`) two card
    runs were identical but the attention probes stopped at epoch 1 on layers 1 and 4 and lost
    0.10-0.20 held-out AUROC (`0xcc/reviews/probe_training_device_2026-10-22.md`); the fixture
    measurement that said zero trains as well did not hold on real data.

    ⚠ A DRAW ON THE CARD IS A DIFFERENT DRAW. CUDA keeps its own random stream, so the draw is made
    on the CPU, from a generator of its own seeded with the run's seed — the same numbers the
    CPU-trained probes started from (`torch.manual_seed(seed)` then `torch.randn`), independent of
    the global random state and of the device — and then moved to `device`.
    """
    generator = torch.Generator().manual_seed(int(seed))
    return (torch.randn(d_model, generator=generator) * 0.02).to(device)


@contextmanager
def deterministic_kernels(device: Any) -> Iterator[None]:
    """Deterministic CUDA kernels while a probe trains on a card; restored afterwards.

    Two card runs of one configuration must produce one probe — the CPU path always did, and SC-4
    rests on it. `use_deterministic_algorithms` is process-wide, so it is switched on only around
    the fit and put back, because the capture and calibration forwards in the same worker use ops
    with no deterministic CUDA kernel. `warn_only` keeps such an op from failing a fit; cuBLAS'
    own reductions additionally need `CUBLAS_WORKSPACE_CONFIG`, which the image sets before any
    handle is created.
    """
    if getattr(device, "type", None) != "cuda":
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


def training_device(
    model_device: Optional[torch.device],
    batch_bytes: int,
    free_bytes: Optional[int],
    *,
    headroom: int = DEVICE_HEADROOM_BYTES,
) -> Tuple[torch.device, str]:
    """Where `train_rule` runs: the model's card when the standardised batches fit beside it.

    ⚠ TRAINING RAN ON THE CPU WHILE THE RUN HELD A GPU. The run passed `torch.device("cpu")`
    unconditionally, so the longest stage — 845 s of the 1,408 s run `pmr_2bbbf15c1a89` — left
    the card idle at 2.2 GB while sixteen cores did full-batch epochs over the whole training
    set. The batches are built once and read on every epoch, so they belong where the epochs run.

    Resident, not streamed: copying every batch over a x4 link on every epoch would cost about
    what the CPU epoch does. So the batches go on the card only when they FIT, and otherwise the
    run trains on the CPU and records why — a refusal of the card is a slower run, never an OOM.
    """
    # A float32-width question to the one rule (`plan_holding`), so this and `batch_holding` cannot drift.
    plan = plan_holding(storage_bytes=4, float32_bytes=batch_bytes, card_present=is_card(model_device),
                        card_free=free_bytes, host_room=None, headroom=headroom)
    return (model_device if plan.on_card else torch.device("cpu")), plan.device_reason


def is_card(device: Any) -> bool:
    """Whether `device` is a CUDA device."""
    return device is not None and getattr(device, "type", None) == "cuda"


def batches_fit_on_card(
    batch_bytes: int, free_bytes: int, *, headroom: int = DEVICE_HEADROOM_BYTES
) -> Tuple[bool, str]:
    """`training_device`'s rule on its own: the batches plus a quarter plus `headroom` against the card's
    free memory. Also what the time forecast (ENH-006 G15) asks of a run that has no card yet."""
    needed = int(batch_bytes) + int(batch_bytes) // 4 + int(headroom)
    if needed > int(free_bytes):
        return False, (
            f"the batches need {needed / 1024 ** 3:.2f} GiB with headroom and the card has "
            f"{int(free_bytes) / 1024 ** 3:.2f} GiB free"
        )
    return True, (
        f"the batches ({int(batch_bytes) / 1024 ** 3:.2f} GiB) fit on the model's card "
        f"({int(free_bytes) / 1024 ** 3:.2f} GiB free)"
    )


class ProbeTrainingTooLarge(RuntimeError):
    """The plan does not fit, said BEFORE allocating anything."""


def slots_for(d_model: int, budget_bytes: int = DEFAULT_BATCH_BYTES) -> int:
    """Padded activation slots that fit in `budget_bytes` at fp32. At least one."""
    if d_model < 1:
        raise ValueError("d_model must be positive")
    return max(1, budget_bytes // (d_model * 4))


def plan_length_buckets(
    widths: Sequence[int], *, budget_slots: int
) -> List[List[int]]:
    """Group row indices into padded batches, SORTED BY LENGTH.

    ⚠ THE DEFECT THIS REPLACES, MEASURED ON THE FIRST STAGE 1 ACCEPTANCE RUN. `_pack`
    right-padded EVERY row to the longest one and trained full-batch over the result. On
    8,000 rows of `Arrrlex/models-under-pressure` at d=4096 the real data is 1,003,703
    tokens — median 89, p99 476, max 1384 — so the padded tensor was

        6,800 x 1,384 x 4,096 x 4 B = 154.2 GB

    for 16.4 GB of actual activations: an ELEVEN-FOLD padding waste, on a 124 GB node.
    The worker was OOMKilled (exit 137) at the training stage after the 17-minute sweep
    and the 14-minute token capture had already succeeded, and the run row was left
    reading `running` because a killed process writes no status.

    Sorting by length is what removes the waste: a batch of 89-token rows pads to 89, not
    to 1384. The waste factor over the whole plan is reported by `forecast_batches` and is
    about 1.02x on this data.

    The order within a batch is irrelevant — each row's aggregate is independent, and the
    loss is summed — so sorting changes no result, only the padding. Row indices travel
    with the batch precisely so labels and predictions can be put back in row order.
    """
    if not widths:
        return []
    if budget_slots < 1:
        raise ValueError("budget_slots must be at least 1")
    order = sorted(range(len(widths)), key=lambda index: widths[index])
    batches: List[List[int]] = []
    current: List[int] = []
    current_max = 0
    for index in order:
        candidate_max = max(current_max, widths[index])
        # A single row wider than the budget still gets its own batch: refusing it would
        # drop data, and `max_length` already bounds how wide it can be.
        if current and (len(current) + 1) * candidate_max > budget_slots:
            batches.append(current)
            current, current_max = [index], widths[index]
        else:
            current.append(index)
            current_max = candidate_max
    if current:
        batches.append(current)
    return batches


def forecast_batches(
    widths: Sequence[int],
    d_model: int,
    *,
    budget_bytes: int = DEFAULT_BATCH_BYTES,
    element_bytes: int = 4,
) -> Dict[str, Any]:
    """What `_standardised_batches` will allocate, before it allocates it.

    An OOM kill is the worst failure mode available here: it takes the worker down
    mid-run, writes no status, and leaves the row claiming to be running until a janitor
    notices 90 minutes later. A forecast that refuses is the same information, an hour
    earlier, with the numbers in the message.

    `element_bytes` is what each held slot costs: 4 for batches standardised once (float32), 2 for
    batches held as their stored float16 and standardised as they are used (`batch_holding`).
    It changes the BYTES only. The bucket plan is always sized at float32 (`slots_for`), because each
    batch is standardised to float32 before the forward, so the batches themselves never change.
    """
    buckets = plan_length_buckets(widths, budget_slots=slots_for(d_model, budget_bytes))
    padded_slots = sum(len(batch) * max(widths[i] for i in batch) for batch in buckets)
    real_tokens = int(sum(widths))
    bytes_needed = padded_slots * d_model * int(element_bytes)
    return {
        "rows": len(widths),
        "batches": len(buckets),
        "real_tokens": real_tokens,
        "padded_slots": padded_slots,
        "waste_factor": round(padded_slots / real_tokens, 4) if real_tokens else None,
        "bytes": bytes_needed,
        "gib": round(bytes_needed / 1024 ** 3, 2),
        "widest_row": int(max(widths)) if widths else 0,
        "batch_bytes": budget_bytes,
        "slots_per_batch": slots_for(d_model, budget_bytes),
        "element_bytes": int(element_bytes),
    }


def stored_as_float16(rows: Sequence[np.ndarray]) -> bool:
    """Whether these rows are a 16-bit model's stored float16 activations, which CAN be held raw (ENH-006 G24).

    Whether they ARE held raw is `batch_holding`'s decision, not this one's.
    """
    return bool(rows) and np.asarray(rows[0]).dtype == np.float16


def storage_bytes_of(rows: Sequence[np.ndarray]) -> int:
    """Bytes per stored slot: 2 for float16 rows, else 4."""
    return 2 if stored_as_float16(rows) else 4


@dataclass(frozen=True)
class BatchHolding:
    """Where a layer's batches live and at what width: `batch_holding`'s answer."""

    device: torch.device
    #: 4 = float32, standardised once at build time; 2 = the stored float16, standardised at every use.
    element_bytes: int
    #: The batches widened to float32: what they cost at 4 B. At 2 B they are exactly half.
    float32_bytes: int
    #: The card's half of the decision (`training_device`'s reason), as the run has always recorded it.
    device_reason: str
    #: Why this width.
    width_reason: str

    @property
    def hold_raw(self) -> bool:
        return self.element_bytes == 2

    @property
    def batch_bytes(self) -> int:
        return held_batch_bytes(self.float32_bytes, self.element_bytes)

    def as_record(self) -> Dict[str, Any]:
        return {
            "element_bytes": self.element_bytes,
            "held": HELD_DESCRIPTION[self.element_bytes],
            "held_reason": self.width_reason,
            "float32_batch_bytes": int(self.float32_bytes),
        }


#: What each width means, as a run record and a forecast state it.
HELD_DESCRIPTION: Dict[int, str] = {
    4: "float32, standardised once",
    2: "float16 as stored, standardised at every use",
}


def held_batch_bytes(float32_bytes: int, element_bytes: int) -> int:
    """The batches at `element_bytes` per slot, from their float32 figure (`forecast_batches` at 4 B)."""
    return int(float32_bytes) * int(element_bytes) // 4


@dataclass(frozen=True)
class HoldingPlan:
    """`plan_holding`'s answer, with the card as a yes or no: what a caller with no device in hand (the time
    forecast) reads. `batch_holding` turns it into a `BatchHolding` on the real device."""

    on_card: bool
    element_bytes: int
    float32_bytes: int
    device_reason: str
    width_reason: str

    @property
    def batch_bytes(self) -> int:
        return held_batch_bytes(self.float32_bytes, self.element_bytes)


def plan_holding(
    *,
    storage_bytes: int,
    float32_bytes: int,
    card_present: bool,
    card_free: Optional[int],
    host_room: Optional[int],
    headroom: int = DEVICE_HEADROOM_BYTES,
) -> HoldingPlan:
    """THE decision (ENH-006 G25): whether a layer's batches go on the model's card, and whether a 16-bit model's
    are held raw. `batch_holding` (a real device) and `training_device` are this, and nothing else decides.

    ⚠ HELD RAW ONLY WHEN FLOAT32 DOES NOT FIT. G24 held every 16-bit model's batches as their stored float16
    and standardised each at use, to fit arm A in memory. It cost the training stage: 9,184 s against arm B's
    1,848 s on the node, and after `879b8dc64` still 2.3-6x float32's epoch on the workstation. Most runs fit
    at float32, so they now widen and standardise ONCE, the pre-G24 path, and only a run that would not fit
    pays the per-use cost. The values are the same either way (`standardise_batch`; pinned bit for bit).

    The order tried, first that fits: the model's card at float32, the card at float16, host memory at float32,
    host memory at float16. The card at float16 beats host memory at float32: on a card the per-use widening
    is one kernel over device memory, far cheaper than a CPU epoch. A float32 model has one width (4).

    Each caller hands in its own figure for room, and this function is the only place the comparison lives:
    `execute_probe_run` per layer (live memory), `forecast_memory` (the forecast's own usable memory, so
    "fits at float32" here is "the forecast at float32 fits"), the time forecast (the card's measured free
    memory), and `train_rule` when called without prebuilt batches. `host_room` None is unknown, and an
    unknown room holds raw: the cautious width, which is what G24 did everywhere.
    """
    widths = (4,) if int(storage_bytes) >= 4 else (4, 2)
    if not card_present:
        device_reason = "the model is not on a CUDA device"
    elif card_free is None:
        device_reason = "the card's free memory could not be read"
    else:
        for width in widths:
            fits, device_reason = batches_fit_on_card(held_batch_bytes(float32_bytes, width), int(card_free),
                                                      headroom=headroom)
            if fits:
                return HoldingPlan(True, width, int(float32_bytes), device_reason,
                                   _width_reason(width, widths, "on the card"))
    for width in widths:
        if host_room is not None and held_batch_bytes(float32_bytes, width) <= int(host_room):
            return HoldingPlan(False, width, int(float32_bytes), device_reason,
                               _width_reason(width, widths, "in host memory", host_room))
    narrowest = widths[-1]
    why = (
        "the room in host memory is not known, so the narrowest width is held" if host_room is None else
        f"no width fits the {int(host_room) / 1024 ** 3:.2f} GiB of host memory there is room for; the narrowest "
        f"is held and the build refuses if it does not fit"
    )
    return HoldingPlan(False, narrowest, int(float32_bytes), device_reason, why)


def batch_holding(
    *,
    storage_bytes: int,
    float32_bytes: int,
    card: Optional[torch.device],
    card_free: Optional[int],
    host_room: Optional[int],
    headroom: int = DEVICE_HEADROOM_BYTES,
) -> BatchHolding:
    """`plan_holding` on a real device: the batches go on `card` when the plan says so, else on the CPU."""
    plan = plan_holding(storage_bytes=storage_bytes, float32_bytes=float32_bytes, card_present=is_card(card),
                        card_free=card_free, host_room=host_room, headroom=headroom)
    return BatchHolding(card if plan.on_card else torch.device("cpu"), plan.element_bytes, plan.float32_bytes,
                        plan.device_reason, plan.width_reason)


def _width_reason(width: int, widths: Tuple[int, ...], where: str, room: Optional[int] = None) -> str:
    room_text = "" if room is None else f" ({int(room) / 1024 ** 3:.2f} GiB of room)"
    if len(widths) == 1:
        return f"float32 rows are standardised once, {where}{room_text}"
    if width == 4:
        return f"float16 rows widened to float32 and standardised once: float32 fits {where}{room_text}"
    return (f"float16 rows held as stored and standardised at every use: float32 does not fit {where}, "
            f"float16 does{room_text}")


def standardisation_terms(
    mean: torch.Tensor, std: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(mean, divisor, degenerate) on `device`, as `standardise_batch` takes them."""
    degenerate = std.abs() <= 1e-6
    divisor = torch.where(degenerate, torch.ones_like(std), std)
    return mean.to(device), divisor.to(device), degenerate.to(device)


def standardise_batch(
    packed: torch.Tensor,
    mask: torch.Tensor,
    terms: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """One float32 batch, standardised IN PLACE: `packed` is consumed. THE ONE PLACE this arithmetic
    lives, so a batch standardised once at build time and one standardised at use are the same
    numbers (`batch_holding`).

    In place because it now runs on every epoch for a held-raw batch: out of place it made five
    batch-sized temporaries and measured 355 ms per 512 MiB batch on 8 CPU threads, against 16 ms for
    the forward it feeds. The operations and their order are unchanged, so the values are too.
    """
    return _standardise_in_place(packed, mask.unsqueeze(-1), terms)


def _standardise_in_place(
    values: torch.Tensor,
    keep: torch.Tensor,
    terms: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """The arithmetic itself, on any run of slots whose last dim is `d_model`. `keep` broadcasts over
    the channels (`mask.unsqueeze(-1)` for a batch, a `(slots, 1)` column for a run of slots). Every
    operation is elementwise, so applying it to a slice gives the same bits as applying it whole."""
    mean_d, divisor_d, degenerate_d = terms
    values.sub_(mean_d).div_(divisor_d)
    # Skipped when no channel is degenerate, which is then a fill of nothing (14 ms per batch).
    if bool(degenerate_d.any()):
        values.masked_fill_(degenerate_d, 0.0)
    # Padding must not carry a standardised value: `combine` masks it, but a non-zero
    # pad would still reach `max` through a mask bug, and zero is the honest filler.
    return values.mul_(keep)


# ⚠ HOW MUCH OF A HELD-RAW BATCH IS WIDENED AND STANDARDISED AT A TIME (ENH-006 G25). Done over a whole
# 512 MiB batch, the widen, subtract, divide and mask are four passes through main memory, every epoch:
# arm A's training stage took 9,184 s against arm B's 1,848 s for the same epochs on the node. Over a
# 4 MiB run of slots the later passes hit the cache. Measured on a 512 MiB batch (d 4096) with the
# forward and backward included, against 29 ms for a float32 batch standardised once: a fresh copy
# per use 150 ms (8 threads) / 298 ms (1 thread); 4 MiB runs in a reused buffer 81 / 87 ms.
STANDARDISE_CHUNK_BYTES: int = 4 * 1024 ** 2


def _standardise_run(
    out: torch.Tensor,
    raw: torch.Tensor,
    keep: torch.Tensor,
    terms: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Widen one run of raw slots into `out` and standardise it there."""
    out.copy_(raw)
    _standardise_in_place(out, keep, terms)


def standardise_into(
    out: torch.Tensor,
    raw: torch.Tensor,
    mask: torch.Tensor,
    terms: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """`raw` (float16) widened into `out` (float32, same shape) and standardised, `STANDARDISE_CHUNK_BYTES`
    of slots at a time on the CPU. The same bits as `standardise_batch(raw.to(torch.float32), ...)`:
    the same elementwise operations in the same order. On a card the batch goes whole, since a loop of
    small kernels would cost more than the memory passes it saves."""
    d_model = int(raw.shape[-1])
    flat_out = out.view(-1, d_model)
    flat_raw = raw.reshape(-1, d_model)
    flat_keep = mask.reshape(-1, 1)
    if out.device.type != "cpu":
        _standardise_run(flat_out, flat_raw, flat_keep, terms)
        return out
    step = max(1, STANDARDISE_CHUNK_BYTES // (d_model * 4))
    for start in range(0, flat_raw.shape[0], step):
        stop = start + step
        _standardise_run(flat_out[start:stop], flat_raw[start:stop], flat_keep[start:stop], terms)
    return out


def raw_batch_scratch(
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor, List[int]]], device: torch.device
) -> Optional[torch.Tensor]:
    """One float32 buffer as large as the largest held-raw batch, or None when none is held raw.

    `train_rule` widens every held-raw batch into it, every epoch (`ready_batch`), instead of
    allocating a fresh float32 copy per use. Reusing it is safe because each batch's backward runs
    before the next batch is widened, and the predictions run under `no_grad`."""
    sizes = [x.numel() for x, _mask, _indices in batches if x.dtype == torch.float16]
    if not sizes:
        return None
    return torch.empty(max(sizes), dtype=torch.float32, device=device)


def ready_batch(
    x: torch.Tensor,
    mask: torch.Tensor,
    terms: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The standardised float32 input for one batch from `_standardised_batches`.

    A float32 batch was standardised when it was built. A float16 batch is the raw stored rows
    (`batch_holding`) and is standardised here: into `scratch` when given (`raw_batch_scratch`, what
    `train_rule` passes), else into a fresh copy. The returned tensor is then a view of `scratch`,
    valid until the next call. Those are the only two dtypes the builder produces; anything else is
    refused rather than guessed at.
    """
    if x.dtype == torch.float32:
        return x
    if x.dtype == torch.float16:
        if scratch is None:
            return standardise_batch(x.to(torch.float32), mask, terms)
        return standardise_into(scratch[: x.numel()].view(x.shape), x, mask, terms)
    raise TypeError(f"a probe training batch is float32 or float16, not {x.dtype}")


def _standardised_batches(
    rows: Sequence[np.ndarray],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    *,
    budget_bytes: int = DEFAULT_BATCH_BYTES,
    max_bytes: Optional[int] = None,
    hold_raw: Optional[bool] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
    """Length-bucketed padded batches, STANDARDISED ONCE.

    Standardisation does not depend on the learned parameters, so doing it per forward —
    as the previous loop did — read and wrote three tensors the size of the whole dataset
    on every one of up to 400 epochs. Applied once here instead, which is also what makes
    the stored batches directly usable as the model's input.

    A degenerate channel (std at or below the floor) is zeroed rather than divided by it,
    matching `ProbeHead.standardise` exactly — dividing by a 1e-6 floor amplified a 0.001
    drift into 1000.0 and made the probe fire on a channel carrying no signal.

    ⚠ EXCEPT FLOAT16 ROWS HELD RAW (`hold_raw`), which stay as stored and are standardised at use: half
    the memory, the same numbers, a slower epoch. Read every batch through `ready_batch`. The caller
    passes `batch_holding`'s decision; None, for a caller that made none, holds float16 rows raw (G24's
    behaviour, the cautious width). Float32 rows cannot be held raw.
    """
    widths = [int(np.asarray(chunk).shape[0]) for chunk in rows]
    d_model = int(np.asarray(rows[0]).shape[1])
    float16 = stored_as_float16(rows)
    raw = float16 if hold_raw is None else bool(hold_raw)
    if raw and not float16:
        raise ValueError("only float16 rows can be held raw; these rows are not float16")
    element_bytes = 2 if raw else 4
    forecast = forecast_batches(
        widths, d_model, budget_bytes=budget_bytes, element_bytes=element_bytes
    )
    ceiling = _ram_ceiling(device, max_bytes)
    if ceiling is not None and forecast["bytes"] > ceiling:
        raise ProbeTrainingTooLarge(
            f"the standardised batches would need {forecast['gib']} GiB "
            f"({forecast['padded_slots']:,} padded slots x {d_model} dims x {element_bytes} B) for "
            f"{forecast['real_tokens']:,} real tokens over {forecast['rows']:,} rows, "
            f"above the {ceiling / 1024 ** 3:.1f} GiB this host can give them now. Free host "
            f"memory or provision more, or reduce max_length, narrow the scope, or train on "
            f"fewer rows."
        )

    terms = standardisation_terms(mean, std, device)
    held_dtype, numpy_dtype = (
        (torch.float16, np.float16) if raw else (torch.float32, np.float32)
    )

    batches: List[Tuple[torch.Tensor, torch.Tensor, List[int]]] = []
    for indices in plan_length_buckets(
        widths, budget_slots=slots_for(d_model, budget_bytes)
    ):
        width = max(widths[i] for i in indices)
        packed = torch.zeros((len(indices), width, d_model), dtype=held_dtype, device=device)
        mask = torch.zeros((len(indices), width), dtype=torch.bool, device=device)
        for position, row_index in enumerate(indices):
            array = np.asarray(rows[row_index], dtype=numpy_dtype)
            if array.shape[0] == 0:
                continue
            packed[position, : array.shape[0]] = torch.from_numpy(array).to(device)
            mask[position, : array.shape[0]] = True
        batches.append((packed if raw else standardise_batch(packed, mask, terms), mask, list(indices)))
    return batches


#: Host memory kept free beside the batches while they are built: one batch being built (packed, standardised
#: and masked at once, about 3 x `DEFAULT_BATCH_BYTES`) and the rendered rows' Python lists (about 70 B per
#: token). `probe_run_memory.RAM_TRANSIENT_BYTES` is this figure.
BATCH_BUILD_TRANSIENT_BYTES: int = 2 * 1024 ** 3


def float32_batch_bytes(
    row_sets: Sequence[Sequence[np.ndarray]], *, budget_bytes: int = DEFAULT_BATCH_BYTES
) -> int:
    """The batches of every set in `row_sets` at float32: what `batch_holding` weighs."""
    total = 0
    for rows in row_sets:
        if rows:
            widths = [int(np.asarray(r).shape[0]) for r in rows]
            d_model = int(np.asarray(rows[0]).shape[1])
            total += forecast_batches(widths, d_model, budget_bytes=budget_bytes, element_bytes=4)["bytes"]
    return total


def standalone_hold_raw(
    train_rows: Sequence[np.ndarray],
    val_rows: Sequence[np.ndarray],
    device: torch.device,
    *,
    budget_bytes: int = DEFAULT_BATCH_BYTES,
) -> bool:
    """`batch_holding`'s width for a `train_rule` that builds its own batches on `device` (the SAE variant,
    tests). The device is the caller's; on a card, host memory is no room for them."""
    if not stored_as_float16(train_rows):
        return False
    on_card = getattr(device, "type", None) == "cuda"
    free = int(torch.cuda.mem_get_info(device)[0]) if on_card else None
    return batch_holding(
        storage_bytes=2,
        float32_bytes=float32_batch_bytes([train_rows, val_rows], budget_bytes=budget_bytes),
        card=device if on_card else None,
        card_free=free,
        host_room=0 if on_card else host_ram_usable_bytes() - BATCH_BUILD_TRANSIENT_BYTES,
    ).hold_raw


def train_rule(
    rule: str,
    train_rows: Sequence[np.ndarray],
    train_labels: Sequence[int],
    val_rows: Sequence[np.ndarray],
    val_labels: Sequence[int],
    *,
    tau: float = 1.0,
    window: int = 16,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = 1337,
    device: Optional[torch.device] = None,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
    prebuilt: Optional[
        Tuple[torch.Tensor, torch.Tensor, List[Any], List[Any]]
    ] = None,
    on_epoch: Optional[Callable[[int, int], None]] = None,
    hold_raw: Optional[bool] = None,
) -> TrainedRule:
    """Fit `w`, `b` (and `q` for `attention`) by BCE on the rule's AGGREGATE.

    `hold_raw` applies only when the batches are built here (no `prebuilt`): None asks `batch_holding`
    (`standalone_hold_raw`), True or False forces the width.

    `on_epoch(epoch, epochs)` is told after every epoch — what a live run's training stage reports
    (`epochs` is the cap; early stopping can end sooner). It reads nothing the fit depends on.

    `epochs` counts FULL-BATCH STEPS. See `DEFAULT_EPOCHS` for why the first
    configuration of these numbers could not converge at all.

    EARLY STOPPING IS ON VALIDATION AUROC, NOT VALIDATION LOSS. The reported metric is
    AUROC, and loss and AUROC do not have the same argmin — stopping on loss can hand
    back a probe that ranks worse than one seen three epochs earlier, under a number
    nobody looks at.

    THE BEST STATE IS RESTORED, not merely remembered. Returning the LAST epoch's
    weights while reporting the BEST epoch's AUROC would report a number the returned
    probe does not achieve.
    """
    from sklearn.metrics import roc_auc_score

    from ..ml.probe_monitor_model import combine, rule_parameters

    if not train_rows:
        raise ValueError("no training rows")
    device = device or torch.device("cpu")
    torch.manual_seed(seed)

    # ⚠ `prebuilt` EXISTS BECAUSE AN R-RULE RUN BUILT THE SAME TENSORS R TIMES. The statistics
    # and the batches depend only on `(train_rows, budget)` — none of which varies by rule — so a
    # four-rule run rebuilt ~16.6 GiB of standardised activations four times from the same memmap.
    #
    # Optional rather than required, so `train_rule` remains callable on its own: its unit tests
    # and any other caller pass rows and get batches built here, exactly as before. A refactor
    # that forced every caller to hoist would be a larger change for the same saving.
    #
    # ⚠ THE STATISTICS MUST COME FROM TRAIN ONLY, and hoisting is where that could quietly stop
    # being true — computing them over train+validation would leak the validation distribution's
    # scale into the model, and it would still train and still look fine.
    # `test_training_refactors_change_nothing` pins both halves: bit-identical weights under
    # reuse, and a fixture that can tell train-only statistics from train+val ones.
    if prebuilt is not None:
        mean, std, train_batches, val_batches = prebuilt
    else:
        mean, std = _standardisation(train_rows)
        if hold_raw is None:
            hold_raw = standalone_hold_raw(train_rows, val_rows, device, budget_bytes=batch_bytes)
        # ⚠ LENGTH-BUCKETED BATCHES, NOT ONE PADDED TENSOR. See `plan_length_buckets`: the
        # single-tensor version needed 154 GB for 16 GB of data and was OOMKilled on the node.
        train_batches = _standardised_batches(
            train_rows, mean, std, device, budget_bytes=batch_bytes, hold_raw=hold_raw
        )
        val_batches = (
            _standardised_batches(val_rows, mean, std, device, budget_bytes=batch_bytes, hold_raw=hold_raw)
            if val_rows else []
        )
    # Batches held as stored float16 are standardised as they are used (`ready_batch`), with the
    # same statistics the float32 path applied once at build time.
    terms = standardisation_terms(mean, std, device)
    # One float32 buffer for every held-raw batch, every epoch (ENH-006 G25), not a copy per use.
    scratch = raw_batch_scratch([*train_batches, *val_batches], device)
    y_train = torch.tensor([float(v) for v in train_labels], dtype=torch.float32, device=device)
    y_val = np.asarray(list(val_labels)) if val_rows else None

    d_model = int(train_batches[0][0].shape[-1])
    weight = torch.zeros(d_model, device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)
    parameters = [weight, bias]
    query: Optional[torch.Tensor] = None
    if rule == "attention":
        # A small seeded draw, made on the CPU — see `attention_query_init` for why not zero and
        # why not on the card.
        query = attention_query_init(d_model, device, seed).requires_grad_(True)
        parameters.append(query)

    # ⚠ THE DECAY REACHES THE BIAS AND THE QUERY, DELIBERATELY LEFT THAT WAY — MEASURED.
    #
    # Standard practice excludes non-weight parameters from weight decay: decaying a bias pulls
    # the decision boundary toward the prior rather than regularising capacity, and decaying an
    # attention query pulls softmax toward uniform, which IS mean pooling. This group contains
    # all of them, which looks like an oversight and is worth not "fixing" twice.
    #
    # Measured 2026-09-29, same seed and data, only the decay group changing:
    #
    #     positive rate   rule        with decay   weight-only
    #     50%             mean          0.9483       0.9483
    #     50%             attention     0.9389       0.9389
    #     10%             mean          0.8769       0.8769
    #     10%             attention     0.7790       0.7790
    #
    # Identical to four decimals, same best epoch, same |bias|. Two structural reasons:
    # `_standardisation` centres every channel, so the optimal intercept sits near zero (|bias|
    # 0.002-0.15 across those runs); and early stopping fires at epoch 3-14, so decay has very
    # few steps to act.
    #
    # It would matter given UNCENTRED inputs or a genuinely large optimal intercept — neither of
    # which this pipeline produces, because standardisation is not optional here. Split the group
    # if that ever changes; until then the split buys nothing and the measurement is the reason.
    optimiser = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    # ⚠ `reduction="sum"`, DIVIDED BY THE ROW COUNT. That is what makes accumulating the
    # gradient over batches EXACTLY the full-batch gradient rather than an approximation:
    # sum(per-batch sums) / n is the mean loss, and the gradient of a sum is the sum of
    # the gradients. A per-batch `mean` would weight a short batch as heavily as a long
    # one, which is a different objective — and the epoch count, patience and learning
    # rate here were tuned for full batches.
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    def aggregate(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`x` is ALREADY standardised — see `_standardised_batches`.

        It used to be standardised here, on every forward, which read and wrote three
        tensors the size of the whole dataset up to 400 times.
        """
        token_scores = x @ weight + bias
        logits = x @ query if rule == "attention" else None
        return combine(
            rule, token_scores, mask=mask, attention_logits=logits, tau=tau, window=window
        )

    def predict(batches: Sequence[Tuple[torch.Tensor, torch.Tensor, List[int]]]) -> np.ndarray:
        """Predictions back IN ROW ORDER. The batches are sorted by length, so returning
        them in batch order would silently pair every prediction with another row's
        label — an AUROC computed on a permutation, which looks like a bad probe rather
        than like a bug."""
        total = sum(len(indices) for _x, _m, indices in batches)
        out = np.zeros(total, dtype=np.float64)
        with torch.no_grad():
            for x, mask, indices in batches:
                values = (
                    aggregate(ready_batch(x, mask, terms, scratch), mask).detach().cpu().numpy()
                )
                out[np.asarray(indices)] = values
        return out

    best_auroc: Optional[float] = None
    best_state: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = None
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    since_improvement = 0
    epochs_run = 0

    n_train = float(len(train_labels))
    for epoch in range(1, max(1, epochs) + 1):
        epochs_run = epoch
        optimiser.zero_grad()
        epoch_loss = 0.0
        for x, mask, indices in train_batches:
            targets = y_train[torch.as_tensor(indices, device=device)]
            loss = (
                loss_fn(aggregate(ready_batch(x, mask, terms, scratch), mask), targets) / n_train
            )
            loss.backward()
            epoch_loss += float(loss.item())
        optimiser.step()

        val_auroc: Optional[float] = None
        if val_batches and y_val is not None and len(set(y_val.tolist())) > 1:
            val_auroc = float(roc_auc_score(y_val, predict(val_batches)))
        history.append({"epoch": epoch, "loss": epoch_loss, "val_auroc": val_auroc})
        if on_epoch is not None:
            on_epoch(epoch, max(1, epochs))

        if val_auroc is None:
            continue
        if best_auroc is None or val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            best_state = (
                weight.detach().clone(),
                bias.detach().clone(),
                query.detach().clone() if query is not None else None,
            )
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                break

    if best_state is not None:
        final_weight, final_bias, final_query = best_state
    else:
        final_weight = weight.detach().clone()
        final_bias = bias.detach().clone()
        final_query = query.detach().clone() if query is not None else None

    return TrainedRule(
        rule=rule,
        weight=final_weight.cpu(),
        bias=float(final_bias.cpu().item()),
        mean=mean,
        std=std,
        attention_query=final_query.cpu() if final_query is not None else None,
        rule_params=rule_parameters(rule, tau=tau, window=window),
        val_auroc=best_auroc,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        history=history,
    )


def select_sae_features(
    features_by_row: Sequence[np.ndarray],
    labels: Sequence[int],
    *,
    k: int,
) -> np.ndarray:
    """The k SAE features that separate the classes best, ranked on TRAIN ONLY.

    `score_j = |mean_pos_j - mean_neg_j| / (pooled_std_j + 1e-6)` over row-pooled
    features — a standardised class-mean difference, which is the planner default
    (P-note in the FPRD) and needs no fitting.

    ⚠ TRAIN ONLY, AND THE PITFALL IS SPECIFIC. Ranking on validation rows leaks them
    into the model's STRUCTURE — which features exist at all — so the validation AUROC
    that then decides early stopping is no longer held out, and the number reported for
    the probe is optimistic by an amount nobody can estimate afterwards.

    Returns indices sorted ASCENDING, so the basis order is stable across runs and an
    exported definition's feature list is comparable between two probes.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    pooled = np.stack(
        [np.asarray(rows, dtype=np.float32).mean(axis=0) for rows in features_by_row]
    )
    y = np.asarray(labels)
    if len(set(y.tolist())) < 2:
        raise ValueError("cannot rank features with a single class")
    positives = pooled[y == 1]
    negatives = pooled[y == 0]
    spread = np.sqrt((positives.var(axis=0) + negatives.var(axis=0)) / 2.0)
    score = np.abs(positives.mean(axis=0) - negatives.mean(axis=0)) / (spread + 1e-6)
    if k >= score.shape[0]:
        return np.arange(score.shape[0])
    chosen = np.argsort(score)[-k:]
    return np.sort(chosen)


@dataclass
class Calibration:
    threshold: Optional[float]
    target_fpr: float
    realised_fpr: float
    source: str
    n_negatives: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "target_fpr": self.target_fpr,
            "realised_fpr": self.realised_fpr,
            "threshold_source": self.source,
            "n_negatives": self.n_negatives,
        }


def calibrate(
    negative_scores: Sequence[float],
    *,
    target_fpr: float,
    source: str,
) -> Calibration:
    """The threshold to serve at, from NEGATIVES ALONE, plus what it really spends.

    Negatives alone, on purpose: a false-positive rate is a property of the negative
    distribution, and involving positives would make the operating point depend on the
    prevalence in whichever set happened to be used.

    ⚠ THE REALISED FPR IS RECORDED BECAUSE IT IS USUALLY NOT THE TARGET. With 100
    negatives the achievable rates are multiples of 0.01, so a 1% request lands on 0%
    or 1% and nothing in between. Reporting the TARGET as though it were achieved is
    the honest-absence-into-silent-lie shape this estate has already shipped once.

    `threshold is None` means fire on NOTHING, which is a real operating point at a 1%
    budget with few negatives — and is not `inf`, which cannot be serialised.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    scores = sorted((float(s) for s in negative_scores), reverse=True)
    n = len(scores)
    if n == 0:
        raise ValueError("cannot calibrate without negatives")

    # How many false positives the budget allows, rounded DOWN: spending more than the
    # budget to hit it exactly is the wrong direction for a monitor.
    allowed = int(target_fpr * n)
    if allowed <= 0:
        # Above every negative: fire on nothing. `None`, not `inf`.
        return Calibration(
            threshold=None,
            target_fpr=target_fpr,
            realised_fpr=0.0,
            source=source,
            n_negatives=n,
        )
    # The (allowed)-th highest negative is the first score we are willing to admit;
    # threshold just above it would admit allowed-1, so the threshold IS that score
    # and the realised rate counts the ties with it.
    threshold = scores[allowed - 1]
    realised = sum(1 for s in scores if s >= threshold) / n
    return Calibration(
        threshold=threshold,
        target_fpr=target_fpr,
        realised_fpr=realised,
        source=source,
        n_negatives=n,
    )


def head_from_trained(
    trained: TrainedRule, layer: int
) -> Any:
    """A `ProbeHead` carrying everything needed to score — including the layer.

    One constructor, so a training path and a serving path cannot disagree about
    which fields a head has. `layer` is part of the identity: the same weights over a
    different layer are a different detector.
    """
    from ..ml.probe_monitor_model import ProbeHead

    return ProbeHead(
        weight=trained.weight,
        bias=trained.bias,
        mean=trained.mean,
        std=trained.std,
        attention_query=trained.attention_query,
        layer=layer,
    )
