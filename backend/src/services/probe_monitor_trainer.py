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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def select_layers(
    pooled: Any,
    labels: Sequence[int],
    train_index: Sequence[int],
    val_index: Sequence[int],
    *,
    top_n: int = 1,
    seed: int = 1337,
) -> LayerSelection:
    """Fit an L2 logistic regression per (layer, pooling) and rank by VALIDATION AUROC.

    sklearn on CPU, because this is a d-dimensional problem over a few thousand rows
    and a torch loop would be slower and less reproducible. Standardisation is fitted
    on TRAIN and applied to validation — fitting it on everything leaks the validation
    distribution's scale into the model.

    Returns the whole grid. A sweep that records only its argmax cannot be audited for
    a near-tie, and a near-tie is exactly when the chosen layer is arbitrary.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels)
    y_train = y[list(train_index)]
    y_val = y[list(val_index)]

    grid: List[LayerScore] = []
    for pooling in POOLINGS:
        source = getattr(pooled, pooling)
        for layer in sorted(source):
            matrix = source[layer].numpy() if hasattr(source[layer], "numpy") else source[layer]
            x_train = np.asarray(matrix)[list(train_index)]
            x_val = np.asarray(matrix)[list(val_index)]
            score: Optional[float] = None
            if len(set(y_train.tolist())) > 1 and len(set(y_val.tolist())) > 1:
                mu = x_train.mean(axis=0)
                sigma = x_train.std(axis=0)
                sigma = np.where(sigma < 1e-8, 1.0, sigma)
                model = LogisticRegression(max_iter=2000, random_state=seed)
                model.fit((x_train - mu) / sigma, y_train)
                predictions = model.decision_function((x_val - mu) / sigma)
                score = float(roc_auc_score(y_val, predictions))
            grid.append(
                LayerScore(
                    layer=layer,
                    pooling=pooling,
                    val_auroc=score,
                    n_train=len(x_train),
                    n_val=len(x_val),
                )
            )

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

#: Refuse rather than allocate more than this for the standardised batches. The node has
#: 124 GB and is shared with a 16 GB model, five other containers and the page cache that
#: makes the memmap readable at all.
MAX_BATCH_BYTES: int = 48 * 1024 ** 3


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
) -> Dict[str, Any]:
    """What `_standardised_batches` will allocate, before it allocates it.

    An OOM kill is the worst failure mode available here: it takes the worker down
    mid-run, writes no status, and leaves the row claiming to be running until a janitor
    notices 90 minutes later. A forecast that refuses is the same information, an hour
    earlier, with the numbers in the message.
    """
    buckets = plan_length_buckets(widths, budget_slots=slots_for(d_model, budget_bytes))
    padded_slots = sum(len(batch) * max(widths[i] for i in batch) for batch in buckets)
    real_tokens = int(sum(widths))
    bytes_needed = padded_slots * d_model * 4
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
    }


def _standardised_batches(
    rows: Sequence[np.ndarray],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    *,
    budget_bytes: int = DEFAULT_BATCH_BYTES,
    max_bytes: int = MAX_BATCH_BYTES,
) -> List[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
    """Length-bucketed padded batches, STANDARDISED ONCE.

    Standardisation does not depend on the learned parameters, so doing it per forward —
    as the previous loop did — read and wrote three tensors the size of the whole dataset
    on every one of up to 400 epochs. Applied once here instead, which is also what makes
    the stored batches directly usable as the model's input.

    A degenerate channel (std at or below the floor) is zeroed rather than divided by it,
    matching `ProbeHead.standardise` exactly — dividing by a 1e-6 floor amplified a 0.001
    drift into 1000.0 and made the probe fire on a channel carrying no signal.
    """
    widths = [int(np.asarray(chunk).shape[0]) for chunk in rows]
    d_model = int(np.asarray(rows[0]).shape[1])
    forecast = forecast_batches(widths, d_model, budget_bytes=budget_bytes)
    if forecast["bytes"] > max_bytes:
        raise ProbeTrainingTooLarge(
            f"the standardised batches would need {forecast['gib']} GiB "
            f"({forecast['padded_slots']:,} padded slots x {d_model} dims x 4 B) for "
            f"{forecast['real_tokens']:,} real tokens over {forecast['rows']:,} rows, "
            f"above the {max_bytes / 1024 ** 3:.0f} GiB ceiling. Reduce max_length, "
            f"narrow the scope, or train on fewer rows."
        )

    degenerate = std.abs() <= 1e-6
    divisor = torch.where(degenerate, torch.ones_like(std), std)
    mean_d = mean.to(device)
    divisor_d = divisor.to(device)
    degenerate_d = degenerate.to(device)

    batches: List[Tuple[torch.Tensor, torch.Tensor, List[int]]] = []
    for indices in plan_length_buckets(
        widths, budget_slots=slots_for(d_model, budget_bytes)
    ):
        width = max(widths[i] for i in indices)
        packed = torch.zeros((len(indices), width, d_model), dtype=torch.float32, device=device)
        mask = torch.zeros((len(indices), width), dtype=torch.bool, device=device)
        for position, row_index in enumerate(indices):
            array = np.asarray(rows[row_index], dtype=np.float32)
            if array.shape[0] == 0:
                continue
            packed[position, : array.shape[0]] = torch.from_numpy(array).to(device)
            mask[position, : array.shape[0]] = True
        standardised = (packed - mean_d) / divisor_d
        standardised = torch.where(degenerate_d, torch.zeros_like(standardised), standardised)
        # Padding must not carry a standardised value: `combine` masks it, but a non-zero
        # pad would still reach `max` through a mask bug, and zero is the honest filler.
        standardised = standardised * mask.unsqueeze(-1)
        batches.append((standardised, mask, list(indices)))
    return batches


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
) -> TrainedRule:
    """Fit `w`, `b` (and `q` for `attention`) by BCE on the rule's AGGREGATE.

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

    from ..ml.probe_monitor_model import ProbeHead, combine, rule_parameters

    if not train_rows:
        raise ValueError("no training rows")
    device = device or torch.device("cpu")
    torch.manual_seed(seed)

    mean, std = _standardisation(train_rows)
    # ⚠ LENGTH-BUCKETED BATCHES, NOT ONE PADDED TENSOR. See `plan_length_buckets`: the
    # single-tensor version needed 154 GB for 16 GB of data and was OOMKilled on the node.
    train_batches = _standardised_batches(
        train_rows, mean, std, device, budget_bytes=batch_bytes
    )
    y_train = torch.tensor([float(v) for v in train_labels], dtype=torch.float32, device=device)
    if val_rows:
        val_batches = _standardised_batches(
            val_rows, mean, std, device, budget_bytes=batch_bytes
        )
        y_val = np.asarray(list(val_labels))
    else:
        val_batches = []
        y_val = None

    d_model = int(train_batches[0][0].shape[-1])
    weight = torch.zeros(d_model, device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)
    parameters = [weight, bias]
    query: Optional[torch.Tensor] = None
    if rule == "attention":
        # ⚠ A SMALL RANDOM INIT, AND THE REASON I FIRST WROTE HERE WAS FALSE.
        #
        # The comment claimed an all-zero query "makes every softmax weight identical and
        # its gradient symmetric, so `attention` would never differentiate from `mean`".
        # MEASURED, after a mutation to zeros survived the suite: a zero-init query trains
        # perfectly well — 0.9675 validation AUROC against 0.9650 for the random init on
        # the same fixture, with |q|max reaching 1.48. The gradient is not symmetric at
        # zero, because it depends on the SPREAD of the per-token scores, which is nonzero.
        #
        # So the init is a mild convenience (it breaks the tie between identical channels
        # on the first step) and NOT a correctness requirement. It is left random because
        # there is no reason to change it, and the false claim is recorded here rather than
        # quietly deleted — a test written to pin it would have been pinning nothing.
        query = (torch.randn(d_model, device=device) * 0.02).requires_grad_(True)
        parameters.append(query)

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
                values = aggregate(x, mask).detach().cpu().numpy()
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
            loss = loss_fn(aggregate(x, mask), targets) / n_train
            loss.backward()
            epoch_loss += float(loss.item())
        optimiser.step()

        val_auroc: Optional[float] = None
        if val_batches and y_val is not None and len(set(y_val.tolist())) > 1:
            val_auroc = float(roc_auc_score(y_val, predict(val_batches)))
        history.append({"epoch": epoch, "loss": epoch_loss, "val_auroc": val_auroc})

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
