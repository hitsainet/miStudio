"""Forward loops for probe monitors: pooled, token-level and streaming (032 FR-5–FR-8).

Three modes over ONE loop shape, because they differ only in what they keep:

  `capture_pooled`   every swept layer in ONE forward pass, two d-vectors per row
  `capture_tokens`   the selected layers only, scored positions, fp16 to a memmap
  `forward_scores`   scores computed IN the loop, activations discarded

⚠ IT IS ITS OWN LOOP, NOT `extract_activations` (FTDD T1). That function stores EVERY
position — which is how 48% of every SAE batch here turned out to be padding — and it
carries no labels and no role mask. A probe needs the scored positions only, pooled
outputs for layer selection, and a streaming scorer for evaluation. Reusing it would
mean storing ~200x the data and then masking it afterwards.

⚠ `resid_post` MEANS THE DECODER LAYER'S OUTPUT, and it is resolved through
`get_hookable_module(layer, "residual", structure)` so this path moves with training,
circuit capture and J-Lens. Until 2026-09-12 that resolved to the post-attention NORM
on LFM2 and Llama, so every SAE on this estate learned a normalised pre-MLP signal
that miLLM never reads. A capture at the wrong point is SILENT — the numbers look
fine — so the point is never chosen locally.

⚠ THE MASK IS THE ROLE MASK TIMES THE ATTENTION MASK. Padding must never reach a
pooled mean, and neither must a token outside the probe's scope. Both exclusions are
applied in one place (`_row_mask`) so a caller cannot honour one and forget the other.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..ml.forward_hooks import HookManager, HookType
from .probe_monitor_render import RenderedExample

logger = logging.getLogger(__name__)

#: How many tokens a micro-batch may carry. Batching by TOKENS rather than by rows
#: is what makes a 4k-token evaluation input and a 40-token training row cost the
#: same peak memory; batching by rows makes the longest row in the set decide whether
#: the pass OOMs.
DEFAULT_TOKEN_BUDGET = 16_384


class ProbeCaptureOOM(RuntimeError):
    """Out of memory twice — once at full budget and once at half (FTID I4)."""


@dataclass
class PooledCapture:
    """Mean and last-scored-token vectors, per row per layer.

    Two poolings because layer selection compares them: `mean` is what a `mean` rule
    would see and `last` is what `last` would see, and which wins is a property of the
    concept, not a constant. Storing only one would pre-decide the sweep.
    """

    #: layer → (n_rows, d_model) float32 on CPU
    mean: Dict[int, torch.Tensor]
    last: Dict[int, torch.Tensor]
    #: Rows that contributed NO scored token — an empty mask. Kept as indices rather
    #: than silently zero-filled, because a zero vector is a legitimate activation and
    #: an all-zero row would train as if it were data.
    empty_rows: List[int] = field(default_factory=list)


@dataclass
class TokenCapture:
    """Scored-token activations on disk, plus where each row's tokens start."""

    path: Path
    #: (n_rows + 1,) int64 — row i occupies [offsets[i], offsets[i + 1]).
    offsets: np.ndarray
    d_model: int
    layer: int
    dtype: str = "float16"

    def open_memmap(self) -> np.memmap:
        total = int(self.offsets[-1])
        return np.memmap(
            self.path, dtype=np.dtype(self.dtype), mode="r", shape=(total, self.d_model)
        )

    def row(self, index: int) -> np.ndarray:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.open_memmap()[start:end]


def _row_mask(example: RenderedExample, scope: str) -> List[bool]:
    """Which positions of ONE row may be scored. The single place scope is applied.

    A row whose role mask is unreliable falls back to `all` here rather than raising,
    because the fallback is a recorded property of the row (FTDD §12) and refusing it
    would discard data for a template quirk. The count of such rows is reported by the
    render pass, so the fallback is visible rather than silent.
    """
    effective = scope if example.role_mask_reliable else "all"
    return example.scored_mask(effective)


def plan_batches(
    examples: Sequence[RenderedExample], *, token_budget: int = DEFAULT_TOKEN_BUDGET
) -> List[List[int]]:
    """Group row indices into micro-batches under a TOKEN budget.

    A row longer than the whole budget still gets its own batch rather than being
    dropped or truncated here: truncation is the render's decision and is recorded on
    the row, and silently skipping a long row would remove exactly the hardest
    examples from an evaluation.
    """
    if token_budget < 1:
        raise ValueError(f"token_budget must be at least 1, got {token_budget}")
    batches: List[List[int]] = []
    current: List[int] = []
    widest = 0
    for index, example in enumerate(examples):
        length = len(example.input_ids)
        # Padding makes the batch cost (rows x widest), not the sum of lengths.
        candidate_width = max(widest, length)
        if current and candidate_width * (len(current) + 1) > token_budget:
            batches.append(current)
            current, widest = [index], length
        else:
            current.append(index)
            widest = candidate_width
    if current:
        batches.append(current)
    return batches


def _pad_batch(
    examples: Sequence[RenderedExample],
    indices: Sequence[int],
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ids and the attention mask. Returns (input_ids, attention_mask)."""
    widths = [len(examples[i].input_ids) for i in indices]
    width = max(widths)
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(indices), width), dtype=torch.long)
    for position, index in enumerate(indices):
        row = examples[index].input_ids
        ids[position, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention[position, : len(row)] = 1
    return ids.to(device), attention.to(device)


def _scored_mask_tensor(
    examples: Sequence[RenderedExample],
    indices: Sequence[int],
    scope: str,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """(batch, width) bool: scored positions, already intersected with real tokens."""
    mask = torch.zeros((len(indices), width), dtype=torch.bool)
    for position, index in enumerate(indices):
        flags = _row_mask(examples[index], scope)
        mask[position, : len(flags)] = torch.tensor(flags, dtype=torch.bool)
    return mask.to(device)


def _hook_key(layer: int) -> str:
    return f"layer_{layer}_residual"


def _run_forward(
    model: Any,
    hooks: HookManager,
    ids: torch.Tensor,
    attention: torch.Tensor,
) -> None:
    hooks.activations.clear()
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attention)


def _layer_activation(hooks: HookManager, layer: int) -> torch.Tensor:
    key = _hook_key(layer)
    captured = hooks.activations.get(key)
    if not captured:
        raise RuntimeError(
            f"no activation was captured at {key}. The hook did not fire, which means "
            f"the layer index is out of range for this model or the hook was registered "
            f"on a different module than the one the forward pass ran"
        )
    # One entry per forward pass; the loop clears between batches.
    return captured[-1]


def _align(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """`tensor` on `like`'s device — the one place the capture loops reconcile the two.

    ⚠ THE ACTIVATION IS NOT WHERE THE MODEL IS. `HookManager`'s forward hook stores
    `output.detach().cpu()`, deliberately: an extraction that kept every swept layer's
    activation in VRAM would not fit. So on a GPU run the model's parameters are on
    cuda:0 while every activation these loops receive is on the CPU.

    All three loops built their mask and index tensors from the MODEL's device, and the
    first Stage 1 acceptance run died mid-sweep on

        RuntimeError: Expected all tensors to be on the same device, but found at least
        two devices, cuda:0 and cpu!

    after loading the model and rendering 8,000 rows. The unit tests could not have
    caught it: they run a real tiny Llama on the CPU, where the model's device and the
    hook's device are the same object, so the two agreed BY CONSTRUCTION — the fixture
    trap this repo keeps paying for.

    The activation is the authority, not the model: it is the tensor that cannot be
    moved cheaply (it is the big one), and moving a bool mask to meet it is free.
    """
    return tensor if tensor.device == like.device else tensor.to(like.device)


def capture_pooled(
    model: Any,
    examples: Sequence[RenderedExample],
    layers: Sequence[int],
    *,
    scope: str = "all",
    architecture: str = "",
    pad_id: int = 0,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    device: Optional[torch.device] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> PooledCapture:
    """ONE forward pass per batch, hooking every swept layer at once.

    This is what makes a full layer sweep affordable: the cost is one pass over the
    data, not one pass per layer. For the reference set (8,000 rows x 7 layers x 2
    poolings x 4096 dims x fp32) the result is about 1.8 GB on CPU.
    """
    if not examples:
        raise ValueError("nothing to capture")
    if not layers:
        raise ValueError("no layers to capture")

    device = device or next(model.parameters()).device
    hooks = HookManager(model)
    hooks.register_hooks(list(layers), [HookType.RESIDUAL], architecture)
    registered = len(hooks.hook_names)
    if registered != len(layers):
        raise RuntimeError(
            # The wording is DISTINCT from `_layer_activation`'s fallback on purpose: a
            # test matching "out of range" passed against either, so a mutation removing
            # this guard survived while the run merely failed later and less clearly.
            f"hook registration mismatch: asked for {len(layers)} residual hooks and "
            f"registered {registered}. `register_hooks` warns and SKIPS a layer beyond "
            f"the model's depth, so the sweep would silently cover fewer layers than the "
            f"run's config claims"
        )

    n = len(examples)
    means: Dict[int, torch.Tensor] = {}
    lasts: Dict[int, torch.Tensor] = {}
    empty_rows: List[int] = []
    try:
        batches = plan_batches(examples, token_budget=token_budget)
        for batch_number, indices in enumerate(batches):
            ids, attention = _pad_batch(examples, indices, pad_id, device)
            mask = _scored_mask_tensor(examples, indices, scope, ids.shape[1], device)
            _run_forward(model, hooks, ids, attention)
            for layer in layers:
                activation = _layer_activation(hooks, layer).to(torch.float32)
                d_model = activation.shape[-1]
                if layer not in means:
                    means[layer] = torch.zeros((n, d_model), dtype=torch.float32)
                    lasts[layer] = torch.zeros((n, d_model), dtype=torch.float32)
                # The hook hands activations back on the CPU whatever card the model is
                # on; see `_align`. Everything below combines with the activation, so it
                # is reconciled to the activation's device, not the model's.
                row_mask = _align(mask, activation)
                counts = row_mask.sum(dim=1)
                # A masked mean: sum over scored positions / their count. Dividing by
                # the row WIDTH instead would make the value depend on the padding,
                # which is the defect that cost this estate an entire SAE corpus.
                summed = (activation * row_mask.unsqueeze(-1)).sum(dim=1)
                safe = counts.clamp_min(1).unsqueeze(-1).to(torch.float32)
                pooled_mean = summed / safe
                # The LAST scored index, searched from the right so left padding (or a
                # trailing generation prompt) cannot make it land on a pad.
                flipped = torch.flip(row_mask.to(torch.int8), dims=[1])
                found = flipped.argmax(dim=1)
                last_index = row_mask.shape[1] - 1 - found
                pooled_last = activation[
                    torch.arange(len(indices), device=activation.device), last_index
                ]
                for position, row_index in enumerate(indices):
                    if counts[position].item() == 0:
                        if layer == layers[0]:
                            empty_rows.append(row_index)
                        continue
                    means[layer][row_index] = pooled_mean[position].detach().cpu()
                    lasts[layer][row_index] = pooled_last[position].detach().cpu()
            if progress is not None:
                progress(batch_number + 1, len(batches))
    finally:
        hooks.remove_hooks()

    if empty_rows:
        logger.warning(
            "probe_monitor: %d of %d rows contributed no scored token under scope %r; "
            "they are reported rather than zero-filled, because a zero vector is a "
            "legitimate activation and would train as if it were data",
            len(empty_rows), n, scope,
        )
    return PooledCapture(mean=means, last=lasts, empty_rows=empty_rows)


def count_scored_tokens(
    examples: Sequence[RenderedExample], scope: str = "all"
) -> Tuple[int, np.ndarray]:
    """(total, offsets). Pre-sizes the memmap without a forward pass (FTID §13)."""
    per_row = [sum(_row_mask(example, scope)) for example in examples]
    offsets = np.zeros(len(per_row) + 1, dtype=np.int64)
    np.cumsum(per_row, out=offsets[1:])
    return int(offsets[-1]), offsets


def capture_tokens(
    model: Any,
    examples: Sequence[RenderedExample],
    layer: int,
    destination: Path,
    *,
    scope: str = "all",
    architecture: str = "",
    pad_id: int = 0,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    device: Optional[torch.device] = None,
    dtype: str = "float16",
    progress: Optional[Callable[[int, int], None]] = None,
) -> TokenCapture:
    """Scored-token activations at ONE layer, written to a pre-sized fp16 memmap.

    Pre-sizing from `count_scored_tokens` rather than appending matters for more than
    speed: a growing file has no single moment at which its length is known, so a
    crash leaves a file whose size cannot be checked against the offsets. Here the
    offsets are computed first and the file is exactly as long as they say.
    """
    total, offsets = count_scored_tokens(examples, scope)
    if total == 0:
        raise ValueError(
            f"no scored tokens at all under scope {scope!r}; every row's mask is empty, "
            f"so there is nothing to train on"
        )

    device = device or next(model.parameters()).device
    hooks = HookManager(model)
    hooks.register_hooks([layer], [HookType.RESIDUAL], architecture)

    d_model: Optional[int] = None
    memmap: Optional[np.memmap] = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        batches = plan_batches(examples, token_budget=token_budget)
        for batch_number, indices in enumerate(batches):
            ids, attention = _pad_batch(examples, indices, pad_id, device)
            mask = _scored_mask_tensor(examples, indices, scope, ids.shape[1], device)
            _run_forward(model, hooks, ids, attention)
            activation = _layer_activation(hooks, layer)
            if d_model is None:
                d_model = int(activation.shape[-1])
                memmap = np.memmap(
                    destination, dtype=np.dtype(dtype), mode="w+", shape=(total, d_model)
                )
            # Indexing a CPU activation with a CUDA bool mask is the same defect as
            # the pooled loop's, raised by `index_select` rather than by the multiply.
            row_mask = _align(mask, activation)
            for position, row_index in enumerate(indices):
                keep = row_mask[position]
                if not bool(keep.any()):
                    continue
                rows = activation[position][keep].to(torch.float32).detach().cpu().numpy()
                start = int(offsets[row_index])
                memmap[start : start + rows.shape[0]] = rows.astype(dtype)
            if progress is not None:
                progress(batch_number + 1, len(batches))
    finally:
        hooks.remove_hooks()
        if memmap is not None:
            memmap.flush()
            del memmap

    if d_model is None:
        raise RuntimeError("no batch produced an activation, so nothing was written")
    return TokenCapture(
        path=destination, offsets=offsets, d_model=d_model, layer=layer, dtype=dtype
    )


@dataclass
class ScoredRow:
    """One row's scores. `aggregate` is the number a rule produced."""

    index: int
    token_scores: List[float]
    aggregate: float
    n_scored: int


def forward_scores(
    model: Any,
    examples: Sequence[RenderedExample],
    head: Any,
    *,
    rule: str,
    rule_params: Optional[Dict[str, Any]] = None,
    scope: str = "all",
    architecture: str = "",
    pad_id: int = 0,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    device: Optional[torch.device] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> List[ScoredRow]:
    """Score rows WITHOUT storing activations — the one scoring path (FTID §11).

    Evaluation, offline scoring and 033's test-vector generation all call this, so the
    vectors an exported definition carries come from the same code that produced the
    metrics. Two scoring paths would be two detectors, and only one of them measured.

    ⚠ `encoder` IS WHAT KEEPS THAT TRUE FOR A k-SPARSE PROBE, and its absence was a defect.
    An SAE-variant probe is TRAINED on SAE features — `train_sae_variant` encodes the capture
    and slices the chosen k columns — but was EVALUATED here against the raw residual, so a
    128-dimensional head met a 2,048-dimensional activation:

        ValueError: activations are d_model=2048 but this probe is d_model=128

    Found by Stage 2 acceptance (`pmr_5d81ad3b81f2`), 98.5% of the way through a 24-minute
    evaluation, after the dense probe's five sets had already succeeded. The k-sparse variant
    was trainable and not evaluable, so it could never leave rung 0.

    The transform is a HOOK INTO THIS FUNCTION rather than a second scoring function, because
    a separate SAE scoring path would be a second detector and only one of them would be
    measured. `encoder` maps `(batch, tokens, d_model)` to `(batch, tokens, k)` and everything
    downstream — the head, `combine`, the mask — is unchanged.
    """
    from ..ml.probe_monitor_model import combine

    if not examples:
        return []
    device = device or next(model.parameters()).device
    layer = head.layer if getattr(head, "layer", None) is not None else -1
    if layer is None or layer < 0:
        raise ValueError(
            "the probe head carries no layer, so there is nothing to hook; a probe is "
            "a readout of ONE layer and the layer is part of its identity"
        )

    hooks = HookManager(model)
    hooks.register_hooks([layer], [HookType.RESIDUAL], architecture)
    results: List[Optional[ScoredRow]] = [None] * len(examples)
    try:
        batches = plan_batches(examples, token_budget=token_budget)
        for batch_number, indices in enumerate(batches):
            ids, attention = _pad_batch(examples, indices, pad_id, device)
            mask = _scored_mask_tensor(examples, indices, scope, ids.shape[1], device)
            _run_forward(model, hooks, ids, attention)
            activation = _layer_activation(hooks, layer)
            hidden = activation.to(torch.float32)
            if encoder is not None:
                hidden = encoder(hidden)
            # The head is trained on the CPU (`train_rule` defaults there) and the hook
            # returns the activation on the CPU, so the mask is the one tensor still on
            # the model's device — and `combine` multiplies it by the scores.
            row_mask = _align(mask, hidden)
            token_scores = head.token_scores(hidden)
            # `attention` weights tokens by `softmax(q · ẑ_t)` while the VALUE stays
            # `w · ẑ_t`, so the logits are a separate tensor from the scores. Passing
            # the scores as the logits would silently turn `attention` into `softmax`
            # at tau=1 — a different detector under the same name.
            params = dict(rule_params or {})
            logits = head.attention_logits(hidden) if rule == "attention" else None
            aggregates = combine(
                rule,
                token_scores,
                mask=row_mask,
                attention_logits=logits,
                **{k: v for k, v in params.items() if k in ("tau", "window")},
            )
            for position, row_index in enumerate(indices):
                keep = row_mask[position]
                kept = token_scores[position][keep].detach().cpu().tolist()
                results[row_index] = ScoredRow(
                    index=row_index,
                    token_scores=[float(v) for v in kept],
                    aggregate=float(aggregates[position].item()),
                    n_scored=len(kept),
                )
            if progress is not None:
                progress(batch_number + 1, len(batches))
    finally:
        hooks.remove_hooks()

    missing = [i for i, row in enumerate(results) if row is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} rows were never scored (first: {missing[0]}); a partial "
            f"score set silently shrinks the evaluation sample"
        )
    return [row for row in results if row is not None]


def with_oom_retry(run: Callable[[int], Any], *, token_budget: int) -> Any:
    """Run `run(budget)`, halving the budget ONCE on OOM, then failing (FTID I4).

    Matches the existing extraction behaviour. The second failure names the card,
    because "CUDA out of memory" without it is unactionable on a two-card node.
    """
    try:
        return run(token_budget)
    except torch.cuda.OutOfMemoryError:
        halved = max(1, token_budget // 2)
        logger.warning(
            "probe_monitor: out of memory at a %d-token budget; retrying once at %d",
            token_budget, halved,
        )
        torch.cuda.empty_cache()
        try:
            return run(halved)
        except torch.cuda.OutOfMemoryError as exc:
            name = "unknown GPU"
            try:
                name = torch.cuda.get_device_name(torch.cuda.current_device())
            except Exception:  # noqa: BLE001 - naming must not mask the OOM
                pass
            raise ProbeCaptureOOM(
                f"out of memory twice on {name}, at token budgets {token_budget} and "
                f"{halved}. Reduce max_length or the number of swept layers."
            ) from exc
