"""What a probe training run needs from host memory, worked out before it spends GPU time.

ENH-006 D116 G10. The refusal in `_standardised_batches` comes at the training stage, after pooled
capture and token capture have already spent hours. Everything it needs to know is known much
earlier: a row's token count comes from the tokenizer and the width from the model's config.

THE ARITHMETIC, each term read from the code it describes:

    batches    padded slots of the train set + the validation set x d_model x the HELD bytes, which
               `batch_holding` decides, the same function the run asks per layer (ENH-006 G25): 4 B
               (float32, standardised once) whenever float32 fits, 2 B (a 16-bit model's stored float16,
               standardised at every use, ENH-006 G24) only when it does not. Fits means: on the card
               when the model's card has room, else within `room` below. Built once per layer and
               dropped before the next (`execute_probe_run`), so ONE layer's worth is held however many
               layers are chosen. `forecast_batches`.
    pooled     rows x d_model x 4 B x 2 poolings x layers swept. Dropped once the layers are chosen,
               so it counts against the batches as the LARGER of the two, not their sum.
    transient  one batch while it is built (packed, standardised and masked: about three times the
               512 MiB budget) plus the rendered rows' Python lists.
    baseline   the worker's own heap. An allowance, not a measurement.

    needed     max(batches, pooled) + transient + baseline

    token file tokens x d_model x storage bytes, ONE PER CHOSEN LAYER. Written to disk and read through
               `np.memmap`, and `execute_probe_run`'s rows are views of it, so it is file-backed page
               cache the kernel can drop. `top_n_layers` therefore costs DISK, not required RAM. The
               `by_top_n` table gives both figures, because a reader sizing a run wants to see the
               resident total if nothing is evicted.

    usable     MemAvailable x (1 - RAM_SAFETY_MARGIN) - what other live jobs have reserved, the rule
               `gpu_job_claim.host_ram_refusal` already applies to a job starting beside another.

    held       anonymous memory the worker that runs the job already holds (`gpu_worker_memory`). The
               run executes in that worker and reuses it, and MemAvailable already excludes it, so the
               run needs only `needed - held` MORE from the host (ENH-006 G14: an idle worker's 19.2
               GiB was counted as unavailable, then reused by the next run, which grew the worker only
               to 31.5 GiB against a 30.9 GiB forecast).

    fits       max(0, needed - held) <= usable

    room       usable + held - transient - baseline: what the batches may take with `fits` still true
               (when the pooled vectors are not the larger term). `batch_holding` widens a 16-bit
               model's batches when their float32 figure is within it, so "float32 fits" is exactly
               "this forecast, counted at 4 B, fits".

`checked: False` always carries a `reason`, and never reads as "fits". The check that cannot run says
so, and the run's own check at the rendering stage still applies.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from . import probe_monitor_trainer as trainer
from .probe_monitor_trainer import (
    BATCH_BUILD_TRANSIENT_BYTES,
    HELD_DESCRIPTION,
    forecast_batches,
    host_ram_usable_bytes,
)
from .resource_config import ResourceConfig

logger = logging.getLogger(__name__)

GIB = 1024 ** 3

#: One batch being built (`_standardised_batches` holds packed, standardised and mask at once, about
#: 3 x `DEFAULT_BATCH_BYTES`) and the rendered rows' Python lists (about 70 B per token). The trainer's
#: figure, so the run's room at the training stage and this forecast keep the same margin.
RAM_TRANSIENT_BYTES: int = BATCH_BUILD_TRANSIENT_BYTES

#: The worker's own heap: Python, torch and the imported libraries. An ALLOWANCE, not a measurement. At
#: the rendering stage MemAvailable already excludes it, so that check counts it twice, the cautious way.
RAM_WORKER_BASELINE_BYTES: int = 4 * GIB

#: The `top_n_layers` values `by_top_n` always reports, beside the one asked for.
REPORTED_TOP_N = (1, 2, 3)


def _gib(value: float) -> float:
    return round(value / GIB, 2)


def _config_value(config: Any, key: str) -> Optional[int]:
    """`key` from a config dict or object, or from its `text_config`. None when neither states it:
    a model's width is a fact about the checkpoint and is never assumed."""
    for source in (config, _read(config, "text_config")):
        value = _read(source, key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _read(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def unchecked(reason: str) -> Dict[str, Any]:
    """The answer when no forecast could be made. Says so; never reads as `fits`."""
    return {"checked": False, "fits": None, "reason": reason}


def other_reserves_bytes(db: Any, exclude_holder: Optional[str] = None) -> int:
    """Host RAM other live jobs have reserved and not yet allocated (`host_ram_refusal`'s rule), less
    `exclude_holder`'s own: a run asking at its training stage has declared its own reserve by then."""
    from .gpu_job_claim import HOST_RESERVE_MB_BY_KIND, holder_kind
    from .gpu_leases import host_reserves

    total_mb = 0
    for holder, reserve_mb in host_reserves(db).items():
        if exclude_holder is not None and holder == exclude_holder:
            continue
        total_mb += reserve_mb if reserve_mb is not None else HOST_RESERVE_MB_BY_KIND.get(holder_kind(holder), 0)
    return int(total_mb) * 2 ** 20


def batch_room_now(db: Any, available_bytes: Optional[int] = None) -> int:
    """The room a run's batches have in host memory at its training stage (`batch_holding`'s `host_room`).

    Live MemAvailable less the safety margin, less what OTHER jobs have reserved (this run's own reserve is
    for these very batches), less the build's transient. The worker's heap is already outside MemAvailable,
    so nothing is credited for it here. A failure to read the reserves counts none, as the guard does.
    """
    from .gpu_job_claim import current_claim

    claim = current_claim()
    try:
        reserved = other_reserves_bytes(db, exclude_holder=getattr(claim, "holder", None))
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning("could not read other jobs' host reserves for the batch width", exc_info=True)
        reserved = 0
    return host_ram_usable_bytes(available_bytes, reserved) - RAM_TRANSIENT_BYTES


def forecast_memory(
    *,
    train_widths: Sequence[int],
    val_widths: Sequence[int],
    d_model: int,
    swept_layers: int,
    top_n_layers: int,
    storage_bytes: int,
    based_on: str,
    available_bytes: Optional[int] = None,
    reserved_bytes: int = 0,
    card: Any = None,
    card_free: Optional[int] = None,
    held_bytes: int = 0,
) -> Dict[str, Any]:
    """The forecast and its verdict. Pure once `available_bytes` is given. See the module docstring.

    `card` and `card_free` are the model's CUDA device and its free memory when the run has one (the run's
    own check); None for the preflight, which cannot know the card, so its batches are counted in host memory.
    """
    # Sized at float32 first: the width the batches are HELD at is `batch_holding`'s decision, below.
    train = forecast_batches(list(train_widths), d_model, element_bytes=4)
    val = forecast_batches(list(val_widths), d_model, element_bytes=4)
    rows = train["rows"] + val["rows"]
    tokens = train["real_tokens"] + val["real_tokens"]
    slots = train["padded_slots"] + val["padded_slots"]
    float32_batches = train["bytes"] + val["bytes"]

    if available_bytes is None:
        import psutil

        available_bytes = int(psutil.virtual_memory().available)
    usable = host_ram_usable_bytes(available_bytes, reserved_bytes)
    held = max(0, int(held_bytes or 0))
    room = usable + held - RAM_TRANSIENT_BYTES - RAM_WORKER_BASELINE_BYTES
    holding = trainer.batch_holding(
        storage_bytes=storage_bytes, float32_bytes=float32_batches, card=card, card_free=card_free,
        host_room=room,
    )
    element_bytes = holding.element_bytes
    batches = holding.batch_bytes
    batches_on_card = holding.device.type == "cuda"
    batches_in_ram = 0 if batches_on_card else batches
    pooled = rows * d_model * 4 * 2 * swept_layers
    needed = max(batches_in_ram, pooled) + RAM_TRANSIENT_BYTES + RAM_WORKER_BASELINE_BYTES
    token_file = tokens * d_model * storage_bytes
    additional = max(0, needed - held)
    fits = additional <= usable

    by_top_n = {
        str(n): {
            "needed_gib": _gib(needed),
            "token_files_gib": _gib(n * token_file),
            "resident_if_nothing_is_evicted_gib": _gib(needed + n * token_file),
        }
        for n in sorted({*REPORTED_TOP_N, int(top_n_layers)})
    }
    arithmetic = (
        f"needed {_gib(needed)} GiB = max(batches {_gib(batches_in_ram)}, pooled {_gib(pooled)}) "
        f"+ transient {_gib(RAM_TRANSIENT_BYTES)} + worker {_gib(RAM_WORKER_BASELINE_BYTES)}. "
        f"Batches: {slots:,} padded slots x {d_model} dims x {element_bytes} B, "
        f"{HELD_DESCRIPTION[element_bytes]} ({holding.width_reason})"
        f"{' (on the card, so not in RAM)' if batches_on_card else ''}. "
        f"Pooled: {rows:,} rows x {d_model} dims x 4 B x 2 x {swept_layers} layers. "
        f"usable {_gib(usable)} GiB = available {_gib(available_bytes)} x {1 - ResourceConfig.RAM_SAFETY_MARGIN:g} "
        f"- reserved {_gib(reserved_bytes)}. "
        f"The worker that runs it already holds {_gib(held)} GiB it will reuse, "
        f"so it needs {_gib(additional)} GiB more."
    )
    answer: Dict[str, Any] = {
        "checked": True,
        "fits": fits,
        "based_on": based_on,
        "needed_gib": _gib(needed),
        "needed_bytes": needed,
        "usable_gib": _gib(usable),
        "usable_bytes": usable,
        "available_gib": _gib(available_bytes),
        "reserved_gib": _gib(reserved_bytes),
        "worker_held_gib": _gib(held),
        "worker_held_bytes": held,
        "additional_gib": _gib(additional),
        "additional_bytes": additional,
        "batches_gib": _gib(batches),
        "batches_bytes": batches,
        "batches_on_card": bool(batches_on_card),
        # ENH-006 G25: the width the batches are held at, and why; what the reservation and the run count.
        "batches_element_bytes": element_bytes,
        "batches_held": HELD_DESCRIPTION[element_bytes],
        "batches_held_reason": holding.width_reason,
        "float32_batches_bytes": float32_batches,
        "storage_bytes": int(storage_bytes),
        "batch_room_bytes": room,
        "train_batches_gib": _gib(train["bytes"] * element_bytes // 4),
        "validation_batches_gib": _gib(val["bytes"] * element_bytes // 4),
        "pooled_gib": _gib(pooled),
        "pooled_bytes": pooled,
        "token_file_per_layer_gib": _gib(token_file),
        "token_file_per_layer_bytes": token_file,
        "rows": rows,
        "real_tokens": tokens,
        "padded_slots": slots,
        "d_model": d_model,
        "swept_layers": swept_layers,
        "top_n_layers": int(top_n_layers),
        "by_top_n": by_top_n,
        "arithmetic": arithmetic,
    }
    answer["message"] = (
        f"this run would need {_gib(needed)} GiB of host memory, {_gib(additional)} GiB beyond what its "
        f"worker already holds, and {_gib(usable)} GiB is usable now"
        f" ({based_on}). {arithmetic}"
        + ("" if fits else " Free host memory or provision more, or reduce max_length, narrow the "
                           "scope, or train on fewer rows, then submit again.")
    )
    return answer


def template_prefix_skipped(rendered: Sequence[Any], scope: str) -> Optional[int]:
    """G38: the template-prefix positions the run leaves out of its scope's window, over every row;
    `None` when the rows were rendered without the skip (a pre-G38 config)."""
    from .probe_monitor_capture import _row_mask
    from .probe_monitor_render import effective_scope

    if not any(getattr(row, "skip_template_prefix", False) for row in rendered):
        return None
    return int(sum(
        sum(row._scope_mask(effective_scope(row, scope))) - sum(_row_mask(row, scope))
        for row in rendered
        if not (scope == "last_user" and not row.role_mask_reliable)
    ))


def _with_prefix_skip(answer: Dict[str, Any], skipped: Optional[int]) -> Dict[str, Any]:
    """The forecast, stating the skipped positions when the run skips any (G38). Every width the
    forecast sized is already net of them (`count_scored_tokens` reads the same masks)."""
    if skipped is not None:
        answer = {**answer, "template_prefix_skipped": skipped}
    return answer


def forecast_for_run(
    db: Any,
    model: Any,
    rendered: Sequence[Any],
    train_index: Sequence[int],
    val_index: Sequence[int],
    *,
    scope: str,
    swept_layers: int,
    top_n_layers: int,
    storage_bytes: int,
) -> Dict[str, Any]:
    """The run's own check, on its own rendered rows, before pooled capture spends any GPU time.

    A failure here is reported as `checked: False` and does not fail the run: `_standardised_batches`
    still refuses at build time, and a forecast's bug must not cost a good run.
    """
    try:
        from .probe_monitor_capture import count_scored_tokens

        d_model = _config_value(getattr(model, "config", None), "hidden_size")
        if d_model is None:
            return unchecked("the loaded model's config states no hidden_size")
        _total, offsets = count_scored_tokens(rendered, scope)
        widths = np.diff(np.asarray(offsets)).tolist()
        train_widths = [widths[i] for i in train_index]
        val_widths = [widths[i] for i in val_index]
        card, card_free = _card_state(model)
        from .gpu_worker_memory import anon_bytes

        return _with_prefix_skip(forecast_memory(
            train_widths=train_widths, val_widths=val_widths, d_model=d_model, swept_layers=swept_layers,
            top_n_layers=top_n_layers, storage_bytes=storage_bytes, reserved_bytes=other_reserves_bytes(db),
            card=card, card_free=card_free,
            # This check runs inside the worker that runs the job: what it holds is reused.
            held_bytes=anon_bytes() or 0,
            based_on=f"every one of {len(widths):,} rows as rendered and tokenised for this run",
        ), template_prefix_skipped(rendered, scope))
    except Exception as exc:  # noqa: BLE001 - see the docstring
        logger.warning("probe run memory forecast failed: %s", exc)
        return unchecked(f"the forecast raised {type(exc).__name__}: {exc}")


def _card_state(model: Any) -> tuple:
    """The model's CUDA device and its free memory now, or (None, None): what the run's per-layer decision
    (`batch_holding`) reads at the training stage, read here for the forecast's."""
    import torch

    from .gpu_stage_memory import cuda_device_of

    device = cuda_device_of(model)
    if device is None:
        return None, None
    return device, int(torch.cuda.mem_get_info(device)[0])


def forecast_for_request(
    model_id: str, train_view_id: str, config: Dict[str, Any], gpu: Optional[str] = "auto"
) -> Dict[str, Any]:
    """The preflight: the same rows the worker will read, tokenised on the CPU. No GPU, no row, no task.

    Blocking (dataset read, tokenizer load, one render per row), so the route runs it in a thread. It
    uses the run's own helpers so the split, the scope and `max_length` cannot differ from the run's.
    """
    from ..core.database import SyncSessionLocal
    from ..models.model import Model
    from ..models.probe_monitor import ProbeMonitorDataset
    from . import probe_monitor_run as run_module
    from .gpu_worker_memory import held_bytes, memory_dir
    from .probe_monitor_capture import count_scored_tokens

    session = SyncSessionLocal()
    try:
        model = session.query(Model).filter(Model.id == model_id).first()
        view = session.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == train_view_id).first()
        if model is None or view is None:
            return unchecked(f"model {model_id} or view {train_view_id} was not found")
        architecture = model.architecture_config or {}
        d_model = _config_value(architecture, "hidden_size")
        if d_model is None:
            return unchecked(f"model {model_id} records no hidden_size, so its width is unknown")
        explicit = config.get("layers")
        depth = _config_value(architecture, "num_hidden_layers")
        if not explicit and depth is None:
            return unchecked(f"model {model_id} records no num_hidden_layers, so the sweep is unknown")
        swept = run_module.resolve_layers(config, depth if depth is not None else 1 + max(explicit))

        context = run_module.context_from_row(SimpleNamespace(id="forecast", config=config))
        tokenizer = run_module._load_tokenizer_for_run(SimpleNamespace(model_id=model_id))
        examples, rendered, _summary, _labels = run_module._prepare_examples(
            session, view, context, tokenizer, role="train"
        )
        train_index, val_index = run_module._split_indices(examples, context)
        _total, offsets = count_scored_tokens(rendered, context.scope)
        widths = np.diff(np.asarray(offsets)).tolist()
        floats = "float32" in str(config.get("dtype") or "") or "FP32" in str(getattr(model, "quantization", ""))
        skipped = template_prefix_skipped(rendered, context.scope)
        return _with_prefix_skip(forecast_memory(
            train_widths=[widths[i] for i in train_index],
            val_widths=[widths[i] for i in val_index],
            d_model=d_model,
            swept_layers=len(swept),
            top_n_layers=context.top_n_layers,
            storage_bytes=4 if floats else 2,
            reserved_bytes=other_reserves_bytes(session),
            # The workers run in another container: their figures come from the supervisor.
            held_bytes=held_bytes(memory_dir(), gpu),
            based_on=(
                f"every one of {len(widths):,} rows of view {train_view_id}, tokenised on the CPU with the "
                f"model's tokenizer at max_length {context.max_length}, scope {context.scope}"
                + (
                    f", without the {skipped:,} template-prefix positions the run skips"
                    if skipped is not None else ""
                )
                + ". Memory is as the host reads now; the run checks again when it starts. Assumes the "
                "batches stay in host memory, since which card a run gets is not known yet"
            ),
        ), skipped)
    except Exception as exc:  # noqa: BLE001 - a preflight that cannot run says so
        logger.info("probe run preflight could not run: %s", exc)
        return unchecked(f"{type(exc).__name__}: {exc}")
    finally:
        session.close()
