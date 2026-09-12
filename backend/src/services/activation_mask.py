"""Which cached activation positions are real tokens, and which are padding.

WHY THIS EXISTS. Tokenization defaults to ``padding="max_length"``
(``schemas/dataset.py:200-206``), so every document is padded out to the full
window. Extraction builds an attention mask, passes it to the model, and then
saves the activation tensor **whole** — the mask is dropped
(``activation_service.py:774,799`` vs ``:802-841``) and is not recorded in
``metadata.json``. Training then flattens ``(N, seq_len, d)`` and samples
uniformly over every position, so the residual stream at PAD positions becomes
SAE training data.

That is not a small effect. Measured on this estate 2026-09-11, real-token
fraction per corpus: OpenWebText 87.7%, the Pile 68.2%, hard-negatives 31.9%,
Bloomberg 3.3% — and a run over all four is roughly **48% padding**.

(Deliberately no line numbers below. An earlier version of this docstring cited
them and every one was invalidated by this arc's own edits to the file it was
describing — while remaining the sole written justification for the row-alignment
assumption.)

The mask is recoverable without re-extracting anything: ``metadata.json``
records ``dataset_path``, the tokenized Arrow dataset keeps its
``attention_mask`` column, and extraction selects a **prefix** of that dataset
(``ActivationService._load_dataset`` does ``select(range(max_samples))``), so
row *i* of the ``.npy`` is row *i* of the tokenization.

FAILS LOUD, NEVER SILENT. When the mask cannot be recovered the caller is told
so and warns; it does not quietly fall back to training on padding, because
that is the defect this module exists to remove.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Written beside the activations by new extractions, so the Arrow lookup below
#: is only ever needed for the pre-existing estate.
MASK_FILENAME = "attention_mask.npy"


class MaskUnavailable(Exception):
    """The real-token mask could not be recovered for an extraction."""


def _from_sidecar(extraction_dir: Path, num_samples: int, seq_len: int) -> Optional[np.ndarray]:
    path = extraction_dir / MASK_FILENAME
    if not path.exists():
        return None
    mask = np.load(path, mmap_mode="r")
    return _conform(np.asarray(mask, dtype=bool), num_samples, seq_len, str(path))


def _fingerprint_matches(ds, metadata) -> bool:
    """Is this directory plausibly the tokenization the extraction read?

    NARROWER THAN IT WAS, because the first version refused correct recoveries.

    The hazard: `metadata.json` records only `dataset_path`, and the tokenized
    directory name used to omit `max_length`, so a 512 tokenization and a 2048
    one occupied the SAME path. A 512 extraction whose directory was later
    overwritten by a 2048 tokenization still row-matches, and trimming that mask
    to its first 512 columns returns a mask describing DIFFERENT rows.

    But a width mismatch is ALSO the signature of a legitimate truncated
    extraction — and the first version of this guard refused those, sending
    training back to `PADDING NOT MASKED` and training on padding: the exact
    defect the arc exists to remove. Round 3 caught it, along with a test of
    mine that pinned the refusal.

    So this only refuses the case it can actually distinguish: a tokenization
    NARROWER than what was saved. Extraction can truncate (stored < source) but
    can never invent columns, so stored > source means the directory is not the
    one that produced these activations. A wider source is accepted and trimmed.

    Note the collision this guards against is now prevented at the source —
    `tokenized_dataset_path` embeds `max_length` — so this covers only
    directories written before that landed.
    """
    recorded = metadata.get("seq_len") or metadata.get("max_length")
    if not recorded:
        return True  # pre-existing extraction; the caller still shape-checks
    try:
        width = len(ds[0]["attention_mask"])
    except Exception:  # noqa: BLE001
        return True
    if width >= recorded:
        # Equal, or wider because extraction truncated. Both legitimate.
        return True
    logger.warning(
        "Mask recovery: the tokenization at %r is %d wide but the extraction "
        "saved %d columns. Extraction cannot widen a mask, so this is not the "
        "directory that produced these activations — refusing rather than "
        "zero-extending a mask that may describe other rows.",
        getattr(ds, "cache_files", "?"), width, recorded,
    )
    return False


def _from_tokenized_dataset(
    dataset_path: str, num_samples: int, seq_len: int, metadata=None
) -> Optional[np.ndarray]:
    if not dataset_path:
        return None
    if not Path(dataset_path).exists():
        logger.warning(
            "Mask recovery: dataset_path %r from metadata.json does not exist", dataset_path
        )
        return None

    from datasets import load_from_disk

    ds = load_from_disk(dataset_path)
    if "attention_mask" not in ds.column_names:
        logger.warning(
            "Mask recovery: %r has no attention_mask column (has %s)",
            dataset_path, ds.column_names,
        )
        return None
    if len(ds) < num_samples:
        # The extraction cannot have read more rows than the dataset holds; if
        # it did, this is not the dataset it was built from.
        logger.warning(
            "Mask recovery: %r has %d rows but the extraction holds %d — refusing "
            "to align them", dataset_path, len(ds), num_samples,
        )
        return None

    if metadata is not None and not _fingerprint_matches(ds, metadata):
        return None

    mask = np.asarray(ds.select(range(num_samples))["attention_mask"], dtype=bool)
    return _conform(mask, num_samples, seq_len, dataset_path)


def _conform(
    mask: np.ndarray, num_samples: int, seq_len: int, source: str
) -> Optional[np.ndarray]:
    """Make a recovered mask match the activation tensor's shape, or refuse.

    Extraction truncates at its own cap before saving, so the stored `seq_len`
    can be SHORTER than the tokenization's `max_length`. Trimming that is
    correct — those positions were never run through the model. A mask that is
    shorter than the tensor is the opposite situation and cannot be repaired by
    guessing, so it is refused.
    """
    if mask.ndim != 2:
        logger.warning("Mask recovery: %s has ndim=%d, expected 2", source, mask.ndim)
        return None
    if mask.shape[0] != num_samples:
        logger.warning(
            "Mask recovery: %s has %d rows, activations have %d",
            source, mask.shape[0], num_samples,
        )
        return None
    if mask.shape[1] == seq_len:
        return mask
    if mask.shape[1] > seq_len:
        # Extraction truncated; the tail was never encoded.
        return mask[:, :seq_len]
    logger.warning(
        "Mask recovery: %s is %d wide but activations are %d — cannot extend a mask",
        source, mask.shape[1], seq_len,
    )
    return None


def load_valid_mask(
    extraction_dir: Path, num_samples: int, seq_len: int
) -> Tuple[Optional[np.ndarray], str]:
    """Recover the real-token mask for one extraction.

    Returns ``(mask, source)`` where `mask` is a bool array shaped
    ``(num_samples, seq_len)`` — True where the position is a real token — or
    ``(None, reason)`` when it cannot be recovered.
    """
    extraction_dir = Path(extraction_dir)

    try:
        mask = _from_sidecar(extraction_dir, num_samples, seq_len)
        if mask is not None:
            return mask, f"sidecar {MASK_FILENAME}"
    except Exception as exc:  # noqa: BLE001 - a bad sidecar must not kill the run
        logger.warning("Mask recovery: sidecar unreadable in %s (%s)", extraction_dir, exc)

    metadata_path = extraction_dir / "metadata.json"
    if not metadata_path.exists():
        return None, f"no {MASK_FILENAME} and no metadata.json in {extraction_dir}"

    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return None, f"metadata.json unreadable ({exc})"

    dataset_path = metadata.get("dataset_path") or ""
    try:
        mask = _from_tokenized_dataset(dataset_path, num_samples, seq_len, metadata)
    except Exception as exc:  # noqa: BLE001
        return None, f"tokenized dataset {dataset_path!r} unreadable ({exc})"

    if mask is not None:
        return mask, f"tokenized dataset {dataset_path}"
    return None, f"no {MASK_FILENAME}, and {dataset_path!r} yielded no usable mask"


def valid_flat_indices(mask: np.ndarray) -> np.ndarray:
    """Flat indices of real tokens, in the same order the tensor flattens.

    ``(N, seq_len) -> (N*seq_len,)`` row-major, matching
    ``activations.reshape(-1, d)``.
    """
    return np.flatnonzero(mask.reshape(-1))


def resolve_flat_indices(
    valid_flat: Optional[np.ndarray], local_indices: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Turn a selection into positions in one file's flat ``(N*seq_len)`` space.

    ``local_indices`` index whatever space `total_tokens` counts. When a mask
    was recovered that space is the VALID subset, so they must be mapped through
    ``valid_flat`` — using them directly still yields in-range, real-looking
    activations, just the **wrong ones**, which nothing downstream could detect.

    Returns ``None`` to mean "every position", which the caller may serve with a
    bulk reshape instead of a gather.
    """
    if valid_flat is None:
        return local_indices
    if local_indices is None:
        return valid_flat
    return valid_flat[local_indices]


def gather_tokens(mmap, flat_indices: np.ndarray, chunk_size: int = 200) -> np.ndarray:
    """Read the given flat positions out of a ``(N, seq_len, d)`` memmap.

    Reads contiguous sample ranges so the access pattern stays sequential — a
    naive per-index read of a 69 GB file is orders of magnitude slower.

    The returned rows are in ascending flat-index order, which is the order
    ``reshape(-1, d)`` would produce for the same positions.
    """
    n, s, d = mmap.shape
    flat_indices = np.asarray(flat_indices)
    if flat_indices.size == 0:
        return np.empty((0, d), dtype=np.float32)

    # Ascending order is what makes the sample ranges contiguous, and is also
    # the order the caller's bulk-reshape path would produce.
    flat_indices = np.sort(flat_indices)
    sample_indices = flat_indices // s
    token_indices = flat_indices % s

    # SLICE, DO NOT SCAN.
    #
    # This used to do `token_indices[sample_indices == s_idx]` inside the
    # per-sample loop — a full pass over every index for every sample, i.e.
    # quadratic. Measured: 0.003 s at 200 samples, 0.117 s at 2,000, 10.07 s at
    # 20,000. A real extraction is ~2x10^5 samples and ~3x10^8 indices, which is
    # hours per (layer, extraction) against the seconds the pre-mask bulk
    # reshape took. `flat_indices` is sorted above, so `sample_indices` is
    # non-decreasing and each sample's slice is contiguous: searchsorted finds
    # the boundaries in O(log n) and the whole gather becomes linear.
    # `sample_indices` is non-decreasing (flat_indices was sorted above), so the
    # group boundaries are just the positions where it changes. `np.unique(...,
    # return_index=True)` would also work but performs a full stable argsort and
    # allocates an extra int64 array the size of the input — ~2.4 GB at the
    # 3x10^8 indices a real extraction reaches. `flatnonzero(diff)` is one pass
    # and one small array.
    boundaries = np.flatnonzero(np.diff(sample_indices)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(sample_indices)]))
    unique_samples = sample_indices[starts]

    chunk_parts = []
    for ci in range(0, len(unique_samples), chunk_size):
        sample_batch = unique_samples[ci:ci + chunk_size]
        lo, hi = ci, min(ci + chunk_size, len(unique_samples))
        s_min, s_max = sample_batch[0], sample_batch[-1]
        block = mmap[s_min:s_max + 1]
        for k in range(lo, hi):
            s_idx = unique_samples[k]
            toks = token_indices[starts[k]:ends[k]]
            chunk_parts.append(block[s_idx - s_min, toks].astype(np.float32))
    return np.concatenate(chunk_parts, axis=0)


def split_documents(
    flat_indices: np.ndarray,
    seq_len: int,
    holdout_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split token positions into train and held-out, BY DOCUMENT.

    WHY BY DOCUMENT AND NOT BY TOKEN. Adjacent positions in one document are
    massively correlated — they share a prefix and often a topic. A token-level
    split therefore puts near-duplicates of the evaluation set into training,
    and the held-out number flatters the model for reasons that have nothing to
    do with generalisation. Splitting whole documents is the only version of
    this that measures anything.

    Every number this project reports about an SAE today is in-sample: batches
    are drawn with replacement from a single pool, and there is no split at all.

    Returns ``(train_flat, eval_flat)``. A fraction of 0 gives everything to
    training and an empty eval set, which is the historical behaviour.
    """
    flat_indices = np.asarray(flat_indices)
    if not (0.0 <= holdout_fraction < 1.0):
        raise ValueError(
            f"holdout_fraction must be in [0, 1), got {holdout_fraction}"
        )
    if holdout_fraction == 0.0 or flat_indices.size == 0:
        return flat_indices, np.empty(0, dtype=flat_indices.dtype)

    documents = np.unique(flat_indices // seq_len)
    n_hold = int(round(len(documents) * holdout_fraction))
    if n_hold == 0:
        # The caller asked for a split and the corpus is too small to give one.
        # Returning an empty eval set silently would report in-sample numbers as
        # if they were held out.
        raise ValueError(
            f"holdout_fraction={holdout_fraction} over {len(documents)} documents "
            f"reserves nothing; use more documents or a larger fraction"
        )

    rng = np.random.RandomState(seed)
    held = set(rng.choice(documents, size=n_hold, replace=False).tolist())
    is_held = np.fromiter(
        ((idx // seq_len) in held for idx in flat_indices),
        dtype=bool,
        count=flat_indices.size,
    )
    return flat_indices[~is_held], flat_indices[is_held]


def build_padded_batch(
    input_ids_rows,
    source_masks,
    max_length: int,
    pad_token_id: int,
):
    """Pad a batch and derive its REAL attention masks and token counts.

    ONE implementation for both extraction paths. They had two, and the second
    one — written in the same commit that diagnosed the first — reproduced the
    identical defect: it built masks as ``[1] * len(input_ids)`` over rows that
    arrive from Arrow **already padded** to ``max_length``, so the mask was
    all-ones and masked nothing.

    Returns ``(padded_ids, attention_masks, real_lengths, non_prefix_rows,
    missing_mask_rows)``:

    * ``attention_masks`` is what the model should attend to.
    * ``real_lengths`` is what the sidecar records — the COUNT of real tokens,
      not the row width.
    * ``non_prefix_rows`` lists rows whose real tokens are not a right-hand
      prefix; a length cannot describe those, so the caller must not write a
      length-based sidecar for them.
    * ``missing_mask_rows`` counts rows with no source mask at all, which the
      caller must report rather than silently treat as fully real.
    """
    padded_ids = []
    attention_masks = []
    real_lengths = []
    non_prefix_rows = []
    missing_mask_rows = 0

    for idx, ids in enumerate(input_ids_rows):
        ids = list(ids)
        padding_length = max(0, max_length - len(ids))

        src = source_masks[idx] if idx < len(source_masks) else None
        if src is None:
            missing_mask_rows += 1
            mask = [1] * len(ids)
        else:
            mask = [int(bool(v)) for v in src][: len(ids)]
            if len(mask) < len(ids):
                # A SHORT mask would mark real tokens as padding. Extend with
                # ones rather than zeros: over-including is recoverable, and
                # dropping real text silently is not.
                mask = mask + [1] * (len(ids) - len(mask))

        real = int(sum(mask))
        if mask[:real] != [1] * real:
            non_prefix_rows.append(idx)

        padded_ids.append(ids + [pad_token_id] * padding_length)
        attention_masks.append(mask + [0] * padding_length)
        real_lengths.append(real)

    return padded_ids, attention_masks, real_lengths, non_prefix_rows, missing_mask_rows
