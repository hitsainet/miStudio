"""Annotate, persist and queue J-space feature annotations (024 tasks 4.1, 4.3, 4.4).

* **Two surfaces (BR-015, 4.1).** A feature has a DECODER direction (what it writes into
  the residual stream) and an ENCODER direction (what it reads from it). Read through the
  lens separately, the pair shows the transformation the feature performs; one readout of
  both hides it. For a transcoder the two live on different sides of the MLP, which is the
  case BR-015 names.
* **Persistence (4.3).** Every annotation is upserted into `jlens_feature_annotations`,
  one row per (SAE, feature, surface, lens type), with the label it was compared against
  AS IT WAS — a relabel must not make an old score describe a label it never saw.
* **The sweep (4.4)** runs this over a bounded, contiguous range of one SAE's features in
  a queued CPU task, priced in advance by the `annotation_sweep` cost class.

WEIGHTS ONLY, NO MODEL LOAD. Annotation needs the unembedding, the tokenizer and the
SAE's weights — no forward pass. The original route called `load_for_readout`, which
instantiated the WHOLE model in the API process for one matvec per feature; this reads
W_U from the checkpoint shard the same way the logit lens does
(`analysis_service.load_unembedding_matrix`).

TOKENS ARE DECODED, NOT CONVERTED. The route decoded with `convert_ids_to_tokens`, which
returns tokenizer pieces (`Ġdog`, `▁dog`), so the disagreement queue compared a label's
words against piece strings that can never equal them and every labelled feature scored
near-maximal disagreement — a queue containing the whole dictionary says nothing.
`tokenizer.decode([id])` per id is what `annotate_direction`'s docstring always required.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

SURFACES = ("decoder", "encoder")

#: Disagreement at or above this is flagged into the queue. Unchanged from the route's
#: original rule, so a persisted flag means what the single-annotation response meant.
DISAGREEMENT_THRESHOLD = 0.8

#: Most features one sweep may annotate. A sweep is a CPU matvec per feature over the
#: vocabulary ([n_vocab, d_model]); 4,096 on a 256k vocabulary at d_model 3840 is
#: minutes, and a 32k-feature dictionary is eight bounded sweeps, not one unbounded one.
MAX_SWEEP_FEATURES = 4096

_WEIGHTS_CACHE: Dict[str, Tuple[torch.Tensor, Any]] = {}


def decode_tokens(tokenizer: Any, ids: Sequence[int]) -> List[str]:
    """Each id decoded ON ITS OWN into the text it stands for."""
    return [tokenizer.decode([int(i)]) for i in ids]


def label_words(label: Optional[str]) -> List[str]:
    """A label's words, for comparison with decoded readout tokens."""
    if not label:
        return []
    return [w for w in re.split(r"[^0-9A-Za-z]+", label) if w]


def unembedding_and_tokenizer(model_record: Any) -> Tuple[torch.Tensor, Any]:
    """W_U [n_vocab, d_model] as fp32 on CPU, and the tokenizer — no model instantiated."""
    from .analysis_service import load_unembedding_matrix, resolve_snapshot_dir
    from .jlens_model_registry import ModelNotAvailable, locate_weights, tokenizer_for

    repo_id, resolved = locate_weights(model_record)
    cached = _WEIGHTS_CACHE.get(repo_id)
    if cached is not None:
        return cached
    snapshot = resolve_snapshot_dir(resolved, repo_id) if resolved is not None else None
    if snapshot is None:
        raise ModelNotAvailable(f"No local weights for {repo_id}; download the model first")
    unembedding = load_unembedding_matrix(snapshot, device="cpu").to(torch.float32)
    tokenizer = tokenizer_for(model_record)
    _WEIGHTS_CACHE.clear()  # one model's W_U at a time: it can be gigabytes
    _WEIGHTS_CACHE[repo_id] = (unembedding, tokenizer)
    return unembedding, tokenizer


def sae_surfaces(sae: Any) -> Dict[str, Optional[torch.Tensor]]:
    """Both weight surfaces, each oriented [d_sae, d_model] — one direction per row.

    Reuses the resolvers steering and circuit discovery use, so the orientation is the
    one already pinned for every SAE format (016 task 0.1).
    """
    from .steering_service import resolve_decoder_weight, resolve_encoder_weight

    decoder = resolve_decoder_weight(sae)  # [d_model, d_sae]
    encoder = resolve_encoder_weight(sae)  # [d_sae, d_model]
    return {
        "decoder": decoder.detach().T.to(torch.float32).cpu() if decoder is not None else None,
        "encoder": encoder.detach().to(torch.float32).cpu() if encoder is not None else None,
    }


def annotation_row(
    *,
    sae_id: str,
    feature_index: int,
    surface: str,
    lens_type: str,
    annotation: Any,
    compared_label: Optional[str],
    score: Optional[float],
    sweep_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "sae_id": sae_id,
        "feature_index": int(feature_index),
        "surface": surface,
        "lens_type": lens_type,
        "layer": int(annotation.layer),
        "lens_kurtosis": annotation.lens_kurtosis,
        "workspace_class": annotation.workspace_class.value,
        "top_tokens": list(annotation.top_tokens),
        "compared_label": (compared_label or None) and compared_label[:500],
        "disagreement_score": score,
        "has_disagreement": bool(score is not None and score >= DISAGREEMENT_THRESHOLD),
        "sweep_task_id": sweep_task_id,
        "evidence_rung": 0,
    }


def upsert_annotation(db: Any, row: Dict[str, Any]) -> None:
    """Insert or replace the row for (sae, feature, surface, lens type). Sync session."""
    from ..models.jlens_feature_annotation import JLensFeatureAnnotation

    existing = (
        db.query(JLensFeatureAnnotation)
        .filter(
            JLensFeatureAnnotation.sae_id == row["sae_id"],
            JLensFeatureAnnotation.feature_index == row["feature_index"],
            JLensFeatureAnnotation.surface == row["surface"],
            JLensFeatureAnnotation.lens_type == row["lens_type"],
        )
        .first()
    )
    if existing is None:
        db.add(JLensFeatureAnnotation(**row))
    else:
        for key, value in row.items():
            setattr(existing, key, value)
        from ..core.clock import utc_now

        existing.updated_at = utc_now()


def annotate_rows(
    *,
    directions: torch.Tensor,
    indices: Sequence[int],
    labels: Dict[int, Optional[str]],
    unembedding: torch.Tensor,
    tokenizer: Any,
    transport: Any,
    layer: int,
    sae_id: str,
    surface: str,
    band_report: Any = None,
    top_k: int = 8,
    sweep_task_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Annotate each listed row of `directions` ([d_sae, d_model]); return DB-ready rows."""
    from .jlens_annotation import annotate_direction, label_disagreement

    rows = []
    for index in indices:
        annotation = annotate_direction(
            directions[int(index)],
            transport,
            unembedding,
            layer=layer,
            feature_id=str(index),
            decode=lambda ids: decode_tokens(tokenizer, ids),
            top_k=top_k,
            band_report=band_report,
        )
        label = labels.get(int(index))
        words = label_words(label)
        score = label_disagreement(words, annotation.top_tokens) if words else None
        rows.append(
            annotation_row(
                sae_id=sae_id,
                feature_index=int(index),
                surface=surface,
                lens_type=transport.lens_type,
                annotation=annotation,
                compared_label=label,
                score=score,
                sweep_task_id=sweep_task_id,
            )
        )
    return rows
