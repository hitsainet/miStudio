"""Export and import for the J-space interchange kinds (027 task 4.1, BR-021/BR-022).

`schemas/jspace_contracts.py` declared four portable kinds on 2026-08-01 and nothing
produced or read one — a contract with no consumer, the estate's signature defect in its
document form. This module is the production consumer:

* **export** — `artifact_document` describes a committed lens directory (Track A: it
  DESCRIBES a mount and carries no weights); `watchlist_document` packages a validated
  watchlist WITH the recipe of the artifact its thresholds were measured under.
* **import** — `parse_document` reads any of the four kinds by its versioned `kind`
  identifier and REFUSES an unknown kind or version rather than reading `.../v2` as v1
  with missing fields (027 FTID pitfall 3). There is deliberately no ingestion API
  (pitfall 5): parsing validates and normalises; it stores nothing.

Every exporter refuses rather than fabricates. An artifact acquired from a repository
that shipped no recipe has no corpus or prompt count to state, and a document claiming
`corpus: "unspecified"` would be the fabricated-value-wearing-a-default failure this
estate keeps finding (memory: never-fabricate-model-dimensions).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel

from ..schemas.jspace_contracts import (
    KIND_JLENS_ARTIFACT,
    KIND_READOUT_RECORD,
    KIND_WATCHLIST,
    KIND_WORKSPACE_ANNOTATION,
    JLensArtifactDocument,
    ReadoutRecordDocument,
    RecipeProvenance,
    WatchlistDocument,
    WorkspaceAnnotationDocument,
)

DOCUMENT_MODELS: Dict[str, Type[BaseModel]] = {
    KIND_JLENS_ARTIFACT: JLensArtifactDocument,
    KIND_WORKSPACE_ANNOTATION: WorkspaceAnnotationDocument,
    KIND_READOUT_RECORD: ReadoutRecordDocument,
    KIND_WATCHLIST: WatchlistDocument,
}

AnyDocument = Union[
    JLensArtifactDocument, WorkspaceAnnotationDocument, ReadoutRecordDocument, WatchlistDocument
]


class DocumentUnavailable(ValueError):
    """The source lacks a fact the document must state; refusing beats inventing it."""


class UnknownDocumentKind(ValueError):
    """A `kind` this build does not implement — including a newer version of a known one."""


def read_config(config_path: Optional[Any]) -> Dict[str, str]:
    """The artifact's `config.yaml` as flat `key: value` strings (the format the fitter writes)."""
    if config_path is None:
        return {}
    try:
        text = config_path.read_text()
    except OSError:
        return {}
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        if ":" not in raw or raw.startswith((" ", "\t", "#")):
            continue
        key, _, value = raw.partition(":")
        out[key.strip()] = value.strip()
    return out


def recipe_from_config(config: Dict[str, str]) -> RecipeProvenance:
    """The recipe an artifact was BUILT with, as its own config records it."""
    required = ("model", "corpus", "n_prompts", "seq_len")
    missing = [k for k in required if not config.get(k)]
    if missing or config.get("corpus") == "unspecified":
        raise DocumentUnavailable(
            "the artifact's recipe does not record "
            + ", ".join(missing or ["its corpus (recorded as 'unspecified')"])
            + "; a document stating them would be inventing provenance"
        )
    try:
        seq_len = max(1, int(round(float(config["seq_len"]))))
        n_prompts = int(config["n_prompts"])
    except ValueError as exc:
        raise DocumentUnavailable(f"unreadable recipe value: {exc}") from exc
    applied = config.get("attention_gradients_applied_to_layers")
    per_layer = []
    if applied and applied.startswith("["):
        try:
            per_layer = [{"layer": int(l), "attention_gradients": config.get(
                "attention_gradients_requested", "full")} for l in json.loads(applied)]
        except (ValueError, TypeError):
            per_layer = []
    library_versions: Dict[str, str] = {}
    raw_versions = config.get("library_versions")
    if raw_versions:
        try:
            parsed = json.loads(raw_versions)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            library_versions = {str(k): str(v) for k, v in parsed.items()}
    return RecipeProvenance(
        model_id=config["model"],
        target_layer=config.get("target_layer", "penultimate"),
        attention_gradients=config.get("attention_gradients_requested", "full"),
        aggregation=config.get("aggregation", "mean"),
        corpus=config["corpus"],
        n_prompts=n_prompts,
        seq_len=seq_len,
        dtype=config.get("dtype", "fp16"),
        # Absent on artifacts fitted before 2026-10-17: {} = not recorded, never guessed.
        library_versions=library_versions,
        per_layer_applicability=per_layer,
    )


def artifact_document(service: Any, ref: Any) -> JLensArtifactDocument:
    """Track A: describe a COMMITTED artifact directory. Carries no weights."""
    config = read_config(ref.config_path)
    recipe = recipe_from_config(config)
    try:
        d_model, n_layers, n_vocab = (int(config[k]) for k in ("d_model", "n_layers", "n_vocab"))
    except (KeyError, ValueError) as exc:
        raise DocumentUnavailable(
            f"the artifact's config does not record its dimensions ({exc}); "
            "they are never defaulted"
        ) from exc
    # The verdict recorded for THIS EXACT FILE (`stored_report` returns None when the
    # lens changed since). Only what RAN: absent means NOT RUN — the suite's own
    # fail-closed rule — so a class missing from the stored report is missing here
    # too, and a DEFERRED or NOT_RUN status is carried as itself, never as "pass".
    report = service.stored_report(ref) or {}
    validation: Dict[str, str] = {
        str(r["check"]): str(r["status"])
        for r in report.get("results") or []
        if isinstance(r, dict) and r.get("check") and r.get("status")
    }
    return JLensArtifactDocument(
        slug=ref.slug,
        d_model=d_model,
        n_layers=n_layers,
        n_vocab=n_vocab,
        layers=service.fitted_layers(ref),
        size_bytes=ref.size_bytes,
        recipe=recipe,
        validation=validation,
    )


def watchlist_document(watchlist: Any, recipe: RecipeProvenance) -> WatchlistDocument:
    """A watchlist WITH the recipe its thresholds were measured under (BR-025)."""
    return WatchlistDocument(
        name=watchlist.name,
        recipe=recipe,
        scoring_definition=watchlist.scoring_definition,
        concepts=[
            {"token": c.token, "threshold": float(c.threshold)} for c in watchlist.concepts
        ],
    )


def parse_document(payload: Dict[str, Any]) -> AnyDocument:
    """Read one interchange document by its `kind`. Unknown kinds and versions RAISE."""
    if not isinstance(payload, dict):
        raise UnknownDocumentKind("a document is a JSON object")
    kind = payload.get("kind")
    model = DOCUMENT_MODELS.get(kind) if isinstance(kind, str) else None
    if model is None:
        raise UnknownDocumentKind(
            f"unknown kind {kind!r}. This build reads exactly: {sorted(DOCUMENT_MODELS)}. "
            "A newer version is refused, not read as an older one with fields missing."
        )
    return model.model_validate(payload)  # type: ignore[return-value]


def dump_document(document: BaseModel) -> Dict[str, Any]:
    """The wire form. `mode="json"`, no aliases exist to apply (027 FTID pitfall 2)."""
    return document.model_dump(mode="json")
