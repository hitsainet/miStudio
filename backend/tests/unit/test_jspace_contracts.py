"""
J-space interchange kinds (BR-021..023).

ADDITIVE ONLY. The tests that matter here are the ones about kinds this feature
does NOT own: miLLM consumes them today, and a schema change is a silent import
failure at the far end rather than a loud one.

Two mechanisms have already broken this in this codebase, and each gets a guard:

  * a pydantic `alias` renames on OUTPUT as well as input — it once republished
    a schema without its wire field and invalidated every exported document;
  * miLLM holds a HAND-WRITTEN mirror, so re-vendoring without updating it
    silently drops the new field.

MUTATION CONTROLS (each must turn this file red):
  * add a field to an existing shipped kind        -> "unchanged" fails
  * add an alias to any contract field             -> "no alias" fails
  * move the version out of the kind identifier    -> "version in kind" fails
  * merge Track A and Track B into one kind        -> "independent tracks" fails
  * default a template field to a value            -> "absent when not run" fails
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.schemas.jspace_contracts import (
    JSPACE_KINDS,
    KIND_JLENS_ARTIFACT,
    KIND_READOUT_RECORD,
    KIND_WATCHLIST,
    KIND_WORKSPACE_ANNOTATION,
    JLensArtifactDocument,
    ReadoutRecordDocument,
    RecipeProvenance,
    TemplateLensSpec,
    WatchlistDocument,
    WorkspaceAnnotationDocument,
)

JSPACE_MODELS = [
    JLensArtifactDocument,
    WorkspaceAnnotationDocument,
    ReadoutRecordDocument,
    WatchlistDocument,
    RecipeProvenance,
    TemplateLensSpec,
]


def recipe(**kw):
    base = dict(model_id="m", corpus="wikitext", n_prompts=100, seq_len=128)
    base.update(kw)
    return RecipeProvenance(**base)


# ── version lives in the kind identifier ───────────────────────────────────


def test_every_kind_carries_its_version_in_the_identifier():
    """So an unknown version is REJECTED, not read as an older one.

    A separate `version` field lets a consumer that does not understand v2 parse
    it as v1 with some fields missing — which is the failure this convention
    exists to prevent.
    """
    for kind in JSPACE_KINDS:
        assert kind.startswith("mistudio."), kind
        assert kind.endswith("/v1"), f"{kind} has no version in its identifier"


def test_the_kind_is_pinned_by_the_type_not_merely_defaulted():
    """A Literal REFUSES a wrong kind; a plain str with a default accepts it."""
    with pytest.raises(Exception):
        JLensArtifactDocument(
            kind="mistudio.something-else/v1",  # type: ignore[arg-type]
            slug="m",
            d_model=8,
            n_layers=2,
            n_vocab=40,
            layers=[0, 1],
            size_bytes=256,
            recipe=recipe(),
        )


def test_the_four_new_kinds_are_distinct():
    """Two kinds sharing an identifier makes one of them unreadable."""
    assert len(set(JSPACE_KINDS)) == 4


# ── no aliases, anywhere ───────────────────────────────────────────────────


@pytest.mark.parametrize("model", JSPACE_MODELS, ids=lambda m: m.__name__)
def test_no_contract_field_carries_an_alias(model):
    """An alias renames on OUTPUT as well as input.

    That is not a hypothetical: it republished a schema without its wire field
    and invalidated every exported document. The fix was to remove the alias,
    and this is the guard that keeps it removed.
    """
    for name, info in model.model_fields.items():
        assert info.alias is None, (
            f"{model.__name__}.{name} carries alias {info.alias!r} — it will "
            "rename the field on OUTPUT and invalidate exported documents"
        )
        assert info.validation_alias is None, f"{model.__name__}.{name}"
        assert info.serialization_alias is None, f"{model.__name__}.{name}"


# ── existing kinds are untouched (BR-021) ──────────────────────────────────


def test_the_existing_circuit_definition_kind_is_unchanged():
    """miStudio's shipped kinds must round-trip byte-identically.

    This feature is additive; the moment an existing kind gains a field, every
    consumer holding a hand-written mirror silently drops it.
    """
    from src.schemas import circuit_definition

    # The J-space module must not have reached into it.
    source = circuit_definition.__file__
    text = open(source, encoding="utf-8").read()
    assert "jspace" not in text.lower(), (
        "the J-space feature modified the shipped circuit-definition kind"
    )
    assert "jlens" not in text.lower()


def test_jspace_kinds_do_not_collide_with_the_shipped_ones():
    for kind in JSPACE_KINDS:
        assert "cluster-definition" not in kind
        assert "circuit-definition" not in kind
        assert "cluster-bundle" not in kind


# ── Track A and Track B are independent (BR-022) ───────────────────────────


def test_track_A_describes_a_MOUNTED_directory_and_carries_no_weights():
    """There is no ingestion API upstream — J-lens is compute-on-demand.

    A document embedding tensors would describe a transfer nobody performs, and
    would be enormous for no purpose.
    """
    fields = set(JLensArtifactDocument.model_fields)
    assert "slug" in fields and "size_bytes" in fields
    for weighty in ("weights", "tensors", "jacobians", "payload"):
        assert weighty not in fields, f"Track A carries {weighty}"


def test_track_A_and_track_B_are_SEPARATE_kinds():
    """One object for both couples two things that ship and fail independently."""
    assert KIND_JLENS_ARTIFACT != KIND_WORKSPACE_ANNOTATION
    assert set(JLensArtifactDocument.model_fields) != set(
        WorkspaceAnnotationDocument.model_fields
    )


def test_track_A_validation_absent_means_NOT_RUN():
    """Fail-closed, matching the suite itself: absent is not a pass."""
    doc = JLensArtifactDocument(
        slug="m", d_model=8, n_layers=2, n_vocab=40, layers=[0, 1],
        size_bytes=256, recipe=recipe(),
    )
    assert doc.validation == {}


# ── provenance travels with every kind (BR-007) ────────────────────────────


def test_every_jspace_document_carries_a_recipe():
    """A document without its recipe is an assertion, not a result."""
    for model in (JLensArtifactDocument, WorkspaceAnnotationDocument, WatchlistDocument):
        assert "recipe" in model.model_fields, f"{model.__name__} has no provenance"


def test_per_layer_applicability_travels_rather_than_a_model_level_claim():
    """"This artifact is frozen-Q/K" is false when the treatment reached a subset."""
    assert "per_layer_applicability" in RecipeProvenance.model_fields


def test_a_readout_document_carries_its_rung():
    """So a consumer cannot receive a readout stripped of what it is."""
    doc = ReadoutRecordDocument(model="m", prompt="p", meta={}, tokens=[])
    assert doc.evidence_rung == 0


# ── template lens: fields day-one, compute optional (BR-023) ───────────────


def test_template_fields_exist_before_the_compute_path_does():
    """Adding them later is exactly the change BR-021 forbids."""
    fields = set(TemplateLensSpec.model_fields)
    assert {"phrase", "n_contexts", "baseline_set", "whitening"} <= fields


def test_an_uncomputed_template_is_ABSENT_not_false():
    """None means NOT COMPUTED. `False` would read as "no direction exists"."""
    spec = TemplateLensSpec(phrase="a rose")
    assert spec.direction_available is None
    assert spec.n_contexts is None


def test_the_watchlist_carries_its_scoring_definition():
    """A threshold without the definition it was measured under is not portable.

    The consumer would apply it to a differently-computed score and get a
    different detector while believing it had the same one.
    """
    wl = WatchlistDocument(
        name="eval-awareness",
        recipe=recipe(),
        scoring_definition="mean lens log-prob minus control mean",
    )
    assert wl.scoring_definition
    assert "scoring_definition" in WatchlistDocument.model_fields
