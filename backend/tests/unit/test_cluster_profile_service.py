"""
Unit tests for the cluster-profile service layer (Feature 014, IDL-30).

Covers the pure pieces without a DB: the import compatibility matrix (every
FTDD §3 row), payload parsing (definition/bundle/hostile), the portable-format
validators (no local paths, caps), and unbound round-trip fidelity
(definition → profile → definition preserves every member field).
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.schemas.cluster_profile import (
    BUNDLE_KIND,
    DEFINITION_KIND,
    MAX_BUNDLE,
    MAX_MEMBERS,
    ClusterBundleV1,
    ClusterDefinitionV1,
    DefinitionModelRef,
    DefinitionSAERef,
    ProfileBudget,
    ProfileMember,
)
from src.services.cluster_profile_service import (
    ClusterProfileService,
    decide_compatibility,
    parse_import_payload,
)


def _definition(**overrides) -> ClusterDefinitionV1:
    base = dict(
        name="Fear cluster",
        narrative="Steers toward fear-adjacent tokens.",
        display_token="fear",
        model=DefinitionModelRef(hf_id="google/gemma-2-2b"),
        sae=DefinitionSAERef(mistudio_sae_id="sae_a", layer=12, n_features=16384),
        members=[
            ProfileMember(feature_idx=100, strength=1.2, similarity=0.9, pinned=True),
            ProfileMember(feature_idx=200, strength=-0.8, sign=-1, activation_frequency=0.2),
        ],
        budget=ProfileBudget(B=2.4, B_dir=2.4, G=1.0, formula_id="freq-budget/sim-alloc@1"),
    )
    base.update(overrides)
    return ClusterDefinitionV1(**base)


LOCAL = [
    {"id": "sae_a", "n_features": 16384, "layer": 12, "model_name": "google/gemma-2-2b"},
    {"id": "sae_b", "n_features": 16384, "layer": 6, "model_name": "google/gemma-2-2b"},
    {"id": "sae_small", "n_features": 4096, "layer": 12, "model_name": "gpt2"},
]


# ── Compatibility matrix (FTDD §3) ──────────────────────────────────────────

def test_matrix_same_id_compatible_binds_silently():
    d = decide_compatibility(_definition(), LOCAL)
    assert d.action == "bind"
    assert d.sae_id == "sae_a"
    assert d.warnings == []


def test_matrix_n_features_mismatch_blocks():
    definition = _definition(sae=DefinitionSAERef(mistudio_sae_id="sae_small", n_features=16384))
    # Same id exists locally but with different n_features? Use explicit choice:
    d = decide_compatibility(definition, LOCAL, explicit_sae_id="sae_small")
    assert d.action == "block"
    assert "n_features mismatch" in d.warnings[0]


def test_matrix_model_mismatch_warn_binds():
    definition = _definition(
        model=DefinitionModelRef(hf_id="meta-llama/Llama-3-8B"),
        sae=DefinitionSAERef(mistudio_sae_id="sae_a", layer=12, n_features=16384),
    )
    d = decide_compatibility(definition, LOCAL)
    assert d.action == "warn_bind"
    assert any("Model mismatch" in w for w in d.warnings)


def test_matrix_layer_mismatch_warn_binds():
    definition = _definition(sae=DefinitionSAERef(mistudio_sae_id="sae_b", layer=12, n_features=16384))
    d = decide_compatibility(definition, LOCAL)
    assert d.action == "warn_bind"
    assert any("Layer mismatch" in w for w in d.warnings)


def test_matrix_no_local_saes_unbound():
    d = decide_compatibility(_definition(), [])
    assert d.action == "unbound"


def test_matrix_no_matching_n_features_blocks_with_hint():
    definition = _definition(sae=DefinitionSAERef(layer=12, n_features=99999))
    d = decide_compatibility(definition, LOCAL)
    assert d.action == "block"
    assert "unbound" in d.warnings[0]


def test_matrix_explicit_choice_overrides_auto():
    # Auto would pick sae_a (same id); explicit sae_b wins (with layer warning).
    d = decide_compatibility(_definition(), LOCAL, explicit_sae_id="sae_b")
    assert d.sae_id == "sae_b"
    assert d.action == "warn_bind"


def test_matrix_explicit_choice_missing_is_unbound():
    d = decide_compatibility(_definition(), LOCAL, explicit_sae_id="sae_ghost")
    assert d.action == "unbound"


def test_matrix_id_unknown_prefers_matching_layer():
    definition = _definition(
        sae=DefinitionSAERef(mistudio_sae_id="sae_elsewhere", layer=6, n_features=16384)
    )
    d = decide_compatibility(definition, LOCAL)
    assert d.sae_id == "sae_b"  # n_features + layer match beats sae_a
    assert d.action == "warn_bind"  # "id differs" warning


# ── Payload parsing ─────────────────────────────────────────────────────────

def test_parse_single_definition():
    payload = _definition().model_dump(mode="json")
    defs = parse_import_payload(payload)
    assert len(defs) == 1
    assert defs[0].name == "Fear cluster"


def test_parse_bundle():
    bundle = ClusterBundleV1(definitions=[_definition(), _definition(name="Joy")])
    defs = parse_import_payload(bundle.model_dump(mode="json"))
    assert [d.name for d in defs] == ["Fear cluster", "Joy"]


def test_parse_unknown_kind_rejected():
    with pytest.raises(ValueError, match="Unknown kind"):
        parse_import_payload({"kind": "mistudio.evil", "schema_version": "1"})


def test_parse_hostile_json_shapes():
    for hostile in ({}, {"kind": None}, {"definitions": "x"}, {"kind": BUNDLE_KIND}):
        with pytest.raises((ValueError, ValidationError)):
            parse_import_payload(hostile)


# ── Contract validators ─────────────────────────────────────────────────────

def test_source_hint_rejects_local_paths():
    for bad in ("/data/saes/x", "~/saes/x", "../x", "C:\\saes\\x"):
        with pytest.raises(ValidationError):
            DefinitionSAERef(source_hint=bad)
    # hf-style hints pass
    assert DefinitionSAERef(source_hint="hf:repo/path").source_hint == "hf:repo/path"


def test_member_cap_enforced():
    members = [ProfileMember(feature_idx=i, strength=1.0) for i in range(MAX_MEMBERS + 1)]
    with pytest.raises(ValidationError):
        _definition(members=members)


def test_bundle_cap_enforced():
    defs = [_definition(name=f"c{i}") for i in range(2)]
    ClusterBundleV1(definitions=defs)  # fine
    with pytest.raises(ValidationError):
        ClusterBundleV1(definitions=[_definition(name=f"c{i}") for i in range(MAX_BUNDLE + 1)])


def test_definition_never_contains_secret_or_path_fields():
    """The serialized artifact must not even have fields that could carry
    secrets/tokens/local paths (BR-level guarantee, not just empty values)."""
    dumped = _definition().model_dump(mode="json")
    flat = str(dumped).lower()
    for forbidden in ("token\":", "secret", "api_key", "local_path", "password"):
        assert forbidden not in flat


# ── Round-trip fidelity (unbound path is DB-free) ───────────────────────────

@pytest.mark.asyncio
async def test_unbound_round_trip_preserves_members_and_budget():
    original = _definition(sae=DefinitionSAERef())  # unbound
    profile = ClusterProfileService.from_definition(original, bind_sae_id=None)
    # to_definition touches the DB only when sae_id is set
    profile.created_at = datetime(2026, 7, 16)
    result = await ClusterProfileService.to_definition(MagicMock(), profile)

    assert result.name == original.name
    assert result.narrative == original.narrative
    assert result.display_token == original.display_token
    assert [m.model_dump() for m in result.members] == [m.model_dump() for m in original.members]
    assert result.budget is not None and original.budget is not None
    assert result.budget.model_dump() == original.budget.model_dump()
    assert result.kind == DEFINITION_KIND
    assert result.schema_version == "1"


@pytest.mark.asyncio
async def test_update_narrative_clearable_via_explicit_null():
    """PATCH {"narrative": null} must CLEAR the narrative (fields_set semantics)."""
    from src.schemas.cluster_profile import ClusterProfileUpdate

    profile = ClusterProfileService.from_definition(_definition(), bind_sae_id=None)
    assert profile.narrative is not None
    db = MagicMock()
    db.commit = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock()
    db.refresh = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock()
    updated = await ClusterProfileService.update(
        db, profile, ClusterProfileUpdate(narrative=None)
    )
    assert updated.narrative is None
    # Omitting the field leaves it untouched
    profile2 = ClusterProfileService.from_definition(_definition(), bind_sae_id=None)
    updated2 = await ClusterProfileService.update(
        db, profile2, ClusterProfileUpdate(name="renamed")
    )
    assert updated2.narrative is not None
    assert updated2.name == "renamed"


def test_from_definition_records_import_provenance():
    original = _definition()
    profile = ClusterProfileService.from_definition(original, bind_sae_id="sae_a")
    assert profile.sae_id == "sae_a"
    assert profile.imported_from["kind"] == DEFINITION_KIND
    assert profile.imported_from["source_sae_id"] == "sae_a"


class TestMemberEnrichment:
    """Contract rev 2026-07-17: label + meta populated from Feature records."""

    @staticmethod
    def _feature(idx=3169, **over):
        from unittest.mock import MagicMock

        f = MagicMock()
        f.neuron_index = idx
        f.name = over.get("name", "free_newsletter")
        f.description = over.get("description", "Fires on: free newsletters.")
        f.category = "semantic"
        f.label_source = "openai"
        f.interpretability_score = 0.78
        f.mean_activation = 1.96
        f.created_at = over.get("created_at")
        f.nlp_analysis = over.get("nlp_analysis", {
            "prime_token_analysis": {
                "word_lowercase_distribution": {"free": 12, "the": 15, "in": 7},
            },
            "context_patterns": {
                "prefix_trigrams": [{"tokens": ["subscribe", "to", "our"]}],
                "suffix_bigrams": [{"tokens": ["news", "letters"]}],
            },
        })
        return f

    @pytest.mark.asyncio
    async def test_enriches_label_and_meta(self):
        from unittest.mock import AsyncMock, MagicMock

        from src.schemas.cluster_profile import ProfileMember
        from src.services.cluster_profile_service import ClusterProfileService

        db = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [self._feature()]
        db.execute = AsyncMock(return_value=result)

        members = [ProfileMember(feature_idx=3169, strength=0.2)]
        out = await ClusterProfileService._enrich_members(
            db, members, "sae_x", None)
        assert out[0].label == "free_newsletter"
        assert out[0].meta.description == "Fires on: free newsletters."
        assert out[0].meta.top_tokens[0] == "the"  # freq-sorted
        assert "___" in out[0].meta.signature
        assert out[0].meta.interpretability == 0.78

    @pytest.mark.asyncio
    async def test_never_overwrites_existing_values(self):
        from unittest.mock import AsyncMock, MagicMock

        from src.schemas.cluster_profile import MemberMeta, ProfileMember
        from src.services.cluster_profile_service import ClusterProfileService

        db = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [self._feature()]
        db.execute = AsyncMock(return_value=result)

        members = [ProfileMember(
            feature_idx=3169, strength=0.2, label="author_label",
            meta=MemberMeta(description="author words"))]
        out = await ClusterProfileService._enrich_members(
            db, members, "sae_x", None)
        assert out[0].label == "author_label"
        assert out[0].meta.description == "author words"

    @pytest.mark.asyncio
    async def test_no_feature_row_leaves_member_unchanged(self):
        from unittest.mock import AsyncMock, MagicMock

        from src.schemas.cluster_profile import ProfileMember
        from src.services.cluster_profile_service import ClusterProfileService

        db = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result)

        members = [ProfileMember(feature_idx=42, strength=0.1)]
        out = await ClusterProfileService._enrich_members(
            db, members, "sae_x", None)
        assert out[0].label is None and out[0].meta is None

    def test_meta_is_extensible_and_optional(self):
        from src.schemas.cluster_profile import MemberMeta

        meta = MemberMeta.model_validate(
            {"description": "d", "future_field": {"x": 1}})
        assert meta.model_dump()["future_field"] == {"x": 1}  # extra=allow
        assert MemberMeta().model_dump(exclude_none=True) == {}  # all optional


class TestEnrichmentResilience:
    """Contract-rev review #1: best-effort enrichment must NEVER 500."""

    @pytest.mark.asyncio
    async def test_oversized_db_values_truncate_not_raise(self):
        from unittest.mock import AsyncMock, MagicMock

        from src.schemas.cluster_profile import ProfileMember
        from src.services.cluster_profile_service import ClusterProfileService

        f = TestMemberEnrichment._feature()
        f.category = "x" * 300          # DB String(255) > schema cap 50
        f.description = "d" * 5000      # unbounded Text > cap 1000
        f.interpretability_score = 1.7  # legacy out-of-range

        db = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [f]
        db.execute = AsyncMock(return_value=result)

        out = await ClusterProfileService._enrich_members(
            db, [ProfileMember(feature_idx=3169, strength=0.2)], "sae_x", None)
        assert out[0].meta is not None
        assert len(out[0].meta.category) == 50
        assert len(out[0].meta.description) == 1000
        assert out[0].meta.interpretability == 1.0

    @pytest.mark.asyncio
    async def test_pathological_feature_row_degrades_to_label_only(self):
        from unittest.mock import AsyncMock, MagicMock

        from src.schemas.cluster_profile import ProfileMember
        from src.services.cluster_profile_service import ClusterProfileService

        f = TestMemberEnrichment._feature()
        f.nlp_analysis = {"context_patterns": {"prefix_trigrams": object()}}

        db = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [f]
        db.execute = AsyncMock(return_value=result)

        out = await ClusterProfileService._enrich_members(
            db, [ProfileMember(feature_idx=3169, strength=0.2)], "sae_x", None)
        assert out[0].label == "free_newsletter"  # label survives
        # meta enrichment failed quietly — no exception escaped

    @pytest.mark.asyncio
    async def test_export_of_bound_profile_carries_label_and_meta(self):
        """Review #11: the EXPORTED definition (to_definition), not just the
        helper, must backfill label+meta — this is what miLLM tiles consume."""
        from unittest.mock import AsyncMock, MagicMock

        from src.services.cluster_profile_service import ClusterProfileService

        profile = ClusterProfileService.from_definition(
            _definition(sae=DefinitionSAERef()), bind_sae_id="sae_x")
        profile.created_at = datetime(2026, 7, 16)
        profile.members[0]["feature_idx"] = 3169

        db = MagicMock()
        sae_lookup = MagicMock()
        sae_lookup.scalar_one_or_none.return_value = None  # SAE row gone; export still enriches
        feature_lookup = MagicMock()
        feature_lookup.scalars.return_value.all.return_value = [
            TestMemberEnrichment._feature()]
        db.execute = AsyncMock(side_effect=[sae_lookup, feature_lookup])

        definition = await ClusterProfileService.to_definition(db, profile)
        exported = {m.feature_idx: m for m in definition.members}
        assert exported[3169].label == "free_newsletter"
        assert exported[3169].meta is not None
        assert exported[3169].meta.description is not None
