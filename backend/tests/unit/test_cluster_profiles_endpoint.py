"""
Contract tests for the cluster-profile endpoints (Feature 014).

The endpoints are DB glue over the service layer; these tests exercise the
glue with the service/DB mocked: import caps and hostile payloads, per-item
isolation inside bundles, export 404, and the SAE-delete 409 guard.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.v1.endpoints.cluster_profiles import (
    export_profile,
    import_profiles,
)
from src.api.v1.endpoints.saes import delete_sae
from src.schemas.cluster_profile import (
    ClusterBundleV1,
    ClusterDefinitionV1,
    DefinitionSAERef,
    ImportRequest,
    ProfileMember,
)


def _definition(name="Fear cluster", **overrides) -> ClusterDefinitionV1:
    base = dict(
        name=name,
        sae=DefinitionSAERef(),
        members=[ProfileMember(feature_idx=1, strength=1.0)],
    )
    base.update(overrides)
    return ClusterDefinitionV1(**base)


def _db_with_no_saes() -> MagicMock:
    db = MagicMock()
    execute_result = MagicMock()
    execute_result.all.return_value = []
    db.execute = AsyncMock(return_value=execute_result)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_import_payload_over_1mb_rejected():
    huge = {"kind": "mistudio.cluster-definition", "blob": "x" * 1_100_000}
    with pytest.raises(HTTPException) as e:
        await import_profiles(ImportRequest(payload=huge), db=MagicMock())
    assert e.value.status_code == 413


@pytest.mark.asyncio
async def test_import_bundle_over_cap_rejected():
    # 51 definition STUBS — the cap must fire before full validation cost.
    payload = {
        "kind": "mistudio.cluster-bundle",
        "schema_version": "1",
        "definitions": [{"name": f"c{i}"} for i in range(51)],
    }
    with pytest.raises(HTTPException) as e:
        await import_profiles(ImportRequest(payload=payload), db=MagicMock())
    assert e.value.status_code == 400
    assert "exceeds" in str(e.value.detail)


@pytest.mark.asyncio
async def test_import_unknown_kind_rejected():
    with pytest.raises(HTTPException) as e:
        await import_profiles(
            ImportRequest(payload={"kind": "mistudio.evil"}), db=MagicMock()
        )
    assert e.value.status_code == 400
    assert "Invalid import payload" in str(e.value.detail)


@pytest.mark.asyncio
async def test_import_unbound_when_no_local_saes():
    db = _db_with_no_saes()
    payload = _definition().model_dump(mode="json")
    resp = await import_profiles(ImportRequest(payload=payload), db=db)
    assert resp.imported == 1
    assert resp.results[0].status == "imported_unbound"


@pytest.mark.asyncio
async def test_import_bundle_per_item_isolation():
    """One failing item must not poison the rest of the bundle."""
    db = _db_with_no_saes()
    # Second commit blows up; first and third succeed.
    db.commit = AsyncMock(side_effect=[None, RuntimeError("db down"), None])
    bundle = ClusterBundleV1(
        definitions=[_definition("a"), _definition("b"), _definition("c")]
    )
    resp = await import_profiles(
        ImportRequest(payload=bundle.model_dump(mode="json")), db=db
    )
    assert resp.imported == 2
    assert resp.errors == 1
    assert [r.status for r in resp.results] == ["imported_unbound", "error", "imported_unbound"]
    # Internal DB error is NOT leaked to the response
    assert resp.results[1].error == "internal error"
    db.rollback.assert_awaited()


@pytest.mark.asyncio
async def test_import_blocks_out_of_bounds_members_against_bound_sae():
    """Definition metadata can lie — indices are checked against the ACTUAL bound SAE."""
    db = _db_with_no_saes()
    rows = MagicMock()
    row = MagicMock()
    row.id, row.n_features, row.layer, row.model_name = "sae_small", 100, 12, "gpt2"
    rows.all.return_value = [row]
    db.execute = AsyncMock(return_value=rows)
    definition = _definition(
        members=[ProfileMember(feature_idx=5000, strength=1.0)],
        sae=DefinitionSAERef(mistudio_sae_id="sae_small"),  # claims no n_features
    )
    resp = await import_profiles(
        ImportRequest(payload=definition.model_dump(mode="json")), db=db
    )
    assert resp.blocked == 1
    assert resp.imported == 0
    assert any("out of bounds" in w for w in resp.results[0].warnings)


@pytest.mark.asyncio
async def test_export_missing_profile_404():
    with patch(
        "src.api.v1.endpoints.cluster_profiles.ClusterProfileService.get",
        new=AsyncMock(return_value=None),
    ):
        with pytest.raises(HTTPException) as e:
            await export_profile("clp_ghost", db=MagicMock())
    assert e.value.status_code == 404


# ── SAE-delete guard ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sae_delete_blocked_by_bound_profiles():
    with patch(
        "src.services.cluster_profile_service.ClusterProfileService.count_for_sae",
        new=AsyncMock(return_value=3),
    ):
        with pytest.raises(HTTPException) as e:
            await delete_sae("sae_x", delete_files=True, force=False, db=MagicMock())
    assert e.value.status_code == 409
    assert e.value.detail["code"] == "PROFILES_BOUND"
    assert e.value.detail["profile_count"] == 3


@pytest.mark.asyncio
async def test_sae_delete_force_unbinds_then_deletes():
    unbind = AsyncMock(return_value=3)
    with patch(
        "src.services.cluster_profile_service.ClusterProfileService.count_for_sae",
        new=AsyncMock(return_value=3),
    ), patch(
        "src.services.cluster_profile_service.ClusterProfileService.unbind_for_sae",
        new=unbind,
    ), patch(
        "src.api.v1.endpoints.saes.SAEManagerService.delete_sae",
        new=AsyncMock(return_value=True),
    ):
        resp = await delete_sae("sae_x", delete_files=True, force=True, db=MagicMock())
    unbind.assert_awaited_once()
    assert "deleted" in resp["message"]


@pytest.mark.asyncio
async def test_sae_delete_unguarded_when_no_profiles():
    with patch(
        "src.services.cluster_profile_service.ClusterProfileService.count_for_sae",
        new=AsyncMock(return_value=0),
    ), patch(
        "src.api.v1.endpoints.saes.SAEManagerService.delete_sae",
        new=AsyncMock(return_value=True),
    ):
        resp = await delete_sae("sae_x", delete_files=True, force=False, db=MagicMock())
    assert "deleted" in resp["message"]
