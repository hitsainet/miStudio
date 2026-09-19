"""
Tests for /api/v1/settings — specifically the round-trip safety of sensitive values.

Regression coverage for a bug where the upsert endpoint mutated the SQLAlchemy
session-bound `setting.value` to the masked form for the response, causing the
masked string to overwrite the encrypted ciphertext on session commit.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.models.app_setting import AppSetting

pytestmark = pytest.mark.asyncio


REAL_OPENAI_KEY = "sk-proj-" + "x" * 40 + "TMKA"  # 52 chars, like a real key


class TestUpsertPreservesEncryptedValue:
    """Saving a sensitive value must not corrupt the at-rest ciphertext."""

    async def test_single_upsert_keeps_ciphertext_in_db(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        response = await client.put(
            "/api/v1/settings",
            json={
                "key": "openai_api_key",
                "value": REAL_OPENAI_KEY,
                "is_sensitive": True,
                "category": "api_keys",
            },
        )
        assert response.status_code == 200
        body = response.json()

        # Response should be masked (sk-...TMKA — first 3 + ... + last 4)
        assert body["value"] != REAL_OPENAI_KEY
        assert body["value"].startswith("sk-")
        assert body["value"].endswith("TMKA")
        assert "..." in body["value"]

        # DB must contain the actual encrypted ciphertext, not the mask.
        # AES-GCM envelope = 12-byte nonce + ciphertext + 16-byte tag, base64-encoded.
        # For a 52-char plaintext, the stored value will be ~110+ chars.
        result = await async_session.execute(
            select(AppSetting).where(AppSetting.key == "openai_api_key")
        )
        row = result.scalar_one()
        assert row.is_sensitive is True
        assert len(row.value) > len(REAL_OPENAI_KEY), (
            f"DB value len={len(row.value)} is not longer than plaintext "
            f"len={len(REAL_OPENAI_KEY)} — masked string was committed over ciphertext"
        )
        assert "..." not in row.value, "DB value contains '...' — the mask leaked into storage"

    async def test_re_save_does_not_progressively_truncate(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        """Multiple saves of the same key must each round-trip cleanly."""
        for _ in range(3):
            response = await client.put(
                "/api/v1/settings",
                json={
                    "key": "openai_api_key",
                    "value": REAL_OPENAI_KEY,
                    "is_sensitive": True,
                    "category": "api_keys",
                },
            )
            assert response.status_code == 200

        result = await async_session.execute(
            select(AppSetting).where(AppSetting.key == "openai_api_key")
        )
        row = result.scalar_one()
        assert len(row.value) > len(REAL_OPENAI_KEY)
        assert "..." not in row.value


class TestPinPersistence:
    """Settings PIN must survive across request/session boundaries.

    Regression coverage for Phase 3 (review finding #1): confirms that the
    get_db dependency's auto-commit causes the PIN hash to actually reach the
    database, not just the in-memory session.
    """

    async def test_pin_set_persists_to_db(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        """POST /pin/set → DB row exists with a PBKDF2 hash value."""
        response = await client.post(
            "/api/v1/settings/pin/set",
            json={"pin": "testpin123"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

        # Verify the hash landed in the database — not just the session.
        result = await async_session.execute(
            select(AppSetting).where(AppSetting.key == "settings_pin_hash")
        )
        row = result.scalar_one_or_none()
        assert row is not None, "PIN hash row not found in DB — commit did not happen"
        # The row is ENCRYPTED at rest now (MIS-E2E-055/-165). It used to be
        # written is_sensitive=False, which is precisely why GET /settings
        # served the PBKDF2 salt+hash of a 4-digit PIN in the clear to an
        # unauthenticated caller — 10,000 offline candidates.
        assert row.is_sensitive is True, (
            "the PIN hash must be encrypted at rest; storing it plaintext is "
            "what made it readable when the row was exposed"
        )
        assert not row.value.startswith("pbkdf2:sha256:"), (
            "the raw PBKDF2 hash is sitting in the column unencrypted"
        )

        # Decrypting it must yield the real hash — otherwise this test would
        # pass against a version that stored garbage.
        from src.core.encryption import decrypt_value

        plaintext = decrypt_value(row.value, setting_key="settings_pin_hash")
        assert plaintext.startswith("pbkdf2:sha256:"), (
            f"Unexpected hash format after decryption: {plaintext[:40]}"
        )

    async def test_pin_status_reflects_configured(self, client: AsyncClient):
        """After setting a PIN, /pin/status must report configured=true."""
        await client.post(
            "/api/v1/settings/pin/set",
            json={"pin": "testpin123"},
        )
        response = await client.get("/api/v1/settings/pin/status")
        assert response.status_code == 200
        body = response.json()
        assert body["configured"] is True

    async def test_pin_verify_correct(self, client: AsyncClient):
        """Correct PIN must verify successfully."""
        await client.post("/api/v1/settings/pin/set", json={"pin": "correct-pin"})
        response = await client.post(
            "/api/v1/settings/pin/verify", json={"pin": "correct-pin"}
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True

    async def test_pin_verify_wrong(self, client: AsyncClient):
        """Wrong PIN must return valid=false, not raise."""
        await client.post("/api/v1/settings/pin/set", json={"pin": "correct-pin"})
        response = await client.post(
            "/api/v1/settings/pin/verify", json={"pin": "wrong-pin"}
        )
        assert response.status_code == 200
        assert response.json()["valid"] is False

    async def test_pin_change_requires_current_pin(self, client: AsyncClient):
        """Changing PIN without current_pin must return 400."""
        await client.post("/api/v1/settings/pin/set", json={"pin": "original-pin"})
        response = await client.post(
            "/api/v1/settings/pin/set",
            json={"pin": "new-pin"},
        )
        assert response.status_code == 400

    async def test_pin_change_with_wrong_current_pin(self, client: AsyncClient):
        """Changing PIN with incorrect current_pin must return 401."""
        await client.post("/api/v1/settings/pin/set", json={"pin": "original-pin"})
        response = await client.post(
            "/api/v1/settings/pin/set",
            json={"pin": "new-pin", "current_pin": "wrong"},
        )
        assert response.status_code == 401

    async def test_pin_change_succeeds_with_correct_current_pin(
        self, client: AsyncClient
    ):
        """Changing PIN with correct current_pin must succeed and new PIN verifies."""
        await client.post("/api/v1/settings/pin/set", json={"pin": "original-pin"})
        change = await client.post(
            "/api/v1/settings/pin/set",
            json={"pin": "new-pin", "current_pin": "original-pin"},
        )
        assert change.status_code == 200

        # Old PIN no longer works
        old = await client.post(
            "/api/v1/settings/pin/verify", json={"pin": "original-pin"}
        )
        assert old.json()["valid"] is False

        # New PIN works
        new = await client.post(
            "/api/v1/settings/pin/verify", json={"pin": "new-pin"}
        )
        assert new.json()["valid"] is True
