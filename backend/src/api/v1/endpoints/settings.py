"""
Application Settings API endpoints.

Provides CRUD operations for persistent application settings with
transparent encryption for sensitive values (API keys, tokens).
"""

import hashlib
import hmac
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.config import settings
from ....core.deps import get_db
from ....schemas.app_setting import (
    AppSettingUpsert,
    AppSettingResponse,
    AppSettingBulkUpsert,
    AppSettingBulkResponse,
    PinStatusResponse,
    PinVerifyRequest,
    PinVerifyResponse,
    PinSetRequest,
)
from ....services.app_setting_service import (
    AppSettingService,
    ProtectedSettingError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])

# Key used to store the PIN hash in app_settings
_PIN_HASH_KEY = "settings_pin_hash"


def _hash_pin(pin: str) -> str:
    """Hash a PIN with PBKDF2-SHA256 and a random salt."""
    salt = os.urandom(32)
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, 600_000)
    return f"pbkdf2:sha256:600000:{salt.hex()}:{dk.hex()}"


def _verify_pin(pin: str, stored: str) -> bool:
    """Verify a PIN against a stored PBKDF2 hash."""
    try:
        _, algo, iters_str, salt_hex, hash_hex = stored.split(":")
        if algo != "sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, int(iters_str))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


@router.get("/pin/status", response_model=PinStatusResponse)
async def get_pin_status(db: AsyncSession = Depends(get_db)):
    """Return whether a settings PIN is configured and whether the bypass flag is active."""
    existing = await AppSettingService.get_by_key(db, _PIN_HASH_KEY, _privileged=True)
    return PinStatusResponse(
        configured=existing is not None,
        bypass_active=settings.bypass_settings_pin,
    )


@router.post("/pin/verify", response_model=PinVerifyResponse)
async def verify_pin(body: PinVerifyRequest, db: AsyncSession = Depends(get_db)):
    """Verify the settings PIN. Returns {valid: true} on success."""
    existing = await AppSettingService.get_by_key(db, _PIN_HASH_KEY, unmask=True, _privileged=True)
    if not existing:
        # No PIN configured — trivially valid
        return PinVerifyResponse(valid=True)
    return PinVerifyResponse(valid=_verify_pin(body.pin, existing.value))


@router.post("/pin/set", status_code=200)
async def set_pin(body: PinSetRequest, db: AsyncSession = Depends(get_db)):
    """Set or change the settings PIN.

    Requires the current PIN when one is already configured, unless
    MISTUDIO_BYPASS_PIN=true is active (the recovery path).
    """
    existing = await AppSettingService.get_by_key(db, _PIN_HASH_KEY, unmask=True, _privileged=True)

    if existing and not settings.bypass_settings_pin:
        if not body.current_pin:
            raise HTTPException(status_code=400, detail="current_pin is required to change an existing PIN")
        if not _verify_pin(body.current_pin, existing.value):
            raise HTTPException(status_code=401, detail="Current PIN is incorrect")

    await AppSettingService.upsert(
        db,
        # The PIN endpoint OWNS this key — the only privileged write in the app.
        _privileged=True,
        data=AppSettingUpsert(
            key=_PIN_HASH_KEY,
            value=_hash_pin(body.pin),
            # Encrypted at rest. Belt and braces: _PROTECTED_KEYS already hides
            # this row from every generic read, but it was served in the clear
            # BECAUSE it was written is_sensitive=False, so if that guard is
            # ever removed the row should still not be readable.
            is_sensitive=True,
            category="system",
        ),
    )
    # Explicit commit before returning so the row is visible to any immediately-
    # following request (e.g. the frontend's set-then-GET /pin/status pattern).
    # get_db also commits after the endpoint returns, but that happens after the
    # HTTP response is sent — too late for rapid sequential callers.
    await db.commit()
    return {"success": True}


@router.get("", response_model=list[AppSettingResponse])
async def list_settings(
    category: Optional[str] = Query(None, description="Filter by category"),
    db: AsyncSession = Depends(get_db),
):
    """List all settings, optionally filtered by category. Sensitive values are masked."""
    if category:
        return await AppSettingService.get_by_category(db, category)
    return await AppSettingService.list_all(db)


@router.get("/{key}", response_model=AppSettingResponse)
async def get_setting(
    key: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a single setting by key. Sensitive values are masked."""
    setting = await AppSettingService.get_by_key(db, key)
    if not setting:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return setting


_URL_VALIDATED_KEYS = frozenset({"ollama_url", "openai_compatible_endpoint"})


@router.put("", response_model=AppSettingResponse, status_code=200)
async def upsert_setting(
    data: AppSettingUpsert,
    db: AsyncSession = Depends(get_db),
):
    """Create or update a setting (upsert). Sensitive values are encrypted at rest."""
    if data.key in _URL_VALIDATED_KEYS:
        from ....utils.url_validation import validate_llm_endpoint_url
        try:
            validate_llm_endpoint_url(data.value)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        setting, is_new = await AppSettingService.upsert(db, data)
    except ProtectedSettingError as e:
        raise HTTPException(status_code=403, detail=str(e))
    try:
        # Return masked response for sensitive values. CRITICAL: expunge the row from
        # the session before mutating its `value` field, otherwise SQLAlchemy will
        # autoflush/commit the masked string back over the encrypted ciphertext on
        # request teardown — silently corrupting the row each save.
        if setting.is_sensitive:
            from ....core.encryption import decrypt_value, mask_value
            db.expunge(setting)
            try:
                setting.value = mask_value(
                    decrypt_value(setting.value, setting_key=setting.key))
            except Exception:
                # A row we cannot authenticate is masked, never echoed back as
                # raw ciphertext. decrypt_value now raises on InvalidTag.
                setting.value = "***"
        return setting
    except HTTPException:
        raise  # see the note in bulk_upsert_settings
    except Exception as e:
        logger.exception(f"Failed to upsert setting '{data.key}'")
        raise HTTPException(status_code=500, detail="Failed to save setting")


@router.put("/bulk", response_model=AppSettingBulkResponse)
async def bulk_upsert_settings(
    data: AppSettingBulkUpsert,
    db: AsyncSession = Depends(get_db),
):
    """Create or update multiple settings at once."""
    results = []
    created = 0
    updated = 0
    try:
        for item in data.settings:
            try:
                setting, is_new = await AppSettingService.upsert(db, item)
            except ProtectedSettingError as e:
                # /bulk must not be the way around the guard on /settings —
                # MIS-E2E-073 is the same shape (bulk skipped the URL validation
                # the single-key route applied).
                raise HTTPException(status_code=403, detail=str(e))
            if setting.is_sensitive:
                from ....core.encryption import decrypt_value, mask_value
                # Expunge before mutating — otherwise the masked string overwrites
                # the encrypted ciphertext on session commit (see upsert_setting).
                db.expunge(setting)
                try:
                    setting.value = mask_value(
                        decrypt_value(setting.value, setting_key=setting.key))
                except Exception:
                    # A row we cannot authenticate is masked, never echoed back as
                    # raw ciphertext. decrypt_value now raises on InvalidTag.
                    setting.value = "***"
            results.append(setting)
            if is_new:
                created += 1
            else:
                updated += 1
        return AppSettingBulkResponse(data=results, created=created, updated=updated)
    except HTTPException:
        # Re-raise deliberate HTTP responses before the generic handler.
        # Without this, the 403 raised for a protected key is caught below and
        # re-raised as a 500 — the MIS-E2E-103 pattern, where a client mistake
        # is reported as a server error and the real reason is lost.
        raise
    except Exception as e:
        logger.exception("Failed to bulk upsert settings")
        raise HTTPException(status_code=500, detail="Failed to save settings")


@router.delete("/{key}", status_code=204)
async def delete_setting(
    key: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a setting by key."""
    try:
        deleted = await AppSettingService.delete_by_key(db, key)
    except ProtectedSettingError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
