"""
AppSetting service layer for business logic.

Handles CRUD operations with transparent encryption/decryption for sensitive values.
All sensitive values are encrypted before storage and decrypted (or masked) on retrieval.
"""

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.encryption import encrypt_value, decrypt_value, mask_value
from ..models.app_setting import AppSetting
from ..schemas.app_setting import AppSettingUpsert, AppSettingResponse

logger = logging.getLogger(__name__)

# Keys that are always encrypted at rest regardless of what the client sends.
_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "openai_api_key",
    "hf_token",
    "neuronpedia_api_key",
})

# Keys the generic settings CRUD must never read, write or delete
# (MIS-E2E-055 / -165).
#
# The Settings PIN was stored as an ordinary row under `settings_pin_hash` with
# `is_sensitive=False`, so the generic routes beside it defeated the gate three
# separate ways, all confirmed against the live deployment:
#
#   READ    GET /settings returned the PBKDF2 salt+hash unmasked, 150 chars.
#           A 4-digit PIN is 10,000 offline candidates.
#   WRITE   PUT /settings validated only membership in a two-element URL set,
#           so the PIN could be overwritten without knowing the current one —
#           bypassing the `current_pin` check in /pin/set entirely.
#   DELETE  DELETE /settings/{key} removed it, after which /pin/set sees
#           `existing is None` and stops requiring a current PIN at all.
#
# The guard lives HERE, in the service every route funnels through, rather than
# on each route — this audit found five instances of a correct guard applied at
# one call site and not its siblings. The PIN endpoints reach the row through
# the explicit `_privileged` escape hatch below, which is the only way in.
_PROTECTED_KEYS: frozenset[str] = frozenset({
    "settings_pin_hash",
})


class ProtectedSettingError(PermissionError):
    """A protected key was addressed through the generic settings CRUD.

    A gate must not be stored in the thing it gates. These keys are reachable
    only through the dedicated endpoints that own them.
    """


class AppSettingService:
    """Service class for application settings operations."""

    @staticmethod
    async def upsert(
        db: AsyncSession,
        data: AppSettingUpsert,
        *,
        _privileged: bool = False,
    ) -> tuple[AppSetting, bool]:
        """Create or update a setting. Returns (setting, is_new).

        Sensitive values are encrypted before storage.

        `_privileged` is the escape hatch for the endpoints that OWN a protected
        key — today only `/settings/pin/set`. It is keyword-only and underscored
        so it cannot be passed by accident from a generic route, and every use
        is greppable.
        """
        if data.key in _PROTECTED_KEYS and not _privileged:
            raise ProtectedSettingError(
                f"'{data.key}' cannot be written through the generic settings "
                f"API — it is the gate, not a setting"
            )
        result = await db.execute(
            select(AppSetting).where(AppSetting.key == data.key)
        )
        existing = result.scalar_one_or_none()

        # Server-side sensitivity: known secrets are always encrypted regardless
        # of the client-supplied is_sensitive flag, preventing plaintext downgrade.
        is_sensitive = data.key in _SENSITIVE_KEYS or data.is_sensitive
        store_value = encrypt_value(data.value) if is_sensitive else data.value

        if existing:
            existing.value = store_value
            existing.is_sensitive = is_sensitive
            existing.category = data.category
            await db.flush()
            await db.refresh(existing)
            return existing, False
        else:
            setting = AppSetting(
                key=data.key,
                value=store_value,
                is_sensitive=is_sensitive,
                category=data.category,
            )
            db.add(setting)
            await db.flush()
            await db.refresh(setting)
            return setting, True

    @staticmethod
    async def get_by_key(
        db: AsyncSession, key: str, unmask: bool = False, *, _privileged: bool = False
    ) -> Optional[AppSetting]:
        """Get a setting by key. Decrypts sensitive values if unmask=True, otherwise masks them.

        Protected keys are invisible here unless `_privileged`. Returning the
        PBKDF2 salt+hash of a 4-digit PIN to an unauthenticated caller is
        MIS-E2E-165, confirmed live.
        """
        if key in _PROTECTED_KEYS and not _privileged:
            # Not "masked" — absent. The generic API has no business confirming
            # this row exists, and `/settings/pin/status` is the endpoint that
            # answers that question properly.
            return None
        result = await db.execute(
            select(AppSetting).where(AppSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        if setting and setting.is_sensitive:
            # Expunge to prevent in-place mutation from dirtying the session
            db.expunge(setting)
            if unmask:
                # Caller wants the real value — a DecryptionError must propagate.
                # Returning unauthenticated ciphertext to a credential consumer
                # is exactly MIS-E2E-056.
                setting.value = decrypt_value(setting.value, setting_key=setting.key)
            else:
                # Display path. A row we cannot authenticate is shown as masked,
                # like the list paths already do — never as raw ciphertext.
                try:
                    decrypted = decrypt_value(setting.value, setting_key=setting.key)
                    setting.value = mask_value(decrypted)
                except Exception:
                    setting.value = "***"
        return setting

    @staticmethod
    async def get_by_category(
        db: AsyncSession, category: str
    ) -> list[AppSetting]:
        """Get all settings in a category. Sensitive values are masked."""
        result = await db.execute(
            select(AppSetting)
            .where(AppSetting.category == category)
            .order_by(AppSetting.key)
        )
        # Protected keys never appear in a listing — see _PROTECTED_KEYS.
        settings = [x for x in result.scalars().all() if x.key not in _PROTECTED_KEYS]
        for s in settings:
            if s.is_sensitive:
                db.expunge(s)
                try:
                    decrypted = decrypt_value(s.value, setting_key=s.key)
                    s.value = mask_value(decrypted)
                except Exception:
                    s.value = "***"
        return settings

    @staticmethod
    async def list_all(db: AsyncSession) -> list[AppSetting]:
        """List all settings. Sensitive values are masked."""
        result = await db.execute(
            select(AppSetting).order_by(AppSetting.category, AppSetting.key)
        )
        # Protected keys never appear in a listing — see _PROTECTED_KEYS.
        settings = [x for x in result.scalars().all() if x.key not in _PROTECTED_KEYS]
        for s in settings:
            if s.is_sensitive:
                db.expunge(s)
                try:
                    decrypted = decrypt_value(s.value, setting_key=s.key)
                    s.value = mask_value(decrypted)
                except Exception:
                    s.value = "***"
        return settings

    @staticmethod
    async def delete_by_key(
        db: AsyncSession, key: str, *, _privileged: bool = False
    ) -> bool:
        """Delete a setting by key. Returns True if deleted.

        Deleting the PIN row is how the gate was removed entirely: with no row,
        `/pin/set` sees `existing is None` and stops asking for a current PIN.
        """
        if key in _PROTECTED_KEYS and not _privileged:
            raise ProtectedSettingError(
                f"'{key}' cannot be deleted through the generic settings API"
            )
        result = await db.execute(
            delete(AppSetting).where(AppSetting.key == key)
        )
        return result.rowcount > 0

    @staticmethod
    async def get_decrypted_value(db: AsyncSession, key: str) -> Optional[str]:
        """Get the plaintext value of a setting (for internal backend use only).

        Returns None if the key doesn't exist.
        """
        result = await db.execute(
            select(AppSetting).where(AppSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        if not setting:
            return None
        if setting.is_sensitive:
            return decrypt_value(setting.value, setting_key=setting.key)
        return setting.value
