"""Runtime environment settings (Feature 036; ENH-003 §3.1, D2–D8, D12, D13).

``/api/v1/settings`` is the free key/value store (API keys, labeling defaults, the PIN), so this
surface has its own path (D3). The environment stays the source of truth: ``PUT`` stores an override
of one allow-listed key (``core.config.RUNTIME_KEYS``), ``DELETE`` removes it and the environment's
value applies again. Every write appends to the change history (D12) and refreshes this process's
override cache before it returns — the first of the three refresh points
(``core.runtime_settings``).

No MCP write tool exists for these routes in v1 (D5): writes are the operator's, from the Settings →
Environment tab, which sits behind the Settings PIN (D8; a UI lock, not authentication). Agents read
the listing through ``get_environment_settings``.

Secrets are never served: a secret row carries only ``is_set``. ``test_environment_api.py`` puts a
sentinel secret in the settings and requires it to appear nowhere in the response body.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ....core import config, runtime_settings
from ....core.deps import get_db
from ....models.environment_setting import EnvironmentOverride, EnvironmentSettingChange
from ....schemas.environment import (
    REASON_MAX,
    EnvironmentChange,
    EnvironmentHistory,
    EnvironmentListing,
    EnvironmentRow,
    EnvironmentWrite,
    EnvironmentWriteResult,
    LastChange,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/environment", tags=["environment"])

#: v1 has one writer: the operator, through the UI (D5, FTID TI15).
ACTOR = "operator"

#: Environment variable names that differ from the field name upper-cased.
_ENV_NAMES = {"bypass_settings_pin": "MISTUDIO_BYPASS_PIN"}


def redact_url(value: Any) -> Any:
    """A URL string with any password replaced; anything else unchanged."""
    if not isinstance(value, str) or "://" not in value:
        return value
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if parts.password is None:
        return value
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    user = parts.username or ""
    return urlunsplit((parts.scheme, f"{user}:<redacted>@{host}", parts.path, parts.query, parts.fragment))


def _env_name(key: str) -> str:
    return _ENV_NAMES.get(key, key.upper())


def _label(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _description(key: str) -> str:
    field = config.Settings.model_fields.get(key)
    return (field.description or "") if field is not None else ""


def _require_runtime(key: str) -> None:
    if key not in config.RUNTIME_KEYS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"setting_not_runtime: {key!r} cannot be changed while miStudio runs. Change it in "
                "the environment and restart the pod that reads it."
            ),
        )


def _check_url(value: Any, *, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError("an address is required")
        return
    parts = urlsplit(str(value))
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("must be an http:// or https:// address with a host name")


def _extra_check(key: str, value: Any) -> None:
    """Checks beyond the field's own type and bounds, run on WRITE only (FTID §3.4)."""
    if key == "ollama_url":
        from ....utils.url_validation import validate_llm_endpoint_url

        _check_url(value, required=True)
        validate_llm_endpoint_url(value)
    elif key == "neuronpedia_local_url":
        _check_url(value, required=False)
    elif key == "neuronpedia_local_admin_user_id":
        if not str(value or "").strip():
            raise ValueError("must not be empty")
    elif key == "steering_cluster_constants_json":
        from ....services.cluster_allocation_service import validate_constants_json

        validate_constants_json(value)


def _validate(key: str, raw: Any) -> Any:
    try:
        value = runtime_settings.validated_value(key, raw)
        _extra_check(key, value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid_value: {key}: {exc}") from None
    return value


async def _last_changes(db: AsyncSession) -> dict[str, LastChange]:
    h = EnvironmentSettingChange
    rows = (
        await db.execute(
            select(h.key, h.actor, h.action, h.changed_at)
            .distinct(h.key)
            .order_by(h.key, h.changed_at.desc(), h.id.desc())
        )
    ).all()
    return {r.key: LastChange(actor=r.actor, action=r.action, at=r.changed_at) for r in rows}


@router.get("", response_model=EnvironmentListing)
async def list_environment(db: AsyncSession = Depends(get_db)) -> EnvironmentListing:
    """Every setting on the page, grouped: changeable now, needs a restart, from the environment."""
    stored = {
        row.key: row
        for row in (await db.execute(select(EnvironmentOverride))).scalars().all()
    }
    last = await _last_changes(db)
    in_force = runtime_settings.effective().model_dump(mode="json")
    env = config.settings.model_dump(mode="json")
    cached = runtime_settings.overrides()
    rows: dict[str, EnvironmentRow] = {}

    for key, spec in config.RUNTIME_KEYS.items():
        row = stored.get(key)
        rows[key] = EnvironmentRow(
            label=spec.label,
            env_name=_env_name(key),
            description=spec.description,
            group="runtime",
            value=in_force[key],
            env_value=env[key],
            source="override" if key in cached else "env",
            runtime_writable=True,
            effect=spec.effect,
            processes=list(spec.processes),
            updated_at=row.updated_at if row is not None else None,
            updated_by=row.updated_by if row is not None else None,
            reason=row.reason if row is not None else None,
            last_change=last.get(key),
        )

    for key, pod in config.RESTART_KEYS.items():
        rows[key] = EnvironmentRow(
            label=_label(key),
            env_name=_env_name(key),
            description=_description(key),
            group="restart",
            value=redact_url(env[key]),
            env_value=redact_url(env[key]),
            source="env",
            runtime_writable=False,
            restart=pod,
            restart_warning=config.BACKEND_RESTART_WARNING if pod.startswith("mistudio-backend") else None,
        )

    for name, pod, description in config.EXTERNAL_RESTART_KEYS:
        rows[name] = EnvironmentRow(
            label=_label(name.lower()),
            env_name=name,
            description=description,
            group="restart",
            source="external",
            runtime_writable=False,
            restart=pod,
            restart_warning=config.BACKEND_RESTART_WARNING if pod.startswith("mistudio-backend") else None,
        )

    for key in sorted(config.SHOWN_READ_ONLY):
        secret = key in config.SECRET_FIELDS
        rows[key] = EnvironmentRow(
            label=_label(key),
            env_name=_env_name(key),
            description=_description(key),
            group="environment",
            value=None if secret else redact_url(env[key]),
            env_value=None if secret else redact_url(env[key]),
            source="env",
            runtime_writable=False,
            secret=secret,
            is_set=bool(getattr(config.settings, key)) if secret else None,
        )

    return EnvironmentListing(
        settings=rows,
        count=len(rows),
        runtime_keys=list(config.RUNTIME_KEYS),
        overrides={k: v for k, v in cached.items()},
    )


@router.get("/history", response_model=EnvironmentHistory)
async def environment_history(
    limit: int = Query(50, ge=1, le=200),
    key: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> EnvironmentHistory:
    """The newest changes first (D12: the page shows the last 50)."""
    h = EnvironmentSettingChange
    stmt = select(h).order_by(h.changed_at.desc(), h.id.desc()).limit(limit)
    if key:
        stmt = stmt.where(h.key == key)
    rows = (await db.execute(stmt)).scalars().all()
    changes = [
        EnvironmentChange(
            id=r.id, key=r.key, action=r.action, previous_value=r.previous_value,
            previous_source=r.previous_source, new_value=r.new_value, new_source=r.new_source,
            actor=r.actor, reason=r.reason, changed_at=r.changed_at,
        )
        for r in rows
    ]
    return EnvironmentHistory(changes=changes, count=len(changes))


def _clean_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    reason = reason.strip()
    if len(reason) > REASON_MAX:
        raise HTTPException(status_code=422, detail=f"invalid_reason: at most {REASON_MAX} characters")
    return reason or None


@router.put("/{key}", response_model=EnvironmentWriteResult)
async def set_environment_override(
    key: str, body: EnvironmentWrite, db: AsyncSession = Depends(get_db)
) -> EnvironmentWriteResult:
    """Store an override of one runtime setting; it applies as the setting's row says."""
    _require_runtime(key)
    value = _validate(key, body.value)
    reason = _clean_reason(body.reason)

    row = await db.get(EnvironmentOverride, key)
    if row is not None:
        previous, previous_source = row.value, "override"
        row.value = value
        row.updated_at = func.now()
        row.updated_by = ACTOR
        row.reason = reason
    else:
        previous, previous_source = runtime_settings.env_value(key), "env"
        db.add(EnvironmentOverride(key=key, value=value, updated_by=ACTOR, reason=reason))
    db.add(
        EnvironmentSettingChange(
            key=key, action="set", previous_value=previous, previous_source=previous_source,
            new_value=value, new_source="override", actor=ACTOR, reason=reason,
        )
    )
    await db.commit()
    # Refresh point 1: this process sees its own change before the response is sent.
    await runtime_settings.refresh_async(db)
    logger.info("runtime setting %s overridden by %s", key, ACTOR)
    return EnvironmentWriteResult(key=key, value=value, previous=previous, source="override")


@router.delete("/{key}", response_model=EnvironmentWriteResult)
async def reset_environment_override(
    key: str,
    reason: Optional[str] = Query(None, max_length=REASON_MAX),
    db: AsyncSession = Depends(get_db),
) -> EnvironmentWriteResult:
    """Remove the override: the environment's value applies again."""
    _require_runtime(key)
    reason = _clean_reason(reason)
    row = await db.get(EnvironmentOverride, key)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"setting_not_overridden: {key!r} has no override; the environment's value is in force",
        )
    previous = row.value
    value = runtime_settings.env_value(key)
    await db.delete(row)
    db.add(
        EnvironmentSettingChange(
            key=key, action="reset", previous_value=previous, previous_source="override",
            new_value=value, new_source="env", actor=ACTOR, reason=reason,
        )
    )
    await db.commit()
    # Refresh point 1 (reset half).
    await runtime_settings.refresh_async(db)
    logger.info("runtime setting %s reset to the environment by %s", key, ACTOR)
    return EnvironmentWriteResult(key=key, value=value, previous=previous, source="env")
