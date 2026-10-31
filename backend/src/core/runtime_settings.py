"""Runtime overrides of the allow-listed settings (Feature 036; ENH-003 §2, §3.1; D2, D4).

Copied in shape from miForge ``backend/src/core/runtime_settings.py`` (F12, D5) and adapted to
miStudio's import-time ``settings`` singleton.

**One function every reader calls: :func:`effective`.** It is ``core.config.settings`` with the
in-process override cache laid over it — no I/O, so it is as cheap to call per request or per task as
reading ``settings`` itself. The singleton is never mutated and ``os.environ`` is never touched (the
steering spawn copies the environment; MIS-E2E-115).

**The cache is filled from ``environment_overrides`` at three wiring points**, each with a test that
fails when its line is removed (``test_environment_refresh_points.py``, ``test_environment_api.py``):

* the environment routes, right after a write commits (the API sees its own change at once);
* the API's lifespan startup (``main.py``), so a restart keeps the overrides;
* a Celery ``task_prerun`` receiver (:func:`connect_signals`, called from ``core/celery_app.py``), so
  every task in every worker process — general, processor, each per-card GPU worker, each steering
  worker — starts with the overrides as stored.

**Values are validated before they are cached**, by building ``Settings`` with the override applied,
so a value the model would refuse can never reach a reader through this path either. A stored row for
a key that is no longer runtime-writable, or whose value no longer validates, is logged and ignored.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()
_overrides: dict[str, Any] = {}
_loaded = False
#: The root log level this module last applied, or None when it has not overridden the process's own
#: logging configuration (FTID TI5).
_applied_log_level: Optional[str] = None


def effective() -> config.Settings:
    """The process settings with the runtime overrides applied."""
    base = config.settings
    with _lock:
        current = dict(_overrides)
    if not current:
        return base
    return base.model_copy(update=current)


def overrides() -> dict[str, Any]:
    """A copy of the overrides currently in force in this process."""
    with _lock:
        return dict(_overrides)


def is_loaded() -> bool:
    """Whether the cache has been filled in this process since the last :func:`reset`."""
    return _loaded


def env_value(key: str) -> Any:
    """The environment's (or the code default's) value of ``key``, JSON-compatible."""
    return config.settings.model_dump(mode="json")[key]


def validated_value(key: str, value: Any) -> Any:
    """``value`` as ``Settings`` would hold it for ``key``, or ``ValueError``.

    Built through the model's own validation, so the field's bounds and validators run.
    ``model_validate`` does not read the environment: validation sees exactly this process's
    values plus the one change. Returns the JSON-compatible form stored in the table.
    """
    if key not in config.RUNTIME_KEYS:
        raise ValueError(f"{key!r} is not a runtime setting")
    from pydantic import ValidationError

    try:
        built = config.Settings.model_validate({**config.settings.model_dump(), key: value})
    except ValidationError as exc:
        errors = exc.errors()
        message = "; ".join(str(e.get("msg", "")) for e in errors if key in tuple(e.get("loc", ()))) or (
            errors[0].get("msg", str(exc)) if errors else str(exc)
        )
        raise ValueError(message) from None
    return built.model_dump(mode="json")[key]


def apply(rows: dict[str, Any]) -> None:
    """Replace the cache with ``rows`` (key -> stored value), allow-listed and valid keys only."""
    global _loaded
    accepted: dict[str, Any] = {}
    for key, value in rows.items():
        if key not in config.RUNTIME_KEYS:
            # Only the environment routes write this table and they refuse every other key, so a
            # row here was written by hand. Ignored and said so, never applied.
            log.warning("ignoring a settings override for %r: it is not runtime-writable "
                        "(setting_override_ignored)", key)
            continue
        try:
            accepted[key] = validated_value(key, value)
        except ValueError as exc:
            log.warning("ignoring the stored override for %r: it no longer validates (%s) "
                        "(setting_override_invalid)", key, exc)
    with _lock:
        _overrides.clear()
        _overrides.update(accepted)
        _loaded = True
    _apply_log_level()


def _apply_log_level() -> None:
    """Make the root logger follow a ``log_level`` override (FTID TI5).

    Only an override moves it, and removing the override returns it to the environment's level.
    With no override ever set, the process's own configuration stands — Celery's ``--loglevel``,
    the API's ``configure_logging``, a test runner's capture — so a refresh is never a surprise.
    """
    global _applied_log_level
    with _lock:
        overridden = _overrides.get("log_level")
    root = logging.getLogger()
    if overridden is not None:
        if overridden != _applied_log_level or root.level != logging.getLevelName(overridden):
            root.setLevel(overridden)
            _applied_log_level = overridden
    elif _applied_log_level is not None:
        root.setLevel(config.settings.log_level)
        _applied_log_level = None


async def refresh_async(db) -> dict[str, Any]:
    """Read every override through the async session ``db`` and replace the cache."""
    from sqlalchemy import select

    from ..models.environment_setting import EnvironmentOverride

    result = await db.execute(select(EnvironmentOverride.key, EnvironmentOverride.value))
    apply({str(row.key): row.value for row in result.all()})
    return overrides()


def refresh_sync(session) -> dict[str, Any]:
    """Read every override through the sync session ``session`` and replace the cache."""
    from sqlalchemy import select

    from ..models.environment_setting import EnvironmentOverride

    rows = session.execute(select(EnvironmentOverride.key, EnvironmentOverride.value)).all()
    apply({str(row.key): row.value for row in rows})
    return overrides()


async def refresh_from_database_async() -> bool:
    """:func:`refresh_async` through a session of its own — the API startup wiring point.

    A failure leaves the previous cache in place and is logged: the environment's values still
    apply, which is what an absent override means, so a settings read never fails the startup.
    """
    try:
        from .database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            await refresh_async(db)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read the runtime settings overrides; environment values apply "
                    "(setting_overrides_unreadable: %s)", type(exc).__name__)
        return False
    return True


def refresh_from_database_sync() -> bool:
    """:func:`refresh_sync` through a session of its own — the Celery ``task_prerun`` wiring point."""
    try:
        from .database import SyncSessionLocal

        session = SyncSessionLocal()
        try:
            refresh_sync(session)
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read the runtime settings overrides; environment values apply "
                    "(setting_overrides_unreadable: %s)", type(exc).__name__)
        return False
    return True


def on_task_prerun(sender: Any = None, task_id: Optional[str] = None, **_: Any) -> None:
    """``task_prerun`` receiver: every task starts with the overrides as stored."""
    refresh_from_database_sync()


def connect_signals() -> None:
    """Wire :func:`on_task_prerun` to every task this process runs. Called from ``core.celery_app``.

    ``weak=False`` so the wiring never depends on a reference being held; ``dispatch_uid`` makes a
    second call a no-op.
    """
    from celery.signals import task_prerun

    task_prerun.connect(on_task_prerun, weak=False, dispatch_uid="mistudio-runtime-settings-prerun")


def reset() -> None:
    """Empty the cache. For tests, and for a test that simulates a fresh process."""
    global _loaded, _applied_log_level
    with _lock:
        _overrides.clear()
        _loaded = False
    if _applied_log_level is not None:
        logging.getLogger().setLevel(config.settings.log_level)
    _applied_log_level = None
