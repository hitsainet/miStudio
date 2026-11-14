"""Refresh points 2 and 3 (Feature 036; ENH-003 §4 "Refresh points"; FTASKS 2.3, 2.4).

Point 1 (after a write) is in ``test_environment_api.py``. Each test here stores an override row the
way the route does, then runs the REAL wiring — the app's lifespan, a Celery task sent through the live
``task_prerun`` signal — and asserts the value the accessor (and a task body) sees afterwards.

Controls (``0xcc/reviews/036_implementation_controls_2026-10-20.md``):
  C-RP2  delete the ``refresh_from_database_async()`` call in ``main.lifespan`` → the startup test is red
  C-RP3  delete ``_connect_runtime_settings()`` in ``core/celery_app.py``     → both prerun tests are red
"""
from __future__ import annotations

import logging

import pytest
from sqlalchemy import delete

from src.core import runtime_settings
from src.core.config import settings
from src.models.environment_setting import EnvironmentOverride


@pytest.fixture(autouse=True)
def _fresh_cache():
    runtime_settings.reset()
    yield
    runtime_settings.reset()


@pytest.fixture
def stored(async_engine):
    """Write override rows through the sync engine the workers use."""
    from src.core.database import SyncSessionLocal

    def put(**values):
        with SyncSessionLocal() as db:
            db.execute(delete(EnvironmentOverride))
            for key, value in values.items():
                db.add(EnvironmentOverride(key=key, value=value, updated_by="operator"))
            db.commit()

    return put


async def test_the_api_startup_loads_the_stored_overrides(stored, monkeypatch):
    """A restarted API keeps the operator's overrides — including the log level (spec §3.4 gap 1)."""
    from src import main
    from src.services import background_monitor

    stored(labeling_batch_size=3, log_level="ERROR")

    class Inert:
        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(main, "get_background_monitor", lambda: Inert())
    monkeypatch.setattr(background_monitor, "_background_monitor", None)
    root = logging.getLogger()
    before = root.level
    assert runtime_settings.effective().labeling_batch_size == settings.labeling_batch_size

    async with main.lifespan(main.app):
        assert runtime_settings.is_loaded()
        assert runtime_settings.effective().labeling_batch_size == 3
        assert root.level == logging.ERROR
    root.setLevel(before)


def test_task_prerun_is_wired_in_the_live_signal_registry():
    """Asserted on the registry the worker really uses, after importing the Celery app."""
    import src.core.celery_app  # noqa: F401  the import is what wires it
    from celery.signals import task_prerun

    keys = [key for key, _ref in task_prerun.receivers if key[0] == "mistudio-runtime-settings-prerun"]
    # Exactly one: `dispatch_uid` makes a second connect a no-op. This test never connects it
    # itself, so the task-body test below still depends on the celery_app line alone.
    assert len(keys) == 1, f"{len(keys)} runtime-settings receivers on task_prerun"


def test_a_task_body_sees_the_override_stored_before_it_started(stored):
    """Refresh point 3, driven: the stored row reaches a real task body through `task_prerun`."""
    from src.core.celery_app import celery_app

    seen = {}

    @celery_app.task(name="tests.environment.probe_036")
    def probe():
        seen["batch"] = runtime_settings.effective().labeling_batch_size
        seen["level"] = logging.getLogger().level
        return seen["batch"]

    before = logging.getLogger().level
    stored(labeling_batch_size=5, log_level="CRITICAL")
    try:
        result = probe.apply()
        assert result.successful(), result.traceback
        assert seen == {"batch": 5, "level": logging.CRITICAL}

        # And a reset reaches the NEXT task.
        stored()
        probe.apply()
        assert seen["batch"] == settings.labeling_batch_size
    finally:
        logging.getLogger().setLevel(before)
        celery_app.tasks.pop("tests.environment.probe_036", None)
