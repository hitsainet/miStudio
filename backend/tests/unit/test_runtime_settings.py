"""The override accessor itself (Feature 036, FTASKS 2.1; ENH-003 §2 "One reader", "Validation").

``test_environment_readers.py`` drives every reader through :func:`runtime_settings.effective`; this
file pins the accessor's own rules, which every one of those tests relies on:

* no override → the import-time ``settings`` singleton itself (no copy, no surprise);
* an override is laid over the singleton, which is never mutated, and ``os.environ`` is never touched
  (the steering spawn copies the environment, MIS-E2E-115);
* a value is validated through ``Settings`` before it is cached — the field's own bounds;
* a row for a key outside ``RUNTIME_KEYS``, or one that no longer validates, is ignored and logged;
* ``apply`` REPLACES the cache, so a reset row (deleted) stops applying at the next refresh;
* the sync and async refresh read the table.
"""
from __future__ import annotations

import logging
import os

import pytest

from src.core import config, runtime_settings
from src.core.config import settings


@pytest.fixture(autouse=True)
def _fresh_cache():
    runtime_settings.reset()
    yield
    runtime_settings.reset()


def _other_batch_size() -> int:
    return 3 if settings.labeling_batch_size != 3 else 4


def test_no_override_is_the_singleton_itself():
    assert runtime_settings.effective() is config.settings
    assert runtime_settings.overrides() == {}


def test_an_override_is_laid_over_and_the_singleton_is_untouched():
    env_before = dict(os.environ)
    base = settings.labeling_batch_size
    runtime_settings.apply({"labeling_batch_size": _other_batch_size()})

    assert runtime_settings.effective().labeling_batch_size == _other_batch_size()
    assert settings.labeling_batch_size == base
    assert runtime_settings.env_value("labeling_batch_size") == base
    assert dict(os.environ) == env_before
    assert runtime_settings.is_loaded()


def test_apply_replaces_the_cache_so_a_reset_row_stops_applying():
    runtime_settings.apply({"labeling_batch_size": _other_batch_size()})
    runtime_settings.apply({})  # the row was deleted; the next refresh reads nothing
    assert runtime_settings.effective().labeling_batch_size == settings.labeling_batch_size


@pytest.mark.parametrize(
    "key, value",
    [("labeling_batch_size", 0), ("labeling_batch_size", 33), ("system_monitor_interval_seconds", 31),
     ("steering_hazard_prior_threshold", 1.5), ("log_level", "LOUD")],
)
def test_validated_value_uses_the_fields_own_bounds(key, value):
    with pytest.raises(ValueError):
        runtime_settings.validated_value(key, value)


def test_validated_value_returns_the_settings_form():
    assert runtime_settings.validated_value("labeling_batch_size", "5") == 5


def test_validated_value_refuses_a_key_outside_the_allow_list():
    with pytest.raises(ValueError, match="not a runtime setting"):
        runtime_settings.validated_value("database_url", "postgresql://x")


def test_a_non_runtime_row_is_ignored_and_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="src.core.runtime_settings"):
        runtime_settings.apply({"database_url": "postgresql://evil", "labeling_batch_size": _other_batch_size()})
    assert runtime_settings.overrides() == {"labeling_batch_size": _other_batch_size()}
    assert runtime_settings.effective().database_url == settings.database_url
    assert "setting_override_ignored" in caplog.text


def test_an_invalid_row_is_ignored_and_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="src.core.runtime_settings"):
        runtime_settings.apply({"labeling_batch_size": 999})
    assert runtime_settings.overrides() == {}
    assert "setting_override_invalid" in caplog.text


def test_removing_a_log_level_override_returns_the_root_logger_to_the_environment():
    root = logging.getLogger()
    before = root.level
    target = "ERROR" if settings.log_level != "ERROR" else "WARNING"
    try:
        runtime_settings.apply({"log_level": target})
        assert root.level == logging.getLevelName(target)
        runtime_settings.apply({})
        assert root.level == logging.getLevelName(settings.log_level)
    finally:
        root.setLevel(before)


async def test_refresh_async_reads_the_table(async_session):
    from src.models.environment_setting import EnvironmentOverride

    async_session.add(EnvironmentOverride(key="labeling_batch_size", value=_other_batch_size()))
    await async_session.commit()
    assert await runtime_settings.refresh_async(async_session) == {"labeling_batch_size": _other_batch_size()}
    assert runtime_settings.effective().labeling_batch_size == _other_batch_size()


def test_on_task_prerun_refreshes_from_the_database(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime_settings, "refresh_from_database_sync", lambda: calls.append(1) or True)
    runtime_settings.on_task_prerun(sender=None, task_id="t")
    assert calls == [1]
