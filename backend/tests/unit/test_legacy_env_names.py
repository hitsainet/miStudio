"""Old environment names are read new-first, honoured alone, and recorded (042 FR-042.8; FTID §3.1, §3.2;
FTASKS 3.6). One parameter per `legacy_names.ENV` row, each read through its REAL reader: the two Settings
fields through `Settings()`, the MCP URLs through `MCPSettings`, the GPU directories through the worker
functions that read them. The values are never returned by the evidence (names only, BR-023).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from src.core import legacy_names

ALL_NAMES = sorted({n for r in legacy_names.ENV for n in (r.new, r.old)})


def _read(new: str):
    """The value the production code ends up with for row `new`."""
    from src.core.config import Settings

    if new == "MISERVER_API_URL":
        backend = Settings(_env_file=None).miserver_api_url
        from src.mcp_server.config import MCPSettings

        mcp = MCPSettings(_env_file=None).miserver_api_url
        assert backend == mcp, (backend, mcp)
        return backend
    if new == "MIWORKBENCH_BYPASS_PIN":
        return Settings(_env_file=None).bypass_settings_pin
    if new == "MIWORKBENCH_API_URL":
        from src.mcp_server.config import MCPSettings

        return MCPSettings(_env_file=None).api_url
    if new == "MIWORKBENCH_GPU_RECYCLE_DIR":
        from src.workers import gpu_worker_recycle

        return legacy_names.env_from(None, gpu_worker_recycle.RECYCLE_DIR_ENV)
    if new == "MIWORKBENCH_GPU_RECYCLE_RSS_GIB":
        from src.workers import gpu_worker_recycle

        return gpu_worker_recycle.threshold_bytes() / (1 << 30)
    if new == "MIWORKBENCH_GPU_WORKER_MEMORY_DIR":
        from src.services import gpu_worker_memory

        return gpu_worker_memory.memory_dir()
    raise AssertionError(f"no reader for {new}")


VALUES = {"MISERVER_API_URL": ("http://new-srv", "http://old-srv"),
          "MIWORKBENCH_BYPASS_PIN": ("true", "true"),
          "MIWORKBENCH_API_URL": ("http://new-wb", "http://old-wb"),
          "MIWORKBENCH_GPU_RECYCLE_DIR": ("/tmp/new-recycle", "/tmp/old-recycle"),
          # Values that cannot occur in a name or an ISO timestamp: the leak check below is a substring test,
          # and "7" matched every snapshot stamped in a minute, second, hour, day or year holding a 7
          # (review rev-mistudio-p0 F8: red on ~1 run in 5 today, on every run on 2026-11-17 and all of 2027).
          "MIWORKBENCH_GPU_RECYCLE_RSS_GIB": ("6.5", "7.0625"),
          "MIWORKBENCH_GPU_WORKER_MEMORY_DIR": ("/tmp/new-mem", "/tmp/old-mem")}
EXPECTED = {"MISERVER_API_URL": lambda v: v, "MIWORKBENCH_BYPASS_PIN": lambda v: True,
            "MIWORKBENCH_API_URL": lambda v: v, "MIWORKBENCH_GPU_RECYCLE_DIR": lambda v: v,
            "MIWORKBENCH_GPU_RECYCLE_RSS_GIB": lambda v: float(v), "MIWORKBENCH_GPU_WORKER_MEMORY_DIR": lambda v: v}
ROWS = {r.new: r for r in legacy_names.ENV}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ALL_NAMES + ["BYPASS_SETTINGS_PIN"]:
        monkeypatch.delenv(name, raising=False)
    legacy_names.reset()
    yield
    legacy_names.reset()


def _uses():
    return {(u["kind"], u["old"]) for u in legacy_names.snapshot()["in_use"]}


def test_every_row_has_a_reader():
    assert set(VALUES) == set(ROWS), "a new ENV row needs a value and a reader here"


@pytest.mark.parametrize("new", sorted(ROWS))
def test_new_name_alone(new, monkeypatch):
    monkeypatch.setenv(new, VALUES[new][0])
    assert _read(new) == EXPECTED[new](VALUES[new][0])
    assert _uses() == set(), "the new name alone records nothing"


def _transition(monkeypatch, new):
    """Put row `new` back in its transition state for one test: the recording mechanism stays tested while
    the retired rows are still in the table (they go at P7's close, FTASKS 18.15)."""
    row = ROWS[new]
    live = legacy_names.EnvName(new=row.new, old=row.old, status="transition")
    monkeypatch.setitem(legacy_names._ENV_BY_NEW, row.new, live)  # noqa: SLF001
    monkeypatch.setitem(legacy_names._ENV_BY_OLD, row.old, live)  # noqa: SLF001
    monkeypatch.setattr(legacy_names, "ENV", tuple(live if r.new == row.new else r for r in legacy_names.ENV))
    return live


@pytest.mark.parametrize("new", sorted(ROWS))
def test_old_name_alone_is_not_read_since_p7(new, monkeypatch):
    """T-07's old-alone case INVERTED at ENH-009 P7 (FTASKS 18.4): every row is retired, so the old name alone
    is not read, not recorded, and reported in `retired_but_set` (`test_legacy_retired.py` holds the inverse
    the P7 control turns red)."""
    row = ROWS[new]
    assert row.status == "retired"
    unset = _read(new)
    legacy_names.reset()
    monkeypatch.setenv(row.old, VALUES[new][1])
    assert _read(new) == unset
    snap = legacy_names.snapshot()
    assert [u for u in snap["in_use"] if u["old"] == row.old] == []
    assert {"old": row.old, "new": row.new} in snap["retired_but_set"]
    assert VALUES[new][1] not in repr(snap), "a value reached the evidence"


@pytest.mark.parametrize("new", sorted(ROWS))
def test_a_transition_old_name_alone_is_honoured_and_recorded(new, monkeypatch):
    """The transition mechanism, on a row put back into transition (the P0 – P6 guarantee)."""
    row = _transition(monkeypatch, new)
    monkeypatch.setenv(row.old, VALUES[new][1])
    assert _read(new) == EXPECTED[new](VALUES[new][1])
    snap = legacy_names.snapshot()
    entries = [u for u in snap["in_use"] if u["old"] == row.old]
    assert entries and set(entries[0]) == {"kind", "old", "new", "last_seen"}
    assert entries[0]["kind"] == "env" and entries[0]["new"] == row.new
    assert VALUES[new][1] not in repr(snap), "a value reached the evidence"


def test_both_set_new_wins_and_also_set(monkeypatch):
    monkeypatch.setenv("MISERVER_API_URL", "http://same")
    monkeypatch.setenv("MILLM_API_URL", "http://same")
    assert _read("MISERVER_API_URL") == "http://same"
    snap = legacy_names.snapshot()
    assert {"old": "MILLM_API_URL", "new": "MISERVER_API_URL"} in snap["also_set"]
    assert snap["conflicts"] == []
    assert ("env", "MILLM_API_URL") not in _uses(), "the new name is in force: the old one is not in use"


def test_differing_values_are_conflicts(monkeypatch):
    monkeypatch.setenv("MIWORKBENCH_API_URL", "http://new")
    monkeypatch.setenv("MISTUDIO_API_URL", "http://old")
    assert _read("MIWORKBENCH_API_URL") == "http://new"
    snap = legacy_names.snapshot()
    pair = {"old": "MISTUDIO_API_URL", "new": "MIWORKBENCH_API_URL"}
    assert pair in snap["also_set"] and pair in snap["conflicts"]
    assert "http://old" not in repr(snap) and "http://new" not in repr(snap)


def test_retired_row_reports_retired_but_set(monkeypatch):
    row = ROWS["MIWORKBENCH_GPU_WORKER_MEMORY_DIR"]
    retired = legacy_names.EnvName(new=row.new, old=row.old, status="retired")
    monkeypatch.setitem(legacy_names._ENV_BY_NEW, row.new, retired)  # noqa: SLF001
    monkeypatch.setattr(legacy_names, "ENV", tuple(retired if r.new == row.new else r for r in legacy_names.ENV))
    monkeypatch.setenv(row.old, "/tmp/old-mem")
    from src.services import gpu_worker_memory

    assert gpu_worker_memory.memory_dir() == gpu_worker_memory.DEFAULT_MEMORY_DIR, "a retired old name is not read"
    snap = legacy_names.snapshot()
    assert {"old": row.old, "new": row.new} in snap["retired_but_set"]
    assert ("env", row.old) not in _uses()


def test_dotenv_old_name_is_not_read_since_p7(tmp_path):
    """INVERTED at ENH-009 P7: a retired old name in the `.env` file is not read; it is reported."""
    from src.core.config import Settings

    env_file = tmp_path / ".env"
    env_file.write_text("MILLM_API_URL=http://from-dotenv\nMISTUDIO_BYPASS_PIN=true\n")
    s = Settings(_env_file=env_file)
    assert s.miserver_api_url == ""
    assert s.bypass_settings_pin is False
    assert _uses() == set()
    assert {"old": "MILLM_API_URL", "new": "MISERVER_API_URL"} in legacy_names.snapshot()["retired_but_set"]


def test_a_transition_dotenv_old_name_is_honoured(tmp_path, monkeypatch):
    from src.core.config import Settings

    _transition(monkeypatch, "MISERVER_API_URL")
    _transition(monkeypatch, "MIWORKBENCH_BYPASS_PIN")
    env_file = tmp_path / ".env"
    env_file.write_text("MILLM_API_URL=http://from-dotenv\nMISTUDIO_BYPASS_PIN=true\n")
    s = Settings(_env_file=env_file)
    assert s.miserver_api_url == "http://from-dotenv"
    assert s.bypass_settings_pin is True
    assert {("env", "MILLM_API_URL"), ("env", "MISTUDIO_BYPASS_PIN")} <= _uses()


def test_reset_has_no_production_caller():
    """`legacy_names.reset()` empties the evidence registry; it is a test hook (FTID §3.1). An AST walk over
    every module under `src/` (not a text search: a comment naming it must not count)."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    files = sorted(src.rglob("*.py"))
    assert len(files) > 100, "the walk saw too few modules"
    callers = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("legacy_names") \
                    and any(a.name == "reset" for a in node.names):
                callers.append(f"{path}:{node.lineno} imports reset")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reset" \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "legacy_names":
                callers.append(f"{path}:{node.lineno}")
    assert callers == [], callers


def test_an_empty_new_name_is_unset_in_the_backend_as_in_the_mcp_pod(monkeypatch):
    """Review rev-mistudio-p0 F6: `MISERVER_API_URL=""` with `MILLM_API_URL` set. `legacy_names.env()` (the MCP
    pod) reads the old address; the backend's Settings read the empty new value and reported miServer
    unconnected. Both now read the old one, and record it."""
    from src.core.config import Settings
    from src.mcp_server.config import MCPSettings

    _transition(monkeypatch, "MISERVER_API_URL")  # the transition rule this pins (retired since P7)
    monkeypatch.setenv("MISERVER_API_URL", "")
    monkeypatch.setenv("MILLM_API_URL", "http://old-name:8000")
    assert MCPSettings(_env_file=None).miserver_api_url == "http://old-name:8000"
    assert Settings(_env_file=None).miserver_api_url == "http://old-name:8000"
    assert ("env", "MILLM_API_URL") in _uses()


def test_the_leak_check_values_cannot_occur_in_a_snapshot():
    """Review rev-mistudio-p0 F8: the substring leak check above is only as good as its values. Each old value
    must be impossible inside a name or a `YYYY-MM-DDTHH:MM:SSZ` stamp, at any time of any day."""
    import re

    names = {n for r in legacy_names.ENV for n in (r.new, r.old)} | {h.old for h in legacy_names.HOSTS} \
        | {h.new for h in legacy_names.HOSTS}
    for new, (_new_value, old_value) in VALUES.items():
        if old_value == "true":  # a boolean flag: no snapshot field holds a boolean's text
            continue
        assert not re.fullmatch(r"[0-9TZ:-]+", old_value) and not any(old_value in n for n in names), (
            f"{new}: {old_value!r} can occur in a timestamp or a name, so the leak check can fail by the clock")
