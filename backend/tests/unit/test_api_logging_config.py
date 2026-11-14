"""The API process honours LOG_LEVEL (live 2026-10-20, tracker 3.4 A1).

`Settings.log_level` was parsed and never applied: uvicorn configures only its
own loggers, so `src.*` INFO lines — the runbook's `[CircuitService] typed N
edge(s)` among them — never reached `kubectl logs`.

Run in a SUBPROCESS: the test runner configures logging in this process, and a
test of "who configured the root logger" must not share it.

NEGATIVE CONTROL: delete the `configure_logging(...)` call in `src/main.py` ->
the INFO test goes red (the record is not emitted).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2]

PROBE = r"""
import logging, src.main
logging.getLogger("src.services.circuit_service").info(
    "[CircuitService] typed %d edge(s): %s", 5, {"computed": 5})
logging.getLogger("src.services.circuit_service").debug("debug-line-should-not-appear")
"""


def _run(**env):
    full = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, "-c", PROBE], cwd=BACKEND, env=full,
        capture_output=True, text=True, timeout=180,
    )


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_an_info_line_from_the_app_is_emitted(fmt):
    r = _run(LOG_LEVEL="INFO", LOG_FORMAT=fmt)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "[CircuitService] typed 5 edge(s)" in r.stderr, (
        "an app INFO line was dropped: LOG_LEVEL=INFO is not applied in the API process"
    )
    assert "debug-line-should-not-appear" not in r.stderr
    if fmt == "json":
        line = next(l for l in r.stderr.splitlines() if "typed 5 edge" in l)
        assert line.startswith("{") and '"level": "INFO"' in line


def test_a_higher_level_is_honoured_too():
    r = _run(LOG_LEVEL="WARNING", LOG_FORMAT="text")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "typed 5 edge" not in r.stderr


def test_an_already_configured_root_is_left_alone():
    from src.core.logging_setup import configure_logging
    import logging

    root = logging.getLogger()
    sentinel = logging.NullHandler()
    saved = (root.handlers[:], root.level)
    root.handlers[:] = [sentinel]
    try:
        assert configure_logging("INFO", "text") is False
        assert root.handlers == [sentinel], "a second handler would print every record twice"
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
