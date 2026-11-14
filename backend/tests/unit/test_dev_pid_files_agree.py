"""The development launchers and the backend agree on every pid file's name (042 FR-042.31; FTASKS 15.7).

P4 renamed the pid files `/tmp/mistudio-*` to `/tmp/miworkbench-*`. Three places name the steering
worker's file (`backend/celery.sh`, `services/steering_worker_card.py`, `api/v1/endpoints/steering.py`) and
two more name the others (`start-miworkbench.sh` reads what `celery.sh` writes; `stop-miworkbench.sh` stops
what `start-miworkbench.sh` started). A rename applied to one side only leaves a worker the backend cannot
find, or a stop script that stops nothing while reporting success. Control M-15 (2026-11-11) changed
`celery.sh` alone and no test failed; this file is the test that was missing.

Read from the scripts' own text (their assignments and the paths they test), never a copy kept here.
On a mirror-shaped tree the launchers are present (only `scripts/` is removed), so nothing skips.
"""
from __future__ import annotations

import re
from pathlib import Path

from tests.support import legacy_export as le

REPO = le.REPO_ROOT
ASSIGN = re.compile(r'^([A-Z_]*PID_FILE)="(/tmp/[^"]+)"', re.MULTILINE)
PID_PATH = re.compile(r"/tmp/[A-Za-z0-9_.-]+\.pid")


def _text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def celery_pid_files() -> dict[str, str]:
    found = dict(ASSIGN.findall(_text("backend/celery.sh")))
    assert {"WORKER_PID_FILE", "STEERING_PID_FILE", "BEAT_PID_FILE"} <= set(found), sorted(found)
    return found


def test_steering_pid_file_is_one_name_in_three_places():
    from src.api.v1.endpoints import steering as endpoint
    from src.services import steering_worker_card as card

    script = Path(celery_pid_files()["STEERING_PID_FILE"]).name
    assert script == card.pid_file(None).name == Path(endpoint.PID_FILE).name
    assert script.startswith("miworkbench-")


def test_start_script_checks_the_files_celery_sh_writes():
    written = set(celery_pid_files().values())
    checked = {p for p in PID_PATH.findall(_text("start-miworkbench.sh")) if "celery" in p}
    assert checked, "start-miworkbench.sh checks no celery pid file: nothing was read"
    assert checked <= written, sorted(checked - written)


def test_stop_script_stops_what_the_start_script_started():
    started = dict(ASSIGN.findall(_text("start-miworkbench.sh")))
    assert {"BACKEND_PID_FILE", "FRONTEND_PID_FILE"} <= set(started), sorted(started)
    stopped = set(PID_PATH.findall(_text("stop-miworkbench.sh")))
    assert {started["BACKEND_PID_FILE"], started["FRONTEND_PID_FILE"]} <= stopped, sorted(stopped)


def test_no_launcher_keeps_an_old_pid_file_name():
    for rel in ("backend/celery.sh", "start-miworkbench.sh", "stop-miworkbench.sh", "restart-miworkbench.sh",
                "miworkbench"):
        old = [p for p in PID_PATH.findall(_text(rel)) if re.search(r"mi(?:studio|llm)", p, re.IGNORECASE)]
        assert old == [], (rel, old)
