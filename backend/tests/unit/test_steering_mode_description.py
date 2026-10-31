"""The steering-mode tools describe what the code does (live GPU checks, 2026-10-20 22:17 ET).

Live: the enter-mode worker held no GPU memory, each submit started a new worker (PIDs 172→239,
380→447), and the task's worker left the card 3-4 s after success. The tool text said enter
"LOADS THE MODEL+SAE ONTO THE GPU" and that without exit the worker "holds VRAM indefinitely".

The code agrees with the live run, by design: `_ensure_steering_worker_running` /
`ensure_card_worker` replace an idle worker with a fresh one at every submit, and
`workers/steering_tasks.py` warm-shuts the worker after each generation task (`_SELF_EXIT_TASKS`).
`--max-tasks-per-child=1` on the spawn line is NOT the mechanism: under `--pool=solo` Celery
ignores it (MIS-E2E-094, `core/celery_app.py`), so pinning that flag would pin a no-op. These tests
pin the real mechanism and the corrected text together, so the text cannot drift from it again.

Controls (review record `0xcc/reviews/2026-10-21_mcp-gaps.md`):
  C-TEXT   restore "holds VRAM indefinitely" in enter_steering_mode → the text test red
  C-EXIT   drop "steering.sweep" from _SELF_EXIT_TASKS → test_every_generation_task_exits_its_worker red
"""
import asyncio

import pytest


def _descriptions():
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import howto, steering

    mcp = MCPServer("t")
    steering.register(mcp, object(), MCPSettings(auth_token="x" * 32, _env_file=None))
    tools = {t.name: t.description for t in asyncio.run(mcp.list_tools())}
    return tools, howto


@pytest.mark.parametrize("claim", ["holds VRAM indefinitely", "LOADS THE MODEL+SAE ONTO THE GPU",
                                   "nothing else reaps"])
def test_no_tool_text_makes_the_disproved_claims(claim):
    tools, _ = _descriptions()
    for name in ("enter_steering_mode", "exit_steering_mode", "get_steering_mode"):
        assert claim not in tools[name], (name, claim)


def test_enter_says_it_loads_nothing_and_each_task_gets_a_fresh_worker():
    tools, _ = _descriptions()
    text = tools["enter_steering_mode"]
    assert "LOADS NOTHING" in text and "FRESH" in text and "exits by itself" in text


def test_the_server_instructions_and_howto_agree():
    from src.mcp_server.server import SERVER_INSTRUCTIONS

    _, howto = _descriptions()
    import inspect

    howto_text = inspect.getsource(howto)
    for text in (SERVER_INSTRUCTIONS, howto_text):
        assert "MANDATORY when done" not in text and "nothing else reaps" not in text
        assert "holds VRAM indefinitely otherwise" not in text
    assert "loads nothing" in SERVER_INSTRUCTIONS


def test_every_generation_task_exits_its_worker():
    """The mechanism the text now relies on."""
    from src.workers import steering_tasks

    assert {"steering.compare", "steering.sweep", "steering.combined"} <= steering_tasks._SELF_EXIT_TASKS


def test_postrun_of_a_generation_task_schedules_the_exit(monkeypatch):
    from src.workers import steering_tasks

    scheduled = []
    monkeypatch.setattr(steering_tasks, "_pidfile_is_ours", lambda: True)
    monkeypatch.setattr(steering_tasks, "_schedule_self_exit", lambda *a, **k: scheduled.append(1))
    monkeypatch.setattr(steering_tasks._worker_state, "clear_busy_marker", lambda *a, **k: None)
    sender = type("S", (), {"name": "steering.sweep"})()
    steering_tasks._steering_task_postrun(sender=sender)
    assert scheduled == [1]


def test_the_worker_runs_solo_so_max_tasks_per_child_is_not_the_mechanism():
    """If the spawn ever moves off --pool=solo, the reasoning above must be revisited."""
    import inspect

    from src.api.v1 import steering_workers

    source = inspect.getsource(steering_workers.spawn_card_worker)
    assert '"--pool=solo"' in source
