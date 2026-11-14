"""A silent worker can be asked where it stands (ENH-006 D116 G12).

A real process, a real signal, a real stack: the property is that the bytes reach stderr and name every
thread. It does not prove the C handler works with the GIL held, which is `faulthandler`'s documented behaviour.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[2]

PROGRAM = """
import os, signal, sys, threading
sys.path.insert(0, {backend!r})
from src.core.celery_app import on_worker_ready
on_worker_ready(type("S", (), {{"hostname": "t"}})())
def waiting_thread():
    threading.Event().wait()
threading.Thread(target=waiting_thread, daemon=True, name="waiter").start()
os.kill(os.getpid(), signal.SIGUSR2)
print("still running")
"""


def test_the_signal_dumps_every_thread_and_the_worker_carries_on():
    done = subprocess.run([sys.executable, "-c", PROGRAM.format(backend=str(BACKEND))], capture_output=True,
                          text=True, timeout=180, cwd=BACKEND)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "still running" in done.stdout
    assert "most recent call first" in done.stderr
    assert "waiting_thread" in done.stderr, "only the current thread was dumped"
