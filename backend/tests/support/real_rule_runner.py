"""miStudio's REAL runner host as a subprocess, for backend tests that need operator code actually run (Feature 040 S4;
FTDD Appendix C.9; the `rule-runner-tests` CI job).

The guest snapshot is the PUBLISHED one (`guest.bundle.snapshot` of the vendored runner contract), fetched by digest
into ``$RUNNER_BUNDLE_DIR`` (``python -m src.rule_runner.fetch --out DIR``); it is never committed. Without it — or
without ``wasmtime`` importable by the runner's interpreter — these tests SKIP LOUDLY, and FAIL instead under
``MISTUDIO_REQUIRE_RUNNER=1`` (the CI job sets it).

The runner is started the way its image runs it: the package `backend/src/rule_runner/` imported as the top-level
`mistudio_rule_runner` (a symlink in a temporary import root), with an environment of ``PATH``, ``PYTHONPATH`` and
``RUNNER_*`` only — no database URL, no secret — plus whatever SENTINELS a test adds on purpose.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest

BACKEND = Path(__file__).resolve().parents[2]
PACKAGE = BACKEND / "src" / "rule_runner"
CONTRACT = BACKEND / "src" / "schemas" / "vendored" / "midataworks-runner-v1.json"
REQUIRED = os.environ.get("MISTUDIO_REQUIRE_RUNNER") == "1"


def _skip(message: str) -> None:
    if REQUIRED:
        pytest.fail("RULE RUNNER: " + message + " (MISTUDIO_REQUIRE_RUNNER=1)")
    pytest.skip("RULE RUNNER: " + message + " — the real-runner tests did NOT run")


def bundle_dir() -> Path:
    env = os.environ.get("RUNNER_BUNDLE_DIR")
    if not env:
        _skip("RUNNER_BUNDLE_DIR is not set; fetch the published snapshot with "
              "`python -m src.rule_runner.fetch --out DIR`")
    directory = Path(env)
    snapshot = directory / "snapshot.wasm"
    if not snapshot.is_file():
        _skip(f"no snapshot.wasm in {directory}")
    want = json.loads(CONTRACT.read_text(encoding="utf-8"))["guest"]["bundle"]["snapshot"]["sha256"]
    got = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if got != want:
        _skip(f"snapshot.wasm digest {got} is not the contract's {want}")
    return directory


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class RealRunner:
    """`with RealRunner(env={...}, files={...}) as runner:` — `runner.url` while it runs; `runner.output` after."""

    def __init__(self, *, env: Optional[Dict[str, str]] = None, files: Optional[Dict[str, str]] = None) -> None:
        self.bundle = bundle_dir()
        self.extra_env = dict(env or {})
        self.files = dict(files or {})
        self.root = Path(tempfile.mkdtemp(prefix="mrr-backend-"))
        (self.root / "mistudio_rule_runner").symlink_to(PACKAGE, target_is_directory=True)
        self.cwd = self.root / "cwd"
        self.cwd.mkdir()
        for name, content in self.files.items():  # sentinel files the runner PROCESS can read
            (self.cwd / name).write_text(content, encoding="utf-8")
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: Optional[subprocess.Popen] = None
        self.output: Tuple[bytes, bytes] = (b"", b"")

    def env(self) -> Dict[str, str]:
        paths = [str(self.root)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": os.pathsep.join(paths),
                "RUNNER_BUNDLE_DIR": str(self.bundle), "RUNNER_CONTRACT_PATH": str(CONTRACT),
                "RUNNER_HOST": "127.0.0.1", "RUNNER_PORT": str(self.port), **self.extra_env}

    def start(self) -> "RealRunner":
        import httpx

        probe = subprocess.run([sys.executable, "-c", "import wasmtime"], env=self.env(), capture_output=True)
        if probe.returncode != 0:
            _skip("wasmtime is not importable by the runner's interpreter (install rule_runner/requirements.txt)")
        self.proc = subprocess.Popen([sys.executable, "-m", "mistudio_rule_runner.server"], env=self.env(),
                                     cwd=str(self.cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                _out, err = self.proc.communicate()
                raise RuntimeError("rule runner exited early: " + err.decode()[-2000:])
            try:
                if httpx.get(self.url + "/health", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        self.stop()
        raise RuntimeError("rule runner did not become ready")

    def stop(self) -> Tuple[bytes, bytes]:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.output = self.proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.output = self.proc.communicate()
        elif self.proc is not None:
            self.output = self.proc.communicate()
        self.proc = None
        return self.output

    def __enter__(self) -> "RealRunner":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
        shutil.rmtree(self.root, ignore_errors=True)


def point_backend_at(monkeypatch, runner: RealRunner, **overrides) -> None:
    """The backend's client → this runner over real HTTP; code rules on (plus any runtime overrides)."""
    from src.core import runtime_settings
    from src.core.config import settings
    from src.services import code_rule_runner

    monkeypatch.setattr(code_rule_runner, "_transport_factory", None)
    monkeypatch.setattr(code_rule_runner, "RETRY_SLEEP_S", 0.0)
    monkeypatch.setattr(settings, "code_rule_runner_url", runner.url)
    runtime_settings.apply({"code_rules_enabled": True, **overrides})
