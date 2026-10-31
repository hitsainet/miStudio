"""Fixtures for miStudio's runner-host tests (Feature 040 S4; FTDD Appendix C.9; the `rule-runner-tests` CI job).

The host under test is `backend/src/rule_runner/`, imported AS THE IMAGE INSTALLS IT: a top-level package
`mistudio_rule_runner`, without the rest of `src` (whose `__init__` imports SQLAlchemy). A session-scoped temporary
directory holds a symlink `mistudio_rule_runner -> backend/src/rule_runner` and goes first on `sys.path`; the
symlink resolves, so the package still reads the vendored contract at `backend/src/schemas/vendored/`.

The guest bundle comes from ``RUNNER_BUNDLE_DIR`` (``python -m src.rule_runner.fetch --out DIR``, from the
backend tree). When it is absent or its snapshot does not match the vendored contract's digest — or ``wasmtime``
is not installed — the tests that need it SKIP LOUDLY, and FAIL instead under ``MISTUDIO_REQUIRE_RUNNER=1`` (set
by the CI job). The real runner is started as a SUBPROCESS whose environment holds only ``PATH``, ``PYTHONPATH``
and ``RUNNER_*`` — no database URL, no secret — so a runner that silently needed one would fail to answer.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

RUNNER = Path(__file__).resolve().parents[1]
REPO = RUNNER.parent
BACKEND = REPO / "backend"
PACKAGE_SRC = BACKEND / "src" / "rule_runner"
PACKAGED_CONTRACT = BACKEND / "src" / "schemas" / "vendored" / "midataworks-runner-v1.json"
REQUIRE = os.environ.get("MISTUDIO_REQUIRE_RUNNER") == "1"

#: The import root: `<tmp>/mistudio_rule_runner -> backend/src/rule_runner` (created at import of this conftest so
#: the test modules' top-level `from mistudio_rule_runner …` resolve).
IMPORT_ROOT = Path(tempfile.mkdtemp(prefix="mrr-import-"))
(IMPORT_ROOT / "mistudio_rule_runner").symlink_to(PACKAGE_SRC, target_is_directory=True)
sys.path.insert(0, str(IMPORT_ROOT))


def _skip(message: str) -> None:
    if REQUIRE:
        pytest.fail("RUNNER: " + message + " (MISTUDIO_REQUIRE_RUNNER=1)")
    pytest.skip("RUNNER: " + message + " — the runner tests did NOT run")


def _bundle_dir() -> Path:
    env = os.environ.get("RUNNER_BUNDLE_DIR")
    if not env:
        _skip("RUNNER_BUNDLE_DIR is not set; fetch it with `python -m src.rule_runner.fetch --out DIR`")
    return Path(env)


def _check_bundle(bundle_dir: Path) -> Path:
    snapshot = bundle_dir / "snapshot.wasm"
    if not snapshot.is_file():
        _skip(f"no snapshot.wasm in {bundle_dir}")
    contract = json.loads(PACKAGED_CONTRACT.read_text(encoding="utf-8"))
    want = contract["guest"]["bundle"]["snapshot"]["sha256"]
    got = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if got != want:
        _skip(f"snapshot.wasm digest {got} does not match the contract's {want}")
    return snapshot


@pytest.fixture(scope="session")
def bundle_dir() -> Path:
    directory = _bundle_dir()
    _check_bundle(directory)
    return directory


@pytest.fixture(scope="session")
def contract() -> dict[str, Any]:
    return json.loads(PACKAGED_CONTRACT.read_text(encoding="utf-8"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class RunnerProcess:
    def __init__(self, base_url: str, env: dict[str, str], proc: subprocess.Popen) -> None:
        self.base_url = base_url
        self.env = env
        self.proc = proc

    def health(self) -> dict[str, Any]:
        import httpx

        return httpx.get(self.base_url + "/health", timeout=10).json()

    def run(self, body: dict[str, Any]) -> Any:
        import httpx

        return httpx.post(self.base_url + "/v1/run", json=body, timeout=60)

    def inspect(self, body: dict[str, Any]) -> Any:
        import httpx

        return httpx.post(self.base_url + "/v1/inspect", json=body, timeout=60)


def runner_env(bundle_dir: Path, port: int, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Deliberately minimal: PATH, PYTHONPATH and RUNNER_* only. No DATABASE_URL, REDIS_URL or any secret."""
    paths = [str(IMPORT_ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.pathsep.join(paths),
        "RUNNER_BUNDLE_DIR": str(bundle_dir),
        "RUNNER_CONTRACT_PATH": str(PACKAGED_CONTRACT),
        "RUNNER_HOST": "127.0.0.1",
        "RUNNER_PORT": str(port),
        **(extra or {}),
    }


def start_runner(bundle_dir: Path, extra_env: dict[str, str] | None = None,
                 cwd: Path | None = None) -> RunnerProcess:
    import httpx

    port = _free_port()
    env = runner_env(bundle_dir, port, extra_env)
    proc = subprocess.Popen([sys.executable, "-m", "mistudio_rule_runner.server"], env=env,
                            cwd=str(cwd or IMPORT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            _out, err = proc.communicate()
            raise RuntimeError(f"runner exited early: {err.decode()[-2000:]}")
        try:
            if httpx.get(base_url + "/health", timeout=2).status_code in (200, 503):
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError("runner did not become ready")
    return RunnerProcess(base_url, env, proc)


def stop_runner(runner: RunnerProcess) -> tuple[bytes, bytes]:
    runner.proc.terminate()
    try:
        return runner.proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        runner.proc.kill()
        return runner.proc.communicate()


@pytest.fixture(scope="session")
def runner_process(bundle_dir: Path) -> Iterator[RunnerProcess]:
    runner = start_runner(bundle_dir)
    try:
        yield runner
    finally:
        stop_runner(runner)
