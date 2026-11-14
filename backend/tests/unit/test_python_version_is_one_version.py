"""Every place that states the backend's Python version agrees, and the image is the authority.

2026-10-21 (operator: the suite moves from 3.11 to 3.13, one app at a time). When the move began,
this repository named FOUR Python versions and compared none of them: the image ran **3.12**
(deadsnakes, ``backend/Dockerfile``), both CI ``setup-python`` steps tested on **3.11**, and
``pyproject.toml`` declared ``^3.10`` with Black, Ruff and MyPy all targeting 3.10. The published
image was built on an interpreter no CI job ever ran.

The authority is the Dockerfile's ``ARG PYTHON_VERSION`` — the interpreter that installs the
requirements and serves production. Everything else is read from its own file and must equal it:

* every ``actions/setup-python`` step in every workflow (and a step with no ``python-version``
  is itself a failure, because it silently takes the runner's default);
* every ``python3.N`` the Dockerfile names outside a comment, and any ``FROM python:`` /
  compose ``image: python:`` in a tracked file;
* ``[tool.poetry.dependencies] python`` floor, Black ``target-version``, MyPy ``python_version``;
* Ruff ``target-version``. Until 2026-10-22 this carried ONE self-expiring exception (``py312``
  while ``requirements.txt`` pinned a 0.1.x ruff, which refused ``py313``). Ruff moved to 0.17.1
  (``0xcc/reviews/deps_sync_2026-10-21.md``); the exception expired as designed — this file went
  red on ``('py312', {'py313'})`` — and was then removed, so Ruff is held to the image's version
  like every other tool;
* and the interpreter running this suite is not older than the image's.

References are found by scanning tracked files, so a new workflow or Dockerfile is covered the day
it lands (the same shape as ``test_node_version_agrees.py``).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND = REPO_ROOT / "backend"
DOCKERFILE = BACKEND / "Dockerfile"
PYPROJECT = BACKEND / "pyproject.toml"

ARG_PYTHON = re.compile(r"^\s*ARG\s+PYTHON_VERSION\s*=\s*(\S+)\s*$", re.MULTILINE)
PY_LITERAL = re.compile(r"python(3\.\d+)")
FROM_PYTHON = re.compile(r"^\s*FROM\s+(?:--\S+\s+)*python:(\d+\.\d+)", re.MULTILINE | re.IGNORECASE)
IMAGE_PYTHON = re.compile(r"""^\s*image:\s*['"]?python:(\d+\.\d+)""", re.MULTILINE)
SETUP_PYTHON = re.compile(r"uses:\s*actions/setup-python@")
PYTHON_VERSION_KEY = re.compile(r"""^\s*python-version:\s*(['"]?)([^'"\s#]+)\1""", re.MULTILINE)


def _tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", *patterns],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if line]


def _uncommented(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def image_version(dockerfile_text: str) -> str:
    """The X.Y the image installs: its one uncommented ``ARG PYTHON_VERSION``."""
    found = ARG_PYTHON.findall(_uncommented(dockerfile_text))
    assert len(found) == 1, f"expected exactly one ARG PYTHON_VERSION, found {found}"
    assert re.fullmatch(r"3\.\d+", found[0]), f"unparseable ARG PYTHON_VERSION {found[0]!r}"
    return found[0]


def setup_python_versions(workflow_text: str) -> list[str]:
    """Every setup-python step's version; a step without one fails here rather than passing."""
    versions = [v for _, v in PYTHON_VERSION_KEY.findall(workflow_text)]
    steps = len(SETUP_PYTHON.findall(workflow_text))
    assert len(versions) >= steps, (
        f"{steps} setup-python step(s) but {len(versions)} python-version line(s): a step "
        "without one takes the runner's default Python"
    )
    return versions


EXPECTED = image_version(DOCKERFILE.read_text())


# ── the references are found, or every comparison below is vacuous ──────────────────────────


def test_the_image_installs_and_links_the_declared_version():
    text = _uncommented(DOCKERFILE.read_text())
    assert (
        "python${PYTHON_VERSION} -m pip install -r requirements.txt" in text
    ), "the requirements must be installed by the ARG's interpreter"
    assert "ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3" in text
    stray = sorted(set(PY_LITERAL.findall(text)) - {EXPECTED})
    assert not stray, f"the Dockerfile names python{stray} beside ARG PYTHON_VERSION={EXPECTED}"


def test_every_ci_python_is_the_image_python():
    seen: dict[str, list[str]] = {}
    for wf in _tracked(".github/workflows/*.yml", ".github/workflows/*.yaml"):
        versions = setup_python_versions(wf.read_text())
        if versions:
            seen[str(wf.relative_to(REPO_ROOT))] = versions
    # The two that run Python today; a blind walk that found nothing must not pass.
    assert len(seen.get(".github/workflows/backend-tests.yml", [])) >= 1, seen
    assert len(seen.get(".github/workflows/sync-to-clean.yml", [])) >= 1, seen
    wrong = {w: v for w, vs in seen.items() for v in vs if v != EXPECTED}
    assert not wrong, f"CI tests on a Python the image ({EXPECTED}) does not run: {wrong}"


def test_no_other_image_or_compose_service_names_another_python():
    wrong = []
    for df in _tracked("*Dockerfile*"):
        wrong += [
            (str(df.relative_to(REPO_ROOT)), v)
            for v in FROM_PYTHON.findall(df.read_text())
            if v != EXPECTED
        ]
    for cf in _tracked("*compose*.yml", "*compose*.yaml"):
        wrong += [
            (str(cf.relative_to(REPO_ROOT)), v)
            for v in IMAGE_PYTHON.findall(cf.read_text())
            if v != EXPECTED
        ]
    assert not wrong, wrong


def test_pyproject_floor_and_tool_targets_are_the_image_python():
    cfg = tomllib.loads(PYPROJECT.read_text())["tool"]
    py = "py" + EXPECTED.replace(".", "")
    floor = cfg["poetry"]["dependencies"]["python"]
    assert floor in (
        f"^{EXPECTED}",
        f">={EXPECTED}",
    ), f"poetry python {floor!r} != image {EXPECTED}"
    assert cfg["black"]["target-version"] == [py], cfg["black"]["target-version"]
    assert cfg["mypy"]["python_version"] == EXPECTED, cfg["mypy"]["python_version"]
    assert cfg["ruff"]["target-version"] == py, cfg["ruff"]["target-version"]


def test_the_running_interpreter_is_not_older_than_the_image():
    """A 3.11 dev venv would test syntax and stdlib behaviour production never runs."""
    major, minor = (int(x) for x in EXPECTED.split("."))
    assert sys.version_info[:2] >= (major, minor), (
        f"this suite runs on {sys.version.split()[0]}; the image runs {EXPECTED}. "
        "Recreate the venv: uv venv --python " + EXPECTED + " venv"
    )


# ── the parsers themselves: each of these is a way the scan above could pass blind ──────────


def test_a_commented_arg_is_not_the_version():
    with pytest.raises(AssertionError, match="exactly one"):
        image_version("# ARG PYTHON_VERSION=3.13\nFROM nvidia/cuda:x\n")
    assert image_version("FROM x\n# ARG PYTHON_VERSION=3.11\nARG PYTHON_VERSION=3.13\n") == "3.13"


def test_quoted_and_unquoted_yaml_versions_both_parse():
    text = (
        "      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.13'\n"
        "      - uses: actions/setup-python@v5\n        with:\n          python-version: 3.13\n"
        '      - uses: actions/setup-python@v5\n        with:\n          python-version: "3.12"\n'
    )
    assert setup_python_versions(text) == ["3.13", "3.13", "3.12"]


def test_a_setup_python_step_without_a_version_fails():
    with pytest.raises(AssertionError, match="runner's default"):
        setup_python_versions("      - uses: actions/setup-python@v5\n")
