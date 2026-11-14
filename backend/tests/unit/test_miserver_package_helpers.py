"""miServer's package is found under either name (042 FR-042.36; FTID §3.12; FTASKS 5.5, control K-5.13).

miServer renames `millm` → `miserver` at ENH-009 P8. The helpers in `tests/support/sibling_repos.py`
resolve whichever the checkout holds (a tree with both resolves to `miserver`, and says so), and the two
drivers that run inside miServer's interpreter take the package as an argument instead of guessing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import sibling_repos as S

SUPPORT = Path(__file__).resolve().parents[1] / "support"
NEW, OLD = S.MISERVER_PACKAGES  # never literals here: test_miserver_package_is_located_by_helper.py
DRIVERS = ("millm_decision_driver.py", "millm_definition_driver.py")


def _tree(tmp_path: Path, *packages: str) -> Path:
    for pkg in packages:
        (tmp_path / pkg / "services").mkdir(parents=True)
        (tmp_path / pkg / "main.py").write_text("")
    return tmp_path


@pytest.mark.parametrize("layout, expected", [
    ((NEW,), NEW),
    ((OLD,), OLD),
    ((OLD, NEW), NEW),
], ids=["miserver-only", "millm-only", "both"])
def test_package_name_path_import(tmp_path, layout, expected):
    repo = _tree(tmp_path, *layout)
    assert S.miserver_package(repo) == expected
    assert S.both_packages(repo) is (len(layout) == 2)
    assert S.miserver_path(repo, "services", "x.py") == repo / expected / "services" / "x.py"
    assert S.miserver_path_or_none(repo, "main.py") == repo / expected / "main.py"
    assert S.miserver_import("core.errors", repo) == f"{expected}.core.errors"
    assert S.miserver_import(f"{OLD}.main", repo) == f"{expected}.main"
    assert S.miserver_import(f"{NEW}.services.x", repo) == f"{expected}.services.x"
    assert S.miserver_import("main", repo) == f"{expected}.main"


def test_neither_package(tmp_path):
    assert S.miserver_package(tmp_path) is None
    assert S.miserver_path_or_none(tmp_path, "main.py") is None
    with pytest.raises(FileNotFoundError):
        S.miserver_path(tmp_path, "main.py")
    with pytest.raises(FileNotFoundError):
        S.miserver_import("main", tmp_path)


def test_a_folder_without_main_py_is_not_the_package(tmp_path):
    (tmp_path / NEW).mkdir()          # e.g. a half-applied rename
    (tmp_path / OLD).mkdir()
    (tmp_path / OLD / "main.py").write_text("")
    assert S.miserver_package(tmp_path) == OLD


def test_the_package_names():
    assert S.MISERVER_PACKAGES == ("miserver", "millm")


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("argv", [[], ["mistudio"], [f"{OLD}.main"]], ids=["none", "other-name", "dotted"])
def test_drivers_refuse_missing_argument(tmp_path, driver, argv):
    """Run as the calling tests run them (`python -c <source> …`), outside any miServer checkout."""
    source = (SUPPORT / driver).read_text(encoding="utf-8")
    args = [*argv, str(tmp_path / "cases.json")] if driver == DRIVERS[0] and argv else argv
    proc = subprocess.run([sys.executable, "-I", "-c", source, *args], cwd=tmp_path,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-500:])
    assert "usage:" in proc.stderr and "millm|miserver" in proc.stderr
