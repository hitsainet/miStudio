"""`docs/schemas/decision-precedence-cases.json` — the serving bar's precedence, pinned by data (040 S2; 7.4).

miStudio OWNS the file (it pins the meaning of `probe-definition-v1.json`'s `decision` block); miLLM 039
vendors it and runs it against `probe_runtime`. Three independent readings must agree with every case:

1. `deployed_decision` — miStudio's production function (threshold AND rule);
2. a standard-library checker written from the file's own `rule` text, importing nothing from `src`
   (a consumer reproduces every case from the file alone);
3. miLLM's REAL arming and serving path (the definition parsers `probe_arming` arms a probe with, then
   `ProbeRequestContext._verdict_for`), run in miLLM's own venv as a subprocess when the sibling is checked out — nothing of miLLM is imported here (ADR-001). Skips loudly
   without the sibling; fails under `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1`.

And, once miLLM has vendored it, its copy must be byte-identical.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from src.services.probe_monitor_metrics import DECISION_RULES, deployed_decision
from tests.support.sibling_repos import millm_repo

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "docs" / "schemas" / "decision-precedence-cases.json"
DRIVER = Path(__file__).resolve().parents[1] / "support" / "millm_decision_driver.py"
FLAG = "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"


def load(path: Path = CASES) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_file_is_well_formed():
    doc = load()
    assert doc["format"] == "mistudio.decision-precedence-cases/1"
    names = [c["name"] for c in doc["cases"]]
    assert len(names) == len(set(names)) and len(names) >= 20
    assert {c["expect"]["rule"] for c in doc["cases"]} == set(DECISION_RULES)
    assert all(isinstance(line, str) for line in doc["rule"])


@pytest.mark.parametrize("case", load()["cases"], ids=lambda c: c["name"])
def test_deployed_decision_reproduces_each_case(case):
    probe = case["probe"]
    got = deployed_decision(probe["threshold"], case["window_entry"], probe["length_bands"], probe["scope"],
                            case["window"], case["n_scored"])
    assert (got.threshold, got.rule) == (case["expect"]["threshold"], case["expect"]["rule"])


# ── a consumer's reading: the standard library and the file's own rule text ─────────────────────


def stdlib_decision(case: dict):
    """Written from `doc["rule"]`, importing nothing from the app."""
    probe, entry, window, n = case["probe"], case["window_entry"], case["window"], case["n_scored"]

    def usable(bands):
        """The file's whole-or-nothing table rule."""
        if not bands:
            return []
        for band in bands:
            lo, hi, t = band.get("min_tokens"), band.get("max_tokens"), band.get("threshold")
            if not isinstance(t, (int, float)) or isinstance(t, bool):
                return []
            if not isinstance(lo, int) or isinstance(lo, bool) or lo < 0:
                return []
            if hi is not None and (not isinstance(hi, int) or isinstance(hi, bool)):
                return []
        return bands if bands[-1]["max_tokens"] is None else []

    def match(bands):
        for band in usable(bands):
            lo, hi = band["min_tokens"] or 0, band["max_tokens"]
            if lo <= n and (hi is None or n <= hi):
                return band
        return None

    own = entry.get("threshold") if entry else None
    bar = own if own is not None else probe["threshold"]
    base = "window_bar" if own is not None else ("global" if bar is not None else "none")
    if own is not None and usable(entry.get("length_bands")):
        band = match(entry["length_bands"])
        if band is None:
            return bar, base
        return band["threshold"], "window_band"
    own_scope = window is None or window == probe["scope"]
    if own_scope and (own is None or probe["scope"] == "all") and usable(probe["length_bands"]):
        band = match(probe["length_bands"])
        if band is not None:
            return band["threshold"], "probe_band"
    return bar, base


def test_a_stdlib_checker_reproduces_every_case_from_the_file_alone():
    for case in load()["cases"]:
        assert stdlib_decision(case) == (case["expect"]["threshold"], case["expect"]["rule"]), case["name"]


def test_the_checkers_see_a_wrong_expectation(tmp_path):
    """Negative control kept in the suite: one changed expectation is caught by both readings."""
    doc = load()
    doc["cases"][0]["expect"]["rule"] = "global"
    case = doc["cases"][0]
    assert stdlib_decision(case) != (case["expect"]["threshold"], case["expect"]["rule"])
    probe = case["probe"]
    got = deployed_decision(probe["threshold"], case["window_entry"], probe["length_bands"], probe["scope"],
                            case["window"], case["n_scored"])
    assert got.rule != case["expect"]["rule"]


# ── miLLM's serving path, in miLLM's venv ───────────────────────────────────────────────────────


def _millm_or_skip() -> Path:
    root = millm_repo()
    python = root / "venv" / "bin" / "python"
    if not (root / "millm" / "services" / "probe_runtime.py").is_file() or not python.is_file():
        message = (f"miLLM (with its venv) is not at {root}; set MILLM_REPO. Skipping is expected on the "
                   f"public mirror; set {FLAG}=1 to make this a failure.")
        if os.environ.get(FLAG) == "1":
            pytest.fail(message)
        pytest.skip(message)
    return root


def run_in_millm(case_file: Path) -> dict:
    root = _millm_or_skip()
    # `-c` with the driver's source and cwd = the miLLM checkout: `millm` imports from there, and no
    # directory of miStudio's lands on miLLM's `sys.path`.
    proc = subprocess.run([str(root / "venv" / "bin" / "python"), "-c", DRIVER.read_text(), str(case_file)], cwd=root,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("DECISIONS "))
    return json.loads(line[len("DECISIONS "):])


def test_the_cases_cover_tables_a_server_will_not_arm():
    """Review R1 (040 S1-S2): the file once pinned bars miLLM never serves, because its tables were
    applied as written rather than as armed. Each unusable shape stays a case, so the rule stays pinned."""
    names = {c["name"] for c in load()["cases"]}
    assert {"unusable-table.null-band-threshold", "unusable-table.closed-last-band",
            "window.own-bar.unusable-own-bands"} <= names


def test_millm_serves_every_case_at_the_expected_bar():
    served = run_in_millm(CASES)
    for case in load()["cases"]:
        assert served[case["name"]] == case["expect"]["threshold"], case["name"]


def test_millms_vendored_copy_is_byte_identical_once_vendored():
    root = _millm_or_skip()
    theirs = root / "docs" / "schemas" / CASES.name
    if not theirs.is_file():
        pytest.skip("miLLM has not vendored decision-precedence-cases.json yet (miLLM 039 S1 does)")
    assert theirs.read_bytes() == CASES.read_bytes()
