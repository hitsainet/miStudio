"""No NameError should be discoverable by a linter and still be in production.

WHY THIS EXISTS. `neuronpedia_export_service.execute_export` used `utc_now()` at
its FIRST statement while the only `from ..core.clock import utc_now` in the file
sat inside a README template's f-string — a python code block in text the export
writes for its users. So the name existed nowhere in the module namespace, and
every export died with `NameError` before reaching its own `try:`, which meant
even the FAILED write never ran and the row stranded in COMPUTING.

Nothing caught it: no test exercised `execute_export`, and reading the file shows
a line that says `from ..core.clock import utc_now` in the right shape at the
wrong depth. A linter finds it in under a second.

Two more real ones surfaced the same way and are fixed:
  * `api/v1/endpoints/steering.py` called `signal.SIGKILL` in the orphan-worker
    reaper without importing `signal`. That reaper is the SANCTIONED
    replacement for `pkill -f steering@`, which is forbidden here — so the
    supported path raised NameError and the forbidden one was the only one that
    worked.
  * `workers/jlens_fit_tasks.py` had `except ArtifactQualityRegression` with the
    name unimported in that function. Evaluating an except clause resolves the
    name, so ANY exception from the publish call became a NameError that
    masked it.
"""

import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]

#: Verified by reading the code, not assumed. Each of these is a real binding
#: that ruff's scope analysis does not follow. Shrink this list; never grow it
#: without recording WHY the finding is not a defect.
KNOWN_FALSE_POSITIVES = {
    # `if 'base_model' in locals(): del base_model` — ruff cannot see the
    # locals() guard, and the del is correct.
    "src/workers/training_tasks.py:base_model",
    "src/workers/training_tasks.py:tokenizer",
}


def _f821_findings():
    """Run ruff, and REFUSE TO PASS IF IT DID NOT RUN.

    The first version of this passed `--output-format concise`, which ruff
    0.1.15 rejected (ruff is 0.17.1 since 2026-10-22, and reads `json`). The command exited non-zero with an empty stdout, the parse
    found nothing, and the gate reported a clean tree — a lint guard that fails
    OPEN, which is the exact failure mode this repo has recorded twice. Ruff
    exits 0 for "no findings" and 1 for "findings"; anything else means the
    check did not happen and must be an error, not a pass.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821",
         "--no-cache", "--output-format", "json", "src/"],
        cwd=BACKEND, capture_output=True, text=True,
    )
    # R1-11: A MISSING RUFF ALSO EXITS 1, with empty stdout — so the returncode
    # gate alone still reported a clean tree it never inspected, which is the
    # exact fail-open this function's docstring claims to have closed. Prove the
    # tool is there before trusting its silence.
    probe = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        cwd=BACKEND, capture_output=True, text=True,
    )
    if probe.returncode != 0 or "ruff" not in probe.stdout.lower():
        raise AssertionError(
            "ruff is not importable in this interpreter, so this gate cannot "
            f"report anything about the tree.\nstdout: {probe.stdout!r}\n"
            f"stderr: {probe.stderr[-500:]!r}"
        )
    if proc.returncode not in (0, 1):
        raise AssertionError(
            f"ruff did not run (exit {proc.returncode}); this gate cannot "
            f"report a clean tree it never inspected.\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    # MACHINE-READABLE OUTPUT (2026-10-22, ruff 0.1.15 -> 0.17.1). The old parse split
    # `path:line:col: F821 Undefined name `x`` lines; ruff 0.17's default output puts the code
    # and the path on DIFFERENT lines, so that parse matched nothing and this gate would have
    # reported a clean tree it never read. JSON is ruff's stable interface; output that does
    # not parse as a list of findings refuses, it does not pass.
    try:
        records = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"ruff output is not JSON ({exc}); this gate cannot read it.\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        ) from exc
    if not isinstance(records, list):
        raise AssertionError(f"ruff JSON is not a list of findings: {type(records).__name__}")
    if proc.returncode == 1 and not records:
        raise AssertionError("ruff exited 1 (findings) but reported none — refusing to pass")
    findings = set()
    for record in records:
        if record.get("code") != "F821":
            continue
        path = Path(record["filename"]).resolve().relative_to(BACKEND).as_posix()
        message = record["message"]
        name = message.split("`")[1] if "`" in message else message.strip()
        findings.add(f"{path}:{name}")
    return findings


def test_the_gate_fails_loudly_when_ruff_cannot_run():
    """NEGATIVE CONTROL for the fail-open bug above.

    Without the returncode check, a broken invocation is indistinguishable from
    a clean tree.
    """
    import unittest.mock as mock

    # Two ways the tool can be unusable, and NEITHER may read as a clean tree:
    #   returncode 2 — a bad invocation (the original bug: a rejected flag)
    #   returncode 1 with empty stdout — ruff is not installed at all, which
    #                 is indistinguishable from "no findings" by exit code
    for returncode, stderr in ((2, "error: invalid value"),
                               (1, "No module named ruff")):
        broken = subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout="", stderr=stderr
        )
        with mock.patch.object(subprocess, "run", return_value=broken):
            try:
                _f821_findings()
            except AssertionError:
                pass  # refused, which is the whole point
            else:
                raise AssertionError(
                    f"a ruff invocation exiting {returncode} with no output was "
                    f"reported as a clean tree"
                )


def test_the_gate_refuses_ruffs_human_output():
    """NEGATIVE CONTROL for the parse: ruff 0.17's default (human) output, from a ruff that ran
    and found something, must not read as a clean tree — the 0.1.x line parse matched nothing
    in it. With a real finding in the text, the gate must refuse rather than return no names."""
    import unittest.mock as mock

    human = (
        "F821 Undefined name `utc_now`\n"
        "  --> src/services/neuronpedia_export_service.py:10:5\n"
        "Found 1 error.\n"
    )
    check = subprocess.CompletedProcess(args=[], returncode=1, stdout=human, stderr="")
    probe = subprocess.CompletedProcess(args=[], returncode=0, stdout="ruff 0.17.1\n", stderr="")
    with mock.patch.object(subprocess, "run", side_effect=[check, probe]):
        try:
            found = _f821_findings()
        except AssertionError:
            return  # refused, which is the point
    raise AssertionError(f"human-format ruff output was parsed as findings={found!r}")


def test_the_gate_reads_a_json_finding():
    """The positive half: one F821 in ruff's JSON is reported as `path:name`."""
    import unittest.mock as mock

    record = [{"code": "F821", "filename": str(BACKEND / "src/x.py"),
               "message": "Undefined name `utc_now`"}]
    check = subprocess.CompletedProcess(args=[], returncode=1, stdout=json.dumps(record), stderr="")
    probe = subprocess.CompletedProcess(args=[], returncode=0, stdout="ruff 0.17.1\n", stderr="")
    with mock.patch.object(subprocess, "run", side_effect=[check, probe]):
        assert _f821_findings() == {"src/x.py:utc_now"}


def test_no_new_undefined_names():
    """A name a linter cannot resolve is a NameError waiting for a code path."""
    findings = _f821_findings()
    unexpected = findings - KNOWN_FALSE_POSITIVES
    assert not unexpected, (
        "undefined names that are not on the verified-false-positive list:\n  "
        + "\n  ".join(sorted(unexpected))
        + "\n\nEach is a NameError that fires the first time its line is "
          "reached. Fix it, or add it above WITH the reason it is not a defect."
    )


def test_the_false_positive_list_does_not_rot():
    """A ratchet that never tightens is a list of excuses.

    If a listed finding disappears — because the code was fixed or removed —
    the entry must go too, or the next real defect in that file hides behind it.
    """
    findings = _f821_findings()
    stale = KNOWN_FALSE_POSITIVES - findings
    assert not stale, (
        "these are no longer reported and must be removed from "
        f"KNOWN_FALSE_POSITIVES: {sorted(stale)}"
    )


def test_the_export_services_clock_import_is_real():
    """The specific bug: the import must be in the module, not in a string.

    Kept as its own case because the general gate above would also pass if
    someone deleted the ruff dependency.
    """
    import src.services.neuronpedia_export_service as module

    assert hasattr(module, "utc_now"), (
        "utc_now is not in the module namespace — the only import is inside "
        "the README template again"
    )
    assert hasattr(module, "utc_now_iso")


def test_the_readme_template_does_not_tell_users_to_import_our_clock():
    """Where the bogus import came from: it was pasted into user-facing text."""
    source = (BACKEND / "src/services/neuronpedia_export_service.py").read_text()
    # R1-12: `readme_block[:800]` was an arbitrary window — the bogus import
    # re-added 900 characters in survived. Bound the block by its real end.
    start = source.index("## Usage with SAELens")
    end = source.index('"""', start)
    readme_block = source[start:end]
    assert "core.clock" not in readme_block, (
        "the exported README tells the reader to import miStudio internals"
    )
