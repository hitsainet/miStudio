"""MIS-E2E-153 / -154 / -012 — the doc→code join the framework requires.

`## Relevant Files` is how a reviewer gets from a task list to the code. Three
findings, one mechanism:

  * **153** The five FTASKS with the MOST implementation (024–028) had no such
    section at all — and PPRD marked their rows "Planned" while their own boxes
    ran 68–100% checked. A shipped feature with no join is one nobody can review.
  * **154** Of 273 paths across the files that DID have the section, 22 were
    genuinely dead — and 15 of 15 checked had **zero add-commits in the entire
    repository history**. Not renames, not deletions: never written. Four `[x]`
    boxes pointed at a `TrainingForm.tsx` that has never existed.
  * **012** 193 of the repo's 348 unchecked boxes sat in six ad-hoc files, none
    with the section. `IMPL_Celery_Steering_Migration.md` alone held 61 open and
    0 done for a migration PADR IDL-13 records as shipped.

Unchecked boxes over shipped work are not neutral — they hide the ones that are
genuinely open.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "0xcc" / "tasks"

#: How a row records a path that was never written. Annotating rather than
#: deleting keeps the record of what was claimed.
NEVER_WRITTEN = "never written"

#: How a row records a path that DID exist and was deliberately removed. This
#: is a different claim from "never written" and must stay distinguishable: one
#: says the plan was never carried out, the other says the code shipped and was
#: later deleted on purpose. Collapsing them would rewrite history — the first
#: version of this guard had only NEVER_WRITTEN, and Wave 7's deletion of
#: `frontend/src/api/websocket.ts` would have been mislabelled as never built.
DELETED = "**Deleted**"

_SOURCE_EXT = (".py", ".tsx", ".ts", ".md", ".yaml", ".yml", ".json", ".sh", ".sql", ".css")


def _unannotated_dead_paths(text: str, exists=None) -> list[str]:
    """Paths a `## Relevant Files` section names that neither resolve nor say so.

    Extracted so the detection itself can be tested. Control C163 made the
    annotation match every table row, which silently accepted every dead path,
    and every existing test stayed green — a guard whose acceptance rule can be
    widened to "anything" without a failure is not a guard.
    """
    if "## Relevant Files" not in text:
        return []
    if exists is None:
        def exists(candidate):
            return (REPO / candidate).exists()

    section = text.split("## Relevant Files", 1)[1]
    offenders = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        match = re.search(r"`([A-Za-z0-9_./\-]+)`", line)
        if not match:
            continue
        candidate = match.group(1)
        if not candidate.endswith(_SOURCE_EXT):
            continue                      # an API route or a directory, not a file
        if exists(candidate):
            continue
        if NEVER_WRITTEN in line or DELETED in line:
            continue                      # recorded honestly, either way
        offenders.append(candidate)
    return offenders


def _task_files():
    files = sorted(TASKS.glob("*.md"))
    assert len(files) > 20, f"only {len(files)} task files found — the scan broke"
    return files


@pytest.mark.parametrize("path", _task_files(), ids=lambda p: p.name)
def test_every_task_file_has_a_relevant_files_section(path):
    """The framework requires it, and eleven files did not have it."""
    assert "## Relevant Files" in path.read_text(), (
        f"{path.name} has no `## Relevant Files` section — there is no way to "
        f"get from its tasks to the code they describe"
    )


@pytest.mark.parametrize("path", _task_files(), ids=lambda p: p.name)
def test_every_listed_path_resolves_or_is_marked_as_never_written(path):
    """A row must either point at a real file or say plainly that it does not.

    Silent dead entries are the defect: a reader following one finds nothing and
    cannot tell whether the file moved, was deleted, or never existed.
    """
    text = path.read_text()
    if "## Relevant Files" not in text:
        pytest.skip("covered by the section test above")

    offenders = _unannotated_dead_paths(text)

    assert not offenders, (
        f"{path.name} lists paths that do not resolve and are not marked: "
        f"{offenders}. Either fix the path, or annotate it: "
        f"'⚠️ **never written**' if it was planned and never built (say what "
        f"ships the capability instead), or '**Deleted**' if it shipped and was "
        f"removed on purpose (say why). Those are different claims."
    )


def test_the_scan_can_see_an_unresolved_path():
    """Negative control. A path check that resolves everything by accident —
    a broken regex, a wrong root — would pass this file forever."""
    assert not (REPO / "backend/src/services/definitely_not_a_real_module.py").exists()
    assert (REPO / "backend/src/services/steering_service.py").exists(), (
        "the repo root is wrong, so every path would 'not resolve' and the "
        "parametrized test above would be failing for the wrong reason"
    )


def test_the_five_untraceable_ftasks_now_have_real_paths():
    """MIS-E2E-153 specifically: 024–028 had the most code and the least join."""
    for stem in ("024_", "025_", "026_", "027_", "028_"):
        matches = list(TASKS.glob(f"{stem}*.md"))
        assert matches, f"no FTASKS file starting {stem}"
        section = matches[0].read_text().split("## Relevant Files", 1)
        assert len(section) == 2, f"{matches[0].name} still has no section"
        rows = [l for l in section[1].splitlines() if l.startswith("| `")]
        assert rows, f"{matches[0].name}'s section lists no files"


def test_no_task_file_claims_a_capability_with_no_implementation():
    """MIS-E2E-154's one substantive item, as opposed to its documentation half.

    `- [x] Zoom and pan` was checked in 003_FTASKS while nothing implements zoom
    or pan anywhere. The other 21 dead entries name the wrong FILE for a
    capability that does ship; this one claimed the capability.
    """
    training = TASKS / "003_FTASKS|SAE_Training.md"
    assert training.exists()
    assert "- [x] Zoom and pan" not in training.read_text(), (
        "the zoom-and-pan box is checked again; nothing implements it"
    )


class TestTheGuardItselfStillRejects:
    """C163: widening the annotation to match every row accepted every dead
    path, and the whole suite stayed green. These exercise the detector
    directly, against text rather than the repository."""

    SECTION = """## Relevant Files

| File | Purpose |
|---|---|
| `backend/src/real_module.py` | exists |
| `backend/src/gone.py` | no annotation at all |
| `backend/src/planned.py` | ⚠️ **never written** — no add-commit anywhere |
| `backend/src/removed.py` | **Deleted** — superseded by the middleware |
"""

    def _exists(self, candidate):
        return candidate == "backend/src/real_module.py"

    def test_an_unannotated_dead_path_is_caught(self):
        found = _unannotated_dead_paths(self.SECTION, exists=self._exists)
        assert found == ["backend/src/gone.py"], found

    def test_never_written_is_accepted(self):
        assert "backend/src/planned.py" not in _unannotated_dead_paths(
            self.SECTION, exists=self._exists)

    def test_deleted_is_accepted(self):
        assert "backend/src/removed.py" not in _unannotated_dead_paths(
            self.SECTION, exists=self._exists)

    def test_the_annotations_are_not_interchangeable_with_any_text(self):
        """The two markers must be specific strings, not something every row has."""
        for marker in (NEVER_WRITTEN, DELETED):
            assert marker.strip() not in ("", "|", "-"), (
                f"{marker!r} would match every table row, accepting every dead path"
            )
            assert len(marker) >= 8, (
                f"{marker!r} is short enough to appear incidentally"
            )
