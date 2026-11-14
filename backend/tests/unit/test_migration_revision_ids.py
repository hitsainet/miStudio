"""New migrations must use hex revision ids. The existing 22 stay as they are.

MIS-E2E-022. Twenty-two revisions carry hand-typed ids like `j6k7l8m9n0o1`
instead of the hex ids `alembic revision` generates. The finding proposed
normalising them. That would be worse than the defect: a revision id is the
migration graph's identity AND the value sitting in `alembic_version` in every
deployed database. Renaming one strands every install that has already applied
it — the upgrade path would look for a revision that no longer exists.

So the 22 are grandfathered by explicit list, and the rule is enforced for
everything new. The list is a snapshot: it may only shrink (if a legacy
migration is ever squashed away), never grow.
"""

import re
from pathlib import Path

import pytest

VERSIONS = Path(__file__).resolve().parents[2] / "alembic" / "versions"

#: Revisions that predate the rule. Frozen — adding to this list is the thing
#: the test exists to prevent.
GRANDFATHERED = {
    "a1c2e3f4g5h6",
    "b2c3d4e5f6g7",
    "b3c4d5e6f7g8",
    "e1f2g3h4i5j6",
    "f2g3h4i5j6k7",
    "g3h4i5j6k7l8",
    "h4i5j6k7l8m9",
    "i5j6k7l8m9n0",
    "j6k7l8m9n0o1",
    "k7l8m9n0o1p2",
    "l8m9n0o1p2q3",
    "m9n0o1p2q3r4",
    "n1o2p3q4r5s6",
    "o2p3q4r5s6t7",
    "p3q4r5s6t7u8",
    "q4r5s6t7u8v9",
    "r5s6t7u8v9w0",
    "s6t7u8v9w0x1",
    "t7u8v9w0x1y2",
    "u8v9w0x1y2z3",
    "v9w0x1y2z3a4",
    "w0x1y2z3a4b5",
}

_HEX_ID = re.compile(r"^[0-9a-f]{8,}$")
_REVISION = re.compile(r"^revision(?::\s*str)?\s*=\s*['\"]([^'\"]+)['\"]", re.M)


def _revisions():
    found = {}
    for path in sorted(VERSIONS.glob("*.py")):
        match = _REVISION.search(path.read_text())
        if match:
            found[match.group(1)] = path.name
    assert len(found) > 90, f"only {len(found)} revisions parsed — the scan broke"
    return found


class TestRevisionIds:
    def test_new_migrations_use_generated_hex_ids(self):
        offenders = {
            rev: name for rev, name in _revisions().items()
            if not _HEX_ID.match(rev) and rev not in GRANDFATHERED
        }
        assert not offenders, (
            f"{offenders} use hand-typed revision ids. Run `alembic revision` and "
            f"keep the id it generates — a hand-typed id collides silently and "
            f"sorts unpredictably in the graph."
        )

    def test_the_grandfathered_list_only_shrinks(self):
        present = set(_revisions())
        stale = GRANDFATHERED - present
        assert len(GRANDFATHERED) == 22 - len(stale), (
            "the grandfathered list grew; new migrations must use hex ids"
        )

    def test_every_grandfathered_id_is_really_non_hex(self):
        """If one turns out to be hex, it never needed an exemption."""
        wrong = {r for r in GRANDFATHERED if _HEX_ID.match(r)}
        assert not wrong, f"{wrong} are hex and should not be exempted"

    def test_the_check_can_fail(self):
        """A regex that matches everything would pass this file forever."""
        assert not _HEX_ID.match("j6k7l8m9n0o1")
        assert _HEX_ID.match("b4d19f0c73ae")
