"""R2-T2: automated migration hygiene guard — the multi-head incident
(cd6c46abac48 merge) must never recur silently, and every new migration must
keep exactly one head."""

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_exactly_one_head():
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert len(heads) == 1, f"Multiple alembic heads: {heads} — merge before shipping"


def test_no_duplicate_revision_ids():
    """R2 A1: two files sharing a revision id still yield ONE head, so
    test_exactly_one_head can't catch it — Alembic silently shadows one and
    drops its migration. Walk the version files and assert every id is unique.
    (The 016 work nearly hit this: d4e5f6a7b8c9 already existed.)"""
    import re
    from pathlib import Path

    versions = Path("alembic/versions")
    seen: dict[str, str] = {}
    pat = re.compile(r"^revision(?::\s*\w+)?\s*=\s*['\"]([^'\"]+)['\"]", re.M)
    for f in versions.glob("*.py"):
        m = pat.search(f.read_text())
        if not m:
            continue
        rev = m.group(1)
        assert rev not in seen, (
            f"Duplicate alembic revision id {rev!r}: {f.name} and "
            f"{seen[rev]} — one silently shadows the other")
        seen[rev] = f.name
