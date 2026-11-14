"""The seeded internal labeling template carries the product's new name, on every path (ENH-009 R25).

Revision ``c25e7a41d9b3`` renames ONE row, by id. Two things can go wrong silently and each has a test:

* **The fresh-database trap.** ``n1o2p3q4r5s6`` rewrites this row's prompts ``WHERE name = '<seeded
  name>'``. If the seed file's name is ever changed instead of (or as well as) adding a revision, that
  UPDATE matches nothing on a fresh database and the row keeps the first prompts, which are inside the
  labeling fingerprint. ``test_a_database_at_head_...`` reads a database migrated to head (CI migrates
  it from empty) and requires BOTH the new name and the prompts that revision writes.
* **The guard.** Only the system row that still carries the seeded name is renamed.
  ``test_only_the_seeded_system_row_is_renamed`` runs the revision's own ``upgrade`` and ``downgrade``
  against rows that must and must not move.

CONTROLS (2026-11-11; each alone, restored from saved bytes) - all red:
  C1 the head database left at the previous head ``0b576a6b9bda``      -> test_a_database_at_head (name)
  C2 the seed JSON's name changed to the new name, fresh database      -> test_a_database_at_head (prompts)
  C3 ``AND name = :from_name`` removed from the UPDATE                  -> test_only_the_seeded_system_row
  C4 ``AND is_system = true`` removed                                   -> test_only_the_seeded_system_row
  C5 ``upgrade`` made a no-op                                           -> test_only_the_seeded_system_row
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import create_engine, text

from tests.support.test_databases import migrated_url, sync_test_url

BACKEND = Path(__file__).resolve().parents[2]
VERSIONS = BACKEND / "alembic" / "versions"
SEED_FILE = BACKEND / "src" / "data" / "templates" / "mistudio_internal.json"


def _revision(filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), VERSIONS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RENAME = _revision("c25e7a41d9b3_rename_internal_labeling_template.py")
PROMPTS = _revision("n1o2p3q4r5s6_improve_default_labeling_template.py")


def test_the_new_name_is_the_products_and_the_seeded_one_is_the_seed_files():
    assert RENAME.NAME == "miWorkbench Internal - Full Context"
    assert RENAME.TEMPLATE_ID == "lpt_mistudio_internal"  # frozen (DR-16)
    # The seed file is NOT renamed: the revision's `from` name is the name the seed still writes.
    assert json.loads(SEED_FILE.read_text(encoding="utf-8"))["name"] == RENAME.SEEDED_NAME


def test_a_database_at_head_has_the_new_name_and_the_prompts_the_name_matched_revision_wrote():
    engine = create_engine(migrated_url())
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT name, is_system, system_message, user_prompt_template "
                     "FROM labeling_prompt_templates WHERE id = :i"), {"i": RENAME.TEMPLATE_ID}).one()
    finally:
        engine.dispose()
    assert row.is_system is True
    assert row.name == RENAME.NAME, "the database is not at head, or the rename matched nothing"
    # The prompts `n1o2p3q4r5s6` writes by NAME: absent when the seed's name was changed under it.
    assert row.system_message == PROMPTS.NEW_SYSTEM_MESSAGE
    assert row.user_prompt_template == PROMPTS.NEW_USER_PROMPT


class _Op:
    def __init__(self, conn):
        self.conn = conn
        self.calls = 0

    def execute(self, statement):
        self.calls += 1
        self.conn.execute(statement)


def test_only_the_seeded_system_row_is_renamed(monkeypatch: pytest.MonkeyPatch):
    engine = create_engine(sync_test_url())
    try:
        with engine.connect() as conn:
            # A session-local table of the same name shadows the real one; nothing is committed.
            conn.execute(text("CREATE TEMP TABLE labeling_prompt_templates "
                              "(id varchar PRIMARY KEY, name varchar NOT NULL, is_system boolean NOT NULL) "
                              "ON COMMIT DROP"))
            rows = {
                RENAME.TEMPLATE_ID: (RENAME.SEEDED_NAME, True),
                "lpt_users_copy": (RENAME.SEEDED_NAME, False),
                "lpt_other_system": (RENAME.SEEDED_NAME, True),
                "lpt_anthropic_style": ("Anthropic Style - Evidence-Based Labeling", True),
            }
            for template_id, (name, is_system) in rows.items():
                conn.execute(text("INSERT INTO labeling_prompt_templates VALUES (:i, :n, :s)"),
                             {"i": template_id, "n": name, "s": is_system})

            def names() -> dict[str, str]:
                return dict(conn.execute(text("SELECT id, name FROM labeling_prompt_templates")).all())

            op = _Op(conn)
            monkeypatch.setattr(RENAME, "op", op)
            before = names()
            RENAME.upgrade()
            assert op.calls == 1
            assert names() == {**before, RENAME.TEMPLATE_ID: RENAME.NAME}
            RENAME.downgrade()
            assert names() == before

            # A name the operator typed on the system row is left alone, in both directions.
            conn.execute(text("UPDATE labeling_prompt_templates SET name = 'Ours' WHERE id = :i"),
                         {"i": RENAME.TEMPLATE_ID})
            # ... and a non-system row holding the seeded name under the frozen id is not a seeded row.
            typed = names()
            RENAME.upgrade()
            RENAME.downgrade()
            assert names() == typed
            conn.execute(text("UPDATE labeling_prompt_templates SET name = :n, is_system = false WHERE id = :i"),
                         {"i": RENAME.TEMPLATE_ID, "n": RENAME.SEEDED_NAME})
            unowned = names()
            RENAME.upgrade()
            assert names() == unowned
            conn.rollback()
    finally:
        engine.dispose()
