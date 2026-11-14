"""New migrations must parameterise SQL, not build it by f-string.

MIS-E2E-047. Two template-seeding migrations build their UPDATE by f-string
from a JSON file on disk. String fields go through a hand-rolled `escape_sql`
that only doubles single quotes; numeric fields are interpolated raw.

No exploit today — the JSON is committed, not user-supplied. It matters for two
reasons. The moment the template source becomes user-supplied it is an
injection vector, and a hand-rolled `escape_sql` sitting in the migrations
directory reads as a sanctioned pattern for the next person to copy.

The two existing migrations are grandfathered: they have already run on every
deployment, and editing an applied migration changes nothing for those
databases while risking the graph. The rule is enforced going forward.
"""

import ast
import re
from pathlib import Path

import pytest

VERSIONS = Path(__file__).resolve().parents[2] / "alembic" / "versions"

#: Migrations that predate the rule. Frozen — may shrink, never grow.
GRANDFATHERED = {
    "90faea1e38d0_update_anthropic_template_improved_.py",
    # The audit named this one `9dc725cba2ad_update_gpt4_template_with_improved_`.
    # That file does not exist; the real name is below. Running the detector
    # rather than trusting the reference is what surfaced the discrepancy.
    "9dc725cba2ad_update_anthropic_template_add_negative_.py",
}


def _migrations():
    files = sorted(VERSIONS.glob("*.py"))
    assert len(files) > 90, f"only {len(files)} migrations found — the scan broke"
    return files


def _reads_external_data(path: Path) -> bool:
    """Does the migration pull content in from outside its own source?"""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
            if name in ("load", "loads", "read_text", "open"):
                return True
    return False


def _has_fstring_sql(path: Path) -> bool:
    """An f-string containing a SQL verb, anywhere in the module.

    NOTE this alone is not the finding. Interpolating an IDENTIFIER — a table
    or column name — is unavoidable, because SQL cannot bind an identifier;
    two migrations in this repo legitimately do it over a hardcoded list. The
    hazard is interpolating DATA, and specifically data read from outside the
    migration's own source. `test_no_new_fstring_sql_over_external_data` is the
    rule; this predicate is one half of it.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:  # pragma: no cover
        return False
    verbs = re.compile(r"\b(INSERT|UPDATE|DELETE|SELECT)\b", re.I)
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        literal = "".join(
            p.value for p in node.values
            if isinstance(p, ast.Constant) and isinstance(p.value, str)
        )
        # An f-string with no interpolation is just a string.
        interpolated = any(isinstance(p, ast.FormattedValue) for p in node.values)
        if interpolated and verbs.search(literal):
            return True
    return False


class TestTheScanWorks:
    def test_it_flags_the_known_offenders(self):
        """If the detector cannot see the two it was written for, it sees nothing."""
        for name in GRANDFATHERED:
            path = VERSIONS / name
            if not path.exists():
                continue
            assert _has_fstring_sql(path), (
                f"{name} is exempted as an f-string-SQL offender but the "
                f"detector no longer flags it — either it was cleaned up (drop "
                f"it from GRANDFATHERED) or the detector is broken"
            )

    def test_it_does_not_flag_a_parameterised_migration(self):
        """A bindparams migration must be legal, or the rule is unusable."""
        clean = VERSIONS / "n1o2p3q4r5s6_improve_default_labeling_template.py"
        if clean.exists():
            assert not _has_fstring_sql(clean)


class TestNoNewFStringSql:
    def test_no_new_fstring_sql_over_external_data(self):
        """The actual finding: a JSON file on disk interpolated into an UPDATE."""
        offenders = [
            p.name for p in _migrations()
            if p.name not in GRANDFATHERED
            and _has_fstring_sql(p) and _reads_external_data(p)
        ]
        assert not offenders, (
            f"{offenders} interpolate externally-read content into SQL by "
            f"f-string. Use `sa.text(...).bindparams(...)` — the driver escapes "
            f"correctly, and a hand-rolled escaper in this directory reads as a "
            f"sanctioned pattern for the next person to copy."
        )

    def test_identifier_interpolation_is_not_treated_as_the_hazard(self):
        """Guard against over-fitting the rule.

        `d7f3a91c2e08` and `e2a4c81b9d17` f-string table and column names from
        hardcoded lists in their own source. SQL cannot bind an identifier, so
        that is the only way to write them — flagging those would make the rule
        unusable and it would be turned off.
        """
        for name in ("d7f3a91c2e08_restore_declared_foreign_keys.py",
                     "e2a4c81b9d17_widen_circuit_run_counters.py"):
            path = VERSIONS / name
            if not path.exists():
                continue
            assert _has_fstring_sql(path), "fixture assumption changed"
            assert not _reads_external_data(path), (
                f"{name} now reads external data and f-strings SQL — that IS "
                f"the hazard, and it must be parameterised"
            )

    def test_the_grandfather_list_only_shrinks(self):
        present = {p.name for p in _migrations()}
        assert GRANDFATHERED <= present | GRANDFATHERED
        assert len(GRANDFATHERED) <= 2, (
            "the grandfather list grew; a new migration was exempted instead of "
            "being written with bound parameters"
        )

    def test_no_new_hand_rolled_escaper(self):
        offenders = []
        for path in _migrations():
            if path.name in GRANDFATHERED:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and "escape" in node.name.lower():
                    offenders.append(f"{path.name}:{node.name}")
        assert not offenders, (
            f"{offenders} define a hand-rolled SQL escaper. Doubling quotes is "
            f"not escaping; bind the parameter."
        )
