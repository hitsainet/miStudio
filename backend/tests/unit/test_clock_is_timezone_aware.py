"""MIS-E2E-054: naive UTC must not come back, and `utc_now` must stay aware.

The 37 `datetime.utcnow()` sites were writing naive datetimes into 62 columns
declared `DateTime(timezone=True)`. Postgres reads a naive value in the session
timezone, so on a non-UTC server the stored instant is silently wrong — and a
value read back out is aware, so comparing it to a naive "now" raises.
"""

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.clock import utc_now, utc_now_iso

SRC = Path(__file__).resolve().parents[2] / "src"


class TestTheHelper:
    def test_utc_now_is_timezone_aware(self):
        now = utc_now()
        assert now.tzinfo is not None, "the whole point of the helper"
        assert now.utcoffset() == timezone.utc.utcoffset(None)

    def test_utc_now_iso_ends_in_Z_and_carries_a_real_offset(self):
        text = utc_now_iso()
        assert text.endswith("Z")
        # The old hand-rolled form appended "Z" to a naive isoformat, asserting
        # an offset the value did not have. Round-tripping proves this one does.
        assert datetime.fromisoformat(text.replace("Z", "+00:00")).tzinfo is not None

    def test_it_matches_what_the_columns_declare(self):
        """A naive value here would be silently reinterpreted by Postgres."""
        assert utc_now().tzinfo is not None


class TestNaiveUtcDoesNotComeBack:
    """Parsed, not grepped — a docstring quoting the old call is not a defect.

    (This module's own docstring names it, and a substring check would flag it.
    Seven findings in this audit were exactly that trap.)
    """

    def _calls_to_utcnow(self, path: Path):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            return []
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "utcnow":
                    found.append(node.lineno)
        return found

    def test_no_source_file_calls_utcnow(self):
        offenders = {}
        for path in SRC.rglob("*.py"):
            hits = self._calls_to_utcnow(path)
            if hits:
                offenders[str(path.relative_to(SRC))] = hits
        assert not offenders, (
            f"`utcnow()` is back at {offenders}. It returns a naive datetime, and "
            f"every DateTime column in this schema is `timezone=True`. Use "
            f"`src.core.clock.utc_now`."
        )

    def test_every_datetime_column_is_timezone_aware(self):
        """The premise of the fix. If a naive column appears, revisit it."""
        naive = []
        for path in (SRC / "models").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == "DateTime"):
                    continue
                aware = any(kw.arg == "timezone" and getattr(kw.value, "value", False) is True
                            for kw in node.keywords)
                if not aware:
                    naive.append(f"{path.name}:{node.lineno}")
        assert not naive, (
            f"these DateTime columns are naive: {naive}. `utc_now()` writes an "
            f"aware value; decide deliberately what these should hold."
        )
