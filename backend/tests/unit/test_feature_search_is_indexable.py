"""Feature search must stay indexable — no cast-and-scan over activations.

THE DEFECT THIS PINS
--------------------
`search_features` timed out on any SELECTIVE search term. Root cause: the
search filter ILIKE'd `CAST(feature_activations.tokens AS TEXT)`, a cast
expression over 15,685,642 rows. A cast cannot use an index, so the planner
had only `Parallel Seq Scan` (cost ~4,009,222).

The failure was inverted in a way that hid it:

    search="a"          instant   (matched 8,161 of 8,164 features)
    search="zzzzqqqq"   TIMEOUT
    name/desc half only 15 ms, and found the 5 features actually wanted

`"a"` matched nearly everything on the INDEXED half, so the `OR`
short-circuited and never evaluated the subquery. The tool was fast exactly
when it was useless and timed out exactly when it was useful — which is why it
survived: nobody's smoke test used a rare word.

It was also undocumented. The tool contract says "search matches
name/description" and never mentioned token contents.

WHAT THIS TEST GUARDS
---------------------
The SQL shape, not the timing. A wall-clock assertion would be flaky on a
shared runner and would not say WHY it regressed — see the project rule about
never asserting a bare timing threshold. Compiling the query and inspecting it
catches the reintroduction directly, on any machine, with no database.
"""

import re

import pytest


def _compiled_search_sql(build) -> str:
    """The SQL a search query compiles to, as text."""
    from sqlalchemy.dialects import postgresql

    stmt = build()
    return str(stmt.compile(dialect=postgresql.dialect()))


class TestSearchDoesNotScanActivations:
    def _sql_for_search(self, term: str) -> str:
        """Build the same filter the service builds, via the service itself."""
        import inspect

        from src.services import feature_service

        source = inspect.getsource(feature_service)
        return source

    def test_no_cast_of_activation_tokens_anywhere(self):
        """The exact expression that made the query unindexable.

        Checked as SOURCE rather than a live query because the defect appeared
        in FOUR places — the list and count queries of both the training and
        extraction paths — and a test that exercised one would have left three.
        """
        source = self._sql_for_search("x")
        offenders = [
            line.strip()
            for line in source.splitlines()
            if "cast(" in line.lower()
            and "featureactivation" in line.lower().replace(" ", "")
        ]
        assert not offenders, (
            "feature search casts an activation column to text and ILIKEs it. "
            "That cannot use an index: it is a sequential scan over ~15.7M "
            f"rows. Offending line(s): {offenders}"
        )

    def test_search_does_not_reference_the_activations_table_at_all(self):
        """Broader than the cast: ANY join from search into feature_activations
        reintroduces the scan, cast or not."""
        import inspect

        from src.services import feature_service

        source = inspect.getsource(feature_service)
        # Isolate each `if search_params.search:` block and check what it touches.
        blocks = re.findall(
            r"if search_params\.search:(.*?)(?=\n        (?:if|#|query|count_query|return)\s)",
            source,
            re.S,
        )
        assert blocks, "could not locate any search block — has the shape changed?"
        for block in blocks:
            assert "FeatureActivation" not in block, (
                "a search filter references FeatureActivation. Searching token "
                "CONTENT needs a real index (pg_trgm GIN on a materialised "
                "column, or tsvector) and an explicit opt-in parameter — not a "
                "subquery attached to every search.\n\n" + block.strip()[:400]
            )

    def test_search_still_covers_name_and_description(self):
        """Specificity: the fix must not have removed the search entirely.

        Without this, deleting the whole filter would pass the tests above —
        the classic over-correction where making a query fast makes it useless.
        """
        import inspect

        from src.services import feature_service

        source = inspect.getsource(feature_service)
        blocks = re.findall(
            r"if search_params\.search:(.*?)(?=\n        (?:if|#|query|count_query|return)\s)",
            source,
            re.S,
        )
        assert blocks, "no search blocks found"
        for block in blocks:
            assert "Feature.name.ilike" in block, (
                "a search block no longer filters on name:\n" + block.strip()[:300]
            )
            assert "Feature.description.ilike" in block, (
                "a search block no longer filters on description:\n"
                + block.strip()[:300]
            )

    def test_all_four_search_paths_are_covered(self):
        """The defect was copy-pasted across four sites: list and count, for
        both the training and extraction paths. Fixing one would have left the
        tool broken on the others, and the count query alone would still hang
        the request."""
        import inspect

        from src.services import feature_service

        source = inspect.getsource(feature_service)
        count = source.count("if search_params.search:")
        assert count >= 4, (
            f"expected at least 4 search filter sites, found {count} — either "
            "a path was removed or the shape changed and this guard is "
            "checking less than it thinks"
        )
