"""The generated MCP contract must match the live registry.

miLLM's hand-maintained contract carried ✅ marks for 16 tools that were never
registered — an entire increment of a document confidently describing a surface
that did not exist. miStudio had no contract at all for its 58 native tools.

The fix is not "write one carefully"; that is what failed. The fix is to DERIVE
it, and have this test regenerate and diff. A stale contract fails the build
instead of quietly lying.
"""

import pytest

from src.mcp_server.contract import collect, contract_path, render


class TestTheContractIsGeneratedAndCurrent:
    def test_the_contract_file_exists(self):
        path = contract_path()
        assert path.exists(), (
            f"{path} is missing. Regenerate it:\n"
            "  python -c 'from src.mcp_server.contract import write_contract; "
            "write_contract()'"
        )

    def test_it_matches_what_the_registry_would_produce_NOW(self):
        """The staleness check. Regenerate; if it differs, the committed file
        describes a surface that has changed underneath it."""
        path = contract_path()
        if not path.exists():
            pytest.skip("covered by test_the_contract_file_exists")

        current = path.read_text()
        expected = render()
        if current == expected:
            return

        # Report WHICH tools moved, not a diff of 300 lines.
        import re

        def names(text):
            return set(re.findall(r"^\| `(\w+)`", text, re.M))

        added = sorted(names(expected) - names(current))
        removed = sorted(names(current) - names(expected))
        detail = ""
        if added:
            detail += f"\n  NEW tools not in the contract: {added}"
        if removed:
            detail += f"\n  tools in the contract that no longer exist: {removed}"
        if not detail:
            detail = "\n  same tools, but their summaries or endpoints changed."

        pytest.fail(
            "docs/mcp-contract.md is STALE." + detail + "\n\nRegenerate:\n"
            "  python -c 'from src.mcp_server.contract import write_contract; "
            "write_contract()'"
        )

    def test_the_collection_is_not_vacuous(self):
        """An empty index would make every assertion above pass."""
        index = collect()
        total = sum(len(v) for v in index.values())
        assert total > 80, f"only collected {total} tools — extraction broken"
        assert len(index) >= 10, f"only {len(index)} categories"

    def test_every_tool_row_carries_its_endpoint(self):
        """A contract row with no endpoint tells a reviewer nothing about what
        the tool reaches. Tools that legitimately call nothing (guidance tools)
        are the only exception."""
        no_endpoint = [
            r["name"]
            for rows in collect().values()
            for r in rows
            if not r["endpoints"]
        ]
        assert set(no_endpoint) <= {"mistudio_howto"}, (
            f"tools with no extractable endpoint: {sorted(no_endpoint)}. Either "
            "they issue no HTTP call, or the AST extractor cannot read their "
            "call shape — the second case means the contract is under-reporting."
        )

    def test_the_destructive_tools_are_flagged(self):
        """An operator scanning this must see the irreversible ones. All five
        are named explicitly: the keyword detector missed millm_delete_circuit
        on its first pass because that docstring says 'permanently' and 'cannot
        be undone' without ever using the word DESTRUCTIVE."""
        flagged = {
            r["name"]
            for rows in collect().values()
            for r in rows
            if r["destructive"]
        }
        for tool in (
            "delete_circuit",
            "delete_experiment",
            "delete_extraction",
            "millm_delete_circuit",
            "millm_circuit_sensing_clear",
        ):
            assert tool in flagged, (
                f"{tool} destroys data and is not flagged in the contract"
            )
