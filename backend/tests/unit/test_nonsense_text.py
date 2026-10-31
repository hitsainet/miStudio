"""ENH-006 D84: the seeded nonsense-text generator, its route and its MCP tool.

The generator (`services/nonsense_text.py`) makes the negatives-only set rung 2's nonsense-text check reads.
Two properties are load-bearing:

* DETERMINISM — the same (rows, seed, prose) gives the same rows, so a set can be re-made and a result
  reproduced;
* NEVER SSN-SHAPED — the probes this check exists for detect SSNs; a nonsense row holding one would be a
  positive read as a negative. No run of 9+ digits and no 3-2-4 digit grouping, ever.

Negative controls (0xcc/reviews/rung2_tighten_controls_2026-10-29.md): `defuse_ssn_shapes` made a no-op ->
red; `digit_noise` emitting 3-2-4 groups -> red; the route's registration line removed -> red.
"""
from __future__ import annotations

import random
import re

import pytest

from src.core.config import settings
from src.services import nonsense_text as N

API = "/api/v1/probe-monitors"
#: An INDEPENDENT statement of the shapes (not `N.SSN_SHAPED`), so a loosened module regex cannot pass.
NINE_DIGITS = re.compile(r"\d{9}")
GROUPED = re.compile(r"\d{3}\D{0,3}\d{2}\D{0,3}\d{4}")
PROSE = [f"The {w} river ran past the old mill at dawn. Number {i} 123-45-6789 was on a card. "
         f"Then 987654321 was read aloud by the {w} keeper!" for i, w in enumerate(N.COMMON_WORDS[:60])]


def _shaped(text: str) -> bool:
    return bool(NINE_DIGITS.search(text) or GROUPED.search(text))


class TestDeterminism:
    def test_the_same_seed_gives_the_same_rows(self):
        a, b = N.generate(700, 11), N.generate(700, 11)
        assert a.texts == b.texts and a.kinds == b.kinds

    def test_a_different_seed_gives_different_rows(self):
        assert N.generate(300, 1).texts != N.generate(300, 2).texts

    def test_with_prose_too(self):
        assert N.generate(400, 5, prose=PROSE).texts == N.generate(400, 5, prose=PROSE).texts


class TestNeverSsnShaped:
    @pytest.mark.parametrize("seed", range(12))
    def test_no_row_of_any_seed_carries_a_shape(self, seed):
        generated = N.generate(350, seed, prose=PROSE if seed % 2 else None)
        bad = [(k, t[:80]) for t, k in zip(generated.texts, generated.kinds) if _shaped(t)]
        assert bad == []

    def test_digit_noise_is_safe_by_construction(self):
        """Before the backstop: the kind that emits digits on purpose never makes a shape on its own."""
        for i in range(300):
            assert not _shaped(N._digit_noise(random.Random(i), 3000))

    @pytest.mark.parametrize("text", ["123-45-6789", "123456789", "123 45 6789", "SSN 123.45.6789 ok",
                                      "x1234567890123x", "١٢٣٤٥٦٧٨٩", "12345 6789", "000-00-0000",
                                      "call 555-12-3456 / 4567 89 0123"])
    def test_the_backstop_breaks_every_crafted_shape(self, text):
        assert _shaped(text)
        out = N.defuse_ssn_shapes(text)
        assert not _shaped(out) and len(out) == len(text)

    def test_shuffled_prose_with_ssns_in_it_comes_out_clean(self):
        generated = N.generate(400, 3, prose=PROSE)
        prose_rows = [t for t, k in zip(generated.texts, generated.kinds) if k == "shuffled_prose"]
        assert prose_rows and all("river" in t or "keeper" in t or "card" in t or "Then" in t for t in prose_rows)
        assert not any(_shaped(t) for t in prose_rows)


class TestKindsAndLengths:
    def test_every_kind_is_present_and_prose_is_skipped_without_a_corpus(self):
        generated = N.generate(700, 0)
        assert set(generated.kinds) == set(N.KINDS) - {"shuffled_prose"}
        assert "shuffled_prose" in generated.skipped
        assert set(N.generate(700, 0, prose=PROSE).kinds) == set(N.KINDS)
        assert max(generated.counts.values()) - min(generated.counts.values()) <= 1   # near-equal shares

    def test_lengths_run_from_a_token_to_thousands(self):
        texts = N.generate(2000, 4).texts
        words = sorted(len(t.split()) for t in texts)
        assert words[0] <= 3 and words[-1] >= 1500          # ~4,000 tokens at the top
        assert max(len(t) for t in texts) < 40_000

    def test_the_mixed_scripts_kind_mixes_scripts(self):
        generated = N.generate(700, 2)
        row = max((t for t, k in zip(generated.texts, generated.kinds) if k == "mixed_scripts"), key=len)
        assert len({ord(c) >> 8 for c in row if not c.isspace()}) >= 5


# ── the route and the MCP tool ─────────────────────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


class TestTheRoute:
    async def test_one_call_makes_the_dataset_and_its_view(self, client, data_dir):
        response = await client.post(f"{API}/nonsense-text", json={"rows": 300, "seed": 9})
        assert response.status_code == 201, response.text
        body = response.json()
        view = body["view"]
        assert view["role"] == "nonsense_text" and view["label_mapping"] == {"*": "negative"}
        assert view["counts"]["negative"] == 300 and view["counts"].get("positive", 0) == 0
        assert view["dataset_id"] == body["dataset"]["id"] and view["group_columns"] == ["kind"]
        assert body["manifest"]["seed"] == 9 and sum(body["manifest"]["kinds"].values()) == 300
        assert "shuffled_prose" in body["manifest"]["skipped"]
        from src.services.probe_monitor_service import open_split, resolve_dataset_path

        rows = open_split(resolve_dataset_path(body["dataset"]["raw_path"]))
        assert rows.column_names == ["text", "kind"]
        assert list(rows["text"]) == N.generate(300, 9).texts       # the stored rows ARE the seeded rows

    async def test_a_prose_corpus_feeds_the_shuffled_kind(self, client, async_session, data_dir):
        from tests.support.detector_fixtures import dataset_row, write_split

        ds = await dataset_row(async_session, write_split(data_dir, {"body": PROSE}))
        await async_session.commit()
        response = await client.post(f"{API}/nonsense-text", json={
            "rows": 200, "seed": 1, "prose_dataset_id": str(ds.id), "prose_column": "body", "register_view": False})
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["view"] is None and body["manifest"]["kinds"]["shuffled_prose"] > 0
        missing = await client.post(f"{API}/nonsense-text", json={
            "rows": 200, "prose_dataset_id": str(ds.id), "prose_column": "nope"})
        assert missing.status_code == 422 and "nope" in missing.text

    async def test_the_route_is_in_the_live_app(self, client):
        paths = (await client.get("/api/openapi.json")).json()["paths"]
        assert "post" in paths[f"{API}/nonsense-text"]


def test_the_mcp_tool_sends_its_defaults_to_the_route():
    """Payload and count: a default call posts exactly the defaults, once."""
    import asyncio

    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import probe_monitors
    from tests.unit.test_reachability import RecordingClient

    mcp = MCPServer("t")
    client = RecordingClient()
    probe_monitors.register(mcp, client, MCPSettings(auth_token="x" * 32))
    assert "create_nonsense_text_dataset" in {t.name for t in asyncio.run(mcp.list_tools())}
    asyncio.run(mcp.call_tool("create_nonsense_text_dataset", {}))
    assert client.calls == [("POST", "/probe-monitors/nonsense-text",
                             {"json_body": {"rows": 2000, "seed": 0, "prose_column": "text", "register_view": True}})]
