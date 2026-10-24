"""Browsing a probe dataset view's rows (operator request 2026-10-21).

THE PROPERTY THAT MATTERS: the label each row shows and the counts the view stores are ONE
computation. The view here is created through the real `POST /datasets` route, so its stored
counts come from creation; the browser is then asked for every outcome and must reproduce those
counts exactly, row for row. The fixture covers all five outcomes on purpose — a fixture with
only positives and negatives would agree with a browser that ignored the keyword filter and the
exclusions (fixtures-agree-by-construction).
"""
from __future__ import annotations

import uuid

import pytest

datasets = pytest.importorskip("datasets")

pytestmark = pytest.mark.asyncio

API = "/api/v1/probe-monitors"
MAPPING = {"yes": "positive", "no": "negative", "skip": "excluded"}
FILTER = {"terms": ["card"], "mode": "any", "case_sensitive": False}
LONG = "x" * 25_000

# (input, label) — and what the view must make of it.
ROWS = (
    [(f"my card number {i}", "yes") for i in range(25)]                     # positive
    + [(f"a card game {i}", "no") for i in range(24)]                       # negative
    + [([{"role": "user", "content": "lost CARD"}, {"role": "assistant", "content": "ok"}], "no")]
    + [(f"nothing here {i}", "yes") for i in range(4)]                      # filtered_out
    + [(f"card skip {i}", "skip") for i in range(3)]                        # excluded
    + [(f"card maybe {i}", "maybe") for i in range(2)]                      # unparseable: unmapped
    + [("", "yes"), ("   ", "no")]                                           # unparseable: empty
)
EXPECTED = {"positive": 25, "negative": 25, "filtered_out": 4, "excluded": 3, "unparseable": 4}


def _records():
    out = []
    for i, (text, label) in enumerate(ROWS):
        out.append({
            "prompt": text if isinstance(text, str) else None,
            "chat": text if not isinstance(text, str) else None,
            "label": label,
            "pair": f"p{i // 2}",
            "source": f"src-{i % 3}",
            "notes": LONG if i == 0 else f"note {i}",
        })
    return out


@pytest.fixture
def download(tmp_path):
    """One split with a string input column, and a second whose input is a chat list."""
    path = tmp_path / "pii-segments"
    recs = _records()
    plain = [{k: v for k, v in r.items() if k != "chat"} for r in recs]
    for r, (text, _) in zip(plain, ROWS):
        r["prompt"] = text if isinstance(text, str) else "lost CARD ok"
    datasets.DatasetDict({
        "train": datasets.Dataset.from_list(plain),
        "test": datasets.Dataset.from_list([
            {"chat": r["chat"] or [{"role": "user", "content": t}], "label": r["label"]}
            for r, (t, _) in zip(recs, ROWS)
        ][:6]),
    }).save_to_disk(str(path))
    return path


async def _view(client, async_session, path, **overrides):
    from src.models.dataset import Dataset, DatasetStatus

    dataset = Dataset(id=uuid.uuid4(), name="pii-segments", source="local",
                      status=DatasetStatus.READY, raw_path=str(path), extra_metadata={})
    async_session.add(dataset)
    await async_session.commit()
    body = {
        "name": "card mentions", "dataset_id": str(dataset.id), "split": "train",
        "input_column": "prompt", "label_column": "label", "label_mapping": MAPPING,
        "keyword_filter": FILTER, "pair_column": "pair", "role": "eval", **overrides,
    }
    r = await client.post(f"{API}/datasets", json=body)
    assert r.status_code == 201, r.text
    return r.json(), dataset


async def _rows(client, view_id, **params):
    r = await client.get(f"{API}/datasets/{view_id}/rows", params=params)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(autouse=True)
def _fresh_cache():
    from src.services import probe_dataset_rows

    probe_dataset_rows.clear_cache()
    yield
    probe_dataset_rows.clear_cache()


class TestTheBrowserAndTheCountsAgree:
    async def test_the_stored_counts_are_the_fixture_s(self, client, async_session, download):
        view, _ = await _view(client, async_session, download)
        stored = {k: view["counts"][k] for k in EXPECTED}
        assert stored == EXPECTED, "the fixture must exercise all five outcomes"

    async def test_every_label_filter_reproduces_its_stored_count(self, client, async_session, download):
        view, _ = await _view(client, async_session, download)
        seen = set()
        for label, n in EXPECTED.items():
            page = await _rows(client, view["id"], label=label, limit=200)
            assert page["matched"] == n, label
            assert page["counts_agree"] is True, (label, page["recount"], page["counts"])
            assert {row["outcome"] for row in page["rows"]} == {label}
            seen |= {row["position"] for row in page["rows"]}
        assert seen == set(range(len(ROWS))), "every row lands in exactly one outcome"

    async def test_an_unfiltered_page_labels_rows_the_same_way(self, client, async_session, download):
        view, _ = await _view(client, async_session, download)
        page = await _rows(client, view["id"], limit=200)
        assert page["recount"] is None and page["counts_agree"] is None
        tally = {}
        for row in page["rows"]:
            tally[row["outcome"]] = tally.get(row["outcome"], 0) + 1
        assert tally == EXPECTED
        first = page["rows"][0]
        assert (first["raw_label"], first["mapped_label"], first["outcome"]) == ("yes", "positive", "positive")
        unmapped = next(r for r in page["rows"] if r["raw_label"] == "maybe")
        assert (unmapped["mapped_label"], unmapped["outcome"]) == (None, "unparseable")

    async def test_a_changed_download_is_reported_not_hidden(self, client, async_session, download, tmp_path):
        """The stored counts stay the authority; a recount that disagrees says so."""
        from src.models.dataset import Dataset
        from sqlalchemy import update

        view, dataset = await _view(client, async_session, download)
        moved = tmp_path / "rewritten"
        datasets.Dataset.from_list(
            [{"prompt": "card", "label": "yes", "pair": "p", "source": "s", "notes": "n"}] * 3
        ).save_to_disk(str(moved))
        await async_session.execute(update(Dataset).where(Dataset.id == dataset.id).values(raw_path=str(moved)))
        await async_session.commit()
        page = await _rows(client, view["id"], label="positive")
        assert page["recount"]["positive"] == 3 and page["counts_agree"] is False


class TestPagingAndProjection:
    async def test_pages_tile_the_rows(self, client, async_session, download):
        view, _ = await _view(client, async_session, download)
        positions = []
        for offset in range(0, 70, 25):
            page = await _rows(client, view["id"], offset=offset, limit=25)
            assert page["matched"] == len(ROWS) == page["total_rows"]
            positions += [r["position"] for r in page["rows"]]
        assert positions == list(range(len(ROWS)))
        last = await _rows(client, view["id"], offset=50, limit=25)
        assert len(last["rows"]) == len(ROWS) - 50

    async def test_past_the_end_is_an_empty_page_not_an_error(self, client, async_session, download):
        view, _ = await _view(client, async_session, download)
        page = await _rows(client, view["id"], offset=10_000)
        assert page["rows"] == [] and page["matched"] == len(ROWS)
        page = await _rows(client, view["id"], offset=10_000, label="positive")
        assert page["rows"] == [] and page["matched"] == 25

    @pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"offset": -1},
                                        {"label": "maybe"}, {"columns": "nope"}])
    async def test_bad_requests_are_422(self, client, async_session, download, params):
        view, _ = await _view(client, async_session, download)
        r = await client.get(f"{API}/datasets/{view['id']}/rows", params=params)
        assert r.status_code == 422, (params, r.text)

    async def test_the_projection_keeps_the_view_columns(self, client, async_session, download):
        view, _ = await _view(client, async_session, download)
        page = await _rows(client, view["id"], columns="source", limit=2)
        assert page["returned_columns"] == ["prompt", "label", "pair", "source"]
        assert set(page["rows"][0]["values"]) == {"prompt", "label", "pair", "source"}
        assert page["other_columns"] == ["source", "notes"]
        none = await _rows(client, view["id"], columns="", limit=1)
        assert none["returned_columns"] == ["prompt", "label", "pair"]
        everything = await _rows(client, view["id"], limit=1)
        assert everything["returned_columns"] == ["prompt", "label", "pair", "source", "notes"]

    async def test_a_long_cell_is_cut_and_flagged(self, client, async_session, download):
        from src.services.probe_dataset_rows import MAX_CELL_CHARS

        view, _ = await _view(client, async_session, download)
        row = (await _rows(client, view["id"], limit=1))["rows"][0]
        assert row["truncated"] == ["notes"]
        assert len(row["values"]["notes"]) == MAX_CELL_CHARS


class TestSearch:
    async def test_search_is_case_insensitive_and_combines_with_the_label(
        self, client, async_session, download
    ):
        view, _ = await _view(client, async_session, download)
        page = await _rows(client, view["id"], search="CARD GAME", limit=200)
        assert page["matched"] == 24
        assert all("card game" in r["values"]["prompt"] for r in page["rows"])
        both = await _rows(client, view["id"], search="card", label="excluded")
        assert both["matched"] == 3

    def test_search_reads_a_conversation_s_message_text(self, download):
        """A chat-list input is searched by its MESSAGE TEXT, never by its JSON spelling."""
        from src.services.probe_dataset_rows import ViewSpec, browse

        spec = ViewSpec(id="pmd_chat", split="test", input_column="chat", label_column="label",
                        pair_column=None, label_mapping=MAPPING, keyword_filter=None)
        page = browse(spec, download, search="NUMBER 3")
        assert page["matched"] == 1 and page["rows"][0]["position"] == 3
        assert page["rows"][0]["input_kind"] == "messages"
        assert browse(spec, download, search="role")["matched"] == 0, "JSON keys are not text"


class TestRefusals:
    async def test_unknown_view_is_404(self, client, async_session):
        r = await client.get(f"{API}/datasets/pmd_nope/rows")
        assert r.status_code == 404

    async def test_a_missing_download_is_409(self, client, async_session, download):
        import shutil

        view, _ = await _view(client, async_session, download)
        shutil.rmtree(download)
        r = await client.get(f"{API}/datasets/{view['id']}/rows")
        assert r.status_code == 409


def test_classify_row_IS_what_build_examples_counts():
    """The shared classifier, without HTTP: a tally of `classify_row` over the fixture equals
    `build_examples`' counts. If either stops calling the other's logic, this is the red."""
    from src.services.probe_monitor_inputs import build_examples, classify_row

    inputs = [t if isinstance(t, str) else t for t, _ in ROWS]
    labels = [label for _, label in ROWS]
    built = build_examples(inputs, labels, MAPPING, keyword_filter=FILTER)
    tally = {}
    for raw_input, raw_label in zip(inputs, labels):
        outcome = classify_row(raw_input, raw_label, MAPPING, FILTER).outcome
        tally[outcome] = tally.get(outcome, 0) + 1
    assert tally == built.counts.as_dict() == EXPECTED


def test_the_service_refuses_what_the_route_refuses(download):
    """The route's Query bounds stop these first; the service refuses them on its own too, because
    the MCP-free callers (and a later route) must not get a silent empty page or a slice from -1."""
    from src.services.probe_dataset_rows import RowsRefused, ViewSpec, browse

    spec = ViewSpec(id="pmd_x", split="train", input_column="prompt", label_column="label",
                    pair_column=None, label_mapping=MAPPING, keyword_filter=None)
    for kwargs in ({"offset": -1}, {"limit": 0}, {"limit": 201}, {"label": "maybe"},
                   {"columns": ["nope"]}):
        with pytest.raises(RowsRefused):
            browse(spec, download, **kwargs)
