"""A page of probe dataset views for an agent (ENH-006 G35).

`list_probe_datasets` read `GET /probe-monitors/datasets`, every view whole: 465,000 characters for
134 views on the node (2026-11-07), most of it each column's snapshot path and digest. Finding a
stack's seven train views meant reading all of them. The tool now reads
`GET /probe-monitors/datasets-page`: filtered by name or id, role and behaviour, paged with the
total that matched, and a summary per view unless `detail=full`.

Through the real route over the real database. The tool's call and payload are pinned in
`test_reachability.EXPECTED_CALLS`; its DEFAULTS are pinned here, because the ledger passes
every argument and a default of `full` would be invisible to it.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.asyncio

API = "/api/v1/probe-monitors"
T0 = datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc)


def _snapshots(view_id: str, names: list[str]) -> dict:
    return {
        n: {"path": f"/data/probe_monitor_views/{view_id}/columns/{uuid.uuid4().hex[:24]}",
            "dtype": "dictionary", "n_rows": 32732, "sha256": "a" * 64, "n_values": 10}
        for n in names
    }


async def _world(session):
    """Nine views: a seven-view stack with one name part, an eval, a calibration, an archived one."""
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.probe_monitor import ProbeMonitorDataset

    dataset = Dataset(id=uuid.uuid4(), name="ssn-stack", source="local",
                      status=DatasetStatus.READY, raw_path="/x", extra_metadata={})
    session.add(dataset)
    await session.flush()

    def view(name, role, minutes, behaviour="ssn", **kw):
        vid = f"pmd_{uuid.uuid4().hex[:12]}"
        return ProbeMonitorDataset(
            id=vid, name=name, dataset_id=dataset.id, input_column="text", label_column="label",
            label_mapping={"1": "positive", "0": "negative"}, role=role, behaviour=behaviour,
            counts={"positive": 300, "negative": 30000}, group_columns=["form", "kind"],
            column_snapshots=_snapshots(vid, ["form", "kind", "genre"]),
            positive_meaning="A US Social Security number written so a reader could recover it. " * 8,
            dataset_version={"kind": "midataworks.dataset-version/v1", "split": "train",
                             "commit": "c" * 40, "repository": "hitsainet/ssn", "version_id": "dv_1",
                             "recorded_at": "2026-11-06T20:00:00Z", "dataset_name": "ssn-d128",
                             "manifest_rows": 32732, "version_number": 54,
                             "downloaded_rows": 32732, "manifest_sha256": "d" * 64,
                             "publication_state": "published"},
            created_at=T0 + timedelta(minutes=minutes), **kw,
        )

    forms = ["digits", "spelled", "split_parts", "spaced", "mixed_words", "across_lines", "glued"]
    stack = [view(f"SSN-D128 {f} train", "train", i) for i, f in enumerate(forms)]
    others = {
        "eval": view("ssn-d128 unseen genre", "eval", 10),
        "calibration": view("everyday calibration", "calibration", 11, behaviour=None),
        # `%` and `_` are text here, not LIKE wildcards.
        "literal": view("rate 100%_kept", "eval", 12, behaviour="pii"),
        "archived": view("ssn-d128 old train", "train", 13, archived_at=T0, archived_by="operator"),
    }
    session.add_all([*stack, *others.values()])
    await session.commit()
    return [v.id for v in stack], {k: v.id for k, v in others.items()}


async def _page(client, **params):
    r = await client.get(f"{API}/datasets-page", params=params)
    assert r.status_code == 200, r.text
    return r.json()


class TestFilters:
    async def test_search_matches_the_name_without_case_and_finds_the_stack(self, client, async_session):
        stack, others = await _world(async_session)
        page = await _page(client, search="ssn-d128", role="train")
        assert page["total"] == 7
        assert {v["id"] for v in page["items"]} == set(stack)

    async def test_search_matches_an_id(self, client, async_session):
        stack, _ = await _world(async_session)
        page = await _page(client, search=stack[3][4:].upper())
        assert [v["id"] for v in page["items"]] == [stack[3]]

    async def test_percent_and_underscore_are_text_not_wildcards(self, client, async_session):
        _, others = await _world(async_session)
        assert [v["id"] for v in (await _page(client, search="100%_k"))["items"]] == [others["literal"]]
        # Read as wildcards, `_` matches any one character and `%` any run: both would find views.
        # "n-d_28", not "d_28": the search reads ids too, and a random `pmd_28…` id holds "d_28" literally
        # (about one run in thirty failed on it). A `-` never occurs in an id.
        assert (await _page(client, search="n-d_28"))["total"] == 0
        assert (await _page(client, search="ssn%train"))["total"] == 0

    async def test_behaviour_and_role_narrow_and_archived_stay_hidden(self, client, async_session):
        stack, others = await _world(async_session)
        assert (await _page(client, behaviour="ssn"))["total"] == 8  # 7 + the eval; not archived
        assert (await _page(client, behaviour="pii"))["total"] == 1
        assert (await _page(client, role="calibration"))["total"] == 1
        with_archived = await _page(client, search="ssn-d128", include_archived="true")
        assert with_archived["total"] == 9
        assert others["archived"] in {v["id"] for v in with_archived["items"]}


class TestPaging:
    async def test_pages_are_newest_first_disjoint_and_cover_the_total(self, client, async_session):
        stack, _ = await _world(async_session)
        seen = []
        for offset in (0, 3, 6):
            page = await _page(client, search="ssn-d128", role="train", offset=offset, limit=3)
            assert (page["total"], page["offset"], page["limit"]) == (7, offset, 3)
            seen += [v["id"] for v in page["items"]]
        assert seen == list(reversed(stack))  # created a minute apart, newest first

    async def test_a_tie_on_created_at_is_broken_by_id(self, client, async_session):
        from sqlalchemy import update

        from src.models.probe_monitor import ProbeMonitorDataset

        stack, _ = await _world(async_session)
        await async_session.execute(update(ProbeMonitorDataset).values(created_at=T0))
        await async_session.commit()
        seen = []
        for offset in range(0, 7, 2):
            seen += [v["id"] for v in (await _page(client, search="ssn-d128", role="train",
                                                   offset=offset, limit=2))["items"]]
        assert seen == sorted(stack, reverse=True)

    async def test_the_default_page_is_fifty_and_the_cap_two_hundred(self, client, async_session):
        await _world(async_session)
        assert (await _page(client))["limit"] == 50
        r = await client.get(f"{API}/datasets-page", params={"limit": 201})
        assert r.status_code == 422


class TestSummary:
    async def test_a_summary_names_its_columns_and_version_and_drops_the_bulk(self, client, async_session):
        stack, _ = await _world(async_session)
        page = await _page(client, search=stack[0])
        assert page["detail"] == "summary"
        item = page["items"][0]
        assert item["columns"] == ["form", "genre", "kind"]
        assert item["dataset_version"] == {"version_id": "dv_1", "version_number": 54,
                                           "dataset_name": "ssn-d128", "split": "train",
                                           "publication_state": "published"}
        assert item["group_columns"] == ["form", "kind"]
        assert item["counts"] == {"positive": 300, "negative": 30000}
        assert item["behaviour"] == "ssn" and item["role"] == "train"
        for gone in ("column_snapshots", "positive_meaning", "positive_labels_confirmed",
                     "rule_set_listing", "label_mapping"):
            assert gone not in item
        text = json.dumps(item)
        assert "/data/probe_monitor_views" not in text and "a" * 64 not in text and "c" * 40 not in text

    async def test_a_summary_is_a_fraction_of_the_whole_row(self, client, async_session):
        stack, _ = await _world(async_session)
        summary = await _page(client, search=stack[0])
        whole = await _page(client, search=stack[0], detail="full")
        assert len(json.dumps(summary["items"])) * 3 < len(json.dumps(whole["items"]))

    async def test_full_is_the_view_as_get_gives_it(self, client, async_session):
        stack, _ = await _world(async_session)
        page = await _page(client, search="ssn-d128", role="train", detail="full", limit=2)
        for item in page["items"]:
            one = await client.get(f"{API}/datasets/{item['id']}")
            assert item == one.json()

    async def test_a_view_without_snapshots_or_manifest_summarises(self, client, async_session):
        from sqlalchemy import update

        from src.models.probe_monitor import ProbeMonitorDataset

        stack, _ = await _world(async_session)
        await async_session.execute(update(ProbeMonitorDataset).where(ProbeMonitorDataset.id == stack[0])
                                    .values(column_snapshots=None, dataset_version=None))
        await async_session.commit()
        item = (await _page(client, search=stack[0]))["items"][0]
        assert item["columns"] == [] and item["dataset_version"] is None


class TestTheUIListIsUnchanged:
    async def test_the_whole_list_still_answers_whole_rows(self, client, async_session):
        await _world(async_session)
        r = await client.get(f"{API}/datasets", params={"include_archived": "true"})
        assert r.status_code == 200
        rows = r.json()
        assert isinstance(rows, list) and len(rows) == 11
        assert all("column_snapshots" in row for row in rows)


def test_the_tool_defaults_to_a_summary_page_of_fifty():
    """The ledger passes every argument, so a default of `full` or a larger page passes it."""
    from tests.unit.test_reachability import _tools_by_name

    manager, recorder = _tools_by_name()
    asyncio.run(manager.call_tool("list_probe_datasets", {}, context=None))
    assert recorder.calls == [("GET", "/probe-monitors/datasets-page",
                               {"role": None, "include_archived": None, "search": None,
                                "behaviour": None, "offset": 0, "limit": 50, "detail": "summary"})]
