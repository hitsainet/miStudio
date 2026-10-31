"""Pinned downloads at the API (035 FR-1, FR-2, FR-3, FR-5; D1, D9, D10, D31, D32, D43).

Every test drives the REAL endpoint through the ASGI app (or, for concurrency, the endpoint
function on two real sessions). The Hub is faked at `hf_revision.HfApi` — so the endpoint's own
resolve call and the real error mapping both run — and `.delay` is a recorder whose PAYLOAD and
CALL COUNT are asserted: a "was called" check passes against a dispatch sending the wrong commit.

MUTATION CONTROLS (recorded in 0xcc/reviews/035_implementation_controls_2026-10-18.md):
  C4  remove the resolve call (commit := request.revision) → tag-stores-commit fails
  C5  dispatch `revision=request.revision` instead of the commit → dispatch payload test fails
  C6  remove `dataset_identity_lock`                          → the concurrency test fails
  C2' identity without the commit (revision=None in the lookup) → other-commit test fails
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

# The Hub's own HTTP module (httpx on hub 1.x, httpx2 on 2.x): a fake raising plain httpx errors
# kept these tests green while every real 2.x timeout escaped the mapping as a 500.
from huggingface_hub.utils import httpx
from sqlalchemy import func, select

from src.models.dataset import Dataset, DatasetStatus
from tests.support.fake_hub import COMMIT_A, COMMIT_B, FakeHub, install

pytestmark = pytest.mark.asyncio

URL = "/api/v1/datasets/download"


@pytest.fixture
def dispatch():
    with patch("src.workers.dataset_tasks.download_dataset_task.delay") as delay:
        delay.return_value = MagicMock(id="task-1")
        yield delay


async def _count(session) -> int:
    return (await session.execute(select(func.count()).select_from(Dataset))).scalar_one()


def _resp(status):
    return httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/x"))


class TestTheRefIsResolvedBeforeAnythingExists:
    async def test_a_tag_stores_the_COMMIT_not_the_tag(self, client, async_session, monkeypatch, dispatch):
        hub = install(monkeypatch, FakeHub(refs={"v1.0": COMMIT_B}))
        r = await client.post(URL, json={"repo_id": "org/ds", "split": "train", "revision": "v1.0"})
        assert r.status_code == 202, r.text
        assert r.json()["revision"] == COMMIT_B
        assert hub.calls and hub.calls[0]["repo_id"] == "org/ds" and hub.calls[0]["revision"] == "v1.0"
        row = (await async_session.execute(select(Dataset))).scalars().one()
        assert row.revision == COMMIT_B

    async def test_no_revision_pins_the_head_resolved_now(self, client, monkeypatch, dispatch):
        """D1: every new row is verifiable, including the ones nobody asked to pin."""
        install(monkeypatch, FakeHub(default=COMMIT_A))
        r = await client.post(URL, json={"repo_id": "org/ds"})
        assert r.status_code == 202 and r.json()["revision"] == COMMIT_A

    async def test_a_blank_revision_is_the_default_branch(self, client, monkeypatch, dispatch):
        hub = install(monkeypatch)
        r = await client.post(URL, json={"repo_id": "org/ds", "revision": "   "})
        assert r.status_code == 202
        assert hub.calls[0]["revision"] is None

    async def test_the_task_gets_the_resolved_commit_ONCE(self, client, monkeypatch, dispatch):
        """C5: a branch can move between the request and the worker, so the ref is never sent."""
        install(monkeypatch, FakeHub(refs={"main": COMMIT_B}))
        r = await client.post(
            URL, json={"repo_id": "org/ds", "config": "c", "split": "s", "revision": "main"}
        )
        assert r.status_code == 202
        assert dispatch.call_count == 1
        kwargs = dispatch.call_args.kwargs
        assert kwargs["revision"] == COMMIT_B
        assert (kwargs["repo_id"], kwargs["config"], kwargs["split"]) == ("org/ds", "c", "s")
        assert kwargs["dataset_id"] == r.json()["id"]

    @pytest.mark.parametrize(
        "raises, status",
        [
            (lambda: RepositoryNotFoundError("nope", response=_resp(404)), 422),
            (lambda: GatedRepoError("gated", response=_resp(403)), 403),
            (lambda: httpx.ReadTimeout("slow"), 502),
        ],
    )
    async def test_a_hub_failure_creates_NOTHING(
        self, client, async_session, monkeypatch, dispatch, raises, status
    ):
        install(monkeypatch, FakeHub(raises=raises))
        r = await client.post(URL, json={"repo_id": "org/ds", "revision": "v9"})
        assert r.status_code == status, r.text
        assert isinstance(r.json()["detail"], str) and "org/ds" in r.json()["detail"]
        assert await _count(async_session) == 0
        assert dispatch.call_count == 0


class TestTheIdentityIsTheCommit:
    async def _seed(self, session, revision, config=None, split=None):
        row = Dataset(
            name="seed", source="HuggingFace", hf_repo_id="org/ds", status=DatasetStatus.READY,
            revision=revision, extra_metadata={"config": config, "split": split},
        )
        session.add(row)
        await session.commit()
        return row

    async def test_the_same_commit_is_a_409_naming_commit_and_id(
        self, client, async_session, monkeypatch, dispatch
    ):
        seeded_id = str((await self._seed(async_session, COMMIT_A, "c", "s")).id)
        install(monkeypatch, FakeHub(default=COMMIT_A))
        r = await client.post(URL, json={"repo_id": "org/ds", "config": "c", "split": "s"})
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert isinstance(detail, str), "the store renders `detail` as a message (D32)"
        assert COMMIT_A in detail and seeded_id in detail and "config c" in detail
        assert dispatch.call_count == 0

    async def test_another_commit_is_a_second_row(self, client, async_session, monkeypatch, dispatch):
        await self._seed(async_session, COMMIT_A, "c", "s")
        install(monkeypatch, FakeHub(default=COMMIT_B))
        r = await client.post(URL, json={"repo_id": "org/ds", "config": "c", "split": "s"})
        assert r.status_code == 202, r.text
        assert await _count(async_session) == 2

    async def test_a_legacy_row_never_duplicates(self, client, async_session, monkeypatch, dispatch):
        """D10: a row with no recorded commit is not the commit the request resolved to."""
        await self._seed(async_session, None, "c", "s")
        install(monkeypatch)
        r = await client.post(URL, json={"repo_id": "org/ds", "config": "c", "split": "s"})
        assert r.status_code == 202
        assert r.json()["revision"] == COMMIT_A
        assert await _count(async_session) == 2

    async def test_a_legacy_row_reads_null_in_the_API(self, client, async_session):
        """FR-5: null, never a fabricated "latest"."""
        row = await self._seed(async_session, None)
        r = await client.get(f"/api/v1/datasets/{row.id}")
        assert r.status_code == 200 and r.json()["revision"] is None
        listed = await client.get("/api/v1/datasets")
        assert [d["revision"] for d in listed.json()["data"]] == [None]


class TestTheDuplicateCheckIsAtomic:
    async def test_two_simultaneous_requests_create_ONE_row(self, async_engine, monkeypatch, dispatch):
        """C6. Pre-existing defect: lookup then insert with nothing between them. The window is
        widened (a pause after the lookup) so the race is certain without the lock, not lucky."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        from fastapi import HTTPException

        from src.api.v1.endpoints import datasets as endpoint
        from src.schemas.dataset import DatasetDownloadRequest
        from src.services.dataset_service import DatasetService

        install(monkeypatch)
        real = DatasetService.get_dataset_by_view

        async def slow_lookup(*args, **kwargs):
            found = await real(*args, **kwargs)
            await asyncio.sleep(0.5)
            return found

        monkeypatch.setattr(DatasetService, "get_dataset_by_view", staticmethod(slow_lookup))
        maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        request = DatasetDownloadRequest(repo_id="org/ds", split="train")

        async def one():
            async with maker() as session:
                try:
                    await endpoint.download_dataset(request, session)
                    return 202
                except HTTPException as exc:
                    return exc.status_code

        outcomes = sorted(await asyncio.gather(one(), one()))
        async with maker() as session:
            rows = await _count(session)
        assert rows == 1, f"two concurrent requests created {rows} rows ({outcomes})"
        assert outcomes == [202, 409]
        assert dispatch.call_count == 1
