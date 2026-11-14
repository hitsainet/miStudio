"""The REAL download task body with a pinned revision (035 FR-4, FR-6, FR-7; TI5, TI6).

`download_dataset_task` is called synchronously (Celery `__call__` binds `self`), against real rows
in the test database. Only the network is faked: `datasets.load_dataset` returns a real in-memory
Arrow dataset, so `save_to_disk`, the row counts and the cleanup run their production code. Every
fake records its calls so the PAYLOAD and the CALL COUNT are asserted.

MUTATION CONTROLS (recorded in 0xcc/reviews/035_implementation_controls_2026-10-18.md):
  C3  `raw_dataset_dirname` ignores the revision      → the pinned-path and two-revision cancel tests fail
  C7  the task builds the path without `revision`     → the pinned-path test fails
  C8  `load_dataset` loses `revision=`                → the payload test fails
  C9  `retry_params` loses "revision"                 → the failure-records-revision test fails
  C10 the success-path in-flight guard removed        → the shared-cache test fails
  C21 `reconcile_revision` returns the kwarg only      → the row-fallback test fails
"""
from __future__ import annotations

import contextlib
import uuid

import pytest

from src.core import cancellation
from src.core.config import settings
from src.core.database import SyncSessionLocal
from src.models.dataset import Dataset, DatasetStatus
from src.models.task_queue import TaskQueue
from src.services.dataset_view_identity import hf_cache_dirname, raw_dataset_dirname
from src.workers import dataset_tasks

pytestmark = pytest.mark.asyncio

C1 = "1" * 40
C2 = "2" * 40
REPO = "org/ds"


@pytest.fixture
def env(async_engine, tmp_path, monkeypatch):
    """Tables exist (async_engine), files go to tmp, the websocket and the watchdog are inert."""
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(dataset_tasks, "emit_dataset_progress", lambda *a, **k: None)
    monkeypatch.setattr(cancellation, "CancelWatchdog", lambda *a, **k: contextlib.nullcontext())
    calls = []
    behaviour = {"raise": None}

    def fake_load_dataset(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        if behaviour["raise"] is not None:
            raise behaviour["raise"]
        from datasets import Dataset as HFDataset

        return HFDataset.from_dict({"text": ["a", "b", "c"]})

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return {"calls": calls, "behaviour": behaviour, "root": tmp_path / "datasets"}


def _row(revision, status=DatasetStatus.DOWNLOADING, config="c", split="train", row_id=None):
    with SyncSessionLocal() as db:
        row = Dataset(
            id=row_id or uuid.uuid4(), name="ds", source="HuggingFace", hf_repo_id=REPO, status=status,
            revision=revision, extra_metadata={"config": config, "split": split},
        )
        db.add(row)
        db.commit()
        return str(row.id)


def _get(dataset_id):
    with SyncSessionLocal() as db:
        return db.query(Dataset).filter(Dataset.id == uuid.UUID(dataset_id)).first()


def _run(dataset_id, revision, config="c", split="train"):
    return dataset_tasks.download_dataset_task(
        dataset_id=dataset_id, repo_id=REPO, split=split, config=config, revision=revision
    )


async def test_it_downloads_EXACTLY_the_commit_into_its_own_directory(env):
    dataset_id = _row(C1)
    result = _run(dataset_id, C1)
    assert len(env["calls"]) == 1, env["calls"]
    call = env["calls"][0]
    assert call["revision"] == C1 and call["name"] == "c" and call["split"] == "train"
    want = env["root"] / raw_dataset_dirname(REPO, "c", "train", C1)
    assert want.name.endswith(f"__rev-{C1}")
    assert result["raw_path"] == str(want) and want.exists()
    row = _get(dataset_id)
    assert row.status == DatasetStatus.READY and row.raw_path == str(want) and row.revision == C1


async def test_a_dispatch_that_dropped_the_kwarg_still_downloads_the_rows_commit(env):
    """TI5 / C21: the column is the authority; a retry row written before 035 cannot unpin."""
    dataset_id = _row(C1)
    _run(dataset_id, None)
    assert [c["revision"] for c in env["calls"]] == [C1]


async def test_a_DIFFERENT_commit_than_the_row_records_is_refused(env):
    dataset_id = _row(C1)
    with pytest.raises(ValueError, match="refusing"):
        _run(dataset_id, C2)
    assert env["calls"] == [], "it downloaded before refusing"
    row = _get(dataset_id)
    assert row.status == DatasetStatus.ERROR and C1 in row.error_message and C2 in row.error_message


async def test_a_refusal_keeps_its_own_words_when_the_id_contains_404(env):
    """CI flake 2026-10-24 (run 40138656496): the refusal names the dataset's random UUID, and the
    error mapper read any text containing "404" as "not found" — so roughly 1 run in 140 replaced the
    refusal with a false "Dataset not found". The refusal must reach the row verbatim."""
    dataset_id = _row(C1, row_id=uuid.UUID("40440440-4044-4404-8404-404404404404"))
    with pytest.raises(ValueError, match="refusing"):
        _run(dataset_id, C2)
    row = _get(dataset_id)
    assert row.status == DatasetStatus.ERROR
    assert C1 in row.error_message and C2 in row.error_message and "not found" not in row.error_message


async def test_a_legacy_row_downloads_unpinned_as_before(env):
    dataset_id = _row(None, config=None, split=None)
    result = _run(dataset_id, None, config=None, split=None)
    assert [c["revision"] for c in env["calls"]] == [None]
    assert result["raw_path"].endswith("/org_ds")


async def test_a_failure_records_the_revision_for_the_retry(env):
    """C9 / FR-7: the retry must download the same commit."""
    dataset_id = _row(C1)
    env["behaviour"]["raise"] = RuntimeError("connection reset")
    with pytest.raises(RuntimeError):
        _run(dataset_id, C1)
    with SyncSessionLocal() as db:
        entry = db.query(TaskQueue).filter(TaskQueue.entity_id == dataset_id).one()
    assert entry.retry_params == {"repo_id": REPO, "split": "train", "config": "c", "revision": C1}


async def test_cancelling_one_revision_leaves_the_others_directory(env):
    """FR-6 / D13: two commits of one split never share a cancel path."""
    keep = _row(C2, status=DatasetStatus.READY)
    kept_dir = env["root"] / raw_dataset_dirname(REPO, "c", "train", C2)
    kept_dir.mkdir(parents=True)
    (kept_dir / "data.arrow").write_text("completed neighbour")

    cancelled = _row(C1)
    partial = env["root"] / raw_dataset_dirname(REPO, "c", "train", C1)
    partial.mkdir(parents=True)
    env["behaviour"]["raise"] = cancellation.OperatorCancelled("dataset_download", cancelled)
    result = _run(cancelled, C1)
    assert result["status"] == "cancelled"
    assert not partial.exists(), "the cancelled revision's own partial output was left"
    assert (kept_dir / "data.arrow").read_text() == "completed neighbour"
    assert _get(keep).raw_path is None  # untouched row


async def test_the_shared_cache_is_KEPT_while_another_revision_downloads(env, monkeypatch):
    """C10 / TI6: the arrow tree is shared by every revision of the repo."""
    monkeypatch.setattr(settings, "auto_cleanup_after_download", True, raising=False)
    _row(C2, status=DatasetStatus.DOWNLOADING)  # another download in flight
    cache = env["root"] / hf_cache_dirname(REPO)
    cache.mkdir(parents=True)
    (cache / "shard").write_text("x")
    _run(_row(C1), C1)
    assert cache.exists(), "the success path deleted a cache another download is writing to"


async def test_the_shared_cache_IS_cleaned_when_nothing_else_downloads(env, monkeypatch):
    """Negative control for the test above: the cleanup still runs, so the guard is not a no-op."""
    monkeypatch.setattr(settings, "auto_cleanup_after_download", True, raising=False)
    _row(C2, status=DatasetStatus.READY)
    cache = env["root"] / hf_cache_dirname(REPO)
    cache.mkdir(parents=True)
    _run(_row(C1), C1)
    assert not cache.exists()


def test_the_in_flight_check_fails_closed(monkeypatch):
    def boom():
        raise RuntimeError("database is down")

    monkeypatch.setattr("src.core.database.SyncSessionLocal", boom)
    assert dataset_tasks._no_other_download_in_flight(REPO, "x") is False


class TestReconcile:
    def test_cases(self):
        r = dataset_tasks.reconcile_revision
        assert r(None, None, "d") is None
        assert r(None, C1, "d") == C1
        assert r(C1, C1, "d") == C1
        with pytest.raises(ValueError):
            r(C1, C2, "d")
        with pytest.raises(ValueError):
            r(C1, None, "d")
