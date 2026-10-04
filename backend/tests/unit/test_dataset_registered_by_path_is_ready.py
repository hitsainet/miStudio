"""A dataset whose data is already on disk is READY, not permanently "processing".

⚠ `create_dataset` read `DOWNLOADING if hf_repo_id else PROCESSING`. Registering an existing
directory by `raw_path` therefore landed in PROCESSING — and NOTHING moved it out, because
PROCESSING means "a download or tokenization job is working on this" and no such job exists for
a path that is already there.

The row sat permanently mid-flight: the UI showed it as not Ready, and `num_samples` and
`size_bytes` stayed NULL, so the record described none of the data it pointed at.

Reported 2026-09-29 against a generated corpus registered by path. It had blocked nothing — probe
views read `raw_path` directly and never consult status — which is exactly why it went unnoticed,
and why a status that is merely decorative is worth fixing rather than shrugging at: the next
thing to read it will believe it.

⚠ READY IS CHECKED, NOT ASSUMED. A status claiming the data is present when the path is missing
would be strictly worse than PROCESSING: a wrong "ready" is believed, and every consumer then
fails somewhere else with an error about a directory rather than about the row.
"""

from __future__ import annotations

import pytest

from src.models.dataset import DatasetStatus
from src.schemas.dataset import DatasetCreate
from src.services.dataset_service import DatasetService

datasets = pytest.importorskip("datasets")


@pytest.fixture
def one_split(tmp_path):
    path = tmp_path / "flat"
    datasets.Dataset.from_list([{"text": f"row {i}"} for i in range(7)]).save_to_disk(str(path))
    return str(path)


@pytest.fixture
def two_splits(tmp_path):
    path = tmp_path / "split"
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list([{"text": f"t{i}"} for i in range(11)]),
            "eval": datasets.Dataset.from_list([{"text": f"e{i}"} for i in range(4)]),
        }
    ).save_to_disk(str(path))
    return str(path)


class TestInspectingAnExistingPath:
    def test_it_counts_a_single_split(self, one_split):
        counted, measured, error = DatasetService._inspect_existing_path(one_split)
        assert error is None
        assert counted == 7
        assert measured and measured > 0

    def test_it_SUMS_the_splits_of_a_DatasetDict(self, two_splits):
        """⚠ `len()` of a DatasetDict is its SPLIT COUNT. Reporting 2 for 15 rows is the same
        mistake that once recorded 1 sample for a million."""
        counted, _measured, error = DatasetService._inspect_existing_path(two_splits)
        assert error is None
        assert counted == 15, f"counted {counted}; 11 train + 4 eval is 15, not the split count"

    def test_a_missing_path_is_an_error_not_a_count(self, tmp_path):
        counted, measured, error = DatasetService._inspect_existing_path(str(tmp_path / "nope"))
        assert counted is None and measured is None
        assert "does not exist" in error

    def test_a_path_that_is_not_a_dataset_is_an_error(self, tmp_path):
        """A directory of unrelated files must not register as a readable corpus."""
        junk = tmp_path / "junk"
        junk.mkdir()
        (junk / "a.txt").write_text("not arrow")
        counted, _measured, error = DatasetService._inspect_existing_path(str(junk))
        assert counted is None
        assert "not a readable dataset" in error

    def test_it_returns_an_error_rather_than_raising(self, tmp_path):
        """A registration that cannot see its data should leave a row saying so, not a 500 that
        records nothing."""
        assert DatasetService._inspect_existing_path(str(tmp_path / "gone"))[2] is not None


@pytest.mark.asyncio
class TestCreateSetsTheRightStatus:
    async def test_an_existing_path_is_READY_and_described(self, async_session, two_splits):
        created = await DatasetService.create_dataset(
            async_session,
            DatasetCreate(name="generated corpus", source="Custom", raw_path=two_splits),
        )
        assert created.status == DatasetStatus.READY
        assert created.num_samples == 15, "the row does not describe the data it points at"
        assert created.size_bytes and created.size_bytes > 0
        assert created.error_message is None

    async def test_a_missing_path_is_ERROR_and_says_why(self, async_session, tmp_path):
        """⚠ NOT READY. A wrong "ready" is believed; the failure then surfaces far away as a
        complaint about a directory rather than about this row."""
        created = await DatasetService.create_dataset(
            async_session,
            DatasetCreate(name="ghost", source="Custom", raw_path=str(tmp_path / "absent")),
        )
        assert created.status == DatasetStatus.ERROR
        assert "does not exist" in (created.error_message or "")
        assert created.num_samples is None

    async def test_an_hf_repo_still_starts_DOWNLOADING(self, async_session):
        """⚠ Specificity. A HuggingFace dataset has work to do before it is ready, and marking
        it READY at creation would claim data nothing has fetched."""
        created = await DatasetService.create_dataset(
            async_session,
            DatasetCreate(name="remote", source="HuggingFace", hf_repo_id="org/corpus"),
        )
        assert created.status == DatasetStatus.DOWNLOADING

    async def test_an_hf_repo_WITH_a_path_still_downloads(self, async_session, one_split):
        """The repo id is what says a fetch is pending; a path beside it is the destination, not
        evidence the fetch already happened."""
        created = await DatasetService.create_dataset(
            async_session,
            DatasetCreate(
                name="remote", source="HuggingFace", hf_repo_id="org/corpus", raw_path=one_split
            ),
        )
        assert created.status == DatasetStatus.DOWNLOADING

    async def test_neither_a_path_nor_a_repo_stays_PROCESSING(self, async_session):
        """The original behaviour, kept for the case it was actually right for: a row created
        ahead of an upload has no data yet and nothing to verify."""
        created = await DatasetService.create_dataset(
            async_session, DatasetCreate(name="pending", source="Local")
        )
        assert created.status == DatasetStatus.PROCESSING
