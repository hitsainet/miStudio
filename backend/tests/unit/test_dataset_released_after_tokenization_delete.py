"""Deleting a tokenization must release the dataset when nothing is in flight.

Reported 2026-08-24. The Datasets modal showed an amber "Processing" badge on
`hard-negatives` beside two READY tokenizations, with an idle worker and no
running job. `datasets.status` had been stuck at PROCESSING since a cancelled
tokenization hours earlier. The delete handler reset the dataset only when the
row it removed was the LAST one, so cancelling on a dataset that already had
finished tokenizations left it PROCESSING forever. Whether other tokenizations
EXIST says nothing about whether work is in flight.

WHY THIS FILE WAS REWRITTEN (2026-08-25). The first version of these tests read
the handler's source with `inspect.getsource` and asserted on substrings. It
never called the handler. The reset block it was checking referenced `Dataset`,
which this module never imported -- so every delete raised

    NameError: name 'Dataset' is not defined

and returned 500, while all four source assertions passed. The delete button
did nothing for a day and the suite stayed green.

A guard that reads source proves the source LOOKS right. Only executing the
path proves it runs. These tests drive the real handler.
"""

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from src.api.v1.endpoints import datasets as ep
from src.models.dataset import DatasetStatus
from src.models.dataset_tokenization import TokenizationStatus


class _Result:
    """One `await db.execute(...)` outcome."""

    def __init__(self, scalar=None, many=None):
        self._scalar, self._many = scalar, many

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return SimpleNamespace(all=lambda: self._many or [])


class _DB:
    """Hands back queued results in order, exactly as the handler asks for them."""

    def __init__(self, results):
        self._results = list(results)
        self.deleted, self.commits = [], 0

    async def execute(self, _stmt):
        assert self._results, "the handler issued more queries than expected"
        return self._results.pop(0)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def commit(self):
        self.commits += 1


def _tokenization(status=TokenizationStatus.READY):
    return SimpleNamespace(
        id="tok_x_m_1_512",
        status=status,
        tokenized_path=None,          # keeps the file-cleanup branch out of it
        model_id="m_1",
    )


async def _delete(db, dataset_id):
    with patch.object(ep, "get_redis_client", lambda: SimpleNamespace(delete=lambda k: 1)), \
         patch.object(ep, "emit_tokenization_status", lambda **kw: True), \
         patch.object(ep, "emit_dataset_progress", lambda **kw: True):
        return await ep.delete_dataset_tokenization(
            dataset_id=dataset_id, tokenization_id="tok_x_m_1_512", db=db
        )


@pytest.mark.asyncio
class TestTheHandlerActuallyRuns:
    async def test_deleting_the_last_active_row_releases_a_stuck_dataset(self):
        """The load-bearing path. A NameError anywhere in it surfaces here."""
        dataset_id = uuid4()
        dataset = SimpleNamespace(
            id=dataset_id,
            status=DatasetStatus.PROCESSING,
            progress=80.0,
            error_message="whatever",
        )
        db = _DB([
            _Result(scalar=_tokenization()),        # find the tokenization
            _Result(many=[]),                       # remaining tokenizations
            _Result(scalar=dataset),                # the parent dataset
        ])

        await _delete(db, dataset_id)

        assert db.commits == 1
        assert dataset.status == DatasetStatus.READY, (
            "the dataset stayed PROCESSING, so the card keeps an amber badge "
            "with no job behind it"
        )
        assert dataset.error_message is None

    async def test_it_releases_even_when_finished_tokenizations_remain(self):
        """The original defect: 'others exist' is not 'work is running'."""
        dataset_id = uuid4()
        dataset = SimpleNamespace(
            id=dataset_id, status=DatasetStatus.PROCESSING, progress=0.0,
            error_message=None,
        )
        db = _DB([
            _Result(scalar=_tokenization()),
            _Result(many=[
                SimpleNamespace(status=TokenizationStatus.READY),
                SimpleNamespace(status=TokenizationStatus.READY),
            ]),
            _Result(scalar=dataset),
        ])

        await _delete(db, dataset_id)
        assert dataset.status == DatasetStatus.READY

    @pytest.mark.parametrize(
        "live", [TokenizationStatus.QUEUED, TokenizationStatus.PROCESSING]
    )
    async def test_it_leaves_the_dataset_alone_while_work_is_in_flight(self, live):
        dataset_id = uuid4()
        dataset = SimpleNamespace(
            id=dataset_id, status=DatasetStatus.PROCESSING, progress=0.0,
            error_message=None,
        )
        db = _DB([
            _Result(scalar=_tokenization()),
            _Result(many=[SimpleNamespace(status=live)]),
            # no third result: the handler must not query for the dataset
        ])

        await _delete(db, dataset_id)
        assert dataset.status == DatasetStatus.PROCESSING

    async def test_a_downloading_dataset_is_not_flipped_to_ready(self):
        """The reset targets a STUCK dataset, not every status."""
        dataset_id = uuid4()
        dataset = SimpleNamespace(
            id=dataset_id, status=DatasetStatus.DOWNLOADING, progress=42.0,
            error_message=None,
        )
        db = _DB([
            _Result(scalar=_tokenization()),
            _Result(many=[]),
            _Result(scalar=dataset),
        ])

        await _delete(db, dataset_id)
        assert dataset.status == DatasetStatus.DOWNLOADING
        assert dataset.progress == 42.0

    async def test_a_missing_tokenization_is_a_404_not_a_500(self):
        from fastapi import HTTPException

        db = _DB([_Result(scalar=None)])
        with pytest.raises(HTTPException) as exc:
            await _delete(db, uuid4())
        assert exc.value.status_code == 404

    async def test_a_running_tokenization_is_a_409_not_a_500(self):
        from fastapi import HTTPException

        db = _DB([_Result(scalar=_tokenization(TokenizationStatus.PROCESSING))])
        with pytest.raises(HTTPException) as exc:
            await _delete(db, uuid4())
        assert exc.value.status_code == 409
