"""Tests for task-queue federation of dataset tokenizations and activation extractions.

BACKGROUND — why these tests exist
----------------------------------
The Monitor page's "Active Operations" panel renders whatever
``GET /api/v1/task-queue/active`` returns. That endpoint federates several job
tables. Two live operations were invisible there:

1. **Dataset tokenization** — has no ``task_queue`` row while running (the worker
   only writes one on failure) and had no federator.
2. **Model activation extraction** — ``_federated_extractions`` federates
   ``extraction_jobs`` (SAE *feature* extraction), a different table. The
   "Extract Activations" flow writes ``activation_extractions`` and had no
   federator.

THE CASING TRAP (the reason these tests are worth their weight)
---------------------------------------------------------------
``dataset_tokenizations.status`` and ``activation_extractions.status`` are
Postgres enums whose labels are the UPPERCASE Python enum *names*
('PROCESSING', 'EXTRACTING'), because SQLAlchemy persists ``.name``. But
``extraction_jobs.status`` uses lowercase labels ('extracting'). Copying the
existing federator's lowercase comparison silently matches ZERO rows — the query
succeeds, returns nothing, and the panel stays empty while the code looks right.

Verified against the live database: comparing these columns to lowercase strings
returned 0 rows while uppercase returned every row.

MUTATION CONTROLS (re-run these to prove the tests bite):
  * Change ``upper(t.status::text) IN :statuses`` back to ``t.status IN :statuses``
    with lowercase params -> the "reports a running X" tests must FAIL.
  * Delete the ``_federated_tokenizations`` / ``_federated_activation_extractions``
    ``.extend(...)`` call in ``list_active_tasks`` -> the wiring tests must FAIL.
"""

import uuid

import pytest

from src.api.v1.endpoints.task_queue import (
    _federated_activation_extractions,
    _federated_tokenizations,
    list_active_tasks,
)
from src.models.activation_extraction import ActivationExtraction, ExtractionStatus
from src.models.dataset import Dataset, DatasetStatus
from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from src.models.model import Model, ModelStatus, QuantizationFormat

pytestmark = pytest.mark.asyncio

# Rows are built through the ORM (not raw INSERTs) so Python-side column defaults
# are applied automatically — the schema gains NOT NULL columns over time and
# hand-written INSERTs rot.


async def _make_model(db) -> str:
    """Insert a minimal Model row (FK target for tokenizations/extractions)."""
    model = Model(
        id=f"m_{uuid.uuid4().hex[:8]}",
        name="granite-test-8b",
        architecture="llama",
        params_count=1_000_000,
        quantization=QuantizationFormat.Q4,
        status=ModelStatus.READY,
    )
    db.add(model)
    await db.flush()
    return model.id


async def _make_dataset(db, name: str = "OpenWebText-Test") -> str:
    dataset = Dataset(
        id=uuid.uuid4(),
        name=name,
        source="HuggingFace",
        status=DatasetStatus.PROCESSING,
    )
    db.add(dataset)
    await db.flush()
    return str(dataset.id)


async def _make_tokenization(db, dataset_id: str, model_id: str,
                             status: TokenizationStatus,
                             progress: float = 79.2) -> str:
    tok = DatasetTokenization(
        id=f"tok_{uuid.uuid4().hex[:12]}",
        dataset_id=uuid.UUID(dataset_id),
        model_id=model_id,
        max_length=512,
        tokenizer_repo_id="test/tokenizer",
        status=status,
        progress=progress,
    )
    db.add(tok)
    await db.flush()
    return tok.id


async def _make_extraction(db, model_id: str, dataset_id: str,
                           status: ExtractionStatus,
                           progress: float = 19.6) -> str:
    ext = ActivationExtraction(
        id=f"ext_{uuid.uuid4().hex[:12]}",
        model_id=model_id,
        # NB: activation_extractions.dataset_id is VARCHAR here, whereas
        # dataset_tokenizations.dataset_id is UUID — pass the string form.
        dataset_id=dataset_id,
        layer_indices=[33, 34],
        hook_types=["residual"],
        max_samples=10000,
        status=status,
        progress=progress,
        samples_processed=1200,
    )
    db.add(ext)
    await db.flush()
    return ext.id


class TestFederatedTokenizations:
    """A running tokenization must surface as an active operation."""

    async def test_reports_running_tokenization(self, async_session):
        """THE CASING REGRESSION. Fails if the query compares lowercase."""
        model_id = await _make_model(async_session)
        dataset_id = await _make_dataset(async_session, "OpenWebText-2M")
        tok_id = await _make_tokenization(async_session, dataset_id, model_id, TokenizationStatus.PROCESSING)

        rows = await _federated_tokenizations(async_session, ("QUEUED", "PROCESSING"))

        assert len(rows) == 1, (
            "a PROCESSING tokenization must be federated; 0 rows means the status "
            "comparison missed the UPPERCASE enum labels"
        )
        row = rows[0]
        assert row["id"] == tok_id
        assert row["status"] == "running"
        assert row["task_type"] == "tokenization"
        assert row["entity_type"] == "dataset"
        assert row["entity_id"] == dataset_id  # matches the failure-path task_queue row
        assert row["progress"] == pytest.approx(79.2)
        assert "OpenWebText-2M" in row["entity_info"]["name"]

    async def test_caller_passing_lowercase_still_matches(self, async_session):
        """The federator normalizes casing, so a lowercase caller still works.

        This pins the defensive ``upper()`` normalization: without it, only the
        exact uppercase spelling would match and the next caller would silently
        get nothing.
        """
        model_id = await _make_model(async_session)
        dataset_id = await _make_dataset(async_session)
        await _make_tokenization(async_session, dataset_id, model_id, TokenizationStatus.PROCESSING)

        rows = await _federated_tokenizations(async_session, ("queued", "processing"))

        assert len(rows) == 1

    async def test_queued_maps_to_queued(self, async_session):
        model_id = await _make_model(async_session)
        dataset_id = await _make_dataset(async_session)
        await _make_tokenization(async_session, dataset_id, model_id, TokenizationStatus.QUEUED, progress=0)

        rows = await _federated_tokenizations(async_session, ("QUEUED", "PROCESSING"))

        assert len(rows) == 1
        assert rows[0]["status"] == "queued"

    async def test_ready_tokenization_is_not_active(self, async_session):
        model_id = await _make_model(async_session)
        dataset_id = await _make_dataset(async_session)
        await _make_tokenization(async_session, dataset_id, model_id, TokenizationStatus.READY, progress=100)

        rows = await _federated_tokenizations(async_session, ("QUEUED", "PROCESSING"))

        assert rows == []


class TestFederatedActivationExtractions:
    """A running activation extraction must surface as an active operation."""

    ACTIVE = ("QUEUED", "LOADING", "EXTRACTING", "SAVING")

    async def test_reports_running_extraction(self, async_session):
        """THE CASING REGRESSION for extractions. Mirrors the live granite job."""
        model_id = await _make_model(async_session)
        dataset_id = await _make_dataset(async_session)
        ext_id = await _make_extraction(async_session, model_id, dataset_id, ExtractionStatus.EXTRACTING)

        rows = await _federated_activation_extractions(async_session, self.ACTIVE)

        assert len(rows) == 1, (
            "an EXTRACTING activation extraction must be federated; 0 rows means "
            "the status comparison missed the UPPERCASE enum labels"
        )
        row = rows[0]
        assert row["id"] == ext_id
        assert row["status"] == "running"
        assert row["task_type"] == "extraction"
        assert row["progress"] == pytest.approx(19.6)
        assert "1200/10000 samples" in row["entity_info"]["details"]

    @pytest.mark.parametrize("status", ["LOADING", "SAVING"])
    async def test_intermediate_statuses_count_as_running(self, async_session, status):
        """LOADING/SAVING are in-flight per cleanup_stuck_activations.py."""
        model_id = await _make_model(async_session)
        dataset_id = await _make_dataset(async_session)
        await _make_extraction(async_session, model_id, dataset_id, ExtractionStatus[status])

        rows = await _federated_activation_extractions(async_session, self.ACTIVE)

        assert len(rows) == 1
        assert rows[0]["status"] == "running"

    async def test_completed_extraction_is_not_active(self, async_session):
        model_id = await _make_model(async_session)
        dataset_id = await _make_dataset(async_session)
        await _make_extraction(async_session, model_id, dataset_id, ExtractionStatus.COMPLETED, progress=100)

        rows = await _federated_activation_extractions(async_session, self.ACTIVE)

        assert rows == []


class TestActiveEndpointWiring:
    """Reachability: the federators must actually be called by /active.

    A capability is not shipped until a test fails when its wiring is removed —
    deleting either ``.extend(await _federated_*(...))`` line in
    ``list_active_tasks`` must turn these red.
    """

    async def test_running_tokenization_appears_in_active_endpoint(self, async_session):
        model_id = await _make_model(async_session)
        dataset_id = await _make_dataset(async_session, "WiringCheck-DS")
        tok_id = await _make_tokenization(async_session, dataset_id, model_id, TokenizationStatus.PROCESSING)

        result = await list_active_tasks(db=async_session)

        ids = [t["id"] for t in result["data"]]
        assert tok_id in ids, (
            "tokenization federation is not wired into list_active_tasks"
        )

    async def test_running_extraction_appears_in_active_endpoint(self, async_session):
        model_id = await _make_model(async_session)
        dataset_id = await _make_dataset(async_session)
        ext_id = await _make_extraction(async_session, model_id, dataset_id, ExtractionStatus.EXTRACTING)

        result = await list_active_tasks(db=async_session)

        ids = [t["id"] for t in result["data"]]
        assert ext_id in ids, (
            "activation-extraction federation is not wired into list_active_tasks"
        )
