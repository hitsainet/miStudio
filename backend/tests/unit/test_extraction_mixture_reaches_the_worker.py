"""A multi-corpus extraction request must actually reach the worker.

WHY THIS FILE EXISTS. `extraction_mixture` and the worker's mixture planning are
unit-tested in isolation, and every one of those tests passes by importing the
module directly. That is precisely the shape of this repo's signature failure:
16 MCP tools fully implemented, unit-tested and documented while never
registered with the server, so the suite was green and the docs said done while
no caller could reach the feature.

So this file asserts REACHABILITY through the real endpoint: the `dataset_ids`
and `dataset_weights` a caller sends must arrive in the config the Celery task
is dispatched with. It asserts the PAYLOAD and the CALL COUNT, because a test
that only checks "was called" passes against a call sending the wrong arguments.

⚠ DATASET IDS HERE ARE UUIDS ON PURPOSE. `Dataset.id` is `UUID(as_uuid=True)`,
so the endpoint refuses an id that is not UUID-shaped as not-found WITHOUT
querying (a non-UUID cannot even be bound to the column). Fixtures using ids
like "ds_code" would therefore exercise the refusal path while appearing to
test the happy one — a fixture agreeing with the wrong branch by construction.

MUTATION CONTROLS (apply one at a time, restore by bytes, verify sha256):
  X1 delete `config_dict["dataset_ids"] = dataset_ids` in saes.py
       -> test_both_corpora_reach_the_dispatched_config FAILS
  X2 make the config read `config_dict["dataset_id"] = dataset_id` (the raw
     query param) instead of `dataset_ids[0]`
       -> test_the_legacy_key_is_the_first_corpus FAILS
  X3 drop `dataset_weights` from ExtractionConfigRequest
       -> test_the_weights_travel_with_the_ids FAILS
  X4 delete the one-to-one length check
       -> test_mismatched_weights_are_refused FAILS
  X5 move the dataset validation inside the `try`
       -> test_an_unknown_corpus_is_a_400_and_no_job_is_created FAILS (500)
  X6 delete the `_cannot_be_a_dataset_id` guard
       -> test_a_malformed_corpus_id_is_refused_not_crashed FAILS (DBAPIError)
"""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

#: Real dataset ids are UUIDs; see the module docstring.
DS_CODE = "11111111-1111-4111-8111-111111111111"
DS_WEB = "22222222-2222-4222-8222-222222222222"
DS_CHAT = "33333333-3333-4333-8333-333333333333"
DS_ONLY = "44444444-4444-4444-8444-444444444444"
DS_REAL = "55555555-5555-4555-8555-555555555555"
DS_MISSING = "66666666-6666-4666-8666-666666666666"


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _AsyncDB:
    """An AsyncSession that answers `execute` BY QUERIED ENTITY, not by position.

    A positional queue would couple this file to how many queries the endpoint
    and service happen to make, and in what order — so adding one query
    elsewhere silently shifts every answer and the failure surfaces as a
    baffling AttributeError three frames away. That is a fixture agreeing with
    one particular call order rather than with the behaviour under test, and it
    is the exact shape this repo keeps getting caught by. Dispatching on the
    entity makes the fake indifferent to ordering.
    """

    def __init__(self, datasets=None, sae=None):
        #: Answers for `select(Dataset)`, consumed in order — so a test can say
        #: "the second corpus does not exist" by passing None in that slot.
        self.datasets = list(datasets) if datasets is not None else []
        self.sae = sae
        self.added = []

    @staticmethod
    def _entity_of(statement):
        try:
            return statement.column_descriptions[0]["entity"].__name__
        except Exception:  # pragma: no cover - defensive; any odd statement
            return None

    async def execute(self, statement):
        if self._entity_of(statement) == "Dataset":
            return _Result(self.datasets.pop(0) if self.datasets else _ready())
        return _Result(self.sae)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def refresh(self, obj):
        pass


@pytest.fixture
def endpoint_harness(monkeypatch):
    from src.api.v1.endpoints import saes as endpoint
    from src.models.external_sae import SAEStatus
    from src.services.extraction_service import ExtractionService
    from src.workers.extraction_tasks import extract_features_from_sae_task

    sae = SimpleNamespace(
        id="sae_a",
        status=SAEStatus.READY.value,
        local_path="/saes/a",
        name="sae a",
        model_id="m_1",
        layer=3,
        hook_type=None,
    )

    dispatch = SimpleNamespace(calls=[])

    def apply_async(**kwargs):
        dispatch.calls.append(kwargs)
        return SimpleNamespace(id="celery-1")

    async def get_sae(_db, sae_id):
        return SimpleNamespace(**{**vars(sae), "id": sae_id})

    async def no_active_extraction(self, training_id=None, sae_id=None):
        return None

    monkeypatch.setattr(endpoint.SAEManagerService, "get_sae", staticmethod(get_sae))
    monkeypatch.setattr(ExtractionService, "_check_active_extraction", no_active_extraction)
    monkeypatch.setattr(extract_features_from_sae_task, "apply_async", apply_async)

    return SimpleNamespace(endpoint=endpoint, sae=sae, dispatch=dispatch)


def _ready(name="ds"):
    return SimpleNamespace(status="ready", name=name)


def _dispatched_config(harness):
    assert len(harness.dispatch.calls) == 1, (
        f"expected exactly one dispatch, got {len(harness.dispatch.calls)}"
    )
    _sae_id, config = harness.dispatch.calls[0]["args"]
    return config


def _start(harness, config, dataset_id=None, datasets=None):
    """Drive the real endpoint. `datasets` answers the corpus lookups in order;
    omit it and every corpus resolves to a ready one."""
    from src.api.v1.endpoints import saes as endpoint

    db = _AsyncDB(datasets=datasets, sae=harness.sae)
    return asyncio.run(
        endpoint.start_sae_extraction("sae_a", config, dataset_id=dataset_id, db=db)
    ), db


class TestAMixtureReachesTheWorker:
    def test_both_corpora_reach_the_dispatched_config(self, endpoint_harness):
        from src.schemas.extraction import ExtractionConfigRequest

        _start(endpoint_harness, ExtractionConfigRequest(dataset_ids=[DS_CODE, DS_WEB]))

        config = _dispatched_config(endpoint_harness)
        assert config["dataset_ids"] == [DS_CODE, DS_WEB], (
            "the worker plans the mixture from dataset_ids; if it does not "
            "arrive, every multi-corpus request silently reads one corpus"
        )

    def test_the_weights_travel_with_the_ids(self, endpoint_harness):
        from src.schemas.extraction import ExtractionConfigRequest

        _start(
            endpoint_harness,
            ExtractionConfigRequest(
                dataset_ids=[DS_CODE, DS_WEB], dataset_weights=[0.7, 0.3]
            ),
        )

        config = _dispatched_config(endpoint_harness)
        # Positional, and in the SAME order as the ids: a reordered pair is not
        # an error, it is a different mixture reported truthfully.
        assert config["dataset_ids"] == [DS_CODE, DS_WEB]
        assert config["dataset_weights"] == [0.7, 0.3]

    def test_the_legacy_key_is_the_first_corpus(self, endpoint_harness):
        from src.schemas.extraction import ExtractionConfigRequest

        _start(endpoint_harness, ExtractionConfigRequest(dataset_ids=[DS_CODE, DS_WEB]))

        config = _dispatched_config(endpoint_harness)
        assert config["dataset_id"] == DS_CODE, (
            "legacy readers (list_extractions, the dataset_name lookup) still "
            "read dataset_id and must see a real corpus, not None"
        )

    def test_a_single_dataset_id_query_param_still_works_unchanged(self, endpoint_harness):
        """The whole backward-compatibility guarantee, in one test."""
        from src.schemas.extraction import ExtractionConfigRequest

        _start(endpoint_harness, ExtractionConfigRequest(), dataset_id=DS_ONLY)

        config = _dispatched_config(endpoint_harness)
        assert config["dataset_id"] == DS_ONLY
        assert config["dataset_ids"] == [DS_ONLY]


class TestBadMixturesAreRefusedBeforeAJobExists:
    def test_mismatched_weights_are_refused(self, endpoint_harness):
        from src.schemas.extraction import ExtractionConfigRequest

        with pytest.raises(HTTPException) as exc:
            _start(
                endpoint_harness,
                ExtractionConfigRequest(
                    dataset_ids=[DS_CODE, DS_WEB, DS_CHAT], dataset_weights=[1.0, 2.0]
                ),
            )

        assert exc.value.status_code == 400
        assert "one-to-one" in str(exc.value.detail)
        assert endpoint_harness.dispatch.calls == [], "a job was dispatched anyway"

    def test_duplicate_ids_are_refused(self, endpoint_harness):
        from src.schemas.extraction import ExtractionConfigRequest

        with pytest.raises(HTTPException) as exc:
            _start(endpoint_harness, ExtractionConfigRequest(dataset_ids=[DS_CODE, DS_CODE]))

        assert exc.value.status_code == 400
        assert endpoint_harness.dispatch.calls == []

    def test_no_dataset_at_all_is_refused(self, endpoint_harness):
        from src.schemas.extraction import ExtractionConfigRequest

        with pytest.raises(HTTPException) as exc:
            _start(endpoint_harness, ExtractionConfigRequest(), dataset_id=None)

        assert exc.value.status_code == 400
        assert endpoint_harness.dispatch.calls == []

    def test_an_unknown_corpus_is_a_400_and_no_job_is_created(self, endpoint_harness):
        """A 400, not a 500 — which is why validation lives outside the `try`.

        400 rather than 404 because that is what an unknown dataset has always
        returned here, and `test_sae_hook_recorded_at_import` asserts the
        message "Dataset {id} not found" verbatim.
        """
        from src.schemas.extraction import ExtractionConfigRequest

        with pytest.raises(HTTPException) as exc:
            _start(
                endpoint_harness,
                ExtractionConfigRequest(dataset_ids=[DS_REAL, DS_MISSING]),
                datasets=[_ready(), None],
            )

        assert exc.value.status_code == 400
        assert f"Dataset {DS_MISSING} not found" in str(exc.value.detail)
        assert endpoint_harness.dispatch.calls == []

    def test_a_malformed_corpus_id_is_refused_not_crashed(self, endpoint_harness):
        """A non-UUID id cannot be bound to `Dataset.id`, and asyncpg raises
        while ENCODING the parameter — before any query runs. That surfaced as
        a bare DBAPIError escaping the endpoint as a 500. It must read as an
        ordinary unknown dataset instead."""
        from src.schemas.extraction import ExtractionConfigRequest

        with pytest.raises(HTTPException) as exc:
            _start(endpoint_harness, ExtractionConfigRequest(), dataset_id="ds_none")

        assert exc.value.status_code == 400
        assert "Dataset ds_none not found" in str(exc.value.detail)
        assert endpoint_harness.dispatch.calls == []
