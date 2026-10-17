"""OSD-8 — deleting a model must not silently destroy its tokenizations.

`dataset_tokenizations.model_id` is `ON DELETE CASCADE`, so deleting a model
removes every tokenization row naming it **while the Arrow directories stay on
disk**, orphaned. It happened on 2026-09-12: `m_88d55564` was deleted to change
the model's precision and took all five tokenization rows for 81 GB of data with
it. They were rebuilt by hand only because the data itself survived — there is no
database backup newer than 2025-12 and WAL archiving is off.

The cascade is correct at the schema level. What was missing is consent.
"""
import ast
import inspect

import pytest

from src.api.v1.endpoints import models as models_endpoint
from src.services.model_service import ModelService


class TestTheSchemaStillCascades:
    """The guard exists BECAUSE of the cascade, so pin the cascade too.

    If someone changes the FK to RESTRICT, the guard becomes redundant and this
    test says so out loud rather than leaving two mechanisms fighting.
    """

    def test_the_tokenization_fk_is_still_on_delete_cascade(self):
        from src.models.dataset_tokenization import DatasetTokenization
        fk = next(
            iter(DatasetTokenization.__table__.c.model_id.foreign_keys)
        )
        assert fk.ondelete == "CASCADE", (
            "the model_id FK no longer cascades — re-read OSD-8 before deleting "
            "the consent guard, because the reason for it has changed"
        )


class TestARefusalNamesWhatWouldBeLost:

    @pytest.mark.asyncio
    async def test_a_model_with_tokenizations_is_refused_by_default(self, monkeypatch):
        calls = _FakeDb(tokenizations=["tok_a", "tok_b"], trainings=[])
        monkeypatch.setattr(ModelService, "get_model", _fake_get_model)
        with pytest.raises(ValueError) as exc:
            await ModelService.delete_model(calls, "m_abc")
        message = str(exc.value)
        assert "2 tokenization" in message
        assert "tok_a" in message and "tok_b" in message, (
            "a refusal that does not name the rows leaves the operator guessing"
        )
        assert "Re-download" in message, "the refusal must name the safe alternative"
        assert calls.deleted == [], "nothing may be deleted on the refusing path"

    @pytest.mark.asyncio
    async def test_the_refusal_does_not_fire_without_tokenizations(self, monkeypatch):
        db = _FakeDb(tokenizations=[], trainings=[])
        monkeypatch.setattr(ModelService, "get_model", _fake_get_model)
        result = await ModelService.delete_model(db, "m_abc")
        assert result["deleted"] is True
        assert result["deleted_tokenizations"] == 0

    @pytest.mark.asyncio
    async def test_the_cascade_proceeds_when_asked_for_explicitly(self, monkeypatch):
        db = _FakeDb(tokenizations=["tok_a"], trainings=[])
        monkeypatch.setattr(ModelService, "get_model", _fake_get_model)
        result = await ModelService.delete_model(db, "m_abc", cascade_tokenizations=True)
        assert result["deleted"] is True
        assert result["deleted_tokenizations"] == 1, (
            "the count is how the caller learns what the cascade took"
        )

    @pytest.mark.asyncio
    async def test_many_tokenizations_are_summarised_not_dumped(self, monkeypatch):
        db = _FakeDb(tokenizations=[f"tok_{i}" for i in range(9)], trainings=[])
        monkeypatch.setattr(ModelService, "get_model", _fake_get_model)
        with pytest.raises(ValueError) as exc:
            await ModelService.delete_model(db, "m_abc")
        assert "and 4 more" in str(exc.value)


class TestTheChoiceIsReachable:
    """A guard only the service can express is a guard no operator can use."""

    def test_the_endpoint_accepts_the_flag(self):
        signature = inspect.signature(models_endpoint.delete_model)
        assert "cascade_tokenizations" in signature.parameters

    def test_the_endpoint_passes_it_through(self):
        """Assert the CALL carries the keyword — not that the name appears."""
        tree = ast.parse(inspect.getsource(models_endpoint.delete_model))
        delete_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "delete_model"
        ]
        assert delete_calls, "the endpoint no longer calls ModelService.delete_model"
        assert any(
            kw.arg == "cascade_tokenizations" for call in delete_calls for kw in call.keywords
        ), (
            "the endpoint must forward cascade_tokenizations, or the query "
            "parameter is decorative and every delete takes the default"
        )

    def test_the_default_is_the_safe_reading(self):
        assert inspect.signature(
            ModelService.delete_model
        ).parameters["cascade_tokenizations"].default is False


# ── fakes ────────────────────────────────────────────────────────────────────
# Deliberately not a mock of `db.execute`: the guard's correctness depends on
# WHICH query it runs, so the fake answers by the entity being selected.

class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeDb:
    def __init__(self, tokenizations, trainings):
        self.tokenizations = tokenizations
        self.trainings = trainings
        self.deleted = []
        self.committed = False

    async def execute(self, statement):
        entity = str(statement).lower()
        if "dataset_tokenizations" in entity:
            return _Result(list(self.tokenizations))
        if "trainings" in entity:
            return _Result(list(self.trainings))
        return _Result([])           # activation_extractions

    async def delete(self, obj):
        self.deleted.append(obj)

    async def commit(self):
        self.committed = True


class _Model:
    id = "m_abc"
    file_path = "/data/models/m_abc"
    quantized_path = None


async def _fake_get_model(db, model_id):
    return _Model()
