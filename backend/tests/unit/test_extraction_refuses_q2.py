"""SAE extraction refuses a Q2 base model (operator decision 2026-10-04).

miStudio loads a Q2 row as bitsandbytes fp4 while miLLM serves it unquantized, so activations
extracted from it — and every SAE trained on them, and every feature read through one — would
describe a model nothing serves. Steering, calibration and the circuit paths already refuse by
`served_quantization`; this extends the same one rule to both extraction paths:

* activation extraction (feeds SAE training): `ActivationService._load_model`, plus a 422 at the
  start and retry endpoints;
* SAE feature extraction: the worker after it resolves the model row, plus a 422 at single start
  and a skip-with-reason in a batch.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.models.model import QuantizationFormat


def _calls(func, name: str) -> list:
    tree = ast.parse(_dedent(func))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) == name or getattr(node.func, "attr", None) == name)
    ]


def _dedent(func) -> str:
    import textwrap

    return textwrap.dedent(inspect.getsource(func))


class TestActivationExtraction:
    def test_the_load_refuses_q2_before_touching_the_snapshot(self, monkeypatch):
        from src.services import activation_service
        from src.services.activation_service import ActivationExtractionError, ActivationService

        touched = []
        monkeypatch.setattr(activation_service, "resolve_model_snapshot", lambda p: touched.append(p) or p)
        service = ActivationService.__new__(ActivationService)
        with pytest.raises(ActivationExtractionError, match="Q2"):
            service._load_model("/data/models/x", QuantizationFormat.Q2, SimpleNamespace())
        assert touched == [], "the snapshot was resolved for a model that must never load"

    @pytest.mark.parametrize("served", ["FP32", "FP16", "Q8", "Q4"])
    def test_served_formats_get_past_the_gate(self, monkeypatch, served):
        from src.services import activation_service
        from src.services.activation_service import ActivationService

        class Reached(Exception):
            pass

        def snapshot(path):
            raise Reached

        monkeypatch.setattr(activation_service, "resolve_model_snapshot", snapshot)
        with pytest.raises(Reached):
            ActivationService.__new__(ActivationService)._load_model(
                "/data/models/x", QuantizationFormat(served), SimpleNamespace()
            )

    def test_the_endpoint_check_is_a_422_for_q2_only(self):
        from src.api.v1.endpoints.models import _refuse_unserved_extraction_model

        with pytest.raises(HTTPException) as exc:
            _refuse_unserved_extraction_model(SimpleNamespace(quantization=QuantizationFormat.Q2, id="m_x"))
        assert exc.value.status_code == 422 and "Q2" in exc.value.detail
        for served in ("FP32", "FP16", "Q8", "Q4"):
            assert _refuse_unserved_extraction_model(SimpleNamespace(quantization=served, id="m_x")) is None

    @pytest.mark.parametrize("endpoint", ["extract_model_activations", "retry_extraction"])
    def test_both_endpoints_call_the_check(self, endpoint):
        from src.api.v1.endpoints import models

        assert len(_calls(getattr(models, endpoint), "_refuse_unserved_extraction_model")) == 1


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _Db:
    def __init__(self, model):
        self.model = model
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(self.model)


class TestSaeFeatureExtraction:
    @pytest.mark.asyncio
    async def test_a_q2_base_model_has_a_reason(self):
        from src.services.extraction_service import ExtractionService

        service = ExtractionService(_Db(SimpleNamespace(id="m_q2", quantization=QuantizationFormat.Q2)))
        reason = await service._unserved_model_reason(SimpleNamespace(model_id="m_q2"))
        assert reason and "Q2" in reason and "m_q2" in reason

    @pytest.mark.asyncio
    async def test_a_served_base_model_and_an_unlinked_sae_have_none(self):
        from src.services.extraction_service import ExtractionService

        service = ExtractionService(_Db(SimpleNamespace(id="m_q4", quantization="Q4")))
        assert await service._unserved_model_reason(SimpleNamespace(model_id="m_q4")) is None
        db = _Db(None)
        assert await ExtractionService(db)._unserved_model_reason(SimpleNamespace(model_id=None)) is None
        assert db.statements == [], "an SAE with no model id needs no lookup"

    def test_single_start_raises_and_batch_skips(self):
        from src.services.extraction_service import ExtractionService

        assert len(_calls(ExtractionService.start_extraction_for_sae, "_unserved_model_reason")) == 1
        assert len(_calls(ExtractionService.start_batch_extraction_for_saes, "_unserved_model_reason")) == 1

    def test_the_worker_refuses_before_loading(self):
        """The authoritative gate: a job queued before the rule, or a model resolved by name."""
        from src.services.extraction_service import ExtractionService

        source = _dedent(ExtractionService.extract_features_for_sae)
        assert len(_calls(ExtractionService.extract_features_for_sae, "refuse_unserved_quantization")) == 1
        assert source.index("refuse_unserved_quantization(") < source.index("load_model_from_hf("), (
            "the refusal must come before the model loads"
        )

    def test_the_endpoint_answers_a_q2_refusal_with_422(self):
        from src.api.v1.endpoints import saes

        tree = ast.parse(inspect.getsource(saes))
        handlers = [
            h for h in ast.walk(tree)
            if isinstance(h, ast.ExceptHandler) and getattr(h.type, "id", None) == "UnservedQuantization"
        ]
        assert len(handlers) == 1
        raised = [n for n in ast.walk(handlers[0]) if isinstance(n, ast.Raise)]
        assert raised and "422" in ast.unparse(raised[0])
