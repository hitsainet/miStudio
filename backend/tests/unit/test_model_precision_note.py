"""F-7 (2026-10-21): the model listing puts `quantization` (storage format) beside the dtype the
model actually runs in, so `FP16` listed and `bfloat16` recorded by a run do not read as a
contradiction.

Control (review record `0xcc/reviews/2026-10-21_mcp-gaps.md`): drop `precision_note` from
ModelResponse → test_the_listing_route_serves_it red.
"""
from datetime import datetime, timezone

import pytest

from src.schemas.model import ModelResponse

NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)


def _row(quantization, native):
    return ModelResponse(id="m_x", name="m", architecture="llama", params_count=1,
                         quantization=quantization, status="ready", native_dtype=native,
                         created_at=NOW, updated_at=NOW)


def test_fp16_beside_a_bfloat16_checkpoint():
    row = _row("FP16", "bfloat16")
    assert row.loads_at == "bfloat16"
    note = row.precision_note
    assert "FP16 = unquantized 16-bit weights" in note
    assert "Runs in bfloat16" in note and "config.json records bfloat16" in note
    assert "model_dtype bfloat16" in note


def test_an_unread_checkpoint_says_it_is_a_default():
    note = _row("FP16", None).precision_note
    assert "rule's default" in note and "Runs in bfloat16" in note


@pytest.mark.parametrize("quant,native,runs", [("FP32", "bfloat16", "float32"),
                                               ("Q4", "float16", "float16")])
def test_other_formats(quant, native, runs):
    row = _row(quant, native)
    assert row.loads_at == runs and f"Runs in {runs}" in row.precision_note
    assert row.precision_note.startswith(quant)


async def test_the_listing_route_serves_it(client, async_session):
    from src.models.model import Model, ModelStatus, QuantizationFormat

    async_session.add(Model(id="m_prec01", name="p", status=ModelStatus.READY, file_path="/x",
                            architecture="llama", params_count=1, native_dtype="bfloat16",
                            quantization=QuantizationFormat.FP16))
    await async_session.commit()
    body = (await client.get("/api/v1/models")).json()
    row = next(r for r in body["data"] if r["id"] == "m_prec01")
    assert row["quantization"] == "FP16" and row["loads_at"] == "bfloat16"
    assert "Runs in bfloat16" in row["precision_note"]


def test_the_mcp_listing_explains_the_two_fields():
    import asyncio

    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import models

    mcp = MCPServer("t")
    models.register(mcp, object(), MCPSettings(auth_token="x" * 32, _env_file=None))
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("list_models", "get_model"):
        assert "loads_at" in tools[name].description and "precision_note" in tools[name].description
