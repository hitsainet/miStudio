"""Attention Q/K/V/O readouts per head (024 task 4.2, BR-015) — on a REAL saved checkpoint.

A tiny Llama is saved with safetensors and read back the way the route reads a
downloaded model: structure from a META instantiation, each projection tensor by tensor.
The planted weights make the expected answer exact, so the test checks the
orientation of every side (read rows vs write columns) rather than "something came back".
"""

from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.services import jlens_attention_readout as ar

D, HEADS, VOCAB = 16, 4, 32
HEAD_DIM = D // HEADS


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    config = LlamaConfig(hidden_size=D, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=HEADS, num_key_value_heads=HEADS,
                         vocab_size=VOCAB, max_position_embeddings=32)
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    attn = model.model.layers[1].self_attn
    with torch.no_grad():
        attn.q_proj.weight.zero_()
        attn.o_proj.weight.zero_()
        # Head 2 of Q READS residual dimension 5 only.
        attn.q_proj.weight[2 * HEAD_DIM, 5] = 3.0
        # Head 1 of O WRITES residual dimension 9 only.
        attn.o_proj.weight[9, 1 * HEAD_DIM] = 4.0
    path = tmp_path_factory.mktemp("tiny") / "snap"
    model.save_pretrained(path, safe_serialization=True)
    return config, Path(path)


def _decode(ids):
    return [f"t{i}" for i in ids]


def test_read_and_write_sides_are_oriented_correctly(checkpoint):
    config, snap = checkpoint
    W_U = torch.eye(VOCAB, D)  # token i <-> residual dimension i
    out = ar.readout(config, snap, W_U, _decode, layer=1, heads=[1, 2],
                     matrices=["q_proj", "o_proj"], components=1, top_k=1)
    q = {r["head"]: r for r in out["matrices"]["q_proj"]["heads"]}
    o = {r["head"]: r for r in out["matrices"]["o_proj"]["heads"]}
    assert out["matrices"]["q_proj"]["side"] == "read"
    assert out["matrices"]["o_proj"]["side"] == "write"
    assert out["matrices"]["q_proj"]["n_heads"] == HEADS
    # Unsigned: the planted dimension is at ONE of the two ends.
    assert "t5" in (q[2]["toward_tokens"] + q[2]["away_tokens"])
    assert q[2]["singular_value"] == pytest.approx(3.0, rel=1e-5)
    assert "t9" in (o[1]["toward_tokens"] + o[1]["away_tokens"])
    assert o[1]["singular_value"] == pytest.approx(4.0, rel=1e-5)
    assert q[1]["singular_value"] == pytest.approx(0.0, abs=1e-6), "head 1 of Q reads nothing"


def test_every_projection_is_found_structurally(checkpoint):
    config, snap = checkpoint
    model = ar._meta_model(config)
    found = ar.attention_projections(model, 1, D)
    assert {k: v["side"] for k, v in found.items()} == {
        "q_proj": "read", "k_proj": "read", "v_proj": "read", "o_proj": "write"}
    assert found["o_proj"]["key"] == "model.layers.1.self_attn.o_proj.weight"


def test_refusals(checkpoint):
    config, snap = checkpoint
    W_U = torch.eye(VOCAB, D)
    with pytest.raises(ValueError, match="no projection named"):
        ar.readout(config, snap, W_U, _decode, layer=1, matrices=["c_attn"])
    with pytest.raises(ValueError, match="heads"):
        ar.readout(config, snap, W_U, _decode, layer=1, heads=[7], matrices=["q_proj"])
    with pytest.raises(ValueError, match="outside range"):
        ar.readout(config, snap, W_U, _decode, layer=5)


def test_the_route_is_served():
    from src.main import app

    assert "post" in app.openapi()["paths"]["/api/v1/jlens/attention-readout"]


@pytest.mark.asyncio
async def test_the_route_reads_the_checkpoint_it_resolves(checkpoint, client, async_session, monkeypatch):
    from src.models.model import Model

    config, snap = checkpoint
    async_session.add(Model(id="m_att", name="tiny", architecture="llama", params_count=1))
    await async_session.commit()

    class _Tok:
        def decode(self, ids):
            return "".join(f"t{i}" for i in ids)

    monkeypatch.setattr("src.services.jlens_annotation_store.unembedding_and_tokenizer",
                        lambda record: (torch.eye(VOCAB, D), _Tok()))
    monkeypatch.setattr("src.services.jlens_model_registry.locate_weights",
                        lambda record: ("org/tiny", snap))
    monkeypatch.setattr("src.services.analysis_service.resolve_snapshot_dir",
                        lambda resolved, repo: snap)
    resp = await client.post("/api/v1/jlens/attention-readout", json={
        "model_id": "m_att", "layer": 1, "heads": [1], "matrices": ["o_proj"], "top_k": 1})
    assert resp.status_code == 200, resp.text
    (row,) = resp.json()["matrices"]["o_proj"]["heads"]
    assert "t9" in row["toward_tokens"] + row["away_tokens"]
    bad = await client.post("/api/v1/jlens/attention-readout",
                            json={"model_id": "m_att", "layer": 1, "matrices": ["nope"]})
    assert bad.status_code == 400
