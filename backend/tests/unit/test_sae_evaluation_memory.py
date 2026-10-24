"""The cross-entropy pass never holds a whole batch of float32 log-probabilities (review R1-C).

FOUND 2026-09-15 (review round 1, R1-C). ``evaluate_spliced_layers`` turned each
batch's logits into float32 log-probabilities, kept the untouched model's copy
while every substitution ran, and took ``F.kl_div`` over the full
``[batch, seq, vocab]`` tensor. Measured on CPU with the module's own helpers at
the default 4,096 tokens a batch: peak growth **5.0x** one float32
``[batch, seq-1, vocab]`` tensor — 5.0 GB for LFM2.5 (vocab 65,536), 20 GB for a
262,144-token vocabulary — against the re-run task's 3,072 MB reservation. The
evaluation then OOMs (recorded as failed) or squeezes a neighbouring job.

The arithmetic now runs a chunk of positions at a time, sized so one float32
``[positions, vocab]`` block stays under ``CE_CHUNK_BYTES``, and holds the base
model's LOGITS in the model's dtype instead of float32 log-probabilities.

MUTATION CONTROLS (2026-09-15; each applied alone, this file run, restored, sha256 verified):
  M1 positions_per_chunk returns every position (one chunk)  RED  test_the_whole_evaluation_uses_the_chunked_arithmetic
                                                                   (+ the `wide` fixture's own chunk-size assertion errors
                                                                   the two per-call guards)
  M2 KL arguments swapped (KL(other || base))                RED  test_chunked_equals_the_whole_tensor_arithmetic
  M3 base log-softmax taken on the OTHER logits              RED  test_chunked_equals_the_whole_tensor_arithmetic
  M4 chunks built from every position, not the valid ones    RED  test_chunked_equals_the_whole_tensor_arithmetic
  M5 ce_working_bytes drops the held base logits (1x not 2x) RED  test_the_reservation_counts_both_logit_tensors
"""

import pytest
import torch
import torch.nn.functional as F

from src.services import sae_evaluation as ev


def _reference(base_logits, logits, input_ids, attention_mask):
    """The first version's whole-tensor arithmetic, kept here as the definition."""
    labels = input_ids[:, 1:]
    valid = ev.prediction_mask(input_ids, attention_mask)
    base_lp = F.log_softmax(base_logits[:, :-1, :].float(), dim=-1)
    lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    nll = -lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    base_nll = -base_lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    kl = F.kl_div(lp, base_lp, log_target=True, reduction="none").sum(-1)
    return float(base_nll[valid].sum()), float(nll[valid].sum()), float(kl[valid].sum())


@pytest.fixture
def batch():
    torch.manual_seed(3)
    B, T, V = 3, 37, 211
    input_ids = torch.randint(0, V, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[1, 30:] = 0          # right padding
    attention_mask[2, :4] = 0           # left padding: a pad token's successor is real
    base_logits = torch.randn(B, T, V, dtype=torch.float16) * 4
    logits = base_logits.float() + torch.randn(B, T, V) * 2
    return input_ids, attention_mask, base_logits, logits.half()


@pytest.mark.parametrize("chunk_positions", [1, 5, 64, 10_000])
def test_chunked_equals_the_whole_tensor_arithmetic(batch, monkeypatch, chunk_positions):
    input_ids, attention_mask, base_logits, logits = batch
    monkeypatch.setattr(ev, "positions_per_chunk", lambda vocab, chunk_bytes=0: chunk_positions)
    labels = input_ids[:, 1:]
    valid = ev.prediction_mask(input_ids, attention_mask)
    chunks = ev._prediction_chunks(valid, base_logits.shape[-1])

    base = ev._nll_sum(base_logits[:, :-1, :], labels, chunks)
    nll, kl = ev._nll_and_kl_sums(base_logits[:, :-1, :], logits[:, :-1, :], labels, chunks)

    ref_base, ref_nll, ref_kl = _reference(base_logits, logits, input_ids, attention_mask)
    assert base == pytest.approx(ref_base, rel=1e-5)
    assert nll == pytest.approx(ref_nll, rel=1e-5)
    assert kl == pytest.approx(ref_kl, rel=1e-4)
    assert kl > 0 and nll != pytest.approx(base, rel=1e-3)  # the fixture separates the two


def _record(monkeypatch, name):
    seen = []
    original = getattr(F, name)

    def spy(input, *args, **kwargs):
        seen.append(input.numel())
        return original(input, *args, **kwargs)

    monkeypatch.setattr(ev.F, name, spy)
    return seen


@pytest.fixture
def wide(monkeypatch):
    """A vocabulary wide enough that a whole batch is many chunks."""
    torch.manual_seed(5)
    B, T, V = 2, 300, 4_000
    monkeypatch.setattr(ev, "CE_CHUNK_BYTES", 4 * V * 64)  # 64 positions a chunk
    input_ids = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    base = torch.randn(B, T, V, dtype=torch.float16)
    other = torch.randn(B, T, V, dtype=torch.float16)
    labels = input_ids[:, 1:]
    valid = ev.prediction_mask(input_ids, mask)
    per_chunk = ev.positions_per_chunk(V, ev.CE_CHUNK_BYTES)
    assert per_chunk == 64 and int(valid.sum()) > 8 * per_chunk
    return SimpleBatch(base[:, :-1], other[:, :-1], labels, valid, V, per_chunk)


class SimpleBatch:
    def __init__(self, base, other, labels, valid, vocab, per_chunk):
        self.base, self.other, self.labels, self.valid = base, other, labels, valid
        self.vocab, self.per_chunk = vocab, per_chunk


def test_no_log_softmax_sees_more_than_a_chunk(wide, monkeypatch):
    sizes = _record(monkeypatch, "log_softmax")
    chunks = ev._prediction_chunks(wide.valid, wide.vocab)
    ev._nll_and_kl_sums(wide.base, wide.other, wide.labels, chunks)
    ev._nll_sum(wide.base, wide.labels, chunks)
    assert sizes and max(sizes) <= wide.per_chunk * wide.vocab


def test_no_kl_sees_more_than_a_chunk(wide, monkeypatch):
    sizes = _record(monkeypatch, "kl_div")
    chunks = ev._prediction_chunks(wide.valid, wide.vocab)
    ev._nll_and_kl_sums(wide.base, wide.other, wide.labels, chunks)
    assert len(sizes) > 1 and max(sizes) <= wide.per_chunk * wide.vocab


def test_the_whole_evaluation_uses_the_chunked_arithmetic(monkeypatch):
    """Wiring: evaluate_spliced_layers itself never takes a whole-batch log-softmax."""
    from test_sae_evaluation import _llama  # a tiny real Llama

    model = _llama()
    vocab = model.config.vocab_size
    monkeypatch.setattr(ev, "CE_CHUNK_BYTES", 4 * vocab * 3)  # 3 positions a chunk
    sizes = _record(monkeypatch, "log_softmax")
    hidden = model.config.hidden_size

    class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, return_loss=False):
            return x, None

    ids = torch.randint(0, vocab, (2, 12))
    mask = torch.ones_like(ids)
    result = ev.evaluate_spliced_layers(
        model, {0: Identity()}, {0: model.model.layers[0]}, lambda: [(ids, mask)],
    )
    assert result["ce_base"] is not None
    assert sizes and max(sizes) <= 3 * vocab, "a log-softmax saw a whole batch"
    assert hidden > 0


def test_the_reservation_counts_both_logit_tensors():
    vocab, tokens = 65_536, 4_096
    working = ev.ce_working_bytes(vocab, tokens, logit_bytes=2)
    assert working >= 2 * tokens * vocab * 2 + 5 * min(ev.CE_CHUNK_BYTES, tokens * vocab * 4)
    # and it is far below the five whole-batch float32 copies it replaced
    assert working < 5 * tokens * vocab * 4
