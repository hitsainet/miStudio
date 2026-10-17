"""The activation sources' resume contract (tracker item 4, cross-workstream contract).

Every source that feeds training exposes ``state_dict()`` / ``load_state_dict(state)``.
The dict holds only JSON-style values and CPU tensors (saved with torch.save, loaded
with weights_only=True), and after ``load_state_dict`` the source serves EXACTLY the
batches it would have served had training not stopped.

The fixtures encode each activation's origin — (source, row, position, layer) — so
"exactly" is checked token by token, in every read mode the buffer has, across
refills and source passes, and with masked rows whose reads take the fancy-index path.

MUTATION CONTROLS (2026-09-15, WS-LOOP; applied alone, this file and the
resume-equivalence file run, bytes restored, sha256 verified). All red:
  B23 state_dict saves the LIVE generators, not the served buffer's snapshot
        -> both resume tests in every mode, equivalence [rolling] (9 failed)
  B24 the position within the buffer not restored  -> same 9
  B25 the row-cycle generators not restored        -> restored-buffer test in every mode,
        equivalence [rolling] (5 failed)
  B26 the fixed-pool generator not restored        -> restored pool, equivalence [pool]
  B27 the permutation generator not restored       -> 7 failed
The fixture writes each extraction ONCE: its first version re-saved the .npy files a
live buffer had memory-mapped, which is a SIGBUS, not a test failure.
"""

import io

import numpy as np
import pytest
import torch

from src.services import activation_buffer as AB

SEQ = 8
KEYS = [(3, "residual"), (4, "residual")]
CPU = torch.device("cpu")
MODES = {
    "sync-1": {"prefetch": False, "read_threads": 1},
    "sync-4": {"prefetch": False, "read_threads": 4},
    "prefetch-1": {"prefetch": True, "read_threads": 1, "host_ram_tokens": 10**9},
    "prefetch-12": {"prefetch": True, "read_threads": 12, "host_ram_tokens": 10**9},
}


def _source(tmp_path, source, rows, holes_seed=None):
    files = {}
    r, p = np.meshgrid(np.arange(rows), np.arange(SEQ), indexing="ij")
    for layer, hook in KEYS:
        path = tmp_path / f"s{source}_layer_{layer}_{hook}.npy"
        # Written once. A second buffer over the same extraction must not rewrite
        # the file the first one still has memory-mapped (that is a SIGBUS).
        if not path.exists():
            arr = np.stack([np.full_like(r, source), r, p, np.full_like(r, layer)], axis=-1).astype(np.float16)
            np.save(path, arr)
        files[(layer, hook)] = path
    valid = None
    if holes_seed is not None:
        mask = np.random.default_rng(holes_seed).random((rows, SEQ)) > 0.3
        mask[:, 0] = True
        valid = np.flatnonzero(mask.reshape(-1))
    return AB.BufferSource(label=f"src{source}", files=files, num_rows=rows, seq_len=SEQ, valid_flat=valid)


@pytest.fixture
def built():
    buffers = []
    yield buffers
    for buf in buffers:
        buf.close()


def _make(built, tmp_path, mode, seed=11):
    sources = [_source(tmp_path, 0, 9), _source(tmp_path, 1, 7, holes_seed=3)]
    buf = AB.RollingActivationBuffer(
        sources, KEYS, [3 * SEQ, 2 * SEQ], seed=seed, storage_device=CPU, train_device=CPU, **MODES[mode]
    )
    built.append(buf)
    return buf


def _draw(buf, batches, size=13):
    """Origins of each layer's batches, and a check that the layers stay aligned."""
    out = []
    for _ in range(batches):
        batch = buf.next_batch(size)
        first, second = (batch[k] for k in KEYS)
        assert torch.equal(first[:, :3], second[:, :3]), "the layers were served different tokens"
        out.append([tuple(int(v) for v in row[:3]) for row in first.tolist()])
    return out


def _through_disk(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def _assert_plain(value, path="state"):
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str), f"{path} has a non-string key {key!r}"
            _assert_plain(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _assert_plain(item, f"{path}[{i}]")
    elif isinstance(value, torch.Tensor):
        assert value.device.type == "cpu", f"{path} is on {value.device}"
    else:
        assert value is None or isinstance(value, (str, int, float, bool)), f"{path} is a {type(value).__name__}"


@pytest.mark.parametrize("mode", list(MODES))
class TestTheRollingBufferResumesExactly:
    def test_a_restored_buffer_serves_the_batches_the_original_would_have(self, built, tmp_path, mode):
        original = _make(built, tmp_path, mode)
        _draw(original, 9)  # several refills, past a pass of the smaller source
        saved = _through_disk(original.state_dict())
        expected = _draw(original, 12)

        resumed = _make(built, tmp_path, mode)
        _draw(resumed, 2)  # a fresh process may have drawn anything before it restores
        resumed.load_state_dict(saved)

        assert _draw(resumed, 12) == expected
        assert resumed.refills == original.refills
        assert resumed.tokens_loaded == original.tokens_loaded
        assert resumed.tokens_dropped == original.tokens_dropped
        assert resumed.epochs_completed() == original.epochs_completed()

    def test_a_save_taken_after_the_next_buffer_was_prepared_skips_nothing(self, built, tmp_path, mode):
        """The background preparation draws the NEXT buffer from the generators as
        soon as one is served. Saving the generators as they are then would make
        a resumed run skip that buffer."""
        original = _make(built, tmp_path, mode)
        _draw(original, 5)
        if original._pending is not None:
            original._pending.exception(timeout=10)  # the next buffer is fully prepared
        saved = _through_disk(original.state_dict())
        expected = _draw(original, 10)

        resumed = _make(built, tmp_path, mode)
        resumed.load_state_dict(saved)
        assert _draw(resumed, 10) == expected

    def test_the_state_is_plain_data_and_cpu_tensors(self, built, tmp_path, mode):
        buf = _make(built, tmp_path, mode)
        _draw(buf, 3)
        _assert_plain(buf.state_dict())


class TestTheRollingBufferRefusesAnotherRun:
    def test_different_quotas_are_refused(self, built, tmp_path):
        original = _make(built, tmp_path, "sync-1")
        _draw(original, 2)
        state = original.state_dict()
        state["quotas"] = [2 * SEQ, 3 * SEQ]
        with pytest.raises(ValueError, match="quotas"):
            _make(built, tmp_path, "sync-1").load_state_dict(state)

    def test_a_source_with_other_rows_is_refused(self, built, tmp_path):
        original = _make(built, tmp_path, "sync-1")
        _draw(original, 2)
        state = original.state_dict()
        other = tmp_path / "other"
        other.mkdir()
        sources = [_source(other, 0, 10), _source(other, 1, 7, holes_seed=3)]
        buf = AB.RollingActivationBuffer(
            sources, KEYS, [3 * SEQ, 2 * SEQ], seed=11, storage_device=CPU, train_device=CPU, **MODES["sync-1"]
        )
        built.append(buf)
        with pytest.raises(ValueError, match="rows"):
            buf.load_state_dict(state)


class TestTheFixedPool:
    def _pool(self, n=50, seed=4):
        tensors = {key: torch.arange(n * 2, dtype=torch.float32).reshape(n, 2) + i for i, key in enumerate(KEYS)}
        return AB.FixedPoolSampler(tensors, KEYS, seed=seed)

    def test_a_restored_pool_serves_the_same_draws(self):
        original = self._pool()
        for _ in range(3):
            original.next_batch(16)
        saved = _through_disk(original.state_dict())
        _assert_plain(saved)
        expected = [original.next_batch(16)[KEYS[0]] for _ in range(4)]

        resumed = self._pool(seed=999)
        resumed.load_state_dict(saved)
        for want in expected:
            assert torch.equal(resumed.next_batch(16)[KEYS[0]], want)

    def test_the_global_rng_does_not_move_the_stream(self):
        a, b = self._pool(), self._pool()
        torch.manual_seed(0)
        first = a.next_batch(16)[KEYS[0]]
        torch.manual_seed(12345)
        torch.rand(100)
        assert torch.equal(b.next_batch(16)[KEYS[0]], first)

    def test_every_layer_gets_the_same_rows(self):
        batch = self._pool().next_batch(16)
        assert torch.equal(batch[KEYS[1]] - 1, batch[KEYS[0]])

    def test_another_pool_is_refused(self):
        state = self._pool(n=50).state_dict()
        with pytest.raises(ValueError, match="tokens"):
            self._pool(n=40).load_state_dict(state)
