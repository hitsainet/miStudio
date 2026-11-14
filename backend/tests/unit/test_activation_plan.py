"""The resume decision (services/activation_plan.py) and the sources' re-planned continuation.

Review round 1 of the SAE training remediation, R1D-1 and R1D-2, 2026-09-15. The
task-level tests are in test_resume_storage_plan.py, and the mutation-control table
for the whole fix is in that file's docstring. Here:

* the fits rule and the reuse / re-plan / not-recorded choice, on plain values;
* the plan and the resume history survive training_state.pt;
* ``load_state_dict_after_replan`` on the rolling buffer and on the on-the-fly
  source: under the SAVED quotas it serves exactly the buffer an uninterrupted
  source serves after the interrupted one (so it consumes exactly that buffer's
  draws from the generators), and under OTHER quotas no token repeats and no token
  of the interrupted buffer is served again in the pass;
* ``restore_source_position`` picks exact loading, the continuation, a restart or
  a refusal.

Every activation in the fixtures encodes its origin, so "no repeat" is checked token
by token.
"""

import io

import numpy as np
import pytest
import torch

from src.services import activation_buffer as AB
from src.services import activation_plan as AP
from src.services import model_activation_source as MAS
from src.services.checkpoint_service import build_training_state, restore_training_state

CPU = torch.device("cpu")


def _through_disk(state):
    blob = io.BytesIO()
    torch.save(state, blob)
    blob.seek(0)
    return torch.load(blob, weights_only=True)


# ── the decision ────────────────────────────────────────────────────────────


def _plan(mode="gpu_rolling", tokens=1_000, quotas=(600, 400), source_tokens=(5_000, 3_000), path="cached"):
    return AP.build_plan(
        path=path, mode=mode, buffer_tokens=tokens, quotas=quotas, source_tokens=source_tokens,
        gpu_capacity_tokens=tokens, ram_capacity_tokens=10**6, min_useful_tokens=500,
    )


def _choose(saved, *, gpu, ram, fresh=("gpu_rolling", 900), resuming=True, source_tokens=(5_000, 3_000),
            path="cached"):
    return AP.choose_storage_plan(
        saved_plan=saved, resuming=resuming, path=path, source_tokens=list(source_tokens), fresh=fresh,
        gpu_capacity_tokens=gpu, ram_capacity_tokens=ram,
    )


class TestTheFitsRule:
    @pytest.mark.parametrize(
        "mode,gpu,ram,fits",
        [
            ("gpu_rolling", 1_000, 0, True),         # exactly the saved buffer
            ("gpu_rolling", 999, 10**9, False),      # a token short on the card; RAM does not count
            ("gpu_all", 1_000, 0, True),
            ("cpu_rolling", 0, 1_000, True),
            ("cpu_rolling", 10**9, 999, False),      # the card does not count for a RAM plan
            ("cpu_all", 0, 999, False),
        ],
    )
    def test_a_plan_fits_only_where_it_put_its_buffer(self, mode, gpu, ram, fits):
        assert AP.plan_fits(_plan(mode=mode), gpu_capacity_tokens=gpu, ram_capacity_tokens=ram) is fits


class TestTheChoice:
    def test_a_fresh_run_takes_the_planner(self):
        choice = _choose(_plan(), gpu=10, ram=10, resuming=False, fresh=("cpu_all", 8_000))
        assert (choice.mode, choice.buffer_tokens, choice.quotas, choice.outcome) == ("cpu_all", 8_000, None, AP.FRESH)

    def test_a_state_with_no_plan_takes_the_planner_and_says_so(self):
        choice = _choose(None, gpu=10**9, ram=10**9, fresh=("gpu_rolling", 1_234))
        assert (choice.mode, choice.buffer_tokens, choice.quotas, choice.outcome) == (
            "gpu_rolling", 1_234, None, AP.NOT_RECORDED,
        )

    def test_a_plan_that_fits_is_reused_even_when_a_fresh_plan_would_be_larger(self):
        choice = _choose(_plan(), gpu=1_500, ram=0, fresh=("gpu_rolling", 1_500))
        assert (choice.mode, choice.buffer_tokens, choice.quotas, choice.outcome) == (
            "gpu_rolling", 1_000, [600, 400], AP.REUSED,
        )

    def test_a_plan_that_no_longer_fits_is_replanned(self):
        choice = _choose(_plan(), gpu=900, ram=0, fresh=("gpu_rolling", 900))
        assert (choice.mode, choice.buffer_tokens, choice.quotas, choice.outcome) == (
            "gpu_rolling", 900, None, AP.REPLANNED,
        )

    def test_a_cycled_pool_keeps_cycling_when_the_replan_would_load_it_whole(self):
        """A fixed pool samples with replacement; a run mid-pass must not switch to it."""
        choice = _choose(_plan(), gpu=0, ram=10**9, fresh=("cpu_all", 8_000))
        assert (choice.mode, choice.buffer_tokens, choice.outcome) == ("cpu_rolling", 8_000, AP.REPLANNED)

    def test_a_whole_pool_moved_off_the_card_stays_a_whole_pool(self):
        choice = _choose(_plan(mode="gpu_all", tokens=8_000, quotas=None), gpu=0, ram=10**9, fresh=("cpu_all", 8_000))
        assert (choice.mode, choice.buffer_tokens, choice.outcome) == ("cpu_all", 8_000, AP.REPLANNED)

    def test_a_plan_for_the_other_path_is_refused(self):
        with pytest.raises(ValueError, match="on_the_fly"):
            _choose(_plan(path="on_the_fly"), gpu=10**9, ram=10**9)

    def test_a_plan_over_other_data_is_refused_with_what_to_do(self):
        with pytest.raises(ValueError, match="changed after the checkpoint"):
            _choose(_plan(), gpu=10**9, ram=10**9, source_tokens=(5_000, 2_999))


class TestRestoredStateIsCreditedBackToTheReading:
    """Reviewer B: a resumed run measures free memory after its optimizer state is on the card."""

    @staticmethod
    def _stepped():
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        return {"k": model}, {"k": optimizer}

    def test_the_optimizer_state_and_gradients_on_a_device_are_counted(self):
        models, optimizers = self._stepped()
        state = sum(
            t.numel() * t.element_size()
            for entry in optimizers["k"].state.values() for t in entry.values() if torch.is_tensor(t)
        )
        grads = sum(p.grad.numel() * p.grad.element_size() for p in models["k"].parameters())
        weights = sum(p.numel() * p.element_size() for p in models["k"].parameters())
        # Adam's per-parameter step counters put moments + gradients a few bytes over the
        # pending allowance (3 x weights), so the cap applies with gradients present.
        assert AP._tensor_bytes_on(optimizers, models, "cpu") == min(state + grads, 3 * weights)
        models["k"].zero_grad(set_to_none=True)
        assert state < 3 * weights, "precondition: the moments alone are under the cap"
        assert AP._tensor_bytes_on(optimizers, models, "cpu") == state
        assert AP._tensor_bytes_on(optimizers, models, "meta") == 0, "state on another device was counted"

    def test_it_never_credits_more_than_the_pending_state_of_the_weights_there(self):
        models, optimizers = self._stepped()
        param = next(iter(models["k"].parameters()))
        optimizers["k"].state[param]["stray"] = torch.zeros(10_000)
        weights = sum(p.numel() * p.element_size() for p in models["k"].parameters())
        assert AP._tensor_bytes_on(optimizers, models, "cpu") == 3 * weights

    def test_nothing_is_credited_off_a_card_or_on_a_fresh_run(self, monkeypatch):
        models, optimizers = self._stepped()
        assert AP.restored_state_bytes(optimizers, models, torch.device("cpu")) == 0
        monkeypatch.setattr(AP, "restored_state_bytes", lambda optimizers, models, device: 700)
        common = dict(optimizers=optimizers, models=models, device=torch.device("cuda", 0))
        assert AP.free_bytes_for_planning(10_000, resuming=False, **common) == 10_000
        assert AP.free_bytes_for_planning(10_000, resuming=True, **common) == 10_700


class TestThePlanIsSavedWithTheState:
    @staticmethod
    def _state(**extra):
        return build_training_state(
            step=3, sae_keys=[], optimizers={}, schedulers={}, scalers={}, dead_latent_trackers={},
            activation_ema={}, firing_rate={}, best_loss=1.0, **extra,
        )

    def test_the_plan_and_the_resume_history_round_trip_through_a_weights_only_load(self):
        plan = _plan()
        report = {"checkpoint_step": 10, "checkpoint_id": "ckpt_x", "activation_plan": "replanned",
                  "bit_identical": False, "tokens_skipped": 7, "source_restarted": False,
                  "saved_plan": {"mode": "gpu_rolling", "buffer_tokens": 9, "quotas": [5, 4]},
                  "plan": {"mode": "gpu_rolling", "buffer_tokens": 8, "quotas": [4, 4]}}
        state = _through_disk({"format": "x", **self._state(activation_plan=plan, resume_history=[report])})
        restored = restore_training_state(
            state, sae_keys=[], optimizers={}, schedulers={}, scalers={}, dead_latent_trackers={},
        )
        assert restored["activation_plan"] == plan
        assert restored["resume_history"] == [report]

    def test_a_state_written_before_plans_restores_none_and_no_history(self):
        state = self._state()
        del state["activation_plan"], state["resume_history"]
        restored = restore_training_state(
            state, sae_keys=[], optimizers={}, schedulers={}, scalers={}, dead_latent_trackers={},
        )
        assert restored["activation_plan"] is None and restored["resume_history"] == []


# ── the rolling buffer ──────────────────────────────────────────────────────

SEQ = 8
KEYS = [(3, "residual"), (4, "residual")]


def _rolling_source(folder, source, rows, holes_seed=None):
    """float32 activations [source, row, position, layer]; written once per folder."""
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    r, p = np.meshgrid(np.arange(rows), np.arange(SEQ), indexing="ij")
    for layer, hook in KEYS:
        path = folder / f"s{source}_layer_{layer}_{hook}.npy"
        if not path.exists():
            np.save(path, np.stack([np.full_like(r, source), r, p, np.full_like(r, layer)], axis=-1).astype(np.float32))
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


def _rolling(built, folder, spec, quotas, *, prefetch, keys=KEYS, seed=11):
    sources = [_rolling_source(folder, *s) for s in spec]
    kwargs = {"prefetch": prefetch, "read_threads": 3}
    if prefetch:
        kwargs["host_ram_tokens"] = 10**9
    buf = AB.RollingActivationBuffer(sources, keys, quotas, seed=seed, storage_device=CPU, train_device=CPU, **kwargs)
    built.append(buf)
    return buf


def _rolling_draw(buf, batches, size=13):
    return [[tuple(int(v) for v in row[:3]) for row in buf.next_batch(size)[KEYS[0]].tolist()] for _ in range(batches)]


def _flat(draws):
    return [origin for batch in draws for origin in batch]


@pytest.mark.parametrize("prefetch", [False, True], ids=["sync", "prefetch"])
class TestTheRollingBufferContinuesAReplannedPass:
    def test_under_the_saved_quotas_it_serves_the_buffer_after_the_interrupted_one(self, built, tmp_path, prefetch):
        """So the continuation consumes EXACTLY the interrupted buffer's draws from the
        generators — not none (the rows would be taken again) and not two buffers."""
        spec = [(0, 9), (1, 7, 3)]
        original = _rolling(built, tmp_path, spec, [3 * SEQ, 2 * SEQ], prefetch=prefetch)
        _rolling_draw(original, 2)
        state = _through_disk(original.state_dict())
        remaining = original.size - original._pos
        assert remaining > 0, "precondition: the save is mid-buffer"
        original.next_batch(remaining)
        expected = _rolling_draw(original, 8)  # the following buffers, across a pass of source 1

        resumed = _rolling(built, tmp_path, spec, [3 * SEQ, 2 * SEQ], prefetch=prefetch)
        _rolling_draw(resumed, 1)
        assert resumed.load_state_dict_after_replan(state) == remaining
        assert _rolling_draw(resumed, 8) == expected
        assert resumed.refills == original.refills
        assert resumed.tokens_loaded == original.tokens_loaded
        assert resumed.epochs_completed() == original.epochs_completed()
        # The uninterrupted source SERVED the tail the continuation skipped; later tails agree.
        assert resumed.tokens_dropped == original.tokens_dropped + remaining

    @pytest.mark.parametrize("new_quotas", [[7 * SEQ, 4 * SEQ], [2 * SEQ, SEQ]], ids=["larger", "smaller"])
    def test_under_other_quotas_no_token_repeats_and_the_interrupted_buffer_is_not_served_again(
        self, built, tmp_path, prefetch, new_quotas
    ):
        spec = [(0, 200), (1, 150, 5)]
        original = _rolling(built, tmp_path, spec, [5 * SEQ, 3 * SEQ], prefetch=prefetch)
        before = _rolling_draw(original, 3)
        interrupted = {tuple(int(v) for v in row[:3]) for row in original.tensors[KEYS[0]][: original.size].tolist()}
        state = _through_disk(original.state_dict())

        resumed = _rolling(built, tmp_path, spec, new_quotas, prefetch=prefetch)
        skipped = resumed.load_state_dict_after_replan(state)
        after = _rolling_draw(resumed, 30)  # 390 tokens: within one pass of either source

        assert skipped == state["size"] - state["position"] > 0
        served = _flat(before) + _flat(after)
        assert len(served) == len(set(served)), f"{len(served) - len(set(served))} tokens were served twice"
        assert not set(_flat(after)) & interrupted, "rows of the interrupted buffer were taken again in its pass"
        assert resumed.quotas == new_quotas
        # The skipped tail is counted; later spent buffers may add their own tails.
        assert resumed.tokens_dropped >= state["tokens_dropped"] + skipped


class TestTheRollingBufferRefusesAnotherRunsPass:
    def test_other_sources_other_layers_and_other_data_are_refused(self, built, tmp_path):
        original = _rolling(built, tmp_path / "a", [(0, 9), (1, 7)], [24, 16], prefetch=False)
        _rolling_draw(original, 1)
        state = original.state_dict()
        with pytest.raises(ValueError, match="labels"):
            _rolling(built, tmp_path / "b", [(0, 9), (2, 7)], [24, 16], prefetch=False).load_state_dict_after_replan(state)
        with pytest.raises(ValueError, match="keys"):
            _rolling(built, tmp_path / "a", [(0, 9), (1, 7)], [24, 16], prefetch=False,
                     keys=KEYS[:1]).load_state_dict_after_replan(state)
        with pytest.raises(ValueError, match="rows"):
            _rolling(built, tmp_path / "c", [(0, 10), (1, 7)], [24, 16], prefetch=False).load_state_dict_after_replan(state)


# ── the on-the-fly source ───────────────────────────────────────────────────

PAD = 7_777_777
MD = 4
MKEYS = [(3, "residual"), (5, "residual")]


def _id(source, row, pos):
    return source * 1_000_000 + row * 1_000 + pos


def _origin(value):
    value = int(round(float(value)))
    return value // 1_000_000, (value // 1_000) % 1_000, value % 1_000


class _Rows:
    def __init__(self, source, lengths, width):
        self.source, self.lengths, self.width = source, list(lengths), width

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, row):
        real = self.lengths[row]
        return {
            "input_ids": [_id(self.source, row, p) for p in range(real)] + [PAD] * (self.width - real),
            "attention_mask": [1] * real + [0] * (self.width - real),
        }


def _capture(padded, masks):
    ids = torch.tensor(padded, dtype=torch.float64)
    out = {}
    for layer, hook in MKEYS:
        acts = torch.zeros(ids.shape[0], ids.shape[1], MD, dtype=torch.float64)
        acts[..., 0] = ids
        acts[..., 1] = layer
        out[(layer, hook)] = acts.float()
    return out


def _model_source(source, rows, width=16, lengths=None, label=None):
    lengths = list(lengths) if lengths is not None else [width] * rows
    return MAS.TokenRowSource(
        label=label or f"ds_{source}", dataset=_Rows(source, lengths, width),
        rows=np.arange(len(lengths), dtype=np.int64), max_row_tokens=width,
    )


def _model(sources, quotas, *, keys=MKEYS, seed=11):
    return MAS.ModelActivationSource(
        sources, keys, quotas, capture=_capture, hidden_dim=MD, seed=seed, storage_device=CPU,
        train_device=CPU, micro_batch_rows=3, pad_token_id=PAD,
    )


def _model_origins(batch):
    return [_origin(v) for v in batch[MKEYS[0]][:, 0].tolist()]


class TestTheOnTheFlySourceContinuesAReplannedPass:
    def test_under_the_saved_quotas_it_serves_the_buffer_after_the_interrupted_one(self):
        def sources():
            return [_model_source(0, 60, lengths=[4 + (3 * r) % 12 for r in range(60)]), _model_source(1, 25)]

        original = _model(sources(), [90, 40])
        for _ in range(3):
            original.next_batch(20)
        state = _through_disk(original.state_dict())
        remaining = original.size - original._pos
        assert remaining > 0, "precondition: the save is mid-buffer"
        original.next_batch(remaining)
        expected = [original.next_batch(20) for _ in range(20)]  # several refills and a pass of source 1

        resumed = _model(sources(), [90, 40])
        assert resumed.load_state_dict_after_replan(state) == remaining
        for step, want in enumerate(expected):
            got = resumed.next_batch(20)
            for key in MKEYS:
                assert torch.equal(got[key], want[key]), f"batch {step} differs for {key}"
        assert resumed.refills == original.refills
        assert resumed.tokens_loaded == original.tokens_loaded
        assert resumed.rows_loaded == original.rows_loaded
        assert resumed.epochs_completed() == original.epochs_completed()
        # The uninterrupted source SERVED the tail the continuation skipped; later tails agree.
        assert resumed.tokens_dropped == original.tokens_dropped + remaining

    @pytest.mark.parametrize("new_quotas", [[150, 70], [60, 30]], ids=["larger", "smaller"])
    def test_under_other_quotas_no_token_repeats_and_the_interrupted_buffer_is_not_served_again(self, new_quotas):
        def sources():
            return [_model_source(0, 400), _model_source(1, 300)]

        original = _model(sources(), [90, 40])
        before = [_model_origins(original.next_batch(20)) for _ in range(3)]
        interrupted = {_origin(v) for v in original.tensors[MKEYS[0]][: original.size, 0].tolist()}
        state = _through_disk(original.state_dict())

        resumed = _model(sources(), new_quotas)
        skipped = resumed.load_state_dict_after_replan(state)
        after = [_model_origins(resumed.next_batch(20)) for _ in range(30)]  # 600 tokens, within a pass

        assert skipped == state["size"] - state["position"] > 0
        served = _flat(before) + _flat(after)
        assert len(served) == len(set(served)), f"{len(served) - len(set(served))} tokens were served twice"
        assert not set(_flat(after)) & interrupted, "rows of the interrupted buffer were taken again in its pass"
        assert resumed.quotas == new_quotas

    @pytest.mark.parametrize("change", ["labels", "rows", "keys"])
    def test_another_runs_pass_is_refused(self, change):
        original = _model([_model_source(0, 20)], [48])
        original.next_batch(10)
        state = original.state_dict()
        if change == "labels":
            other = _model([_model_source(0, 20, label="other")], [32])
        elif change == "rows":
            other = _model([_model_source(0, 21)], [32])
        else:
            other = _model([_model_source(0, 20)], [32], keys=MKEYS[:1])
        with pytest.raises(ValueError, match=change if change != "rows" else "rows_digest"):
            other.load_state_dict_after_replan(state)


# ── restore_source_position ─────────────────────────────────────────────────


def _choice(outcome):
    return AP.StoragePlanChoice(
        mode="cpu_rolling", buffer_tokens=0, quotas=None, outcome=outcome, saved=None, fresh=("cpu_rolling", 0),
        gpu_capacity_tokens=0, ram_capacity_tokens=0,
    )


class TestRestoreSourcePosition:
    SPEC = [(0, 60), (1, 50)]

    def test_a_matching_state_loads_exactly_whatever_the_outcome(self, built, tmp_path):
        original = _rolling(built, tmp_path, self.SPEC, [24, 16], prefetch=False)
        _rolling_draw(original, 2)
        state = original.state_dict()
        expected = _rolling_draw(original, 5)
        resumed = _rolling(built, tmp_path, self.SPEC, [24, 16], prefetch=False)
        assert AP.restore_source_position(resumed, state, _choice(AP.REPLANNED)) == {
            "bit_identical": True, "tokens_skipped": 0, "source_restarted": False,
        }
        assert _rolling_draw(resumed, 5) == expected

    def test_a_replanned_rolling_state_continues_the_pass(self, built, tmp_path):
        original = _rolling(built, tmp_path, self.SPEC, [24, 16], prefetch=False)
        _rolling_draw(original, 2)
        state = original.state_dict()
        resumed = _rolling(built, tmp_path, self.SPEC, [16, 8], prefetch=False)
        assert AP.restore_source_position(resumed, state, _choice(AP.REPLANNED)) == {
            "bit_identical": False, "tokens_skipped": state["size"] - state["position"], "source_restarted": False,
        }

    def test_a_replanned_fixed_pool_state_restarts_the_rolling_buffer(self, built, tmp_path):
        pool = AB.FixedPoolSampler({key: torch.zeros(96, 4) for key in KEYS}, KEYS, seed=1)
        pool.next_batch(8)
        fresh = _rolling(built, tmp_path, self.SPEC, [16, 8], prefetch=False)
        untouched = _rolling(built, tmp_path, self.SPEC, [16, 8], prefetch=False)
        assert AP.restore_source_position(fresh, pool.state_dict(), _choice(AP.REPLANNED)) == {
            "bit_identical": False, "tokens_skipped": 0, "source_restarted": True,
        }
        assert _rolling_draw(fresh, 3) == _rolling_draw(untouched, 3)

    @pytest.mark.parametrize("outcome", [AP.REUSED, AP.NOT_RECORDED])
    def test_a_mismatch_that_was_not_replanned_is_refused_as_before(self, built, tmp_path, outcome):
        original = _rolling(built, tmp_path, self.SPEC, [24, 16], prefetch=False)
        _rolling_draw(original, 1)
        other = _rolling(built, tmp_path, self.SPEC, [16, 8], prefetch=False)
        with pytest.raises(ValueError, match="quotas"):
            AP.restore_source_position(other, original.state_dict(), _choice(outcome))

    def test_a_rolling_pass_is_never_continued_by_a_fixed_pool(self, built, tmp_path):
        original = _rolling(built, tmp_path, self.SPEC, [24, 16], prefetch=False)
        _rolling_draw(original, 1)
        pool = AB.FixedPoolSampler({key: torch.zeros(96, 4) for key in KEYS}, KEYS, seed=1)
        with pytest.raises(ValueError, match="without repeating rows"):
            AP.restore_source_position(pool, original.state_dict(), _choice(AP.REPLANNED))
