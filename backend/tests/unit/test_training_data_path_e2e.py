"""train_sae_task, driven end to end on the CPU: the on-the-fly path on a tiny REAL model, and
the held-out sample of the cached path over two real extractions.

SAE training remediation items 2 and 3 (2026-09-15). Before this, a training with
no extractions FAILED at step 0 (`holdout_activations` unbound), and no test ran
that path past the base-model load. The cached path's held-out sample was the
lowest positions of the first extraction.

WHAT IS REAL HERE. The step loop, the SAEs, their optimizers, the storage plan,
the mixture, the held-out split and evaluation, and — on the fly — a real
transformers Llama run through `LayerCapture` and HookManager. What is faked: the
database session (in-memory rows), placement (the CPU), model/dataset loading
(handing over the tiny model and in-memory Arrow datasets), and progress emits.

HOW ORIGINS ARE CHECKED WITH A REAL MODEL. Every forward the capture makes is
recorded, so each served activation vector can be looked up, byte for byte,
among the real-token positions of the rows that were forwarded. A vector not
found there is padding (or invented); the row it came from says which dataset,
whether it was held out, and which block a batch mixed.

MUTATION CONTROLS (2026-09-15). Each broke one line of src/workers/training_tasks.py
(D29: src/services/model_activation_source.py), ran the listed tests, and was restored from
the original bytes with the sha256 and `git diff` re-checked. All KILLED.
  D3   the cached held-out quotas ignore dataset_weights
         -> test_each_extraction_supplies_its_weighted_share_of_whole_held_out_rows
  D4   the held-out sample taken from the first extraction only
         -> test_each_extraction_supplies_its_weighted_share_of_whole_held_out_rows
  D11  the log step evaluates the whole held-out set in one chunk
         -> both held-out E2E tests, test_r4_survivors no-grad guard
  D12  the cached budget reserves nothing for the evaluation chunk
         -> test_the_gpu_budget_leaves_one_evaluation_chunk_free, the budget AST test
  D13  the on-the-fly budget reserves no forward
         -> test_both_paths_reserve_the_held_out_chunk_and_the_fly_reserves_a_forward
  D14  the shared `holdout_activations = {}` removed (the original step-0 crash)
         -> test_a_run_with_no_held_out_split_and_no_weights_completes, three cached tests
  D18  the on-the-fly allocator given no weights
         -> test_the_weights_set_each_datasets_share_of_every_refill, two AST guards
  D20  the source given every row, held-out rows included
         -> test_no_block_is_read_twice_in_a_pass_and_held_out_blocks_are_never_trained_on
  D21  b_dec mean init gated back to cached runs only
         -> test_b_dec_and_thresholds_are_set_from_the_first_buffer
  D22  threshold calibration gated back to cached runs only
         -> test_b_dec_and_thresholds_are_set_from_the_first_buffer
  D27  mismatched weights not refused before the model loads
         -> test_a_weight_per_extraction_count_that_does_not_match_the_datasets_is_refused
  D28  the source never registered on the task for after_return
         -> after_return test and three E2E tests that read the registered source (4)
  D29  close() keeps the capture, and with it the base model
         -> test_after_return_releases_the_source_and_the_model_it_holds
  D32  the SAE width not corrected from the model row
         -> test_a_wrong_hidden_dim_is_corrected_from_the_model_row_before_the_saes_are_built
  D34  the capture's stop hook never registered (the head runs)
         -> test_it_returns_each_layers_output_and_runs_nothing_above_the_deepest
  D37  held-out activations moved off the CPU
         -> both cached held-out tests
Frontend controls F1/F2 are in TrainingPanel.mixture.test.tsx; evaluator and source controls
in test_holdout_evaluation.py and test_model_activation_source.py.
"""

import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.dataset import Dataset
from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from src.models.model import Model
from src.models.training import Training
from src.services.gpu_placement import Placement

#: Large enough that every row's FIRST token is unique across both datasets. A
#: causal model's position-p output depends on tokens 0..p, so a unique first
#: token makes every activation name exactly one row. At 64 the first token was
#: shared by thousands of rows and position-0 vectors could not be attributed.
VOCAB = 16_384
WIDTH = 16
HIDDEN = 32
LAYERS = [1, 2]
KEYS = [(1, "residual"), (2, "residual")]


# ── an in-memory session that honours equality filters ─────────────────────

def _matches(row, condition):
    left, right = getattr(condition, "left", None), getattr(condition, "right", None)
    key = getattr(left, "key", None)
    if key is None or not hasattr(right, "value"):
        return True
    return getattr(row, key, None) == right.value


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter_by(self, **criteria):
        return _Query(r for r in self._rows if all(getattr(r, k, None) == v for k, v in criteria.items()))

    def filter(self, *conditions):
        return _Query(r for r in self._rows if all(_matches(r, c) for c in conditions))

    def order_by(self, *columns):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _Session:
    def __init__(self, rows):
        self.rows = rows

    def query(self, model):
        return _Query(self.rows.get(model, []))

    def commit(self):
        pass

    def rollback(self):
        pass

    def add(self, obj):
        pass

    def close(self):
        pass


def _training(hp, **fields):
    base = dict(
        id="train_e2e", model_id="m_tiny", status="pending", current_step=0, current_loss=None,
        dataset_id="ds_a", dataset_ids=["ds_a", "ds_b"], extraction_id=None, extraction_ids=None,
        gpu_request="auto", gpu_uuid=None, gpu_uuids=None, checkpoint_dir=None,
        error_message=None, error_traceback=None, completed_at=None, progress=0.0,
    )
    base.update(fields)
    base["hyperparameters"] = {
        "hidden_dim": HIDDEN, "latent_dim": 64, "batch_size": 64, "learning_rate": 1e-3,
        "total_steps": 6, "log_interval": 2, "checkpoint_interval": 10_000, "seed": 5,
        "training_layers": LAYERS, "hook_types": ["residual"], "architecture_type": "jumprelu",
        "sparsity_coeff": 1e-3, "evaluate_ce_delta": False, "sparsity_warmup_steps": 0,
        **hp,
    }
    return SimpleNamespace(**base)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Everything train_sae_task touches outside the step loop, faked; the loop itself real."""
    from src.services import activation_service
    from src.workers import base_task, training_tasks, websocket_emitter

    state = SimpleNamespace(rows={}, metrics=[], progress=[], draws=[], captures=[], evaluations=[])

    @contextmanager
    def get_sync_db():
        yield _Session(state.rows)

    def draw(*args, **kwargs):
        batch = real_draw(*args, **kwargs)
        state.draws.append({k: v.detach().clone() for k, v in batch.items()})
        return batch

    real_draw = training_tasks.draw_cached_batch
    real_capture = training_tasks.LayerCapture.__call__
    real_evaluate = training_tasks.holdout_evaluation.evaluate_holdout

    def capture(self, padded, masks):
        out = real_capture(self, padded, masks)
        state.captures.append((padded, masks, {k: v.detach().clone() for k, v in out.items()}))
        return out

    def evaluate(model, held, device, chunk_tokens, **kwargs):
        state.evaluations.append((held.detach().clone(), chunk_tokens, held.device.type))
        return real_evaluate(model, held, device, chunk_tokens, **kwargs)

    monkeypatch.setattr(base_task, "get_sync_db", get_sync_db)
    monkeypatch.setattr(training_tasks, "draw_cached_batch", draw)
    monkeypatch.setattr(training_tasks.LayerCapture, "__call__", capture)
    monkeypatch.setattr(training_tasks.holdout_evaluation, "evaluate_holdout", evaluate)
    monkeypatch.setattr(
        training_tasks, "estimate_multilayer_training_memory",
        lambda **kw: {"total_gb": 0.1, "total_mb": 100.0, "fits_in_6gb": True, "available_gpu_gb": 8.0,
                      "per_layer_gb": 0.05, "max_layers_in_6gb": 8},
    )
    monkeypatch.setattr(
        training_tasks, "estimate_training_memory",
        lambda **kw: {"total_gb": 0.1, "total_mb": 100.0, "fits_in_6gb": True, "available_gpu_gb": 8.0},
    )
    monkeypatch.setattr(
        training_tasks, "place_job",
        lambda requested, required_mb=None, cards=None, allow_shard=False: Placement(card=None, device=torch.device("cpu")),
    )
    monkeypatch.setattr(
        training_tasks.TrainingValidator, "validate_sparsity_config", staticmethod(lambda hp: ([], []))
    )
    monkeypatch.setattr(training_tasks.settings, "data_dir", tmp_path)
    monkeypatch.setattr(websocket_emitter, "emit_training_progress", lambda *a, **k: True)
    monkeypatch.setattr(websocket_emitter, "emit_checkpoint_created", lambda *a, **k: True)
    monkeypatch.setattr(training_tasks.CheckpointService, "save_multilayer_community_checkpoint", staticmethod(lambda **kw: {}))
    task = training_tasks.train_sae_task
    monkeypatch.setattr(task, "log_metric", lambda **kw: state.metrics.append(kw), raising=False)
    monkeypatch.setattr(task, "update_training_progress", lambda **kw: state.progress.append(kw), raising=False)
    monkeypatch.setattr(task, "_activation_stream", None, raising=False)
    state.set_ram = lambda nbytes: monkeypatch.setattr(activation_service, "_available_memory_bytes", lambda: nbytes)
    state.run = lambda training_id="train_e2e": task.run(training_id)
    state.task = task
    yield state
    stream = getattr(task, "_activation_stream", None)
    if stream is not None:
        stream.close()


# ── on the fly ─────────────────────────────────────────────────────────────

def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2,
    )
    return LlamaForCausalLM(config).eval()


def _tokenized(offset: int, rows: int, seed: int):
    """Rows whose FIRST id names the row uniquely across datasets; every eleventh row is padded."""
    from datasets import Dataset as ArrowDataset

    rng = np.random.default_rng(seed)
    ids, masks = [], []
    for r in range(rows):
        k = offset + r
        assert 2 + k < VOCAB
        real = WIDTH if r % 11 else 6
        row = [2 + k] + rng.integers(2, VOCAB, WIDTH - 1).tolist()
        ids.append(row[:real] + [0] * (WIDTH - real))
        masks.append([1] * real + [0] * (WIDTH - real))
    return ArrowDataset.from_dict({"input_ids": ids, "attention_mask": masks})


ROWS_PER_DATASET = 4_000


@pytest.fixture
def on_the_fly(harness, monkeypatch):
    from src.workers import training_tasks

    model = _tiny_llama()
    datasets = {"ds_a": _tokenized(0, ROWS_PER_DATASET, 1), "ds_b": _tokenized(ROWS_PER_DATASET, ROWS_PER_DATASET, 2)}
    harness.model, harness.datasets = model, datasets
    harness.rows = {
        Training: [_training({"holdout_fraction": 0.2, "holdout_eval_tokens": 400, "dataset_weights": [3.0, 1.0]})],
        Model: [SimpleNamespace(id="m_tiny", repo_id="org/tiny-llama", quantization="FP16", file_path=None,
                                architecture="llama", architecture_config={"hidden_size": HIDDEN})],
        Dataset: [SimpleNamespace(id="ds_a"), SimpleNamespace(id="ds_b")],
        DatasetTokenization: [
            SimpleNamespace(dataset_id=name, model_id="m_tiny", status=TokenizationStatus.READY,
                            tokenized_path=f"/tok/{name}", tokenizer_repo_id="org/tiny-llama",
                            vocab_size=VOCAB, max_length=WIDTH)
            for name in datasets
        ],
    }
    harness.state_rows = harness.rows
    monkeypatch.setattr(training_tasks, "select_tokenization_for_model",
                        lambda candidates, model_id, ds_id: next(c for c in candidates if c.dataset_id == ds_id))
    monkeypatch.setattr(training_tasks, "load_from_disk", lambda path: datasets[path.rsplit("/", 1)[-1]])
    harness.loads = []
    monkeypatch.setattr(
        training_tasks, "load_model_from_hf",
        lambda **kw: harness.loads.append(kw) or (
            model, SimpleNamespace(pad_token_id=0, eos_token_id=1, vocab_size=VOCAB), model.config, {}
        ),
    )
    # The RAM budget decides the buffer: 50,000 tokens per layer, so the ~128,000
    # token corpus is CYCLED (cpu_rolling) and the weights bind on every refill.
    harness.set_ram(50_000 * HIDDEN * 4 * len(KEYS) * 2)
    return harness


def _row_identity(datasets):
    """Maps a forwarded row's ids to (dataset, row)."""
    table = {}
    for name, ds in datasets.items():
        for r, ids in enumerate(ds["input_ids"]):
            table[tuple(ids)] = (name, r)
    return table


def _origin_index(captures, identity, key):
    """Bytes of every REAL-position activation the model produced -> {(dataset, row, pos)}."""
    index = defaultdict(set)
    for padded, masks, acts in captures:
        tensor = acts[key]
        for i, (ids, mask) in enumerate(zip(padded, masks)):
            origin = identity[tuple(ids[:WIDTH])]
            for p, real in enumerate(mask):
                if real:
                    index[tensor[i, p].numpy().tobytes()].add((*origin, p))
    return index


class TestOnTheFlyTrainingEndToEnd:
    def test_it_trains_every_step_and_completes(self, on_the_fly):
        result = on_the_fly.run()
        assert result["status"] == "completed" and result["steps"] == 6
        assert len(on_the_fly.draws) == 6
        assert all(b[k].shape == (64, HIDDEN) for b in on_the_fly.draws for k in KEYS)

    def test_batches_are_real_tokens_from_many_blocks_at_the_same_positions_for_every_layer(self, on_the_fly):
        on_the_fly.run()
        identity = _row_identity(on_the_fly.datasets)
        index = {key: _origin_index(on_the_fly.captures, identity, key) for key in KEYS}
        for batch in on_the_fly.draws:
            rows = set()
            for i in range(64):
                found = [index[key].get(batch[key][i].numpy().tobytes()) for key in KEYS]
                assert all(found), "a served activation is not any real token the model produced: padding"
                assert found[0] & found[1], "layer 1 and layer 2 were served different positions"
                rows |= {(d, r) for d, r, p in found[0] if p >= 3}
            assert len(rows) > 20, f"a 64-token batch came from only {len(rows)} blocks"

    def test_the_weights_set_each_datasets_share_of_every_refill(self, on_the_fly):
        on_the_fly.run()
        stream = on_the_fly.task._activation_stream
        assert stream.quotas == [37_500, 12_500], stream.quotas
        a, b = stream.tokens_loaded
        assert 37_500 - WIDTH <= a <= 37_500 and 12_500 - WIDTH <= b <= 12_500
        served = Counter()
        identity = _row_identity(on_the_fly.datasets)
        index = _origin_index(on_the_fly.captures, identity, KEYS[0])
        for batch in on_the_fly.draws:
            for vec in batch[KEYS[0]]:
                served[next(iter(index[vec.numpy().tobytes()]))[0]] += 1
        assert 2.0 < served["ds_a"] / served["ds_b"] < 4.5, served

    def test_no_block_is_read_twice_in_a_pass_and_held_out_blocks_are_never_trained_on(self, on_the_fly):
        from src.services import activation_mask

        on_the_fly.run()
        stream = on_the_fly.task._activation_stream
        held = {
            name: set(activation_mask.split_rows(ROWS_PER_DATASET, 0.2, 5)[1].tolist())
            for name in ("ds_a", "ds_b")
        }
        identity = _row_identity(on_the_fly.datasets)
        refill_rows = Counter()
        held_out_rows = set()
        for padded, _, _ in on_the_fly.captures:
            for ids in padded:
                refill_rows[identity[tuple(ids[:WIDTH])]] += 1
        # The source reads training rows; the held-out collection reads held rows.
        for (name, row), count in refill_rows.items():
            if row in held[name]:
                held_out_rows.add((name, row))
                continue
            assert count == 1, f"{name} row {row} was forwarded {count} times within one pass"
        assert sum(stream.rows_loaded) == sum(c for (n, r), c in refill_rows.items() if r not in held[n])
        assert held_out_rows, "no held-out row was ever forwarded for evaluation"
        index = _origin_index(on_the_fly.captures, identity, KEYS[0])
        for batch in on_the_fly.draws:
            for vec in batch[KEYS[0]]:
                for name, row, _ in index[vec.numpy().tobytes()]:
                    assert row not in held[name], f"held-out {name} row {row} was served to training"

    def test_the_held_out_set_follows_the_weights_and_is_evaluated_in_chunks_on_the_cpu(self, on_the_fly):
        from src.services import activation_mask

        on_the_fly.run()
        assert on_the_fly.evaluations, "the held-out set was never evaluated"
        held_tensor, chunk, device_type = on_the_fly.evaluations[0]
        assert device_type == "cpu" and chunk == 2_048
        identity = _row_identity(on_the_fly.datasets)
        index = _origin_index(on_the_fly.captures, identity, KEYS[0])
        held = {
            name: set(activation_mask.split_rows(ROWS_PER_DATASET, 0.2, 5)[1].tolist())
            for name in ("ds_a", "ds_b")
        }
        per = Counter()
        for vec in held_tensor:
            origins = index[vec.numpy().tobytes()]
            assert origins and all(r in held[n] for n, r, _ in origins)
            per[next(iter(origins))[0]] += 1
        assert per == {"ds_a": 300, "ds_b": 100}, per
        rows = [m for m in on_the_fly.metrics if m.get("layer_idx") is not None and m["layer_idx"] < 0]
        assert {m["layer_idx"] for m in rows} == {-2, -3}
        assert all(m["fvu"] is not None and m["l0_mean"] is not None for m in rows)

    def test_a_padding_heavy_datasets_held_out_shortfall_is_evaluated_from_the_other(self, on_the_fly):
        """R1-B F3, through the task (2026-09-15). ds_b holds ONE real token per row: 800
        held-out tokens against an equal share of 1,000. Its shortfall was not evaluated
        (1,800 of 2,000 tokens); the other dataset now supplies it.

        MUTATION CONTROL R1-B C7 (the collection's redistribution round removed) -> this test fails.
        """
        from datasets import Dataset as ArrowDataset

        first = 2 + ROWS_PER_DATASET
        on_the_fly.datasets["ds_b"] = ArrowDataset.from_dict({
            "input_ids": [[first + r] + [0] * (WIDTH - 1) for r in range(ROWS_PER_DATASET)],
            "attention_mask": [[1] + [0] * (WIDTH - 1) for _ in range(ROWS_PER_DATASET)],
        })
        hp = on_the_fly.rows[Training][0].hyperparameters
        hp.pop("dataset_weights")
        hp["holdout_eval_tokens"] = 2_000
        assert on_the_fly.run()["status"] == "completed"
        held_tensor, _, _ = on_the_fly.evaluations[0]
        index = _origin_index(on_the_fly.captures, _row_identity(on_the_fly.datasets), KEYS[0])
        per = Counter(next(iter(index[v.numpy().tobytes()]))[0] for v in held_tensor)
        assert per == {"ds_a": 1_200, "ds_b": 800}, per

    def test_the_held_out_collection_is_given_the_weights_and_only_held_out_rows(self, on_the_fly, monkeypatch):
        """The task's call carries dataset_weights, so a shortfall is re-allocated by the
        mixture rather than equally (MUTATION CONTROL R1-B C10: `weights=` dropped -> red)."""
        from src.services import activation_mask
        from src.workers import training_tasks

        calls = []
        real = training_tasks.model_activation_source.collect_holdout_activations

        def spy(sources, keys, quotas, **kwargs):
            calls.append((sources, list(quotas), kwargs))
            return real(sources, keys, quotas, **kwargs)

        monkeypatch.setattr(training_tasks.model_activation_source, "collect_holdout_activations", spy)
        assert on_the_fly.run()["status"] == "completed"
        assert len(calls) == 1, calls
        sources, quotas, kwargs = calls[0]
        assert kwargs["weights"] == [3.0, 1.0] and quotas == [300, 100]
        held = activation_mask.split_rows(ROWS_PER_DATASET, 0.2, 5)[1]
        assert [label for label, _, _ in sources] == ["ds_a", "ds_b"]
        assert all(np.array_equal(rows, held) for _, _, rows in sources)

    def test_b_dec_and_thresholds_are_set_from_the_first_buffer(self, on_the_fly, monkeypatch):
        from src.ml.sparse_autoencoder import JumpReLUSAE
        from src.services import model_activation_source as MAS

        first = {}
        real_refill = MAS.ModelActivationSource.refill

        def refill(self):
            real_refill(self)
            if not first:
                first.update({k: v.clone() for k, v in self.tensors.items()})

        calibrations = []
        real_calibrate = JumpReLUSAE.calibrate_thresholds

        def calibrate(self, x, target_l0=0.05):
            calibrations.append((x.clone(), self.b_dec.detach().clone(), self))
            return real_calibrate(self, x, target_l0)

        monkeypatch.setattr(MAS.ModelActivationSource, "refill", refill)
        monkeypatch.setattr(JumpReLUSAE, "calibrate_thresholds", calibrate)
        on_the_fly.run()

        assert len(calibrations) == len(KEYS)
        for (x, b_dec, sae), key in zip(calibrations, KEYS):
            expected = sae.normalize(first[key])[0].mean(dim=0)
            torch.testing.assert_close(b_dec, expected, rtol=1e-5, atol=1e-6)
            rows = {r.numpy().tobytes() for r in first[key]}
            assert all(r.numpy().tobytes() in rows for r in x), "calibration drew tokens outside the first buffer"

    def test_a_weight_per_extraction_count_that_does_not_match_the_datasets_is_refused(self, on_the_fly):
        on_the_fly.rows[Training][0].hyperparameters["dataset_weights"] = [1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="dataset_weights has 3 entries"):
            on_the_fly.run()
        assert on_the_fly.loads == [] and on_the_fly.captures == [], "refused only after loading the model"

    def test_a_wrong_hidden_dim_is_corrected_from_the_model_row_before_the_saes_are_built(self, on_the_fly):
        on_the_fly.rows[Training][0].hyperparameters["hidden_dim"] = 999
        assert on_the_fly.run()["status"] == "completed"


# ── cached: the held-out sample spans every extraction ─────────────────────

SEQ, D_CACHED, ROWS = 8, 8, 60


def _extraction(tmp_path, index: int):
    """A real extraction directory whose every activation encodes (extraction, row, position)."""
    out = tmp_path / f"ext_{index}"
    out.mkdir()
    acts = np.zeros((ROWS, SEQ, D_CACHED), dtype=np.float32)
    rows, positions = np.meshgrid(np.arange(ROWS), np.arange(SEQ), indexing="ij")
    acts[..., 0] = index * 10_000 + rows * 100 + positions
    acts[..., 1:] = np.random.default_rng(index).normal(size=(ROWS, SEQ, D_CACHED - 1))
    np.save(out / "layer_1_residual.npy", acts)
    mask = np.ones((ROWS, SEQ), dtype=bool)
    mask[::7, 5:] = False
    np.save(out / "attention_mask.npy", mask)
    (out / "metadata.json").write_text(json.dumps({
        "num_samples_processed": ROWS, "layer_indices": [1], "hook_types": ["residual"], "seq_len": SEQ,
    }))
    return SimpleNamespace(id=f"ext_{index}", status="completed", output_path=str(out), dataset_id=f"ds_{index}"), mask


class TestTheOnTheFlyEvaluationReadsOnlyHeldOutRows:
    """Integration of WS-DATA and WS-EVAL (2026-09-15).

    An on-the-fly run's post-run evaluation must read rows the SAEs never trained
    on. The only rows this path guarantees that for are its held-out rows, so the
    run hands the evaluation exactly those, one source per dataset, weighted like
    the mixture. Before the integration it handed nothing and the evaluation
    recorded "skipped".

    MUTATION CONTROLS (2026-09-15; each restored, sha256 verified):
      INT-M1 candidate rows are the training rows, not the held-out rows
             -> test_the_post_run_evaluation_gets_each_datasets_held_out_rows
      INT-M2 the call site passes eval_sources=None
             -> both on-the-fly tests above
      INT-M3 run_post_run_evaluation ignores eval_sources
             -> test_run_post_run_evaluation_evaluates_the_sources_it_is_given
      INT-M4 every source weighted 1.0 regardless of dataset_weights
             -> test_the_post_run_evaluation_gets_each_datasets_held_out_rows
      INT-M5 the held-out log_metric call drops fvu_centred
             -> SURVIVED first; red after test_every_log_step_writes_a_held_out_row_with_both_fvus
                gained its centred-FVU assertion
    """

    @staticmethod
    def _spy(monkeypatch):
        from src.workers import training_tasks

        calls = []
        monkeypatch.setattr(training_tasks, "run_post_run_evaluation", lambda task, **kw: calls.append(kw))
        return calls

    def test_the_post_run_evaluation_gets_each_datasets_held_out_rows(self, on_the_fly, monkeypatch):
        from src.services import activation_mask

        calls = self._spy(monkeypatch)
        on_the_fly.run()
        assert len(calls) == 1, calls
        call = calls[0]
        assert call["extractions"] is None
        sources = call["eval_sources"]
        assert [source.label for source in sources] == ["ds_a", "ds_b"]
        assert [source.weight for source in sources] == [3.0, 1.0]
        held = activation_mask.split_rows(ROWS_PER_DATASET, 0.2, 5)[1]
        for source in sources:
            assert source.rows_read is None
            assert source.dataset_path.endswith(f"tok/{source.label}"), source.dataset_path
            assert source.candidate_rows == tuple(int(row) for row in held.tolist())

    def test_a_run_with_no_held_out_split_hands_the_evaluation_no_rows(self, on_the_fly, monkeypatch):
        calls = self._spy(monkeypatch)
        on_the_fly.rows[Training] = [_training({"holdout_fraction": 0.0})]
        on_the_fly.run()
        assert len(calls) == 1 and calls[0]["eval_sources"] == []

    def test_run_post_run_evaluation_evaluates_the_sources_it_is_given(self, on_the_fly, monkeypatch):
        from src.services import training_evaluation
        from src.workers import training_tasks

        seen = {}

        def run_evaluation(**kw):
            seen["sources"] = kw["sources"]()
            return {"status": "completed"}

        monkeypatch.setattr(training_evaluation, "run_evaluation", run_evaluation)
        monkeypatch.setattr(training_evaluation, "base_model_loader", lambda **kw: (lambda *a, **k: None))
        source = training_evaluation.EvalSource(label="ds_a", dataset_path="/tok/ds_a", candidate_rows=(3, 7))
        training_tasks.run_post_run_evaluation(
            on_the_fly.task, training_id="train_e2e", hp={}, models={}, placement=None,
            base_model=object(), extractions=None, sae_mb=0.0, eval_sources=[source],
        )
        assert seen["sources"] == [source]


class TestTheCachedHeldOutSampleSpansEveryExtraction:
    @pytest.fixture
    def cached(self, harness, tmp_path):
        from src.models.activation_extraction import ActivationExtraction

        (ext_a, mask_a), (ext_b, mask_b) = _extraction(tmp_path, 1), _extraction(tmp_path, 2)
        harness.masks = {1: mask_a, 2: mask_b}
        harness.rows = {
            Training: [_training(
                {"hidden_dim": D_CACHED, "latent_dim": 16, "training_layers": [1], "architecture_type": "jumprelu",
                 "l1_alpha": 1e-3, "holdout_fraction": 0.25, "holdout_eval_tokens": 120,
                 "holdout_eval_chunk_tokens": 32, "dataset_weights": [1.0, 3.0], "total_steps": 4},
                extraction_ids=["ext_1", "ext_2"], dataset_ids=["ds_1", "ds_2"],
            )],
            Model: [SimpleNamespace(id="m_tiny", repo_id="org/tiny", quantization="FP16", file_path=None)],
            ActivationExtraction: [ext_a, ext_b],
        }
        harness.set_ram(10**9)
        return harness

    def test_each_extraction_supplies_its_weighted_share_of_whole_held_out_rows(self, cached):
        from src.services import activation_mask

        assert cached.run()["status"] == "completed"
        assert cached.evaluations, "the held-out set was never evaluated"
        held, chunk, device_type = cached.evaluations[0]
        assert device_type == "cpu" and chunk == 32
        origins = [(int(v) // 10_000, (int(v) // 100) % 100, int(v) % 100) for v in held[:, 0].tolist()]
        per = Counter(e for e, _, _ in origins)
        assert per == {1: 30, 2: 90}, f"held-out tokens per extraction {dict(per)}; asked for 1:3 of 120"

        for ext in (1, 2):
            flat = activation_mask.valid_flat_indices(cached.masks[ext])
            _, held_flat = activation_mask.split_documents(flat, SEQ, 0.25, 5)
            held_rows = set((held_flat // SEQ).tolist())
            rows = {r for e, r, _ in origins if e == ext}
            assert rows <= held_rows, f"extraction {ext} evaluated rows it trained on: {sorted(rows - held_rows)}"
            lowest = sorted(held_rows)[: len(rows)]
            assert sorted(rows) != lowest, f"extraction {ext}'s held-out rows were taken lowest first"
            for e, r, p in origins:
                if e == ext:
                    assert cached.masks[ext][r, p], "a padding position was evaluated"

    def test_a_held_out_split_is_honoured_when_no_mask_can_be_recovered(self, cached):
        """R1-B F4 (2026-09-15). With no `attention_mask.npy` and no tokenization to recover
        one from, the split ran only in the MASKED branch: `holdout_fraction` was dropped
        without a word, every document trained, and no held-out row was ever written — an
        operator who asked for out-of-sample numbers got in-sample ones, unlabelled. The
        split must hold out whole documents over every position (padding cannot be told
        apart there, which the PADDING NOT MASKED warning already says).

        MUTATION CONTROL R1-B C4: the no-mask branch skips the split again -> this test fails.
        """
        from pathlib import Path

        from src.models.activation_extraction import ActivationExtraction
        from src.services import activation_mask

        for extraction in cached.rows[ActivationExtraction]:
            (Path(extraction.output_path) / "attention_mask.npy").unlink()
        assert cached.run()["status"] == "completed"
        assert cached.evaluations, "holdout_fraction was requested and no held-out evaluation ran"

        _, held_flat = activation_mask.split_documents(np.arange(ROWS * SEQ), SEQ, 0.25, 5)
        held_rows = set((held_flat // SEQ).tolist())
        held, _, _ = cached.evaluations[0]
        evaluated = {(int(v) // 10_000, (int(v) // 100) % 100) for v in held[:, 0].tolist()}
        assert {e for e, _ in evaluated} == {1, 2}, evaluated
        assert all(row in held_rows for _, row in evaluated), "a training document was evaluated as held out"
        trained = {
            (int(v) // 10_000, (int(v) // 100) % 100)
            for batch in cached.draws for v in batch[(1, "residual")][:, 0].tolist()
        }
        assert trained and not any(row in held_rows for _, row in trained), (
            f"held-out documents were trained on: {sorted(r for _, r in trained if r in held_rows)[:10]}"
        )
        assert [m["step"] for m in cached.metrics if m.get("layer_idx") == -2] == [0, 2]

    def test_a_held_out_split_without_a_mask_is_warned_about_and_recorded_in_the_settings(self, cached, caplog):
        """R1-D L7, fixed in R1-B (2026-09-15). With no recorded mask the split now runs (F4),
        but padding cannot be told apart there, so the held-out documents include padding
        positions: a different evaluation from a masked extraction's, and silent until now.
        It is made visible twice: one WARNING naming the extractions, and their ids recorded in
        the training's hyperparameters, where the run's settings are reported. The ids are
        written as a NEW dict through the session, never into the task's in-memory `hp` — this
        fixture's row shares that dict, and an in-place write would satisfy the assertion even
        with the database write deleted.

        MUTATION CONTROLS R1-B C11 (the warning removed) and C12 (the settings write removed) -> red.
        """
        import logging
        from pathlib import Path

        from src.models.activation_extraction import ActivationExtraction
        from src.workers import training_tasks

        for extraction in cached.rows[ActivationExtraction]:
            (Path(extraction.output_path) / "attention_mask.npy").unlink()
        with caplog.at_level(logging.WARNING, logger=training_tasks.logger.name):
            assert cached.run()["status"] == "completed"
        warnings = [
            r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING and "held-out" in r.getMessage().lower()
        ]
        assert len(warnings) == 1, warnings
        assert "ext_1" in warnings[0] and "ext_2" in warnings[0], warnings[0]
        settings = cached.rows[Training][0].hyperparameters
        assert settings.get("holdout_unmasked_extractions") == ["ext_1", "ext_2"], settings

    def test_a_masked_run_records_no_unmasked_held_out_extractions(self, cached):
        """NEGATIVE CONTROL for the test above: with every mask recorded, nothing is recorded."""
        assert cached.run()["status"] == "completed"
        assert "holdout_unmasked_extractions" not in cached.rows[Training][0].hyperparameters

    def test_every_log_step_writes_a_held_out_row_with_both_fvus(self, cached, monkeypatch):
        """The legacy FVU keeps its column, and the centred FVU the chunked evaluator
        measured reaches its own (integration, 2026-09-15; control INT-M5 dropped the
        `fvu_centred` argument and survived before the second assertion existed)."""
        from src.workers import training_tasks

        results = []
        measured = training_tasks.holdout_evaluation.evaluate_holdout

        def record(*args, **kwargs):
            out = measured(*args, **kwargs)
            results.append(out)
            return out

        monkeypatch.setattr(training_tasks.holdout_evaluation, "evaluate_holdout", record)
        cached.run()
        rows = [m for m in cached.metrics if m.get("layer_idx") == -2]
        assert [m["step"] for m in rows] == [0, 2]
        assert all(m["fvu"] is not None and m["l0_sparsity"] is not None for m in rows)
        assert len(results) == len(rows)
        assert all(r["fvu_centred"] is not None for r in results)
        assert [m.get("fvu_centred") for m in rows] == [r["fvu_centred"] for r in results]

    def test_the_gpu_budget_leaves_one_evaluation_chunk_free(self, cached, monkeypatch):
        """Driven: the storage plan's GPU capacity falls by exactly one chunk's bound.

        Two runs differing only in the chunk size; every other budget term is
        identical between them, so the difference isolates the reservation.
        """
        from src.services import activation_buffer
        from src.services.holdout_evaluation import holdout_eval_peak_bytes
        from src.workers import training_tasks

        plans = []
        real_plan = activation_buffer.plan_activation_storage
        monkeypatch.setattr(
            training_tasks.activation_buffer, "plan_activation_storage",
            lambda *args: plans.append(args) or real_plan(*args),
        )
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (4 * 1024**3, 8 * 1024**3))
        row = cached.rows[Training][0]
        small, big = 32, 32 * 1024
        for chunk in (small, big):
            row.status = "pending"
            row.hyperparameters["holdout_eval_chunk_tokens"] = chunk
            assert cached.run()["status"] == "completed"
        gpu_small, gpu_big = plans[0][1], plans[1][1]
        delta = holdout_eval_peak_bytes(big, D_CACHED, 16) - holdout_eval_peak_bytes(small, D_CACHED, 16)
        assert gpu_small - gpu_big == pytest.approx(delta * 0.9 / (D_CACHED * 4), abs=1.0)


class TestTheOnTheFlyDefaults:
    def test_a_run_with_no_held_out_split_and_no_weights_completes(self, on_the_fly):
        """The configuration that FAILED at step 0: holdout_activations was unbound."""
        hp = on_the_fly.rows[Training][0].hyperparameters
        for name in ("dataset_weights", "holdout_fraction", "holdout_eval_tokens"):
            hp.pop(name)
        assert on_the_fly.run()["status"] == "completed"
        assert on_the_fly.evaluations == []
        a, b = on_the_fly.task._activation_stream.quotas
        assert abs(a - b) <= 0.05 * (a + b), "without weights, equal-size datasets get equal shares"

    def test_after_return_releases_the_source_and_the_model_it_holds(self, on_the_fly, monkeypatch):
        """The source is CLOSED before the post-run evaluation and DETACHED by after_return.

        REVIEW R1-C, for R1-D's R1D-5 (2026-09-15). This asserted `stream.capture is not
        None` after the run: that the source still held its buffer and its capture until
        after_return. That is the defect R1D-5 reports — the on-the-fly buffer was sized to
        fill the card, and the evaluation ran the model with full-vocabulary logits beside
        it. The release before the evaluation now closes the source in place; after_return
        remains the idempotent backstop and the detach.
        """
        from src.services import model_activation_source as MAS

        on_the_fly.run()
        stream = on_the_fly.task._activation_stream
        assert isinstance(stream, MAS.ModelActivationSource)
        assert stream._closed and stream.tensors == {}, "the buffer outlived the training into the evaluation"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        on_the_fly.task.after_return("SUCCESS", None, "task-1", (), {}, None)
        assert stream._closed and stream.tensors == {}
        assert stream.capture is None, "the closed source still holds the capture, and with it the base model"
        assert on_the_fly.task._activation_stream is None


class TestLayerCapture:
    def test_it_returns_each_layers_output_and_runs_nothing_above_the_deepest(self):
        from src.ml.forward_hooks import HookType
        from src.workers.training_tasks import LayerCapture

        model = _tiny_llama()
        ran = []
        model.model.layers[3].register_forward_hook(lambda *a: ran.append("layer 3"))
        model.lm_head.register_forward_hook(lambda *a: ran.append("head"))
        ids = torch.randint(2, VOCAB, (2, 10), generator=torch.Generator().manual_seed(0)).tolist()
        masks = [[1] * 10, [1] * 6 + [0] * 4]
        with torch.no_grad():
            expected = model(
                input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks), output_hidden_states=True
            ).hidden_states
        ran.clear()

        capture = LayerCapture(model, LAYERS, [HookType.RESIDUAL], "llama", KEYS)
        with capture:
            out = capture(ids, masks)
        assert ran == [], f"the forward ran {ran} after the deepest trained layer"
        for layer in LAYERS:
            torch.testing.assert_close(out[(layer, "residual")], expected[layer + 1], rtol=0, atol=0)

        with torch.no_grad():
            model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks))
        assert ran == ["layer 3", "head"], "the capture's hooks outlived it"


class TestTheBudgetAndTheKnobs:
    def test_reserved_bytes_come_out_of_the_buffer_one_for_one(self):
        from src.workers.training_tasks import sae_buffer_budget

        shape = dict(num_keys=3, hidden_dim=2048, latent_dim=16384, batch_size=4096)
        a = sae_buffer_budget(20 * 1024**3, reserved_bytes=0, **shape)
        b = sae_buffer_budget(20 * 1024**3, reserved_bytes=768 * 1024**2, **shape)
        assert a["available"] - b["available"] == 768 * 1024**2
        assert a["pending_sae_bytes"] == 3 * 2 * 2048 * 16384 * 12

    @pytest.mark.parametrize("arch", ["jumprelu", "standard", "topk"])
    def test_the_step_reservation_covers_a_measured_training_step(self, arch):
        """R1-B F5 (2026-09-15). `sae_buffer_budget` reserved max(1 GiB, a per-key formula)
        for the training step. Measured on CPU allocations at the production 16K shape, one
        JumpReLU step at batch 4,096 holds ~2.8 GiB beyond its weights, gradients and Adam
        moments — the formula gave a single-layer run 1 GiB, so its buffer was sized into
        memory the first step then needed. Here a smaller shape is measured and the
        reservation must cover it.

        MUTATION CONTROL R1-B C8: the step term dropped from the overhead -> the production
        shape assertion fails (1.00 GiB).
        """
        from src.ml.sparse_autoencoder import create_sae
        from src.workers.training_tasks import sae_buffer_budget, training_step_bytes
        from tests.unit.test_holdout_evaluation import _LiveTensorBytes, _storages

        hidden, latent, batch = 256, 8_192, 1_024
        torch.manual_seed(0)
        kwargs = dict(architecture_type=arch, hidden_dim=hidden, latent_dim=latent, l1_alpha=1e-3)
        if arch == "jumprelu":
            kwargs.update(initial_threshold=0.05, sparsity_coeff=1e-3)
        if arch == "topk":
            kwargs.update(top_k=64)
        sae = create_sae(**kwargs)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
        x = torch.randn(batch, hidden)
        meter = _LiveTensorBytes(_storages(*sae.parameters(), *sae.buffers(), x))
        with meter:
            for _ in range(2):  # the second step runs with the moments already allocated
                optimizer.zero_grad()
                sae(x, return_loss=True)[2]["loss"].backward()
                optimizer.step()
        weights = sum(p.numel() * 4 for p in sae.parameters())
        step = meter.peak - 3 * weights  # beyond gradients and the two moments
        reserved = training_step_bytes(batch_size=batch, hidden_dim=hidden, latent_dim=latent, architecture_type=arch)
        assert 0 < step <= reserved, f"{arch}: a step holds {step / 2**20:.0f} MiB, {reserved / 2**20:.0f} MiB reserved"

        production = sae_buffer_budget(
            20 * 1024**3, num_keys=1, hidden_dim=2048, latent_dim=16384, batch_size=4096,
            architecture_type="jumprelu",
        )
        assert production["training_overhead"] >= 2816 * 2**20, production

    def test_both_calls_size_the_step_for_the_architecture(self):
        import ast
        from pathlib import Path

        from src.workers import training_tasks

        tree = ast.parse(Path(training_tasks.__file__).read_text())
        task = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_sae_task")
        calls = [
            call for call in ast.walk(task)
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "sae_buffer_budget"
        ]
        archs = [
            [n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name)]
            for call in calls for kw in call.keywords if kw.arg == "architecture_type"
        ]
        assert len(calls) == 2 and archs == [["architecture_type"], ["architecture_type"]], archs

    def test_both_paths_reserve_the_held_out_chunk_and_the_fly_reserves_a_forward(self):
        import ast
        from pathlib import Path

        from src.workers import training_tasks

        tree = ast.parse(Path(training_tasks.__file__).read_text())
        task = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_sae_task")
        reserved = [
            {n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name)}
            for call in ast.walk(task)
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "sae_buffer_budget"
            for kw in call.keywords if kw.arg == "reserved_bytes"
        ]
        assert len(reserved) == 2 and all("holdout_eval_bytes" in names for names in reserved), reserved
        assert any("forward_reserve" in names for names in reserved), reserved

    def test_the_evaluation_knobs_survive_the_schema_dump(self):
        from src.schemas.training import TrainingHyperparameters

        dumped = TrainingHyperparameters(
            hidden_dim=8, latent_dim=16, batch_size=64, total_steps=10, learning_rate=1e-3,
            architecture_type="jumprelu", holdout_eval_tokens=500, holdout_eval_chunk_tokens=64,
        ).model_dump()
        assert (dumped["holdout_eval_tokens"], dumped["holdout_eval_chunk_tokens"]) == (500, 64)
