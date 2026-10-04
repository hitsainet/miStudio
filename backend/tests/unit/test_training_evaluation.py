"""The post-run evaluation: unseen blocks, a recorded result, and a failure that never fails a training.

SAE TRAINING REMEDIATION, ITEM 6 (2026-09-15). What this file pins:

* SELECTION. Extraction reads the first ``max_samples`` rows of its tokenization, so
  rows at or above that bound are the only rows an SAE trained on it never saw.
  ``select_unseen_rows`` must never return a row below it, whatever the budget,
  weights, widths or seed — checked over randomised configurations, not one case.
* THE MIXTURE. The token budget splits by the training's weights; a source with no
  unseen rows gives nothing and the others take its share.
* RECORDING. ``run_evaluation`` writes one document to ``trainings.evaluation``:
  ``skipped`` with a reason when there is nothing to do, ``failed`` with the
  exception when anything raises — including the model load and the sources —
  and ``completed`` with the numbers. It never raises and never stores NaN.
* THE EXPORT. The re-run loads SAEs from ``community_format/`` strictly: a weight
  the architecture does not have, or one missing, refuses to load rather than
  evaluating a partly random SAE.
* THE WIRING. ``train_sae_task`` calls the step after the community export and
  inside a handler that does not re-raise; the re-run task records every refusal.

MUTATION CONTROLS (2026-09-15; each applied alone, the named files run, source restored
and verified by sha256):
  U1 unseen rows drawn from row 0, not first_unseen            RED  randomised_configurations, rows_it_reads_are_rows..., rows_processed...
  U2 num_samples_processed floor ignored                       RED  the_larger_bound_wins..., rows_processed_beyond_max_samples...
  U3 unseen count ignores rows read                            RED  8 tests (selection, recording, budget)
  U4 candidate rows replaced by their indices                  RED  test_candidate_rows_are_the_only_rows_taken
  U5 mixture weights ignored                                   RED  test_weights_split_the_budget
  U6 max_samples 0 read as "no rows read"                      RED  zero_max_samples_means_every_row, ...read_every_row_leaves_nothing_unseen
  R1 run_evaluation re-raises instead of recording             RED  model_that_will_not_load..., sources_that_cannot_be_read..., measurement_that_raises...
  R2 a failure recorded as completed                           RED  the same three
  R3 the document is never stored on the row                  RED  completed..., model_that_will_not_load..., disabled_is_skipped...
  R4 a result recorded without measuring                       RED  completed..., measurement_that_raises..., non_residual_and_transcoder...
  R5 the loaded model is never released                        RED  test_a_measurement_that_raises_is_recorded_and_the_model_is_released
  R6 the loader accepts weights of another architecture        RED  test_an_export_from_another_architecture_refuses_to_load
  X1 the TopK export drops b_pre again (community_format)      RED  test_an_exported_sae_loads_back_as_the_trained_one[topk]
  P1 train_sae never calls run_post_run_evaluation             RED  all three TestTheTrainingTaskRunsTheStep call tests + test_wiring_reachable
  P2 the post-run handler re-raises                            RED  test_it_is_called_inside_a_handler_that_does_not_re_raise
  P3 extractions=None                                          RED  test_the_step_passes_the_extractions_and_the_placement (AST only; the
                                                                    behavioural guard is test_post_run_evaluation_receives_the_extractions.py)
  P4 on-the-fly model reloaded instead of kept                 RED  test_training_gpu_placement::test_the_evaluation_inputs_go_to_the_embedding_card
  P5 loader ignores the placement (device_map="auto")          RED  test_training_gpu_placement::test_the_evaluation_loader_uses_...placed_card
  P6 `except JobHandoff: raise` deleted                        SURVIVED — an EQUIVALENT mutant: JobHandoff is a BaseException, so
                                                                    `except GpuPlacementError` could never catch it and behaviour is identical
  P6b the handler swallows the hand-off and records a failure  RED  TestTheRerunTask::test_a_hand_off_is_not_a_failure
"""

import ast
import inspect
import json
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.services import training_evaluation as te
from src.services.training_evaluation import EvalSource, rows_read_by_extraction, select_unseen_rows


# ── selection ────────────────────────────────────────────────────────────────


class TestRowsReadByExtraction:
    def test_max_samples_bounds_the_rows_read(self):
        assert rows_read_by_extraction({"max_samples": 100, "num_samples_processed": 100}, 1000) == 100

    def test_a_tokenization_shorter_than_max_samples_was_read_whole(self):
        assert rows_read_by_extraction({"max_samples": 5000, "num_samples_processed": 300}, 300) == 300

    def test_zero_max_samples_means_every_row(self):
        assert rows_read_by_extraction({"max_samples": 0}, 777) == 777

    def test_the_larger_bound_wins_when_the_record_disagrees(self):
        """Only ever shrinks the unseen pool; never lets a read row into it."""
        assert rows_read_by_extraction({"max_samples": 100, "num_samples_processed": 150}, 1000) == 150


class TestUnseenRowsNeverOverlapWhatWasRead:
    def test_randomised_configurations(self):
        rng = np.random.default_rng(0)
        for trial in range(300):
            n = int(rng.integers(1, 4))
            totals = [int(rng.integers(0, 60)) for _ in range(n)]
            reads = [int(rng.integers(1, t + 3)) for t in totals]
            widths = [int(rng.integers(1, 9)) for _ in range(n)]
            sources = [
                EvalSource(label=str(i), dataset_path="x", rows_read=r, weight=float(rng.random() + 0.01))
                for i, r in enumerate(reads)
            ]
            chosen = select_unseen_rows(sources, totals, widths, int(rng.integers(0, 500)), seed=trial)
            for rows, total, read in zip(chosen, totals, reads):
                assert all(read <= row < total for row in rows), (trial, rows, total, read)
                assert len(rows) == len(set(rows))
                assert rows == sorted(rows)

    def test_candidate_rows_are_the_only_rows_taken(self):
        held_out = (3, 9, 11, 40)
        source = EvalSource(label="otf", dataset_path="x", candidate_rows=held_out)
        [rows] = select_unseen_rows([source], [100], [1], 1000, seed=1)
        assert set(rows) == set(held_out)

    def test_a_run_with_no_named_rows_on_the_fly_has_nothing_unseen(self):
        source = EvalSource(label="otf", dataset_path="x", rows_read=None)
        assert select_unseen_rows([source], [100], [4], 1000, seed=0) == [[]]

    def test_an_extraction_that_read_every_row_leaves_nothing_unseen(self):
        """max_samples 0 is "every row" to the extractor, not "no rows"."""
        source = EvalSource(label="all", dataset_path="x", rows_read=0)
        assert select_unseen_rows([source], [100], [4], 1000, seed=0) == [[]]

    def test_rows_processed_beyond_max_samples_are_not_unseen(self):
        source = EvalSource(label="a", dataset_path="x", rows_read=10, rows_processed=30)
        [rows] = select_unseen_rows([source], [100], [1], 1000, seed=0)
        assert rows and min(rows) >= 30

    def test_the_selection_is_reproducible_and_seeded(self):
        source = EvalSource(label="a", dataset_path="x", rows_read=10)
        first = select_unseen_rows([source], [10_000], [8], 800, seed=5)
        assert first == select_unseen_rows([source], [10_000], [8], 800, seed=5)
        assert first != select_unseen_rows([source], [10_000], [8], 800, seed=6)


class TestTheBudgetFollowsTheMixture:
    def test_weights_split_the_budget(self):
        sources = [
            EvalSource(label="code", dataset_path="x", rows_read=1, weight=3.0),
            EvalSource(label="chat", dataset_path="x", rows_read=1, weight=1.0),
        ]
        code, chat = select_unseen_rows(sources, [10_001, 10_001], [10, 10], 4_000, seed=0)
        assert len(code) == 300 and len(chat) == 100

    def test_a_source_with_nothing_unseen_gives_its_share_to_the_others(self):
        sources = [
            EvalSource(label="exhausted", dataset_path="x", rows_read=50, weight=9.0),
            EvalSource(label="fresh", dataset_path="x", rows_read=1, weight=1.0),
        ]
        exhausted, fresh = select_unseen_rows(sources, [50, 10_001], [10, 10], 1_000, seed=0)
        assert exhausted == [] and len(fresh) == 100

    def test_the_budget_is_in_tokens_not_rows(self):
        sources = [EvalSource(label="wide", dataset_path="x", rows_read=1)]
        [rows] = select_unseen_rows(sources, [10_001], [2048], 131_072, seed=0)
        assert len(rows) == 64


class TestSourcesFromExtractions:
    def _extraction(self, tmp_path, name, max_samples, processed):
        out = tmp_path / name
        out.mkdir()
        (out / "metadata.json").write_text(json.dumps({
            "dataset_path": f"datasets/{name}_tok", "max_samples": max_samples,
            "num_samples_processed": processed,
        }))
        return SimpleNamespace(id=f"ext_m_{name}", output_path=str(out))

    def test_each_source_reads_its_extractions_metadata(self, tmp_path):
        exts = [self._extraction(tmp_path, "a", 4300, 4300), self._extraction(tmp_path, "b", 7000, 7000)]
        sources = te.sources_from_extractions(exts, lambda p: Path("/data") / p if not str(p).startswith("/") else Path(p))
        assert [s.label for s in sources] == ["ext_m_a", "ext_m_b"]
        assert [s.rows_read for s in sources] == [4300, 7000]
        assert [s.rows_processed for s in sources] == [4300, 7000]
        assert sources[0].dataset_path == "/data/datasets/a_tok"
        # With no stated mixture, the share the SAE actually trained on.
        assert [s.weight for s in sources] == [4300.0, 7000.0]

    def test_stated_weights_are_used_one_to_one(self, tmp_path):
        exts = [self._extraction(tmp_path, "a", 10, 10), self._extraction(tmp_path, "b", 10, 10)]
        sources = te.sources_from_extractions(exts, Path, dataset_weights=[0.9, 0.1])
        assert [s.weight for s in sources] == [0.9, 0.1]
        with pytest.raises(ValueError):
            te.sources_from_extractions(exts, Path, dataset_weights=[1.0])


# ── recording ────────────────────────────────────────────────────────────────


class _Row(SimpleNamespace):
    pass


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *conditions):
        return self

    def first(self):
        return self.rows[0] if self.rows else None


class _Db:
    def __init__(self, row):
        self.row = row
        self.writes = []

    def query(self, model):
        return _Query([self.row] if self.row is not None else [])

    def commit(self):
        self.writes.append(dict(self.row.evaluation))


def _get_db(db):
    @contextmanager
    def get_db():
        yield db

    return get_db


def _tokenization(tmp_path, rows=10, width=8, name="tok"):
    from datasets import Dataset as HFDataset

    path = tmp_path / name
    HFDataset.from_dict({
        "input_ids": [[(r * 7 + c * 3) % 63 + 1 for c in range(width)] for r in range(rows)],
        "attention_mask": [[1] * width for _ in range(rows)],
    }).save_to_disk(str(path))
    return str(path)


def _llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2,
    )).eval()


def _jumprelu():
    from src.ml.sparse_autoencoder import create_sae

    torch.manual_seed(1)
    return create_sae("jumprelu", hidden_dim=32, latent_dim=64)


class _RecordingOpen:
    """load_from_disk, remembering every row the evaluation selects."""

    def __init__(self):
        self.selected = []

    def __call__(self, path):
        from datasets import load_from_disk

        dataset = load_from_disk(path)
        recorder = self

        class _Wrapped:
            column_names = dataset.column_names

            def __len__(self):
                return len(dataset)

            def select(self, rows):
                rows = list(rows)
                recorder.selected.extend(rows)
                return dataset.select(rows)

            def __getitem__(self, index):
                return dataset[index]

        return _Wrapped()


def _run(tmp_path, *, hp=None, sources=None, load=None, saes=None, open_dataset=None, **kwargs):
    row = _Row(id="t1", evaluation=None)
    db = _Db(row)
    if sources is None:
        sources = [EvalSource(label="ext", dataset_path=_tokenization(tmp_path), rows_read=4)]
    loads = []

    def default_load():
        loads.append(True)
        return _llama(), None

    document = te.run_evaluation(
        get_db=_get_db(db), training_id="t1", hp=hp if hp is not None else {"seed": 3},
        saes=saes if saes is not None else {(1, "residual"): _jumprelu()},
        sources=sources, load_base_model=load or default_load, trigger="post_run",
        open_dataset=open_dataset, **kwargs,
    )
    return document, row, db, loads


class TestRunEvaluationRecordsEveryOutcome:
    def test_a_completed_evaluation_is_stored_with_every_number(self, tmp_path):
        opener = _RecordingOpen()
        document, row, db, loads = _run(tmp_path, token_budget=24, open_dataset=opener)

        assert document["status"] == "completed", document.get("reason")
        assert row.evaluation == document
        # Running first, completed last, and nothing but running between them: the
        # heartbeat rewrites the running record (review R1-C), and a reaper judges
        # a running record by its last write.
        statuses = [w["status"] for w in db.writes]
        assert statuses[0] == "running" and statuses[-1] == "completed"
        assert set(statuses[1:-1]) <= {"running"} and statuses.count("completed") == 1
        [layer] = document["layers"]
        for key in ("ce_spliced", "ce_mean_ablated", "ce_zero_ablated", "ce_delta",
                    "loss_recovered_vs_mean", "loss_recovered_vs_zero", "kl", "l0", "fvu_centred"):
            assert isinstance(layer[key], float), key
        assert layer["layer"] == 1 and layer["hook_type"] == "residual"
        assert document["all_layers_spliced"]["layers"] == [1]
        assert document["ce_base"] is not None and "floor" in document["zero_ablation_note"]
        assert document["trigger"] == "post_run" and document["hook_point"] == "resid_post"
        [source] = document["sources"]
        assert source["rows_read_by_training"] == 4 and source["blocks_evaluated"] == 3
        json.dumps(document, allow_nan=False)
        assert len(loads) == 1

    def test_the_rows_it_reads_are_rows_the_training_never_read(self, tmp_path):
        opener = _RecordingOpen()
        document, *_ = _run(tmp_path, token_budget=10_000, open_dataset=opener)
        assert document["status"] == "completed"
        assert opener.selected, "nothing was read"
        assert min(opener.selected) >= 4, opener.selected
        assert sorted(set(opener.selected)) == list(range(4, 10))

    def test_disabled_is_skipped_without_loading_a_model(self, tmp_path):
        document, row, _db, loads = _run(tmp_path, hp={"evaluate_ce_delta": False})
        assert document["status"] == "skipped" and "evaluate_ce_delta" in document["reason"]
        assert row.evaluation["status"] == "skipped" and loads == []

    def test_a_zero_budget_is_skipped(self, tmp_path):
        document, _row, _db, loads = _run(tmp_path, hp={"evaluation_token_budget": 0})
        assert document["status"] == "skipped" and loads == []

    def test_nothing_unseen_is_skipped_with_the_reason_and_no_model_load(self, tmp_path):
        sources = [EvalSource(label="ext", dataset_path=_tokenization(tmp_path), rows_read=10)]
        document, _row, _db, loads = _run(tmp_path, sources=sources)
        assert document["status"] == "skipped" and "never read" in document["reason"]
        assert document["sources"][0]["unseen_rows"] == 0
        assert loads == []

    def test_a_model_that_will_not_load_is_recorded_as_failed_and_does_not_raise(self, tmp_path):
        def load():
            raise RuntimeError("CUDA out of memory")

        document, row, _db, _loads = _run(tmp_path, load=load)
        assert document["status"] == "failed" and "CUDA out of memory" in document["reason"]
        assert row.evaluation["status"] == "failed"

    def test_sources_that_cannot_be_read_are_recorded_as_failed(self, tmp_path):
        def sources():
            raise FileNotFoundError("metadata.json")

        document, _row, _db, loads = _run(tmp_path, sources=sources)
        assert document["status"] == "failed" and "metadata.json" in document["reason"]
        assert loads == []

    def test_a_measurement_that_raises_is_recorded_and_the_model_is_released(self, tmp_path, monkeypatch):
        released = []

        def load():
            return _llama(), lambda: released.append(True)

        def explode(*args, **kwargs):
            raise ValueError("boom in the measurement")

        monkeypatch.setattr(te, "evaluate_spliced_layers", explode)
        document, *_ = _run(tmp_path, load=load)
        assert document["status"] == "failed" and "boom" in document["reason"]
        assert released == [True]

    def test_non_residual_and_transcoder_saes_are_listed_as_skipped(self, tmp_path):
        from src.ml.sparse_autoencoder import create_sae

        saes = {(1, "residual"): _jumprelu(), (1, "mlp"): _jumprelu(),
                (2, "residual"): create_sae("transcoder", hidden_dim=32, latent_dim=64)}
        document, *_ = _run(tmp_path, saes=saes, token_budget=16)
        assert document["status"] == "completed"
        assert [e["layer"] for e in document["layers"]] == [1]
        assert {(s["layer"], s["hook_type"]) for s in document["skipped_saes"]} == {(1, "mlp"), (2, "residual")}

    def test_a_deleted_training_does_not_make_it_raise(self, tmp_path):
        db = _Db(None)
        document = te.run_evaluation(
            get_db=_get_db(db), training_id="gone", hp={}, saes={(1, "residual"): _jumprelu()},
            sources=[], load_base_model=lambda: (_llama(), None), trigger="rerun",
        )
        assert document["status"] == "skipped"

    def test_evaluate_ce_delta_false_skips_the_AUTOMATIC_evaluation(self, tmp_path):
        """The flag still governs the evaluation that follows a run, untouched."""
        document, *_ = _run(tmp_path, hp={"seed": 3, "evaluate_ce_delta": False}, token_budget=16)

        assert document["status"] == "skipped"
        assert "evaluate_ce_delta" in document["reason"]

    def test_an_EXPLICIT_rerun_runs_even_with_evaluate_ce_delta_false(self, tmp_path):
        """The operator asked for THIS evaluation (user decision, 2026-09-16).

        Before this, POST /trainings/{id}/evaluate on such a run answered 202, took a
        GPU lease, released it a second later and recorded "skipped" with the reason
        buried in the JSONB — a button that quietly did nothing. The flag is meant for
        the automatic post-run evaluation; an explicit re-run overrides it.
        """
        row = _Row(id="t1", evaluation=None)
        db = _Db(row)
        document = te.run_evaluation(
            get_db=_get_db(db), training_id="t1",
            hp={"seed": 3, "evaluate_ce_delta": False},
            saes={(1, "residual"): _jumprelu()},
            sources=[EvalSource(label="ext", dataset_path=_tokenization(tmp_path), rows_read=4)],
            load_base_model=lambda: (_llama(), None),
            trigger="rerun",
            token_budget=16,
        )

        assert document["status"] == "completed", document.get("reason")
        assert document["trigger"] == "rerun"
        assert document["layers"], "the explicit run produced no layer results"


# ── the export ───────────────────────────────────────────────────────────────


EXPORTABLE = [
    pytest.param({"architecture_type": "standard"}, id="standard"),
    pytest.param({"architecture_type": "standard_anthropic"}, id="anthropic"),
    pytest.param({"architecture_type": "skip"}, id="skip"),
    pytest.param({"architecture_type": "topk", "top_k": 4}, id="topk"),
    pytest.param({"architecture_type": "jumprelu"}, id="jumprelu"),
]


def _export(tmp_path, hp):
    from src.services.checkpoint_service import CheckpointService

    torch.manual_seed(7)
    model = te.build_sae_for_training(hp, 16, 48)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.rand_like(p) * 0.5 + 0.01)
    CheckpointService.save_community_checkpoint(
        model=model, output_dir=str(tmp_path / "layer_3_residual"), model_name="org/tiny",
        layer=3, hyperparams={**hp, "hidden_dim": 16, "latent_dim": 48}, training_id="t1",
    )
    return model


@pytest.mark.parametrize("hp", EXPORTABLE)
def test_an_exported_sae_loads_back_as_the_trained_one(tmp_path, hp):
    trained = _export(tmp_path, hp)
    loaded = te.load_exported_sae(hp, te.exported_sae_dir(tmp_path, 3, "residual"))
    x = torch.randn(10, 16)
    torch.testing.assert_close(loaded(x, return_loss=False)[0], trained(x, return_loss=False)[0])


def test_an_export_from_another_architecture_refuses_to_load(tmp_path):
    """Hyperparameters that disagree with the weights must not evaluate a random SAE."""
    _export(tmp_path, {"architecture_type": "jumprelu"})
    with pytest.raises(ValueError, match="does not have"):
        te.load_exported_sae({"architecture_type": "standard"}, tmp_path / "layer_3_residual")


def test_a_missing_weight_refuses_to_load(tmp_path):
    """Strict: a parameter left at its random initialisation is a different SAE."""
    from safetensors.torch import load_file, save_file

    hp = {"architecture_type": "jumprelu"}
    _export(tmp_path, hp)
    weights_file = tmp_path / "layer_3_residual" / "sae_weights.safetensors"
    weights = load_file(str(weights_file))
    weights.pop("b_enc")
    save_file(weights, str(weights_file))
    with pytest.raises(RuntimeError, match="b_enc"):
        te.load_exported_sae(hp, tmp_path / "layer_3_residual")


# ── the wiring ───────────────────────────────────────────────────────────────


def _function(module, name):
    tree = ast.parse(inspect.getsource(module))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(node, name):
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and (
            getattr(n.func, "attr", None) == name or getattr(n.func, "id", None) == name
        )
    ]


class TestTheTrainingTaskRunsTheStep:
    def test_train_sae_task_calls_it_once_after_the_export(self):
        from src.workers import training_tasks

        body = _function(training_tasks, "train_sae_task")
        [call] = _calls(body, "run_post_run_evaluation")
        [export] = _calls(body, "save_multilayer_community_checkpoint")
        assert call.lineno > export.lineno, "the evaluation runs before the SAEs are saved"

    def test_it_is_called_inside_a_handler_that_does_not_re_raise(self):
        from src.workers import training_tasks

        body = _function(training_tasks, "train_sae_task")
        guarded = [
            node for node in ast.walk(body)
            if isinstance(node, ast.Try) and _calls(ast.Module(body=node.body, type_ignores=[]),
                                                   "run_post_run_evaluation")
        ]
        assert guarded, "the call is not inside a try"
        # The INNERMOST try: the whole task sits in an outer try that re-raises
        # after marking the training failed, which is exactly what must not happen here.
        innermost = max(guarded, key=lambda node: node.lineno)
        assert innermost.handlers, "the innermost try around the call has no handler"
        for handler in innermost.handlers:
            assert not any(isinstance(n, ast.Raise) for n in ast.walk(handler)), (
                "a failed evaluation would fail a finished training"
            )

    def test_the_step_passes_the_extractions_and_the_placement(self):
        from src.workers import training_tasks

        body = _function(training_tasks, "train_sae_task")
        [call] = _calls(body, "run_post_run_evaluation")
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert {"models", "placement", "base_model", "extractions", "hp", "sae_mb"} <= set(keywords)
        assert ast.unparse(keywords["placement"]) == "placement"
        assert "extractions" in ast.unparse(keywords["extractions"])

    def test_the_step_records_a_post_run_trigger(self):
        from src.workers import training_tasks

        step = _function(training_tasks, "run_post_run_evaluation")
        [call] = _calls(step, "run_evaluation")
        trigger = next(kw.value for kw in call.keywords if kw.arg == "trigger")
        assert isinstance(trigger, ast.Constant) and trigger.value == "post_run"


# ── the re-run task ──────────────────────────────────────────────────────────


@pytest.fixture
def rerun(monkeypatch, tmp_path):
    from src.models.activation_extraction import ActivationExtraction
    from src.models.model import Model
    from src.models.training import Training
    from src.services.gpu_placement import Placement
    from src.workers import base_task
    from src.workers import training_evaluation_tasks as task_module

    training = SimpleNamespace(
        id="t1", status="completed", model_id="m_x", extraction_id=None, extraction_ids=["ext_1"],
        hyperparameters={"architecture_type": "jumprelu", "training_layers": [3], "hook_types": ["residual"]},
        evaluation=None,
    )
    rows = {
        Training: [training],
        Model: [SimpleNamespace(id="m_x", repo_id="org/tiny", quantization="FP16", file_path=None,
                                params_count=1000, architecture_config={})],
        ActivationExtraction: [SimpleNamespace(id="ext_1", output_path=str(tmp_path / "ext"))],
    }

    class _Q:
        def __init__(self, rows):
            self.rows = rows

        def filter(self, *a):
            return self

        def first(self):
            return self.rows[0] if self.rows else None

    class _S:
        def query(self, model):
            return _Q(rows.get(model, []))

        def commit(self):
            pass

    @contextmanager
    def get_sync_db():
        yield _S()

    state = SimpleNamespace(training=training, rows=rows, placements=[], evaluations=[])

    def place(requested, required_mb=None, allow_shard=False):
        state.placements.append((requested, allow_shard))
        if isinstance(getattr(state, "refusal", None), BaseException):
            raise state.refusal
        return Placement(card=None, device=torch.device("cpu"))

    def run_evaluation(**kwargs):
        state.evaluations.append(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(base_task, "get_sync_db", get_sync_db)
    monkeypatch.setattr(task_module, "place_job", place)
    monkeypatch.setattr(task_module, "run_evaluation", run_evaluation)
    monkeypatch.setattr(task_module.settings, "data_dir", tmp_path)
    _export(tmp_path / "trainings" / "t1" / "community_format", {"architecture_type": "jumprelu"})
    state.run = lambda **kw: task_module.evaluate_training_task.run(training_id="t1", **kw)
    return state


class TestTheRerunTask:
    def test_it_evaluates_the_exported_saes_on_the_placement(self, rerun):
        result = rerun.run(gpu_request="auto", token_budget=4096)

        assert result["status"] == "completed"
        assert rerun.placements == [("auto", True)]
        [call] = rerun.evaluations
        assert call["trigger"] == "rerun" and call["token_budget"] == 4096
        assert list(call["saes"]) == [(3, "residual")]
        assert type(call["saes"][(3, "residual")]).__name__ == "JumpReLUSAE"

    def test_a_training_that_is_not_completed_is_recorded_as_failed(self, rerun):
        rerun.training.status = "running"
        result = rerun.run()
        assert result["status"] == "failed" and rerun.evaluations == [] and rerun.placements == []

    def test_a_missing_export_is_recorded_as_failed(self, rerun, tmp_path):
        import shutil

        shutil.rmtree(tmp_path / "trainings" / "t1" / "community_format")
        result = rerun.run()
        assert result["status"] == "failed" and "Community Standard export" in result["reason"]

    def test_a_placement_refusal_is_recorded_as_failed(self, rerun):
        from src.services.gpu_placement import GpuPlacementError

        rerun.refusal = GpuPlacementError("no card has room", requested="auto")
        result = rerun.run()
        assert result["status"] == "failed" and "no card has room" in result["reason"]

    def test_a_hand_off_is_not_a_failure(self, rerun):
        from src.services.gpu_job_claim import JobHandoff

        rerun.refusal = JobHandoff(queue="gpu.auto", reason="busy")
        with pytest.raises(JobHandoff):
            rerun.run()
        assert rerun.evaluations == []


# ── the rebuilt SAE is the trained SAE ───────────────────────────────────────


#: Keywords that change what the SAE's forward pass computes. The rest (penalty
#: coefficients, STE bandwidths) matter only to training.
FORWARD_KEYWORDS = ("architecture_type", "normalize_activations", "top_k", "top_k_sparsity")


def _create_sae_keywords(module, function_name):
    tree = ast.parse(inspect.getsource(module))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function_name)
    [call] = [n for n in ast.walk(function)
              if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "create_sae"]
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


def test_the_rebuilt_sae_is_constructed_like_the_trained_one():
    """REVIEW R1-C C21. `build_sae_for_training` could default `normalize_activations` to
    "none" with the whole suite green: the export round trip builds BOTH the trained and
    the reloaded SAE through it, so they agree by construction, while a rebuild that
    normalises differently from the training feeds the model a reconstruction at the
    wrong scale and reports its CE as the SAE's cost.

    The authority is the training task's own `create_sae` call. Every keyword that
    changes the forward pass must be the same expression in both calls.

    NEGATIVE CONTROL: `normalize_activations="none"` in build_sae_for_training -> RED here.
    """
    from src.workers import training_tasks

    trained = _create_sae_keywords(training_tasks, "train_sae_task")
    rebuilt = _create_sae_keywords(te, "build_sae_for_training")
    for key in FORWARD_KEYWORDS:
        assert key in trained, f"the training task no longer passes {key}; update FORWARD_KEYWORDS"
        assert rebuilt.get(key) == trained[key], key


@pytest.mark.parametrize("architecture", ["standard", "standard_anthropic", "skip", "topk", "jumprelu"])
def test_an_unstated_normalisation_is_the_frameworks_default(architecture):
    """The behaviour behind the expression above, for a run whose hyperparameters omit it."""
    from src.core.framework_defaults import get_framework_defaults

    mapped = "standard_saelens" if architecture == "standard" else architecture
    expected = get_framework_defaults(mapped)["normalize_activations"]
    if architecture == "standard_anthropic":
        expected = "anthropic_rescale"
    sae = te.build_sae_for_training({"architecture_type": architecture, "top_k": 4}, 16, 48)
    assert sae.normalize_activations == expected
