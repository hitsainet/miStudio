"""G38 end to end: a REAL run with `skip_template_prefix` on leaves the template prefix out of every
stage — render, token capture, training, calibration (global and per-window bars, length arrays),
evaluation, the exported definition's test vectors, the offline score and the memory forecast.

⚠ WHY A WHOLE RUN AND NOT A UNIT TEST PER STAGE. Every stage reads the flag the same way — from the
rendered rows `_prepare_examples` builds — so the load-bearing lines are few and upstream: the run
config read (`context_from_row`), the hand-off to the renderer (`_prepare_examples` → `render_all`)
and the mask itself. A test that renders rows by hand and passes them to one stage would stay green
with the hand-off deleted. This one drives `execute_probe_run` and the real builder, and checks what
each stage WROTE, so removing any of those lines turns it red (controls in the review record).

The fixtures (database, tiny random LM, Arrow corpus) are the pipeline test's own; the tokenizer here
has a Llama-3.1-shaped preamble, so every row begins with a 7-token prefix: P = 7, and every row is
`prefix (7) + 5 words + </turn> + <turn> assistant` = 15 served tokens, 8 of them scored.
"""
from __future__ import annotations

import glob
import os
import uuid

import numpy as np
import pytest

from tests.unit.test_probe_monitor_pipeline import (  # noqa: F401 - fixtures
    D_MODEL,
    N_ROWS,
    arrow_dataset,
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
)

pytest.importorskip("datasets")

P = 7
ROW_TOKENS = 15


@pytest.fixture(scope="module")
def prefix_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<s>", "<turn>", "</turn>", "user", "assistant", "system", "today", "x", "y"]
    words += [f"risk{i}" for i in range(8)] + [f"calm{i}" for i in range(8)]
    words += [f"filler{i}" for i in range(16)]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>")
    tokenizer.chat_template = (
        "<s> <turn> system today </turn> {% for m in messages %}<turn> {{ m['role'] }} "
        "{{ m['content'] }} </turn> {% endfor %}{% if add_generation_prompt %}<turn> assistant {% endif %}"
    )
    tokenizer.pad_token = "<unk>"
    return tokenizer


@pytest.fixture(scope="module")
def flagged(pipeline_db, arrow_dataset, monkeypatch_module, tiny_lm, prefix_tokenizer, tmp_path_factory):
    """One real run with the flag on, a calibration set and an out-of-distribution eval set."""
    from src.core import config as config_module
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun
    from src.services import probe_monitor_run as run_module

    artifacts = tmp_path_factory.mktemp("g38_artifacts")
    monkeypatch_module.setattr(config_module.settings, "data_dir", artifacts, raising=False)

    model_row = Model(
        id=f"m_{uuid.uuid4().hex[:12]}", name="tiny-random-g38", status=ModelStatus.READY,
        file_path=str(arrow_dataset), architecture="LlamaForCausalLM", params_count=1_000,
    )
    dataset_row = Dataset(
        id=uuid.uuid4(), name="probe-corpus-g38", source="local", status=DatasetStatus.READY,
        raw_path=str(arrow_dataset), extra_metadata={},
    )
    pipeline_db.add_all([model_row, dataset_row])
    pipeline_db.commit()
    mapping = {"high": "positive", "low": "negative"}
    train_view = ProbeMonitorDataset(
        name="train", dataset_id=dataset_row.id, input_column="inputs", label_column="stakes",
        label_mapping=mapping, pair_column="pair", role="train", counts={},
    )
    eval_view = ProbeMonitorDataset(
        name="unseen", dataset_id=dataset_row.id, input_column="inputs", label_column="stakes",
        label_mapping=mapping, role="eval", distribution="out_of_distribution", counts={},
    )
    calibration_view = ProbeMonitorDataset(
        name="calibration", dataset_id=dataset_row.id, input_column="inputs", label_column="stakes",
        label_mapping={"low": "negative", "high": "excluded"}, role="calibration", counts={},
        config="default", split="train",
    )
    pipeline_db.add_all([train_view, eval_view, calibration_view])
    pipeline_db.commit()
    run = ProbeMonitorRun(
        model_id=model_row.id, train_dataset_id=train_view.id, eval_dataset_ids=[eval_view.id],
        calibration_dataset_id=calibration_view.id,
        config={
            "layers": [1, 2], "rules": ["max"], "scope": "all", "max_length": 64,
            "val_fraction": 0.25, "seed": 1337, "top_n_layers": 1, "target_fpr": 0.1,
            "calibration_max_rows": 100,
            # What `ProbeRunConfig` stamps on every run submitted from G38 on.
            "skip_template_prefix": True,
        },
        behaviour="pii", behaviour_label="Personal data present", status="pending", environment={},
    )
    pipeline_db.add(run)
    pipeline_db.commit()

    def loader(row):
        row.gpu_request = "auto"
        row.gpu_uuid = "GPU-00000000-1111-2222-3333-444444444444"
        return tiny_lm, prefix_tokenizer, "LlamaForCausalLM"

    run_module.execute_probe_run(pipeline_db, run.id, model_loader=loader)
    pipeline_db.refresh(run)
    return {"run": run, "model": model_row, "eval_view": eval_view, "artifacts": artifacts}


def _probe(pipeline_db, flagged):
    from src.models.probe_monitor import ProbeMonitor

    return (
        pipeline_db.query(ProbeMonitor)
        .filter(ProbeMonitor.run_id == flagged["run"].id, ProbeMonitor.selected.is_(True))
        .first()
    )


class TestARunWithTheFlag:
    def test_it_completes_and_records_the_flag(self, flagged):
        run = flagged["run"]
        assert run.status == "completed", run.error_message if hasattr(run, "error_message") else run.status
        assert run.environment["skip_template_prefix"] is True

    def test_the_render_skipped_the_prefix_of_every_row(self, flagged):
        render = flagged["run"].environment["render"]
        assert render["template_prefix_skipped"] == P * render["rendered"]
        assert render["scored_tokens"] == (ROW_TOKENS - P) * render["rendered"]

    def test_the_forecast_counted_the_skipped_positions(self, flagged):
        forecast = flagged["run"].environment["memory_forecast"]
        assert forecast.get("checked", True) is not False, forecast
        assert forecast["template_prefix_skipped"] == P * flagged["run"].environment["render"]["rendered"]

    def test_the_token_capture_holds_only_the_positions_after_the_prefix(self, flagged):
        from src.services.probe_monitor_run import artifact_dir_for

        rendered = flagged["run"].environment["render"]["rendered"]
        captures = glob.glob(str(artifact_dir_for(flagged["run"].id) / "tokens_layer*"))
        assert captures
        for path in captures:
            element = 4 if path.endswith(".f32") else 2
            size = os.path.getsize(path)
            assert size == rendered * (ROW_TOKENS - P) * D_MODEL * element, path

    def test_calibration_cut_every_window_over_the_served_positions(self, pipeline_db, flagged):
        probe = _probe(pipeline_db, flagged)
        assert probe.threshold_source == "calibration_set"
        lengths = np.load(probe.calibration_lengths_path).reshape(-1).tolist()
        assert lengths and set(lengths) == {ROW_TOKENS - P}
        windows = probe.window_decisions or {}
        assert {"all", "prompt", "last_user"} <= set(windows)
        # `last_user` skips the prefix too: its span starts at the first user header (position 5),
        # inside the 7-token prefix, so `[5, 13)` becomes `[7, 13)`.
        path = probe.calibration_lengths_path.replace("lengths", "lengths_window_last_user")
        assert set(np.load(path).reshape(-1).tolist()) == {ROW_TOKENS - P - 2}
        for window in ("all", "prompt"):
            path = probe.calibration_lengths_path.replace("lengths", f"lengths_window_{windows[window]['scope']}")
            assert set(np.load(path).reshape(-1).tolist()) == {ROW_TOKENS - P}

    def test_evaluation_scored_the_positions_after_the_prefix(self, flagged):
        from src.services.probe_monitor_run import artifact_dir_for

        files = glob.glob(str(artifact_dir_for(flagged["run"].id) / f"*__{flagged['eval_view'].id}__*__n_scored.npy"))
        assert files
        for path in files:
            assert set(np.load(path).reshape(-1).tolist()) == {ROW_TOKENS - P}, path

    def test_the_offline_score_skips_the_prefix(self, pipeline_db, flagged, tiny_lm, prefix_tokenizer):
        from src.services.probe_monitor_run import score_one

        probe = _probe(pipeline_db, flagged)
        result = score_one(
            pipeline_db, probe.id, text="risk1 risk2 filler3 calm4 filler5",
            loaded=(tiny_lm, prefix_tokenizer, "LlamaForCausalLM"),
        )
        assert result["template_prefix_tokens"] == P
        assert result["n_scored"] == ROW_TOKENS - P
        assert [t["scored"] for t in result["tokens"]][:P] == [False] * P


class TestTheExportedDefinition:
    @pytest.fixture(scope="class")
    def built(self, pipeline_db, flagged, tiny_lm, prefix_tokenizer):
        from _pytest.monkeypatch import MonkeyPatch

        from src.models.dataset import Dataset
        from src.services import probe_definition_builder as builder
        from src.services import probe_monitor_run as run_module

        patcher = MonkeyPatch()
        try:
            model_row = flagged["model"]
            model_row.repo_id = "org/tiny-random"
            model_row.architecture_config = {"hidden_size": D_MODEL, "num_hidden_layers": 4}
            for dataset in pipeline_db.query(Dataset).all():
                dataset.hf_repo_id = "org/probe-corpus"
            pipeline_db.commit()
            patcher.setattr(builder, "resolve_model_revision", lambda row: "a" * 40)
            patcher.setattr(builder, "_vector_dtype_or_refuse", lambda run, model: "float32")
            patcher.setattr(run_module, "probe_precision", lambda db, run: {"refusal": None})
            definition, _record = builder.build(
                pipeline_db, _probe(pipeline_db, flagged).id,
                acknowledge_below_rung2={"reason": "CPU pipeline fixture, random model"},
                vector_count=8,
                model_loader=lambda run: (tiny_lm, prefix_tokenizer, "LlamaForCausalLM"),
            )
            yield definition
        finally:
            patcher.undo()

    def test_it_records_the_flag(self, built):
        assert built.aggregation.skip_template_prefix is True

    def test_every_vector_states_its_prefix_and_scores_the_rest(self, built, prefix_tokenizer):
        from src.services.probe_monitor_render import template_prefix_length

        for vector in built.test_vectors.vectors:
            assert vector.template_prefix_tokens == P
            assert template_prefix_length(prefix_tokenizer, vector.token_ids) == P
            assert len(vector.token_scores) == len(vector.token_ids) - P

    def test_the_identity_carries_the_flag(self, built):
        from src.schemas.probe_identity import detector_payload

        payload = detector_payload(built.model_dump(mode="json"))
        assert payload["aggregation"]["skip_template_prefix"] is True
        assert built.identity is not None
