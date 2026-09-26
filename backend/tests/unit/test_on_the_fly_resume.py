"""An on-the-fly training resumes exactly where it stopped, and fills its first buffer once.

REVIEW R1-B (2026-09-15). The resumed source was built with the constructor's default
prefill and then replayed in `load_state_dict`: a full refill of base-model forwards,
thrown away on every resume. The source is now built with `prefill=False` whenever the
checkpoint carries its state, and this test pins that the resumed run fills exactly once
before its first batch and serves exactly the batches an uninterrupted run served.

NOT COVERED HERE: a resume whose memory reading differs from the interrupted run's. The
buffer plan is re-derived at set-up and `load_state_dict` refuses changed quotas — R1D-2
(and R1D-1 on the rolling buffer), reproduced in R1-D's `test_r1d_seam_reproductions.py`
and fixed by the integrator by saving the storage plan in `training_state.pt`. The RAM
reading here is the same for both processes.

WHAT IS REAL: the task, a tiny transformers Llama read through `LayerCapture`, the source, the
SAEs, their optimizers, the checkpoint files and `resume_training_task`. Faked: the database,
placement, model/dataset loading, and WebSocket emits.

MUTATION CONTROL (R1-B, restored by sha256):
  C2  the source is prefilled on resume (prefill=True)
        -> two refills before the first resumed batch
"""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from src.models.checkpoint import Checkpoint
from src.models.dataset import Dataset
from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from src.models.model import Model
from src.models.training import Training
from src.services import model_activation_source as MAS
from src.services.gpu_placement import Placement
from src.workers import training_tasks
from tests.unit.test_training_data_path_e2e import HIDDEN, KEYS, LAYERS, VOCAB, WIDTH, _tiny_llama, _tokenized
from tests.unit.test_training_resume_equivalence import _Session

_REAL_CREATE_SAE = training_tasks.create_sae
_REAL_STOP = training_tasks.stop_signal_for
_REAL_DRAW = training_tasks.draw_cached_batch
_REAL_REFILL = MAS.ModelActivationSource.refill
_REAL_INIT = MAS.ModelActivationSource.__init__

#: 50,000 tokens per layer — the smallest buffer the storage plan accepts — so the ~96,000
#: training tokens are CYCLED: at 1,024 tokens a batch, a refill every 48 steps (0, 48, 96, 144).
RAM_TOKENS = 50_000
BATCH = 1_024
#: Checkpoints at 50, 100 and 150. The pause at 125 saves its own checkpoint at step 124 (review
#: R1-A, finding L8: before that fix a pause saved nothing and the run resumed from 100). Step
#: 124 lies inside the buffer filled at 96, and the resumed steps 125-159 cross the refill at 144.
TOTAL, CHECKPOINT_EVERY, STOP_AT = 160, 50, 125
ROWS = 4_000


class _World:
    """One training row and everything it touches; the recorders wrap the real functions."""

    def __init__(self, monkeypatch, tmp_path):
        from src.core import database
        from src.services import activation_service
        from src.workers import base_task, websocket_emitter

        model = _tiny_llama()
        datasets = {"ds_a": _tokenized(0, ROWS, 1), "ds_b": _tokenized(ROWS, ROWS, 2)}
        self.training = SimpleNamespace(
            id="train_fly_resume", model_id="m_tiny", status="pending", current_step=0, current_loss=None,
            dataset_id="ds_a", dataset_ids=["ds_a", "ds_b"], extraction_id=None, extraction_ids=None,
            gpu_request="auto", gpu_uuid=None, gpu_uuids=None, checkpoint_dir=None,
            error_message=None, error_traceback=None, completed_at=None, progress=0.0,
            hyperparameters={
                "hidden_dim": HIDDEN, "latent_dim": 64, "batch_size": BATCH, "learning_rate": 1e-3,
                "total_steps": TOTAL, "log_interval": 25, "checkpoint_interval": CHECKPOINT_EVERY,
                "seed": 5, "training_layers": LAYERS, "hook_types": ["residual"],
                "architecture_type": "jumprelu", "sparsity_coeff": 1e-3, "evaluate_ce_delta": False,
                "sparsity_warmup_steps": 0, "holdout_fraction": 0.2, "holdout_eval_tokens": 200,
                "dataset_weights": [3.0, 1.0],
            },
        )
        self.store = {
            Training: [self.training],
            Model: [SimpleNamespace(id="m_tiny", repo_id="org/tiny-llama", quantization="FP16", file_path=None,
                                    params_count=None, architecture="llama",
                                    architecture_config={"hidden_size": HIDDEN})],
            Dataset: [SimpleNamespace(id="ds_a"), SimpleNamespace(id="ds_b")],
            DatasetTokenization: [
                SimpleNamespace(dataset_id=name, model_id="m_tiny", status=TokenizationStatus.READY,
                                tokenized_path=f"/tok/{name}", tokenizer_repo_id="org/tiny-llama",
                                vocab_size=VOCAB, max_length=WIDTH)
                for name in datasets
            ],
        }
        session = _Session(self.store)

        @contextmanager
        def get_sync_db():
            yield session

        self.ram = RAM_TOKENS * HIDDEN * 4 * len(KEYS) * 2
        self.stop_at = None
        self.events, self.draws, self.models, self.prefills, self.dispatched = [], [], [], [], []
        memory = {"total_gb": 0.01, "total_mb": 10.0, "fits_in_6gb": True, "available_gpu_gb": 20.0,
                  "per_layer_gb": 0.01, "max_layers_in_6gb": 10}

        def create(**kwargs):
            sae = _REAL_CREATE_SAE(**kwargs)
            self.models.append(sae)
            return sae

        def stop(row, step, lease_lost=None, **kw):
            if self.stop_at is not None and step == self.stop_at:
                return {"status": "paused", "step": step}
            return _REAL_STOP(row, step, lease_lost, **kw)

        def draw(*args, **kwargs):
            batch = _REAL_DRAW(*args, **kwargs)
            self.events.append("draw")
            self.draws.append({k: v.detach().clone() for k, v in batch.items()})
            return batch

        def refill(source):
            self.events.append("refill")
            return _REAL_REFILL(source)

        def init(source, *args, **kwargs):
            self.prefills.append(kwargs.get("prefill", True))
            return _REAL_INIT(source, *args, **kwargs)

        monkeypatch.setattr(base_task, "get_sync_db", get_sync_db)
        monkeypatch.setattr(database, "get_sync_db", get_sync_db)
        monkeypatch.setattr(training_tasks, "estimate_training_memory", lambda **kw: memory)
        monkeypatch.setattr(training_tasks, "estimate_multilayer_training_memory", lambda **kw: memory)
        monkeypatch.setattr(training_tasks, "place_job", lambda *a, **k: Placement(card=None, device=torch.device("cpu")))
        monkeypatch.setattr(training_tasks, "record_progress", lambda *a, **k: True)
        monkeypatch.setattr(
            training_tasks.TrainingValidator, "validate_sparsity_config", staticmethod(lambda hp: ([], []))
        )
        monkeypatch.setattr(training_tasks.settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(activation_service, "_available_memory_bytes", lambda: self.ram)
        monkeypatch.setattr(websocket_emitter, "emit_training_progress", lambda *a, **k: True)
        monkeypatch.setattr(websocket_emitter, "emit_checkpoint_created", lambda *a, **k: True)
        monkeypatch.setattr(
            training_tasks.CheckpointService, "save_multilayer_community_checkpoint", staticmethod(lambda **kw: {})
        )
        monkeypatch.setattr(training_tasks, "select_tokenization_for_model",
                            lambda candidates, model_id, ds_id: next(c for c in candidates if c.dataset_id == ds_id))
        monkeypatch.setattr(training_tasks, "load_from_disk", lambda path: datasets[path.rsplit("/", 1)[-1]])
        monkeypatch.setattr(
            training_tasks, "load_model_from_hf",
            lambda **kw: (model, SimpleNamespace(pad_token_id=0, eos_token_id=1, vocab_size=VOCAB), model.config, {}),
        )
        monkeypatch.setattr(training_tasks, "create_sae", create)
        monkeypatch.setattr(training_tasks, "stop_signal_for", stop)
        monkeypatch.setattr(training_tasks, "draw_cached_batch", draw)
        monkeypatch.setattr(training_tasks, "gpu_delay", lambda task, request: lambda **kw: self.dispatched.append(kw))
        monkeypatch.setattr(MAS.ModelActivationSource, "refill", refill)
        monkeypatch.setattr(MAS.ModelActivationSource, "__init__", init)

    def run(self, **kwargs):
        try:
            return training_tasks.train_sae_task.run(self.training.id, **kwargs)
        finally:
            training_tasks.train_sae_task._close_activation_stream()


def test_an_on_the_fly_run_resumes_exactly_and_fills_its_first_buffer_once(monkeypatch, tmp_path):
    straight = _World(monkeypatch, tmp_path / "straight")
    assert straight.run()["status"] == "completed", straight.training.error_message
    assert len(straight.draws) == TOTAL
    assert straight.events.count("refill") >= 3, "the fixture never cycled the corpus through the buffer"

    resumed = _World(monkeypatch, tmp_path / "resumed")
    resumed.stop_at = STOP_AT
    assert resumed.run()["status"] == "paused"

    resumed.stop_at = None
    training_tasks.resume_training_task.run(resumed.training.id)
    kwargs = resumed.dispatched[-1]
    checkpoint = next(c for c in resumed.store[Checkpoint] if c.id == kwargs["checkpoint_id"])
    start = checkpoint.step + 1
    # The pause's own checkpoint (R1-A L8), not the interval checkpoint at 100.
    assert checkpoint.step == STOP_AT - 1 and kwargs["start_step"] == start
    resumed.events.clear()
    before = len(resumed.draws)

    outcome = resumed.run(start_step=kwargs["start_step"], checkpoint_id=kwargs["checkpoint_id"])
    assert outcome["status"] == "completed", resumed.training.error_message

    # Built without a prefill, the source fills exactly once — the replay — before its first
    # batch, and a later refill falls where the straight run's did.
    assert resumed.prefills[-1] is False, "the resumed source was prefilled and then replayed"
    first_draw = resumed.events.index("draw")
    assert resumed.events[:first_draw] == ["refill"], resumed.events[:5]
    assert "refill" in resumed.events[first_draw:], "the resumed steps never crossed a refill"

    after = resumed.draws[before:]
    assert len(after) == TOTAL - start
    for offset, (a, b) in enumerate(zip(straight.draws[start:], after)):
        for key in KEYS:
            assert torch.equal(a[key], b[key]), f"step {start + offset}: the resumed run was served other tokens"

    for a, b in zip(straight.models[-len(KEYS):], resumed.models[-len(KEYS):]):
        assert a is not b
        for name, value in a.state_dict().items():
            assert torch.equal(value, b.state_dict()[name]), f"{name} differs after resuming"
