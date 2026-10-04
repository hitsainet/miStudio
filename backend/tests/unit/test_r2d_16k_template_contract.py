"""R2-D: the planned 16K run's configuration against this branch's schema and create path (2026-09-15).

Template `6a460fbe-1314-4b2e-b724-de2e032f807b` ("LFM2.5-1.2B · L11-13 residual · JumpReLU 8x",
latent_dim 16,384) as exported from production on 2026-09-14 16:06 UTC
(scratchpad `main/train_and_templates.json`), verbatim. The planned run (memory
`sae-16k-unique-run-constraints`) is 150,000 steps; the stored template says 50,000, so both
are built. Each is created through the real `TrainingCreate` schema and
`TrainingService.create_training` in the lane's own test database, over five residual
extractions of layers 11-13 with a five-way `dataset_weights` mixture.

These PASS: they record that the template is accepted by this branch, and they pin the
checkpoint-size arithmetic the R2-D record's disk budget rests on.
"""

import copy
import uuid
from unittest.mock import patch

import pytest
import torch

from src.models.activation_extraction import ActivationExtraction, ExtractionStatus
from src.models.dataset import Dataset, DatasetStatus
from src.models.model import Model, ModelStatus, QuantizationFormat
from src.schemas.training import TrainingCreate, TrainingHyperparameters
from src.services.training_service import TrainingService

#: The template's stored hyperparameters, verbatim.
TEMPLATE_6A460FBE = {
    "seed": None, "aux_k": None, "top_k": None, "l1_alpha": None, "bandwidth": 0.01, "target_l0": None,
    "batch_size": 2048, "hidden_dim": 2048, "hook_types": ["residual"], "latent_dim": 16384,
    "total_steps": 50000, "adam_epsilon": None, "log_interval": 100, "warmup_steps": 2000,
    "weight_decay": 0.0, "learning_rate": 0.00007, "ste_bandwidth": None, "aux_loss_alpha": None,
    "grad_clip_norm": 1.0, "sparsity_coeff": 0.001, "top_k_sparsity": None, "dataset_weights": None,
    "training_layers": [11, 12, 13], "holdout_fraction": 0.0, "architecture_type": "jumprelu",
    "evaluate_ce_delta": True, "initial_threshold": 0.5, "normalize_decoder": True,
    "resample_interval": 5000, "checkpoint_interval": 2000, "dead_neuron_threshold": 10000,
    "normalize_activations": "constant_norm_rescale", "resample_dead_neurons": True,
    "sparsity_warmup_steps": 10000,
}

#: The mixture planned for the run: chat, OWT, code, Pile, Bloomberg.
PLANNED_WEIGHTS = [0.35, 0.30, 0.15, 0.10, 0.10]


def _planned_150k():
    hp = copy.deepcopy(TEMPLATE_6A460FBE)
    hp.update(total_steps=150_000, lr_decay_steps=30_000, dataset_weights=PLANNED_WEIGHTS)
    return hp


async def _seed(async_session, n_extractions=5):
    model = Model(
        id="m_r2d16k", name="LFM2.5-1.2B-Instruct", repo_id="LiquidAI/LFM2.5-1.2B-Instruct",
        status=ModelStatus.READY.value, quantization=QuantizationFormat.FP16.value,
        architecture="lfm2", params_count=1_170_000_000,
    )
    async_session.add(model)
    dataset_ids, extraction_ids = [], []
    for index in range(n_extractions):
        dataset = Dataset(id=uuid.uuid4(), name=f"corpus-{index}", source="HuggingFace", status=DatasetStatus.READY)
        async_session.add(dataset)
        await async_session.flush()
        extraction_id = f"ext_m_r2d16k_{index}"
        async_session.add(ActivationExtraction(
            id=extraction_id, model_id="m_r2d16k", dataset_id=str(dataset.id), layer_indices=[11, 12, 13],
            hook_types=["residual"], max_samples=50_000, status=ExtractionStatus.COMPLETED, progress=100.0,
        ))
        dataset_ids.append(str(dataset.id))
        extraction_ids.append(extraction_id)
    await async_session.commit()
    return dataset_ids, extraction_ids


@pytest.mark.parametrize(
    "hyperparameters",
    [
        pytest.param(dict(TEMPLATE_6A460FBE, dataset_weights=PLANNED_WEIGHTS), id="as-stored-50k"),
        pytest.param(_planned_150k(), id="planned-150k-with-decay"),
    ],
)
@pytest.mark.asyncio
async def test_the_16k_template_is_created_through_the_real_schema_and_service(async_session, hyperparameters):
    dataset_ids, extraction_ids = await _seed(async_session)
    request = TrainingCreate(
        model_id="m_r2d16k", dataset_ids=dataset_ids, extraction_ids=extraction_ids,
        hyperparameters=TrainingHyperparameters(**hyperparameters),
    )
    # THIS IS A CONTRACT TEST, NOT A CAPACITY TEST. `create_training` now refuses a run
    # whose checkpoints will not fit (services/checkpoint_disk), and the planned 150k
    # variant genuinely does not fit on a developer workstation: 74 checkpoints x ~3.0 GiB
    # is ~223 GiB, against 196 GiB free here. That refusal is correct and is exercised
    # directly in test_checkpoint_disk_guards_the_run.py; letting it decide THIS test would
    # make the template contract depend on how full the machine happens to be.
    with patch("src.services.training_service._emit_training_event_sync"), \
            patch("src.services.checkpoint_disk.free_bytes_for", return_value=8 * 1024 ** 4):
        training = await TrainingService.create_training(async_session, request)

    stored = training.hyperparameters
    assert stored["latent_dim"] == 16384 and stored["training_layers"] == [11, 12, 13]
    assert stored["dataset_weights"] == PLANNED_WEIGHTS
    # Defaults this branch adds, which the stored template does not carry.
    assert stored["evaluation_token_budget"] == 131_072
    assert stored["holdout_eval_tokens"] is None and stored["holdout_eval_chunk_tokens"] is None
    assert stored["lr_decay_steps"] == hyperparameters.get("lr_decay_steps", 0)
    assert stored["resample_dead_neurons"] is True and stored["dead_neuron_threshold"] == 10000


def test_warmup_plus_decay_beyond_the_run_is_refused_at_create():
    hp = _planned_150k()
    hp["lr_decay_steps"] = 148_001  # 2,000 warmup + 148,001 decay > 150,000
    with pytest.raises(ValueError, match="must not exceed total_steps"):
        TrainingHyperparameters(**hp)


def test_training_state_holds_two_adam_moments_per_weight_so_a_checkpoint_step_is_three_times_the_weights(tmp_path):
    """The disk budget's basis, at a small width. Measured at the run's own width on CPU
    (scratchpad r2-d/measure_ckpt.py): 268,575,168 bytes of weights and 537,301,604 bytes of
    training_state.pt per 16K SAE, so 2,417,630,316 bytes per checkpoint step for three layers."""
    from safetensors.torch import save_file

    from src.ml.sparse_autoencoder import create_sae
    from src.services.checkpoint_service import build_training_state, save_training_state

    hidden, latent = 64, 512
    sae = create_sae("jumprelu", hidden, latent, sparsity_coeff=1e-3, bandwidth=0.01,
                     initial_threshold=0.5, normalize_decoder=True)
    optimizer = torch.optim.Adam(sae.parameters(), lr=7e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    for parameter in sae.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    key = (11, "residual")
    state = build_training_state(
        step=2000, sae_keys=[key], optimizers={key: optimizer}, schedulers={key: scheduler}, scalers={},
        dead_latent_trackers={}, activation_ema={key: torch.zeros(latent)},
        firing_rate={key: torch.zeros(latent)}, best_loss=1.0, models={key: sae},
    )
    state_bytes = save_training_state(tmp_path / "step", state)
    weights = tmp_path / "checkpoint.safetensors"
    save_file({f"model.{k}": v.contiguous() for k, v in sae.state_dict().items()}, str(weights))
    parameter_bytes = sum(p.numel() * p.element_size() for p in sae.parameters())

    assert state_bytes >= 2 * parameter_bytes, (state_bytes, parameter_bytes)
    # Small fixed overhead only (RNG state, pickle framing, the two per-latent trackers).
    assert state_bytes - 2 * parameter_bytes < 64 * 1024 + 2 * latent * 4, (state_bytes, parameter_bytes)
    assert weights.stat().st_size >= parameter_bytes
