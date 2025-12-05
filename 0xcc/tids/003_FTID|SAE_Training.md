# Technical Implementation Document: SAE Training

**Document ID:** 003_FTID|SAE_Training
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related TDD:** [003_FTDD|SAE_Training](../tdds/003_FTDD|SAE_Training.md)

---

## 1. Implementation Order

### Phase 1: SAE Architectures
1. Base SAE class with abstract methods
2. Standard SAE (L1 sparsity)
3. JumpReLU SAE (L0 penalty)
4. Skip SAE (residual connections)
5. Transcoder SAE (layer-to-layer)

### Phase 2: Training Infrastructure
1. Database migration (trainings, training_metrics, checkpoints)
2. Training service layer
3. Activation extraction utilities
4. Training task with progress tracking

### Phase 3: Real-time Monitoring
1. WebSocket progress emission
2. Metric streaming (loss, L0, FVU)
3. Checkpoint save/load
4. Training control (pause/resume/stop)

### Phase 4: Frontend
1. Training form with hyperparameters
2. Training card with progress
3. Loss curves visualization
4. Checkpoint management

---

## 2. File-by-File Implementation

### 2.1 SAE Architectures

#### `backend/src/ml/sparse_autoencoder.py`
```python
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

class BaseSAE(nn.Module, ABC):
    """Base class for all SAE architectures."""

    def __init__(self, d_in: int, d_sae: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.dtype = dtype

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation."""
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation to reconstruction."""
        pass

    @abstractmethod
    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
        aux: Dict
    ) -> Dict[str, torch.Tensor]:
        """Compute loss components."""
        pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward pass returning reconstruction, latents, and auxiliary info."""
        z = self.encode(x)
        x_hat = self.decode(z)
        aux = {"pre_activation": getattr(self, "_pre_act", None)}
        return x_hat, z, aux


class StandardSAE(BaseSAE):
    """Standard SAE with L1 sparsity penalty."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        l1_coeff: float = 5e-3,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__(d_in, d_sae, dtype)
        self.l1_coeff = l1_coeff

        # Encoder: d_in -> d_sae
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype))

        # Decoder: d_sae -> d_in
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._pre_act = x @ self.W_enc + self.b_enc
        return torch.relu(self._pre_act)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def loss(self, x, x_hat, z, aux) -> Dict[str, torch.Tensor]:
        # Reconstruction loss (MSE)
        recon_loss = (x - x_hat).pow(2).mean()

        # L1 sparsity loss
        l1_loss = z.abs().mean()

        # Total loss
        total_loss = recon_loss + self.l1_coeff * l1_loss

        # Metrics
        l0 = (z > 0).float().sum(dim=-1).mean()  # Average active features
        fvu = 1 - (x - x_hat).var() / x.var()  # Fraction of variance unexplained

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
            "l0": l0,
            "fvu": fvu
        }


class JumpReLUSAE(BaseSAE):
    """JumpReLU SAE with learnable thresholds (Gemma Scope style)."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        l0_target: float = 50.0,
        l0_coeff: float = 1e-3,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__(d_in, d_sae, dtype)
        self.l0_target = l0_target
        self.l0_coeff = l0_coeff

        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype))

        # Learnable thresholds (log space for stability)
        self.log_threshold = nn.Parameter(torch.zeros(d_sae, dtype=dtype))

        # Decoder
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

    @property
    def threshold(self):
        return torch.exp(self.log_threshold)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_act = x @ self.W_enc + self.b_enc
        self._pre_act = pre_act
        # JumpReLU: output is (pre_act - threshold) * (pre_act > threshold)
        return torch.relu(pre_act - self.threshold) * (pre_act > self.threshold).float()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def loss(self, x, x_hat, z, aux) -> Dict[str, torch.Tensor]:
        recon_loss = (x - x_hat).pow(2).mean()

        # L0 penalty (deviation from target)
        l0 = (z > 0).float().sum(dim=-1).mean()
        l0_loss = (l0 - self.l0_target).pow(2)

        total_loss = recon_loss + self.l0_coeff * l0_loss

        fvu = 1 - (x - x_hat).var() / x.var()

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "l0_loss": l0_loss,
            "l0": l0,
            "fvu": fvu
        }


class SkipSAE(StandardSAE):
    """SAE with skip connection (residual)."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        skip_coeff: float = 0.1,
        l1_coeff: float = 5e-3,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__(d_in, d_sae, l1_coeff, dtype)
        self.skip_coeff = skip_coeff

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        z = self.encode(x)
        x_hat_sae = self.decode(z)
        # Add skip connection
        x_hat = (1 - self.skip_coeff) * x_hat_sae + self.skip_coeff * x
        return x_hat, z, {"pre_activation": self._pre_act}


class TranscoderSAE(BaseSAE):
    """Transcoder: maps from one layer's activations to another's."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_sae: int,
        l1_coeff: float = 5e-3,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__(d_in, d_sae, dtype)
        self.d_out = d_out
        self.l1_coeff = l1_coeff

        # Encoder: d_in -> d_sae
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype))

        # Decoder: d_sae -> d_out (different from d_in!)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_out, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_out, dtype=dtype))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._pre_act = x @ self.W_enc + self.b_enc
        return torch.relu(self._pre_act)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def loss(self, x_in, x_out_target, z, aux) -> Dict[str, torch.Tensor]:
        """Note: x_in is input, x_out_target is target output layer activation."""
        x_hat = self.decode(z)
        recon_loss = (x_out_target - x_hat).pow(2).mean()
        l1_loss = z.abs().mean()

        return {
            "loss": recon_loss + self.l1_coeff * l1_loss,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
            "l0": (z > 0).float().sum(dim=-1).mean()
        }
```

**Key Implementation Notes:**
- All SAEs share common interface via `BaseSAE`
- Decoder weights normalized to unit norm (important for interpretability)
- JumpReLU uses log-space for threshold stability
- Transcoder has different input/output dimensions

#### `backend/src/ml/activation_extraction.py`
```python
import torch
from typing import List, Dict, Callable
from transformers import AutoModelForCausalLM
from functools import partial

class ActivationExtractor:
    """Extract activations from transformer model layers."""

    def __init__(self, model: AutoModelForCausalLM, device: str = "cuda"):
        self.model = model
        self.device = device
        self.hooks = []
        self.activations = {}

    def _create_hook(self, name: str) -> Callable:
        """Create hook function that stores activation."""
        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            self.activations[name] = activation.detach()
        return hook

    def register_hooks(self, layer_indices: List[int], hook_point: str = "resid_post"):
        """Register hooks at specified layers."""
        self.clear_hooks()

        for layer_idx in layer_indices:
            # Get the appropriate module
            if hook_point == "resid_post":
                # After residual stream (post-MLP)
                module = self.model.model.layers[layer_idx]
            elif hook_point == "mlp_out":
                module = self.model.model.layers[layer_idx].mlp
            elif hook_point == "attn_out":
                module = self.model.model.layers[layer_idx].self_attn
            else:
                raise ValueError(f"Unknown hook point: {hook_point}")

            hook_name = f"layer_{layer_idx}_{hook_point}"
            handle = module.register_forward_hook(self._create_hook(hook_name))
            self.hooks.append(handle)

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    @torch.no_grad()
    def extract(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass and return stored activations."""
        self.activations = {}
        input_ids = input_ids.to(self.device)
        _ = self.model(input_ids)
        return self.activations

    def __del__(self):
        self.clear_hooks()
```

**Key Implementation Notes:**
- Hook functions capture activations during forward pass
- Clear hooks after extraction to avoid memory leaks
- Support multiple hook points (residual stream, MLP, attention)

### 2.2 Training Service

#### `backend/src/services/training_service.py`
```python
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from src.models.training import Training, TrainingMetric, Checkpoint
from src.schemas.training import TrainingCreate

class TrainingService:
    def __init__(self, db: Session):
        self.db = db

    def create(self, data: TrainingCreate) -> Training:
        training = Training(**data.model_dump())
        self.db.add(training)
        self.db.commit()
        self.db.refresh(training)
        return training

    def get_by_id(self, training_id: UUID) -> Optional[Training]:
        return self.db.query(Training).filter(Training.id == training_id).first()

    def list_all(self) -> List[Training]:
        return self.db.query(Training).order_by(Training.created_at.desc()).all()

    def update_status(self, training_id: UUID, status: str, **kwargs):
        training = self.get_by_id(training_id)
        if training:
            training.status = status
            for key, value in kwargs.items():
                setattr(training, key, value)
            self.db.commit()
        return training

    def add_metric(
        self,
        training_id: UUID,
        step: int,
        loss: float,
        **metrics
    ) -> TrainingMetric:
        metric = TrainingMetric(
            training_id=training_id,
            step=step,
            loss=loss,
            **metrics
        )
        self.db.add(metric)
        self.db.commit()
        return metric

    def save_checkpoint(
        self,
        training_id: UUID,
        step: int,
        path: str,
        is_best: bool = False
    ) -> Checkpoint:
        checkpoint = Checkpoint(
            training_id=training_id,
            step=step,
            path=path,
            is_best=is_best
        )
        self.db.add(checkpoint)
        self.db.commit()
        return checkpoint

    def get_metrics(self, training_id: UUID) -> List[TrainingMetric]:
        return self.db.query(TrainingMetric).filter(
            TrainingMetric.training_id == training_id
        ).order_by(TrainingMetric.step).all()
```

### 2.3 Training Task

#### `backend/src/workers/training_tasks.py`
```python
import torch
from celery import shared_task
from pathlib import Path
from src.db.session import SessionLocal
from src.services.training_service import TrainingService
from src.ml.sparse_autoencoder import StandardSAE, JumpReLUSAE, SkipSAE, TranscoderSAE
from src.ml.activation_extraction import ActivationExtractor
from src.workers.websocket_emitter import emit_training_progress
from src.core.config import settings

def create_sae(config: dict):
    """Factory function for SAE architectures."""
    arch = config.get("architecture", "standard")
    d_in = config["d_in"]
    d_sae = config["d_sae"]

    if arch == "standard":
        return StandardSAE(d_in, d_sae, l1_coeff=config.get("l1_coeff", 5e-3))
    elif arch == "jumprelu":
        return JumpReLUSAE(
            d_in, d_sae,
            l0_target=config.get("l0_target", 50),
            l0_coeff=config.get("l0_coeff", 1e-3)
        )
    elif arch == "skip":
        return SkipSAE(
            d_in, d_sae,
            skip_coeff=config.get("skip_coeff", 0.1),
            l1_coeff=config.get("l1_coeff", 5e-3)
        )
    elif arch == "transcoder":
        return TranscoderSAE(
            d_in=d_in,
            d_out=config["d_out"],
            d_sae=d_sae,
            l1_coeff=config.get("l1_coeff", 5e-3)
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


@shared_task(bind=True, queue='sae')
def train_sae_task(self, training_id: str, config: dict):
    """Main SAE training task."""
    db = SessionLocal()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        service = TrainingService(db)

        # Update status
        service.update_status(training_id, "running")
        emit_training_progress(training_id, 0, 0, "Initializing...")

        # Create SAE
        sae = create_sae(config).to(device)

        # Load activations
        activations_path = Path(settings.data_dir) / "activations" / config["activation_source"]
        activations = torch.load(activations_path).to(device)

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(activations)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.get("batch_size", 4096),
            shuffle=True
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=config.get("learning_rate", 1e-4)
        )

        # Training loop
        total_steps = config.get("num_steps", 10000)
        step = 0
        best_loss = float("inf")

        for epoch in range(config.get("num_epochs", 100)):
            for batch, in dataloader:
                if step >= total_steps:
                    break

                # Forward
                x_hat, z, aux = sae(batch)
                losses = sae.loss(batch, x_hat, z, aux)
                loss = losses["loss"]

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log metrics
                if step % config.get("log_every", 100) == 0:
                    service.add_metric(
                        training_id,
                        step=step,
                        loss=loss.item(),
                        recon_loss=losses["recon_loss"].item(),
                        l0=losses["l0"].item(),
                        fvu=losses.get("fvu", torch.tensor(0)).item()
                    )

                    progress = (step / total_steps) * 100
                    emit_training_progress(
                        training_id,
                        progress=progress,
                        step=step,
                        loss=loss.item(),
                        l0=losses["l0"].item()
                    )

                # Save checkpoint
                if step % config.get("checkpoint_every", 1000) == 0:
                    checkpoint_path = save_checkpoint(sae, training_id, step)
                    is_best = loss.item() < best_loss
                    if is_best:
                        best_loss = loss.item()
                    service.save_checkpoint(training_id, step, checkpoint_path, is_best)

                step += 1

        # Save final model
        final_path = save_final_model(sae, training_id, config)
        service.update_status(
            training_id,
            "completed",
            output_path=final_path,
            final_loss=best_loss
        )
        emit_training_progress(training_id, 100, step, "Complete", completed=True)

    except Exception as e:
        service.update_status(training_id, "failed", error_message=str(e))
        emit_training_progress(training_id, 0, 0, str(e), failed=True)
    finally:
        db.close()


def save_checkpoint(sae, training_id: str, step: int) -> str:
    """Save training checkpoint."""
    checkpoint_dir = Path(settings.data_dir) / "checkpoints" / training_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{step}.pt"
    torch.save(sae.state_dict(), path)
    return str(path)


def save_final_model(sae, training_id: str, config: dict) -> str:
    """Save final model in SAELens community format."""
    from safetensors.torch import save_file
    import json

    output_dir = Path(settings.data_dir) / "saes" / training_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    state_dict = {
        "W_enc": sae.W_enc.data,
        "b_enc": sae.b_enc.data,
        "W_dec": sae.W_dec.data,
        "b_dec": sae.b_dec.data
    }
    if hasattr(sae, "log_threshold"):
        state_dict["log_threshold"] = sae.log_threshold.data

    save_file(state_dict, output_dir / "sae_weights.safetensors")

    # Save config
    cfg = {
        "d_in": sae.d_in,
        "d_sae": sae.d_sae,
        "architecture": config.get("architecture", "standard"),
        "model_name": config.get("model_name"),
        "hook_name": config.get("hook_name"),
        "dtype": str(sae.dtype)
    }
    with open(output_dir / "cfg.json", "w") as f:
        json.dump(cfg, f, indent=2)

    return str(output_dir)
```

**Key Implementation Notes:**
- Factory pattern for SAE architecture creation
- Save in SAELens community format for interoperability
- Regular checkpoint saving for recovery
- WebSocket emission at configurable intervals

### 2.4 Frontend Components

#### `frontend/src/components/training/TrainingCard.tsx`
```typescript
import React from 'react';
import { Training } from '../../types/training';
import { Play, Pause, Square, Trash2, RotateCcw, Settings } from 'lucide-react';
import { ProgressBar } from '../common/ProgressBar';
import { formatDuration } from '../../utils/formatters';

interface TrainingCardProps {
  training: Training;
  onPause?: () => void;
  onResume?: () => void;
  onStop?: () => void;
  onDelete?: () => void;
  onRetry?: () => void;
  onViewDetails?: () => void;
}

export function TrainingCard({
  training,
  onPause,
  onResume,
  onStop,
  onDelete,
  onRetry,
  onViewDetails
}: TrainingCardProps) {
  const isRunning = training.status === 'running';
  const isPaused = training.status === 'paused';
  const isComplete = training.status === 'completed';
  const isFailed = training.status === 'failed';

  return (
    <div className="bg-slate-800 rounded-lg p-4">
      {/* Header */}
      <div className="flex justify-between items-start mb-3">
        <div>
          <h3 className="font-medium">{training.name}</h3>
          <p className="text-sm text-slate-400">
            {training.architecture} | {training.d_sae} features
          </p>
        </div>
        <div className="flex gap-1">
          {isRunning && (
            <>
              <button onClick={onPause} className="p-1 hover:bg-slate-700 rounded">
                <Pause className="w-4 h-4" />
              </button>
              <button onClick={onStop} className="p-1 hover:bg-slate-700 rounded">
                <Square className="w-4 h-4" />
              </button>
            </>
          )}
          {isPaused && (
            <button onClick={onResume} className="p-1 hover:bg-slate-700 rounded">
              <Play className="w-4 h-4" />
            </button>
          )}
          {isFailed && (
            <button onClick={onRetry} className="p-1 hover:bg-slate-700 rounded">
              <RotateCcw className="w-4 h-4" />
            </button>
          )}
          <button onClick={onViewDetails} className="p-1 hover:bg-slate-700 rounded">
            <Settings className="w-4 h-4" />
          </button>
          {(isComplete || isFailed) && (
            <button onClick={onDelete} className="p-1 hover:bg-slate-700 rounded text-red-400">
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Progress */}
      {(isRunning || isPaused) && (
        <div className="space-y-2">
          <ProgressBar progress={training.progress} />
          <div className="flex justify-between text-xs text-slate-400">
            <span>Step {training.current_step} / {training.total_steps}</span>
            <span>Loss: {training.current_loss?.toFixed(4)}</span>
          </div>
        </div>
      )}

      {/* Metrics for running training */}
      {isRunning && training.metrics && (
        <div className="grid grid-cols-3 gap-2 mt-3 text-sm">
          <div>
            <span className="text-slate-400">L0:</span>{' '}
            {training.metrics.l0?.toFixed(1)}
          </div>
          <div>
            <span className="text-slate-400">FVU:</span>{' '}
            {training.metrics.fvu?.toFixed(3)}
          </div>
          <div>
            <span className="text-slate-400">Time:</span>{' '}
            {formatDuration(training.elapsed_seconds)}
          </div>
        </div>
      )}

      {/* Status badge */}
      <div className="mt-3">
        <span className={`
          text-xs px-2 py-1 rounded
          ${isRunning ? 'bg-emerald-500/20 text-emerald-400' : ''}
          ${isPaused ? 'bg-yellow-500/20 text-yellow-400' : ''}
          ${isComplete ? 'bg-green-500/20 text-green-400' : ''}
          ${isFailed ? 'bg-red-500/20 text-red-400' : ''}
        `}>
          {training.status}
        </span>
      </div>
    </div>
  );
}
```

---

## 3. Common Patterns

### 3.1 Gradient Accumulation
```python
# For large effective batch sizes with limited memory
accumulation_steps = config.get("gradient_accumulation", 1)

for i, (batch,) in enumerate(dataloader):
    x_hat, z, aux = sae(batch)
    loss = sae.loss(batch, x_hat, z, aux)["loss"]
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3.2 Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    x_hat, z, aux = sae(batch)
    loss = sae.loss(batch, x_hat, z, aux)["loss"]

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3.3 Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

# Or OneCycle for faster convergence
scheduler = OneCycleLR(
    optimizer,
    max_lr=config["learning_rate"] * 10,
    total_steps=total_steps
)

# In training loop
optimizer.step()
scheduler.step()
```

---

## 4. Testing Strategy

### 4.1 SAE Architecture Tests
```python
# backend/tests/test_sae_architectures.py
import torch
import pytest
from src.ml.sparse_autoencoder import StandardSAE, JumpReLUSAE

def test_standard_sae_shapes():
    sae = StandardSAE(d_in=256, d_sae=1024)
    x = torch.randn(32, 256)
    x_hat, z, aux = sae(x)

    assert x_hat.shape == x.shape
    assert z.shape == (32, 1024)

def test_standard_sae_sparsity():
    sae = StandardSAE(d_in=256, d_sae=1024, l1_coeff=0.1)
    x = torch.randn(32, 256)
    x_hat, z, aux = sae(x)

    # Should have some zeros after ReLU
    assert (z == 0).any()

def test_jumprelu_threshold():
    sae = JumpReLUSAE(d_in=256, d_sae=1024, l0_target=10)
    x = torch.randn(32, 256)
    x_hat, z, aux = sae(x)

    l0 = (z > 0).float().sum(dim=-1).mean()
    # L0 should be trainable toward target
    assert l0 > 0
```

### 4.2 Training Task Tests
```python
# backend/tests/test_training_tasks.py
import pytest
from unittest.mock import patch, MagicMock

def test_create_sae_standard():
    config = {"architecture": "standard", "d_in": 256, "d_sae": 1024}
    sae = create_sae(config)
    assert isinstance(sae, StandardSAE)

def test_create_sae_jumprelu():
    config = {"architecture": "jumprelu", "d_in": 256, "d_sae": 1024, "l0_target": 50}
    sae = create_sae(config)
    assert isinstance(sae, JumpReLUSAE)
```

---

## 5. Common Pitfalls

### Pitfall 1: Decoder Weight Normalization
```python
# WRONG - Weights drift during training
# No normalization applied

# RIGHT - Normalize after each update
with torch.no_grad():
    sae.W_dec.data = sae.W_dec.data / sae.W_dec.data.norm(dim=1, keepdim=True)
```

### Pitfall 2: Memory Leak in Hooks
```python
# WRONG - Hooks keep references
self.activations[name] = output  # Keeps computation graph!

# RIGHT - Detach from graph
self.activations[name] = output.detach()
```

### Pitfall 3: Loss Scale with Batch Size
```python
# WRONG - Loss not normalized
recon_loss = (x - x_hat).pow(2).sum()

# RIGHT - Mean reduction
recon_loss = (x - x_hat).pow(2).mean()
```

---

## 6. Performance Tips

1. **Pre-compute Activations**
   ```python
   # Extract once, train many times
   activations = extract_activations(model, dataset)
   torch.save(activations, "activations.pt")
   # Training loads from disk, not GPU
   ```

2. **Use torch.compile (PyTorch 2.0+)**
   ```python
   sae = torch.compile(sae, mode="reduce-overhead")
   ```

3. **Pin Memory for DataLoader**
   ```python
   dataloader = DataLoader(
       dataset,
       pin_memory=True,
       num_workers=4
   )
   ```

---

*Related: [PRD](../prds/003_FPRD|SAE_Training.md) | [TDD](../tdds/003_FTDD|SAE_Training.md) | [FTASKS](../tasks/003_FTASKS|SAE_Training.md)*
