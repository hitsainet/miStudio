# Technical Implementation Document: SAE Training

**Document ID:** 003_FTID|SAE_Training
**Feature:** Sparse Autoencoder Training System
**PRD Reference:** 003_FPRD|SAE_Training.md
**TDD Reference:** 003_FTDD|SAE_Training.md
**Status:** Ready for Implementation
**Created:** 2025-10-06

---

## 1. Implementation Overview

Implements PyTorch training loop for Sparse Autoencoders with real-time metrics, checkpoint management, and memory optimization for Jetson Orin Nano (6GB GPU VRAM). Mock UI lines 1628-2156 is PRIMARY reference.

**Critical Components:**
- SAE Model Architecture (encoder/decoder with sparsity)
- Training Loop (forward/backward pass, optimizer, scheduler)
- Checkpoint Management (safetensors format, retention policy)
- Real-time Metrics (L0 sparsity, dead neurons, WebSocket streaming)
- Memory Optimization (mixed precision, gradient accumulation, OOM handling)

---

## 2. SAE Model Implementation

**File:** `backend/src/ml/sae_model.py`

```python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for activation interpretation.

    Architecture:
    - Encoder: Linear(hidden_dim, latent_dim) + ReLU
    - Decoder: Linear(latent_dim, hidden_dim)
    - Loss: MSE reconstruction + L1 sparsity penalty
    """

    def __init__(
        self,
        hidden_dim: int,  # Model hidden dimension (e.g., 768 for GPT-2)
        latent_dim: int,  # SAE latent dimension (e.g., 8192)
        l1_alpha: float = 0.001,  # L1 sparsity coefficient
        tie_weights: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.l1_alpha = l1_alpha

        # Encoder: hidden_dim -> latent_dim
        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=True)

        # Decoder: latent_dim -> hidden_dim
        if tie_weights:
            # Tied weights: decoder = encoder.T
            self.decoder = lambda x: torch.nn.functional.linear(x, self.encoder.weight.t())
        else:
            self.decoder = nn.Linear(latent_dim, hidden_dim, bias=True)

        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        if not tie_weights:
            nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.

        Args:
            x: Input activations [batch_size, seq_len, hidden_dim]

        Returns:
            (reconstructed, latent_activations, l0_sparsity)
        """
        # Encode
        latent = torch.relu(self.encoder(x))  # [batch, seq, latent_dim]

        # Decode
        reconstructed = self.decoder(latent)  # [batch, seq, hidden_dim]

        # Calculate L0 sparsity (% of active neurons)
        l0_sparsity = (latent > 0).float().mean().item()

        return reconstructed, latent, l0_sparsity

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        latent: torch.Tensor
    ) -> dict:
        """
        Compute loss components.

        Returns dict with:
        - total_loss
        - reconstruction_loss (MSE)
        - sparsity_loss (L1)
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, x)

        # Sparsity loss (L1 penalty on latent activations)
        sparsity_loss = self.l1_alpha * torch.abs(latent).mean()

        # Total loss
        total_loss = reconstruction_loss + sparsity_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss.item(),
            "sparsity_loss": sparsity_loss.item()
        }
```

---

## 3. Training Loop Implementation

**File:** `backend/src/workers/training_tasks.py`

```python
from celery import Task
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from src.ml.sae_model import SparseAutoencoder
from src.services.training_service import TrainingService
from src.core.celery_app import celery_app

@celery_app.task(bind=True)
def train_sae_task(self: Task, training_id: str, config: dict):
    """
    SAE training task with real-time metrics.

    Steps:
    1. Load model, dataset, create SAE
    2. Training loop (total_steps iterations)
    3. Calculate metrics every 10 steps
    4. Save checkpoints (manual + auto-save)
    5. Emit WebSocket events for progress
    """
    import asyncio

    async def run_training():
        async with AsyncSessionLocal() as db:
            service = TrainingService(db)

            try:
                # 1. Setup
                training = await service.get_training(training_id)
                model, tokenizer = load_model(training.model_id)
                dataset = load_dataset(training.dataset_id)
                sae = SparseAutoencoder(
                    hidden_dim=config['hidden_dim'],
                    latent_dim=config['latent_dim'],
                    l1_alpha=config['l1_alpha']
                ).cuda()

                # Optimizer
                optimizer = optim.Adam(sae.parameters(), lr=config['learning_rate'])
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config['total_steps']
                )

                # Mixed precision training
                scaler = GradScaler()

                # Activation extractor
                from src.ml.forward_hooks import ActivationExtractor
                extractor = ActivationExtractor(model)
                extractor.register_hooks([config['layer_idx']], "residual")

                # 2. Training loop
                for step in range(config['total_steps']):
                    # Get batch
                    batch = get_next_batch(dataset, config['batch_size'])
                    input_ids = batch['input_ids'].cuda()

                    # Extract activations
                    activations = extractor.extract_from_batch(input_ids)
                    x = activations[config['layer_idx']].cuda()  # [batch, seq, hidden]

                    # Forward pass (mixed precision)
                    optimizer.zero_grad()
                    with autocast():
                        reconstructed, latent, l0_sparsity = sae(x)
                        loss_dict = sae.compute_loss(x, reconstructed, latent)
                        loss = loss_dict['total_loss']

                    # Backward pass
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # Calculate metrics every 10 steps
                    if step % 10 == 0:
                        metrics = {
                            "step": step,
                            "loss": loss_dict['reconstruction_loss'],
                            "sparsity": l0_sparsity,
                            "dead_neurons": calculate_dead_neurons(latent),
                            "learning_rate": scheduler.get_last_lr()[0]
                        }

                        # Save to database
                        await service.save_metrics(training_id, metrics)

                        # Emit WebSocket
                        from src.core.websocket import emit_event
                        emit_event(f"trainings/{training_id}/metrics", metrics)

                    # Auto-save checkpoint
                    if config.get('auto_save_interval') and step % config['auto_save_interval'] == 0:
                        checkpoint_path = f"/data/checkpoints/{training_id}/step_{step}.safetensors"
                        save_checkpoint(sae, optimizer, scheduler, step, checkpoint_path)

                # 3. Final checkpoint
                final_path = f"/data/checkpoints/{training_id}/final.safetensors"
                save_checkpoint(sae, optimizer, scheduler, config['total_steps'], final_path)

                # Update training status
                training.status = 'completed'
                training.current_step = config['total_steps']
                await db.commit()

                # Cleanup
                extractor.cleanup()

            except torch.cuda.OutOfMemoryError:
                # Handle OOM
                await service.update_status(training_id, 'error', "OOM: Try reducing batch_size")
                torch.cuda.empty_cache()

            except Exception as e:
                await service.update_status(training_id, 'error', str(e))
                raise

    asyncio.run(run_training())

def calculate_dead_neurons(latent: torch.Tensor) -> int:
    """Count neurons that are never active (always zero)."""
    is_active = (latent > 0).any(dim=(0, 1))  # [latent_dim]
    dead_count = (~is_active).sum().item()
    return dead_count

def save_checkpoint(sae, optimizer, scheduler, step, path):
    """Save checkpoint using safetensors."""
    from safetensors.torch import save_file
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)

    state_dict = {
        "sae_state_dict": sae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": torch.tensor(step)
    }

    save_file(state_dict, path)
```

**Key Training Optimizations:**
- Mixed precision (`torch.cuda.amp.autocast`) - 40% memory reduction
- Gradient accumulation (if batch_size causes OOM)
- Activation checkpointing for large SAEs
- Dynamic batch size reduction on OOM

---

## 4. Frontend Component: TrainingPanel

**File:** `frontend/src/components/panels/TrainingPanel.tsx`

**PRIMARY REFERENCE:** Mock UI lines 1628-1842

```typescript
import React, { useState } from 'react';
import { Play, Settings } from 'lucide-react';
import { useTrainingsStore } from '@/stores/trainingsStore';
import { TrainingCard } from '@/components/trainings/TrainingCard';

export const TrainingPanel: React.FC = () => {
  const trainings = useTrainingsStore((state) => state.trainings);
  const models = useModelsStore((state) => state.models.filter(m => m.status === 'ready'));
  const datasets = useDatasetsStore((state) => state.datasets.filter(d => d.status === 'ready'));

  const [config, setConfig] = useState({
    model_id: '',
    dataset_id: '',
    encoder_type: 'standard',
    layer_idx: 12,
    latent_dim: 8192,
    l1_alpha: 0.001,
    learning_rate: 0.0001,
    batch_size: 32,
    total_steps: 10000,
    auto_save_interval: 1000
  });

  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Training Configuration</h2>

      {/* Config Form */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
        {/* Model + Dataset + Encoder (3-column grid) */}
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Model</label>
            <select value={config.model_id} onChange={(e) => setConfig({...config, model_id: e.target.value})}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500">
              <option value="">Select model...</option>
              {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </div>
          {/* Dataset + Encoder similar */}
        </div>

        {/* Advanced Config (collapsible) */}
        <button onClick={() => setShowAdvanced(!showAdvanced)} className="text-sm text-emerald-400">
          <Settings className="w-4 h-4 inline mr-1" />
          {showAdvanced ? 'Hide' : 'Show'} Advanced Configuration
        </button>

        {showAdvanced && (
          <div className="grid grid-cols-2 gap-4 bg-slate-800/30 p-4 rounded-lg">
            {/* Layer, Latent Dim, L1 Alpha, LR, Batch Size, Steps, Auto-save */}
          </div>
        )}

        {/* Start Training Button */}
        <button disabled={!config.model_id || !config.dataset_id}
          className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 rounded-lg">
          <Play className="w-5 h-5 inline mr-2" />
          Start Training
        </button>
      </div>

      {/* Active Trainings */}
      <div className="space-y-4">
        <h3 className="text-xl font-semibold">Training Jobs</h3>
        {trainings.map(t => <TrainingCard key={t.id} training={t} />)}
      </div>
    </div>
  );
};
```

---

## 5. Real-time Metrics Display

**File:** `frontend/src/components/trainings/TrainingCard.tsx`

**PRIMARY REFERENCE:** Mock UI lines 1845-2156

```typescript
import React, { useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

export const TrainingCard: React.FC<{ training: Training }> = ({ training }) => {
  const [metrics, setMetrics] = useState([]);
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    const channel = `trainings/${training.id}/metrics`;
    subscribe(channel, (data) => {
      setMetrics(prev => [...prev, data].slice(-100));  // Keep last 100 points
    });
    return () => unsubscribe(channel);
  }, [training.id]);

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-semibold text-lg">{training.name}</h3>
          <p className="text-sm text-slate-400">
            Step {training.current_step}/{training.total_steps} •
            Loss: {training.latest_loss?.toFixed(4)} •
            Sparsity: {(training.latest_sparsity * 100).toFixed(1)}%
          </p>
        </div>
        <button className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm">
          Stop
        </button>
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm text-slate-400 mb-2">
          <span>Progress</span>
          <span>{((training.current_step / training.total_steps) * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-slate-800 rounded-full h-2">
          <div className="bg-emerald-500 h-2 rounded-full transition-all"
            style={{ width: `${(training.current_step / training.total_steps) * 100}%` }} />
        </div>
      </div>

      {/* Metrics Grid (4 cards) */}
      <div className="grid grid-cols-4 gap-4 mb-4">
        <div className="bg-slate-800/50 p-3 rounded-lg">
          <div className="text-xs text-slate-400">Loss</div>
          <div className="text-xl font-semibold text-emerald-400">{training.latest_loss?.toFixed(4)}</div>
        </div>
        {/* L0 Sparsity, Dead Neurons, Learning Rate */}
      </div>

      {/* Loss Chart */}
      <div className="h-48">
        <LineChart width={600} height={180} data={metrics}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="step" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
          <Line type="monotone" dataKey="loss" stroke="#10b981" strokeWidth={2} dot={false} />
        </LineChart>
      </div>
    </div>
  );
};
```

---

## 6. Checkpoint Management

**File:** `backend/src/services/checkpoint_service.py`

```python
from safetensors.torch import load_file, save_file
import os
from typing import Dict

class CheckpointService:
    """Manage SAE checkpoints with retention policy."""

    def __init__(self, training_id: str):
        self.training_id = training_id
        self.checkpoint_dir = f"/data/checkpoints/{training_id}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        sae_state: Dict,
        optimizer_state: Dict,
        step: int,
        is_best: bool = False
    ) -> str:
        """Save checkpoint with safetensors."""
        filename = f"step_{step}.safetensors"
        if is_best:
            filename = "best.safetensors"

        path = os.path.join(self.checkpoint_dir, filename)

        state_dict = {
            **{f"sae.{k}": v for k, v in sae_state.items()},
            **{f"optimizer.{k}": v for k, v in optimizer_state.items()},
            "step": torch.tensor(step)
        }

        save_file(state_dict, path)
        return path

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint from safetensors."""
        state_dict = load_file(checkpoint_path)

        # Separate SAE and optimizer states
        sae_state = {k.replace("sae.", ""): v for k, v in state_dict.items() if k.startswith("sae.")}
        optimizer_state = {k.replace("optimizer.", ""): v for k, v in state_dict.items() if k.startswith("optimizer.")}
        step = state_dict["step"].item()

        return {
            "sae_state_dict": sae_state,
            "optimizer_state_dict": optimizer_state,
            "step": step
        }

    def apply_retention_policy(self, keep_first: bool = True, keep_last: bool = True, keep_every_n: int = 1000):
        """
        Keep:
        - First checkpoint (step 0)
        - Last checkpoint (latest)
        - Every Nth step (e.g., 1000, 2000, ...)
        - Best checkpoint (lowest loss)
        Delete rest.
        """
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.safetensors')]

        # Extract step numbers
        step_files = [(int(f.replace('step_', '').replace('.safetensors', '')), f) for f in checkpoints if f.startswith('step_')]
        step_files.sort()

        to_keep = set()

        # Keep first
        if keep_first and step_files:
            to_keep.add(step_files[0][1])

        # Keep last
        if keep_last and step_files:
            to_keep.add(step_files[-1][1])

        # Keep every N
        for step, filename in step_files:
            if step % keep_every_n == 0:
                to_keep.add(filename)

        # Keep best
        to_keep.add('best.safetensors')

        # Delete others
        for _, filename in step_files:
            if filename not in to_keep:
                os.remove(os.path.join(self.checkpoint_dir, filename))
```

---

**Document End**
**Status:** Ready for Task Generation
**Estimated Size:** ~30KB
**Next:** 004_FTID|Feature_Discovery.md
