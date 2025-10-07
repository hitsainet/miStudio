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

## 8. Training Template Management Implementation

This section provides implementation guidance for training template save/load/favorite/export/import functionality (FR-1A from PRD).

### Component: Saved Templates Section

**Location:** Inside TrainingPanel, above the "Start Training" button (Mock UI lines 1628-1842)

**Implementation Pattern:**

```typescript
// Add to TrainingPanel component state
const [templates, setTemplates] = useState<TrainingTemplate[]>([]);
const [showTemplates, setShowTemplates] = useState(false);
const [templateName, setTemplateName] = useState('');
const [templateDescription, setTemplateDescription] = useState('');

// Fetch templates on mount
useEffect(() => {
  fetchTrainingTemplates();
}, []);

const fetchTrainingTemplates = async () => {
  try {
    const response = await fetch('/api/templates/training?limit=50');
    const data = await response.json();
    setTemplates(data.templates);
  } catch (error) {
    console.error('Failed to fetch templates:', error);
  }
};

// Auto-generate template name
const generateTemplateName = () => {
  const timestamp = new Date().toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit'
  }).replace(':', '');

  return `${encoderType}_${expansionFactor}x_${trainingSteps}steps_${timestamp}`;
};

// Save template handler
const handleSaveTemplate = async () => {
  const name = templateName || generateTemplateName();

  const template = {
    name,
    description: templateDescription,
    model_id: selectedModel,
    dataset_id: selectedDataset,
    encoder_type: encoderType,
    hyperparameters: {
      learningRate,
      batchSize,
      l1Coefficient,
      expansionFactor,
      trainingSteps,
      trainingLayers: selectedLayers,  // Multi-layer support
      optimizer,
      lrSchedule,
      ghostGradPenalty
    },
    is_favorite: false
  };

  try {
    const response = await fetch('/api/templates/training', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(template)
    });

    if (response.ok) {
      await fetchTrainingTemplates();
      setTemplateName('');
      setTemplateDescription('');
      // Show success toast
    }
  } catch (error) {
    console.error('Failed to save template:', error);
    // Show error toast
  }
};

// Load template handler
const handleLoadTemplate = (template: TrainingTemplate) => {
  // Load optional references
  if (template.model_id) setSelectedModel(template.model_id);
  if (template.dataset_id) setSelectedDataset(template.dataset_id);

  // Load encoder type
  setEncoderType(template.encoder_type);

  // Load all hyperparameters
  const hp = template.hyperparameters;
  setLearningRate(hp.learningRate);
  setBatchSize(hp.batchSize);
  setL1Coefficient(hp.l1Coefficient);
  setExpansionFactor(hp.expansionFactor);
  setTrainingSteps(hp.trainingSteps);
  setSelectedLayers(hp.trainingLayers || [0]);  // Multi-layer
  setOptimizer(hp.optimizer);
  setLrSchedule(hp.lrSchedule);
  setGhostGradPenalty(hp.ghostGradPenalty);

  // Show success toast
};
```

**UI Structure (similar to extraction templates):**

Add collapsible "Saved Templates" section in TrainingPanel with same UI pattern:
- Save form (name + description + save button)
- Template list with star/load/delete buttons
- Export/Import buttons

Implementation code follows same pattern as extraction templates section in Model Management TID.

### Backend: Training Template Endpoints

API endpoints follow same pattern as extraction templates. Refer to TDD section for complete specifications.

Key differences:
- Includes model_id and dataset_id (optional nullable references)
- encoder_type validation: must be one of ['sparse', 'skip', 'transcoder']
- hyperparameters JSONB validation (all fields required except trainingLayers defaults to [0])

---

## 9. Multi-Layer Training Implementation

This section provides implementation guidance for training SAEs on multiple transformer layers simultaneously (FR-1B from PRD).

### Component: Training Layers Selector

**Location:** Inside "Advanced Hyperparameters" collapsible section (Mock UI lines 1740-1789)

**Implementation Pattern:**

```typescript
// Add to TrainingPanel state
const [selectedLayers, setSelectedLayers] = useState<number[]>([0]);
const [modelNumLayers, setModelNumLayers] = useState<number | null>(null);

// Fetch model architecture when model selected
useEffect(() => {
  if (selectedModel) {
    fetchModelArchitecture(selectedModel);
  }
}, [selectedModel]);

const fetchModelArchitecture = async (modelId: string) => {
  try {
    const response = await fetch(`/api/models/${modelId}`);
    const data = await response.json();
    setModelNumLayers(data.num_layers);
  } catch (error) {
    console.error('Failed to fetch model architecture:', error);
  }
};

// Layer selection handlers
const handleLayerToggle = (layer: number) => {
  setSelectedLayers(prev => {
    if (prev.includes(layer)) {
      return prev.filter(l => l !== layer);
    } else {
      return [...prev, layer].sort((a, b) => a - b);
    }
  });
};

const handleSelectAllLayers = () => {
  if (modelNumLayers) {
    setSelectedLayers(Array.from({ length: modelNumLayers }, (_, i) => i));
  }
};

const handleClearAllLayers = () => {
  setSelectedLayers([]);
};

// Memory estimation
const estimateMemoryRequirements = () => {
  if (!modelNumLayers || selectedLayers.length === 0) return null;

  const hiddenSize = 2048;  // From model architecture
  const memoryPerSAE = hiddenSize * expansionFactor * 4 * 3; // FP32 * 3 (params + 2 optimizer states)
  const totalSAEMemory = memoryPerSAE * selectedLayers.length;
  const baseMemory = 2e9;  // Model + activations
  const totalMemory = baseMemory + totalSAEMemory;

  return {
    totalGB: totalMemory / 1e9,
    exceedsLimit: totalMemory > 6e9,  // Jetson 6GB limit
    recommendation: totalMemory > 6e9 ? `Reduce to ≤${Math.floor(6e9 / memoryPerSAE)} layers` : null
  };
};

const memoryEstimate = estimateMemoryRequirements();
```

**UI Structure:**

```typescript
{/* Add in Advanced Hyperparameters section */}
<div className="mb-4">
  <label className="block mb-2 text-sm font-medium text-gray-300">
    Training Layers
    <span className="ml-2 text-xs text-gray-500">
      ({selectedLayers.length} selected)
    </span>
  </label>

  {/* Select All / Clear All buttons */}
  <div className="flex gap-2 mb-2">
    <button
      onClick={handleSelectAllLayers}
      disabled={!modelNumLayers}
      className="px-3 py-1 text-xs font-medium text-gray-300 bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50"
    >
      Select All
    </button>
    <button
      onClick={handleClearAllLayers}
      className="px-3 py-1 text-xs font-medium text-gray-300 bg-gray-700 rounded hover:bg-gray-600"
    >
      Clear All
    </button>
  </div>

  {/* 8-column checkbox grid */}
  {modelNumLayers ? (
    <div className="grid grid-cols-8 gap-2 p-3 bg-gray-800 rounded max-h-48 overflow-y-auto">
      {Array.from({ length: modelNumLayers }, (_, i) => (
        <label
          key={i}
          className={`flex items-center justify-center px-2 py-1 text-xs rounded cursor-pointer transition-colors ${
            selectedLayers.includes(i)
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          <input
            type="checkbox"
            checked={selectedLayers.includes(i)}
            onChange={() => handleLayerToggle(i)}
            className="sr-only"
          />
          L{i}
        </label>
      ))}
    </div>
  ) : (
    <div className="p-3 text-sm text-center text-gray-500 bg-gray-800 rounded">
      Select a model to choose training layers
    </div>
  )}

  {/* Memory warning */}
  {memoryEstimate && memoryEstimate.exceedsLimit && (
    <div className="flex items-start gap-2 p-2 mt-2 text-sm text-yellow-400 bg-yellow-900/20 rounded">
      <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
      <div>
        <div className="font-medium">Memory Warning</div>
        <div className="text-xs text-yellow-400/80">
          Estimated: {memoryEstimate.totalGB.toFixed(1)} GB exceeds 6GB limit.
          {memoryEstimate.recommendation}
        </div>
      </div>
    </div>
  )}

  {/* Helpful hint for multi-layer */}
  {selectedLayers.length > 4 && (
    <div className="p-2 mt-2 text-xs text-blue-400 bg-blue-900/20 rounded">
      Training {selectedLayers.length} layers simultaneously will take longer but provides comprehensive layer analysis.
    </div>
  )}
</div>
```

### Backend: Multi-Layer Training Pipeline

**File:** `backend/src/services/training_service.py`

**Initialization:**

```python
def initialize_multilayer_training(config: TrainingConfig, model, dataset):
    """Initialize separate SAE for each selected layer."""
    training_layers = config.hyperparameters.get('trainingLayers', [0])

    # Validate memory requirements
    estimated_memory = calculate_memory_requirements(
        model_hidden_size=model.config.hidden_size,
        expansion_factor=config.hyperparameters['expansionFactor'],
        num_layers=len(training_layers)
    )

    if estimated_memory > 7.5e9:  # 7.5GB threshold
        raise ValueError(
            f"Estimated memory {estimated_memory / 1e9:.1f}GB exceeds 7.5GB limit. "
            f"Reduce number of layers or expansion factor."
        )

    # Initialize separate SAE for each layer
    saes = {}
    optimizers = {}

    for layer_idx in training_layers:
        # Validate layer index
        if layer_idx >= model.config.num_hidden_layers:
            raise ValueError(f"Layer {layer_idx} exceeds model layer count {model.config.num_hidden_layers}")

        # Initialize SAE
        sae = SparseAutoencoder(
            d_model=model.config.hidden_size,
            d_sae=model.config.hidden_size * config.hyperparameters['expansionFactor'],
            l1_coefficient=config.hyperparameters['l1Coefficient']
        ).to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=config.hyperparameters['learningRate']
        )

        saes[layer_idx] = sae
        optimizers[layer_idx] = optimizer

    return saes, optimizers
```

**Training Loop:**

```python
def train_multilayer_step(batch, saes: dict, optimizers: dict, training_layers: list, model):
    """Execute one training step for all layers."""

    # Extract activations from all layers simultaneously
    activations_by_layer = extract_multilayer_activations(model, batch, training_layers)

    # Train each layer's SAE independently
    losses = {}
    metrics = {}

    for layer_idx in training_layers:
        activations = activations_by_layer[layer_idx]  # (batch_size, hidden_size)
        sae = saes[layer_idx]
        optimizer = optimizers[layer_idx]

        # Forward pass
        reconstructed, latents = sae(activations)

        # Compute loss
        reconstruction_loss = F.mse_loss(reconstructed, activations)
        sparsity_loss = sae.l1_coefficient * latents.abs().mean()
        total_loss = reconstruction_loss + sparsity_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track metrics
        losses[layer_idx] = total_loss.item()
        metrics[layer_idx] = {
            "loss": total_loss.item(),
            "sparsity": (latents.abs() > 1e-5).float().sum(dim=-1).mean().item(),
            "reconstruction_error": reconstruction_loss.item()
        }

    # Aggregate metrics for progress tracking
    aggregated_metrics = {
        "avg_loss": np.mean([m["loss"] for m in metrics.values()]),
        "avg_sparsity": np.mean([m["sparsity"] for m in metrics.values()]),
        "avg_reconstruction_error": np.mean([m["reconstruction_error"] for m in metrics.values()])
    }

    return metrics, aggregated_metrics
```

**Activation Extraction:**

```python
def extract_multilayer_activations(model, batch, layers: list):
    """Extract activations from multiple layers in single forward pass."""
    activations_by_layer = {}
    hooks = []

    # Register forward hooks for all layers
    def create_hook(layer_idx):
        def hook(module, input, output):
            # Detach to avoid gradient tracking
            activations_by_layer[layer_idx] = output.detach()
        return hook

    for layer_idx in layers:
        layer = model.transformer.h[layer_idx]
        hook = layer.register_forward_hook(create_hook(layer_idx))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        model(**batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations_by_layer
```

**Checkpoint Management:**

```python
def save_multilayer_checkpoint(training_id: str, step: int, saes: dict, optimizers: dict, metrics: dict):
    """Save multi-layer checkpoint with subdirectory structure."""
    checkpoint_dir = Path(f"/data/trainings/{training_id}/checkpoints/checkpoint_{step}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save each layer's SAE independently
    for layer_idx, sae in saes.items():
        layer_dir = checkpoint_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        # Save SAE weights
        torch.save(sae.encoder.state_dict(), layer_dir / "encoder.pt")
        torch.save(sae.decoder.state_dict(), layer_dir / "decoder.pt")

        # Save optimizer state
        torch.save(optimizers[layer_idx].state_dict(), layer_dir / "optimizer.pt")

    # Save shared metadata
    metadata = {
        "step": step,
        "training_id": training_id,
        "trainingLayers": list(saes.keys()),
        "metrics_by_layer": {str(k): v for k, v in metrics.items()},
        "aggregated_metrics": aggregate_metrics(metrics),
        "created_at": datetime.utcnow().isoformat()
    }

    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def load_multilayer_checkpoint(checkpoint_path: Path, model_config):
    """Load multi-layer checkpoint."""
    # Load metadata
    with open(checkpoint_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    training_layers = metadata["trainingLayers"]

    # Initialize SAEs for each layer
    saes = {}
    optimizers = {}

    for layer_idx in training_layers:
        layer_dir = checkpoint_path / f"layer_{layer_idx}"

        # Initialize SAE
        sae = SparseAutoencoder(
            d_model=model_config.hidden_size,
            d_sae=model_config.hidden_size * metadata["hyperparameters"]["expansionFactor"],
            l1_coefficient=metadata["hyperparameters"]["l1Coefficient"]
        )

        # Load weights
        sae.encoder.load_state_dict(torch.load(layer_dir / "encoder.pt"))
        sae.decoder.load_state_dict(torch.load(layer_dir / "decoder.pt"))

        # Initialize and load optimizer
        optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=metadata["hyperparameters"]["learningRate"]
        )
        optimizer.load_state_dict(torch.load(layer_dir / "optimizer.pt"))

        saes[layer_idx] = sae
        optimizers[layer_idx] = optimizer

    return saes, optimizers, metadata
```

### Progress Tracking for Multi-Layer

**WebSocket Updates:**

```python
# Emit progress with aggregated metrics
await ws_manager.emit_training_progress(training_id, {
    "current_step": step,
    "progress": (step / total_steps) * 100,
    "latest_loss": aggregated_metrics["avg_loss"],
    "latest_sparsity": aggregated_metrics["avg_sparsity"],
    "per_layer_metrics": metrics,  # Detailed per-layer metrics
    "timestamp": datetime.utcnow().isoformat()
})
```

**Frontend Display:**

```typescript
// Show aggregated metrics in main training card
<div className="text-xs text-gray-400">
  Loss: {training.latest_loss?.toFixed(4)} (avg across {training.num_layers} layers)
</div>

// Expandable per-layer metrics detail
{showLayerDetails && training.per_layer_metrics && (
  <div className="mt-2 space-y-1">
    {Object.entries(training.per_layer_metrics).map(([layer, metrics]) => (
      <div key={layer} className="flex justify-between text-xs text-gray-500">
        <span>Layer {layer}:</span>
        <span>Loss {metrics.loss.toFixed(4)} | Sparsity {metrics.sparsity.toFixed(1)}</span>
      </div>
    ))}
  </div>
)}
```

### Error Handling for Multi-Layer

```python
try:
    # Training loop
    for step in range(current_step, total_steps):
        metrics, aggregated = train_multilayer_step(batch, saes, optimizers, training_layers, model)

except torch.cuda.OutOfMemoryError as e:
    # OOM - save partial checkpoint and notify
    logger.error(f"OOM during multi-layer training at step {step}")

    # Save checkpoint for completed steps
    save_multilayer_checkpoint(training_id, step, saes, optimizers, metrics)

    # Update training status
    await training_repo.update_status(training_id, "error",
        error_message=f"Out of memory during multi-layer training. Try reducing layers (currently {len(training_layers)}) or expansion factor."
    )

    # Emit error to frontend
    await ws_manager.emit_training_error(training_id, {
        "error": "OOM",
        "message": "Insufficient memory for multi-layer training",
        "suggestion": "Reduce number of layers or expansion factor",
        "current_layers": len(training_layers)
    })

except Exception as e:
    logger.exception(f"Error during multi-layer training: {e}")
    # Handle other errors...
```

### Implementation Checklist

- [ ] Add training_templates table migration
- [ ] Implement all 7 API endpoints for training templates
- [ ] Add "Saved Templates" section to TrainingPanel
- [ ] Implement training layers selector (8-column checkbox grid)
- [ ] Add memory estimation with warnings
- [ ] Implement multi-layer SAE initialization
- [ ] Implement multi-layer activation extraction with hooks
- [ ] Implement multi-layer training loop
- [ ] Implement multi-layer checkpoint save/load with subdirectories
- [ ] Update progress tracking to show aggregated + per-layer metrics
- [ ] Add OOM error handling with memory reduction suggestions
- [ ] Add validation for layer indices vs. model architecture
- [ ] Test with 1 layer (backward compatible)
- [ ] Test with 4 layers (typical multi-layer scenario)
- [ ] Test with 8+ layers (memory pressure scenario)
- [ ] Test checkpoint resume for multi-layer training

---

**Document End**
**Status:** Ready for Task Generation
**Total Sections:** 10 (focused implementation guide with templates and multi-layer)
**Estimated Size:** ~45KB
**Next:** 004_FTID|Feature_Discovery.md
