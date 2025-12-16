# Technical Design Document: SAE Training

**Document ID:** 003_FTDD|SAE_Training
**Version:** 1.1
**Last Updated:** 2025-12-16
**Status:** Implemented
**Related PRD:** [003_FPRD|SAE_Training](../prds/003_FPRD|SAE_Training.md)

---

## 1. System Architecture

### 1.1 Training Pipeline Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Flow                             │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Dataset  │───→│  Model   │───→│Activation│───→│   SAE    │  │
│  │(tokenized)│   │(loaded)  │    │  Buffer  │    │ Training │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                        │                              │          │
│                        ▼                              ▼          │
│              ┌──────────────────┐          ┌──────────────────┐ │
│              │ Forward Hook     │          │ Metrics + Ckpts  │ │
│              │ (activation      │          │ → WebSocket      │ │
│              │  extraction)     │          │ → Database       │ │
│              └──────────────────┘          └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                         │
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ TrainingPanel │  │StartTrainingModal│ │ TrainingCard  │ │
│  └───────┬───────┘  └────────┬────────┘  └───────┬───────┘ │
│          │                   │                    │         │
│  ┌───────┴───────────────────┴────────────────────┴───────┐ │
│  │               trainingsStore (Zustand)                  │ │
│  │  + useTrainingWebSocket hook                           │ │
│  └────────────────────────────┬────────────────────────────┘ │
└───────────────────────────────┼─────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────┐
│                    Backend                                   │
│  ┌────────────────────────────┴────────────────────────────┐│
│  │                  TrainingService                        ││
│  │  - create_training()                                    ││
│  │  - get_metrics()                                        ││
│  │  - stop_training()                                      ││
│  └────────────────────────────┬────────────────────────────┘│
│                               │                             │
│  ┌────────────────────────────┴────────────────────────────┐│
│  │              Celery Worker (GPU Queue)                  ││
│  │  ┌────────────────────────────────────────────────┐     ││
│  │  │              train_sae_task                    │     ││
│  │  │  ┌────────────────┐  ┌────────────────────┐   │     ││
│  │  │  │SparseAutoencoder│  │   JumpReLUSAE     │   │     ││
│  │  │  └────────────────┘  └────────────────────┘   │     ││
│  │  └────────────────────────────────────────────────┘     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 2. SAE Architecture Classes

### 2.1 Class Hierarchy
```
                    ┌───────────────────┐
                    │  BaseSAE (ABC)    │
                    │  - encode()       │
                    │  - decode()       │
                    │  - forward()      │
                    │  - loss()         │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│SparseAutoencoder│  │  JumpReLUSAE  │   │    SkipSAE    │
│  (Standard)    │  │               │   │               │
│  L1 penalty    │  │  L0 penalty   │   │  Skip connect │
└───────────────┘   └───────────────┘   └───────────────┘
```

### 2.2 Standard SAE Implementation
```python
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_sae: int, ...):
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: Tensor) -> Tensor:
        return F.relu(x @ self.W_enc + self.b_enc)

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, dict]:
        x_norm = self.normalize(x)
        z = self.encode(x_norm)
        if self.top_k:
            z = self.apply_topk(z)
        x_hat = self.decode(z)
        return x_hat, z, {"x_norm": x_norm}

    def loss(self, x: Tensor, x_hat: Tensor, z: Tensor, aux: dict) -> dict:
        recon_loss = F.mse_loss(x_hat, aux["x_norm"])
        l1_loss = self.l1_alpha * z.abs().mean()
        return {
            "loss": recon_loss + l1_loss,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
            "l0": (z > 0).float().sum(dim=-1).mean()
        }
```

### 2.3 JumpReLU SAE (Gemma Scope Style)
```python
class JumpReLUSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, ...):
        super().__init__()
        self.threshold = nn.Parameter(torch.ones(d_sae) * 0.1)
        # ... other params

    def encode(self, x: Tensor) -> Tensor:
        pre_act = x @ self.W_enc + self.b_enc
        return F.relu(pre_act - self.threshold) * (pre_act > self.threshold)

    def loss(self, x, x_hat, z, aux):
        recon_loss = F.mse_loss(x_hat, aux["x_norm"])
        l0 = (z > 0).float().sum(dim=-1).mean()
        l0_loss = self.l0_alpha * l0
        return {
            "loss": recon_loss + l0_loss,
            "l0": l0,
            "threshold_mean": self.threshold.mean()
        }
```

---

## 3. Database Schema

### 3.1 Training Table
```sql
CREATE TABLE trainings (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    model_id UUID REFERENCES models(id),
    dataset_id UUID REFERENCES datasets(id),
    tokenization_id UUID REFERENCES dataset_tokenizations(id),
    layer INTEGER NOT NULL,
    architecture VARCHAR(50) NOT NULL,  -- standard, jumprelu, skip, transcoder
    hyperparameters JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    current_step INTEGER DEFAULT 0,
    final_loss FLOAT,
    sae_path VARCHAR(500),
    celery_task_id VARCHAR(255),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    CONSTRAINT trainings_arch_check CHECK (
        architecture IN ('standard', 'jumprelu', 'skip', 'transcoder')
    )
);
```

### 3.2 Training Metrics Table
```sql
CREATE TABLE training_metrics (
    id UUID PRIMARY KEY,
    training_id UUID REFERENCES trainings(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    loss FLOAT NOT NULL,
    recon_loss FLOAT,
    l0 FLOAT,
    l1_loss FLOAT,
    fvu FLOAT,  -- Fraction of Variance Unexplained
    dead_neurons INTEGER,
    learning_rate FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(training_id, step)
);

CREATE INDEX idx_metrics_training_step ON training_metrics(training_id, step);
```

### 3.3 Checkpoint Table
```sql
CREATE TABLE checkpoints (
    id UUID PRIMARY KEY,
    training_id UUID REFERENCES trainings(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 4. Training Loop Design

### 4.1 Celery Task
```python
@celery_app.task(bind=True, queue='sae')
def train_sae_task(self, training_id: str, config: dict):
    """
    Main training task executed on GPU worker.

    Steps:
    1. Load model and dataset
    2. Initialize SAE based on architecture
    3. Create activation buffer
    4. Training loop with metrics emission
    5. Checkpoint at intervals
    6. Save final SAE
    """
    training = load_training(training_id)
    training.status = 'running'
    training.started_at = datetime.now()
    save_training(training)

    try:
        # Initialize components
        model = load_model(training.model_id)
        dataset = load_tokenized_dataset(training.tokenization_id)
        sae = create_sae(config['architecture'], config['hyperparameters'])
        optimizer = torch.optim.Adam(sae.parameters(), lr=config['learning_rate'])

        # Training loop
        for step in range(config['num_steps']):
            # Get activation batch
            activations = get_activations(model, dataset, training.layer)

            # Forward pass
            x_hat, z, aux = sae(activations)
            losses = sae.loss(activations, x_hat, z, aux)

            # Backward pass
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()

            # Emit metrics (every 10 steps)
            if step % 10 == 0:
                emit_training_progress(training_id, step, losses)

            # Checkpoint (configurable interval)
            if step % config['checkpoint_interval'] == 0:
                save_checkpoint(sae, optimizer, step, training_id)

        # Save final SAE
        sae_path = save_sae(sae, training_id)
        training.status = 'completed'
        training.sae_path = sae_path
        training.completed_at = datetime.now()

    except Exception as e:
        training.status = 'failed'
        training.error_message = str(e)
        emit_training_failed(training_id, str(e))
        raise
    finally:
        save_training(training)
```

### 4.2 Activation Buffer
```python
class ActivationBuffer:
    """Efficiently collect and serve activation batches."""

    def __init__(self, model, dataset, layer: int, buffer_size: int = 100000):
        self.model = model
        self.dataset = dataset
        self.layer = layer
        self.buffer = torch.zeros(buffer_size, model.config.hidden_size)
        self.position = 0
        self.hook_handle = None

    def fill_buffer(self):
        """Run model forward passes to fill activation buffer."""
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        # Register hook
        layer_module = get_layer(self.model, self.layer)
        self.hook_handle = layer_module.register_forward_hook(hook_fn)

        # Run forward passes
        for batch in self.dataset:
            with torch.no_grad():
                self.model(batch)

        self.hook_handle.remove()
        self.buffer = torch.cat(activations)

    def sample(self, batch_size: int) -> Tensor:
        """Sample random batch from buffer."""
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        return self.buffer[indices]
```

---

## 5. Metrics Computation

### 5.1 Core Metrics
```python
def compute_metrics(x: Tensor, x_hat: Tensor, z: Tensor) -> dict:
    return {
        "loss": F.mse_loss(x_hat, x) + l1_alpha * z.abs().mean(),
        "recon_loss": F.mse_loss(x_hat, x).item(),
        "l0": (z > 0).float().sum(dim=-1).mean().item(),
        "l1": z.abs().mean().item(),
        "fvu": compute_fvu(x, x_hat),
        "dead_neurons": count_dead_neurons(z),
    }

def compute_fvu(x: Tensor, x_hat: Tensor) -> float:
    """Fraction of Variance Unexplained."""
    residual_var = (x - x_hat).var()
    total_var = x.var()
    return (residual_var / total_var).item()

def count_dead_neurons(z: Tensor, threshold: float = 1e-8) -> int:
    """Count neurons that never activate above threshold."""
    max_activations = z.max(dim=0).values
    return (max_activations < threshold).sum().item()
```

---

## 6. WebSocket Events

### 6.1 Channel: `training/{training_id}`

| Event | Payload | Frequency |
|-------|---------|-----------|
| `progress` | `{step, loss, l0, fvu, progress_pct}` | Every 10 steps |
| `metrics` | `{step, all_metrics_dict}` | Every 100 steps |
| `checkpoint` | `{step, checkpoint_path}` | At checkpoint interval |
| `completed` | `{sae_path, final_loss, final_l0}` | Once |
| `failed` | `{error, step}` | Once |

---

## 7. File Storage

### 7.1 SAE Output Structure
```
DATA_DIR/
└── saes/
    └── {training_id}/
        ├── cfg.json                    # SAELens-compatible config
        ├── sae_weights.safetensors     # Model weights
        ├── training_config.json        # Full hyperparameters
        └── checkpoints/
            ├── step_10000/
            │   ├── sae_weights.safetensors
            │   └── optimizer.pt
            └── step_20000/
                └── ...
```

### 7.2 cfg.json (SAELens Format)
```json
{
  "d_in": 2304,
  "d_sae": 18432,
  "dtype": "float32",
  "model_name": "google/gemma-2-2b",
  "hook_name": "blocks.12.hook_resid_post",
  "architecture": "standard",
  "normalize_activations": "constant_norm_rescale"
}
```

---

## 8. Training Templates

### 8.1 Template Schema
```sql
CREATE TABLE training_templates (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    hyperparameters JSONB NOT NULL,
    is_favorite BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 8.2 Template Structure
```json
{
  "name": "Gemma Scope Standard",
  "hyperparameters": {
    "architecture": "jumprelu",
    "latent_dim_multiplier": 8,
    "l1_alpha": 0.001,
    "learning_rate": 1e-4,
    "batch_size": 4096,
    "num_steps": 100000,
    "normalize_activations": "constant_norm_rescale",
    "checkpoint_interval": 10000
  }
}
```

---

## 9. Error Handling

| Error | Recovery |
|-------|----------|
| CUDA OOM | Reduce batch size, restart |
| NaN loss | Reduce learning rate, restart |
| Dead neurons > 50% | Enable resampling |
| Checkpoint corrupt | Restore previous checkpoint |

---

## 10. Celery Resilience (Added Dec 2025)

### 10.1 Task Configuration
```python
@celery_app.task(
    bind=True,
    queue='sae',
    max_retries=3,
    soft_time_limit=3600,    # 1 hour soft limit
    time_limit=7200,         # 2 hour hard limit
    acks_late=True,          # Acknowledge after completion
    reject_on_worker_lost=True
)
def train_sae_task(self, training_id: str, config: dict):
    """Training task with resilience patterns."""
```

### 10.2 Graceful Shutdown
```python
import signal

def handle_shutdown(signum, frame):
    """Handle graceful shutdown on SIGTERM/SIGINT."""
    logger.info("Received shutdown signal, saving checkpoint...")
    if current_sae and current_step:
        save_emergency_checkpoint(current_sae, current_step, training_id)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)
```

### 10.3 Retry with Exponential Backoff
```python
@celery_app.task(bind=True, max_retries=3)
def train_sae_task(self, training_id: str, config: dict):
    try:
        # ... training logic
    except SoftTimeLimitExceeded:
        # Save state before timeout
        save_checkpoint(sae, optimizer, current_step, training_id)
        update_training_status(training_id, 'paused', f'Paused at step {current_step}')
        raise
    except (ConnectionError, TimeoutError) as exc:
        # Retry with exponential backoff
        countdown = 60 * (2 ** self.request.retries)  # 60s, 120s, 240s
        raise self.retry(exc=exc, countdown=countdown)
    except Exception as exc:
        # Log full traceback for debugging
        logger.exception(f"Training {training_id} failed: {exc}")
        update_training_status(training_id, 'failed', str(exc))
        raise
```

### 10.4 State Persistence
Training state is persisted to database and Redis for recovery:
- **Database**: Training record with current_step, status, error_message
- **Redis**: Live progress cache for fast access
- **Checkpoints**: Full SAE + optimizer state at intervals

### 10.5 Dead Letter Queue
Failed tasks are routed to dead letter queue for manual inspection:
```python
celery_app.conf.task_routes = {
    'train_sae_task': {'queue': 'sae'},
    'train_sae_task.retry': {'queue': 'sae_retry'},
    'train_sae_task.failed': {'queue': 'sae_dlq'}
}
```

---

*Related: [PRD](../prds/003_FPRD|SAE_Training.md) | [TID](../tids/003_FTID|SAE_Training.md) | [FTASKS](../tasks/003_FTASKS|SAE_Training.md)*
