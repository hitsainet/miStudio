# Technical Design Document: Multi-GPU Scalability

**Document ID:** 009_FTDD|Multi_GPU_Scalability
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Planned
**Related PRD:** [009_FPRD|Multi_GPU_Scalability](../prds/009_FPRD|Multi_GPU_Scalability.md)

---

## 1. System Architecture

### 1.1 Distributed Training Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                   Distributed Training Architecture              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Training Orchestrator                     │ │
│  │                 (Main Process - Rank 0)                     │ │
│  └────────────────────────┬───────────────────────────────────┘ │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐            │
│  │   GPU 0    │   │   GPU 1    │   │   GPU 2    │            │
│  │  (Rank 0)  │   │  (Rank 1)  │   │  (Rank 2)  │            │
│  │            │   │            │   │            │            │
│  │ SAE Replica│   │ SAE Replica│   │ SAE Replica│            │
│  │ Data Shard │   │ Data Shard │   │ Data Shard │            │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘            │
│        │                │                │                    │
│        └────────────────┼────────────────┘                    │
│                         ▼                                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              NCCL All-Reduce                            │  │
│  │         (Gradient Synchronization)                      │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Monitoring View Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                   Enhanced Monitoring Views                      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    View Toggle                              │ │
│  │            [Aggregated ●] [Per-GPU ○]                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                     │
│         ┌─────────────────┴─────────────────┐                  │
│         ▼                                   ▼                  │
│  ┌────────────────────┐          ┌────────────────────────┐   │
│  │   Aggregated View  │          │     Per-GPU View       │   │
│  │                    │          │                        │   │
│  │ Total VRAM: 48GB   │          │ GPU 0: 16GB  ████████ │   │
│  │ ████████████████   │          │ GPU 1: 14GB  ███████░ │   │
│  │                    │          │ GPU 2: 18GB  █████████│   │
│  │ Avg Util: 78%      │          │                        │   │
│  └────────────────────┘          └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Database Schema Extensions

### 2.1 Training Table Extension
```sql
-- Add multi-GPU support columns
ALTER TABLE trainings ADD COLUMN gpu_ids INTEGER[];
ALTER TABLE trainings ADD COLUMN distributed BOOLEAN DEFAULT FALSE;
ALTER TABLE trainings ADD COLUMN world_size INTEGER DEFAULT 1;

-- Example: training on GPUs 0 and 1
-- gpu_ids = [0, 1], distributed = true, world_size = 2
```

### 2.2 GPU Metrics Table
```sql
CREATE TABLE gpu_metrics_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gpu_index INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    utilization FLOAT NOT NULL,
    memory_used BIGINT NOT NULL,
    memory_total BIGINT NOT NULL,
    temperature FLOAT,
    power_draw FLOAT,
    job_id UUID,           -- Which job is using this GPU (nullable)
    job_type VARCHAR(50),  -- 'training', 'extraction', etc.

    UNIQUE(gpu_index, timestamp)
);

CREATE INDEX idx_gpu_metrics_time ON gpu_metrics_history(timestamp DESC);
CREATE INDEX idx_gpu_metrics_gpu ON gpu_metrics_history(gpu_index);
```

---

## 3. Distributed Training Implementation

### 3.1 PyTorch DDP Setup
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedTrainer:
    def __init__(
        self,
        sae: nn.Module,
        gpu_ids: List[int],
        training_config: TrainingConfig
    ):
        self.gpu_ids = gpu_ids
        self.world_size = len(gpu_ids)
        self.config = training_config

    def setup(self, rank: int):
        """Initialize distributed training for this process."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            world_size=self.world_size,
            rank=rank
        )

        # Set device for this process
        torch.cuda.set_device(self.gpu_ids[rank])

    def wrap_model(self, sae: nn.Module, rank: int) -> DDP:
        """Wrap model with DDP."""
        device = torch.device(f'cuda:{self.gpu_ids[rank]}')
        sae = sae.to(device)
        return DDP(sae, device_ids=[self.gpu_ids[rank]])

    def create_distributed_dataloader(
        self,
        dataset: Dataset,
        rank: int
    ) -> DataLoader:
        """Create dataloader with distributed sampler."""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=rank,
            shuffle=True
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size // self.world_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
```

### 3.2 Distributed Training Task
```python
def run_distributed_training(rank: int, world_size: int, training_id: str, config: dict):
    """Training function for each process/GPU."""
    trainer = DistributedTrainer(config['gpu_ids'], config)
    trainer.setup(rank)

    # Load and wrap model
    sae = create_sae(config)
    sae = trainer.wrap_model(sae, rank)

    # Create distributed dataloader
    dataset = load_dataset(config)
    dataloader = trainer.create_distributed_dataloader(dataset, rank)

    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=config['learning_rate'])

    # Training loop
    for epoch in range(config['num_epochs']):
        dataloader.sampler.set_epoch(epoch)

        for batch in dataloader:
            batch = batch.to(f'cuda:{trainer.gpu_ids[rank]}')

            # Forward
            x_hat, z, aux = sae(batch)
            loss = sae.module.loss(batch, x_hat, z, aux)['loss']

            # Backward (gradients auto-synced by DDP)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Only rank 0 emits metrics
            if rank == 0:
                emit_training_progress(training_id, step, loss.item())

    # Cleanup
    dist.destroy_process_group()


@celery_app.task(bind=True, queue='sae')
def train_sae_distributed_task(self, training_id: str, config: dict):
    """Celery task that spawns distributed training processes."""
    world_size = len(config['gpu_ids'])

    mp.spawn(
        run_distributed_training,
        args=(world_size, training_id, config),
        nprocs=world_size,
        join=True
    )
```

---

## 4. GPU Selection System

### 4.1 GPU Availability Service
```python
class GPUAvailabilityService:
    def get_available_gpus(self) -> List[GPUInfo]:
        """Get list of available GPUs with their status."""
        pynvml.nvmlInit()
        gpus = []

        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            gpu = GPUInfo(
                index=i,
                name=pynvml.nvmlDeviceGetName(handle),
                memory_total=pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                memory_used=pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                utilization=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                is_busy=self._check_if_busy(i),
                current_job=self._get_current_job(i)
            )
            gpus.append(gpu)

        return gpus

    def _check_if_busy(self, gpu_index: int) -> bool:
        """Check if GPU is currently running a job."""
        # Check active trainings
        active = db.query(Training).filter(
            Training.status == 'running',
            Training.gpu_ids.contains([gpu_index])
        ).first()
        return active is not None

    def recommend_gpus(self, memory_required: int) -> List[int]:
        """Recommend GPUs based on memory requirements."""
        gpus = self.get_available_gpus()
        available = [g for g in gpus if not g.is_busy]

        # Sort by free memory
        available.sort(key=lambda g: g.memory_total - g.memory_used, reverse=True)

        # Find GPUs that can fit the workload
        recommended = []
        for gpu in available:
            free_memory = gpu.memory_total - gpu.memory_used
            if free_memory >= memory_required:
                recommended.append(gpu.index)

        return recommended
```

### 4.2 API Endpoints
```python
@router.get("/system/gpus")
async def list_gpus() -> List[GPUInfo]:
    """List all GPUs with availability status."""
    service = GPUAvailabilityService()
    return service.get_available_gpus()

@router.get("/system/gpus/recommend")
async def recommend_gpus(memory_mb: int) -> List[int]:
    """Recommend GPUs for a workload."""
    service = GPUAvailabilityService()
    return service.recommend_gpus(memory_mb * 1024 * 1024)
```

---

## 5. Enhanced Monitoring

### 5.1 Aggregated Metrics Calculation
```python
class AggregatedMetricsService:
    def compute_aggregated(self, gpu_metrics: List[GPUMetrics]) -> AggregatedMetrics:
        """Compute aggregated metrics across all GPUs."""
        return AggregatedMetrics(
            total_memory_used=sum(g.memory_used for g in gpu_metrics),
            total_memory_total=sum(g.memory_total for g in gpu_metrics),
            average_utilization=sum(g.utilization for g in gpu_metrics) / len(gpu_metrics),
            max_temperature=max(g.temperature for g in gpu_metrics),
            total_power_draw=sum(g.power_draw for g in gpu_metrics),
            gpu_count=len(gpu_metrics)
        )
```

### 5.2 Frontend View Toggle
```typescript
interface MonitoringViewState {
  viewMode: 'aggregated' | 'per_gpu';
  setViewMode: (mode: 'aggregated' | 'per_gpu') => void;
}

// Component
function SystemMonitor() {
  const { viewMode, setViewMode } = useMonitoringViewStore();
  const gpuMetrics = useSystemMonitorStore(s => s.gpuMetrics);

  return (
    <div>
      <ViewToggle
        options={['aggregated', 'per_gpu']}
        value={viewMode}
        onChange={setViewMode}
      />

      {viewMode === 'aggregated' ? (
        <AggregatedView metrics={computeAggregated(gpuMetrics)} />
      ) : (
        <PerGPUView metrics={gpuMetrics} />
      )}
    </div>
  );
}
```

### 5.3 Per-GPU View Component
```typescript
function PerGPUView({ metrics }: { metrics: Record<number, GPUMetrics[]> }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {Object.entries(metrics).map(([gpuIndex, history]) => (
        <GPUCard
          key={gpuIndex}
          gpuIndex={parseInt(gpuIndex)}
          currentMetrics={history[history.length - 1]}
          history={history}
        />
      ))}
    </div>
  );
}

function GPUCard({ gpuIndex, currentMetrics, history }) {
  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h3>GPU {gpuIndex}</h3>

      <div className="space-y-2">
        <ProgressBar
          label="Utilization"
          value={currentMetrics.utilization}
          max={100}
          unit="%"
        />
        <ProgressBar
          label="Memory"
          value={currentMetrics.memory_used / (1024**3)}
          max={currentMetrics.memory_total / (1024**3)}
          unit="GB"
        />
        <div className="flex justify-between text-sm">
          <span>Temp: {currentMetrics.temperature}°C</span>
          <span>Power: {currentMetrics.power_draw}W</span>
        </div>
      </div>

      <MiniChart data={history} dataKey="utilization" />
    </div>
  );
}
```

---

## 6. Training Configuration Updates

### 6.1 Extended Training Request
```python
class TrainingCreateRequest(BaseModel):
    # ... existing fields ...

    # Multi-GPU options
    gpu_ids: Optional[List[int]] = None  # None = auto-select
    distributed: bool = False
```

### 6.2 Batch Size Scaling
```python
def calculate_effective_batch_size(
    base_batch_size: int,
    num_gpus: int,
    distributed: bool
) -> Tuple[int, int]:
    """
    Calculate effective and per-GPU batch sizes.

    Returns:
        (effective_batch_size, per_gpu_batch_size)
    """
    if distributed:
        # Each GPU processes base_batch_size
        # Effective = base * num_gpus
        return base_batch_size * num_gpus, base_batch_size
    else:
        # Single GPU, no change
        return base_batch_size, base_batch_size
```

---

## 7. Implementation Phases

### Phase 1: Enhanced Monitoring (Week 1-2)
- [ ] Add per-GPU metrics storage
- [ ] Implement aggregated vs. per-GPU view toggle
- [ ] Create per-GPU chart components
- [ ] Update WebSocket channels for multi-GPU

### Phase 2: GPU Selection (Week 3)
- [ ] Implement GPUAvailabilityService
- [ ] Add GPU selection UI in training modal
- [ ] Implement busy GPU detection
- [ ] Add memory-based recommendations

### Phase 3: Distributed Training (Week 4-6)
- [ ] Set up PyTorch DDP infrastructure
- [ ] Modify training task for multi-process
- [ ] Implement gradient synchronization
- [ ] Handle checkpointing in distributed mode
- [ ] Test with 2, 4, and 8 GPUs

---

## 8. Error Handling

| Error | Cause | Handling |
|-------|-------|----------|
| NCCL timeout | Network issue | Retry with longer timeout |
| GPU OOM | Uneven memory | Rebalance batch sizes |
| Process crash | GPU error | Cleanup and restart |
| Straggler GPU | Slow GPU | Log warning, continue |

---

*Related: [PRD](../prds/009_FPRD|Multi_GPU_Scalability.md) | [TID](../tids/009_FTID|Multi_GPU_Scalability.md) | [FTASKS](../tasks/009_FTASKS|Multi_GPU_Scalability.md)*
