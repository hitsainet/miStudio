# Technical Implementation Document: Multi-GPU Scalability

**Document ID:** 009_FTID|Multi_GPU_Scalability
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Planned
**Related TDD:** [009_FTDD|Multi_GPU_Scalability](../tdds/009_FTDD|Multi_GPU_Scalability.md)

---

## 1. Implementation Order

### Phase 1: Enhanced Monitoring
1. Per-GPU metrics storage
2. Aggregated vs. per-GPU view toggle
3. Per-GPU chart components
4. Update WebSocket channels for multi-GPU

### Phase 2: GPU Selection
1. GPU availability service
2. GPU selection UI in training modal
3. Busy GPU detection
4. Memory-based recommendations

### Phase 3: Distributed Training
1. PyTorch DDP infrastructure
2. Modify training task for multi-process
3. Gradient synchronization
4. Distributed checkpointing

### Phase 4: Testing & Optimization
1. Test with 2, 4, 8 GPUs
2. Scaling efficiency benchmarks
3. Memory profiling
4. NCCL optimization

---

## 2. File-by-File Implementation

### 2.1 Backend - GPU Availability

#### `backend/src/services/gpu_availability_service.py`
```python
import pynvml
from typing import List, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session
from src.models.training import Training

@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total: int
    memory_used: int
    memory_free: int
    utilization: int
    is_busy: bool
    current_job: Optional[str] = None
    current_job_type: Optional[str] = None

class GPUAvailabilityService:
    """Service for GPU availability and selection."""

    def __init__(self, db: Session):
        self.db = db
        pynvml.nvmlInit()

    def get_all_gpus(self) -> List[GPUInfo]:
        """Get status of all GPUs."""
        gpus = []
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            is_busy, job_id, job_type = self._check_gpu_busy(i)

            gpus.append(GPUInfo(
                index=i,
                name=name,
                memory_total=mem.total,
                memory_used=mem.used,
                memory_free=mem.free,
                utilization=util.gpu,
                is_busy=is_busy,
                current_job=job_id,
                current_job_type=job_type
            ))

        return gpus

    def _check_gpu_busy(self, gpu_index: int) -> tuple:
        """Check if GPU is running a job."""
        # Check active trainings
        from sqlalchemy import func
        from sqlalchemy.dialects.postgresql import ARRAY

        training = self.db.query(Training).filter(
            Training.status == 'running',
            Training.gpu_ids.contains([gpu_index])
        ).first()

        if training:
            return True, str(training.id), 'training'

        # Could check other job types here (extraction, export, etc.)

        return False, None, None

    def get_available_gpus(self) -> List[GPUInfo]:
        """Get only available (not busy) GPUs."""
        return [g for g in self.get_all_gpus() if not g.is_busy]

    def recommend_gpus(
        self,
        memory_required_mb: int,
        count: int = 1
    ) -> List[int]:
        """Recommend GPUs based on memory requirements."""
        available = self.get_available_gpus()

        # Sort by free memory (most free first)
        available.sort(key=lambda g: g.memory_free, reverse=True)

        # Filter by memory requirement
        memory_bytes = memory_required_mb * 1024 * 1024
        suitable = [g for g in available if g.memory_free >= memory_bytes]

        # Return top N
        return [g.index for g in suitable[:count]]

    def validate_gpu_selection(self, gpu_ids: List[int]) -> dict:
        """Validate that selected GPUs are available."""
        all_gpus = {g.index: g for g in self.get_all_gpus()}
        result = {"valid": True, "errors": []}

        for gpu_id in gpu_ids:
            if gpu_id not in all_gpus:
                result["valid"] = False
                result["errors"].append(f"GPU {gpu_id} does not exist")
            elif all_gpus[gpu_id].is_busy:
                result["valid"] = False
                job = all_gpus[gpu_id].current_job
                result["errors"].append(
                    f"GPU {gpu_id} is busy (job: {job})"
                )

        return result

    def estimate_memory_requirement(
        self,
        model_name: str,
        d_sae: int,
        batch_size: int,
        dtype: str = "float32"
    ) -> int:
        """Estimate GPU memory requirement in MB."""
        # Rough estimation based on model size and SAE dimensions
        # This is a simplified heuristic

        # Model-specific hidden sizes (approximate)
        model_sizes = {
            "google/gemma-2-2b": 2304,
            "google/gemma-2-9b": 3584,
            "meta-llama/Llama-2-7b": 4096,
            "gpt2": 768
        }

        d_in = model_sizes.get(model_name, 2048)

        # SAE weights: W_enc + W_dec + biases
        sae_params = 2 * d_in * d_sae + d_in + d_sae

        # Bytes per param
        dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}
        bytes_per_param = dtype_bytes.get(dtype, 4)

        # Activations per batch
        activation_memory = batch_size * d_in * bytes_per_param

        # Total with overhead (2x for gradients, optimizer states)
        total_bytes = (sae_params * bytes_per_param * 4) + (activation_memory * 2)

        return int(total_bytes / (1024 * 1024))  # Convert to MB
```

### 2.2 Backend - Distributed Training

#### `backend/src/ml/distributed_training.py`
```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import List, Optional

class DistributedTrainer:
    """Manages distributed training across multiple GPUs."""

    def __init__(
        self,
        gpu_ids: List[int],
        master_addr: str = "localhost",
        master_port: str = "12355"
    ):
        self.gpu_ids = gpu_ids
        self.world_size = len(gpu_ids)
        self.master_addr = master_addr
        self.master_port = master_port
        self._initialized = False

    def setup(self, rank: int):
        """Initialize process group for this rank."""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port

        # Set visible devices for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_ids[rank])

        dist.init_process_group(
            backend='nccl',
            world_size=self.world_size,
            rank=rank
        )

        # After CUDA_VISIBLE_DEVICES is set, device 0 is the correct one
        torch.cuda.set_device(0)
        self._initialized = True

    def cleanup(self):
        """Clean up distributed process group."""
        if self._initialized:
            dist.destroy_process_group()
            self._initialized = False

    def wrap_model(self, model, rank: int) -> DDP:
        """Wrap model with DistributedDataParallel."""
        device = torch.device('cuda:0')  # Always 0 due to CUDA_VISIBLE_DEVICES
        model = model.to(device)
        return DDP(model, device_ids=[0])

    def create_distributed_dataloader(
        self,
        dataset,
        batch_size: int,
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
            batch_size=batch_size // self.world_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True  # Important for distributed training
        )

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """Synchronize tensor across all processes."""
        dist.all_reduce(tensor, op=op)
        return tensor

    def barrier(self):
        """Synchronization point for all processes."""
        dist.barrier()

    def is_main_process(self, rank: int) -> bool:
        """Check if this is the main (rank 0) process."""
        return rank == 0


def run_distributed_training(
    rank: int,
    world_size: int,
    gpu_ids: List[int],
    training_id: str,
    config: dict
):
    """Training function executed by each process."""
    from src.db.session import SessionLocal
    from src.services.training_service import TrainingService
    from src.ml.sparse_autoencoder import create_sae
    from src.workers.websocket_emitter import emit_training_progress

    trainer = DistributedTrainer(gpu_ids)
    trainer.setup(rank)

    db = SessionLocal()

    try:
        # Only main process updates database
        if trainer.is_main_process(rank):
            service = TrainingService(db)
            service.update_status(training_id, "running")
            emit_training_progress(training_id, 0, 0, "Initializing distributed training...")

        # Create and wrap model
        sae = create_sae(config)
        sae = trainer.wrap_model(sae, rank)

        # Load dataset and create distributed dataloader
        dataset = load_activations(config["activation_source"])
        dataloader = trainer.create_distributed_dataloader(
            dataset,
            config["batch_size"],
            rank
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=config["learning_rate"]
        )

        # Training loop
        total_steps = config["num_steps"]
        global_step = 0

        for epoch in range(config["num_epochs"]):
            dataloader.sampler.set_epoch(epoch)  # Important for shuffling

            for batch in dataloader:
                if global_step >= total_steps:
                    break

                batch = batch.cuda()

                # Forward
                x_hat, z, aux = sae(batch)
                losses = sae.module.loss(batch, x_hat, z, aux)
                loss = losses["loss"]

                # Backward (gradients auto-synced by DDP)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Only main process logs metrics
                if trainer.is_main_process(rank) and global_step % 100 == 0:
                    # Average loss across processes
                    avg_loss = trainer.all_reduce(loss.clone()) / world_size

                    service.add_metric(training_id, global_step, avg_loss.item())
                    progress = (global_step / total_steps) * 100
                    emit_training_progress(
                        training_id,
                        progress,
                        global_step,
                        avg_loss.item()
                    )

                global_step += 1

        # Save final model (only main process)
        trainer.barrier()  # Ensure all processes finished
        if trainer.is_main_process(rank):
            save_path = save_distributed_model(sae.module, training_id, config)
            service.update_status(training_id, "completed", output_path=save_path)
            emit_training_progress(training_id, 100, global_step, "Complete", completed=True)

    except Exception as e:
        if trainer.is_main_process(rank):
            service.update_status(training_id, "failed", error_message=str(e))
            emit_training_progress(training_id, 0, 0, str(e), failed=True)
        raise

    finally:
        trainer.cleanup()
        db.close()


def save_distributed_model(sae, training_id: str, config: dict) -> str:
    """Save model weights (called only from main process)."""
    from safetensors.torch import save_file
    from pathlib import Path
    import json

    output_dir = Path(settings.data_dir) / "saes" / training_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights (from unwrapped model)
    state_dict = {
        "W_enc": sae.W_enc.data.cpu(),
        "b_enc": sae.b_enc.data.cpu(),
        "W_dec": sae.W_dec.data.cpu(),
        "b_dec": sae.b_dec.data.cpu()
    }
    save_file(state_dict, output_dir / "sae_weights.safetensors")

    # Save config
    with open(output_dir / "cfg.json", "w") as f:
        json.dump({
            "d_in": sae.d_in,
            "d_sae": sae.d_sae,
            "architecture": config.get("architecture"),
            "model_name": config.get("model_name")
        }, f)

    return str(output_dir)
```

#### `backend/src/workers/distributed_tasks.py`
```python
import torch.multiprocessing as mp
from celery import shared_task
from src.ml.distributed_training import run_distributed_training

@shared_task(bind=True, queue='sae')
def train_sae_distributed_task(self, training_id: str, config: dict):
    """Celery task that spawns distributed training processes."""
    gpu_ids = config.get("gpu_ids", [0])
    world_size = len(gpu_ids)

    if world_size == 1:
        # Single GPU, use regular training
        from src.workers.training_tasks import train_sae_task
        return train_sae_task(training_id, config)

    # Multi-GPU: spawn processes
    mp.spawn(
        run_distributed_training,
        args=(world_size, gpu_ids, training_id, config),
        nprocs=world_size,
        join=True
    )
```

### 2.3 Frontend - GPU Selection

#### `frontend/src/components/training/GPUSelector.tsx`
```typescript
import React, { useEffect, useState } from 'react';
import { systemApi } from '../../api/system';
import { formatBytes } from '../../utils/formatters';

interface GPU {
  index: number;
  name: string;
  memory_total: number;
  memory_used: number;
  memory_free: number;
  utilization: number;
  is_busy: boolean;
  current_job?: string;
}

interface GPUSelectorProps {
  selected: number[];
  onChange: (gpuIds: number[]) => void;
  estimatedMemoryMB?: number;
}

export function GPUSelector({
  selected,
  onChange,
  estimatedMemoryMB
}: GPUSelectorProps) {
  const [gpus, setGpus] = useState<GPU[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadGpus();
    const interval = setInterval(loadGpus, 5000);  // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  const loadGpus = async () => {
    try {
      const data = await systemApi.getGpus();
      setGpus(data);
    } finally {
      setLoading(false);
    }
  };

  const toggleGpu = (index: number) => {
    if (selected.includes(index)) {
      onChange(selected.filter(i => i !== index));
    } else {
      onChange([...selected, index].sort());
    }
  };

  const getMemoryBarColor = (gpu: GPU) => {
    const usedPercent = (gpu.memory_used / gpu.memory_total) * 100;
    if (usedPercent > 90) return 'bg-red-500';
    if (usedPercent > 70) return 'bg-yellow-500';
    return 'bg-emerald-500';
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="text-sm text-slate-400">GPU Selection</label>
        <span className="text-xs text-slate-500">
          {selected.length} GPU{selected.length !== 1 ? 's' : ''} selected
        </span>
      </div>

      {estimatedMemoryMB && (
        <div className="text-xs text-slate-500 mb-2">
          Estimated memory: {estimatedMemoryMB.toLocaleString()} MB per GPU
        </div>
      )}

      <div className="space-y-2">
        {gpus.map(gpu => {
          const isSelected = selected.includes(gpu.index);
          const canSelect = !gpu.is_busy || isSelected;
          const hasEnoughMemory = !estimatedMemoryMB ||
            (gpu.memory_free / (1024 * 1024)) >= estimatedMemoryMB;

          return (
            <div
              key={gpu.index}
              onClick={() => canSelect && hasEnoughMemory && toggleGpu(gpu.index)}
              className={`
                p-3 rounded-lg border cursor-pointer transition
                ${isSelected
                  ? 'border-emerald-500 bg-emerald-500/10'
                  : gpu.is_busy
                    ? 'border-slate-700 bg-slate-800/50 cursor-not-allowed opacity-50'
                    : !hasEnoughMemory
                      ? 'border-yellow-500/50 bg-slate-800/50 cursor-not-allowed'
                      : 'border-slate-700 bg-slate-800 hover:border-slate-600'
                }
              `}
            >
              <div className="flex justify-between items-start">
                <div>
                  <div className="font-medium">GPU {gpu.index}</div>
                  <div className="text-xs text-slate-400">{gpu.name}</div>
                </div>

                {gpu.is_busy && (
                  <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded">
                    Busy
                  </span>
                )}

                {!hasEnoughMemory && !gpu.is_busy && (
                  <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded">
                    Low Memory
                  </span>
                )}
              </div>

              {/* Memory bar */}
              <div className="mt-2">
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Memory</span>
                  <span>
                    {formatBytes(gpu.memory_used)} / {formatBytes(gpu.memory_total)}
                  </span>
                </div>
                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${getMemoryBarColor(gpu)}`}
                    style={{
                      width: `${(gpu.memory_used / gpu.memory_total) * 100}%`
                    }}
                  />
                </div>
              </div>

              {/* Utilization */}
              <div className="mt-2 text-xs text-slate-400">
                Utilization: {gpu.utilization}%
              </div>
            </div>
          );
        })}
      </div>

      {selected.length > 1 && (
        <div className="text-xs text-emerald-400 mt-2">
          Multi-GPU training enabled. Effective batch size will be {selected.length}x.
        </div>
      )}
    </div>
  );
}
```

#### `frontend/src/components/training/TrainingForm.tsx` (Excerpt)
```typescript
// Add GPU selection to training form
import { GPUSelector } from './GPUSelector';

export function TrainingForm({ onSubmit }) {
  const [config, setConfig] = useState({
    // ... other fields
    gpu_ids: [0],
    distributed: false
  });

  const estimatedMemory = useMemo(() => {
    return estimateMemoryRequirement(
      config.model_name,
      config.d_sae,
      config.batch_size
    );
  }, [config.model_name, config.d_sae, config.batch_size]);

  return (
    <form onSubmit={handleSubmit}>
      {/* ... other fields ... */}

      <GPUSelector
        selected={config.gpu_ids}
        onChange={(gpuIds) => setConfig({
          ...config,
          gpu_ids: gpuIds,
          distributed: gpuIds.length > 1
        })}
        estimatedMemoryMB={estimatedMemory}
      />

      {/* ... submit button ... */}
    </form>
  );
}
```

### 2.4 Frontend - Per-GPU Monitoring

#### `frontend/src/components/SystemMonitor/PerGPUView.tsx`
```typescript
import React from 'react';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';
import { formatBytes } from '../../utils/formatters';
import { MiniChart } from './MiniChart';

export function PerGPUView() {
  const gpuMetrics = useSystemMonitorStore(s => s.gpuMetrics);
  const gpuHistory = useSystemMonitorStore(s => s.gpuHistory);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {Object.entries(gpuMetrics).map(([gpuId, metrics]) => (
        <div key={gpuId} className="bg-slate-800 rounded-lg p-4">
          <h3 className="font-medium mb-2">GPU {gpuId}</h3>
          <div className="text-xs text-slate-400 mb-3">{metrics.name}</div>

          {/* Mini utilization chart */}
          <MiniChart
            data={gpuHistory[parseInt(gpuId)] || []}
            dataKey="utilization"
            color="#10b981"
            height={60}
          />

          {/* Stats grid */}
          <div className="grid grid-cols-2 gap-2 mt-3 text-sm">
            <div>
              <div className="text-slate-400">Utilization</div>
              <div className="font-medium">{metrics.utilization}%</div>
            </div>
            <div>
              <div className="text-slate-400">Temperature</div>
              <div className="font-medium">{metrics.temperature}°C</div>
            </div>
            <div>
              <div className="text-slate-400">Memory</div>
              <div className="font-medium">
                {formatBytes(metrics.memory_used)} / {formatBytes(metrics.memory_total)}
              </div>
            </div>
            <div>
              <div className="text-slate-400">Power</div>
              <div className="font-medium">{metrics.power_draw}W</div>
            </div>
          </div>

          {/* Memory bar */}
          <div className="mt-3">
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500"
                style={{
                  width: `${(metrics.memory_used / metrics.memory_total) * 100}%`
                }}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
```

---

## 3. Common Patterns

### 3.1 Rank-Aware Logging
```python
# Only log from main process
if trainer.is_main_process(rank):
    logger.info(f"Step {step}: loss={loss.item():.4f}")
    emit_training_progress(...)
```

### 3.2 Gradient Synchronization Check
```python
# Verify gradients are synced (for debugging)
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_sum = trainer.all_reduce(param.grad.sum().clone())
        # grad_sum should be same across all ranks
```

### 3.3 Checkpoint Loading in Distributed Mode
```python
def load_checkpoint_distributed(model, checkpoint_path, rank):
    # Only rank 0 loads, then broadcasts
    if rank == 0:
        state_dict = torch.load(checkpoint_path)
    else:
        state_dict = None

    # Broadcast from rank 0
    objects = [state_dict]
    dist.broadcast_object_list(objects, src=0)
    state_dict = objects[0]

    model.load_state_dict(state_dict)
```

---

## 4. Testing Strategy

### 4.1 Multi-GPU Simulation
```python
# Test with simulated multi-GPU (single GPU, multiple processes)
def test_distributed_training_simulation():
    # Use NCCL backend with single GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Spawn 2 processes sharing the GPU
    mp.spawn(
        run_test_training,
        args=(2,),  # world_size=2
        nprocs=2
    )
```

### 4.2 Gradient Sync Tests
```python
def test_gradient_synchronization():
    # Create identical models on each GPU
    # Run same batch, verify gradients match
    pass
```

---

## 5. Common Pitfalls

### Pitfall 1: CUDA_VISIBLE_DEVICES Timing
```python
# WRONG - Set after CUDA initialized
import torch
torch.cuda.init()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Too late!

# RIGHT - Set before any CUDA operations
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
# Now torch.cuda sees only GPU 1 as device 0
```

### Pitfall 2: DataLoader Shuffle in DDP
```python
# WRONG - Using shuffle=True with DistributedSampler
dataloader = DataLoader(ds, shuffle=True, sampler=DistributedSampler(ds))

# RIGHT - Let sampler handle shuffling
sampler = DistributedSampler(ds, shuffle=True)
dataloader = DataLoader(ds, sampler=sampler, shuffle=False)
# Also: sampler.set_epoch(epoch) each epoch
```

### Pitfall 3: Model Access After DDP Wrap
```python
# WRONG - Accessing attributes on DDP model
ddp_model = DDP(model)
d_sae = ddp_model.d_sae  # AttributeError!

# RIGHT - Access via .module
d_sae = ddp_model.module.d_sae
```

---

## 6. Performance Tips

1. **NCCL Environment Variables**
   ```python
   # Optimize NCCL for your network
   os.environ['NCCL_DEBUG'] = 'INFO'
   os.environ['NCCL_IB_DISABLE'] = '1'  # If no InfiniBand
   os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Specify interface
   ```

2. **Gradient Bucket Size**
   ```python
   # Larger buckets = fewer sync ops = faster
   ddp_model = DDP(model, bucket_cap_mb=25)
   ```

3. **Mixed Precision with DDP**
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()

   with autocast():
       output = ddp_model(input)
       loss = compute_loss(output)

   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

---

*Related: [PRD](../prds/009_FPRD|Multi_GPU_Scalability.md) | [TDD](../tdds/009_FTDD|Multi_GPU_Scalability.md) | [FTASKS](../tasks/009_FTASKS|Multi_GPU_Scalability.md)*
