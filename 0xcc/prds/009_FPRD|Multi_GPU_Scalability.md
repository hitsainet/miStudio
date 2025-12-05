# Feature PRD: Multi-GPU Scalability

**Document ID:** 009_FPRD|Multi_GPU_Scalability
**Version:** 1.0 (Planning)
**Last Updated:** 2025-12-05
**Status:** Planned
**Priority:** P2 (Future Feature)

---

## 1. Overview

### 1.1 Purpose
Enable distributed SAE training across multiple GPUs and provide enhanced monitoring with aggregated vs. per-GPU views.

### 1.2 User Problem
Researchers with multi-GPU systems cannot fully utilize their hardware:
- Training runs on single GPU only
- No visibility into per-GPU resource usage
- Cannot leverage data parallelism for faster training
- Memory constraints on single GPU limit model/SAE size

### 1.3 Solution
Multi-GPU support with distributed training, configurable GPU selection, and enhanced monitoring views.

---

## 2. Functional Requirements

### 2.1 Distributed Training
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | Data parallel training across GPUs | Planned |
| FR-1.2 | Gradient synchronization | Planned |
| FR-1.3 | GPU selection for training jobs | Planned |
| FR-1.4 | Automatic batch size scaling | Planned |
| FR-1.5 | Mixed precision per GPU | Planned |

### 2.2 GPU Selection
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | List available GPUs with specs | Planned |
| FR-2.2 | Select GPUs for training job | Planned |
| FR-2.3 | Exclude busy GPUs | Planned |
| FR-2.4 | Memory-based GPU recommendation | Planned |

### 2.3 Enhanced Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | Toggle: Aggregated vs. Per-GPU view | Planned |
| FR-3.2 | Aggregated VRAM usage (total across GPUs) | Planned |
| FR-3.3 | Aggregated utilization (average) | Planned |
| FR-3.4 | Per-GPU separate meters | Planned |
| FR-3.5 | Per-GPU temperature/power display | Planned |

### 2.4 Load Balancing
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | Automatic workload distribution | Planned |
| FR-4.2 | Memory-aware batch allocation | Planned |
| FR-4.3 | Straggler detection and handling | Planned |

---

## 3. Architecture Design

### 3.1 Distributed Training Pattern
```
┌─────────────────────────────────────────────────────────────┐
│                    Training Orchestrator                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Data Loader (Shared)                     │   │
│  │         Batch splitting across GPUs                   │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│         ┌───────────────┼───────────────┐                   │
│         ▼               ▼               ▼                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   GPU 0    │  │   GPU 1    │  │   GPU 2    │            │
│  │ SAE Replica│  │ SAE Replica│  │ SAE Replica│            │
│  │            │  │            │  │            │            │
│  │ Forward    │  │ Forward    │  │ Forward    │            │
│  │ Backward   │  │ Backward   │  │ Backward   │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
│        │               │               │                    │
│        └───────────────┼───────────────┘                    │
│                        ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Gradient Synchronization                 │   │
│  │           (All-Reduce via NCCL)                      │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Optimizer Step (Primary)                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Monitoring View Toggle
```
                    [Aggregated ●] [Per-GPU ○]

Aggregated View:                 Per-GPU View:
┌────────────────┐               ┌────────────────┐
│ Total VRAM     │               │ GPU 0: 6.2/8GB │
│ ████████░░░░░░ │               │ ████████░░░░░░ │
│ 18.6 / 24 GB   │               │ GPU 1: 5.8/8GB │
│                │               │ ███████░░░░░░░ │
│ Avg Util: 82%  │               │ GPU 2: 6.6/8GB │
└────────────────┘               │ █████████░░░░░ │
                                 └────────────────┘
```

---

## 4. User Interface

### 4.1 GPU Selection in Training Modal
```
┌─────────────────────────────────────────────────────────────┐
│ Start Training                                              │
├─────────────────────────────────────────────────────────────┤
│ GPU Selection                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [✓] GPU 0: NVIDIA RTX 3090 (24GB) - Available          │ │
│ │ [✓] GPU 1: NVIDIA RTX 3090 (24GB) - Available          │ │
│ │ [ ] GPU 2: NVIDIA RTX 3080 (10GB) - In Use (Training)  │ │
│ └─────────────────────────────────────────────────────────┘ │
│ Selected: 2 GPUs | Effective Batch Size: 8192               │
├─────────────────────────────────────────────────────────────┤
│ [Continue to Hyperparameters →]                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Enhanced System Monitor
```
┌─────────────────────────────────────────────────────────────┐
│ System Monitor              [Aggregated ●] [Per-GPU ○]      │
├─────────────────────────────────────────────────────────────┤
│ Per-GPU View:                                               │
│ ┌─────────────────────────┐ ┌─────────────────────────────┐ │
│ │ GPU 0 (RTX 3090)        │ │ GPU 1 (RTX 3090)            │ │
│ │ Util: ████████░░ 82%    │ │ Util: ███████░░░ 75%        │ │
│ │ Mem:  ████████░░ 6.2GB  │ │ Mem:  ███████░░░ 5.8GB      │ │
│ │ Temp: 68°C | 245W       │ │ Temp: 65°C | 230W           │ │
│ └─────────────────────────┘ └─────────────────────────────┘ │
│ ┌─────────────────────────┐                                 │
│ │ GPU 2 (RTX 3080)        │                                 │
│ │ Util: █████████░ 92%    │                                 │
│ │ Mem:  █████████░ 8.9GB  │                                 │
│ │ Temp: 72°C | 280W       │                                 │
│ └─────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/system/gpus` | GET | List available GPUs with status |
| `/api/v1/system/gpus/{id}` | GET | Get specific GPU details |
| `/api/v1/trainings` | POST | Extended with `gpu_ids` field |
| `/api/v1/system/metrics?view=aggregated` | GET | Aggregated metrics |
| `/api/v1/system/metrics?view=per_gpu` | GET | Per-GPU metrics |

---

## 6. Data Model Extensions

### 6.1 Training Table Extension
```sql
ALTER TABLE trainings ADD COLUMN gpu_ids INTEGER[];  -- Selected GPU indices
ALTER TABLE trainings ADD COLUMN distributed BOOLEAN DEFAULT FALSE;
```

### 6.2 SystemMetrics Extension
```sql
CREATE TABLE gpu_metrics (
    id UUID PRIMARY KEY,
    gpu_index INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    utilization FLOAT,
    memory_used BIGINT,
    memory_total BIGINT,
    temperature FLOAT,
    power_draw FLOAT,
    job_id UUID,  -- Which job is using this GPU
    UNIQUE(gpu_index, timestamp)
);
```

---

## 7. Implementation Approach

### 7.1 PyTorch Distributed
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(sae, device_ids=[local_rank])

# Training loop with gradient sync
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # Gradients auto-synced
    optimizer.step()
```

### 7.2 Batch Size Scaling
```python
effective_batch_size = base_batch_size * num_gpus
per_gpu_batch_size = base_batch_size
```

---

## 8. Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `multi_gpu_enabled` | Enable multi-GPU features | false |
| `default_gpu_selection` | all, available, manual | available |
| `monitor_view_default` | aggregated, per_gpu | aggregated |

---

## 9. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| SAE Training | Extends training system |
| System Monitoring | Extends monitoring views |

---

## 10. Implementation Phases

### Phase 1: Enhanced Monitoring
- [ ] Per-GPU metrics collection
- [ ] Aggregated vs. per-GPU view toggle
- [ ] Per-GPU charts in dashboard

### Phase 2: GPU Selection
- [ ] GPU availability detection
- [ ] GPU selection UI in training modal
- [ ] Busy GPU exclusion

### Phase 3: Distributed Training
- [ ] PyTorch DDP integration
- [ ] Gradient synchronization
- [ ] Batch size scaling
- [ ] Multi-GPU progress tracking

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| NCCL communication overhead | Performance | Profile and optimize |
| Memory imbalance across GPUs | Training failure | Memory-aware allocation |
| Straggler GPUs | Slowdown | Load balancing |
| Single point of failure | Training loss | Checkpoint frequently |

---

## 12. Success Metrics

| Metric | Target |
|--------|--------|
| Training speedup (2 GPUs) | 1.8x |
| Training speedup (4 GPUs) | 3.2x |
| Memory efficiency | >90% |
| GPU utilization (distributed) | >80% |

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/009_FTDD|Multi_GPU_Scalability.md) | [TID](../tids/009_FTID|Multi_GPU_Scalability.md)*
