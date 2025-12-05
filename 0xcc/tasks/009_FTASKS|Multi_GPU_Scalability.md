# Feature Tasks: Multi-GPU Scalability

**Document ID:** 009_FTASKS|Multi_GPU_Scalability
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Planned
**Related PRD:** [009_FPRD|Multi_GPU_Scalability](../prds/009_FPRD|Multi_GPU_Scalability.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Enhanced Monitoring | 4 tasks | ⏳ Planned |
| Phase 2: GPU Selection | 5 tasks | ⏳ Planned |
| Phase 3: Distributed Training | 6 tasks | ⏳ Planned |
| Phase 4: Testing | 4 tasks | ⏳ Planned |

**Total: 19 tasks**

---

## Phase 1: Enhanced Monitoring (Week 1-2)

### Task 1.1: Per-GPU Metrics Storage
- [ ] Create gpu_metrics_history table
- [ ] Add job association columns
- [ ] Add time-based indexes
- [ ] Implement cleanup for old data

**Files:**
- `backend/alembic/versions/xxx_create_gpu_metrics_history.py`

### Task 1.2: Update System Monitor Service
- [ ] Store metrics in database (optional)
- [ ] Add job tracking per GPU
- [ ] Track which job uses which GPU

### Task 1.3: Per-GPU vs Aggregated View
- [ ] Create view toggle in frontend
- [ ] Implement AggregatedView component
- [ ] Implement PerGPUView component
- [ ] Add to systemMonitorStore

**Files:**
- `frontend/src/components/SystemMonitor/PerGPUView.tsx`
- `frontend/src/components/SystemMonitor/AggregatedView.tsx`

### Task 1.4: Multi-GPU WebSocket Channels
- [ ] Dynamic channel subscription
- [ ] Handle variable GPU count
- [ ] Efficient subscription management

---

## Phase 2: GPU Selection (Week 3)

### Task 2.1: GPU Availability Service
- [ ] Implement get_all_gpus() method
- [ ] Implement _check_gpu_busy() method
- [ ] Implement get_available_gpus() method
- [ ] Implement recommend_gpus() method

**Files:**
- `backend/src/services/gpu_availability_service.py`

### Task 2.2: Memory Estimation
- [ ] Implement estimate_memory_requirement()
- [ ] Consider model size
- [ ] Consider batch size
- [ ] Consider dtype

### Task 2.3: GPU Selection API
- [ ] GET /system/gpus - List with availability
- [ ] GET /system/gpus/recommend - Recommendations
- [ ] POST /system/gpus/validate - Validate selection

**Files:**
- `backend/src/api/v1/endpoints/system.py`

### Task 2.4: GPUSelector Component
- [ ] Display all GPUs
- [ ] Show availability status
- [ ] Show memory usage
- [ ] Multi-select support
- [ ] Memory requirement indicator

**Files:**
- `frontend/src/components/training/GPUSelector.tsx`

### Task 2.5: Integrate into Training Form
- [ ] Add GPUSelector to form
- [ ] Update config schema
- [ ] Pass gpu_ids to backend

---

## Phase 3: Distributed Training (Week 4-6)

### Task 3.1: Create DistributedTrainer Class
- [ ] Implement setup() for process group
- [ ] Implement wrap_model() for DDP
- [ ] Implement create_distributed_dataloader()
- [ ] Implement cleanup()

**Files:**
- `backend/src/ml/distributed_training.py`

### Task 3.2: Distributed Training Function
- [ ] Handle rank-based initialization
- [ ] Set CUDA_VISIBLE_DEVICES
- [ ] Create per-process dataloader
- [ ] Handle gradient synchronization

### Task 3.3: Modify Training Task
- [ ] Check for multi-GPU config
- [ ] Use mp.spawn for multi-process
- [ ] Fall back to single GPU for 1 GPU

**Files:**
- `backend/src/workers/distributed_tasks.py`

### Task 3.4: Distributed Checkpointing
- [ ] Save only from rank 0
- [ ] Broadcast checkpoint for loading
- [ ] Handle model unwrapping

### Task 3.5: Progress Emission
- [ ] Only emit from rank 0
- [ ] Average metrics across GPUs
- [ ] Track per-GPU stats

### Task 3.6: Database Schema Updates
- [ ] Add gpu_ids column to trainings
- [ ] Add distributed column
- [ ] Add world_size column

**Files:**
- `backend/alembic/versions/xxx_add_multi_gpu_columns.py`

---

## Phase 4: Testing & Optimization (Week 7)

### Task 4.1: Multi-GPU Simulation Test
- [ ] Test with simulated multi-GPU
- [ ] Verify gradient sync
- [ ] Test checkpoint save/load

### Task 4.2: Scaling Efficiency Benchmark
- [ ] Test 2 GPU scaling
- [ ] Test 4 GPU scaling
- [ ] Test 8 GPU scaling
- [ ] Document scaling efficiency

### Task 4.3: Memory Profiling
- [ ] Profile memory per GPU
- [ ] Optimize batch distribution
- [ ] Identify memory bottlenecks

### Task 4.4: NCCL Optimization
- [ ] Configure NCCL environment
- [ ] Test bucket sizes
- [ ] Optimize for network topology

---

## Implementation Notes

### Prerequisites
- PyTorch with NCCL support
- Multiple NVIDIA GPUs with NVLink (preferred)
- Sufficient CPU memory for data loading

### Key Considerations
1. **CUDA_VISIBLE_DEVICES**: Must set before CUDA initialization
2. **DistributedSampler**: Required for proper data sharding
3. **Gradient Sync**: Handled automatically by DDP
4. **Checkpointing**: Only save from rank 0
5. **Logging**: Only log/emit from rank 0

### Testing Strategy
1. Start with 2 GPUs
2. Verify training completes
3. Verify loss matches single-GPU (approximately)
4. Scale to 4 GPUs
5. Benchmark throughput scaling

---

## Relevant Files Summary

### Backend (To Create)
| File | Purpose |
|------|---------|
| `backend/src/services/gpu_availability_service.py` | GPU availability |
| `backend/src/ml/distributed_training.py` | DDP infrastructure |
| `backend/src/workers/distributed_tasks.py` | Distributed Celery task |
| `backend/alembic/versions/xxx_add_multi_gpu_columns.py` | Schema updates |

### Frontend (To Create)
| File | Purpose |
|------|---------|
| `frontend/src/components/training/GPUSelector.tsx` | GPU selection UI |
| `frontend/src/components/SystemMonitor/PerGPUView.tsx` | Per-GPU view |
| `frontend/src/components/SystemMonitor/AggregatedView.tsx` | Aggregated view |

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| PyTorch | >= 2.0 | DDP support |
| NCCL | >= 2.10 | GPU communication |
| pynvml | >= 11.0 | GPU detection |

---

*Related: [PRD](../prds/009_FPRD|Multi_GPU_Scalability.md) | [TDD](../tdds/009_FTDD|Multi_GPU_Scalability.md) | [TID](../tids/009_FTID|Multi_GPU_Scalability.md)*
