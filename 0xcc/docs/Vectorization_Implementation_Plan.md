# Feature Extraction Vectorization - Research & Implementation Plan

## Executive Summary

**Goal:** Reduce extraction time from 31 hours to 2-4 hours (10-50x speedup) by replacing sequential Python loops with vectorized NumPy/PyTorch operations.

**Clarification:** This is about **code vectorization** (using NumPy/PyTorch vector operations), NOT PostgreSQL vector extensions (pgvector). The bottleneck is in Python code processing, not database operations.

---

## Part 1: Research & Analysis

### Current Bottleneck Analysis

#### Current Implementation (Sequential)
```python
# Line 1117-1155 in extraction_service.py
for neuron_idx in range(latent_dim):  # 16,384 iterations per sample
    neuron_activations = sae_features[:, neuron_idx].cpu().numpy()  # CPU←GPU copy
    max_activation = float(neuron_activations.max())
    
    if max_activation > 0:
        feature_activation_counts[neuron_idx] += 1
        
        top_positions = np.argsort(neuron_activations)[-5:][::-1]
        example = {...}
        
        # Heap management (find min, insert, or replace)
        if len(feature_activations[neuron_idx]) < top_k_examples:
            heapq.heappush(...)
        elif max_activation > feature_activations[neuron_idx][0][0]:
            heapq.heapreplace(...)
```

**Performance Issues:**
1. **16,384 iterations** per sample (1.6 billion total for 100k samples)
2. **CPU↔GPU memory transfer** for each feature individually
3. **Sequential heap operations** cannot be parallelized
4. **Python loop overhead** dominates execution time

**Time Breakdown Per Sample:**
- GPU forward pass: ~30ms (1%)
- CPU feature processing: ~500-1000ms (99%)
- **Total:** ~1 second per sample

---

### Vectorization Strategy Research

#### Option 1: NumPy Vectorization (Recommended - Easiest)
**Approach:** Replace Python loops with NumPy array operations

**Key Operations:**
```python
# Instead of looping over features:
for neuron_idx in range(latent_dim):
    max_activation = neuron_activations.max()
    
# Use vectorized:
all_activations = sae_features.cpu().numpy()  # (seq_len, latent_dim)
max_activations = all_activations.max(axis=0)  # All features at once
```

**Pros:**
- Simple to implement (4-6 hours)
- 10-20x speedup
- CPU-bound operations stay on CPU (good for memory)
- Well-tested NumPy operations

**Cons:**
- Still requires GPU→CPU transfer (once per sample)
- Limited by single-core NumPy operations for some ops

---

#### Option 2: PyTorch GPU Vectorization (Recommended - Best Performance)
**Approach:** Keep all operations on GPU using PyTorch tensors

**Key Operations:**
```python
# Keep everything on GPU
sae_features_gpu = sae.encode(sample_activations)  # Already on GPU
max_activations = sae_features_gpu.max(dim=0).values  # GPU operation
top_k_indices = torch.topk(sae_features_gpu, k=5, dim=0)  # GPU operation
```

**Pros:**
- Maximum performance (20-50x speedup)
- No GPU↔CPU transfers until final results
- Parallelizes across GPU cores
- Memory efficient (fewer copies)

**Cons:**
- Heap management needs redesign (PyTorch doesn't have heaps)
- More complex implementation (6-8 hours)
- GPU memory constraints for very large feature sets

---

#### Option 3: Hybrid Approach (Recommended - Balanced)
**Approach:** GPU vectorization + CPU batch processing

**Strategy:**
1. Process entire batch on GPU (vectorized)
2. Transfer final results to CPU once
3. Use NumPy for aggregation and storage

**Pros:**
- Best balance of performance and simplicity
- 15-30x speedup
- Handles large batches well
- Clear separation of GPU (compute) and CPU (storage)

**Cons:**
- Slightly more complex than pure NumPy
- Requires careful memory management

---

### Technical Decisions

#### Decision 1: Heap Replacement Strategy

**Current:** Python `heapq` for maintaining top-k examples per feature

**Problem:** Heaps are inherently sequential (can't be vectorized)

**Solutions:**

**A. Deferred Heap Construction (Recommended)**
```python
# During extraction: Store ALL examples in memory (if feasible)
feature_examples = []  # List of (feature_idx, activation, example)

# After extraction: Build heaps in batch
for neuron_idx in range(latent_dim):
    examples = [e for e in feature_examples if e[0] == neuron_idx]
    top_k = heapq.nlargest(100, examples, key=lambda x: x[1])
```

**Trade-off:** Uses more memory during extraction, but enables vectorization

**B. Approximate Top-K (Alternative)**
```python
# Use torch.topk to find approximate top-k across batch
batch_top_k = torch.topk(batch_activations, k=100, dim=0)
# Merge with existing top-k (simple comparison, not heap)
```

**Trade-off:** Not exact top-k, but close enough and much faster

**C. Database-Deferred (Best for Large Scale)**
```python
# Store ALL activations to database temporarily
# Post-process to select top-k using SQL
# DELETE FROM feature_activations WHERE ranking > 100
```

**Trade-off:** Slower insertion, but unlimited storage and exact top-k

**Recommendation:** Use **Option A (Deferred Heap)** for feasibility and exact results

---

#### Decision 2: Batch Processing Strategy (CONFIGURABLE)

**Current:** Process samples one-by-one

**Proposed:** Process entire batch at once with **configurable batch size**

```python
# Current (per-sample):
for batch_idx in range(batch_size):
    sample = sae_features[batch_idx]
    for neuron_idx in range(latent_dim):
        process_feature(sample, neuron_idx)

# Proposed (vectorized batch - CONFIGURABLE):
batch_sae_features = sae_features  # (batch_size, seq_len, latent_dim)
# All samples, all features processed together
batch_max_activations = batch_sae_features.max(dim=1).values  # (batch_size, latent_dim)
```

**Configuration Options:**

**New Config Parameter: `vectorization_batch_size`**
```python
# Config options:
vectorization_batch_size = config.get("vectorization_batch_size", "auto")

# Values:
# - "auto": Automatically determine based on available GPU memory (recommended)
# - 1: Process one sample at a time (legacy mode - slowest)
# - 32: Process 32 samples simultaneously
# - 64: Process 64 samples simultaneously
# - 128: Process 128 samples simultaneously (default for auto)
# - 256: Process 256 samples simultaneously (high memory)
```

**Implementation:**
```python
# In extraction config
config = {
    "batch_size": 128,  # Data loading batch size
    "vectorization_batch_size": "auto",  # NEW: Vectorization processing batch
    "num_workers": 8,
    "evaluation_samples": 100000
}

# Auto-calculation logic
if vectorization_batch_size == "auto":
    available_vram_gb = get_available_gpu_memory()
    # Calculate optimal batch size based on:
    # - Available VRAM
    # - Feature dimensions (latent_dim, seq_len)
    # - Safety margin
    vectorization_batch_size = calculate_optimal_vectorization_batch(
        available_vram_gb=available_vram_gb,
        latent_dim=latent_dim,
        seq_len=max_length
    )
```

**Benefits:**
- **Flexibility:** Users can tune based on their hardware
- **Backward Compatibility:** Setting to 1 provides legacy behavior
- **Auto-optimization:** "auto" mode maximizes hardware utilization
- **Memory Control:** Users with limited VRAM can use smaller batches
- **Progressive Adoption:** Can start with small batches and scale up

**UI Integration:**
```typescript
// In extraction job configuration UI
<FormField label="Vectorization Batch Size" tooltip="Number of samples to process simultaneously. 'auto' is recommended.">
  <Select defaultValue="auto">
    <option value="auto">Auto (Recommended)</option>
    <option value="1">Sequential (Legacy - Slowest)</option>
    <option value="32">32 samples</option>
    <option value="64">64 samples</option>
    <option value="128">128 samples</option>
    <option value="256">256 samples (High Memory)</option>
  </Select>
</FormField>
```

---

#### Decision 3: Memory Management

**Challenge:** Storing all examples in memory before heap construction

**Analysis:**
```
Per example size: ~250 bytes (5 tokens + metadata)
Total examples: 100,000 samples × 16,384 features × 5 tokens avg/feature
               = 800M potential examples
               
If 10% features activate:
               = 80M examples × 250 bytes = 20 GB

If we store only activations > 0:
               = Much smaller (sparse activation)
```

**Strategy:**
1. Store examples in chunked list (not dict)
2. Process in batches of 1000 samples to limit memory
3. Construct heaps incrementally every 1000 samples

---

## Part 2: Implementation Plan

### Phase 1: Prototype & Validation (4-6 hours)

#### Task 1.1: Create Vectorized Helper Functions
**File:** `backend/src/services/extraction_vectorized.py` (NEW)

**Implement:**
```python
def calculate_optimal_vectorization_batch(
    available_vram_gb: float,
    latent_dim: int,
    seq_len: int,
    safety_margin: float = 0.3
) -> int:
    """
    Calculate optimal vectorization batch size based on available GPU memory.
    
    Args:
        available_vram_gb: Available GPU VRAM in GB
        latent_dim: SAE latent dimension size
        seq_len: Maximum sequence length
        safety_margin: Safety margin (30% by default)
        
    Returns:
        Optimal batch size for vectorized processing
    """
    # Memory per sample estimate (in GB)
    # seq_len * latent_dim * 4 bytes (float32) * 3 (activations, indices, masks)
    memory_per_sample_gb = (seq_len * latent_dim * 4 * 3) / (1024**3)
    
    # Apply safety margin
    usable_vram = available_vram_gb * (1 - safety_margin)
    
    # Calculate batch size
    optimal_batch = int(usable_vram / memory_per_sample_gb)
    
    # Clamp to reasonable range
    return max(1, min(optimal_batch, 256))

def vectorized_max_activations(sae_features: torch.Tensor) -> torch.Tensor:
    """
    Vectorized computation of max activation per feature.
    
    Args:
        sae_features: (seq_len, latent_dim) or (batch_size, seq_len, latent_dim) tensor
        
    Returns:
        (latent_dim,) or (batch_size, latent_dim) tensor of max activations
    """
    if sae_features.dim() == 2:
        # Single sample: (seq_len, latent_dim) → (latent_dim,)
        return sae_features.max(dim=0).values
    else:
        # Batch: (batch_size, seq_len, latent_dim) → (batch_size, latent_dim)
        return sae_features.max(dim=1).values

def vectorized_top_k_positions(sae_features: torch.Tensor, k: int = 5) -> tuple:
    """
    Find top-k token positions for each feature (vectorized).
    
    Args:
        sae_features: (seq_len, latent_dim) or (batch_size, seq_len, latent_dim) tensor
        k: Number of top positions to return
        
    Returns:
        values: (k, latent_dim) or (batch_size, k, latent_dim) tensor of top activations
        indices: (k, latent_dim) or (batch_size, k, latent_dim) tensor of token positions
    """
    if sae_features.dim() == 2:
        # Single sample
        return torch.topk(sae_features, k=k, dim=0)
    else:
        # Batch
        return torch.topk(sae_features, k=k, dim=1)

def batch_process_features(
    batch_sae_features: torch.Tensor,
    token_strings_batch: List[List[str]],
    sample_indices: List[int],
    vectorization_batch_size: int,
    top_k: int = 5
) -> List[Dict]:
    """
    Process features in configurable mini-batches for memory efficiency.
    
    Args:
        batch_sae_features: (batch_size, seq_len, latent_dim) tensor
        token_strings_batch: List of token strings per sample
        sample_indices: Global sample indices
        vectorization_batch_size: Number of samples to process simultaneously
        top_k: Number of top examples per feature
        
    Returns:
        List of example dictionaries
    """
    all_results = []
    num_samples = batch_sae_features.size(0)
    
    # Process in mini-batches of vectorization_batch_size
    for start_idx in range(0, num_samples, vectorization_batch_size):
        end_idx = min(start_idx + vectorization_batch_size, num_samples)
        
        # Extract mini-batch
        mini_batch = batch_sae_features[start_idx:end_idx]
        
        # Vectorized processing of mini-batch
        batch_results = process_mini_batch_vectorized(
            mini_batch,
            token_strings_batch[start_idx:end_idx],
            sample_indices[start_idx:end_idx],
            top_k
        )
        
        all_results.extend(batch_results)
    
    return all_results
```

**Validation:**
- Unit tests comparing output to sequential version
- Performance benchmark on small dataset (100 samples)
- Memory usage profiling with different vectorization_batch_size values

**Time:** 3-4 hours

---

#### Task 1.2: Implement Deferred Heap Construction
**File:** `backend/src/services/extraction_vectorized.py`

**Implement:**
```python
class DeferredTopKHeap:
    """
    Stores all examples during extraction, builds heaps at end.
    """
    
    def __init__(self, num_features: int, top_k: int = 100):
        self.num_features = num_features
        self.top_k = top_k
        self.examples = []  # List of (feature_idx, activation, example)
        
    def add_batch(self, feature_indices: np.ndarray, activations: np.ndarray, examples: List[Dict]):
        """Add batch of examples (vectorized)."""
        for i in range(len(feature_indices)):
            self.examples.append((
                feature_indices[i],
                activations[i],
                examples[i]
            ))
    
    def build_heaps(self) -> Dict[int, List[tuple]]:
        """Build top-k heaps for each feature."""
        heaps = defaultdict(list)
        
        # Group by feature
        feature_groups = defaultdict(list)
        for feat_idx, activation, example in self.examples:
            feature_groups[feat_idx].append((activation, example))
        
        # Build top-k for each feature
        for feat_idx, examples in feature_groups.items():
            # Sort and take top-k
            top_examples = sorted(examples, key=lambda x: x[0], reverse=True)[:self.top_k]
            heaps[feat_idx] = top_examples
            
        return heaps
```

**Validation:**
- Compare final heaps with sequential version (exact match)
- Test with various top_k values (10, 100, 1000)
- Memory usage test (ensure doesn't OOM)

**Time:** 2-3 hours

---

### Phase 2: Integration & Testing (6-8 hours)

#### Task 2.1: Add Configuration Support
**File:** `backend/src/models/extraction_job.py` or config schema

**Changes:**
1. Add `vectorization_batch_size` to extraction config schema
2. Default to "auto"
3. Validate values (must be "auto" or positive integer)

**File:** `backend/src/api/v1/endpoints/extractions.py`

**Changes:**
1. Accept `vectorization_batch_size` in extraction creation endpoint
2. Pass to extraction service

**Time:** 1 hour

---

#### Task 2.2: Refactor Extraction Service
**File:** `backend/src/services/extraction_service.py`

**Changes:**
1. Import vectorized functions
2. Calculate optimal vectorization batch size if "auto"
3. Replace sequential loop with vectorized batch processing
4. Update progress tracking (batch-level instead of feature-level)

**Modified Section (lines 1086-1165):**
```python
# OLD: for batch_idx in range(len(batch_input_ids)):
#          for neuron_idx in range(latent_dim):

# NEW: Vectorized batch processing with configurable batch size
from src.services.extraction_vectorized import (
    batch_process_features,
    DeferredTopKHeap,
    calculate_optimal_vectorization_batch
)

# Get vectorization batch size from config
vectorization_batch_size = config.get("vectorization_batch_size", "auto")

# Calculate optimal batch size if "auto"
if vectorization_batch_size == "auto":
    available_vram = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    gpu_total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    available_vram_gb = max(1.0, gpu_total_vram - (available_vram / (1024**3)))
    
    vectorization_batch_size = calculate_optimal_vectorization_batch(
        available_vram_gb=available_vram_gb,
        latent_dim=latent_dim,
        seq_len=max_length
    )
    logger.info(f"Auto-calculated vectorization batch size: {vectorization_batch_size}")
else:
    vectorization_batch_size = int(vectorization_batch_size)
    logger.info(f"Using user-specified vectorization batch size: {vectorization_batch_size}")

# Initialize deferred heap
deferred_heap = DeferredTopKHeap(
    num_features=latent_dim,
    top_k=top_k_examples
)

# Process entire batch at once with configurable vectorization
for batch_start in range(0, len(dataset), batch_size):
    batch_end = min(batch_start + batch_size, len(dataset))
    
    # ... (existing data loading code) ...
    
    # Run model forward pass (unchanged)
    base_model_activations = ...
    
    # Process batch through SAE (keep on GPU)
    batch_sae_features = sae.encode(base_model_activations)  # (batch_size, seq_len, latent_dim)
    
    # VECTORIZED: Process features with configurable batch size
    batch_results = batch_process_features(
        batch_sae_features=batch_sae_features,
        token_strings_batch=batch_texts,
        sample_indices=list(range(batch_start, batch_end)),
        vectorization_batch_size=vectorization_batch_size,  # CONFIGURABLE
        top_k=5
    )
    
    # Add to deferred heap
    deferred_heap.add_batch(batch_results)
    
    # Update progress (batch-level)
    progress = batch_end / len(dataset)
    self.update_extraction_status_sync(...)

# After all batches: Build final heaps
logger.info("Building final top-k heaps...")
final_heaps = deferred_heap.build_heaps()

# Convert to feature_activations format for database storage
feature_activations = {
    neuron_idx: [(act, 0, ex) for act, ex in examples]
    for neuron_idx, examples in final_heaps.items()
}
```

**Time:** 4-5 hours

---

#### Task 2.3: Frontend Configuration UI
**File:** `frontend/src/components/extractions/ExtractionConfigForm.tsx`

**Changes:**
Add vectorization batch size selector:
```typescript
<FormField>
  <label>Vectorization Batch Size</label>
  <select 
    value={config.vectorization_batch_size || "auto"}
    onChange={(e) => setConfig({
      ...config,
      vectorization_batch_size: e.target.value
    })}
  >
    <option value="auto">Auto (Recommended)</option>
    <option value="1">Sequential (Legacy)</option>
    <option value="32">32 samples</option>
    <option value="64">64 samples</option>
    <option value="128">128 samples</option>
    <option value="256">256 samples (High Memory)</option>
  </select>
  <p className="text-sm text-slate-400">
    Controls how many samples are processed simultaneously. 
    Higher values are faster but use more GPU memory.
  </p>
</FormField>
```

**Time:** 1 hour

---

#### Task 2.4: Performance Testing
**Scope:** Test with different vectorization batch sizes

**Test Cases:**
1. Sequential (batch_size=1): Baseline performance
2. Small (batch_size=32): Low memory systems
3. Medium (batch_size=64): Balanced
4. Large (batch_size=128): Default auto
5. XL (batch_size=256): High memory systems
6. Auto: Verify auto-calculation works correctly

**Test Dataset Sizes:**
- 100 samples → Verify correctness
- 1,000 samples → Benchmark
- 10,000 samples → Profile
- 100,000 samples → Full test

**Metrics:**
- Total extraction time
- Samples/second processing rate
- CPU utilization
- GPU utilization
- Memory usage (peak RAM and VRAM)
- Correctness (compare with sequential on small dataset)

**Time:** 2-3 hours

---

### Phase 3: Optimization & Deployment (4-6 hours)

#### Task 3.1: Memory Optimization
**If memory issues occur:**

**Strategy A: Dynamic Batch Size Reduction**
```python
try:
    batch_results = batch_process_features(
        vectorization_batch_size=vectorization_batch_size
    )
except RuntimeError as e:
    if "out of memory" in str(e):
        # Reduce batch size and retry
        vectorization_batch_size = max(1, vectorization_batch_size // 2)
        logger.warning(f"GPU OOM, reducing batch size to {vectorization_batch_size}")
        torch.cuda.empty_cache()
        batch_results = batch_process_features(
            vectorization_batch_size=vectorization_batch_size
        )
```

**Strategy B: Chunked Processing**
```python
chunk_size = 10000  # Process 10k samples at a time
for chunk_start in range(0, len(dataset), chunk_size):
    chunk_end = min(chunk_start + chunk_size, len(dataset))
    
    # Process chunk
    chunk_results = process_chunk(chunk_start, chunk_end)
    
    # Build intermediate heaps
    intermediate_heaps = build_heaps(chunk_results)
    
    # Merge with global heaps
    merge_heaps(global_heaps, intermediate_heaps)
```

**Time:** 2-3 hours (if needed)

---

#### Task 3.2: Code Documentation & Testing
**Actions:**
1. Add comprehensive docstrings to all new functions
2. Document `vectorization_batch_size` configuration parameter
3. Add inline comments explaining vectorization logic
4. Write unit tests for vectorized functions
5. Write integration tests for full extraction pipeline
6. Update extraction service documentation
7. Add usage examples for different batch sizes

**Time:** 1-2 hours

---

#### Task 3.3: Backward Compatibility
**Ensure:**
- Old extraction jobs continue to work (default to "auto")
- UI provides default selection ("auto")
- API accepts missing parameter (default to "auto")
- Database schema unchanged

**Add feature flag:**
```python
# In config.py
USE_VECTORIZED_EXTRACTION = os.getenv("USE_VECTORIZED_EXTRACTION", "true").lower() == "true"

# In extraction_service.py
if USE_VECTORIZED_EXTRACTION:
    results = vectorized_extraction_pipeline(
        vectorization_batch_size=vectorization_batch_size
    )
else:
    results = legacy_extraction_pipeline(...)  # Keep old code as fallback
```

**Time:** 1 hour

---

## Part 3: Implementation Task List

### Phase 1: Prototype & Validation (4-6 hours)
- [ ] 1.1.1 Create `extraction_vectorized.py` file
- [ ] 1.1.2 Implement `calculate_optimal_vectorization_batch()`
- [ ] 1.1.3 Implement `vectorized_max_activations()` with batch support
- [ ] 1.1.4 Implement `vectorized_top_k_positions()` with batch support
- [ ] 1.1.5 Implement `batch_process_features()` with configurable batch size
- [ ] 1.1.6 Write unit tests for vectorized functions
- [ ] 1.1.7 Benchmark with different batch sizes (1, 32, 64, 128, 256)
- [ ] 1.2.1 Implement `DeferredTopKHeap` class
- [ ] 1.2.2 Implement `add_batch()` method
- [ ] 1.2.3 Implement `build_heaps()` method
- [ ] 1.2.4 Write unit tests for heap construction
- [ ] 1.2.5 Validate correctness against sequential version

### Phase 2: Integration & Testing (6-8 hours)
- [ ] 2.1.1 Add `vectorization_batch_size` to extraction config schema
- [ ] 2.1.2 Update API endpoint to accept new parameter
- [ ] 2.1.3 Add validation for vectorization_batch_size values
- [ ] 2.2.1 Import vectorized functions in extraction_service.py
- [ ] 2.2.2 Implement auto-calculation of optimal batch size
- [ ] 2.2.3 Replace sequential loop (lines 1086-1165) with vectorized version
- [ ] 2.2.4 Update progress tracking logic
- [ ] 2.2.5 Test with small dataset (100 samples, various batch sizes)
- [ ] 2.3.1 Add batch size selector to frontend UI
- [ ] 2.3.2 Update TypeScript types for new config parameter
- [ ] 2.3.3 Test UI configuration flow
- [ ] 2.4.1 Test batch_size=1 (sequential - baseline)
- [ ] 2.4.2 Test batch_size=32 (low memory)
- [ ] 2.4.3 Test batch_size=64 (balanced)
- [ ] 2.4.4 Test batch_size=128 (default)
- [ ] 2.4.5 Test batch_size=256 (high memory)
- [ ] 2.4.6 Test "auto" mode (verify calculation)
- [ ] 2.4.7 Profile CPU/GPU/RAM usage for each batch size
- [ ] 2.4.8 Validate correctness on medium dataset (10k samples)
- [ ] 2.4.9 Full-scale test (100k samples, auto mode)

### Phase 3: Optimization & Deployment (4-6 hours)
- [ ] 3.1.1 Implement dynamic batch size reduction on OOM
- [ ] 3.1.2 Implement chunked processing (if needed)
- [ ] 3.2.1 Add comprehensive docstrings
- [ ] 3.2.2 Document vectorization_batch_size parameter
- [ ] 3.2.3 Write integration tests
- [ ] 3.2.4 Update extraction documentation with batch size guide
- [ ] 3.3.1 Add feature flag for backward compatibility
- [ ] 3.3.2 Ensure default values work for legacy behavior
- [ ] 3.3.3 Final deployment and monitoring

### Total Estimated Time: 14-20 hours

---

## Part 4: Expected Results

### Performance Metrics by Vectorization Batch Size

| Batch Size | Extraction Time | Speedup | CPU Usage | GPU Usage | VRAM Usage |
|------------|----------------|---------|-----------|-----------|------------|
| **1 (Sequential)** | 31 hours | 1x (baseline) | 7% | 7% | 3 GB |
| **32** | 8-10 hours | 3-4x | 10% | 15% | 4 GB |
| **64** | 4-6 hours | 5-8x | 12% | 20% | 6 GB |
| **128 (Auto Default)** | 2-4 hours | 10-15x | 15% | 25% | 8 GB |
| **256** | 2-3 hours | 12-18x | 18% | 30% | 12 GB |
| **Auto** | 2-4 hours | 10-15x | Varies | Varies | Optimized |

**Processing Rate Comparison:**

| Batch Size | Samples/Second | Improvement |
|------------|---------------|-------------|
| 1 (Sequential) | 0.9 | 1x |
| 32 | 2.8-3.5 | 3-4x |
| 64 | 4.6-6.9 | 5-8x |
| 128 | 6.9-13.9 | 10-15x |
| 256 | 9.3-15.3 | 12-18x |

### Risk Mitigation

**Risk 1: Memory OOM**
- Mitigation: Dynamic batch size reduction on OOM detection
- Fallback: Start with batch_size=32 and scale up
- User Control: Manual override via UI

**Risk 2: Correctness Issues**
- Mitigation: Extensive unit tests, validation against sequential
- Fallback: Keep legacy code path with feature flag
- Testing: Compare outputs on small dataset before full deployment

**Risk 3: GPU OOM**
- Mitigation: Auto-calculation considers available VRAM
- Fallback: Reduce batch size or use CPU fallback
- User Control: Manual batch size selection

**Risk 4: Longer Development Time**
- Mitigation: Phased approach, can stop after Phase 1
- Fallback: Deploy partial optimizations incrementally
- Flexibility: Start with conservative batch sizes

---

## Part 5: Configuration Guide

### When to Use Each Batch Size

**Sequential (batch_size=1):**
- Legacy systems or testing
- Very limited GPU memory (<4 GB VRAM)
- Debugging or correctness validation
- Slowest but most stable

**Small (batch_size=32):**
- Low-end GPUs (4-6 GB VRAM)
- Systems with limited RAM
- Safe starting point
- 3-4x speedup

**Medium (batch_size=64):**
- Mid-range GPUs (6-12 GB VRAM)
- Balanced performance/memory
- Good for most users
- 5-8x speedup

**Large (batch_size=128):**
- High-end GPUs (12-16 GB VRAM)
- Default for "auto" mode
- Recommended for RTX 3090/4090
- 10-15x speedup

**XL (batch_size=256):**
- Enterprise GPUs (24+ GB VRAM)
- Maximum performance
- A100, H100, or multi-GPU setups
- 12-18x speedup

**Auto (Recommended):**
- Automatically determines optimal batch size
- Considers available VRAM dynamically
- Safe for all systems
- Best general-purpose option

---

## Part 6: Alternative Approaches (Not Recommended)

### PostgreSQL pgvector Extension
**Why NOT Applicable:**
- pgvector is for vector similarity search (e.g., finding similar embeddings)
- Our bottleneck is in Python feature processing, NOT database queries
- Database operations are <1% of total time
- Would not solve the main bottleneck

### Multi-processing (Python multiprocessing)
**Why NOT Recommended as Primary:**
- Python GIL limits effectiveness
- Overhead of process spawning
- Difficult to share GPU resources
- Better as secondary optimization after vectorization

### Numba JIT Compilation
**Why NOT Recommended as Primary:**
- JIT compilation overhead
- Limited GPU support
- NumPy/PyTorch vectorization is simpler and faster
- Better as secondary optimization

---

## Part 7: Success Criteria

### Must Have (Phase 1-2)
- [ ] 10x speedup on 100k sample extraction
- [ ] Correctness validated (exact match with sequential)
- [ ] No regression in memory usage (<2x increase)
- [ ] Backward compatibility maintained
- [ ] Configurable batch size working in UI

### Should Have (Phase 3)
- [ ] 15x speedup on 100k sample extraction
- [ ] GPU utilization >20%
- [ ] Comprehensive test coverage
- [ ] Documentation complete
- [ ] Auto-calculation working correctly

### Nice to Have
- [ ] 20x+ speedup with batch_size=256
- [ ] Automatic batch size optimization per GPU
- [ ] Streaming database insertion
- [ ] Multi-GPU support
- [ ] Performance monitoring dashboard

---

## Part 8: Next Steps

### Immediate Actions (Before Starting Implementation)
1. **Review this plan with stakeholders**
2. **Decide on timeline:** 1 week (part-time) or 2-3 days (full-time)
3. **Set up development branch:** `feature/vectorized-extraction`
4. **Create backup:** Export current extraction code as `extraction_service_legacy.py`
5. **Set up monitoring:** Track extraction performance metrics

### Starting Implementation
1. Create feature branch: `git checkout -b feature/vectorized-extraction`
2. Create new file: `backend/src/services/extraction_vectorized.py`
3. Follow Phase 1 tasks sequentially
4. Test incrementally after each task with different batch sizes
5. Commit frequently with descriptive messages

### Getting Help
- **NumPy documentation:** https://numpy.org/doc/stable/
- **PyTorch documentation:** https://pytorch.org/docs/stable/
- **Vectorization guide:** https://numpy.org/doc/stable/user/basics.broadcasting.html

---

## Conclusion

This vectorization project will reduce extraction time from 31 hours to 2-4 hours by:
1. Replacing sequential Python loops with vectorized NumPy/PyTorch operations
2. Processing configurable batches simultaneously instead of one-by-one
3. Minimizing CPU↔GPU memory transfers
4. Better utilizing available hardware (CPU cores and GPU)
5. **Providing user control over batch size for different hardware configurations**

The implementation is **straightforward** (14-20 hours), **low-risk** (backward compatibility maintained), and **flexible** (configurable batch sizes). The performance gain is **substantial** (10-18x speedup depending on configuration).

**Key Innovation:** The configurable `vectorization_batch_size` parameter allows users to:
- Optimize for their specific hardware (4GB GPU vs 24GB GPU)
- Start conservatively and scale up
- Use "auto" mode for automatic optimization
- Fall back to sequential mode if needed

**Recommendation:** Proceed with implementation starting with Phase 1 prototype, with emphasis on testing different batch sizes.

---

**Document Location:** `0xcc/docs/Vectorization_Implementation_Plan.md`
**Last Updated:** 2025-11-16
**Status:** Ready for Implementation
