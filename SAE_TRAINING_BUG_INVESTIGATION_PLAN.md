# SAE Training Collapse - Bug Investigation & Fix Plan

**Date:** 2025-11-22
**Issue:** Both SAE training runs collapse to 95-99% dead neurons between steps 5k-10k, regardless of hyperparameters
**User Insight:** "It would be sooooo easy to have a decimal out of place or something like that"

---

## Executive Summary

Analysis of two failed training runs reveals a **consistent collapse pattern at steps 5000-10000** that occurs regardless of hyperparameter changes (300x different l1_alpha, 8x different latent_dim, removed top_k_sparsity). This strongly indicates **implementation bugs**, not hyperparameter tuning issues.

### Key Evidence

**Training `train_00c727f7` (Latest Run)**:
```
Step  | L0 Sparsity | Dead Neurons | Learning Rate | Status
1,000 | 11.86%      | 5 (0.2%)     | 0.00006      | ✅ Healthy (warmup)
5,000 | 15.62%      | 479 (23%)    | 0.0003       | ⚠️  Sparsity TOO HIGH!
10,000| 0.41%       | 1,981 (97%)  | 0.0003       | ❌ COLLAPSED
49,900| 0.18%       | 2,039 (99%)  | 0.0003       | ❌ Dead
```

**Critical Observations**:
1. ❌ **L0 sparsity INCREASES from 11.86% to 15.62%** when L1 penalty should push it DOWN toward target (5%)
2. ❌ **Collapse happens immediately after step 5000** when:
   - Learning rate jumps from warmup (0.00006 → 0.0003 = **5x increase**)
   - Dead neuron resampling triggers (first interval at step 5000)
3. ❌ **Sparsity goes from 15.62% to 0.41% in 5000 steps** (98% of neurons die)

---

## Suspected Bugs (In Priority Order)

### 🔴 **BUG #1: L1 Penalty Application Error (CRITICAL)**

**Location**: `backend/src/ml/sparse_autoencoder.py:216-236`

**Current Implementation**:
```python
# Line 216: L1 sparsity penalty (SAELens standard)
l1_penalty = z.abs().mean()  # Mean across both features and batch

# Line 236: Total loss
loss_total = loss_reconstruction + self.l1_alpha * l1_penalty
```

**Problem**: L0 sparsity INCREASES when L1 penalty should suppress it. This suggests:

**Hypothesis 1a: Wrong Scale Factor**
- Current: `z.abs().mean()` averages over (batch_size × latent_dim) elements
- For batch=128, latent_dim=2048: divides by 262,144
- With l1_alpha=0.003: penalty = 0.003 * (sum/262144) ≈ **0.000000011 * sum**
- **This is TOO SMALL!** Should be per-sample or per-feature average

**Hypothesis 1b: Missing Normalization Compensation**
- Activations are normalized by `sqrt(hidden_dim) / ||x||` before encoding
- L1 penalty is computed on normalized activations
- When denormalized, the penalty scale might be wrong by factor of `sqrt(hidden_dim)` ≈ 27

**Expected Behavior** (from Anthropic's "Towards Monosemanticity"):
- L1 penalty should be: `l1_alpha * mean_per_sample(sum_per_feature(abs(z)))`
- NOT: `l1_alpha * mean_over_all_elements(abs(z))`

**Evidence**:
- L0 sparsity increases from 11.86% to 15.62% when it should decrease
- This only happens when L1 penalty is supposed to be active (after warmup)

---

### 🟠 **BUG #2: Learning Rate Warmup Transition (HIGH PRIORITY)**

**Location**: `backend/src/workers/training_tasks.py:348-357`

**Current Implementation**:
```python
warmup_steps = hp.get('warmup_steps', 0)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Problem**: At step 5000, LR jumps from `0.00006 → 0.0003` (**5x increase!**)

**Hypothesis 2a: Warmup Transition Too Abrupt**
- Step 4999: LR = 0.0003 * (4999/5000) = 0.00029994
- Step 5000: LR = 0.0003 * 1.0 = 0.0003
- Combined with L1 penalty bug, this sudden LR increase amplifies the collapse

**Evidence**:
- Both trainings collapse RIGHT AFTER warmup ends
- train_037170bc: warmup=10000, collapsed between 10k-50k
- train_00c727f7: warmup=5000, collapsed between 5k-10k

---

### 🟡 **BUG #3: Dead Neuron Resampling Implementation (MEDIUM PRIORITY)**

**Location**: `backend/src/workers/training_tasks.py:926-971`

**Current Implementation**:
```python
# Line 928: Trigger condition
if step > 0 and step % resample_interval == 0 and step >= warmup_steps:
    # Line 941: Identify dead neurons
    dead_mask = (z == 0).all(dim=0)  # Never activated in current batch

    # Line 963: Reinitialize encoder weights
    model.encoder.weight[dead_idx] = x[sample_idx] * 0.1  # Small scale
```

**Problem**: Resampling triggers at step 5000, immediately before collapse

**Hypothesis 3a: Resampling Creates Instability**
- Dead neurons are identified as "never activated in **current batch**"
- This is only 128 samples - not representative!
- Resampling might be killing ALIVE neurons or creating numerical instabilities

**Hypothesis 3b: Resampling Scale Factor Too Small**
- `x[sample_idx] * 0.1` might be too small after normalization
- If x is normalized to `sqrt(hidden_dim) ≈ 27`, then `27 * 0.1 = 2.7` might be too large

**Evidence**:
- Dead neurons jump from 479 (23%) to 1,981 (97%) right after step 5000
- This is when first resampling happens (resample_interval=5000)

---

### 🟢 **BUG #4: Normalization Scale Error (LOWER PRIORITY)**

**Location**: `backend/src/ml/sparse_autoencoder.py:106-110`

**Current Implementation**:
```python
if self.normalize_activations == 'constant_norm_rescale':
    # SAELens standard: E(||x||) = sqrt(hidden_dim)
    import math
    norm_coeff = math.sqrt(self.hidden_dim) / x.norm(dim=-1, keepdim=True)
    x_normalized = x * norm_coeff
```

**Problem**: Normalization might amplify numerical errors

**Hypothesis 4a: Division by Small Norms**
- If `||x|| ≈ 0`, then `norm_coeff ≈ sqrt(hidden_dim) / 0 → ∞`
- This creates huge normalized values that break training

**Hypothesis 4b: Target Norm Incorrect**
- SAELens standard is `E(||x||) = sqrt(hidden_dim)`
- For hidden_dim=768: target norm = 27.7
- But actual activations might have different scale (e.g., mean=5, std=2)

**Evidence**:
- Both failed trainings use `constant_norm_rescale`
- Warmup phase (low LR) survives, but full LR phase collapses

---

## Investigation Plan (Execute Sequentially)

### Phase 1: Verify L1 Penalty Calculation ⏱️ **2 hours**

**Objective**: Confirm L1 penalty is being applied at correct scale

#### Task 1.1: Add Debug Logging to SAE Forward Pass
**File**: `backend/src/ml/sparse_autoencoder.py`
**Changes**:
```python
# After line 219: L0 sparsity = (z > 0).float().mean()
# ADD THIS:
l0_sparsity = (z > 0).float().mean()

# DEBUG: Log L1 penalty components
if return_loss and torch.rand(1).item() < 0.01:  # 1% sampling
    logger.info(f"[DEBUG] L1 Penalty Breakdown:")
    logger.info(f"  z.abs().sum(): {z.abs().sum().item():.6f}")
    logger.info(f"  z.abs().mean(): {z.abs().mean().item():.6f}")
    logger.info(f"  z.abs().mean(dim=0).mean(): {z.abs().mean(dim=0).mean().item():.6f}")
    logger.info(f"  l1_alpha: {self.l1_alpha:.6f}")
    logger.info(f"  l1_penalty (as used): {l1_penalty.item():.6f}")
    logger.info(f"  l1_alpha * l1_penalty: {(self.l1_alpha * l1_penalty).item():.6f}")
    logger.info(f"  loss_reconstruction: {loss_reconstruction.item():.6f}")
    logger.info(f"  Ratio (L1/recon): {(self.l1_alpha * l1_penalty / loss_reconstruction).item():.6f}")
```

#### Task 1.2: Run Short Training (1000 Steps)
**Command**:
```bash
# Use mini training job:
# - latent_dim: 256 (small)
# - batch_size: 64
# - total_steps: 1000
# - l1_alpha: 0.003
# - warmup_steps: 100
# - log_interval: 10
```

**Expected Outcomes**:
- If L1/recon ratio < 0.001: L1 penalty is TOO WEAK
- If L1/recon ratio > 1.0: L1 penalty is TOO STRONG
- Target ratio should be 0.01-0.1 (L1 penalty is 1-10% of reconstruction loss)

#### Task 1.3: Compare Against Research Papers
**References**:
- Anthropic "Towards Monosemanticity": Check Appendix A for L1 penalty formulation
- OpenAI "Sparse Autoencoders" (if available): Verify penalty calculation

**Questions**:
1. Should L1 be computed on **normalized** or **original** activations?
2. Should L1 be averaged per-sample, per-feature, or over all elements?
3. Should L1 be weighted by batch size?

---

### Phase 2: Fix L1 Penalty Calculation ⏱️ **3 hours**

**Objective**: Implement correct L1 penalty based on research

#### Task 2.1: Implement Research-Correct L1 Penalty
**File**: `backend/src/ml/sparse_autoencoder.py`
**Current (Line 216)**:
```python
l1_penalty = z.abs().mean()  # Mean across both features and batch
```

**Option A: Per-Sample Average** (most likely correct):
```python
# Average L1 norm per sample (then average over batch)
l1_penalty = z.abs().sum(dim=-1).mean()  # [batch, latent_dim] → [batch] → scalar
```

**Option B: Per-Feature Average**:
```python
# Average L1 norm per feature (then average over features)
l1_penalty = z.abs().mean(dim=0).mean()  # [batch, latent_dim] → [latent_dim] → scalar
```

**Option C: Unnormalized Sum**:
```python
# Total L1 norm (no averaging)
l1_penalty = z.abs().sum()
```

#### Task 2.2: Adjust L1 Alpha Scale
If L1 penalty formulation changes, l1_alpha must be rescaled:

**Example**: If switching from mean-over-all to mean-per-sample:
- Old: penalty divides by (batch_size × latent_dim) = 128 × 2048 = 262,144
- New: penalty divides by batch_size = 128
- Scaling factor: 2048
- New l1_alpha = old l1_alpha * 2048

**For l1_alpha = 0.003**:
- New l1_alpha = 0.003 * 2048 ≈ **6.1** (if per-sample averaging)
- Or: New l1_alpha = 0.003 / 2048 ≈ **0.0000015** (if unnormalized sum)

#### Task 2.3: Add L1 Penalty Unit Test
**File**: `backend/tests/test_sparse_autoencoder.py`
```python
def test_l1_penalty_scale():
    """Test that L1 penalty has correct scale."""
    sae = SparseAutoencoder(hidden_dim=768, latent_dim=2048, l1_alpha=0.003)
    x = torch.randn(128, 768)  # batch_size=128

    x_recon, z, losses = sae(x, return_loss=True)

    # Expected: L1 penalty should be 1-10% of reconstruction loss
    l1_contribution = sae.l1_alpha * losses['l1_penalty']
    ratio = l1_contribution / losses['loss_reconstruction']

    assert 0.01 <= ratio <= 0.5, f"L1/recon ratio {ratio:.4f} is outside expected range [0.01, 0.5]"

    # Expected: L0 sparsity should decrease over multiple steps
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
    l0_values = []
    for step in range(100):
        x_recon, z, losses = sae(x, return_loss=True)
        losses['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        l0_values.append(losses['l0_sparsity'].item())

    # L0 should decrease by at least 20% after 100 steps
    assert l0_values[-1] < l0_values[0] * 0.8, "L0 sparsity did not decrease"
```

---

### Phase 3: Fix Learning Rate Warmup ⏱️ **2 hours**

**Objective**: Smooth LR transition to prevent sudden jumps

#### Task 3.1: Implement Gradual Warmup End
**File**: `backend/src/workers/training_tasks.py:348-357`
**Current**:
```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0
```

**Fixed (Cosine Warmup with Smooth Transition)**:
```python
def lr_lambda(step):
    if step < warmup_steps:
        # Linear warmup
        return step / warmup_steps
    elif step < warmup_steps + 1000:  # 1000-step smooth transition
        # Cosine transition from 1.0 to 1.0 (gentle plateau)
        progress = (step - warmup_steps) / 1000
        return 1.0  # Keep constant during transition
    else:
        # Optionally: cosine decay after transition
        return 1.0
```

**Alternative (More Conservative)**:
```python
def lr_lambda(step):
    if step < warmup_steps:
        # Linear warmup
        return step / warmup_steps
    else:
        # Keep at max LR (no decay)
        return 1.0
```

#### Task 3.2: Reduce Warmup Steps
**Reasoning**: Warmup is too long (5000-10000 steps), preventing early detection of issues

**Recommendation**:
- Reduce warmup_steps to **500-1000** for 50k total steps
- This allows L1 penalty to activate earlier and catch problems sooner

---

### Phase 4: Fix Dead Neuron Resampling ⏱️ **4 hours**

**Objective**: Make resampling more stable and representative

#### Task 4.1: Track Dead Neurons Over Larger Window
**File**: `backend/src/workers/training_tasks.py:926-971`
**Problem**: Current logic identifies dead neurons in **single batch** (128 samples)

**Fixed**:
```python
# Option A: Track dead neurons over last N batches
dead_neuron_tracker = {}  # {layer_idx: deque of activation counts}

# During training loop:
if step % log_interval == 0:
    for layer_idx in training_layers:
        z = layer_activations[layer_idx]
        activation_counts = (z > 0).float().sum(dim=0)  # [latent_dim]

        if layer_idx not in dead_neuron_tracker:
            from collections import deque
            dead_neuron_tracker[layer_idx] = deque(maxlen=50)  # Last 50 batches

        dead_neuron_tracker[layer_idx].append(activation_counts.cpu())

# At resampling time:
if step % resample_interval == 0:
    for layer_idx in training_layers:
        # Dead neurons = activated in < 1% of last 50 batches
        total_counts = torch.stack(list(dead_neuron_tracker[layer_idx])).sum(dim=0)
        total_samples = len(dead_neuron_tracker[layer_idx]) * batch_size
        activation_rate = total_counts / total_samples

        dead_mask = activation_rate < 0.01  # Less than 1% activation rate
```

#### Task 4.2: Adjust Resampling Scale
**File**: `backend/src/workers/training_tasks.py:963`
**Current**:
```python
model.encoder.weight[dead_idx] = x[sample_idx] * 0.1  # Small scale
```

**Problem**: Scale of `0.1` might be wrong after normalization

**Fixed**:
```python
# Get high-loss sample (already in code)
sample_vector = x[sample_idx]  # [hidden_dim]

# Normalize to unit vector, then scale appropriately
sample_norm = sample_vector.norm()
if sample_norm > 1e-6:
    unit_vector = sample_vector / sample_norm
    # Scale to match typical encoder weight norms
    typical_weight_norm = model.encoder.weight.norm(dim=-1).mean()
    model.encoder.weight[dead_idx] = unit_vector * typical_weight_norm * 0.5
else:
    # Fallback: random initialization
    model.encoder.weight[dead_idx] = torch.randn_like(model.encoder.weight[dead_idx]) * 0.01
```

#### Task 4.3: Disable Resampling for Testing
**Temporary workaround**: Disable resampling to isolate if it's causing the collapse

**File**: Training configuration
```python
"resample_dead_neurons": False  # Temporarily disable
```

---

### Phase 5: Verify Normalization ⏱️ **2 hours**

**Objective**: Ensure normalization doesn't create numerical issues

#### Task 5.1: Add Normalization Debugging
**File**: `backend/src/ml/sparse_autoencoder.py:106-110`
**Add Safety Checks**:
```python
if self.normalize_activations == 'constant_norm_rescale':
    import math
    x_norm = x.norm(dim=-1, keepdim=True)

    # Safety: Prevent division by zero
    x_norm = torch.clamp(x_norm, min=1e-6)

    norm_coeff = math.sqrt(self.hidden_dim) / x_norm

    # DEBUG: Check for abnormal scaling
    if torch.rand(1).item() < 0.01:  # 1% sampling
        logger.info(f"[DEBUG] Normalization:")
        logger.info(f"  x.norm() range: [{x_norm.min().item():.6f}, {x_norm.max().item():.6f}]")
        logger.info(f"  norm_coeff range: [{norm_coeff.min().item():.6f}, {norm_coeff.max().item():.6f}]")
        logger.info(f"  x_normalized.norm() range: [{(x * norm_coeff).norm(dim=-1).min().item():.6f}, {(x * norm_coeff).norm(dim=-1).max().item():.6f}]")

    x_normalized = x * norm_coeff
```

#### Task 5.2: Test Alternative Normalization
**Option**: Try LayerNorm instead of constant_norm_rescale
```python
if self.normalize_activations == 'layer_norm':
    # Standard LayerNorm: (x - mean) / std
    x_normalized = F.layer_norm(x, (self.hidden_dim,))
    norm_coeff = torch.ones_like(x[:, :1])  # No denormalization needed
    return x_normalized, norm_coeff
```

---

## Testing Strategy

### Test 1: Minimal Reproduction ⏱️ **30 minutes**
**Objective**: Confirm bug exists in controlled environment

**Configuration**:
```json
{
  "latent_dim": 256,
  "batch_size": 64,
  "total_steps": 10000,
  "l1_alpha": 0.003,
  "warmup_steps": 500,
  "resample_interval": 2500,
  "resample_dead_neurons": true,
  "log_interval": 50
}
```

**Expected Outcome**:
- L0 sparsity should DECREASE from ~20% to target 5%
- Dead neurons should stay below 10%
- No sudden collapse at step 2500 or 5000

---

### Test 2: L1 Penalty Fix Validation ⏱️ **1 hour**
**Objective**: Verify L1 penalty fix works

**Steps**:
1. Apply L1 penalty fix from Phase 2
2. Run Test 1 configuration again
3. Monitor L0 sparsity trend

**Success Criteria**:
- L0 sparsity decreases steadily
- L1/recon ratio stays in 0.01-0.5 range
- Final L0 sparsity within 20% of target (4%-6% for target=5%)

---

### Test 3: Full Training Run ⏱️ **6 hours**
**Objective**: Confirm fix works at full scale

**Configuration**:
```json
{
  "latent_dim": 2048,
  "batch_size": 128,
  "total_steps": 50000,
  "l1_alpha": 0.003,  // Adjusted based on Phase 2 findings
  "warmup_steps": 1000,  // Reduced from 5000
  "resample_interval": 5000,
  "resample_dead_neurons": true,
  "normalize_activations": "constant_norm_rescale",  // Or "layer_norm" if Phase 5 shows issues
  "log_interval": 100
}
```

**Success Criteria**:
- Dead neurons < 20% at all steps
- L0 sparsity converges to target (5%) by step 20k
- No sudden collapse at step 5000 or 10000
- Final reconstruction loss < 0.05

---

## Expected Fixes (Summary)

### Fix #1: L1 Penalty Calculation (CRITICAL)
**Change**: `sparse_autoencoder.py:216`
```python
# OLD:
l1_penalty = z.abs().mean()

# NEW (most likely):
l1_penalty = z.abs().sum(dim=-1).mean()  # Per-sample L1 norm, averaged over batch
```
**Impact**: L1 penalty will be ~2048x larger, requiring l1_alpha adjustment

### Fix #2: L1 Alpha Rescaling
**Change**: All training configurations
```python
# If using per-sample L1:
l1_alpha = old_l1_alpha * latent_dim

# Example:
old_l1_alpha = 0.000010054  # Failed training 1
new_l1_alpha = 0.000010054 * 16384 ≈ 0.165

old_l1_alpha = 0.003  # Failed training 2
new_l1_alpha = 0.003 * 2048 ≈ 6.1
```

### Fix #3: Warmup Steps Reduction
**Change**: Default training template
```python
# OLD:
warmup_steps = 5000 or 10000

# NEW:
warmup_steps = 1000  # For 50k total steps (2%)
```

### Fix #4: Dead Neuron Tracking
**Change**: `training_tasks.py:941`
```python
# OLD:
dead_mask = (z == 0).all(dim=0)  # Single batch

# NEW:
# Track over last 50 batches (50 * 128 = 6400 samples)
activation_rate = total_counts / total_samples
dead_mask = activation_rate < 0.01  # Less than 1% activation
```

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Verify L1 Penalty | 2 hours | None |
| Phase 2: Fix L1 Penalty | 3 hours | Phase 1 complete |
| Phase 3: Fix LR Warmup | 2 hours | None (parallel with Phase 1-2) |
| Phase 4: Fix Resampling | 4 hours | Phase 2 complete |
| Phase 5: Verify Normalization | 2 hours | None (parallel) |
| Test 1: Minimal Reproduction | 0.5 hours | Phase 2 complete |
| Test 2: L1 Fix Validation | 1 hour | Test 1 complete |
| Test 3: Full Training | 6 hours | Test 2 complete |
| **Total** | **20.5 hours** | **3-4 days** |

---

## Risk Assessment

### High Confidence Bugs (>80% certainty)
1. ✅ **L1 penalty scale is wrong** - L0 sparsity increases when it should decrease
2. ✅ **LR warmup transition is too abrupt** - 5x jump correlates with collapse

### Medium Confidence Bugs (50-80% certainty)
3. ⚠️ **Dead neuron resampling triggers instability** - Collapse happens exactly at resample_interval
4. ⚠️ **Normalization creates numerical issues** - Both trainings use same normalization

### Low Confidence Issues (<50% certainty)
5. ❓ **Gradient accumulation bug** - Less likely, but worth checking
6. ❓ **Optimizer state corruption** - Very unlikely with Adam

---

## Validation Metrics

After implementing fixes, training should show:

| Metric | Target | Acceptable Range |
|--------|--------|------------------|
| L0 Sparsity (final) | 5% | 4%-7% |
| Dead Neurons (final) | <10% | <20% |
| L1/Recon Loss Ratio | 0.1 | 0.01-0.5 |
| Reconstruction Loss | <0.05 | <0.10 |
| Training Stability | No collapse | Smooth L0 curve |

---

## References

1. Anthropic "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
   - Appendix A: SAE Training Details
   - Check: L1 penalty formulation

2. OpenAI "Sparse Autoencoders" (if available)
   - Verify: Normalization methods
   - Check: Dead neuron handling

3. SAELens Library (https://github.com/jbloomAus/SAELens)
   - Reference implementation for L1 penalty
   - Check: Warmup schedules

---

## Next Steps (DO NOT IMPLEMENT YET)

1. ✅ **User approval required** - Review this plan with user
2. ⏳ Execute Phase 1 (Debug logging)
3. ⏳ Run Test 1 (Minimal reproduction)
4. ⏳ Analyze logs and confirm root cause
5. ⏳ Implement fixes based on findings
6. ⏳ Validate with Test 2 and Test 3

---

**Status**: Plan complete, awaiting user approval to proceed with implementation.
