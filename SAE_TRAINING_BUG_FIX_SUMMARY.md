# SAE Training Bug Fixes - Implementation Summary

**Date:** 2025-11-22
**Status:** ✅ FIXES IMPLEMENTED
**Files Modified:** 1 file (sparse_autoencoder.py)

---

## Bug Analysis Summary

Both training runs collapsed at steps 5000-10000 with **95-99% dead neurons** regardless of hyperparameter changes. Root cause analysis identified **implementation bugs**, not hyperparameter tuning issues.

### Key Evidence
- **L0 sparsity INCREASED from 11.86% to 15.62%** when L1 penalty should suppress it
- **Collapse immediately after step 5000** when warmup ended and resampling triggered
- **Consistent pattern across different hyperparameters** (300x different l1_alpha, 8x different latent_dim)

---

## Fixes Implemented

### ✅ FIX #1: L1 Penalty Calculation (CRITICAL)
**File**: `backend/src/ml/sparse_autoencoder.py`
**Lines Modified**: 218-222 (SparseAutoencoder), 364-365 (SkipAutoencoder), 492-493 (Transcoder)

**Bug**: L1 penalty was averaging over ALL elements (batch_size × latent_dim = 128 × 2048 = 262,144), making it **~2000x too weak**.

**OLD (WRONG)**:
```python
l1_penalty = z.abs().mean()  # Divides by 262,144
```

**NEW (CORRECT)**:
```python
# L1 sparsity penalty (per-sample L1 norm, then averaged over batch)
# This is the correct formulation from Anthropic's "Towards Monosemanticity"
# Sum L1 norm per sample, then average across batch
# Shape: [batch, latent_dim] -> sum over latent_dim -> [batch] -> mean over batch -> scalar
l1_penalty = z.abs().sum(dim=-1).mean()  # Divides by 128 only
```

**Impact**: L1 penalty is now **2048x stronger** (for latent_dim=2048), correctly pushing sparsity DOWN toward target.

**Expected Behavior After Fix**:
- L0 sparsity should DECREASE from ~12% → 5% (not increase to 15%!)
- Training should be stable (no sudden collapse)
- Dead neurons should stay < 10% throughout training

---

### ✅ FIX #2: Normalization Safety (MEDIUM PRIORITY)
**File**: `backend/src/ml/sparse_autoencoder.py`
**Lines Modified**: 109-112

**Bug**: Division by zero possible if input norm is very small.

**OLD (UNSAFE)**:
```python
norm_coeff = math.sqrt(self.hidden_dim) / x.norm(dim=-1, keepdim=True)
```

**NEW (SAFE)**:
```python
x_norm = x.norm(dim=-1, keepdim=True)

# Safety: Prevent division by zero
x_norm = torch.clamp(x_norm, min=1e-6)

norm_coeff = math.sqrt(self.hidden_dim) / x_norm
```

**Impact**: Prevents NaN/Inf values that could crash training.

---

## L1 Alpha Rescaling Guide

**CRITICAL**: Because L1 penalty calculation changed, `l1_alpha` values must be rescaled.

### Scaling Factor
If you used l1_alpha with the OLD (buggy) implementation:
```
new_l1_alpha = old_l1_alpha / latent_dim
```

### Examples

**For latent_dim = 2048:**
| Old l1_alpha | New l1_alpha | Calculation |
|--------------|--------------|-------------|
| 0.003        | 0.00000146   | 0.003 / 2048 |
| 0.001        | 0.00000049   | 0.001 / 2048 |
| 0.0001       | 0.000000049  | 0.0001 / 2048 |

**For latent_dim = 16384:**
| Old l1_alpha | New l1_alpha | Calculation |
|--------------|--------------|-------------|
| 0.000010054  | 0.000000000614 | 0.000010054 / 16384 |

### Recommended Starting Values (NEW Implementation)

| Latent Dim | Recommended l1_alpha | Target L0 | Expected Range |
|------------|---------------------|-----------|----------------|
| 256        | 0.00001 - 0.00005   | 5%        | 3%-8% |
| 512        | 0.000005 - 0.00002  | 5%        | 3%-8% |
| 1024       | 0.000002 - 0.00001  | 5%        | 3%-8% |
| 2048       | 0.000001 - 0.000005 | 5%        | 3%-8% |
| 4096       | 0.0000005 - 0.000002| 5%        | 3%-8% |
| 8192       | 0.00000025 - 0.000001| 5%       | 3%-8% |
| 16384      | 0.0000001 - 0.0000005| 5%       | 3%-8% |

**Rule of thumb**: `l1_alpha ≈ 5e-6 / (latent_dim / 2048)`

---

## Training Configuration Recommendations

### Recommended Hyperparameters (After Fixes)

**For latent_dim = 2048:**
```json
{
  "latent_dim": 2048,
  "hidden_dim": 768,
  "batch_size": 128,
  "total_steps": 50000,
  "learning_rate": 0.0003,
  "l1_alpha": 0.000002,  // ← NEW RECOMMENDED VALUE
  "warmup_steps": 1000,  // ← REDUCED from 5000 (2% of total_steps)
  "resample_interval": 5000,
  "resample_dead_neurons": true,
  "dead_neuron_threshold": 1000,
  "normalize_activations": "constant_norm_rescale",
  "target_l0": 0.05,
  "log_interval": 100,
  "checkpoint_interval": 5000
}
```

**For latent_dim = 16384:**
```json
{
  "latent_dim": 16384,
  "hidden_dim": 768,
  "batch_size": 64,  // Reduced for larger model
  "total_steps": 100000,
  "learning_rate": 0.0001,
  "l1_alpha": 0.0000002,  // ← NEW RECOMMENDED VALUE
  "warmup_steps": 2000,  // 2% of total_steps
  "resample_interval": 5000,
  "resample_dead_neurons": true,
  "dead_neuron_threshold": 1000,
  "normalize_activations": "constant_norm_rescale",
  "target_l0": 0.05,
  "log_interval": 100,
  "checkpoint_interval": 10000
}
```

### Key Changes from Previous Configurations
1. ✅ **l1_alpha reduced by 2048x** (for latent_dim=2048) to compensate for L1 penalty fix
2. ✅ **warmup_steps reduced from 5000-10000 to 1000-2000** (2% of total_steps)
3. ✅ **normalization includes safety clamp** to prevent division by zero

---

## Testing Plan

### Test 1: Minimal Validation (30 minutes)
**Objective**: Verify fixes work with small model

**Configuration**:
```json
{
  "latent_dim": 256,
  "batch_size": 64,
  "total_steps": 5000,
  "l1_alpha": 0.00002,
  "warmup_steps": 100,
  "log_interval": 50
}
```

**Success Criteria**:
- ✅ L0 sparsity DECREASES from ~15-20% → ~5% over 5000 steps
- ✅ Dead neurons stay < 10% throughout training
- ✅ No sudden collapse at any step
- ✅ Loss decreases steadily

---

### Test 2: Medium-Scale Validation (2 hours)
**Objective**: Test at production scale with shorter training

**Configuration**:
```json
{
  "latent_dim": 2048,
  "batch_size": 128,
  "total_steps": 20000,
  "l1_alpha": 0.000002,
  "warmup_steps": 500,
  "log_interval": 100
}
```

**Success Criteria**:
- ✅ L0 sparsity converges to 4-7% by step 10000
- ✅ Dead neurons < 15% at all steps
- ✅ Reconstruction loss < 0.10 by step 20000
- ✅ Stable training (no collapse)

---

### Test 3: Full Production Run (6-8 hours)
**Objective**: Validate full training run

**Configuration**: Use recommended hyperparameters above (50k steps for 2048 latent_dim)

**Success Criteria**:
- ✅ L0 sparsity: 4%-7% (target: 5%)
- ✅ Dead neurons: < 10% final
- ✅ Reconstruction loss: < 0.05
- ✅ Training completes without collapse
- ✅ Extracted features have meaningful activations (not 1e-16)

---

## Validation Metrics

After implementing fixes, monitor these metrics during training:

| Metric | Target | Acceptable Range | Red Flag |
|--------|--------|------------------|----------|
| L0 Sparsity (final) | 5.0% | 3%-8% | > 10% or < 1% |
| Dead Neurons (final) | < 5% | < 10% | > 20% |
| L1/Recon Loss Ratio | 0.05 | 0.01-0.20 | > 0.5 or < 0.001 |
| Reconstruction Loss | < 0.05 | < 0.10 | > 0.15 |
| L0 Trend | Decreasing | Steady decrease | Increasing |

---

## Expected Results

### Before Fix (BROKEN)
```
Step  | L0 Sparsity | Dead Neurons | Status
1,000 | 11.86%      | 5 (0.2%)     | ✅ Healthy (warmup)
5,000 | 15.62%      | 479 (23%)    | ❌ Sparsity INCREASED!
10,000| 0.41%       | 1,981 (97%)  | ❌ COLLAPSED
49,900| 0.18%       | 2,039 (99%)  | ❌ Dead
```

### After Fix (EXPECTED)
```
Step  | L0 Sparsity | Dead Neurons | Status
1,000 | 12.0%       | 5 (0.2%)     | ✅ Healthy (warmup)
5,000 | 7.5%        | 50 (2.4%)    | ✅ Sparsity decreasing
10,000| 5.8%        | 80 (3.9%)    | ✅ Converging to target
20,000| 5.2%        | 90 (4.4%)    | ✅ Near target
49,900| 5.0%        | 100 (4.9%)   | ✅ At target, stable
```

---

## Troubleshooting

### If L0 sparsity is still too high (> 10%)
**Problem**: L1 penalty still too weak
**Solution**: Increase l1_alpha by 2-5x

### If L0 sparsity is too low (< 2%)
**Problem**: L1 penalty too strong
**Solution**: Decrease l1_alpha by 2-5x

### If dead neurons increase during training
**Problem**: Learning rate too high or resampling not working
**Solution**:
1. Reduce learning_rate by 2x
2. Check dead neuron resampling is enabled
3. Reduce resample_interval to 2500 or 1000

### If training is unstable (NaN loss)
**Problem**: Normalization or gradient explosion
**Solution**:
1. Verify normalization safety fix is applied
2. Add gradient clipping: `grad_clip_norm: 1.0`
3. Reduce learning rate by 2x

---

## Migration Guide

### For Existing Training Jobs
1. ✅ **DO NOT resume** old training jobs - the L1 penalty change makes them incompatible
2. ✅ **Start fresh training** with new recommended l1_alpha values
3. ✅ **Delete failed training results** (train_037170bc, train_00c727f7) - they used broken code

### For Extraction from Old Trainings
1. ❌ **DO NOT use** features from old trainings (train_037170bc) - they have 95% dead neurons
2. ❌ **Extraction will produce zero activations** because the SAE is broken
3. ✅ **Wait for new training** with fixed code, then extract features

---

## Next Steps

1. ✅ **Fixes Implemented** - All code changes complete
2. ⏳ **Run Test 1** - Minimal validation (30 min)
3. ⏳ **Verify L0 Trend** - Check that sparsity decreases correctly
4. ⏳ **Run Test 2** - Medium-scale validation (2 hours)
5. ⏳ **Run Test 3** - Full production run (6-8 hours)
6. ⏳ **Extract Features** - Verify extraction produces non-zero activations
7. ⏳ **Test Labeling** - Verify LLM receives meaningful activation values

---

## Success Criteria for Fix Validation

✅ **FIXED** if ALL of these are true:
1. L0 sparsity DECREASES during training (not increases)
2. L0 sparsity converges to target ±2% (e.g., 3%-7% for target=5%)
3. Dead neurons stay < 10% throughout training
4. No sudden collapse at any step
5. Extracted features have activation values > 1e-6 (not 1e-16)
6. LLM receives non-zero activation values in prompts

❌ **NOT FIXED** if ANY of these occur:
1. L0 sparsity increases during training
2. Training collapses (> 90% dead neurons)
3. Extracted features have zero activations
4. LLM prompts still show "activation: 0.000"

---

## Technical Details

### Why L1 Penalty Was Wrong

**Research Standard** (Anthropic "Towards Monosemanticity"):
```
L1 = λ * mean_over_batch(sum_over_features(|z_i|))
```

**Our OLD Implementation**:
```
L1 = λ * mean_over_batch_and_features(|z_i|)
```

**Difference**:
- Research: Divides by batch_size (128)
- Old: Divides by batch_size × latent_dim (128 × 2048 = 262,144)
- **Factor difference: 2048x!**

### Why This Caused Collapse

1. L1 penalty too weak → features not sparse enough
2. Many features activate simultaneously → redundancy
3. Gradient updates favor active features
4. Inactive features die (no gradient signal)
5. Death cascade accelerates (positive feedback)
6. Result: 95-99% dead neurons

---

## Files Modified

1. `backend/src/ml/sparse_autoencoder.py` - **3 changes**
   - Added normalization safety (lines 109-112)
   - Fixed L1 penalty in SparseAutoencoder (lines 218-222)
   - Fixed L1 penalty in SkipAutoencoder (lines 364-365)
   - Fixed L1 penalty in Transcoder (lines 492-493)

---

## Commit Message

```
fix: correct L1 penalty calculation in SAE training (CRITICAL BUG FIX)

BREAKING CHANGE: L1 penalty now correctly sums over latent dimensions before
averaging over batch, making it ~2000x stronger. This matches the formulation
from Anthropic's "Towards Monosemanticity" paper.

Fixes:
1. L1 penalty calculation: z.abs().mean() → z.abs().sum(dim=-1).mean()
2. Normalization safety: added torch.clamp(x_norm, min=1e-6) to prevent division by zero
3. Applied fixes to all SAE architectures: SparseAutoencoder, SkipAutoencoder, Transcoder

Impact:
- L1 penalty is now 2048x stronger for latent_dim=2048
- l1_alpha values must be reduced by factor of latent_dim (see migration guide)
- Training should now be stable (no 95-99% neuron death)
- L0 sparsity should decrease toward target (not increase)

Migration:
- Old trainings are incompatible - start fresh with new l1_alpha values
- Recommended l1_alpha for latent_dim=2048: 0.000002 (was 0.003)
- See SAE_TRAINING_BUG_FIX_SUMMARY.md for full migration guide

Root Cause:
- Bug caused consistent training collapse at steps 5k-10k regardless of hyperparameters
- L0 sparsity INCREASED from 11.86% to 15.62% when it should have decreased
- Both failed trainings had 95-99% dead neurons at completion

Related: SAE_TRAINING_BUG_INVESTIGATION_PLAN.md
```

---

**Status**: ✅ All fixes implemented, ready for testing
**Date**: 2025-11-22
**Reviewed**: User approved plan execution
