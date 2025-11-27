# Task List: Gemma Scope Recreation Implementation

**Feature:** JumpReLU SAE Architecture & Gemma Scope Compatibility
**Priority:** P0 (Critical for Research Compatibility)
**Estimated Effort:** 5-7 days
**Created:** 2025-11-26
**Status:** Not Started

## Overview

This task list implements all missing components required to replicate Google DeepMind's Gemma Scope methodology in miStudio. Based on the Gemma Scope Technical Report (arXiv:2408.05147v2) and the JumpReLU paper (arXiv:2407.14435).

## Reference Documents
- [Gemma Scope Recreation Cookbook](../docs/Gemma_Scope_Recreation_Cookbook.md)
- [Gemma Scope Paper](https://arxiv.org/abs/2408.05147)
- [JumpReLU Paper](https://arxiv.org/abs/2407.14435)

---

## Phase 1: JumpReLU Architecture Implementation

### Task 1.1: Create JumpReLU Activation Class
**File:** `backend/src/ml/sparse_autoencoder.py`
**Effort:** 4 hours
**Priority:** P0

- [ ] **1.1.1** Create `JumpReLU` class with learnable thresholds
  - Initialize thresholds as `nn.Parameter` with small positive values (0.001)
  - Implement forward: `z * (z > threshold).float()`
  - Store pre-activations for STE gradient computation

- [ ] **1.1.2** Implement Straight-Through Estimator (STE) for threshold gradients
  - Use kernel density estimation (KDE) for threshold gradient approximation
  - Bandwidth parameter `ε = 0.001` (configurable)
  - Reference: Rajamanoharan et al. 2024, Section 3.2

- [ ] **1.1.3** Add threshold initialization options
  - Default: 0.001 for all features
  - Option: Initialize from data statistics
  - Option: Load from pretrained checkpoint

**Acceptance Criteria:**
- JumpReLU produces sparse outputs with learnable thresholds
- Gradients flow through both z and θ during backprop
- Unit tests pass for forward/backward

---

### Task 1.2: Create JumpReLUSAE Model Class
**File:** `backend/src/ml/sparse_autoencoder.py`
**Effort:** 6 hours
**Priority:** P0

- [ ] **1.2.1** Create `JumpReLUSAE` class extending base architecture
  - Encoder: `z = W_enc @ x + b_enc`
  - Activation: `f = JumpReLU_θ(z)`
  - Decoder: `x_hat = W_dec @ f + b_dec`

- [ ] **1.2.2** Implement weight initialization (Gemma Scope methodology)
  - He-uniform initialization for decoder
  - Normalize decoder columns to unit norm
  - Initialize encoder as transpose of decoder

- [ ] **1.2.3** Add `normalize_decoder()` method
  - Project decoder columns to unit norm after each optimizer step
  - Use `F.normalize(W_dec, dim=0, p=2)`

- [ ] **1.2.4** Implement L0 loss computation
  - Count non-zero activations: `(f != 0).float().sum(dim=-1).mean()`
  - Multiply by sparsity coefficient λ
  - Add to reconstruction loss

- [ ] **1.2.5** Add forward pass with loss dictionary
  - Return: `x_hat, f, losses_dict`
  - Include: `loss_total`, `loss_reconstruction`, `loss_l0`, `l0_sparsity`

**Acceptance Criteria:**
- JumpReLUSAE produces correct reconstructions
- L0 loss properly computed and differentiable (via STE)
- Decoder columns remain unit norm after training steps

---

### Task 1.3: Update SAE Factory and Schemas
**Files:** `backend/src/ml/sparse_autoencoder.py`, `backend/src/schemas/training.py`
**Effort:** 2 hours
**Priority:** P0

- [ ] **1.3.1** Add `JUMPRELU` to `SAEArchitectureType` enum
  ```python
  class SAEArchitectureType(str, Enum):
      STANDARD = "standard"
      SKIP = "skip"
      TRANSCODER = "transcoder"
      JUMPRELU = "jumprelu"  # NEW
  ```

- [ ] **1.3.2** Update `create_sae()` factory function
  - Add case for `architecture_type == 'jumprelu'`
  - Pass JumpReLU-specific parameters (kde_bandwidth, initial_threshold)

- [ ] **1.3.3** Add JumpReLU hyperparameters to `TrainingHyperparameters` schema
  - `kde_bandwidth: float = 0.001` (STE bandwidth)
  - `initial_threshold: float = 0.001` (initial θ value)
  - `normalize_decoder: bool = True` (unit norm constraint)
  - `project_gradients: bool = True` (gradient projection)

**Acceptance Criteria:**
- JumpReLU SAE can be created via factory
- All hyperparameters validated by Pydantic
- API accepts JumpReLU training requests

---

## Phase 2: Training Infrastructure Updates

### Task 2.1: Implement Gradient Projection
**File:** `backend/src/workers/training_tasks.py`
**Effort:** 3 hours
**Priority:** P1

- [ ] **2.1.1** Create `project_decoder_gradients()` utility function
  ```python
  def project_decoder_gradients(model):
      """Project decoder gradients orthogonal to decoder columns."""
      if model.W_dec.grad is not None:
          W = model.W_dec.data
          G = model.W_dec.grad
          # G_perp = G - W * (W^T G)
          parallel = (W * G).sum(dim=0, keepdim=True)
          model.W_dec.grad = G - W * parallel
  ```

- [ ] **2.1.2** Integrate into training loop
  - Call after `loss.backward()` and before `optimizer.step()`
  - Only when `hp['project_gradients'] == True`
  - Only for JumpReLU architecture

- [ ] **2.1.3** Add decoder normalization step
  - Call `model.normalize_decoder()` after `optimizer.step()`
  - Only when `hp['normalize_decoder'] == True`

**Acceptance Criteria:**
- Decoder columns remain unit norm throughout training
- Gradient projection doesn't break training dynamics
- Performance impact < 5%

---

### Task 2.2: Update Optimizer Configuration
**File:** `backend/src/workers/training_tasks.py`
**Effort:** 2 hours
**Priority:** P1

- [ ] **2.2.1** Add Adam beta configuration to hyperparameters
  ```python
  adam_beta1: float = Field(0.9, ge=0.0, le=1.0)  # Default 0.9, Gemma Scope uses 0.0
  adam_beta2: float = Field(0.999, ge=0.0, le=1.0)
  adam_epsilon: float = Field(1e-8, gt=0)
  ```

- [ ] **2.2.2** Update optimizer creation in training loop
  ```python
  optimizer = torch.optim.Adam(
      model.parameters(),
      lr=hp['learning_rate'],
      betas=(hp.get('adam_beta1', 0.9), hp.get('adam_beta2', 0.999)),
      eps=hp.get('adam_epsilon', 1e-8),
      weight_decay=hp.get('weight_decay', 0.0)
  )
  ```

- [ ] **2.2.3** Add Gemma Scope preset to training templates
  - Pre-configured hyperparameters matching paper
  - `adam_beta1=0.0`, `kde_bandwidth=0.001`, etc.

**Acceptance Criteria:**
- Optimizer correctly uses custom beta values
- Gemma Scope preset available in UI
- Training matches paper methodology

---

### Task 2.3: Implement Sparsity Warmup
**File:** `backend/src/workers/training_tasks.py`
**Effort:** 2 hours
**Priority:** P1

- [ ] **2.3.1** Add sparsity warmup hyperparameters
  ```python
  sparsity_warmup_steps: int = Field(10000, ge=0)
  ```

- [ ] **2.3.2** Implement warmup in loss computation
  ```python
  if step < hp['sparsity_warmup_steps']:
      warmup_factor = step / hp['sparsity_warmup_steps']
      sparsity_loss = warmup_factor * sparsity_loss
  ```

- [ ] **2.3.3** Log warmup factor in metrics
  - Track when warmup completes
  - Visible in training dashboard

**Acceptance Criteria:**
- Sparsity penalty gradually increases during warmup
- Training more stable in early steps
- Warmup progress visible in UI

---

## Phase 3: Evaluation Metrics

### Task 3.1: Implement FVU (Fraction of Variance Unexplained)
**Files:** `backend/src/ml/sparse_autoencoder.py`, `backend/src/models/training_metric.py`
**Effort:** 3 hours
**Priority:** P1

- [ ] **3.1.1** Add FVU computation to SAE forward pass
  ```python
  # In forward() method:
  var_original = x.var()
  var_residuals = (x - x_reconstructed).var()
  fvu = var_residuals / var_original
  losses['fvu'] = fvu
  ```

- [ ] **3.1.2** Add `fvu` column to TrainingMetric model
  ```python
  fvu = Column(Float, nullable=True, comment="Fraction of Variance Unexplained")
  ```

- [ ] **3.1.3** Create database migration for new column
  - Alembic migration script
  - Nullable to support existing data

- [ ] **3.1.4** Update training loop to log FVU
  - Store in TrainingMetric table
  - Emit via WebSocket with other metrics

- [ ] **3.1.5** Add FVU to API response schemas
  - `TrainingMetricResponse.fvu: Optional[float]`
  - Include in metrics endpoint

**Acceptance Criteria:**
- FVU computed and logged every log_interval
- FVU visible in training dashboard
- FVU values match expected range (0.0-1.0)

---

### Task 3.2: Implement Delta Loss (LM Impact)
**Files:** `backend/src/services/evaluation_service.py` (new), `backend/src/workers/training_tasks.py`
**Effort:** 6 hours
**Priority:** P2

- [ ] **3.2.1** Create EvaluationService for SAE quality metrics
  ```python
  class EvaluationService:
      async def compute_delta_loss(self, model, sae, eval_tokens, layer_name):
          """Compute LM loss increase when SAE is spliced in."""
  ```

- [ ] **3.2.2** Implement baseline perplexity computation
  - Forward pass without SAE
  - Compute cross-entropy loss on next-token prediction

- [ ] **3.2.3** Implement SAE-spliced perplexity computation
  - Register hook to replace activations with SAE reconstruction
  - Forward pass with hook
  - Compute cross-entropy loss

- [ ] **3.2.4** Add delta loss to checkpoint evaluation
  - Compute at checkpoint intervals (optional, expensive)
  - Store in checkpoint metadata
  - New `CheckpointEvaluation` table

- [ ] **3.2.5** Add API endpoint for delta loss computation
  ```
  POST /api/v1/trainings/{training_id}/evaluate
  {
    "checkpoint_id": "ckpt_xxx",
    "metrics": ["delta_loss", "fvu"]
  }
  ```

**Acceptance Criteria:**
- Delta loss computed correctly (positive = worse)
- Results cached for expensive computations
- API returns evaluation results

---

### Task 3.3: Implement Sparsity-Fidelity Curve Generation
**Files:** `backend/src/services/evaluation_service.py`, `frontend/src/components/training/`
**Effort:** 8 hours
**Priority:** P2

- [ ] **3.3.1** Create sparsity coefficient sweep endpoint
  ```
  POST /api/v1/trainings/sweep
  {
    "model_id": "m_xxx",
    "dataset_id": "ds_xxx",
    "sparsity_coefficients": [1e-5, 3e-5, 1e-4, ...],
    "base_hyperparameters": {...}
  }
  ```

- [ ] **3.3.2** Implement sweep orchestration
  - Queue multiple training jobs with different λ values
  - Track sweep as parent job with child trainings
  - New `TrainingSweep` model

- [ ] **3.3.3** Implement results aggregation
  - Collect final L0 and FVU from each training
  - Compute Pareto frontier

- [ ] **3.3.4** Create frontend visualization component
  - Scatter plot: L0 vs FVU
  - Scatter plot: L0 vs Delta Loss (if computed)
  - Pareto frontier line
  - Clickable points to view training details

- [ ] **3.3.5** Add export functionality
  - CSV export of sweep results
  - PNG/SVG export of plots

**Acceptance Criteria:**
- User can launch sparsity sweep from UI
- Results plotted automatically
- Pareto frontier clearly visible

---

## Phase 4: Community Format & Compatibility

### Task 4.1: Update Community Format for JumpReLU
**File:** `backend/src/ml/community_format.py`
**Effort:** 3 hours
**Priority:** P1

- [ ] **4.1.1** Add JumpReLU fields to `CommunitySAEConfig`
  ```python
  @dataclass
  class CommunitySAEConfig:
      architecture: str = "jumprelu"  # NEW option
      activation_fn_str: str = "jumprelu"  # NEW option
      threshold: Optional[List[float]] = None  # Per-feature thresholds
  ```

- [ ] **4.1.2** Update `save_community_format()` for JumpReLU
  - Save threshold parameters in weights file
  - Save JumpReLU-specific config in cfg.json

- [ ] **4.1.3** Update `load_community_format()` for JumpReLU
  - Detect JumpReLU architecture from config
  - Load threshold parameters
  - Initialize JumpReLUSAE correctly

- [ ] **4.1.4** Add compatibility with Gemma Scope HuggingFace format
  - Test loading from `google/gemma-scope-*` repos
  - Map their format to miStudio format

**Acceptance Criteria:**
- JumpReLU SAEs save/load in community format
- Can load official Gemma Scope SAEs from HuggingFace
- Format compatible with SAELens and Neuronpedia

---

### Task 4.2: Add Gemma Scope SAE Import
**Files:** `backend/src/services/huggingface_sae_service.py`, `backend/src/api/v1/endpoints/saes.py`
**Effort:** 4 hours
**Priority:** P1

- [ ] **4.2.1** Add Gemma Scope repo detection
  - Recognize `google/gemma-scope-*` format
  - Parse layer/width from repo structure

- [ ] **4.2.2** Implement Gemma Scope weight conversion
  - Map Gemma Scope tensor names to miStudio format
  - Handle threshold parameters

- [ ] **4.2.3** Add UI for browsing Gemma Scope SAEs
  - Dropdown to select model (2B, 9B, 27B)
  - Dropdown to select layer and width
  - One-click import

- [ ] **4.2.4** Test with official Gemma Scope SAEs
  - Verify steering works with imported SAEs
  - Verify feature extraction works

**Acceptance Criteria:**
- Can import any Gemma Scope SAE from HuggingFace
- Imported SAEs work for steering
- Feature browser shows correct activations

---

## Phase 5: Testing

### Task 5.1: Unit Tests for JumpReLU
**File:** `backend/tests/unit/test_jumprelu_sae.py` (new)
**Effort:** 4 hours
**Priority:** P0

- [ ] **5.1.1** Test JumpReLU activation function
  - Test forward pass produces sparse output
  - Test threshold gating works correctly
  - Test gradients flow through STE

- [ ] **5.1.2** Test JumpReLUSAE model
  - Test encode/decode roundtrip
  - Test loss computation
  - Test decoder normalization

- [ ] **5.1.3** Test gradient projection
  - Verify gradients orthogonal to decoder columns
  - Verify decoder stays unit norm

- [ ] **5.1.4** Test STE gradient computation
  - Verify threshold gradients computed via KDE
  - Test bandwidth parameter effect

**Acceptance Criteria:**
- All unit tests pass
- Coverage > 80% for new code
- Edge cases covered (zero input, all zeros, etc.)

---

### Task 5.2: Integration Tests
**File:** `backend/tests/integration/test_jumprelu_training.py` (new)
**Effort:** 4 hours
**Priority:** P1

- [ ] **5.2.1** Test JumpReLU training end-to-end
  - Create training job with JumpReLU architecture
  - Verify training completes
  - Verify checkpoint saved correctly

- [ ] **5.2.2** Test FVU and metrics logging
  - Verify FVU computed and stored
  - Verify WebSocket emissions include FVU

- [ ] **5.2.3** Test community format save/load
  - Train JumpReLU SAE
  - Save in community format
  - Load and verify identical

- [ ] **5.2.4** Test Gemma Scope import
  - Import official SAE
  - Verify steering works

**Acceptance Criteria:**
- All integration tests pass
- Training produces valid SAE
- Import/export cycle preserves weights

---

## Phase 6: Documentation

### Task 6.1: Update Documentation
**Effort:** 3 hours
**Priority:** P2

- [ ] **6.1.1** Update CLAUDE.md with JumpReLU architecture
  - Add to supported architectures list
  - Document new hyperparameters

- [ ] **6.1.2** Create JumpReLU training guide
  - Recommended hyperparameters
  - Expected training behavior
  - Troubleshooting tips

- [ ] **6.1.3** Update API documentation
  - New endpoints
  - New schema fields
  - Example requests

- [ ] **6.1.4** Update Gemma Scope Recreation Cookbook
  - Mark implemented features as complete
  - Add miStudio-specific instructions

**Acceptance Criteria:**
- Documentation complete and accurate
- Examples work as shown
- Cookbook reflects current capabilities

---

## Summary

### Tasks by Priority

**P0 (Critical - Must Have):**
- Task 1.1: JumpReLU Activation Class
- Task 1.2: JumpReLUSAE Model Class
- Task 1.3: Factory and Schema Updates
- Task 5.1: Unit Tests

**P1 (High - Should Have):**
- Task 2.1: Gradient Projection
- Task 2.2: Optimizer Configuration
- Task 2.3: Sparsity Warmup
- Task 3.1: FVU Metric
- Task 4.1: Community Format Update
- Task 4.2: Gemma Scope Import
- Task 5.2: Integration Tests

**P2 (Medium - Nice to Have):**
- Task 3.2: Delta Loss
- Task 3.3: Sparsity-Fidelity Curves
- Task 6.1: Documentation

### Estimated Timeline

| Phase | Tasks | Effort | Dependencies |
|-------|-------|--------|--------------|
| Phase 1 | 1.1, 1.2, 1.3 | 12 hours | None |
| Phase 2 | 2.1, 2.2, 2.3 | 7 hours | Phase 1 |
| Phase 3 | 3.1, 3.2, 3.3 | 17 hours | Phase 1, 2 |
| Phase 4 | 4.1, 4.2 | 7 hours | Phase 1 |
| Phase 5 | 5.1, 5.2 | 8 hours | Phase 1, 2, 3, 4 |
| Phase 6 | 6.1 | 3 hours | All |

**Total Estimated Effort:** 54 hours (~5-7 working days)

### Files to Create/Modify

**New Files:**
- `backend/tests/unit/test_jumprelu_sae.py`
- `backend/tests/integration/test_jumprelu_training.py`
- `backend/src/services/evaluation_service.py`
- `backend/alembic/versions/xxx_add_fvu_column.py`
- `docs/JumpReLU_Training_Guide.md`

**Modified Files:**
- `backend/src/ml/sparse_autoencoder.py` (major changes)
- `backend/src/schemas/training.py`
- `backend/src/workers/training_tasks.py`
- `backend/src/models/training_metric.py`
- `backend/src/ml/community_format.py`
- `backend/src/services/huggingface_sae_service.py`
- `frontend/src/types/training.ts`
- `frontend/src/components/training/TrainingForm.tsx`
- `CLAUDE.md`

---

## Relevant Existing Files

| File | Purpose | Lines |
|------|---------|-------|
| `backend/src/ml/sparse_autoencoder.py` | SAE implementations | 586 |
| `backend/src/workers/training_tasks.py` | Training loop | 1269 |
| `backend/src/schemas/training.py` | Training schemas | 311 |
| `backend/src/models/training_metric.py` | Metrics model | 70 |
| `backend/src/ml/community_format.py` | Community format | 200+ |
| `backend/src/services/huggingface_sae_service.py` | HF integration | 300+ |

---

*Last Updated: 2025-11-26*
*Review Status: Multi-agent review complete*
