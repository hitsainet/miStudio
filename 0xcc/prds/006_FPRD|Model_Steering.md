# Feature PRD: Model Steering

**Document ID:** 006_FPRD|Model_Steering
**Version:** 1.0 (MVP Complete)
**Last Updated:** 2025-12-05
**Status:** Implemented
**Priority:** P0 (Core Feature)

---

## 1. Overview

### 1.1 Purpose
Enable users to control model behavior by intervening on SAE feature activations during generation.

### 1.2 User Problem
Researchers need to test feature hypotheses by:
- Amplifying or suppressing specific features
- Comparing steered vs. unsteered outputs
- Testing multiple steering strengths
- Documenting experiments with saved prompts

### 1.3 Solution
A comprehensive steering interface with feature selection, strength control, comparison mode, and prompt templates.

---

## 2. Functional Requirements

### 2.1 Feature Selection
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | Browse features from SAE | Implemented |
| FR-1.2 | Search features by label | Implemented |
| FR-1.3 | Select multiple features | Implemented |
| FR-1.4 | View feature details before selection | Implemented |
| FR-1.5 | Quick-select from recent features | Planned |

### 2.2 Steering Configuration
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | Set steering strength per feature (-10 to +10) | Implemented |
| FR-2.2 | Steering type: activation or suppression | Implemented |
| FR-2.3 | Multi-feature steering (combine interventions) | Implemented |
| FR-2.4 | Neuronpedia-compatible calibration | Implemented |

### 2.3 Generation
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | Generate text with steering applied | Implemented |
| FR-3.2 | Configure generation parameters (temp, top_p, max_tokens) | Implemented |
| FR-3.3 | Comparison mode (steered vs. baseline) | Implemented |
| FR-3.4 | Strength sweep (multiple strengths in one run) | Implemented |

### 2.4 Results & Export
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | Display side-by-side comparison | Implemented |
| FR-4.2 | Highlight differences in outputs | Implemented |
| FR-4.3 | Export results to JSON | Implemented |
| FR-4.4 | Save results to history | Planned |

### 2.5 Prompt Templates (Sub-feature)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-5.1 | Create prompt templates | Implemented |
| FR-5.2 | Variable substitution in prompts | Implemented |
| FR-5.3 | Template favorites | Implemented |
| FR-5.4 | Template import/export | Implemented |

---

## 3. Steering Mechanics

### 3.1 Activation Steering
```python
def steering_hook(output, feature_idx, strength):
    # Encode to feature space
    features = sae.encode(output)

    # Modify feature activation
    # strength > 0: amplify, strength < 0: suppress
    features[:, :, feature_idx] += strength * calibration_factor

    # Decode back to residual stream
    return sae.decode(features)
```

### 3.2 Calibration
- Based on Neuronpedia calibration approach
- Feature direction normalized to unit norm
- Strength scaled by activation standard deviation

### 3.3 Multi-Feature Steering
```python
for feature_config in selected_features:
    output = steer(output, feature_config.idx, feature_config.strength)
```

---

## 4. User Interface

### 4.1 Steering Panel
```
┌─────────────────────────────────────────────────────────────┐
│ Model Steering                                              │
├─────────────────────────────────────────────────────────────┤
│ SAE: [gemma-2b-layer12 ▾]  Model: [google/gemma-2-2b]      │
├─────────────────────────────────────────────────────────────┤
│ Selected Features (3)                       [+ Add Feature] │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Feature #42: "romantic love"                            │ │
│ │ Strength: [═══════●═══] +3.5                   [Remove]│ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Feature #156: "formal language"                         │ │
│ │ Strength: [═══●═══════] -2.0                   [Remove]│ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Prompt:                                                     │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Write a letter to...                                    │ │
│ └─────────────────────────────────────────────────────────┘ │
│ [Load Template]                                             │
├─────────────────────────────────────────────────────────────┤
│ [✓] Comparison Mode  [Sweep Strengths: 3]                  │
│                                                             │
│ [Generate]                                                  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Results View
```
┌─────────────────────────────────────────────────────────────┐
│ Results                                         [Export]    │
├──────────────────────────┬──────────────────────────────────┤
│ Baseline                 │ Steered (+3.5)                   │
├──────────────────────────┼──────────────────────────────────┤
│ Dear John,               │ My dearest beloved John,         │
│ I hope this letter       │ I hope this [love] letter        │
│ finds you well...        │ finds you [forever] well...      │
└──────────────────────────┴──────────────────────────────────┘
```

### 4.3 Feature Browser Modal
- Search/filter features
- View feature statistics
- Preview top activations
- Quick-add to steering

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/steering/generate` | POST | Generate with steering |
| `/api/v1/steering/compare` | POST | Compare steered vs. baseline |
| `/api/v1/steering/sweep` | POST | Test multiple strengths |
| `/api/v1/steering/calibrate` | POST | Get calibration factors |
| `/api/v1/prompt-templates` | GET/POST | Prompt template CRUD |
| `/api/v1/prompt-templates/{id}` | GET/PUT/DELETE | Template by ID |

---

## 6. Data Model

### 6.1 PromptTemplate Table
```sql
CREATE TABLE prompt_templates (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    variables JSONB,  -- Variable definitions
    template_type VARCHAR(50),  -- steering, labeling, etc.
    is_favorite BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 6.2 SteeringResult (Transient)
```python
class SteeringResult:
    prompt: str
    baseline_output: str
    steered_outputs: List[SteerOutput]
    features: List[FeatureConfig]
    generation_params: GenerationParams
    timestamp: datetime
```

---

## 7. Key Files

### Backend
- `backend/src/services/steering_service.py` - Steering logic
- `backend/src/ml/forward_hooks.py` - Hook implementations
- `backend/src/api/v1/endpoints/steering.py` - API routes
- `backend/src/schemas/steering.py` - Request/response schemas

### Frontend
- `frontend/src/components/panels/SteeringPanel.tsx` - Main panel
- `frontend/src/components/steering/FeatureBrowser.tsx` - Feature selection
- `frontend/src/components/steering/SelectedFeatureCard.tsx` - Feature card
- `frontend/src/components/steering/StrengthSlider.tsx` - Strength control
- `frontend/src/components/steering/ComparisonResults.tsx` - Results view
- `frontend/src/components/steering/PromptListEditor.tsx` - Prompt input
- `frontend/src/stores/steeringStore.ts` - State management

---

## 8. Generation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_new_tokens` | Maximum tokens to generate | 100 | 10 - 500 |
| `temperature` | Sampling temperature | 0.7 | 0.0 - 2.0 |
| `top_p` | Nucleus sampling | 0.9 | 0.0 - 1.0 |
| `top_k` | Top-k sampling | 50 | 1 - 100 |
| `repetition_penalty` | Penalize repeats | 1.0 | 1.0 - 2.0 |

---

## 9. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| Model Management | Provides base model |
| SAE Management | Provides SAE for steering |
| Feature Discovery | Provides features to select |

---

## 10. Testing Checklist

- [x] Single feature steering
- [x] Multi-feature steering
- [x] Comparison mode
- [x] Strength sweep
- [x] Feature browser search
- [x] Prompt templates
- [x] Export results
- [x] Negative strength (suppression)
- [x] Generation parameter configuration

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/006_FTDD|Model_Steering.md) | [TID](../tids/006_FTID|Model_Steering.md)*
