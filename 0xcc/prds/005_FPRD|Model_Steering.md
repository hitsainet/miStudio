# Feature PRD: Model Steering

**Document ID:** 005_FPRD|Model_Steering
**Feature:** Feature-Based Model Behavior Modification and Intervention
**Status:** Draft
**Created:** 2025-10-06
**Last Updated:** 2025-10-06
**Owner:** miStudio Development Team

---

## 1. Feature Overview

### Purpose
Enable users to modify language model behavior by directly intervening on discovered SAE features during text generation. This feature allows researchers to amplify or suppress specific interpretable features to steer model outputs toward desired characteristics, providing a controllable interface for mechanistic interpretability research and AI safety experiments.

### Business Value
- **Primary Value:** Transforms discovered features into actionable control mechanisms for model behavior
- **Research Value:** Enables causal analysis of feature importance and mechanistic understanding
- **Safety Value:** Provides controlled testing ground for model behavior modification techniques
- **Educational Value:** Demonstrates direct connection between interpretable features and model outputs

### Success Criteria
- Users can successfully select features and configure steering coefficients
- Side-by-side comparison shows clear behavioral differences between steered and unsteered outputs
- Steering interventions execute within 3 seconds for 100-token generation
- Comparison metrics (KL divergence, semantic similarity) provide meaningful quantitative feedback
- System handles multiple simultaneous feature interventions without performance degradation

### Target Users
- **ML Researchers:** Investigating causal relationships between features and model behavior
- **AI Safety Practitioners:** Testing robustness and controllability of interpretable interventions
- **Interpretability Scientists:** Validating feature interpretations through steering experiments

---

## 2. Feature Context

### Relationship to Project PRD
This feature implements **Core Feature #5: Model Steering** from the Project PRD (000_PPRD|miStudio.md), specifically:
- Priority: **P0 (Critical MVP feature)**
- Dependencies: Feature Discovery (requires extracted interpretable features)
- Enables: Practical application of interpretability research (controlled model outputs)

### Integration with Existing Features

**Upstream Dependencies:**
- **Feature Discovery (004_FPRD):** Requires extracted features with known semantic meanings
- **Model Management (002_FPRD):** Requires models with activation extraction and forward hooks
- **SAE Training (003_FPRD):** Requires trained SAE models for feature-to-activation mapping

**Downstream Consumers:**
- None (terminal feature in the workflow)

### UI/UX Reference
**PRIMARY REFERENCE:** `@0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
- **SteeringPanel Component:** Lines 3512-3952 (complete steering interface)
- **Feature Selection Panel:** Lines 3582-3706 (search, add features, coefficient sliders)
- **Generation Controls:** Lines 3710-3817 (model, prompt, intervention layer, parameters)
- **Comparative Output Display:** Lines 3820-3895 (side-by-side unsteered vs steered)
- **Comparison Metrics:** Lines 3898-3947 (KL divergence, perplexity, similarity, word overlap)

---

## 3. User Stories & Scenarios

### US-1: Select Features for Steering (Priority: P0)
**As a** researcher
**I want to** search and select interpretable features to use for steering
**So that** I can configure interventions to modify model behavior

**Acceptance Criteria:**
- Given I'm on the Steering panel, I see a "Select Features to Steer" section
- When I type in the feature search box, I see a dropdown of matching features (filtered by label)
- When I click a feature from the search results, it is added to "Selected Features" list
- When I add a feature, it initializes with coefficient 0.0 (neutral)
- When a feature is already selected, it does not appear in search results
- Given I have selected features, I see count indicator: "Selected Features (N)"

**UI Reference:** Mock-embedded-interp-ui.tsx Feature Selection Panel (lines 3582-3619)

---

### US-2: Configure Steering Coefficients (Priority: P0)
**As a** researcher with selected features
**I want to** adjust steering coefficients for each feature
**So that** I can control the strength and direction of interventions

**Acceptance Criteria:**
- Given I have selected features, each feature shows: Feature ID, Feature label, Remove button (X icon), Coefficient slider, Current coefficient value
- When I drag the coefficient slider, the value updates in real-time (range: -5.0 to +5.0, step 0.1)
- When I set coefficient to negative values (< 0), I am suppressing the feature
- When I set coefficient to positive values (> 0), I am amplifying the feature
- When I set coefficient to 0.0, the feature has no effect (neutral)
- Given coefficient slider, I see labels: "-5.0 (suppress)" | "0.0" | "+5.0 (amplify)"
- When I click quick preset buttons, coefficient updates: "Suppress" → -2.0, "Reset" → 0.0, "Amplify" → +2.0

**UI Reference:** Mock-embedded-interp-ui.tsx Coefficient Sliders (lines 3632-3702)

---

### US-3: Remove Features from Steering (Priority: P0)
**As a** user configuring steering
**I want to** remove features from the selected list
**So that** I can refine my intervention configuration

**Acceptance Criteria:**
- When I click the X button on a selected feature, it is removed from the list
- When a feature is removed, its coefficient configuration is discarded
- When a feature is removed, it reappears in search results for re-addition

**UI Reference:** Mock-embedded-interp-ui.tsx Remove Feature (lines 3641-3648)

---

### US-4: Configure Generation Parameters (Priority: P0)
**As a** researcher setting up steering experiment
**I want to** configure model, prompt, and generation parameters
**So that** I can control the experimental conditions

**Acceptance Criteria:**
- Given I'm in the "Generation Configuration" panel, I can select: Model (dropdown, filtered to ready models), Prompt (textarea, 4 rows), Intervention Layer (slider, range 0-24, default 12), Temperature (number input, range 0-2, step 0.1, default 0.7), Max Tokens (number input, range 1-2048, default 100)
- When I adjust the intervention layer slider, I see current value displayed: "Intervention Layer: {value}"
- When I change any parameter, the configuration updates immediately
- When prompt is empty OR no features are selected, "Generate Comparison" button is disabled

**UI Reference:** Mock-embedded-interp-ui.tsx Generation Controls (lines 3710-3817)

---

### US-5: Generate Steered vs Unsteered Comparison (Priority: P0)
**As a** researcher with configured steering
**I want to** generate side-by-side comparison of steered and unsteered outputs
**So that** I can observe the effect of feature interventions

**Acceptance Criteria:**
- When I click "Generate Comparison" button, both unsteered and steered generations execute
- When generation is running, I see: Loading spinner, "Generating..." text, Disabled button
- When generation completes, I see two output panels: "Unsteered (Baseline)" (slate-400 indicator dot), "Steered" (emerald-400 indicator dot)
- When outputs are displayed, I see: Generated text (whitespace preserved, wrapped), Copy to clipboard button (Copy icon)
- Given generation takes >2 seconds, I see loading indicator in both output panels

**UI Reference:** Mock-embedded-interp-ui.tsx Comparative Output (lines 3820-3895)

---

### US-6: View Comparison Metrics (Priority: P1)
**As a** researcher analyzing steering results
**I want to** see quantitative metrics comparing steered and unsteered outputs
**So that** I can assess the magnitude and type of behavioral change

**Acceptance Criteria:**
- Given generation is complete, I see "Comparison Metrics" panel with 4 metrics
- When I view metrics, I see: KL Divergence (purple-400, 4 decimal places, "Distribution shift" label), Perplexity Delta (red-400 if positive, emerald-400 if negative, 2 decimal places, "Higher/Lower uncertainty" label), Semantic Similarity (blue-400, percentage, 1 decimal place, "Cosine similarity" label), Word Overlap (emerald-400, percentage, 1 decimal place, "Shared tokens" label)
- When perplexity delta is positive, I understand steered output is less confident
- When semantic similarity is high (>0.8), I understand outputs are semantically similar despite steering

**UI Reference:** Mock-embedded-interp-ui.tsx Comparison Metrics (lines 3898-3947)

---

### US-7: Copy Generated Outputs (Priority: P1)
**As a** user with generated results
**I want to** copy outputs to clipboard
**So that** I can use them in external analysis or reports

**Acceptance Criteria:**
- When I click the copy icon on unsteered output, the text is copied to clipboard
- When I click the copy icon on steered output, the text is copied to clipboard
- When copy succeeds, I see brief visual feedback (button state change or toast)

**UI Reference:** Mock-embedded-interp-ui.tsx Copy buttons (lines 3832-3840, 3867-3876)

---

### SC-1: Save Steering Configuration as Preset (Priority: P2)
**As a** researcher with a useful steering configuration
**I want to** save the configuration as a reusable preset
**So that** I can quickly apply the same intervention to different prompts

**Acceptance Criteria:**
- When I click "Save as Preset", I provide a name and optional description
- When preset is saved, it includes: Selected features and coefficients, Intervention layer, Generation parameters (temperature, max_tokens)
- Given presets exist, I can load them via "Load Preset" dropdown
- When I load a preset, all configuration fields populate automatically

---

### SC-2: Batch Steering Experiments (Priority: P2)
**As a** researcher
**I want to** run multiple steering configurations on the same prompt
**So that** I can compare different intervention strategies

**Acceptance Criteria:**
- When I create multiple steering configurations, I can run them all at once
- When batch completes, I see a comparison table showing all outputs and metrics
- When I view results, I can sort by metric values to identify best configurations

---

## 4. Functional Requirements

### FR-1: Feature Selection and Search (8 requirements)

**FR-1.1:** System SHALL provide API endpoint `GET /api/features/search?q={query}` to search features by label
**FR-1.2:** System SHALL filter search results to exclude already-selected features (client-side)
**FR-1.3:** System SHALL display search results dropdown when query is non-empty and results exist
**FR-1.4:** System SHALL limit search results to top 20 features (for performance)
**FR-1.5:** System SHALL support adding features to selected list via click action
**FR-1.6:** System SHALL initialize new features with coefficient 0.0 (neutral intervention)
**FR-1.7:** System SHALL clear search query after feature is added
**FR-1.8:** System SHALL display count of selected features: "Selected Features ({count})"

---

### FR-2: Coefficient Configuration (10 requirements)

**FR-2.1:** System SHALL provide slider input for each selected feature with range -5.0 to +5.0, step 0.1
**FR-2.2:** System SHALL display current coefficient value with 2 decimal places
**FR-2.3:** System SHALL update coefficient in real-time as slider moves
**FR-2.4:** System SHALL interpret coefficient semantics as:
- Negative values: Suppress feature (reduce activation)
- Zero: Neutral (no intervention)
- Positive values: Amplify feature (increase activation)

**FR-2.5:** System SHALL provide quick preset buttons with values:
- "Suppress": -2.0
- "Reset": 0.0
- "Amplify": +2.0

**FR-2.6:** System SHALL display slider end labels: "-5.0 (suppress)" | "0.0" | "+5.0 (amplify)"
**FR-2.7:** System SHALL persist coefficient values in component state (not database)
**FR-2.8:** System SHALL support removing features via X button click
**FR-2.9:** System SHALL discard coefficient configuration when feature is removed
**FR-2.10:** System SHALL use emerald accent color for slider styling (CSS accent-emerald-500)

---

### FR-3: Generation Configuration (12 requirements)

**FR-3.1:** System SHALL provide model selection dropdown filtered to models with status='ready'
**FR-3.2:** System SHALL provide prompt textarea input (4 rows, resizable)
**FR-3.3:** System SHALL provide intervention layer slider with range 0 to max_layers (model-dependent, default 24)
**FR-3.4:** System SHALL display current intervention layer value: "Intervention Layer: {value}"
**FR-3.5:** System SHALL provide temperature number input with range 0 to 2, step 0.1, default 0.7
**FR-3.6:** System SHALL provide max_tokens number input with range 1 to 2048, default 100
**FR-3.7:** System SHALL validate that intervention layer does not exceed model's actual layer count
**FR-3.8:** System SHALL disable "Generate Comparison" button when:
- Prompt is empty
- No features are selected
- No model is selected

**FR-3.9:** System SHALL display button text: "Generate Comparison" (with Zap icon) when idle, "Generating..." (with Loader spinner) when running
**FR-3.10:** System SHALL prevent multiple simultaneous generations (button disabled during generation)
**FR-3.11:** System SHALL use emerald-600 button color (bg-emerald-600 hover:bg-emerald-700)
**FR-3.12:** System SHALL validate temperature range (0-2) before submission

---

### FR-4: Steered Text Generation (15 requirements)

**FR-4.1:** System SHALL provide API endpoint `POST /api/steering/generate` to execute steering experiment
**FR-4.2:** System SHALL accept request payload:
```json
{
  "model_id": "m_gpt2_456",
  "prompt": "The cat sat on",
  "features": [
    {"feature_id": 42, "coefficient": 2.0},
    {"feature_id": 137, "coefficient": -1.5}
  ],
  "intervention_layer": 12,
  "temperature": 0.7,
  "max_tokens": 100
}
```

**FR-4.3:** System SHALL execute TWO generations in parallel (or sequential):
- Unsteered generation: Normal model forward pass without intervention
- Steered generation: Model forward pass with feature interventions applied

**FR-4.4:** System SHALL implement steering intervention algorithm:
1. Load model and SAE decoder
2. For each generation step:
   a. Compute model activations at intervention layer
   b. Pass activations through SAE encoder to get feature activations
   c. For each steered feature: Modify feature activation by coefficient (multiplicative or additive)
   d. Pass modified features through SAE decoder to reconstruct activations
   e. Replace original activations with reconstructed (steered) activations
   f. Continue model forward pass with steered activations
   g. Sample next token from output logits
   h. Repeat until max_tokens or EOS token

**FR-4.5:** System SHALL use multiplicative steering: `steered_activation[feature_id] = original_activation[feature_id] * (1 + coefficient)`
**FR-4.6:** System SHALL clip steered activations to prevent numerical instability: `clip(steered_activation, min=-10, max=10)`
**FR-4.7:** System SHALL implement forward hooks for intervention:
```python
def steering_hook(module, input, output):
    # Extract activations
    activations = output
    # Pass through SAE encoder
    feature_activations = sae_encoder(activations)
    # Apply steering coefficients
    for feature_id, coeff in steering_config.items():
        feature_activations[:, feature_id] *= (1 + coeff)
    # Reconstruct with SAE decoder
    steered_activations = sae_decoder(feature_activations)
    return steered_activations
```

**FR-4.8:** System SHALL register forward hook on model layer at intervention_layer index
**FR-4.9:** System SHALL generate unsteered output WITHOUT any hooks registered
**FR-4.10:** System SHALL use same random seed for both generations (for fair comparison) - optional, document trade-offs
**FR-4.11:** System SHALL return response payload:
```json
{
  "unsteered_output": "The cat sat on the mat peacefully.",
  "steered_output": "The cat sat on the mat joyfully, radiating happiness!",
  "metrics": {
    "kl_divergence": 0.0234,
    "perplexity_delta": -2.3,
    "semantic_similarity": 0.87,
    "word_overlap": 0.65
  }
}
```

**FR-4.12:** System SHALL implement generation timeout of 30 seconds (failsafe)
**FR-4.13:** System SHALL handle OOM errors gracefully: reduce max_tokens, return partial generation
**FR-4.14:** System SHALL log steering configurations and results for debugging
**FR-4.15:** System SHALL clean up forward hooks after generation completes

---

### FR-5: Comparison Metrics Calculation (11 requirements)

**FR-5.1:** System SHALL calculate KL divergence between unsteered and steered token distributions:
- Compute token-level log probabilities for both generations
- Calculate KL(unsteered || steered) = sum(p_unsteered * log(p_unsteered / p_steered))
- Average across all token positions

**FR-5.2:** System SHALL calculate perplexity delta:
- Compute perplexity for unsteered output: `perplexity = exp(mean(negative_log_likelihoods))`
- Compute perplexity for steered output
- Delta = steered_perplexity - unsteered_perplexity
- Positive delta: Steered output is less confident (higher uncertainty)
- Negative delta: Steered output is more confident (lower uncertainty)

**FR-5.3:** System SHALL calculate semantic similarity using sentence embeddings:
- Encode unsteered and steered outputs using sentence transformer (e.g., all-MiniLM-L6-v2)
- Compute cosine similarity between embeddings: `similarity = cosine(embed_unsteered, embed_steered)`
- Range: 0 (completely different) to 1 (identical meaning)

**FR-5.4:** System SHALL calculate word overlap:
- Tokenize both outputs into words
- Count shared words: `shared = len(set(words_unsteered) & set(words_steered))`
- Calculate Jaccard similarity: `overlap = shared / len(set(words_unsteered) | set(words_steered))`

**FR-5.5:** System SHALL return all metrics in response payload with precision:
- kl_divergence: 4 decimal places
- perplexity_delta: 2 decimal places
- semantic_similarity: 2 decimal places (displayed as percentage)
- word_overlap: 2 decimal places (displayed as percentage)

**FR-5.6:** System SHALL handle edge cases:
- If outputs are identical: kl_divergence=0, perplexity_delta=0, similarity=1.0, overlap=1.0
- If outputs are empty: return null for metrics

**FR-5.7:** System SHALL use consistent tokenization for both outputs (same tokenizer used by model)
**FR-5.8:** System SHALL cache sentence embeddings for performance (if using external model)
**FR-5.9:** System SHALL log metric calculation errors without failing entire request
**FR-5.10:** System SHALL provide interpretations for metric ranges (in UI tooltips):
- KL divergence: <0.01 (minimal change), 0.01-0.1 (moderate change), >0.1 (significant change)
- Perplexity delta: <0 (more confident), 0 (no change), >0 (less confident)
- Semantic similarity: >0.9 (very similar), 0.7-0.9 (similar), <0.7 (different)
- Word overlap: >0.5 (high overlap), 0.3-0.5 (moderate), <0.3 (low overlap)

**FR-5.11:** System SHALL support optional detailed metrics (for debugging):
- Per-token KL divergence
- Per-token perplexity
- Top token changes (unsteered → steered)

---

### FR-6: Output Display and Interaction (8 requirements)

**FR-6.1:** System SHALL display outputs in side-by-side layout (2-column grid)
**FR-6.2:** System SHALL render unsteered output panel with:
- Header: "Unsteered (Baseline)" with slate-400 indicator dot
- Output box: bg-slate-900/50 border border-slate-800, min-h-200px
- Copy button: Copy icon (slate-400 hover:slate-200)

**FR-6.3:** System SHALL render steered output panel with:
- Header: "Steered" with emerald-400 indicator dot
- Output box: bg-slate-900/50 border border-emerald-800/30, min-h-200px
- Copy button: Copy icon

**FR-6.4:** System SHALL display loading state during generation:
- Spinner icon (Loader, w-6 h-6, animate-spin)
- Centered in output box

**FR-6.5:** System SHALL preserve whitespace in output display (whitespace-pre-wrap)
**FR-6.6:** System SHALL implement copy to clipboard functionality:
- Use navigator.clipboard.writeText() API
- Show visual feedback on successful copy (future: toast notification)

**FR-6.7:** System SHALL display "No generation yet" placeholder when outputs are empty
**FR-6.8:** System SHALL show "Comparison Metrics" panel only when both outputs exist

---

### FR-7: Comparison Metrics Display (7 requirements)

**FR-7.1:** System SHALL display metrics in 4-column grid layout
**FR-7.2:** System SHALL render KL Divergence metric card:
- Label: "KL Divergence" (slate-400)
- Value: {value} (purple-400, 4 decimal places)
- Description: "Distribution shift" (slate-500)

**FR-7.3:** System SHALL render Perplexity Delta metric card:
- Label: "Perplexity Δ" (slate-400)
- Value: {sign}{value} (red-400 if positive, emerald-400 if negative, 2 decimal places)
- Description: "Higher uncertainty" or "Lower uncertainty" (slate-500)

**FR-7.4:** System SHALL render Semantic Similarity metric card:
- Label: "Similarity" (slate-400)
- Value: {percentage}% (blue-400, 1 decimal place)
- Description: "Cosine similarity" (slate-500)

**FR-7.5:** System SHALL render Word Overlap metric card:
- Label: "Word Overlap" (slate-400)
- Value: {percentage}% (emerald-400, 1 decimal place)
- Description: "Shared tokens" (slate-500)

**FR-7.6:** System SHALL use consistent card styling: bg-slate-800/50 rounded-lg p-4
**FR-7.7:** System SHALL provide tooltips on hover explaining each metric (future enhancement)

---

## 5. Non-Functional Requirements

### Performance Requirements
- **Generation Speed:** System SHALL complete 100-token generation (both unsteered and steered) within 3 seconds on Jetson Orin Nano
- **Metrics Calculation Time:** Comparison metrics SHALL be calculated within 500ms
- **UI Responsiveness:** Coefficient slider SHALL update value display within 100ms of input
- **Copy Latency:** Clipboard copy SHALL complete within 200ms

### Scalability Requirements
- **Multiple Features:** System SHALL support steering with up to 10 simultaneous features without performance degradation
- **Long Prompts:** System SHALL handle prompts up to 1024 tokens
- **Long Generations:** System SHALL support max_tokens up to 2048 (though edge device may be slow)

### Reliability Requirements
- **Generation Consistency:** System SHALL produce deterministic outputs when using fixed random seeds
- **Error Recovery:** System SHALL handle intervention failures gracefully (fallback to unsteered generation)
- **Hook Cleanup:** System SHALL ensure forward hooks are removed even if generation fails

### Usability Requirements
- **Immediate Feedback:** Slider changes SHALL show immediate visual feedback
- **Clear Comparison:** Side-by-side outputs SHALL clearly distinguish steered vs unsteered
- **Interpretable Metrics:** Metric descriptions SHALL make values understandable to non-experts

---

## 6. Technical Requirements

### Architecture Constraints (from ADR)
- **Backend Framework:** FastAPI with async endpoints for generation
- **ML Framework:** PyTorch 2.0+ with forward hooks for intervention
- **Frontend:** React 18+ with real-time state updates during generation

### Steering Algorithm Implementation

#### Multiplicative Steering (Recommended)
```python
def apply_steering(feature_activations, steering_config):
    """Apply steering coefficients to feature activations."""
    steered = feature_activations.clone()
    for feature_id, coefficient in steering_config.items():
        # Multiplicative: positive coeff amplifies, negative suppresses
        steered[:, feature_id] *= (1 + coefficient)
    return steered
```

#### Additive Steering (Alternative)
```python
def apply_steering(feature_activations, steering_config):
    """Apply steering coefficients to feature activations (additive)."""
    steered = feature_activations.clone()
    for feature_id, coefficient in steering_config.items():
        # Additive: directly add/subtract coefficient
        steered[:, feature_id] += coefficient
    return steered
```

**Recommendation:** Use multiplicative steering as it scales with original activation magnitude.

---

### API Specifications

#### POST /api/steering/generate
**Purpose:** Execute steered generation experiment with side-by-side comparison
**Authentication:** Required
**Request Body:**
```json
{
  "model_id": "m_gpt2_medium_456",
  "prompt": "The cat sat on the mat",
  "features": [
    {"feature_id": 42, "coefficient": 2.0},
    {"feature_id": 137, "coefficient": -1.5}
  ],
  "intervention_layer": 12,
  "temperature": 0.7,
  "max_tokens": 100,
  "seed": 42  // optional, for deterministic comparison
}
```

**Response:** 200 OK
```json
{
  "unsteered_output": "The cat sat on the mat, looking peaceful and content in the afternoon sun.",
  "steered_output": "The cat sat on the mat joyfully, radiating happiness and pure delight in the warm afternoon sun!",
  "metrics": {
    "kl_divergence": 0.0234,
    "perplexity_delta": -2.3,
    "semantic_similarity": 0.87,
    "word_overlap": 0.65
  },
  "generation_time_ms": 2340
}
```

**Error Responses:**
- 400 Bad Request: Invalid coefficients, intervention_layer exceeds model layers
- 404 Not Found: Model or features not found
- 408 Request Timeout: Generation exceeded 30 second timeout
- 507 Insufficient Storage: OOM error during generation

---

#### GET /api/features/search
**Purpose:** Search features by label for steering selection
**Query Parameters:**
- `q` (string): Search query
- `training_id` (string): Filter to features from specific training
- `limit` (int, default 20): Max results

**Response:** 200 OK
```json
{
  "features": [
    {
      "id": 42,
      "neuron_index": 42,
      "label": "Sentiment Positive",
      "training_id": "tr_abc123",
      "activation_frequency": 0.23,
      "interpretability_score": 0.94
    }
  ]
}
```

---

### Database Schema

#### steering_presets Table (Future Enhancement)
```sql
CREATE TABLE steering_presets (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,

    -- Configuration
    feature_configs JSONB NOT NULL,  -- [{"feature_id": 42, "coefficient": 2.0}, ...]
    intervention_layer INTEGER,
    temperature FLOAT,
    max_tokens INTEGER,

    -- User association
    user_id VARCHAR(255),  -- Future: multi-user support

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_steering_presets_user ON steering_presets(user_id);
```

---

## 7. UI/UX Specifications

### Steering Panel Layout

**Reference:** Mock-embedded-interp-ui.tsx SteeringPanel (lines 3512-3952)

**Structure:**

1. **Header:**
   - Title: "Model Steering" (text-2xl font-semibold)

2. **Two-Column Grid Layout:** (grid grid-cols-2 gap-6)

   **Left Column: Feature Selection Panel** (bg-slate-900/50 border border-slate-800 rounded-lg p-6)
   - **Title:** "Select Features to Steer" (text-lg font-semibold)
   - **Feature Search:**
     - Search icon (left side, slate-400)
     - Input: "Search features to add..." placeholder
     - Dropdown results: bg-slate-900 border border-slate-700, max-h-48 overflow-y-auto
     - Each result: Feature ID (mono, slate-400) + Label + "Add" button (emerald-400)

   - **Selected Features List:**
     - Header: "Selected Features ({count})" (text-sm font-medium slate-300)
     - Empty state: "No features selected. Search and add features above." (centered, py-8, slate-400)
     - **Feature Cards:** (bg-slate-800/30 rounded-lg p-4)
       - Header row: Feature ID + Label, Remove button (X icon, red-400)
       - Coefficient slider: Range -5 to +5, step 0.1, accent-emerald-500
       - Coefficient value: Displayed as {value.toFixed(2)} (emerald-400, mono font)
       - Slider labels: "-5.0 (suppress)" | "0.0" | "+5.0 (amplify)"
       - Quick preset buttons: "Suppress" (-2.0) | "Reset" (0.0) | "Amplify" (+2.0)

   **Right Column: Generation Controls Panel** (bg-slate-900/50 border border-slate-800 rounded-lg p-6)
   - **Title:** "Generation Configuration" (text-lg font-semibold)
   - **Model Selection:** Dropdown (filtered to ready models)
   - **Prompt Input:** Textarea (4 rows, resizable)
   - **Intervention Layer:** Slider (0-24, displays current value)
   - **Generation Parameters:** (grid-cols-2)
     - Temperature: Number input (0-2, step 0.1)
     - Max Tokens: Number input (1-2048)
   - **Generate Button:** Full width, bg-emerald-600, Zap icon, "Generate Comparison" text
     - Disabled state: bg-slate-700 when prompt empty or no features selected
     - Loading state: Loader spinner, "Generating..." text

3. **Comparative Output Display:** (shown after generation, space-y-6)

   **Two-Column Grid:**
   - **Unsteered (Baseline):**
     - Header: "Unsteered (Baseline)" with slate-400 dot indicator
     - Copy button (Copy icon)
     - Output box: bg-slate-900/50 border border-slate-800 rounded-lg p-4 min-h-200px
     - Text: slate-300, whitespace-pre-wrap, text-sm

   - **Steered:**
     - Header: "Steered" with emerald-400 dot indicator
     - Copy button (Copy icon)
     - Output box: bg-slate-900/50 border border-emerald-800/30 rounded-lg p-4 min-h-200px
     - Text: slate-300, whitespace-pre-wrap, text-sm

   **Comparison Metrics Panel:** (bg-slate-900/50 border border-slate-800 rounded-lg p-6)
   - Title: "Comparison Metrics" (text-lg font-semibold)
   - **4-Column Grid:**
     - KL Divergence (purple-400)
     - Perplexity Δ (red/emerald based on sign)
     - Similarity (blue-400)
     - Word Overlap (emerald-400)

---

## 8. Testing Requirements

### Unit Tests
- Test steering coefficient application (multiplicative and additive methods)
- Test forward hook registration and cleanup
- Test metric calculations (KL divergence, perplexity delta, semantic similarity, word overlap)
- Test coefficient slider range and step values
- Test feature search filtering

### Integration Tests
- Test end-to-end steering: select features → configure → generate → display
- Test side-by-side generation (unsteered and steered execute correctly)
- Test multiple feature interventions (coefficients combine correctly)
- Test intervention at different layers (early, middle, late)
- Test OOM handling with large generation parameters

### Edge Case Tests
- Test steering with all features at coefficient 0.0 (should match unsteered)
- Test steering with extreme coefficients (-5.0, +5.0)
- Test empty prompt handling
- Test generation timeout (30 second limit)
- Test model without sufficient layers for intervention_layer value

### Performance Tests
- Benchmark generation speed for 100 tokens on Jetson Orin Nano
- Measure overhead of steering vs unsteered generation
- Test responsiveness of coefficient slider UI updates

---

## 9. Dependencies

### Upstream Dependencies (Must be complete before implementation)
- **Feature Discovery (004_FPRD):** Requires extracted features with interpretable labels
- **Model Management (002_FPRD):** Requires models with activation extraction capability
- **SAE Training (003_FPRD):** Requires trained SAE models for feature-to-activation mapping

### Downstream Dependents
- None (terminal feature)

### External Dependencies
- **PyTorch:** For forward hooks and activation manipulation
- **Sentence Transformers:** For semantic similarity calculation (optional, can use model embeddings)
- **NumPy:** For metric calculations

---

## 10. Risks & Mitigations

### Technical Risks

**Risk 1: Steering Has No Observable Effect**
- **Likelihood:** Medium
- **Impact:** High (feature unusable if steering doesn't work)
- **Mitigation:**
  - Validate steering algorithm with synthetic tests (known features)
  - Start with strong coefficients (±2.0) to ensure visible effects
  - Test on features with high interpretability scores
  - Provide diagnostic logging (pre/post activation values)

**Risk 2: Steering Causes Model Instability**
- **Likelihood:** Medium
- **Impact:** High (generation fails or produces garbage)
- **Mitigation:**
  - Clip steered activations to prevent extreme values
  - Implement fallback to unsteered generation on errors
  - Limit coefficient range to tested values (-5 to +5)
  - Monitor perplexity delta as stability indicator

**Risk 3: Generation Too Slow on Edge Device**
- **Likelihood:** High
- **Impact:** Medium (poor UX if generation takes >10 seconds)
- **Mitigation:**
  - Set realistic max_tokens defaults (100 instead of 2048)
  - Show progress indicator during generation
  - Implement generation timeout (30 seconds)
  - Consider offloading heavy computation to cloud (post-MVP)

---

### Usability Risks

**Risk 4: Users Don't Understand Coefficients**
- **Likelihood:** Medium
- **Impact:** Medium (users configure ineffective interventions)
- **Mitigation:**
  - Provide clear slider labels (-5.0 suppress, +5.0 amplify)
  - Include quick preset buttons (Suppress, Reset, Amplify)
  - Show tooltips explaining coefficient semantics
  - Provide example steering configurations in documentation

**Risk 5: Comparison Metrics Are Not Intuitive**
- **Likelihood:** High
- **Impact:** Low (users ignore metrics but can still use feature)
- **Mitigation:**
  - Include plain-language descriptions for each metric
  - Provide tooltips with interpretation guides
  - Consider removing less useful metrics (e.g., word overlap may be redundant)
  - Add visual indicators (colors, icons) for metric ranges

---

## 11. Future Enhancements

### Post-MVP Features (Not included in initial implementation)

1. **Multi-Layer Steering**
   - Apply steering at multiple layers simultaneously
   - Visualize cascade effects through layers
   - Priority: P2, Timeline: Q3 2026

2. **Steering Strength Visualization**
   - Show per-token activation changes in real-time
   - Heatmap of steered vs unsteered activations
   - Priority: P2, Timeline: Q3 2026

3. **Automated Coefficient Tuning**
   - Suggest optimal coefficients based on desired output characteristics
   - Use gradient-based optimization to find steering configuration
   - Priority: P2, Timeline: Q4 2026

4. **Steering Templates Library**
   - Pre-configured steering presets for common use cases (e.g., "Make more positive", "Remove toxicity")
   - Community-shared steering configurations
   - Priority: P2, Timeline: Q2 2026

5. **Batch Steering Experiments**
   - Run multiple steering configurations on same prompt
   - Compare results in tabular format
   - Priority: P2, Timeline: Q3 2026

---

## 12. Open Questions

### Questions for Stakeholders

1. **Steering Algorithm:** Should we use multiplicative or additive coefficient application?
   - **Recommendation:** Multiplicative (scales with activation magnitude), provide both options in settings

2. **Random Seed Control:** Should we use same random seed for both generations to ensure fair comparison?
   - **Recommendation:** Yes for research, but document trade-off (reduces output diversity)

3. **Intervention Layer Default:** What layer should be default for intervention (early/middle/late)?
   - **Recommendation:** Middle layer (layer 12 for GPT-2, scales with model depth)

4. **Metrics to Display:** Are all 4 metrics useful, or should we focus on fewer?
   - **Recommendation:** Keep all 4 initially, remove based on user feedback

---

## 13. Success Metrics

### Feature Adoption Metrics
- **Steering Usage Rate:** Percentage of users who create steering experiments (target: >60%)
- **Repeat Usage:** Average number of steering experiments per user (target: >5)
- **Feature Utilization:** Average number of features used per experiment (target: 2-4)

### Quality Metrics
- **Observable Effect Rate:** Percentage of steering experiments showing KL divergence > 0.01 (target: >80%)
- **Semantic Preservation:** Average semantic similarity for steering experiments (target: 0.7-0.9)
- **User Satisfaction:** Post-feature survey rating (target: >4/5 stars)

### Performance Metrics
- **Generation Speed:** Average time to complete 100-token generation on Jetson Orin Nano (target: <3 seconds)
- **Metrics Calculation Time:** Average time to compute comparison metrics (target: <500ms)

---

## 14. Appendices

### A. Glossary

- **Steering:** Modifying model behavior by intervening on feature activations during generation
- **Intervention Layer:** The transformer layer where feature activations are modified
- **Steering Coefficient:** Scalar value controlling strength and direction of intervention (-5 to +5)
- **Multiplicative Steering:** Scaling feature activation: `new_activation = old_activation * (1 + coefficient)`
- **Additive Steering:** Adding to feature activation: `new_activation = old_activation + coefficient`
- **KL Divergence:** Measure of distribution difference between steered and unsteered token probabilities
- **Perplexity Delta:** Change in model confidence (perplexity) due to steering
- **Semantic Similarity:** Cosine similarity between sentence embeddings of outputs

### B. References

**Research Papers:**
- Turner et al. (2023): "Activation Addition: Steering Language Models Without Optimization"
- Li et al. (2023): "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"

**Technical Documentation:**
- PyTorch Forward Hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
- Sentence Transformers: https://www.sbert.net/

**UI/UX Reference:**
- Mock-embedded-interp-ui.tsx (PRIMARY REFERENCE)

---

**Document End**
**Total Sections:** 14
**Total Functional Requirements:** 71 (across FR-1 through FR-7)
**Estimated Implementation Time:** 3-4 weeks (2 developers)
**Review Status:** Pending stakeholder review
