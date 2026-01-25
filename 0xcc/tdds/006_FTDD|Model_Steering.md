# Technical Design Document: Model Steering

**Document ID:** 006_FTDD|Model_Steering
**Version:** 1.1 (Combined Mode Enhancement)
**Last Updated:** 2026-01-24
**Status:** Partially Implemented (Combined Mode Planned)
**Related PRD:** [006_FPRD|Model_Steering](../prds/006_FPRD|Model_Steering.md)

---

## 1. System Architecture

### 1.1 Steering Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                     Steering Pipeline                            │
│                                                                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────────┐ │
│  │   Prompt   │───→│   Model    │───→│    SAE Intervention    │ │
│  │            │    │  Forward   │    │    (steering hook)     │ │
│  └────────────┘    └────────────┘    └────────────────────────┘ │
│                          │                      │                │
│                          ▼                      ▼                │
│                 ┌────────────────┐    ┌────────────────────────┐│
│                 │  Activations   │───→│  Modified Activations  ││
│                 │  (original)    │    │  (steered)             ││
│                 └────────────────┘    └────────────────────────┘│
│                                                  │               │
│                                                  ▼               │
│                                       ┌────────────────────────┐│
│                                       │   Generation Output    ││
│                                       │   (steered text)       ││
│                                       └────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend                                      │
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────────┐ │
│  │ SteeringPanel │  │ FeatureBrowser  │  │ComparisonResults  │ │
│  └───────┬───────┘  └────────┬────────┘  └─────────┬─────────┘ │
│          │                   │                      │           │
│  ┌───────┴───────────────────┴──────────────────────┴─────────┐ │
│  │                    steeringStore                            │ │
│  └────────────────────────────┬────────────────────────────────┘ │
└───────────────────────────────┼─────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────┐
│                    Backend                                       │
│  ┌────────────────────────────┴────────────────────────────────┐│
│  │                   SteeringService                           ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      ││
│  │  │generate_with_│  │compare_      │  │strength_     │      ││
│  │  │steering()    │  │steering()    │  │sweep()       │      ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘      ││
│  └────────────────────────────┬────────────────────────────────┘│
│                               │                                 │
│  ┌────────────────────────────┴────────────────────────────────┐│
│  │                    forward_hooks.py                         ││
│  │  ┌──────────────────────────────────────────────────────┐  ││
│  │  │              SteeringHook                             │  ││
│  │  │  - encode through SAE                                 │  ││
│  │  │  - modify feature activations                         │  ││
│  │  │  - decode back to residual stream                     │  ││
│  │  └──────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Steering Hook Implementation

### 2.1 Core Steering Hook
```python
class SteeringHook:
    def __init__(
        self,
        sae: SparseAutoencoder,
        feature_configs: List[FeatureSteeringConfig],
        calibration_factor: float = 1.0
    ):
        self.sae = sae
        self.feature_configs = feature_configs
        self.calibration_factor = calibration_factor

    def __call__(self, module, input, output):
        """Forward hook that applies steering."""
        # Output shape: [batch, seq, hidden_dim]
        residual = output[0] if isinstance(output, tuple) else output

        # Encode through SAE
        z = self.sae.encode(residual)  # [batch, seq, d_sae]

        # Apply steering modifications
        for config in self.feature_configs:
            feature_idx = config.feature_index
            strength = config.strength * self.calibration_factor

            if config.steering_type == 'activation':
                # Add to feature activation
                z[:, :, feature_idx] += strength
            elif config.steering_type == 'suppression':
                # Scale feature activation toward zero
                z[:, :, feature_idx] *= (1.0 - abs(strength))

        # Decode back to residual stream
        modified_residual = self.sae.decode(z)

        if isinstance(output, tuple):
            return (modified_residual,) + output[1:]
        return modified_residual

    def register(self, model, layer: int):
        """Register hook on specified layer."""
        target_module = model.model.layers[layer]
        return target_module.register_forward_hook(self)
```

### 2.2 Calibration
```python
def compute_calibration_factor(
    sae: SparseAutoencoder,
    feature_idx: int,
    dataset_activations: Tensor
) -> float:
    """
    Compute Neuronpedia-compatible calibration factor.

    The calibration factor normalizes steering strength based on
    the feature's activation distribution in the training data.
    """
    # Get feature activations
    z = sae.encode(dataset_activations)
    feature_activations = z[:, :, feature_idx]

    # Use standard deviation as calibration base
    active_mask = feature_activations > 0
    if active_mask.sum() > 0:
        std = feature_activations[active_mask].std()
        return std.item()
    return 1.0
```

### 2.3 Combined Multi-Feature Steering Hook (Planned)

The combined mode applies all selected features simultaneously in a single forward pass:

```python
class CombinedSteeringHook:
    """
    Hook that accumulates steering vectors from multiple features
    and applies them together in a single generation pass.
    """
    def __init__(
        self,
        sae: SparseAutoencoder,
        feature_configs: List[FeatureSteeringConfig],
        calibration_factors: Dict[int, float]
    ):
        self.sae = sae
        self.feature_configs = feature_configs
        self.calibration_factors = calibration_factors
        # Pre-compute combined steering direction
        self._precompute_steering_vector()

    def _precompute_steering_vector(self):
        """
        Pre-compute the combined steering direction from decoder weights.
        This avoids encoding/decoding overhead during generation.
        """
        # W_dec shape: [d_sae, d_in] or [d_in, d_sae] depending on SAE
        W_dec = self.sae.W_dec  # Get decoder weights

        self.combined_steering = torch.zeros(W_dec.shape[-1], device=W_dec.device)

        for config in self.feature_configs:
            feature_idx = config.feature_index
            strength = config.strength
            calibration = self.calibration_factors.get(feature_idx, 1.0)

            # Get feature's steering direction (column of decoder)
            steering_direction = W_dec[feature_idx, :]  # or [:, feature_idx]

            # Accumulate scaled steering
            self.combined_steering += strength * calibration * steering_direction

    def __call__(self, module, input, output):
        """
        Apply combined steering to residual stream.
        No SAE encode/decode - direct residual modification.
        """
        hidden_states = output[0] if isinstance(output, tuple) else output

        # Add combined steering to all positions
        # Shape: [batch, seq, hidden] + [hidden]
        modified = hidden_states + self.combined_steering.unsqueeze(0).unsqueeze(0)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
```

**Key Design Decisions:**

1. **Pre-computed Steering Vector**: Instead of encoding through SAE each token,
   we pre-compute the combined steering direction from decoder weights.

2. **Direct Residual Modification**: Steering is applied directly to the residual
   stream without SAE round-trip, which is faster and more stable.

3. **Accumulated Strength**: Multiple features' steering vectors are summed together,
   allowing synergistic or counteracting effects.

**Comparison: Isolated vs Combined Mode**

| Aspect | Isolated Mode | Combined Mode |
|--------|---------------|---------------|
| Outputs | N outputs for N features | 1 output with all features |
| Purpose | Analyze individual impact | Test feature interactions |
| Speed | N × generation time | 1 × generation time |
| Use Case | Feature hypothesis testing | Behavioral composition |

---

## 3. API Design

### 3.1 Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/steering/generate` | Generate with steering (single feature) |
| POST | `/steering/compare` | Compare steered vs baseline |
| POST | `/steering/sweep` | Multi-strength sweep |
| POST | `/steering/calibrate` | Get calibration factors |
| POST | `/steering/combined` | Combined multi-feature generation (Planned) |
| GET | `/prompt-templates` | List prompt templates |
| POST | `/prompt-templates` | Create template |

### 3.2 Request Schemas
```python
class FeatureSteeringConfig(BaseModel):
    feature_id: UUID
    feature_index: int
    strength: float  # -10 to +10
    steering_type: Literal['activation', 'suppression'] = 'activation'

class SteeringGenerateRequest(BaseModel):
    sae_id: UUID
    model_id: UUID
    prompt: str
    features: List[FeatureSteeringConfig]
    generation_config: GenerationConfig

class GenerationConfig(BaseModel):
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0

class SteeringCompareRequest(BaseModel):
    sae_id: UUID
    model_id: UUID
    prompt: str
    features: List[FeatureSteeringConfig]
    generation_config: GenerationConfig

class StrengthSweepRequest(BaseModel):
    sae_id: UUID
    model_id: UUID
    prompt: str
    feature: FeatureSteeringConfig
    strengths: List[float]  # e.g., [-5, -2, 0, 2, 5]
    generation_config: GenerationConfig

# Planned: Combined multi-feature generation
class CombinedSteeringRequest(BaseModel):
    sae_id: UUID
    model_id: UUID
    prompt: str
    features: List[FeatureSteeringConfig]  # Multiple features applied together
    generation_config: GenerationConfig
    include_baseline: bool = True  # Generate baseline for comparison
```

### 3.3 Response Schemas
```python
class SteeringGenerateResponse(BaseModel):
    output: str
    tokens_generated: int
    steering_applied: List[FeatureSteeringConfig]

class SteeringCompareResponse(BaseModel):
    baseline_output: str
    steered_output: str
    differences: List[DifferenceSpan]

class StrengthSweepResponse(BaseModel):
    results: List[SweepResult]

class SweepResult(BaseModel):
    strength: float
    output: str

# Planned: Combined generation response
class CombinedSteeringResponse(BaseModel):
    combined_output: str  # Text generated with all features applied
    features_applied: List[FeatureSteeringConfig]
    baseline_output: Optional[str] = None  # If include_baseline was True
    tokens_generated: int
```

---

## 4. Service Layer

### 4.1 SteeringService
```python
class SteeringService:
    def __init__(self):
        self.model_cache = {}
        self.sae_cache = {}

    async def generate_with_steering(
        self,
        request: SteeringGenerateRequest
    ) -> SteeringGenerateResponse:
        """Generate text with steering applied."""

        # Load model and SAE
        model = await self._get_model(request.model_id)
        sae = await self._get_sae(request.sae_id)
        tokenizer = await self._get_tokenizer(request.model_id)

        # Create steering hook
        hook = SteeringHook(sae, request.features)
        layer = self._get_sae_layer(request.sae_id)
        handle = hook.register(model, layer)

        try:
            # Tokenize prompt
            inputs = tokenizer(request.prompt, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate with steering
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.generation_config.max_new_tokens,
                    temperature=request.generation_config.temperature,
                    top_p=request.generation_config.top_p,
                    top_k=request.generation_config.top_k,
                    repetition_penalty=request.generation_config.repetition_penalty,
                    do_sample=True
                )

            # Decode output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from output
            output_text = output_text[len(request.prompt):]

            return SteeringGenerateResponse(
                output=output_text,
                tokens_generated=len(outputs[0]) - len(inputs['input_ids'][0]),
                steering_applied=request.features
            )

        finally:
            handle.remove()

    async def compare_steering(
        self,
        request: SteeringCompareRequest
    ) -> SteeringCompareResponse:
        """Generate baseline and steered outputs for comparison."""

        # Generate baseline (no steering)
        baseline_request = SteeringGenerateRequest(
            sae_id=request.sae_id,
            model_id=request.model_id,
            prompt=request.prompt,
            features=[],  # Empty = no steering
            generation_config=request.generation_config
        )
        baseline = await self.generate_with_steering(baseline_request)

        # Generate steered
        steered_request = SteeringGenerateRequest(
            sae_id=request.sae_id,
            model_id=request.model_id,
            prompt=request.prompt,
            features=request.features,
            generation_config=request.generation_config
        )
        steered = await self.generate_with_steering(steered_request)

        # Compute differences
        differences = self._compute_differences(baseline.output, steered.output)

        return SteeringCompareResponse(
            baseline_output=baseline.output,
            steered_output=steered.output,
            differences=differences
        )

    async def strength_sweep(
        self,
        request: StrengthSweepRequest
    ) -> StrengthSweepResponse:
        """Test multiple steering strengths."""
        results = []

        for strength in request.strengths:
            modified_feature = request.feature.copy()
            modified_feature.strength = strength

            gen_request = SteeringGenerateRequest(
                sae_id=request.sae_id,
                model_id=request.model_id,
                prompt=request.prompt,
                features=[modified_feature],
                generation_config=request.generation_config
            )
            response = await self.generate_with_steering(gen_request)

            results.append(SweepResult(
                strength=strength,
                output=response.output
            ))

        return StrengthSweepResponse(results=results)

    # Planned: Combined multi-feature generation
    async def generate_combined(
        self,
        request: CombinedSteeringRequest
    ) -> CombinedSteeringResponse:
        """
        Generate with multiple features applied together in a single pass.

        Unlike compare_steering which generates separate outputs per feature,
        this applies ALL features simultaneously for a combined effect.
        """
        # Load model and SAE
        model = await self._get_model(request.model_id)
        sae = await self._get_sae(request.sae_id)
        tokenizer = await self._get_tokenizer(request.model_id)

        # Get calibration factors for each feature
        calibration_factors = {
            config.feature_index: self._get_calibration(config.feature_index)
            for config in request.features
        }

        # Create combined steering hook
        hook = CombinedSteeringHook(sae, request.features, calibration_factors)
        layer = self._get_sae_layer(request.sae_id)
        handle = hook.register(model, layer)

        try:
            # Generate baseline if requested
            baseline_output = None
            if request.include_baseline:
                baseline_output = await self._generate_raw(
                    request.prompt, request.generation_config
                )

            # Tokenize prompt
            inputs = tokenizer(request.prompt, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate with combined steering
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.generation_config.max_new_tokens,
                    temperature=request.generation_config.temperature,
                    do_sample=True
                )

            # Decode output
            combined_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            combined_output = combined_output[len(request.prompt):]

            return CombinedSteeringResponse(
                combined_output=combined_output,
                features_applied=request.features,
                baseline_output=baseline_output,
                tokens_generated=len(outputs[0]) - len(inputs['input_ids'][0])
            )

        finally:
            handle.remove()
```

---

## 5. Frontend Components

### 5.1 SteeringPanel Layout
```typescript
<SteeringPanel>
  <SAEModelSelector
    sae={selectedSAE}
    onSAEChange={setSelectedSAE}
  />

  <SelectedFeatures>
    {selectedFeatures.map(f => (
      <SelectedFeatureCard
        key={f.id}
        feature={f}
        strength={strengths[f.id]}
        onStrengthChange={(s) => updateStrength(f.id, s)}
        onRemove={() => removeFeature(f.id)}
      />
    ))}
    <AddFeatureButton onClick={openFeatureBrowser} />
  </SelectedFeatures>

  <PromptInput
    value={prompt}
    onChange={setPrompt}
    onLoadTemplate={openTemplateModal}
  />

  <SteeringOptions>
    <Checkbox checked={comparisonMode} onChange={setComparisonMode}>
      Comparison Mode
    </Checkbox>
    {/* Planned: Combined mode toggle */}
    <Checkbox
      checked={combinedMode}
      onChange={setCombinedMode}
      disabled={selectedFeatures.length < 2}
    >
      Combined Mode (apply all features together)
    </Checkbox>
    <SweepConfig
      enabled={sweepEnabled}
      strengths={sweepStrengths}
      onChange={setSweepStrengths}
    />
  </SteeringOptions>

  <GenerateButton onClick={generate} />

  {results && <ComparisonResults results={results} />}
</SteeringPanel>
```

### 5.2 StrengthSlider Component
```typescript
interface StrengthSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;  // default -10
  max?: number;  // default +10
  step?: number; // default 0.5
}

// Visual representation:
// Suppression ←───────────[●]───────────→ Activation
//     -10                  0                  +10
```

### 5.3 ComparisonResults Component
```typescript
interface ComparisonResultsProps {
  baseline: string;
  steered: string;
  differences?: DifferenceSpan[];
}

// Side-by-side display:
// ┌──────────────────┬───────────────────┐
// │    Baseline      │     Steered       │
// ├──────────────────┼───────────────────┤
// │ Normal text      │ Modified text     │
// │ here...          │ [highlighted]...  │
// └──────────────────┴───────────────────┘
```

---

## 6. Prompt Templates

### 6.1 Database Schema
```sql
CREATE TABLE prompt_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    variables JSONB,              -- [{name, description, default}]
    template_type VARCHAR(50) DEFAULT 'steering',
    is_favorite BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 6.2 Template Structure
```json
{
  "name": "Character Description",
  "content": "Write a short story about {character} who {action}.",
  "variables": [
    {"name": "character", "description": "Main character", "default": "a wizard"},
    {"name": "action", "description": "What happens", "default": "discovers a secret"}
  ]
}
```

---

## 7. State Management

### 7.1 steeringStore
```typescript
interface SteeringState {
  selectedSAE: ExternalSAE | null;
  selectedModel: Model | null;
  selectedFeatures: SelectedFeature[];
  prompt: string;
  comparisonMode: boolean;
  combinedMode: boolean;  // Planned: apply all features together
  sweepEnabled: boolean;
  sweepStrengths: number[];
  generationConfig: GenerationConfig;
  results: SteeringResults | null;
  combinedResults: CombinedSteeringResult | null;  // Planned
  isGenerating: boolean;

  // Actions
  selectSAE: (sae: ExternalSAE) => void;
  addFeature: (feature: Feature, strength: number) => void;
  updateStrength: (featureId: string, strength: number) => void;
  removeFeature: (featureId: string) => void;
  setPrompt: (prompt: string) => void;
  setCombinedMode: (enabled: boolean) => void;  // Planned
  generate: () => Promise<void>;
  generateCombined: () => Promise<void>;  // Planned
  exportResults: () => void;
}

// Planned: Combined results type
interface CombinedSteeringResult {
  combinedOutput: string;
  featuresApplied: SelectedFeature[];
  baselineOutput?: string;
}
```

---

## 8. Error Handling

| Error | Cause | Handling |
|-------|-------|----------|
| Model not loaded | Missing model | Load model first |
| SAE mismatch | SAE doesn't match model | Validate compatibility |
| OOM during generation | Large model/batch | Reduce max_tokens |
| Invalid strength | Out of range | Clamp to [-10, 10] |

---

*Related: [PRD](../prds/006_FPRD|Model_Steering.md) | [TID](../tids/006_FTID|Model_Steering.md) | [FTASKS](../tasks/006_FTASKS|Model_Steering.md)*
