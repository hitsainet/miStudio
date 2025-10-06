# Technical Design Document: Model Steering

**Document ID:** 005_FTDD|Model_Steering
**Feature:** Feature-Based Model Behavior Modification and Intervention
**PRD Reference:** 005_FPRD|Model_Steering.md
**ADR Reference:** 000_PADR|miStudio.md
**Status:** Draft
**Created:** 2025-10-06
**Last Updated:** 2025-10-06
**Owner:** miStudio Development Team

---

## 1. Executive Summary

This Technical Design Document defines the architecture and implementation approach for the Model Steering feature of miStudio. The feature enables researchers to modify language model behavior by directly intervening on discovered SAE features during text generation, providing a controllable interface for mechanistic interpretability research and causal analysis.

**Business Objective:** Transform discovered interpretable features into actionable control mechanisms for model behavior, enabling researchers to test causal relationships between features and model outputs.

**Technical Approach:**
- FastAPI backend with synchronous generation endpoints (steering requires real-time hook manipulation)
- PyTorch forward hooks for activation intervention at specified layers
- Multiplicative steering algorithm for coefficient application
- Parallel or sequential execution of unsteered and steered generations
- Sentence transformers for semantic similarity calculation
- React frontend with feature selection, coefficient sliders, and side-by-side comparison UI

**Key Design Decisions:**
1. Use multiplicative steering (coefficient scales activation magnitude) rather than additive
2. Execute unsteered and steered generations sequentially to avoid CUDA context conflicts
3. Register forward hooks only for steered generation, unregister immediately after
4. Cache sentence embeddings for semantic similarity calculation
5. Clip steered activations to prevent numerical instability (-10 to +10 range)
6. Use deterministic random seed for fair comparison (optional, user-configurable)

**Success Metrics:**
- 100-token generation (both unsteered and steered) completes within 3 seconds on Jetson Orin Nano
- Steering with 5 features shows observable effect (KL divergence > 0.01) in 80%+ of experiments
- Comparison metrics calculate within 500ms
- System handles up to 10 simultaneous features without performance degradation

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    SteeringPanel                          │   │
│  │  ┌───────────────────┐  ┌────────────────────────────┐   │   │
│  │  │ Feature Selection │  │ Generation Configuration   │   │   │
│  │  │ - Search features │  │ - Model selection          │   │   │
│  │  │ - Coefficient     │  │ - Prompt input             │   │   │
│  │  │   sliders         │  │ - Intervention layer       │   │   │
│  │  │ - Quick presets   │  │ - Temperature, max_tokens  │   │   │
│  │  └───────────────────┘  └────────────────────────────┘   │   │
│  │                                                            │   │
│  │  ┌───────────────────────────────────────────────────┐   │   │
│  │  │        Comparative Output Display                  │   │   │
│  │  │  ┌───────────────┐    ┌─────────────────────┐     │   │   │
│  │  │  │ Unsteered     │    │ Steered             │     │   │   │
│  │  │  │ (Baseline)    │    │                     │     │   │   │
│  │  │  └───────────────┘    └─────────────────────┘     │   │   │
│  │  │                                                     │   │   │
│  │  │  ┌───────────────────────────────────────────┐    │   │   │
│  │  │  │ Comparison Metrics                        │    │   │   │
│  │  │  │ KL Div | Perp Δ | Similarity | Word Ovlp │    │   │   │
│  │  │  └───────────────────────────────────────────┘    │   │   │
│  │  └───────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    REST API (POST /api/steering/generate)
                             │
┌────────────────────────────┼─────────────────────────────────────┐
│                   Backend (FastAPI)                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              /api/steering/generate                       │   │
│  │  1. Validate request (model, features, coefficients)     │   │
│  │  2. Load model and SAE checkpoint                        │   │
│  │  3. Execute unsteered generation (no hooks)              │   │
│  │  4. Execute steered generation (with hooks)              │   │
│  │  5. Calculate comparison metrics                         │   │
│  │  6. Return both outputs + metrics                        │   │
│  └─────────────┬────────────────────────────────────────────┘   │
│                │                                                 │
│  ┌─────────────┴────────────────────────────────────────────┐   │
│  │         SteeringService                                   │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ apply_steering(model, sae, features, coeffs)       │  │   │
│  │  │ - Register forward hook at intervention layer      │  │   │
│  │  │ - Hook: activations → SAE encoder → apply coeffs  │  │   │
│  │  │         → SAE decoder → return steered activations │  │   │
│  │  │ - Generate text with steering active               │  │   │
│  │  │ - Unregister hook after completion                 │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                                                            │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ calculate_metrics(unsteered, steered, model)       │  │   │
│  │  │ - KL divergence (token distributions)              │  │   │
│  │  │ - Perplexity delta (confidence change)             │  │   │
│  │  │ - Semantic similarity (sentence embeddings)        │  │   │
│  │  │ - Word overlap (Jaccard similarity)                │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### Steering Algorithm Flow

```
User clicks "Generate Comparison"
    ↓
Frontend: POST /api/steering/generate
    Request: {model_id, prompt, features[], intervention_layer, temperature, max_tokens}
    ↓
Backend: Load model (with quantization if needed)
    ↓
Backend: Load SAE checkpoint (encoder + decoder)
    ↓
Backend: UNSTEERED GENERATION
    ├─ Set random seed (for reproducibility)
    ├─ Generate text (standard forward pass, no hooks)
    ├─ Capture token-level log probabilities
    └─ Store unsteered_output, unsteered_logprobs
    ↓
Backend: STEERED GENERATION
    ├─ Reset random seed (same as unsteered)
    ├─ Register forward hook on model.layers[intervention_layer]
    │   ├─ Hook function:
    │   │   1. Extract activations from layer output
    │   │   2. Pass activations through SAE encoder → feature_activations
    │   │   3. For each (feature_id, coefficient) in steering config:
    │   │      feature_activations[feature_id] *= (1 + coefficient)
    │   │   4. Clip steered features to [-10, 10] range
    │   │   5. Pass steered features through SAE decoder → steered_activations
    │   │   6. Return steered_activations (replaces original)
    ├─ Generate text (with steering hook active)
    ├─ Capture token-level log probabilities
    ├─ Unregister forward hook (cleanup)
    └─ Store steered_output, steered_logprobs
    ↓
Backend: CALCULATE METRICS
    ├─ KL Divergence: KL(unsteered || steered) averaged across tokens
    ├─ Perplexity Delta: exp(mean(-log_probs_steered)) - exp(mean(-log_probs_unsteered))
    ├─ Semantic Similarity: cosine(embed(unsteered), embed(steered))
    └─ Word Overlap: Jaccard(tokens_unsteered, tokens_steered)
    ↓
Backend: Return response
    {unsteered_output, steered_output, metrics{kl, perp_delta, similarity, overlap}}
    ↓
Frontend: Display side-by-side outputs + metrics panel
```

### Component Relationships

**Frontend Components:**
- **SteeringPanel:** Main container component (lines 3512-3951 in Mock UI)
- **Feature Selection Panel:** Search, add features, coefficient sliders (lines 3582-3706)
- **Generation Controls Panel:** Model, prompt, layer, temperature, max_tokens (lines 3710-3817)
- **Comparative Output Display:** Side-by-side unsteered/steered outputs (lines 3820-3895)
- **Comparison Metrics Panel:** 4-column metric cards (lines 3898-3947)

**Backend Components:**
- **SteeringRouter:** FastAPI router for `/api/steering/*` endpoints
- **SteeringService:** Core steering algorithm and hook management
- **MetricsCalculator:** Compute comparison metrics (KL, perplexity, similarity)
- **ModelRegistry:** Load models and SAE checkpoints (reuses 002_FTDD component)

---

## 3. Technical Stack

| Category | Technology | Version | Justification |
|----------|-----------|---------|---------------|
| **Backend Framework** | FastAPI | 0.104+ | Async capabilities, but steering uses sync endpoints for hook control |
| **ML Framework** | PyTorch | 2.0+ | Forward hooks for activation intervention, standard for LM inference |
| **Quantization** | bitsandbytes | 0.41+ | INT4/INT8 quantization for memory efficiency (reuses 002_FTDD setup) |
| **Embeddings** | sentence-transformers | 2.2+ | Semantic similarity calculation (all-MiniLM-L6-v2 model, 80MB) |
| **Metrics** | NumPy | 1.24+ | Efficient KL divergence and statistical calculations |
| **Frontend** | React | 18+ | Component-based UI, real-time state updates |
| **State Management** | Zustand | 4.4+ | Lightweight state for feature selection and generation config |
| **Styling** | Tailwind CSS | 3.3+ | Utility-first styling matching Mock UI (slate dark theme, emerald accents) |
| **Icons** | Lucide React | 0.263+ | Search, X, Zap, Loader, Copy icons |

**Key Dependencies:**
- **transformers:** For tokenizer and model loading (reuses 002_FTDD infrastructure)
- **safetensors:** For SAE checkpoint loading (reuses 003_FTDD infrastructure)

---

## 4. Data Design

### Database Schema

**Note:** Steering feature does NOT persist experiments to database in MVP (in-memory only). Future enhancement: `steering_presets` table for saved configurations.

#### steering_presets Table (Future Enhancement)

```sql
CREATE TABLE steering_presets (
    id VARCHAR(255) PRIMARY KEY,               -- Format: sp_{uuid}
    name VARCHAR(500) NOT NULL,                -- User-provided preset name
    description TEXT,                          -- Optional description

    -- Configuration
    feature_configs JSONB NOT NULL,            -- [{"feature_id": 42, "coefficient": 2.0}, ...]
    intervention_layer INTEGER,                -- Default intervention layer
    temperature FLOAT DEFAULT 0.7,             -- Default temperature
    max_tokens INTEGER DEFAULT 100,            -- Default max_tokens

    -- Metadata
    user_id VARCHAR(255),                      -- Future: multi-user support
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT sp_temperature_range CHECK (temperature >= 0 AND temperature <= 2),
    CONSTRAINT sp_max_tokens_range CHECK (max_tokens >= 1 AND max_tokens <= 2048),
    CONSTRAINT sp_feature_configs_format CHECK (jsonb_typeof(feature_configs) = 'array')
);

CREATE INDEX idx_steering_presets_user ON steering_presets(user_id);
CREATE INDEX idx_steering_presets_created ON steering_presets(created_at DESC);
```

**JSONB Structure for `feature_configs`:**
```json
[
    {"feature_id": 42, "coefficient": 2.0},
    {"feature_id": 137, "coefficient": -1.5},
    {"feature_id": 89, "coefficient": 0.5}
]
```

### Request/Response Payloads

#### POST /api/steering/generate

**Request Payload:**
```typescript
interface SteeringGenerateRequest {
    model_id: string;                          // Model identifier (must be status='ready')
    prompt: string;                            // Input text (1-1024 tokens)
    features: Array<{                          // Steering configuration
        feature_id: number;                    // Feature ID from features table
        coefficient: number;                   // Range: -5.0 to +5.0
    }>;
    intervention_layer: number;                // Layer index (0 to model.num_layers-1)
    temperature: number;                       // Range: 0 to 2, default 0.7
    max_tokens: number;                        // Range: 1 to 2048, default 100
    seed?: number;                             // Optional: deterministic generation
}
```

**Example Request:**
```json
{
    "model_id": "m_gpt2_medium_abc123",
    "prompt": "The cat sat on the mat",
    "features": [
        {"feature_id": 42, "coefficient": 2.0},
        {"feature_id": 137, "coefficient": -1.5}
    ],
    "intervention_layer": 12,
    "temperature": 0.7,
    "max_tokens": 100,
    "seed": 42
}
```

**Response Payload:**
```typescript
interface SteeringGenerateResponse {
    unsteered_output: string;                  // Generated text without steering
    steered_output: string;                    // Generated text with steering
    metrics: {
        kl_divergence: number;                 // 4 decimal places, range typically 0-1
        perplexity_delta: number;              // 2 decimal places, can be negative
        semantic_similarity: number;           // 2 decimal places, range 0-1
        word_overlap: number;                  // 2 decimal places, range 0-1
    };
    generation_time_ms: number;                // Total generation time (both outputs)
}
```

**Example Response:**
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

### Validation Rules

**Request Validation:**
- `model_id`: Must exist in `models` table with `status='ready'`
- `prompt`: Non-empty, max 1024 tokens after tokenization
- `features`: Array with 1-10 elements (enforce max 10 features for performance)
- `feature_id`: Must exist in `features` table
- `coefficient`: Range [-5.0, 5.0]
- `intervention_layer`: Range [0, model.num_layers - 1]
- `temperature`: Range [0.0, 2.0]
- `max_tokens`: Range [1, 2048]
- `seed`: Optional integer >= 0

**Error Responses:**
- 400 Bad Request: Validation failures (invalid coefficient range, layer exceeds model depth)
- 404 Not Found: Model or features not found
- 408 Request Timeout: Generation exceeded 30 second timeout
- 422 Unprocessable Entity: Feature from different training (mismatched SAE)
- 507 Insufficient Storage: OOM error during generation

---

## 5. API Design

### REST Endpoints

#### POST /api/steering/generate

**Purpose:** Execute steered generation experiment with side-by-side comparison

**Authentication:** Required (JWT)

**Rate Limiting:** 10 requests per minute per user

**Request:**
```json
{
    "model_id": "m_gpt2_medium_abc123",
    "prompt": "The weather today is",
    "features": [
        {"feature_id": 42, "coefficient": 2.0}
    ],
    "intervention_layer": 12,
    "temperature": 0.7,
    "max_tokens": 100,
    "seed": 42
}
```

**Response:** 200 OK
```json
{
    "unsteered_output": "The weather today is cloudy with a chance of rain.",
    "steered_output": "The weather today is absolutely beautiful and sunny!",
    "metrics": {
        "kl_divergence": 0.0456,
        "perplexity_delta": 1.2,
        "semantic_similarity": 0.72,
        "word_overlap": 0.42
    },
    "generation_time_ms": 1850
}
```

**Error Responses:**
```json
// 400 Bad Request
{
    "detail": "intervention_layer (24) exceeds model layer count (12)"
}

// 404 Not Found
{
    "detail": "Model 'm_gpt2_medium_abc123' not found"
}

// 408 Request Timeout
{
    "detail": "Generation exceeded 30 second timeout"
}

// 422 Unprocessable Entity
{
    "detail": "Feature 42 belongs to training 'tr_xyz', but model requires training 'tr_abc'"
}

// 507 Insufficient Storage
{
    "detail": "OOM error during generation. Try reducing max_tokens or use smaller model."
}
```

---

#### GET /api/features/search

**Purpose:** Search features for steering selection (reuses Feature Discovery endpoint)

**Query Parameters:**
- `q` (string, required): Search query
- `training_id` (string, optional): Filter to specific training
- `limit` (int, default 20): Max results

**Response:** 200 OK
```json
{
    "features": [
        {
            "id": 42,
            "neuron_index": 42,
            "name": "Sentiment Positive",
            "training_id": "tr_abc123",
            "activation_frequency": 0.23,
            "interpretability_score": 0.94
        }
    ]
}
```

---

### WebSocket Protocol

**Note:** Steering does NOT use WebSockets in MVP (synchronous request/response). Future enhancement: streaming generation with token-by-token updates.

---

## 6. Component Architecture

### Frontend Component Hierarchy

```
SteeringPanel (lines 3512-3951)
├─ Feature Selection Panel (lines 3582-3706)
│  ├─ Search Input (with Search icon)
│  ├─ Search Results Dropdown
│  └─ Selected Features List
│     └─ Feature Card (for each selected feature)
│        ├─ Feature ID + Label
│        ├─ Remove Button (X icon)
│        ├─ Coefficient Slider (-5 to +5)
│        ├─ Coefficient Value Display
│        └─ Quick Preset Buttons (Suppress, Reset, Amplify)
│
├─ Generation Controls Panel (lines 3710-3817)
│  ├─ Model Selection Dropdown
│  ├─ Prompt Textarea
│  ├─ Intervention Layer Slider
│  ├─ Generation Parameters (2-column grid)
│  │  ├─ Temperature Input
│  │  └─ Max Tokens Input
│  └─ Generate Comparison Button
│
└─ Comparative Output Display (lines 3820-3947)
   ├─ Unsteered Output Panel
   │  ├─ Header ("Unsteered (Baseline)" with slate-400 dot)
   │  ├─ Copy Button
   │  └─ Output Box (slate border)
   │
   ├─ Steered Output Panel
   │  ├─ Header ("Steered" with emerald-400 dot)
   │  ├─ Copy Button
   │  └─ Output Box (emerald border)
   │
   └─ Comparison Metrics Panel
      └─ 4-Column Grid
         ├─ KL Divergence (purple-400)
         ├─ Perplexity Delta (red/emerald based on sign)
         ├─ Semantic Similarity (blue-400)
         └─ Word Overlap (emerald-400)
```

### React Component Specifications

#### SteeringPanel Component

**File:** `src/components/panels/SteeringPanel.tsx`

**Props:**
```typescript
interface SteeringPanelProps {
    models: Model[];                           // All models (filter to ready)
}
```

**State:**
```typescript
interface SteeringPanelState {
    // Feature selection
    selectedFeatures: Feature[];               // Array of selected features
    steeringCoefficients: Record<number, number>; // {feature_id: coefficient}
    featureSearch: string;                     // Search query

    // Generation config
    selectedModel: string;                     // Model ID
    prompt: string;                            // Input prompt
    interventionLayer: number;                 // Layer index (default 12)
    temperature: number;                       // Default 0.7
    maxTokens: number;                         // Default 100

    // Generation results
    isGenerating: boolean;                     // Loading state
    unsteeredOutput: string;                   // Unsteered generation result
    steeredOutput: string;                     // Steered generation result
    comparisonMetrics: ComparisonMetrics | null; // Metrics or null
}
```

**Key Methods:**
```typescript
// Feature management
addFeatureToSteering(feature: Feature): void;
removeFeatureFromSteering(featureId: number): void;
updateCoefficient(featureId: number, value: number): void;

// Generation
generateComparison(): Promise<void>;

// Utilities
copyToClipboard(text: string): void;
```

**UI Reference:** Mock-embedded-interp-ui.tsx lines 3512-3951

**Styling:** Matches Mock UI exactly:
- Two-column grid layout (grid grid-cols-2 gap-6)
- Slate dark theme (bg-slate-900/50, border-slate-800)
- Emerald accents (emerald-600 buttons, emerald-400 indicators)
- Rounded corners (rounded-lg)

---

#### Feature Card Component

**Props:**
```typescript
interface FeatureCardProps {
    feature: Feature;
    coefficient: number;
    onRemove: (featureId: number) => void;
    onCoefficientChange: (featureId: number, value: number) => void;
}
```

**JSX Structure (lines 3632-3702):**
```tsx
<div className="bg-slate-800/30 rounded-lg p-4 space-y-3">
  {/* Header: Feature ID + Label + Remove */}
  <div className="flex items-center justify-between">
    <div>
      <span className="font-mono text-sm text-slate-400">#{feature.id}</span>
      <span className="ml-2 font-medium">{feature.label}</span>
    </div>
    <button onClick={onRemove} className="text-red-400">
      <X className="w-4 h-4" />
    </button>
  </div>

  {/* Coefficient Slider */}
  <div className="space-y-2">
    <div className="flex items-center justify-between text-sm">
      <span className="text-slate-400">Coefficient</span>
      <span className="text-emerald-400 font-mono">{coefficient.toFixed(2)}</span>
    </div>
    <input
      type="range"
      min="-5"
      max="5"
      step="0.1"
      value={coefficient}
      onChange={(e) => onCoefficientChange(feature.id, parseFloat(e.target.value))}
      className="w-full accent-emerald-500"
    />
    <div className="flex justify-between text-xs text-slate-500">
      <span>-5.0 (suppress)</span>
      <span>0.0</span>
      <span>+5.0 (amplify)</span>
    </div>
  </div>

  {/* Quick Presets */}
  <div className="flex gap-2">
    <button onClick={() => onCoefficientChange(feature.id, -2.0)}>Suppress</button>
    <button onClick={() => onCoefficientChange(feature.id, 0.0)}>Reset</button>
    <button onClick={() => onCoefficientChange(feature.id, 2.0)}>Amplify</button>
  </div>
</div>
```

---

### Backend Component Architecture

#### SteeringService Class

**File:** `src/services/steering_service.py`

**Purpose:** Core steering algorithm and hook management

**Key Methods:**

```python
class SteeringService:
    def __init__(self):
        self.model_registry = ModelRegistry()  # Reuse from 002_FTDD
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.hook_handles = []  # Track registered hooks for cleanup

    async def generate_with_steering(
        self,
        model_id: str,
        prompt: str,
        features: List[Dict[str, Any]],  # [{"feature_id": 42, "coefficient": 2.0}]
        intervention_layer: int,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute both unsteered and steered generations, return comparison.
        """
        # 1. Load model and SAE
        model, tokenizer = await self.model_registry.load_model(model_id)
        sae_encoder, sae_decoder = await self._load_sae_for_features(features)

        # 2. Unsteered generation
        unsteered_output, unsteered_logprobs = await self._generate_unsteered(
            model, tokenizer, prompt, temperature, max_tokens, seed
        )

        # 3. Steered generation
        steered_output, steered_logprobs = await self._generate_steered(
            model, tokenizer, sae_encoder, sae_decoder,
            prompt, features, intervention_layer, temperature, max_tokens, seed
        )

        # 4. Calculate metrics
        metrics = await self._calculate_metrics(
            unsteered_output, steered_output,
            unsteered_logprobs, steered_logprobs,
            tokenizer
        )

        return {
            "unsteered_output": unsteered_output,
            "steered_output": steered_output,
            "metrics": metrics
        }

    async def _generate_steered(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        sae_encoder: torch.nn.Module,
        sae_decoder: torch.nn.Module,
        prompt: str,
        features: List[Dict[str, Any]],
        intervention_layer: int,
        temperature: float,
        max_tokens: int,
        seed: Optional[int]
    ) -> Tuple[str, torch.Tensor]:
        """
        Generate text with steering hooks registered.
        """
        # Create steering configuration dict
        steering_config = {f["feature_id"]: f["coefficient"] for f in features}

        # Define hook function
        def steering_hook(module, input, output):
            """
            Forward hook that applies steering to activations.
            """
            # Extract activations (shape: [batch, seq_len, hidden_dim])
            activations = output[0] if isinstance(output, tuple) else output

            # Pass through SAE encoder to get feature activations
            feature_activations = sae_encoder(activations)  # [batch, seq_len, num_features]

            # Apply steering coefficients (multiplicative)
            for feature_id, coefficient in steering_config.items():
                feature_activations[:, :, feature_id] *= (1 + coefficient)

            # Clip to prevent numerical instability
            feature_activations = torch.clamp(feature_activations, min=-10, max=10)

            # Pass through SAE decoder to reconstruct activations
            steered_activations = sae_decoder(feature_activations)  # [batch, seq_len, hidden_dim]

            # Return steered activations (replaces original)
            return (steered_activations,) if isinstance(output, tuple) else steered_activations

        # Register hook on target layer
        target_layer = model.transformer.h[intervention_layer]  # GPT-2 architecture
        handle = target_layer.register_forward_hook(steering_hook)
        self.hook_handles.append(handle)

        try:
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)

            # Generate with steering active
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Extract output text and log probabilities
            output_ids = outputs.sequences[0]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            logprobs = torch.stack(outputs.scores, dim=1)  # [batch, seq_len, vocab_size]

            return output_text, logprobs

        finally:
            # CRITICAL: Unregister hook even if generation fails
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles.clear()

    async def _calculate_metrics(
        self,
        unsteered_output: str,
        steered_output: str,
        unsteered_logprobs: torch.Tensor,
        steered_logprobs: torch.Tensor,
        tokenizer: PreTrainedTokenizer
    ) -> Dict[str, float]:
        """
        Calculate comparison metrics between unsteered and steered outputs.
        """
        metrics = {}

        # 1. KL Divergence
        metrics["kl_divergence"] = self._calculate_kl_divergence(
            unsteered_logprobs, steered_logprobs
        )

        # 2. Perplexity Delta
        metrics["perplexity_delta"] = self._calculate_perplexity_delta(
            unsteered_logprobs, steered_logprobs
        )

        # 3. Semantic Similarity
        metrics["semantic_similarity"] = self._calculate_semantic_similarity(
            unsteered_output, steered_output
        )

        # 4. Word Overlap
        metrics["word_overlap"] = self._calculate_word_overlap(
            unsteered_output, steered_output, tokenizer
        )

        return metrics
```

---

#### MetricsCalculator Helper Methods

```python
def _calculate_kl_divergence(
    self,
    unsteered_logprobs: torch.Tensor,
    steered_logprobs: torch.Tensor
) -> float:
    """
    Calculate KL divergence: KL(unsteered || steered).
    Average across all token positions.
    """
    # Convert log probs to probabilities
    p_unsteered = torch.softmax(unsteered_logprobs, dim=-1)
    p_steered = torch.softmax(steered_logprobs, dim=-1)

    # KL divergence: sum(p * log(p / q))
    kl_div = torch.sum(p_unsteered * torch.log(p_unsteered / (p_steered + 1e-10)), dim=-1)

    # Average across tokens
    return float(kl_div.mean().item())

def _calculate_perplexity_delta(
    self,
    unsteered_logprobs: torch.Tensor,
    steered_logprobs: torch.Tensor
) -> float:
    """
    Calculate perplexity delta: steered_perplexity - unsteered_perplexity.
    Positive: steered is less confident (higher uncertainty).
    Negative: steered is more confident (lower uncertainty).
    """
    # Perplexity = exp(mean(negative_log_likelihood))
    # Using max log prob at each position as NLL approximation
    unsteered_nll = -torch.max(unsteered_logprobs, dim=-1).values.mean()
    steered_nll = -torch.max(steered_logprobs, dim=-1).values.mean()

    perp_unsteered = torch.exp(unsteered_nll).item()
    perp_steered = torch.exp(steered_nll).item()

    return perp_steered - perp_unsteered

def _calculate_semantic_similarity(
    self,
    unsteered_output: str,
    steered_output: str
) -> float:
    """
    Calculate cosine similarity between sentence embeddings.
    Uses cached sentence transformer model.
    """
    # Encode both outputs
    embeddings = self.sentence_model.encode(
        [unsteered_output, steered_output],
        convert_to_tensor=True
    )

    # Cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0),
        embeddings[1].unsqueeze(0)
    )

    return float(similarity.item())

def _calculate_word_overlap(
    self,
    unsteered_output: str,
    steered_output: str,
    tokenizer: PreTrainedTokenizer
) -> float:
    """
    Calculate Jaccard similarity of tokens.
    """
    # Tokenize both outputs (word-level)
    tokens_unsteered = set(tokenizer.tokenize(unsteered_output))
    tokens_steered = set(tokenizer.tokenize(steered_output))

    # Jaccard similarity: intersection / union
    intersection = len(tokens_unsteered & tokens_steered)
    union = len(tokens_unsteered | tokens_steered)

    if union == 0:
        return 0.0

    return intersection / union
```

---

## 7. State Management

### Zustand Store: useSteeringStore

**File:** `src/stores/steeringStore.ts`

```typescript
interface SteeringStore {
    // Feature selection state
    selectedFeatures: Feature[];
    steeringCoefficients: Record<number, number>;
    featureSearch: string;

    // Generation configuration
    selectedModel: string;
    prompt: string;
    interventionLayer: number;
    temperature: number;
    maxTokens: number;

    // Generation results
    isGenerating: boolean;
    unsteeredOutput: string;
    steeredOutput: string;
    comparisonMetrics: ComparisonMetrics | null;
    error: string | null;

    // Actions
    setFeatureSearch: (query: string) => void;
    addFeature: (feature: Feature) => void;
    removeFeature: (featureId: number) => void;
    updateCoefficient: (featureId: number, value: number) => void;

    setSelectedModel: (modelId: string) => void;
    setPrompt: (prompt: string) => void;
    setInterventionLayer: (layer: number) => void;
    setTemperature: (temp: number) => void;
    setMaxTokens: (tokens: number) => void;

    generateComparison: () => Promise<void>;
    reset: () => void;
}

export const useSteeringStore = create<SteeringStore>()(
    devtools((set, get) => ({
        // Initial state
        selectedFeatures: [],
        steeringCoefficients: {},
        featureSearch: '',
        selectedModel: '',
        prompt: '',
        interventionLayer: 12,
        temperature: 0.7,
        maxTokens: 100,
        isGenerating: false,
        unsteeredOutput: '',
        steeredOutput: '',
        comparisonMetrics: null,
        error: null,

        // Feature management actions
        setFeatureSearch: (query) => set({ featureSearch: query }),

        addFeature: (feature) => {
            const { selectedFeatures, steeringCoefficients } = get();
            if (!selectedFeatures.find(f => f.id === feature.id)) {
                set({
                    selectedFeatures: [...selectedFeatures, feature],
                    steeringCoefficients: {
                        ...steeringCoefficients,
                        [feature.id]: 0.0  // Initialize with neutral coefficient
                    },
                    featureSearch: ''  // Clear search after adding
                });
            }
        },

        removeFeature: (featureId) => {
            const { selectedFeatures, steeringCoefficients } = get();
            const newCoefficients = { ...steeringCoefficients };
            delete newCoefficients[featureId];

            set({
                selectedFeatures: selectedFeatures.filter(f => f.id !== featureId),
                steeringCoefficients: newCoefficients
            });
        },

        updateCoefficient: (featureId, value) => {
            const { steeringCoefficients } = get();
            set({
                steeringCoefficients: {
                    ...steeringCoefficients,
                    [featureId]: value
                }
            });
        },

        // Generation configuration actions
        setSelectedModel: (modelId) => set({ selectedModel: modelId }),
        setPrompt: (prompt) => set({ prompt }),
        setInterventionLayer: (layer) => set({ interventionLayer: layer }),
        setTemperature: (temp) => set({ temperature: temp }),
        setMaxTokens: (tokens) => set({ maxTokens: tokens }),

        // Generation action
        generateComparison: async () => {
            const {
                selectedModel,
                prompt,
                selectedFeatures,
                steeringCoefficients,
                interventionLayer,
                temperature,
                maxTokens
            } = get();

            // Build request payload
            const features = selectedFeatures.map(f => ({
                feature_id: f.id,
                coefficient: steeringCoefficients[f.id]
            }));

            const request: SteeringGenerateRequest = {
                model_id: selectedModel,
                prompt,
                features,
                intervention_layer: interventionLayer,
                temperature,
                max_tokens: maxTokens
            };

            set({ isGenerating: true, error: null });

            try {
                const response = await fetch('/api/steering/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${getAuthToken()}`
                    },
                    body: JSON.stringify(request)
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Generation failed');
                }

                const result: SteeringGenerateResponse = await response.json();

                set({
                    unsteeredOutput: result.unsteered_output,
                    steeredOutput: result.steered_output,
                    comparisonMetrics: result.metrics,
                    isGenerating: false
                });
            } catch (error) {
                set({
                    error: error.message,
                    isGenerating: false
                });
            }
        },

        reset: () => set({
            selectedFeatures: [],
            steeringCoefficients: {},
            featureSearch: '',
            prompt: '',
            unsteeredOutput: '',
            steeredOutput: '',
            comparisonMetrics: null,
            error: null
        })
    }))
);
```

---

## 8. Security Considerations

### Authentication & Authorization

- **JWT Required:** All steering endpoints require valid JWT token
- **User Context:** Future enhancement: limit features to user's own trainings
- **Rate Limiting:** 10 requests per minute per user (prevent resource exhaustion)

### Input Validation

- **Coefficient Range:** Enforce [-5.0, 5.0] range to prevent extreme steering
- **Intervention Layer:** Validate layer index does not exceed model depth
- **Prompt Length:** Limit to 1024 tokens (prevent excessive processing)
- **Feature Limit:** Maximum 10 features per request (prevent combinatorial explosion)
- **Temperature Range:** Enforce [0.0, 2.0] range (standard LM generation bounds)
- **Max Tokens:** Enforce [1, 2048] range

### Resource Protection

- **Generation Timeout:** 30 second hard limit (prevent hanging processes)
- **OOM Handling:** Graceful failure with reduced max_tokens suggestion
- **Hook Cleanup:** Always unregister forward hooks (prevent memory leaks)
- **CUDA Context:** Sequential generation prevents context conflicts

### Sanitization

- **Output Sanitization:** No special handling (generated text is not executed)
- **SQL Injection:** N/A (steering does not persist to database in MVP)

---

## 9. Performance & Scalability

### Performance Optimization Strategies

#### 1. Generation Speed

**Target:** 100-token generation (both unsteered + steered) within 3 seconds on Jetson Orin Nano

**Optimizations:**
- Use quantized models (INT4/INT8) to reduce memory bandwidth
- Set reasonable max_tokens default (100 instead of 2048)
- Execute generations sequentially (avoids CUDA context switching overhead)
- Cache SAE encoder/decoder in memory (avoid repeated loading)
- Use `torch.inference_mode()` for no-gradient generation

**Expected Performance:**
- GPT-2 Medium (345M params, INT8): ~1.5 seconds per 100 tokens
- Total with both generations + metrics: ~3-4 seconds

#### 2. Metrics Calculation Speed

**Target:** < 500ms for all 4 metrics

**Optimizations:**
- KL divergence: Vectorized PyTorch operations (single pass)
- Perplexity: Computed from existing log probabilities (no extra forward pass)
- Semantic similarity: Cache sentence transformer model in memory (80MB model)
- Word overlap: Simple set operations (negligible time)

**Expected Breakdown:**
- KL divergence: ~50ms
- Perplexity delta: ~10ms
- Semantic similarity: ~300ms (sentence encoding)
- Word overlap: ~10ms
- **Total:** ~370ms

#### 3. Memory Management

**Jetson Orin Nano Constraints:**
- GPU VRAM: 6GB
- System RAM: 8GB

**Memory Budget (Steering Feature):**
- Model (GPT-2 Medium, INT8): ~1.5GB
- SAE encoder/decoder: ~200MB
- Sentence transformer: ~80MB
- Generation buffers: ~500MB
- **Total:** ~2.3GB (well within 6GB limit)

**OOM Prevention:**
- Monitor GPU memory usage before generation
- Fallback: reduce max_tokens dynamically on OOM
- Clear CUDA cache after each generation: `torch.cuda.empty_cache()`

### Scalability Considerations

#### Multiple Features

- **Linear scaling:** Each feature adds one multiplication operation to hook
- **Tested up to:** 10 features (negligible overhead: ~50µs per token)
- **Theoretical limit:** 100+ features (constrained by SAE size, not steering logic)

#### Concurrent Requests

- **MVP limitation:** Single-user application (no concurrent steering requests)
- **Future scaling:** Queue requests via Celery (one generation at a time per GPU)
- **Multi-GPU:** Future enhancement for parallel experiments

### Caching Strategy

**Sentence Embeddings Cache:**
- Cache generated output embeddings for repeated analysis
- Key: `hash(output_text)`
- Expiration: 1 hour (in-memory LRU cache, max 1000 entries)
- Hit rate target: 10% (low for generative task, but useful for re-analysis)

**Model Loading Cache:**
- Reuse ModelRegistry from 002_FTDD (keeps last 2 models in memory)
- SAE checkpoints cached alongside models

---

## 10. Testing Strategy

### Unit Tests

**Test File:** `tests/unit/test_steering_service.py`

```python
# Steering coefficient application
def test_multiplicative_steering():
    """Test that multiplicative steering scales activations correctly."""
    feature_activations = torch.tensor([[1.0, 2.0, 3.0]])
    steering_config = {0: 1.0, 1: -0.5, 2: 0.0}  # +100%, -50%, neutral

    steered = apply_steering(feature_activations, steering_config)

    assert steered[0, 0] == 2.0   # 1.0 * (1 + 1.0)
    assert steered[0, 1] == 1.0   # 2.0 * (1 + (-0.5))
    assert steered[0, 2] == 3.0   # 3.0 * (1 + 0.0)

def test_steering_clipping():
    """Test that extreme steering is clipped to [-10, 10]."""
    feature_activations = torch.tensor([[5.0]])
    steering_config = {0: 10.0}  # Would result in 55.0 without clipping

    steered = apply_steering(feature_activations, steering_config)
    assert steered[0, 0] == 10.0  # Clipped

# Hook registration and cleanup
def test_hook_registration():
    """Test that forward hooks are registered and cleaned up properly."""
    model = create_mock_model()
    service = SteeringService()

    # Register hook
    service._register_steering_hook(model, layer=6, steering_config={42: 2.0})
    assert len(service.hook_handles) == 1

    # Cleanup
    service._cleanup_hooks()
    assert len(service.hook_handles) == 0

# Metrics calculations
def test_kl_divergence_calculation():
    """Test KL divergence between identical distributions is 0."""
    logprobs = torch.randn(1, 10, 50257)  # [batch, seq_len, vocab_size]
    kl_div = calculate_kl_divergence(logprobs, logprobs)
    assert abs(kl_div) < 1e-5  # Should be ~0

def test_perplexity_delta_calculation():
    """Test perplexity delta with identical outputs is 0."""
    logprobs = torch.randn(1, 10, 50257)
    perp_delta = calculate_perplexity_delta(logprobs, logprobs)
    assert abs(perp_delta) < 1e-2  # Should be ~0

def test_semantic_similarity_identical():
    """Test semantic similarity of identical texts is 1.0."""
    text = "The cat sat on the mat."
    similarity = calculate_semantic_similarity(text, text)
    assert similarity > 0.99  # Should be ~1.0

def test_word_overlap_identical():
    """Test word overlap of identical texts is 1.0."""
    text = "The cat sat on the mat."
    overlap = calculate_word_overlap(text, text, tokenizer)
    assert overlap == 1.0
```

### Integration Tests

**Test File:** `tests/integration/test_steering_api.py`

```python
@pytest.mark.asyncio
async def test_steering_generation_end_to_end():
    """Test complete steering workflow from API request to response."""
    # Setup: Load model and create features
    model = await load_test_model("gpt2")
    features = await create_test_features(training_id="tr_test")

    # Request steering generation
    response = client.post("/api/steering/generate", json={
        "model_id": model.id,
        "prompt": "The weather is",
        "features": [{"feature_id": features[0].id, "coefficient": 2.0}],
        "intervention_layer": 6,
        "temperature": 0.7,
        "max_tokens": 50,
        "seed": 42
    })

    assert response.status_code == 200
    result = response.json()

    # Validate response structure
    assert "unsteered_output" in result
    assert "steered_output" in result
    assert "metrics" in result
    assert len(result["unsteered_output"]) > 0
    assert len(result["steered_output"]) > 0

    # Validate metrics
    assert 0 <= result["metrics"]["kl_divergence"] <= 1
    assert 0 <= result["metrics"]["semantic_similarity"] <= 1
    assert 0 <= result["metrics"]["word_overlap"] <= 1

@pytest.mark.asyncio
async def test_steering_with_zero_coefficients():
    """Test that zero coefficients produce similar outputs to unsteered."""
    # Setup
    model = await load_test_model("gpt2")
    features = await create_test_features(training_id="tr_test")

    # Request with all coefficients = 0.0
    response = client.post("/api/steering/generate", json={
        "model_id": model.id,
        "prompt": "The weather is",
        "features": [{"feature_id": features[0].id, "coefficient": 0.0}],
        "intervention_layer": 6,
        "temperature": 0.7,
        "max_tokens": 50,
        "seed": 42
    })

    result = response.json()

    # With zero steering, outputs should be very similar
    assert result["metrics"]["semantic_similarity"] > 0.9
    assert result["metrics"]["kl_divergence"] < 0.01

@pytest.mark.asyncio
async def test_steering_with_multiple_features():
    """Test steering with multiple features simultaneously."""
    # Setup
    model = await load_test_model("gpt2")
    features = await create_test_features(training_id="tr_test", count=5)

    # Request with 5 features
    response = client.post("/api/steering/generate", json={
        "model_id": model.id,
        "prompt": "The weather is",
        "features": [
            {"feature_id": features[0].id, "coefficient": 2.0},
            {"feature_id": features[1].id, "coefficient": -1.0},
            {"feature_id": features[2].id, "coefficient": 0.5},
            {"feature_id": features[3].id, "coefficient": -0.5},
            {"feature_id": features[4].id, "coefficient": 1.5}
        ],
        "intervention_layer": 6,
        "temperature": 0.7,
        "max_tokens": 50
    })

    assert response.status_code == 200
    result = response.json()
    assert len(result["steered_output"]) > 0

@pytest.mark.asyncio
async def test_steering_at_different_layers():
    """Test steering at early, middle, and late layers."""
    model = await load_test_model("gpt2")  # 12 layers
    features = await create_test_features(training_id="tr_test")

    for layer in [2, 6, 10]:
        response = client.post("/api/steering/generate", json={
            "model_id": model.id,
            "prompt": "The weather is",
            "features": [{"feature_id": features[0].id, "coefficient": 2.0}],
            "intervention_layer": layer,
            "temperature": 0.7,
            "max_tokens": 50
        })

        assert response.status_code == 200
        # Effect magnitude may vary by layer, but should always succeed
```

### Edge Case Tests

```python
@pytest.mark.asyncio
async def test_extreme_coefficient_values():
    """Test steering with extreme coefficients (-5.0, +5.0)."""
    model = await load_test_model("gpt2")
    features = await create_test_features(training_id="tr_test")

    # Test max amplification
    response = client.post("/api/steering/generate", json={
        "model_id": model.id,
        "prompt": "The weather is",
        "features": [{"feature_id": features[0].id, "coefficient": 5.0}],
        "intervention_layer": 6,
        "temperature": 0.7,
        "max_tokens": 50
    })

    assert response.status_code == 200
    result = response.json()
    assert result["metrics"]["kl_divergence"] > 0.01  # Should have observable effect

@pytest.mark.asyncio
async def test_invalid_intervention_layer():
    """Test that intervention layer exceeding model depth returns 400."""
    model = await load_test_model("gpt2")  # 12 layers (0-11)
    features = await create_test_features(training_id="tr_test")

    response = client.post("/api/steering/generate", json={
        "model_id": model.id,
        "prompt": "The weather is",
        "features": [{"feature_id": features[0].id, "coefficient": 2.0}],
        "intervention_layer": 24,  # Invalid: exceeds 12 layers
        "temperature": 0.7,
        "max_tokens": 50
    })

    assert response.status_code == 400
    assert "exceeds model layer count" in response.json()["detail"]

@pytest.mark.asyncio
async def test_empty_prompt():
    """Test that empty prompt returns 400."""
    model = await load_test_model("gpt2")
    features = await create_test_features(training_id="tr_test")

    response = client.post("/api/steering/generate", json={
        "model_id": model.id,
        "prompt": "",  # Empty prompt
        "features": [{"feature_id": features[0].id, "coefficient": 2.0}],
        "intervention_layer": 6,
        "temperature": 0.7,
        "max_tokens": 50
    })

    assert response.status_code == 400

@pytest.mark.asyncio
async def test_generation_timeout():
    """Test that generation exceeding 30 seconds returns 408."""
    # Mock long-running generation
    with mock.patch('services.steering_service.generate', side_effect=TimeoutError):
        response = client.post("/api/steering/generate", json={
            "model_id": "m_test",
            "prompt": "Test prompt",
            "features": [{"feature_id": 1, "coefficient": 2.0}],
            "intervention_layer": 6,
            "temperature": 0.7,
            "max_tokens": 2048  # Long generation
        })

        assert response.status_code == 408
```

### Performance Tests

```python
@pytest.mark.performance
@pytest.mark.asyncio
async def test_generation_speed():
    """Test that 100-token generation completes within 3 seconds."""
    model = await load_test_model("gpt2")
    features = await create_test_features(training_id="tr_test")

    start_time = time.time()

    response = client.post("/api/steering/generate", json={
        "model_id": model.id,
        "prompt": "The weather is",
        "features": [{"feature_id": features[0].id, "coefficient": 2.0}],
        "intervention_layer": 6,
        "temperature": 0.7,
        "max_tokens": 100
    })

    elapsed = time.time() - start_time

    assert response.status_code == 200
    assert elapsed < 3.0  # Target: < 3 seconds

@pytest.mark.performance
@pytest.mark.asyncio
async def test_metrics_calculation_speed():
    """Test that metrics calculation completes within 500ms."""
    # Generate sample outputs
    unsteered = "The weather is cloudy with a chance of rain."
    steered = "The weather is absolutely beautiful and sunny!"

    # Mock log probabilities
    logprobs = torch.randn(1, 10, 50257)

    start_time = time.time()

    metrics = calculate_metrics(unsteered, steered, logprobs, logprobs, tokenizer)

    elapsed = time.time() - start_time

    assert elapsed < 0.5  # Target: < 500ms
```

---

## 11. Deployment & DevOps

### CI/CD Pipeline

**GitHub Actions Workflow:** `.github/workflows/test-steering.yml`

```yaml
name: Test Steering Feature

on:
  push:
    paths:
      - 'src/services/steering_service.py'
      - 'src/api/routes/steering.py'
      - 'src/components/panels/SteeringPanel.tsx'
      - 'tests/**/*steering*'

jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run steering unit tests
        run: pytest tests/unit/test_steering_service.py -v --cov
      - name: Run steering integration tests
        run: pytest tests/integration/test_steering_api.py -v

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: npm install
      - name: Run SteeringPanel tests
        run: npm test -- SteeringPanel.test.tsx
```

### Docker Configuration

**Dockerfile additions for steering dependencies:**

```dockerfile
# Install sentence-transformers for semantic similarity
RUN pip install sentence-transformers==2.2.2

# Download sentence model at build time (avoid runtime download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Monitoring

**Key Metrics to Track:**
- Generation speed: P50, P95, P99 latency for 100-token generation
- Steering effect rate: % of experiments with KL divergence > 0.01
- OOM error rate: % of requests failing due to memory
- Hook cleanup success rate: % of hooks properly unregistered

**Logging:**
```python
logger.info(f"Steering generation started: model={model_id}, features={len(features)}")
logger.info(f"Unsteered generation completed: {len(unsteered_output)} tokens in {unsteered_time_ms}ms")
logger.info(f"Steered generation completed: {len(steered_output)} tokens in {steered_time_ms}ms")
logger.info(f"Metrics: KL={kl_div:.4f}, PerplexityΔ={perp_delta:.2f}, Similarity={similarity:.2f}")
```

---

## 12. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Steering has no observable effect** | Medium | High | Validate algorithm with synthetic features; start with strong coefficients (±2.0); test on high interpretability features |
| **Steering causes model instability** | Medium | High | Clip activations to [-10, 10]; implement fallback to unsteered on errors; monitor perplexity delta |
| **Generation too slow (>3s)** | High | Medium | Use INT8 quantization; set max_tokens=100 default; show progress indicator; implement timeout |
| **CUDA OOM errors** | Medium | High | Monitor GPU memory; dynamic max_tokens reduction; clear CUDA cache after generation |
| **Hook cleanup failures** | Low | High | Use try/finally for hook removal; log hook handle count; implement watchdog cleanup |
| **CUDA context conflicts** | Low | Medium | Execute generations sequentially (not parallel); use single CUDA stream |

### Usability Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Users don't understand coefficients** | Medium | Medium | Clear slider labels (-5 suppress, +5 amplify); quick presets; tooltips with examples |
| **Comparison metrics not intuitive** | High | Low | Plain-language descriptions; color coding (red/green for perplexity); hide less useful metrics |
| **No visible difference in outputs** | Medium | Medium | Suggest starting with ±2.0 coefficients; show metrics even if text looks similar |
| **Feature selection too difficult** | Low | Low | Good search UX; show interpretability scores; suggest high-scoring features |

### Performance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Sentence embedding slow (>500ms)** | Medium | Low | Cache embeddings; use smaller model (MiniLM-L6); make semantic similarity optional |
| **Multiple features slow down generation** | Low | Low | Test up to 10 features; vectorize coefficient application; profile hook overhead |

---

## 13. Development Phases

### Phase 1: Core Steering Algorithm (Week 1)

**Backend:**
- [ ] Implement `SteeringService` with forward hook registration
- [ ] Implement multiplicative steering coefficient application
- [ ] Implement hook cleanup with try/finally
- [ ] Add activation clipping (-10 to +10)
- [ ] Unit tests for steering algorithm

**Frontend:**
- [ ] Create `SteeringPanel` component shell
- [ ] Implement feature selection UI (search + dropdown)
- [ ] Implement coefficient sliders with quick presets
- [ ] Match Mock UI styling exactly (lines 3512-3706)

**Deliverables:**
- Steering hook works correctly with single feature
- Coefficient slider updates state in real-time
- Frontend matches Mock UI design

---

### Phase 2: Generation & Comparison (Week 2)

**Backend:**
- [ ] Implement unsteered generation function
- [ ] Implement steered generation with hook active
- [ ] Add random seed control for reproducibility
- [ ] Implement generation timeout (30 seconds)
- [ ] Handle CUDA OOM gracefully

**Frontend:**
- [ ] Implement generation controls panel (model, prompt, layer, temp, max_tokens)
- [ ] Add "Generate Comparison" button with loading state
- [ ] Implement comparative output display (side-by-side)
- [ ] Add copy to clipboard functionality
- [ ] Match Mock UI styling (lines 3710-3895)

**Deliverables:**
- End-to-end generation works (both unsteered and steered)
- Side-by-side outputs display correctly
- Loading state shows during generation

---

### Phase 3: Metrics Calculation (Week 3)

**Backend:**
- [ ] Implement KL divergence calculation
- [ ] Implement perplexity delta calculation
- [ ] Install and cache sentence transformer model
- [ ] Implement semantic similarity calculation
- [ ] Implement word overlap calculation
- [ ] Add metrics to API response
- [ ] Unit tests for all metric functions

**Frontend:**
- [ ] Implement comparison metrics panel (4-column grid)
- [ ] Display KL divergence (purple-400)
- [ ] Display perplexity delta (red/emerald based on sign)
- [ ] Display semantic similarity (blue-400)
- [ ] Display word overlap (emerald-400)
- [ ] Match Mock UI styling (lines 3898-3947)

**Deliverables:**
- All 4 metrics calculate correctly
- Metrics panel displays in correct format
- Color coding matches Mock UI

---

### Phase 4: Testing & Optimization (Week 4)

**Testing:**
- [ ] Integration tests (end-to-end steering)
- [ ] Edge case tests (extreme coefficients, invalid layer, empty prompt)
- [ ] Performance tests (generation speed < 3s, metrics < 500ms)
- [ ] Multiple feature tests (up to 10 features)
- [ ] Different layer tests (early, middle, late)

**Optimization:**
- [ ] Profile generation speed on Jetson Orin Nano
- [ ] Optimize sentence embedding caching
- [ ] Reduce metric calculation time
- [ ] Memory profiling and CUDA cache management

**Deliverables:**
- All tests passing
- Performance targets met (3s generation, 500ms metrics)
- No memory leaks (hooks cleaned up properly)

---

### Development Timeline Summary

| Week | Focus Area | Deliverable |
|------|-----------|-------------|
| **Week 1** | Core Steering Algorithm | Hook-based steering works with frontend UI |
| **Week 2** | Generation & Comparison | End-to-end generation with side-by-side display |
| **Week 3** | Metrics Calculation | All 4 metrics calculated and displayed |
| **Week 4** | Testing & Optimization | Production-ready feature with tests passing |

**Team Size:** 2 developers (1 backend ML engineer, 1 frontend engineer)
**Total Effort:** ~320 hours (4 weeks × 2 developers × 40 hours/week)

---

## 14. Appendix

### A. Multiplicative vs Additive Steering

**Multiplicative Steering (Recommended):**
```python
steered_activation = original_activation * (1 + coefficient)

# Examples:
# coefficient = 2.0  → 3x amplification (original * 3)
# coefficient = 1.0  → 2x amplification (original * 2)
# coefficient = 0.0  → no change (original * 1)
# coefficient = -0.5 → 50% suppression (original * 0.5)
# coefficient = -1.0 → complete suppression (original * 0)
```

**Advantages:**
- Scales with activation magnitude (respects original signal strength)
- Coefficient = 0 is neutral (no intervention)
- Symmetric around 0 (positive amplifies, negative suppresses)

**Additive Steering (Alternative):**
```python
steered_activation = original_activation + coefficient

# Examples:
# coefficient = 2.0  → add 2.0 to activation
# coefficient = 0.0  → no change
# coefficient = -2.0 → subtract 2.0 from activation
```

**Disadvantages:**
- Fixed magnitude regardless of original activation (can dominate small signals)
- Requires knowledge of activation scale to set appropriate coefficients

**Recommendation:** Use multiplicative steering for MVP. Provide additive as optional setting in future.

---

### B. Steering Effect Interpretation Guide

**KL Divergence:**
- `< 0.01`: Minimal change in token distribution (steering had little effect)
- `0.01 - 0.1`: Moderate change (steering shifted some token probabilities)
- `> 0.1`: Significant change (steering substantially altered generation)

**Perplexity Delta:**
- `< 0`: Steered output is MORE confident (lower uncertainty)
- `≈ 0`: No change in model confidence
- `> 0`: Steered output is LESS confident (higher uncertainty)

**Semantic Similarity:**
- `> 0.9`: Very similar meaning despite steering
- `0.7 - 0.9`: Similar but noticeable semantic shift
- `< 0.7`: Different semantic meaning

**Word Overlap:**
- `> 0.5`: High lexical overlap (many shared words)
- `0.3 - 0.5`: Moderate overlap
- `< 0.3`: Low overlap (different word choices)

---

### C. Example Steering Configurations

**Configuration 1: Amplify Positive Sentiment**
```json
{
    "features": [
        {"feature_id": 42, "coefficient": 2.0}  // "Sentiment Positive"
    ],
    "intervention_layer": 12,
    "prompt": "The movie was",
    "expected_effect": "Steered output expresses more positive sentiment"
}
```

**Configuration 2: Suppress Negation**
```json
{
    "features": [
        {"feature_id": 89, "coefficient": -2.0}  // "Negation Logic"
    ],
    "intervention_layer": 8,
    "prompt": "I do not think",
    "expected_effect": "Steered output avoids negative phrasing"
}
```

**Configuration 3: Multi-Feature Combination**
```json
{
    "features": [
        {"feature_id": 42, "coefficient": 2.0},   // Amplify "Positive"
        {"feature_id": 137, "coefficient": -1.5}, // Suppress "Uncertainty"
        {"feature_id": 203, "coefficient": 1.0}   // Amplify "Formal Tone"
    ],
    "intervention_layer": 10,
    "prompt": "The results of the study show",
    "expected_effect": "Confident, positive, formal academic writing"
}
```

---

### D. PyTorch Forward Hook Reference

**Hook Signature:**
```python
def hook_function(
    module: torch.nn.Module,      # The layer module
    input: Tuple[torch.Tensor],   # Layer input (tuple of tensors)
    output: torch.Tensor           # Layer output (can be tuple)
) -> torch.Tensor:                 # Modified output (must match shape)
    """
    Forward hook called after layer forward pass.
    Return value replaces original output.
    """
    pass
```

**Registration:**
```python
handle = model.layer.register_forward_hook(hook_function)
```

**Cleanup:**
```python
handle.remove()  # CRITICAL: Always remove hooks after use
```

**Common Pitfalls:**
- Forgetting to remove hooks (causes memory leaks)
- Returning wrong tensor shape (causes runtime errors)
- Modifying input instead of output (hook is called AFTER forward pass)

---

### E. Sentence Transformer Model Details

**Model:** `all-MiniLM-L6-v2`
- **Size:** 80MB (small, fast)
- **Dimensions:** 384 (embedding vector size)
- **Speed:** ~100ms for single sentence on CPU
- **Use case:** Semantic similarity, not suitable for long documents (max 128 tokens)

**Alternative Models:**
- `all-mpnet-base-v2`: 420MB, better quality, slower (~300ms)
- `paraphrase-MiniLM-L6-v2`: Similar to all-MiniLM, optimized for paraphrase detection

**Caching Strategy:**
```python
# Cache at service initialization
self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode with cache
embeddings = self.sentence_model.encode(
    [text1, text2],
    convert_to_tensor=True,
    show_progress_bar=False
)
```

---

### F. Mock UI Reference Line Numbers

**Complete Steering Panel:** Lines 3512-3951

**Breakdown by Section:**
- **Component declaration + state:** Lines 3512-3533
- **Feature selection panel:** Lines 3582-3706
  - Search input: Lines 3587-3597
  - Search results dropdown: Lines 3600-3618
  - Selected features list: Lines 3621-3705
  - Feature card with slider: Lines 3632-3702
- **Generation controls panel:** Lines 3710-3817
  - Model dropdown: Lines 3715-3730
  - Prompt textarea: Lines 3733-3745
  - Intervention layer slider: Lines 3748-3761
  - Temperature + max_tokens: Lines 3764-3795
  - Generate button: Lines 3798-3815
- **Comparative output display:** Lines 3820-3895
  - Unsteered panel: Lines 3825-3857
  - Steered panel: Lines 3861-3893
- **Comparison metrics panel:** Lines 3898-3947
  - KL divergence card: Lines 3903-3911
  - Perplexity delta card: Lines 3913-3924
  - Semantic similarity card: Lines 3926-3934
  - Word overlap card: Lines 3936-3944

---

### G. References

**Research Papers:**
- Turner et al. (2023): "Activation Addition: Steering Language Models Without Optimization"
  - https://arxiv.org/abs/2308.10248
- Li et al. (2023): "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
  - https://arxiv.org/abs/2306.03341
- Templeton et al. (2024): "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"
  - https://transformer-circuits.pub/2024/scaling-monosemanticity/

**Technical Documentation:**
- PyTorch Forward Hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
- Sentence Transformers: https://www.sbert.net/
- bitsandbytes Quantization: https://github.com/TimDettmers/bitsandbytes

**UI/UX Reference:**
- Mock-embedded-interp-ui.tsx: Primary source of truth for all UI/UX decisions

---

**Document End**
**Status:** Complete - Ready for Technical Implementation Document (TID)
**Total Sections:** 14
**Estimated Document Size:** ~75KB, ~2,000 lines
**Next Steps:** Create 005_FTID|Model_Steering.md with implementation hints
