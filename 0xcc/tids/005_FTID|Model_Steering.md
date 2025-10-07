# Technical Implementation Document: Model Steering

**Document ID:** 005_FTID|Model_Steering
**Feature:** Feature-Based Model Behavior Modification
**PRD Reference:** 005_FPRD|Model_Steering.md
**TDD Reference:** 005_FTDD|Model_Steering.md
**Status:** Ready for Implementation
**Created:** 2025-10-06

---

## 1. Implementation Overview

Implements PyTorch forward hooks for real-time activation intervention during text generation, enabling feature-based model steering with side-by-side comparison and quantitative metrics. Mock UI lines 3512-3952 is PRIMARY reference.

**Key Components:**
- Forward Hook Registration (PyTorch hook system for activation modification)
- Multiplicative Steering (coefficient scales activation magnitude: `steered = original * (1 + coeff)`)
- Dual Generation (unsteered + steered with same random seed for fair comparison)
- Comparison Metrics (KL divergence, perplexity delta, semantic similarity, word overlap)
- UI with Feature Selection + Coefficient Sliders + Side-by-Side Output

---

## 2. Steering Hook Implementation

**File:** `backend/src/services/steering_service.py`

```python
import torch
from typing import List, Dict

class SteeringService:
    """Implement feature-based model steering via forward hooks."""

    def __init__(self):
        self.hook_handles = []

    async def generate_with_steering(
        self,
        model_id: str,
        prompt: str,
        features: List[Dict],  # [{"feature_id": 42, "coefficient": 2.0}, ...]
        intervention_layer: int,
        temperature: float,
        max_tokens: int,
        seed: int = None
    ):
        """
        Generate text with steering.

        Returns:
        {
            "unsteered_output": str,
            "steered_output": str,
            "metrics": {
                "kl_divergence": float,
                "perplexity_delta": float,
                "semantic_similarity": float,
                "word_overlap": float
            }
        }
        """
        # 1. Load model and SAE
        model, tokenizer = load_model(model_id)
        sae_encoder, sae_decoder = await self._load_sae_for_features(features)

        # 2. Unsteered generation
        if seed is not None:
            torch.manual_seed(seed)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        with torch.no_grad():
            unsteered_outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )

        unsteered_text = tokenizer.decode(unsteered_outputs.sequences[0], skip_special_tokens=True)
        unsteered_logprobs = torch.stack(unsteered_outputs.scores, dim=1)

        # 3. Steered generation
        if seed is not None:
            torch.manual_seed(seed)

        # Build steering config {feature_id: coefficient}
        steering_config = {f["feature_id"]: f["coefficient"] for f in features}

        # Register steering hook
        self._register_steering_hook(
            model,
            sae_encoder,
            sae_decoder,
            intervention_layer,
            steering_config
        )

        try:
            with torch.no_grad():
                steered_outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            steered_text = tokenizer.decode(steered_outputs.sequences[0], skip_special_tokens=True)
            steered_logprobs = torch.stack(steered_outputs.scores, dim=1)

        finally:
            # CRITICAL: Remove hooks
            self._cleanup_hooks()

        # 4. Calculate metrics
        metrics = self._calculate_metrics(
            unsteered_text,
            steered_text,
            unsteered_logprobs,
            steered_logprobs,
            tokenizer
        )

        return {
            "unsteered_output": unsteered_text,
            "steered_output": steered_text,
            "metrics": metrics
        }

    def _register_steering_hook(
        self,
        model,
        sae_encoder,
        sae_decoder,
        intervention_layer: int,
        steering_config: Dict[int, float]
    ):
        """Register forward hook on target layer."""

        def steering_hook(module, input, output):
            """
            Apply steering to activations.

            Steps:
            1. Extract activations [batch, seq, hidden_dim]
            2. Pass through SAE encoder → [batch, seq, latent_dim]
            3. Apply steering: steered[feature_id] = original[feature_id] * (1 + coeff)
            4. Clip to [-10, 10] for numerical stability
            5. Pass through SAE decoder → [batch, seq, hidden_dim]
            6. Return steered activations
            """
            # Extract activations
            activations = output[0] if isinstance(output, tuple) else output

            # SAE encode
            feature_activations = torch.relu(sae_encoder(activations))

            # Apply steering (multiplicative)
            for feature_id, coefficient in steering_config.items():
                feature_activations[:, :, feature_id] *= (1 + coefficient)

            # Clip for stability
            feature_activations = torch.clamp(feature_activations, min=-10, max=10)

            # SAE decode
            steered_activations = sae_decoder(feature_activations)

            # Return steered activations
            return (steered_activations,) if isinstance(output, tuple) else steered_activations

        # Register hook on intervention layer
        target_layer = model.transformer.h[intervention_layer]  # GPT-2 architecture
        handle = target_layer.register_forward_hook(steering_hook)
        self.hook_handles.append(handle)

    def _cleanup_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def _calculate_metrics(
        self,
        unsteered_text: str,
        steered_text: str,
        unsteered_logprobs: torch.Tensor,
        steered_logprobs: torch.Tensor,
        tokenizer
    ) -> Dict[str, float]:
        """Calculate comparison metrics."""

        # 1. KL Divergence
        p_unsteered = torch.softmax(unsteered_logprobs, dim=-1)
        p_steered = torch.softmax(steered_logprobs, dim=-1)
        kl_div = torch.sum(p_unsteered * torch.log(p_unsteered / (p_steered + 1e-10)), dim=-1).mean().item()

        # 2. Perplexity Delta
        unsteered_nll = -torch.max(unsteered_logprobs, dim=-1).values.mean()
        steered_nll = -torch.max(steered_logprobs, dim=-1).values.mean()
        perp_unsteered = torch.exp(unsteered_nll).item()
        perp_steered = torch.exp(steered_nll).item()
        perplexity_delta = perp_steered - perp_unsteered

        # 3. Semantic Similarity
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = sentence_model.encode([unsteered_text, steered_text], convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        ).item()

        # 4. Word Overlap
        tokens_unsteered = set(tokenizer.tokenize(unsteered_text))
        tokens_steered = set(tokenizer.tokenize(steered_text))
        intersection = len(tokens_unsteered & tokens_steered)
        union = len(tokens_unsteered | tokens_steered)
        word_overlap = intersection / union if union > 0 else 0.0

        return {
            "kl_divergence": round(kl_div, 4),
            "perplexity_delta": round(perplexity_delta, 2),
            "semantic_similarity": round(similarity, 2),
            "word_overlap": round(word_overlap, 2)
        }
```

**Steering Algorithm:**
- **Multiplicative:** `steered = original * (1 + coefficient)`
  - `coeff = 2.0` → 3x amplification
  - `coeff = -0.5` → 50% suppression
  - `coeff = 0.0` → no change
- **Clipping:** Prevents numerical instability (`-10 to +10`)
- **Sequential Generation:** Unsteered first, then steered (avoids CUDA context conflicts)

---

## 3. Frontend: SteeringPanel Component

**File:** `frontend/src/components/panels/SteeringPanel.tsx`

**PRIMARY REFERENCE:** Mock UI lines 3512-3951

```typescript
import React, { useState } from 'react';
import { Search, X, Zap, Loader, Copy } from 'lucide-react';
import { useSteeringStore } from '@/stores/steeringStore';

export const SteeringPanel: React.FC = () => {
  const models = useModelsStore((state) => state.models.filter(m => m.status === 'ready'));

  const {
    selectedFeatures,
    steeringCoefficients,
    selectedModel,
    prompt,
    interventionLayer,
    temperature,
    maxTokens,
    isGenerating,
    unsteeredOutput,
    steeredOutput,
    comparisonMetrics,
    setSelectedModel,
    setPrompt,
    setInterventionLayer,
    setTemperature,
    setMaxTokens,
    addFeature,
    removeFeature,
    updateCoefficient,
    generateComparison
  } = useSteeringStore();

  const [featureSearch, setFeatureSearch] = useState('');

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Model Steering</h2>

      <div className="grid grid-cols-2 gap-6">
        {/* LEFT: Feature Selection Panel */}
        <div className="space-y-4">
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Select Features to Steer</h3>

            {/* Feature Search */}
            <div className="relative mb-4">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search features to add..."
                value={featureSearch}
                onChange={(e) => setFeatureSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              />
            </div>

            {/* Selected Features */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-slate-300">
                Selected Features ({selectedFeatures.length})
              </h4>

              {selectedFeatures.length === 0 ? (
                <div className="text-center py-8 text-slate-400 text-sm">
                  No features selected. Search and add features above.
                </div>
              ) : (
                selectedFeatures.map(feature => (
                  <div key={feature.id} className="bg-slate-800/30 rounded-lg p-4 space-y-3">
                    {/* Header: Feature ID + Label + Remove */}
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-mono text-sm text-slate-400">#{feature.id}</span>
                        <span className="ml-2 font-medium">{feature.label}</span>
                      </div>
                      <button onClick={() => removeFeature(feature.id)} className="text-red-400 hover:text-red-300">
                        <X className="w-4 h-4" />
                      </button>
                    </div>

                    {/* Coefficient Slider */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-400">Coefficient</span>
                        <span className="text-emerald-400 font-mono">
                          {(steeringCoefficients[feature.id] || 0).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="-5"
                        max="5"
                        step="0.1"
                        value={steeringCoefficients[feature.id] || 0}
                        onChange={(e) => updateCoefficient(feature.id, parseFloat(e.target.value))}
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
                      <button onClick={() => updateCoefficient(feature.id, -2.0)}
                        className="px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600">Suppress</button>
                      <button onClick={() => updateCoefficient(feature.id, 0.0)}
                        className="px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600">Reset</button>
                      <button onClick={() => updateCoefficient(feature.id, 2.0)}
                        className="px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600">Amplify</button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: Generation Configuration Panel */}
        <div className="space-y-4">
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
            <h3 className="text-lg font-semibold">Generation Configuration</h3>

            {/* Model */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Model</label>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500">
                <option value="">Select model...</option>
                {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
            </div>

            {/* Prompt */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your prompt here..."
                rows={4}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg resize-none focus:outline-none focus:border-emerald-500"
              />
            </div>

            {/* Intervention Layer */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Intervention Layer: {interventionLayer}
              </label>
              <input
                type="range"
                min="0"
                max="24"
                value={interventionLayer}
                onChange={(e) => setInterventionLayer(parseInt(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Temperature + Max Tokens */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Temperature</label>
                <input
                  type="number"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  min="0"
                  max="2"
                  step="0.1"
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Max Tokens</label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  min="1"
                  max="2048"
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                />
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={generateComparison}
              disabled={!prompt || selectedFeatures.length === 0 || isGenerating}
              className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium"
            >
              {isGenerating ? (
                <><Loader className="w-5 h-5 animate-spin" />Generating...</>
              ) : (
                <><Zap className="w-5 h-5" />Generate Comparison</>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Comparative Output Display */}
      {(unsteeredOutput || steeredOutput) && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-6">
            {/* Unsteered */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-slate-400" />
                  Unsteered (Baseline)
                </h3>
                <button onClick={() => navigator.clipboard.writeText(unsteeredOutput)}
                  className="text-sm text-slate-400 hover:text-slate-200">
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4 min-h-[200px]">
                <div className="text-slate-300 whitespace-pre-wrap text-sm leading-relaxed">
                  {unsteeredOutput}
                </div>
              </div>
            </div>

            {/* Steered */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-400" />
                  Steered
                </h3>
                <button onClick={() => navigator.clipboard.writeText(steeredOutput)}
                  className="text-sm text-slate-400 hover:text-slate-200">
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <div className="bg-slate-900/50 border border-emerald-800/30 rounded-lg p-4 min-h-[200px]">
                <div className="text-slate-300 whitespace-pre-wrap text-sm leading-relaxed">
                  {steeredOutput}
                </div>
              </div>
            </div>
          </div>

          {/* Comparison Metrics */}
          {comparisonMetrics && (
            <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-4">Comparison Metrics</h3>
              <div className="grid grid-cols-4 gap-4">
                {/* KL Divergence */}
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="text-xs text-slate-400 mb-1">KL Divergence</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {comparisonMetrics.kl_divergence.toFixed(4)}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">Distribution shift</div>
                </div>

                {/* Perplexity Delta */}
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="text-xs text-slate-400 mb-1">Perplexity Δ</div>
                  <div className={`text-2xl font-bold ${
                    comparisonMetrics.perplexity_delta > 0 ? 'text-red-400' : 'text-emerald-400'
                  }`}>
                    {comparisonMetrics.perplexity_delta > 0 ? '+' : ''}
                    {comparisonMetrics.perplexity_delta.toFixed(2)}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {comparisonMetrics.perplexity_delta > 0 ? 'Higher' : 'Lower'} uncertainty
                  </div>
                </div>

                {/* Semantic Similarity */}
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="text-xs text-slate-400 mb-1">Similarity</div>
                  <div className="text-2xl font-bold text-blue-400">
                    {(comparisonMetrics.semantic_similarity * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-slate-500 mt-1">Cosine similarity</div>
                </div>

                {/* Word Overlap */}
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="text-xs text-slate-400 mb-1">Word Overlap</div>
                  <div className="text-2xl font-bold text-emerald-400">
                    {(comparisonMetrics.word_overlap * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-slate-500 mt-1">Shared tokens</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
```

**Key Styling:**
- Feature cards: `bg-slate-800/30` with emerald slider (`accent-emerald-500`)
- Unsteered output: `border border-slate-800` with slate dot indicator
- Steered output: `border border-emerald-800/30` with emerald dot indicator
- Metrics: Purple (KL), Red/Emerald (Perplexity), Blue (Similarity), Emerald (Overlap)

---

## 4. Testing Implementation

**File:** `backend/tests/unit/test_steering.py`

```python
import pytest
import torch
from src.services.steering_service import SteeringService

@pytest.mark.asyncio
async def test_multiplicative_steering():
    """Test that multiplicative steering scales activations correctly."""
    feature_activations = torch.tensor([[1.0, 2.0, 3.0]])
    steering_config = {0: 1.0, 1: -0.5, 2: 0.0}

    # Apply steering
    steered = feature_activations.clone()
    for feature_id, coeff in steering_config.items():
        steered[:, feature_id] *= (1 + coeff)

    # Assertions
    assert steered[0, 0].item() == 2.0  # 1.0 * (1 + 1.0)
    assert steered[0, 1].item() == 1.0  # 2.0 * (1 + (-0.5))
    assert steered[0, 2].item() == 3.0  # 3.0 * (1 + 0.0)

@pytest.mark.asyncio
async def test_zero_coefficients_no_effect():
    """Test that zero coefficients produce similar outputs."""
    # Initialize service
    service = SteeringService()

    # Generate with all coefficients = 0
    result = await service.generate_with_steering(
        model_id="gpt2",
        prompt="The weather is",
        features=[{"feature_id": 42, "coefficient": 0.0}],
        intervention_layer=6,
        temperature=0.7,
        max_tokens=50,
        seed=42
    )

    # Outputs should be very similar (high semantic similarity)
    assert result["metrics"]["semantic_similarity"] > 0.9
    assert result["metrics"]["kl_divergence"] < 0.01
```

---

## 5. Training Job Selector Implementation

This section provides implementation guidance for the training job selector dropdown (FR-3A from PRD).

### Component: Training Job Dropdown

**Location:** At the top of SteeringPanel, before feature selection (Mock UI lines 3512-3951)

**Implementation Pattern:**

```typescript
// Add to SteeringPanel state
const [trainings, setTrainings] = useState<CompletedTraining[]>([]);
const [selectedTrainingId, setSelectedTrainingId] = useState<string>('');
const [selectedFeatures, setSelectedFeatures] = useState<SelectedFeature[]>([]);

// Fetch completed trainings on mount
useEffect(() => {
  fetchCompletedTrainings();
}, []);

const fetchCompletedTrainings = async () => {
  try {
    const response = await fetch('/api/trainings/completed?limit=100');
    const data = await response.json();
    setTrainings(data.trainings);

    // Auto-select first training if available
    if (data.trainings.length > 0 && !selectedTrainingId) {
      setSelectedTrainingId(data.trainings[0].id);
    }
  } catch (error) {
    console.error('Failed to fetch completed trainings:', error);
  }
};

// Handle training selection change
const handleTrainingChange = (newTrainingId: string) => {
  if (selectedFeatures.length > 0) {
    const confirmed = window.confirm(
      'Changing training job will clear selected features. Continue?'
    );
    if (!confirmed) return;
  }

  // Clear feature-related state
  setSelectedTrainingId(newTrainingId);
  setSelectedFeatures([]);
  setFeatureCoefficients({});
  setActivePreset(null);

  // Fetch features for new training
  if (newTrainingId) {
    fetchFeaturesForTraining(newTrainingId);
  }
};

// Format training display name
const formatTrainingDisplayName = (training: CompletedTraining) => {
  const date = new Date(training.created_at).toLocaleDateString('en-US', {
    month: 'numeric',
    day: 'numeric',
    year: 'numeric'
  });

  return `${training.encoder_type} SAE • ${training.model_name} • ${training.dataset_name} • Started ${date}`;
};
```

**UI Structure:**

```typescript
<div className="mb-6">
  <label className="block mb-2 text-sm font-medium text-gray-300">
    Training Job
    <span className="ml-2 text-xs text-gray-500">
      (defines feature space)
    </span>
  </label>

  <select
    value={selectedTrainingId}
    onChange={(e) => handleTrainingChange(e.target.value)}
    className="w-full px-3 py-2 text-sm bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-emerald-500"
  >
    <option value="">Select a training job...</option>
    {trainings.map((training) => (
      <option key={training.id} value={training.id}>
        {formatTrainingDisplayName(training)}
      </option>
    ))}
  </select>

  {trainings.length === 0 && (
    <p className="mt-2 text-xs text-gray-500">
      No completed trainings available. Complete a training first to use steering.
    </p>
  )}

  {selectedFeatures.length > 0 && (
    <div className="flex items-center gap-2 px-3 py-2 mt-2 text-xs bg-blue-900/20 rounded">
      <Info className="w-4 h-4 text-blue-400 flex-shrink-0" />
      <span className="text-blue-400">
        Features are specific to this training. Changing training will clear selections.
      </span>
    </div>
  )}
</div>
```

### Backend: Completed Trainings Endpoint

**File:** `backend/src/api/routes/trainings.py`

```python
@router.get("/trainings/completed")
async def list_completed_trainings(
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List completed training jobs with model and dataset names."""
    # Query with JOINs to fetch names dynamically
    query = db.query(
        Training.id,
        Training.encoder_type,
        Training.status,
        Training.created_at,
        Model.id.label('model_id'),
        Model.name.label('model_name'),
        Dataset.id.label('dataset_id'),
        Dataset.name.label('dataset_name')
    ).join(
        Model, Training.model_id == Model.id
    ).join(
        Dataset, Training.dataset_id == Dataset.id
    ).filter(
        Training.status == 'completed'
    ).order_by(
        Training.created_at.desc()
    )

    total = query.count()
    results = query.offset(offset).limit(limit).all()

    trainings = [
        {
            "id": r.id,
            "encoder_type": r.encoder_type,
            "model_name": r.model_name,
            "dataset_name": r.dataset_name,
            "created_at": r.created_at.isoformat()
        }
        for r in results
    ]

    return {
        "trainings": trainings,
        "total": total,
        "limit": limit,
        "offset": offset
    }
```

### TypeScript Interfaces

```typescript
export interface CompletedTraining {
  id: string;
  encoder_type: string;
  model_name: string;
  dataset_name: string;
  created_at: string;
}
```

---

## 6. Steering Preset Management Implementation

This section provides implementation guidance for steering preset save/load/favorite/export/import functionality (FR-3B from PRD).

### Component: Saved Presets Section

**Location:** Inside SteeringPanel, below training job selector (Mock UI lines 3512-3951)

**Implementation Pattern:**

```typescript
// Add to SteeringPanel state
const [presets, setPresets] = useState<SteeringPreset[]>([]);
const [showPresets, setShowPresets] = useState(false);
const [presetName, setPresetName] = useState('');
const [presetDescription, setPresetDescription] = useState('');

// Fetch presets when training selected
useEffect(() => {
  if (selectedTrainingId) {
    fetchSteeringPresets(selectedTrainingId);
  }
}, [selectedTrainingId]);

const fetchSteeringPresets = async (trainingId: string) => {
  try {
    const response = await fetch(`/api/presets/steering?training_id=${trainingId}&limit=50`);
    const data = await response.json();
    setPresets(data.presets);
  } catch (error) {
    console.error('Failed to fetch presets:', error);
  }
};

// Auto-generate preset name
const generatePresetName = () => {
  const timestamp = new Date().toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit'
  }).replace(':', '');

  if (interventionLayers.length === 1) {
    return `steering_${selectedFeatures.length}features_layer${interventionLayers[0]}_${timestamp}`;
  } else {
    const min = Math.min(...interventionLayers);
    const max = Math.max(...interventionLayers);
    return `steering_${selectedFeatures.length}features_layers${min}-${max}_${timestamp}`;
  }
};

// Save preset handler
const handleSavePreset = async () => {
  const name = presetName || generatePresetName();

  const preset = {
    training_id: selectedTrainingId,
    name,
    description: presetDescription,
    features: selectedFeatures.map(f => ({
      feature_id: f.id,
      coefficient: f.coefficient
    })),
    intervention_layers: interventionLayers,
    temperature,
    max_tokens: maxTokens,
    is_favorite: false
  };

  try {
    const response = await fetch('/api/presets/steering', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(preset)
    });

    if (response.ok) {
      await fetchSteeringPresets(selectedTrainingId);
      setPresetName('');
      setPresetDescription('');
      // Show success toast
    }
  } catch (error) {
    console.error('Failed to save preset:', error);
  }
};

// Load preset handler
const handleLoadPreset = (preset: SteeringPreset) => {
  // Load features with coefficients
  setSelectedFeatures(preset.features.map(f => ({
    id: f.feature_id,
    name: '', // Will be populated from feature lookup
    coefficient: f.coefficient
  })));

  // Load intervention layers
  setInterventionLayers(preset.intervention_layers);

  // Load generation settings
  setTemperature(preset.temperature);
  setMaxTokens(preset.max_tokens);

  // Show success toast
};
```

**UI Structure:** (Similar to training/extraction templates with same patterns)

---

## 7. Multi-Layer Steering Implementation

This section provides implementation guidance for applying steering across multiple layers (FR-3C from PRD).

### Component: Intervention Layers Selector

**Location:** Inside SteeringPanel generation configuration section

**Implementation Pattern:**

```typescript
// Add to SteeringPanel state
const [interventionLayers, setInterventionLayers] = useState<number[]>([10]);
const [modelNumLayers, setModelNumLayers] = useState<number | null>(null);

// Fetch model architecture from selected training
useEffect(() => {
  if (selectedTrainingId) {
    fetchTrainingModelArchitecture(selectedTrainingId);
  }
}, [selectedTrainingId]);

const fetchTrainingModelArchitecture = async (trainingId: string) => {
  try {
    const response = await fetch(`/api/trainings/${trainingId}`);
    const data = await response.json();

    // Fetch model details
    const modelResponse = await fetch(`/api/models/${data.model_id}`);
    const modelData = await modelResponse.json();
    setModelNumLayers(modelData.num_layers);
  } catch (error) {
    console.error('Failed to fetch model architecture:', error);
  }
};

// Layer selection handlers (same as training layers)
const handleLayerToggle = (layer: number) => {
  setInterventionLayers(prev => {
    if (prev.includes(layer)) {
      return prev.filter(l => l !== layer);
    } else {
      return [...prev, layer].sort((a, b) => a - b);
    }
  });
};

const handleSelectAllLayers = () => {
  if (modelNumLayers) {
    setInterventionLayers(Array.from({ length: modelNumLayers }, (_, i) => i));
  }
};

const handleClearAllLayers = () => {
  setInterventionLayers([10]); // Reset to default middle layer
};
```

**UI Structure:** (8-column checkbox grid, same as training layers selector)

```typescript
<div className="mb-4">
  <label className="block mb-2 text-sm font-medium text-gray-300">
    Intervention Layers
    <span className="ml-2 text-xs text-gray-500">
      ({interventionLayers.length} selected)
    </span>
  </label>

  {/* Select All / Clear All buttons */}
  <div className="flex gap-2 mb-2">
    <button onClick={handleSelectAllLayers} /* ... */>Select All</button>
    <button onClick={handleClearAllLayers} /* ... */>Clear All</button>
  </div>

  {/* 8-column checkbox grid */}
  {modelNumLayers && (
    <div className="grid grid-cols-8 gap-2 p-3 bg-gray-800 rounded max-h-48 overflow-y-auto">
      {Array.from({ length: modelNumLayers }, (_, i) => (
        <label key={i} className={/* ... */}>
          <input
            type="checkbox"
            checked={interventionLayers.includes(i)}
            onChange={() => handleLayerToggle(i)}
            className="sr-only"
          />
          L{i}
        </label>
      ))}
    </div>
  )}

  {/* Warning for many layers */}
  {interventionLayers.length > 4 && (
    <div className="p-2 mt-2 text-xs text-yellow-400 bg-yellow-900/20 rounded">
      <AlertTriangle className="inline w-4 h-4 mr-1" />
      Steering at {interventionLayers.length} layers may produce unexpected results. Effects compound across layers.
    </div>
  )}
</div>
```

### Backend: Multi-Layer Steering Pipeline

**File:** `backend/src/services/steering_service.py`

**Hook Registration:**

```python
def register_multilayer_hooks(model, intervention_layers: list, steering_vector: torch.Tensor):
    """Register forward hooks at multiple layers."""
    hooks = []

    for layer_idx in intervention_layers:
        # Validate layer index
        if layer_idx >= model.config.num_hidden_layers:
            raise ValueError(f"Layer {layer_idx} exceeds model layer count")

        layer = model.transformer.h[layer_idx]

        def steering_hook(module, input, output):
            # Apply steering vector by adding to activations
            return output + steering_vector.to(output.device)

        hook = layer.register_forward_hook(steering_hook)
        hooks.append(hook)

    return hooks

def remove_multilayer_hooks(hooks: list):
    """Remove all registered hooks."""
    for hook in hooks:
        hook.remove()
```

**Steering Vector Computation:**

```python
def compute_steering_vector(sae, features: list, coefficients: list):
    """Compute steering vector from SAE features and coefficients."""
    steering_vector = torch.zeros(sae.d_model, device=sae.device)

    for feature_id, coefficient in zip(features, coefficients):
        # Get decoder direction for this feature
        feature_direction = sae.decoder.weight[feature_id]  # (hidden_size,)

        # Scale by coefficient and add to steering vector
        steering_vector += coefficient * feature_direction

    return steering_vector
```

**Generation Pipeline:**

```python
async def generate_with_multilayer_steering(
    training_id: str,
    model_id: str,
    prompt: str,
    features: list,
    coefficients: list,
    intervention_layers: list,
    temperature: float = 1.0,
    max_tokens: int = 100,
    seed: Optional[int] = None
):
    # 1. Load model and SAE checkpoint
    model = load_model(model_id)
    sae = load_sae_checkpoint(training_id)

    # 2. Compute steering vector (once, applied to all layers)
    steering_vector = compute_steering_vector(sae, features, coefficients)

    # 3. Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 4. Generate unsteered baseline (no hooks)
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        unsteered_output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )

    # 5. Register multi-layer steering hooks
    hooks = register_multilayer_hooks(model, intervention_layers, steering_vector)

    # 6. Generate steered output (with hooks active)
    try:
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            steered_output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
    finally:
        # 7. Always remove hooks (even if generation fails)
        remove_multilayer_hooks(hooks)

    # 8. Decode outputs
    unsteered_text = tokenizer.decode(unsteered_output[0], skip_special_tokens=True)
    steered_text = tokenizer.decode(steered_output[0], skip_special_tokens=True)

    # 9. Calculate comparison metrics
    metrics = calculate_comparison_metrics(unsteered_text, steered_text)

    return {
        "unsteered_output": unsteered_text,
        "steered_output": steered_text,
        "metrics": metrics
    }
```

**API Request/Response Updates:**

```python
class SteeringGenerateRequest(BaseModel):
    training_id: str  # NEW: Required
    model_id: str
    prompt: str
    features: List[dict]  # [{"feature_id": 42, "coefficient": 2.0}, ...]
    intervention_layers: List[int]  # NEW: Array instead of single layer
    temperature: float = 1.0
    max_tokens: int = 100
    seed: Optional[int] = None

    @validator('intervention_layers')
    def validate_intervention_layers(cls, v):
        if not v or len(v) == 0:
            raise ValueError("intervention_layers cannot be empty")
        if len(v) > 20:
            raise ValueError("intervention_layers cannot exceed 20 layers")
        return sorted(v)  # Ensure sorted order

    @validator('features')
    def validate_features(cls, v, values):
        # Validate all features belong to training_id
        training_id = values.get('training_id')
        if training_id:
            # Query database to verify features belong to this training
            pass
        return v
```

### Error Handling:

```python
try:
    result = await generate_with_multilayer_steering(...)

except ValueError as e:
    # Validation error (invalid layers, features don't match training, etc.)
    raise HTTPException(status_code=400, detail=str(e))

except torch.cuda.OutOfMemoryError:
    # OOM during generation
    raise HTTPException(
        status_code=507,
        detail="Out of memory during generation. Try reducing max_tokens or using fewer intervention layers."
    )

except Exception as e:
    logger.exception(f"Error during multi-layer steering: {e}")
    raise HTTPException(status_code=500, detail="Internal server error during steering")
```

### Implementation Checklist

- [ ] Add steering_presets table migration with intervention_layers array and training_id FK
- [ ] Implement GET /api/trainings/completed with dynamic JOINs
- [ ] Add training job selector dropdown to SteeringPanel
- [ ] Implement training change confirmation dialog
- [ ] Implement all 7 API endpoints for steering presets
- [ ] Add "Saved Presets" section to SteeringPanel
- [ ] Implement intervention layers selector (8-column checkbox grid)
- [ ] Update POST /api/steering/generate to accept training_id and intervention_layers array
- [ ] Implement multi-layer hook registration
- [ ] Implement steering vector computation from SAE features
- [ ] Update generation pipeline to use multi-layer hooks
- [ ] Add validation for features belonging to training_id
- [ ] Add validation for intervention_layers vs. model architecture
- [ ] Test with single layer (backward compatible)
- [ ] Test with multiple layers (3-5 layers typical)
- [ ] Test preset save/load with multi-layer configuration
- [ ] Test training job change behavior (clears features)
- [ ] Test OOM handling for generation

---

**Document End**
**Status:** Ready for Task Generation
**Total Sections:** 8 (comprehensive implementation guide)
**Estimated Size:** ~50KB
**Next:** Generate task lists for all three enhanced features
