# Technical Implementation Document: Model Steering

**Document ID:** 006_FTID|Model_Steering
**Version:** 1.1 (Combined Mode Enhancement)
**Last Updated:** 2026-01-24
**Status:** Partially Implemented (Combined Mode Planned)
**Related TDD:** [006_FTDD|Model_Steering](../tdds/006_FTDD|Model_Steering.md)

---

## 1. Implementation Order

### Phase 1: Steering Service
1. Forward hook infrastructure
2. Steering service with intervention logic
3. Calibration utilities

### Phase 2: API Endpoints
1. Generate endpoint
2. Compare endpoint
3. Sweep endpoint
4. Calibrate endpoint

### Phase 3: Prompt Templates
1. Database migration
2. Template service
3. Template CRUD API

### Phase 4: Frontend
1. Steering panel layout
2. Feature browser integration
3. Strength slider component
4. Comparison results view
5. Prompt template editor

### Phase 5: Combined Multi-Feature Mode (Planned)
1. Combined steering hook implementation
2. Combined generation API endpoint
3. Combined mode UI toggle
4. Combined results display component
5. Combined mode store integration

---

## 2. File-by-File Implementation

### 2.1 Backend - Steering Service

#### `backend/src/ml/forward_hooks.py`
```python
import torch
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

@dataclass
class SteeringConfig:
    feature_index: int
    strength: float
    sae_path: str
    layer: int
    hook_point: str = "resid_post"

class SteeringHookManager:
    """Manages forward hooks for feature steering."""

    def __init__(self, model, sae):
        self.model = model
        self.sae = sae
        self.hooks = []
        self.steering_configs: List[SteeringConfig] = []
        self._calibration_scale = 1.0

    def set_steering(self, configs: List[SteeringConfig]):
        """Set steering configuration."""
        self.steering_configs = configs

    def set_calibration_scale(self, scale: float):
        """Set calibration scale for steering strength."""
        self._calibration_scale = scale

    def _create_steering_hook(self, layer: int) -> Callable:
        """Create hook function for a specific layer."""
        def hook(module, input, output):
            if not self.steering_configs:
                return output

            # Get relevant configs for this layer
            layer_configs = [c for c in self.steering_configs if c.layer == layer]
            if not layer_configs:
                return output

            # Output shape: [batch, seq, hidden]
            hidden_states = output[0] if isinstance(output, tuple) else output
            device = hidden_states.device

            # Encode through SAE
            with torch.no_grad():
                # Flatten for SAE
                original_shape = hidden_states.shape
                flat = hidden_states.view(-1, original_shape[-1])

                # Get feature activations
                features = self.sae.encode(flat)

                # Apply steering interventions
                for config in layer_configs:
                    # Scale strength by calibration
                    scaled_strength = config.strength * self._calibration_scale

                    # Add to feature activation
                    features[:, config.feature_index] += scaled_strength

                # Decode back
                modified = self.sae.decode(features)
                modified = modified.view(original_shape)

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook

    def register_hooks(self, layers: List[int]):
        """Register hooks at specified layers."""
        self.clear_hooks()

        for layer_idx in layers:
            module = self.model.model.layers[layer_idx]
            hook = module.register_forward_hook(self._create_steering_hook(layer_idx))
            self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        self.clear_hooks()
```

#### `backend/src/services/steering_service.py`
```python
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ml.forward_hooks import SteeringHookManager, SteeringConfig
from src.ml.sae_loader import load_sae
from src.schemas.steering import (
    SteeringRequest,
    SteeringResult,
    ComparisonResult,
    SweepResult
)

class SteeringService:
    """Service for model steering via SAE feature interventions."""

    def __init__(self, model, tokenizer, sae, layer: int):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.layer = layer
        self.hook_manager = SteeringHookManager(model, sae)
        self._calibration_cache = {}

    def generate_steered(
        self,
        prompt: str,
        feature_configs: List[Dict],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """Generate text with steering applied."""
        # Convert configs to SteeringConfig objects
        configs = [
            SteeringConfig(
                feature_index=c["feature_index"],
                strength=c["strength"],
                sae_path="",  # Already loaded
                layer=self.layer
            )
            for c in feature_configs
        ]

        # Set up steering
        self.hook_manager.set_steering(configs)
        self.hook_manager.register_hooks([self.layer])

        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return generated

        finally:
            self.hook_manager.clear_hooks()

    def generate_comparison(
        self,
        prompt: str,
        feature_configs: List[Dict],
        **gen_kwargs
    ) -> ComparisonResult:
        """Generate both baseline and steered outputs."""
        # Baseline (no steering)
        self.hook_manager.clear_hooks()
        baseline = self._generate_raw(prompt, **gen_kwargs)

        # Steered
        steered = self.generate_steered(prompt, feature_configs, **gen_kwargs)

        return ComparisonResult(
            prompt=prompt,
            baseline_output=baseline,
            steered_output=steered,
            feature_configs=feature_configs
        )

    def generate_sweep(
        self,
        prompt: str,
        feature_index: int,
        strengths: List[float],
        **gen_kwargs
    ) -> SweepResult:
        """Generate outputs at multiple steering strengths."""
        results = []

        for strength in strengths:
            config = [{"feature_index": feature_index, "strength": strength}]
            output = self.generate_steered(prompt, config, **gen_kwargs)
            results.append({
                "strength": strength,
                "output": output
            })

        return SweepResult(
            prompt=prompt,
            feature_index=feature_index,
            results=results
        )

    def calibrate_feature(self, feature_index: int) -> float:
        """Compute calibration factor for a feature.

        Based on Neuronpedia approach: normalize by decoder weight norm
        and activation standard deviation.
        """
        if feature_index in self._calibration_cache:
            return self._calibration_cache[feature_index]

        # Get decoder weight for this feature
        decoder_weight = self.sae.W_dec[:, feature_index]
        weight_norm = decoder_weight.norm().item()

        # Calibration: strength 1.0 should produce meaningful effect
        # Scale inversely with weight norm
        calibration = 1.0 / (weight_norm + 1e-6)

        self._calibration_cache[feature_index] = calibration
        return calibration

    def _generate_raw(self, prompt: str, **gen_kwargs) -> str:
        """Generate without any steering."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 100),
                temperature=gen_kwargs.get("temperature", 0.7),
                top_p=gen_kwargs.get("top_p", 0.9),
                top_k=gen_kwargs.get("top_k", 50),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
```

#### `backend/src/schemas/steering.py`
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class FeatureConfig(BaseModel):
    feature_index: int
    strength: float = Field(ge=-10, le=10)

class SteeringGenerateRequest(BaseModel):
    sae_id: str
    prompt: str
    features: List[FeatureConfig]
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

class SteeringGenerateResponse(BaseModel):
    output: str
    prompt: str

class SteeringCompareRequest(BaseModel):
    sae_id: str
    prompt: str
    features: List[FeatureConfig]
    max_new_tokens: int = 100
    temperature: float = 0.7

class SteeringCompareResponse(BaseModel):
    prompt: str
    baseline_output: str
    steered_output: str
    features: List[FeatureConfig]

class SteeringSweepRequest(BaseModel):
    sae_id: str
    prompt: str
    feature_index: int
    strengths: List[float] = [-5, -2.5, 0, 2.5, 5]
    max_new_tokens: int = 100

class SweepResultItem(BaseModel):
    strength: float
    output: str

class SteeringSweepResponse(BaseModel):
    prompt: str
    feature_index: int
    results: List[SweepResultItem]
```

#### `backend/src/api/v1/endpoints/steering.py`
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.db.session import get_db
from src.schemas.steering import (
    SteeringGenerateRequest,
    SteeringGenerateResponse,
    SteeringCompareRequest,
    SteeringCompareResponse,
    SteeringSweepRequest,
    SteeringSweepResponse
)
from src.services.steering_service import SteeringService
from src.services.sae_manager_service import SAEManagerService
from src.ml.sae_loader import load_sae
from src.ml.model_loader import ModelLoader

router = APIRouter()

def get_steering_service(sae_id: str, db: Session) -> SteeringService:
    """Load SAE and model, create steering service."""
    sae_manager = SAEManagerService(db)
    sae_info = sae_manager.get_sae_by_id(sae_id)

    if not sae_info:
        raise HTTPException(status_code=404, detail="SAE not found")

    # Load model and SAE
    model_loader = ModelLoader.get_instance()
    model = model_loader.load_model(sae_info["model_name"])
    tokenizer = model_loader.load_tokenizer(sae_info["model_name"])
    sae = load_sae(sae_info["local_path"])

    return SteeringService(model, tokenizer, sae, sae_info["layer"])

@router.post("/generate", response_model=SteeringGenerateResponse)
async def generate_steered(
    request: SteeringGenerateRequest,
    db: Session = Depends(get_db)
):
    """Generate text with steering applied."""
    service = get_steering_service(request.sae_id, db)

    output = service.generate_steered(
        prompt=request.prompt,
        feature_configs=[f.model_dump() for f in request.features],
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k
    )

    return SteeringGenerateResponse(output=output, prompt=request.prompt)

@router.post("/compare", response_model=SteeringCompareResponse)
async def compare_outputs(
    request: SteeringCompareRequest,
    db: Session = Depends(get_db)
):
    """Generate baseline and steered outputs for comparison."""
    service = get_steering_service(request.sae_id, db)

    result = service.generate_comparison(
        prompt=request.prompt,
        feature_configs=[f.model_dump() for f in request.features],
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature
    )

    return SteeringCompareResponse(
        prompt=result.prompt,
        baseline_output=result.baseline_output,
        steered_output=result.steered_output,
        features=request.features
    )

@router.post("/sweep", response_model=SteeringSweepResponse)
async def sweep_strengths(
    request: SteeringSweepRequest,
    db: Session = Depends(get_db)
):
    """Generate outputs at multiple steering strengths."""
    service = get_steering_service(request.sae_id, db)

    result = service.generate_sweep(
        prompt=request.prompt,
        feature_index=request.feature_index,
        strengths=request.strengths,
        max_new_tokens=request.max_new_tokens
    )

    return SteeringSweepResponse(
        prompt=result.prompt,
        feature_index=result.feature_index,
        results=[
            {"strength": r["strength"], "output": r["output"]}
            for r in result.results
        ]
    )
```

### 2.2 Frontend Components

#### `frontend/src/components/steering/StrengthSlider.tsx`
```typescript
import React from 'react';

interface StrengthSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

export function StrengthSlider({
  value,
  onChange,
  min = -10,
  max = 10,
  step = 0.5
}: StrengthSliderProps) {
  // Calculate percentage for gradient
  const percentage = ((value - min) / (max - min)) * 100;

  // Color gradient: red (negative) -> gray (zero) -> green (positive)
  const getColor = (val: number) => {
    if (val < 0) return 'rgb(239, 68, 68)';  // red-500
    if (val > 0) return 'rgb(34, 197, 94)';  // green-500
    return 'rgb(100, 116, 139)';  // slate-500
  };

  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-red-400 w-8">{min}</span>

      <div className="relative flex-1">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
          style={{
            background: `linear-gradient(to right,
              rgb(239, 68, 68) 0%,
              rgb(100, 116, 139) 50%,
              rgb(34, 197, 94) 100%)`
          }}
        />

        {/* Zero marker */}
        <div
          className="absolute top-4 transform -translate-x-1/2 text-xs text-slate-500"
          style={{ left: '50%' }}
        >
          0
        </div>
      </div>

      <span className="text-xs text-green-400 w-8 text-right">{max}</span>

      {/* Current value display */}
      <div
        className="w-16 text-center font-mono text-sm rounded px-2 py-1"
        style={{ backgroundColor: getColor(value) + '20', color: getColor(value) }}
      >
        {value > 0 ? '+' : ''}{value.toFixed(1)}
      </div>
    </div>
  );
}
```

#### `frontend/src/components/steering/SelectedFeatureCard.tsx`
```typescript
import React from 'react';
import { X, Info } from 'lucide-react';
import { StrengthSlider } from './StrengthSlider';
import { Feature } from '../../types/features';

interface SelectedFeatureCardProps {
  feature: Feature;
  strength: number;
  onStrengthChange: (strength: number) => void;
  onRemove: () => void;
  onViewDetails: () => void;
}

export function SelectedFeatureCard({
  feature,
  strength,
  onStrengthChange,
  onRemove,
  onViewDetails
}: SelectedFeatureCardProps) {
  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-slate-400">#{feature.feature_index}</span>
            {feature.semantic_label && (
              <span className="text-emerald-400">{feature.semantic_label}</span>
            )}
          </div>
          {feature.category_label && (
            <div className="text-xs text-slate-500 mt-1">
              {feature.category_label}
            </div>
          )}
        </div>

        <div className="flex gap-1">
          <button
            onClick={onViewDetails}
            className="p-1 hover:bg-slate-700 rounded"
            title="View details"
          >
            <Info className="w-4 h-4 text-slate-400" />
          </button>
          <button
            onClick={onRemove}
            className="p-1 hover:bg-slate-700 rounded"
            title="Remove"
          >
            <X className="w-4 h-4 text-slate-400" />
          </button>
        </div>
      </div>

      <div className="mt-2">
        <div className="text-xs text-slate-400 mb-1">Steering Strength</div>
        <StrengthSlider value={strength} onChange={onStrengthChange} />
      </div>

      {/* Top tokens preview */}
      {feature.top_tokens && feature.top_tokens.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {feature.top_tokens.slice(0, 5).map((t, i) => (
            <span
              key={i}
              className="text-xs bg-slate-700/50 px-2 py-0.5 rounded"
            >
              {t.token}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
```

#### `frontend/src/components/steering/ComparisonResults.tsx`
```typescript
import React from 'react';
import { Download, Copy } from 'lucide-react';

interface ComparisonResultsProps {
  prompt: string;
  baseline: string;
  steered: string;
  features: Array<{ feature_index: number; strength: number }>;
  onExport: () => void;
}

export function ComparisonResults({
  prompt,
  baseline,
  steered,
  features,
  onExport
}: ComparisonResultsProps) {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  // Simple diff highlighting (words that differ)
  const highlightDiff = (text1: string, text2: string, isSecond: boolean) => {
    const words1 = text1.split(/\s+/);
    const words2 = text2.split(/\s+/);
    const targetWords = isSecond ? words2 : words1;
    const compareWords = isSecond ? words1 : words2;

    return targetWords.map((word, i) => {
      const isDifferent = compareWords[i] !== word;
      return (
        <span
          key={i}
          className={isDifferent ? 'bg-emerald-500/20 text-emerald-300' : ''}
        >
          {word}{' '}
        </span>
      );
    });
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h3 className="font-medium">Results</h3>
        <button
          onClick={onExport}
          className="flex items-center gap-2 px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-sm"
        >
          <Download className="w-4 h-4" />
          Export JSON
        </button>
      </div>

      {/* Prompt */}
      <div className="bg-slate-800 rounded p-3">
        <div className="text-xs text-slate-400 mb-1">Prompt</div>
        <div className="text-slate-300">{prompt}</div>
      </div>

      {/* Side by side comparison */}
      <div className="grid grid-cols-2 gap-4">
        {/* Baseline */}
        <div className="bg-slate-800 rounded p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-slate-400">Baseline</span>
            <button
              onClick={() => copyToClipboard(baseline)}
              className="p-1 hover:bg-slate-700 rounded"
            >
              <Copy className="w-4 h-4 text-slate-500" />
            </button>
          </div>
          <div className="text-slate-300 whitespace-pre-wrap">
            {highlightDiff(baseline, steered, false)}
          </div>
        </div>

        {/* Steered */}
        <div className="bg-slate-800 rounded p-4 border border-emerald-500/30">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-emerald-400">
              Steered
              {features.length > 0 && (
                <span className="text-slate-400 font-normal ml-2">
                  ({features.map(f =>
                    `#${f.feature_index}: ${f.strength > 0 ? '+' : ''}${f.strength}`
                  ).join(', ')})
                </span>
              )}
            </span>
            <button
              onClick={() => copyToClipboard(steered)}
              className="p-1 hover:bg-slate-700 rounded"
            >
              <Copy className="w-4 h-4 text-slate-500" />
            </button>
          </div>
          <div className="text-slate-300 whitespace-pre-wrap">
            {highlightDiff(baseline, steered, true)}
          </div>
        </div>
      </div>
    </div>
  );
}
```

#### `frontend/src/stores/steeringStore.ts`
```typescript
import { create } from 'zustand';
import { Feature } from '../types/features';
import { steeringApi } from '../api/steering';

interface SelectedFeature {
  feature: Feature;
  strength: number;
}

interface SteeringState {
  // Selected SAE
  selectedSaeId: string | null;

  // Selected features for steering
  selectedFeatures: SelectedFeature[];

  // Prompt
  prompt: string;

  // Generation config
  maxNewTokens: number;
  temperature: number;
  topP: number;
  topK: number;

  // Results
  baselineOutput: string | null;
  steeredOutput: string | null;
  loading: boolean;

  // Actions
  setSaeId: (id: string) => void;
  addFeature: (feature: Feature) => void;
  removeFeature: (featureIndex: number) => void;
  setFeatureStrength: (featureIndex: number, strength: number) => void;
  setPrompt: (prompt: string) => void;
  setGenerationConfig: (config: Partial<{
    maxNewTokens: number;
    temperature: number;
    topP: number;
    topK: number;
  }>) => void;
  generate: () => Promise<void>;
  compare: () => Promise<void>;
  clear: () => void;
}

export const useSteeringStore = create<SteeringState>((set, get) => ({
  selectedSaeId: null,
  selectedFeatures: [],
  prompt: '',
  maxNewTokens: 100,
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  baselineOutput: null,
  steeredOutput: null,
  loading: false,

  setSaeId: (id) => set({ selectedSaeId: id }),

  addFeature: (feature) => {
    const { selectedFeatures } = get();
    if (selectedFeatures.some(sf => sf.feature.feature_index === feature.feature_index)) {
      return; // Already added
    }
    set({
      selectedFeatures: [...selectedFeatures, { feature, strength: 1.0 }]
    });
  },

  removeFeature: (featureIndex) => {
    set(state => ({
      selectedFeatures: state.selectedFeatures.filter(
        sf => sf.feature.feature_index !== featureIndex
      )
    }));
  },

  setFeatureStrength: (featureIndex, strength) => {
    set(state => ({
      selectedFeatures: state.selectedFeatures.map(sf =>
        sf.feature.feature_index === featureIndex
          ? { ...sf, strength }
          : sf
      )
    }));
  },

  setPrompt: (prompt) => set({ prompt }),

  setGenerationConfig: (config) => set(config),

  generate: async () => {
    const { selectedSaeId, selectedFeatures, prompt, maxNewTokens, temperature, topP, topK } = get();
    if (!selectedSaeId || !prompt) return;

    set({ loading: true });
    try {
      const result = await steeringApi.generate({
        sae_id: selectedSaeId,
        prompt,
        features: selectedFeatures.map(sf => ({
          feature_index: sf.feature.feature_index,
          strength: sf.strength
        })),
        max_new_tokens: maxNewTokens,
        temperature,
        top_p: topP,
        top_k: topK
      });
      set({ steeredOutput: result.output });
    } finally {
      set({ loading: false });
    }
  },

  compare: async () => {
    const { selectedSaeId, selectedFeatures, prompt, maxNewTokens, temperature } = get();
    if (!selectedSaeId || !prompt) return;

    set({ loading: true });
    try {
      const result = await steeringApi.compare({
        sae_id: selectedSaeId,
        prompt,
        features: selectedFeatures.map(sf => ({
          feature_index: sf.feature.feature_index,
          strength: sf.strength
        })),
        max_new_tokens: maxNewTokens,
        temperature
      });
      set({
        baselineOutput: result.baseline_output,
        steeredOutput: result.steered_output
      });
    } finally {
      set({ loading: false });
    }
  },

  clear: () => set({
    selectedFeatures: [],
    prompt: '',
    baselineOutput: null,
    steeredOutput: null
  })
}));
```

---

## 3. Common Patterns

### 3.1 Multi-Feature Steering (Isolated Mode - Implemented)
```python
# Current: Generate separate output per feature
def apply_multi_feature_steering(hidden_states, sae, configs):
    features = sae.encode(hidden_states)

    for config in configs:
        features[:, config.feature_index] += config.strength

    return sae.decode(features)
```

### 3.2 Combined Multi-Feature Steering (Planned)
```python
# Planned: Apply all features together in single generation
class CombinedSteeringHook:
    """
    Pre-computes combined steering vector from multiple features
    for efficient single-pass generation.
    """

    def __init__(self, sae, feature_configs, calibration_factors):
        self.sae = sae
        self.feature_configs = feature_configs
        self.calibration_factors = calibration_factors

        # Pre-compute combined steering direction
        self.combined_steering = self._compute_combined_vector()

    def _compute_combined_vector(self):
        """
        Sum all feature steering directions into one vector.

        Mathematical basis:
        - Each feature's steering direction is W_dec[feature_idx, :]
        - Combined steering = Σ (strength_i × calibration_i × direction_i)
        """
        W_dec = self.sae.W_dec
        device = W_dec.device
        d_model = W_dec.shape[-1]

        combined = torch.zeros(d_model, device=device)

        for config in self.feature_configs:
            idx = config.feature_index
            strength = config.strength
            calibration = self.calibration_factors.get(idx, 1.0)

            # Direction from SAE decoder
            direction = W_dec[idx, :]

            # Accumulate with scaling
            combined += strength * calibration * direction

        return combined

    def __call__(self, module, input, output):
        """Hook function - adds combined steering to residual."""
        hidden = output[0] if isinstance(output, tuple) else output

        # Broadcast add: [batch, seq, d_model] + [d_model]
        modified = hidden + self.combined_steering

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


# Usage in steering service
async def generate_combined(self, request):
    # Create combined hook
    hook = CombinedSteeringHook(
        sae=self.sae,
        feature_configs=request.features,
        calibration_factors=self._get_all_calibrations(request.features)
    )

    # Register and generate
    handle = hook.register(self.model, self.layer)
    try:
        output = self._generate_raw(request.prompt, **request.gen_config)
        return CombinedResult(
            combined_output=output,
            features_applied=request.features
        )
    finally:
        handle.remove()
```

**Key Implementation Notes:**

1. **Vector Accumulation**: Steering directions are summed, not applied sequentially
2. **Pre-computation**: Combined vector computed once before generation starts
3. **Direct Modification**: No SAE encode/decode during inference - faster
4. **Calibration**: Each feature scaled by its calibration factor

### 3.2 Streaming Generation
```python
# For long outputs, stream tokens
from transformers import TextIteratorStreamer
from threading import Thread

def generate_streaming(model, tokenizer, prompt, steering_service):
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    thread = Thread(target=steering_service.generate_steered, kwargs={
        "prompt": prompt,
        "streamer": streamer
    })
    thread.start()

    for token in streamer:
        yield token
```

---

## 4. Testing Strategy

### 4.1 Steering Service Tests
```python
# backend/tests/test_steering_service.py
import torch
import pytest
from src.services.steering_service import SteeringService
from src.ml.sparse_autoencoder import StandardSAE

def test_generate_steered():
    # Create mock model and SAE
    sae = StandardSAE(d_in=256, d_sae=1024)
    # ... setup mock model

    service = SteeringService(mock_model, mock_tokenizer, sae, layer=12)

    output = service.generate_steered(
        prompt="Hello",
        feature_configs=[{"feature_index": 0, "strength": 1.0}],
        max_new_tokens=10
    )

    assert isinstance(output, str)
    assert len(output) > 0

def test_comparison_mode():
    service = SteeringService(mock_model, mock_tokenizer, sae, layer=12)

    result = service.generate_comparison(
        prompt="Test",
        feature_configs=[{"feature_index": 0, "strength": 2.0}]
    )

    assert result.baseline_output != result.steered_output
```

---

## 5. Common Pitfalls

### Pitfall 1: Hook Registration Order
```python
# WRONG - Hooks persist across calls
service.generate_steered(prompt1, config1)
service.generate_steered(prompt2, config2)  # Has BOTH configs!

# RIGHT - Clear hooks each time
def generate_steered(self, ...):
    self.hook_manager.clear_hooks()  # Clear first
    self.hook_manager.register_hooks([self.layer])
    try:
        # generate
    finally:
        self.hook_manager.clear_hooks()  # Clear after
```

### Pitfall 2: Strength Calibration
```python
# WRONG - Raw strength values
features[:, idx] += strength  # May be too weak or strong

# RIGHT - Calibrated strength
calibration = self.calibrate_feature(idx)
features[:, idx] += strength * calibration
```

### Pitfall 3: Memory Accumulation
```python
# WRONG - Storing activations in hook
def hook(module, input, output):
    self.all_activations.append(output)  # Memory leak!

# RIGHT - Process immediately, don't store
def hook(module, input, output):
    return self.process_and_return(output)
```

### Pitfall 4: Combined Mode Feature Conflicts (Planned)
```python
# POTENTIAL ISSUE - Opposing features may cancel out
features = [
    {"feature_index": 42, "strength": +5},   # "formal language"
    {"feature_index": 99, "strength": +5},   # "casual language"
]
# These may partially cancel, producing weak or unpredictable effect

# RECOMMENDED - Start with complementary features
features = [
    {"feature_index": 42, "strength": +3},   # "formal language"
    {"feature_index": 156, "strength": +2},  # "positive sentiment"
]
# Complementary features create more predictable combined effects

# UI HINT - Show warning when potentially conflicting features selected
def check_feature_conflicts(features):
    # Check category labels for potential conflicts
    categories = [f.category_label for f in features]
    if has_opposing_categories(categories):
        return "Warning: Selected features may have opposing effects"
    return None
```

---

## 6. Performance Tips

1. **Batch Multiple Prompts**
   ```python
   # Process multiple prompts in one forward pass
   inputs = tokenizer(prompts, padding=True, return_tensors="pt")
   outputs = model.generate(**inputs)
   ```

2. **Cache Model and SAE**
   ```python
   # Don't reload for each request
   _cached_service = None

   def get_steering_service(sae_id):
       global _cached_service
       if _cached_service is None or _cached_service.sae_id != sae_id:
           _cached_service = SteeringService(...)
       return _cached_service
   ```

3. **Use KV Cache**
   ```python
   # Enable KV cache for faster generation
   outputs = model.generate(
       **inputs,
       use_cache=True
   )
   ```

---

*Related: [PRD](../prds/006_FPRD|Model_Steering.md) | [TDD](../tdds/006_FTDD|Model_Steering.md) | [FTASKS](../tasks/006_FTASKS|Model_Steering.md)*
