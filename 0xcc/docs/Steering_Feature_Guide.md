# Feature Steering Guide

This guide covers the Feature Steering functionality in miStudio, which allows you to modify model behavior by adjusting the activation strength of specific SAE features during text generation.

## Overview

Feature steering enables:
- **Behavior modification**: Increase or decrease specific behaviors in model outputs
- **Comparative analysis**: Compare steered vs unsteered outputs side-by-side
- **Strength exploration**: Sweep through different steering strengths
- **Experiment saving**: Save and replay steering configurations

## Core Concepts

### Feature Selection
Select up to 4 features simultaneously for steering. Each feature is assigned a unique color (teal, blue, purple, amber) for visual distinction.

### Steering Strength
Strength is measured from -100 to +300:
- **Negative values** (-100 to 0): Suppress the feature behavior
- **Zero** (0): No modification (baseline)
- **Positive values** (0 to +300): Amplify the feature behavior

**Multiplier Formula:** `multiplier = 1 + strength/100`
- Strength 50 = 1.5x activation
- Strength 100 = 2x activation
- Strength -50 = 0.5x activation

### Strength Presets
Quick presets for common use cases:
- **Subtle** (10): Gentle modification
- **Moderate** (50): Noticeable effect
- **Strong** (100): Strong behavioral shift

## API Endpoints

### Generate Comparison
```http
POST /api/v1/steering/compare
Content-Type: application/json

{
  "sae_id": "sae_abc123",
  "prompt": "Once upon a time",
  "selected_features": [
    {
      "feature_idx": 1234,
      "layer": 6,
      "strength": 50,
      "label": "storytelling"
    }
  ],
  "generation_params": {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "num_samples": 1
  },
  "include_unsteered": true,
  "compute_metrics": false
}
```

**Response:**
```json
{
  "comparison_id": "comp_abc123",
  "sae_id": "sae_abc123",
  "model_id": "model_xyz",
  "prompt": "Once upon a time",
  "unsteered": {
    "text": "Once upon a time there was...",
    "metrics": null
  },
  "steered": [
    {
      "text": "Once upon a time in a magical kingdom...",
      "feature_config": {...},
      "metrics": null
    }
  ],
  "total_time_ms": 1234,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Strength Sweep
```http
POST /api/v1/steering/sweep
Content-Type: application/json

{
  "sae_id": "sae_abc123",
  "prompt": "The weather today is",
  "feature_idx": 1234,
  "layer": 6,
  "strength_values": [0, 25, 50, 75, 100, 150, 200],
  "generation_params": {
    "max_new_tokens": 50,
    "temperature": 0.7
  }
}
```

### Save Experiment
```http
POST /api/v1/steering/experiments
Content-Type: application/json

{
  "name": "Storytelling Enhancement",
  "description": "Testing story-related features",
  "comparison_id": "comp_abc123",
  "tags": ["storytelling", "gpt2"]
}
```

### List Experiments
```http
GET /api/v1/steering/experiments?skip=0&limit=20&search=storytelling
```

## Rate Limiting

Steering endpoints are rate-limited to prevent resource exhaustion:
- **Limit**: 5 requests per minute per client
- **Response on limit**: HTTP 429 with `Retry-After` header

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 45
Content-Type: application/json

{
  "detail": "Rate limit exceeded. Try again in 45 seconds."
}
```

## Request Timeout

Steering requests have a 30-second timeout:
- Generation starts immediately
- If not complete in 30s, request is aborted
- Error response: HTTP 504 Gateway Timeout

## Frontend Components

### SteeringPanel
Main orchestrator component providing:
- SAE selection dropdown
- Feature browser integration
- Prompt input
- Generation configuration
- Comparison results display

### FeatureSelector
Manages selected features with:
- Feature cards with color indicators
- Strength sliders (-100 to +300)
- Strength presets (Subtle/Moderate/Strong)
- Drag-and-drop reordering
- Remove feature button

### ComparisonPreview
Visual preview of steering configuration:
- Shows unsteered baseline card
- Shows each selected feature card
- Displays feature color, name, strength, and layer

### GenerationConfig
Configure generation parameters:
- Max tokens (10-500)
- Temperature (0-2)
- Top-p (0-1)
- Top-k (1-100)
- Number of samples
- Advanced options (repetition penalty, etc.)

## steeringStore Actions

```typescript
const {
  selectSAE,
  addFeature,
  removeFeature,
  updateFeatureStrength,
  applyStrengthPreset,
  setPrompt,
  setGenerationParams,
  generateComparison,
  abortComparison,
  runStrengthSweep,
  saveExperiment,
  loadExperiment,
} = useSteeringStore();

// Select an SAE for steering
selectSAE(sae);

// Add a feature (auto-assigns color)
const added = addFeature({
  feature_idx: 1234,
  layer: 6,
  strength: 50,
  label: 'storytelling'
});

// Apply preset strength to all features
applyStrengthPreset(100); // Strong

// Generate comparison
const result = await generateComparison(true, false);
```

## Selectors

```typescript
import {
  selectCanGenerate,
  selectFeature,
  selectAvailableColors
} from '../stores/steeringStore';

// Check if ready to generate
const canGenerate = selectCanGenerate(state);

// Find specific feature
const feature = selectFeature(1234, 6)(state);

// Get unused colors
const colors = selectAvailableColors(state);
```

## Strength Warning Levels

Visual feedback for extreme strength values:

| Range | Level | Indicator |
|-------|-------|-----------|
| -50 to 150 | Normal | None |
| -80 to -50 or 150 to 250 | Caution | Yellow warning |
| < -80 or > 250 | Extreme | Red warning |

## Best Practices

### Feature Selection
1. Start with single feature to understand its effect
2. Add complementary features gradually
3. Avoid conflicting features (e.g., formal + informal)

### Strength Tuning
1. Start with Subtle (10) preset
2. Gradually increase if effect is too weak
3. Use strength sweep to find optimal value
4. Watch for quality degradation at high strengths

### Prompt Design
1. Use prompts that allow the feature to manifest
2. Longer prompts provide more context
3. Test multiple prompt styles

### Experiment Management
1. Save successful configurations as experiments
2. Use descriptive names and tags
3. Document what works and what doesn't

## Troubleshooting

### No visible steering effect
- Check that SAE is loaded on correct model layer
- Increase steering strength
- Verify feature is relevant to prompt topic

### Quality degradation
- Reduce steering strength
- Try fewer simultaneous features
- Use lower temperature for consistency

### Rate limit errors
- Wait for cooldown period
- Batch related generations
- Use experiments to replay results

### Timeout errors
- Reduce max_new_tokens
- Simplify feature configuration
- Check GPU availability
