# Technical Implementation Document: Feature Discovery

**Document ID:** 004_FTID|Feature_Discovery
**Feature:** Feature Extraction and Analysis
**PRD Reference:** 004_FPRD|Feature_Discovery.md
**TDD Reference:** 004_FTDD|Feature_Discovery.md
**Status:** Ready for Implementation
**Created:** 2025-10-06

---

## 1. Implementation Overview

Implements SAE feature extraction, max-activating example identification, token-level highlighting, and analysis tools (logit lens, correlations, ablation). Mock UI lines 2159-2800 is PRIMARY reference.

**Key Components:**
- Feature Extraction Pipeline (SAE encoder inference on eval samples)
- Top-K Max-Activating Examples (store only top 100 per feature to save space: ~3GB vs ~50GB)
- Token Highlighting (intensity-based gradient, CSS inline styles)
- Analysis Tools (logit lens, feature correlations, ablation experiments)
- Auto-Labeling (pattern matching heuristics)

---

## 2. Feature Extraction Implementation

**File:** `backend/src/workers/feature_extraction_tasks.py`

```python
from celery import Task
import torch
from safetensors.torch import load_file
from src.core.celery_app import celery_app

@celery_app.task(bind=True)
def extract_features_task(self: Task, training_id: str, config: dict):
    """
    Extract features from trained SAE.

    Steps:
    1. Load SAE checkpoint
    2. Load eval dataset samples
    3. For each sample:
       - Pass through model to get activations
       - Pass activations through SAE encoder
       - Record feature activations per token
    4. For each feature (neuron):
       - Identify top-K max-activating examples
       - Calculate activation frequency
       - Auto-generate label
       - Store to database
    """
    import asyncio

    async def run_extraction():
        async with AsyncSessionLocal() as db:
            service = FeatureExtractionService(db)

            try:
                # 1. Load SAE
                training = await service.get_training(training_id)
                sae_checkpoint = load_file(training.final_checkpoint_path)
                sae = reconstruct_sae_from_checkpoint(sae_checkpoint)
                sae = sae.cuda().eval()

                # Load model
                model, tokenizer = load_model(training.model_id)

                # 2. Load eval samples
                dataset = load_dataset(training.dataset_id)
                eval_samples = dataset[:config.get('max_samples', 1000)]

                # 3. Extract activations per feature
                feature_activations = {}  # {feature_id: [(sample_idx, tokens, activations, max_act), ...]}

                for sample_idx, sample in enumerate(eval_samples):
                    input_ids = tokenizer.encode(sample['text'], return_tensors='pt').cuda()

                    # Extract model activations
                    with torch.no_grad():
                        outputs = model(input_ids, output_hidden_states=True)
                        activations = outputs.hidden_states[training.layer_idx]  # [1, seq_len, hidden_dim]

                        # Pass through SAE encoder
                        latent = torch.relu(sae.encoder(activations))  # [1, seq_len, latent_dim]

                    # Record per-feature activations
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                    for feature_id in range(latent.shape[-1]):
                        feature_acts = latent[0, :, feature_id].cpu().numpy()  # [seq_len]
                        max_act = float(feature_acts.max())

                        if max_act > 0:  # Only store if feature activated
                            if feature_id not in feature_activations:
                                feature_activations[feature_id] = []

                            feature_activations[feature_id].append({
                                'sample_idx': sample_idx,
                                'tokens': tokens,
                                'activations': feature_acts.tolist(),
                                'max_activation': max_act
                            })

                # 4. Process each feature
                for feature_id, examples in feature_activations.items():
                    # Sort by max activation, keep top-K
                    examples.sort(key=lambda x: x['max_activation'], reverse=True)
                    top_k = examples[:100]  # Store only top 100

                    # Calculate activation frequency
                    activation_freq = len(examples) / len(eval_samples)

                    # Auto-generate label
                    label = auto_label_feature(top_k)

                    # Save feature to database
                    feature = await service.create_feature(
                        training_id=training_id,
                        neuron_index=feature_id,
                        name=label,
                        activation_frequency=activation_freq,
                        max_activation_value=top_k[0]['max_activation']
                    )

                    # Save top-K examples
                    for example in top_k:
                        await service.create_feature_activation(
                            feature_id=feature.id,
                            tokens=example['tokens'],
                            activations=example['activations'],
                            max_activation=example['max_activation'],
                            sample_index=example['sample_idx']
                        )

                # Update job status
                await service.update_extraction_job(training_id, 'completed')

            except Exception as e:
                await service.update_extraction_job(training_id, 'error', str(e))

    asyncio.run(run_extraction())

def auto_label_feature(top_examples):
    """
    Auto-generate feature label using pattern matching.

    Heuristics:
    - If high-activation tokens are all punctuation → "Punctuation"
    - If tokens contain question words → "Question Pattern"
    - If tokens are uppercase → "Capitalization"
    - Default: "Feature {feature_id}"
    """
    import string

    # Extract high-activation tokens (activation > 0.7 * max)
    all_tokens = []
    for ex in top_examples[:5]:
        max_act = ex['max_activation']
        high_tokens = [
            ex['tokens'][i]
            for i, act in enumerate(ex['activations'])
            if act > 0.7 * max_act
        ]
        all_tokens.extend(high_tokens)

    if not all_tokens:
        return "Unknown Feature"

    # Pattern matching
    if all(t in string.punctuation for t in all_tokens[:10]):
        return "Punctuation"

    if any(t.lower() in ['what', 'how', 'why', 'when', 'where', 'who'] for t in all_tokens):
        return "Question Pattern"

    if all(t.isupper() for t in all_tokens[:5] if t.isalpha()):
        return "Uppercase/Capitalization"

    if all(t.isdigit() for t in all_tokens[:5]):
        return "Numerical"

    # Default
    return f"Feature {top_examples[0].get('feature_id', 'Unknown')}"
```

**Storage Optimization:**
- Store only top-100 examples per feature
- Use JSONB for tokens/activations (PostgreSQL native type, indexed)
- Partition `feature_activations` table by `feature_id` ranges
- Result: ~3GB per training vs ~50GB if storing all activations

---

## 3. Token Highlighting Implementation

**File:** `frontend/src/components/features/MaxActivatingExamples.tsx`

**PRIMARY REFERENCE:** Mock UI lines 2728-2800

```typescript
import React from 'react';

interface TokenHighlightProps {
  tokens: string[];
  activations: number[];
}

export const TokenHighlight: React.FC<TokenHighlightProps> = ({ tokens, activations }) => {
  const maxActivation = Math.max(...activations);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 font-mono text-sm whitespace-pre-wrap">
      {tokens.map((token, idx) => {
        const activation = activations[idx];
        const intensity = activation / maxActivation;  // Normalize 0-1

        // Color gradient based on intensity
        const backgroundColor = `rgba(16, 185, 129, ${intensity * 0.4})`;  // emerald-500 with opacity
        const textColor = intensity > 0.6 ? '#fff' : '#cbd5e1';  // white or slate-300
        const border = intensity > 0.7 ? '1px solid rgba(16, 185, 129, 0.5)' : 'none';

        return (
          <span
            key={idx}
            className="px-1 py-0.5 rounded relative group cursor-help"
            style={{
              backgroundColor,
              color: textColor,
              border
            }}
            title={`Activation: ${activation.toFixed(3)}`}
          >
            {token}
            {/* Tooltip on hover */}
            <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-slate-800 text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
              {activation.toFixed(3)}
            </span>
          </span>
        );
      })}
    </div>
  );
};
```

**Styling Formula:**
- `intensity = activation / max_activation` (0 to 1)
- `backgroundColor = rgba(16, 185, 129, intensity * 0.4)` (emerald with 0-40% opacity)
- `textColor = intensity > 0.6 ? white : slate-300` (high contrast)
- `border = intensity > 0.7 ? emerald border : none` (emphasize strongest)

---

## 4. Feature Browser Component

**File:** `frontend/src/components/panels/FeaturesPanel.tsx`

**PRIMARY REFERENCE:** Mock UI lines 2159-2584

```typescript
import React, { useState, useEffect } from 'react';
import { Search, Filter, Star, Eye, EyeOff } from 'lucide-react';
import { useFeaturesStore } from '@/stores/featuresStore';

export const FeaturesPanel: React.FC = () => {
  const features = useFeaturesStore((state) => state.features);
  const trainings = useTrainingsStore((state) => state.trainings.filter(t => t.status === 'completed'));

  const [selectedTraining, setSelectedTraining] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('activation_freq');
  const [filterFavorites, setFilterFavorites] = useState(false);

  // Fetch features when training selected
  useEffect(() => {
    if (selectedTraining) {
      fetchFeatures(selectedTraining.id);
    }
  }, [selectedTraining]);

  // Client-side filtering and sorting
  const filteredFeatures = features
    .filter(f => filterFavorites ? f.is_favorite : true)
    .filter(f => !f.is_hidden)
    .filter(f => f.name?.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'activation_freq') return b.activation_frequency - a.activation_frequency;
      if (sortBy === 'interpretability') return b.interpretability_score - a.interpretability_score;
      return 0;
    });

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Feature Discovery</h2>

      {/* Training Selector */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
        <label className="block text-sm font-medium text-slate-300 mb-2">Select Training</label>
        <select value={selectedTraining?.id || ''} onChange={(e) => setSelectedTraining(trainings.find(t => t.id === e.target.value))}
          className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500">
          <option value="">Choose a training...</option>
          {trainings.map(t => (
            <option key={t.id} value={t.id}>{t.name} ({t.num_features} features)</option>
          ))}
        </select>
      </div>

      {/* Feature Browser */}
      {selectedTraining && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
          {/* Stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-800/50 p-3 rounded-lg">
              <div className="text-xs text-slate-400">Total Features</div>
              <div className="text-2xl font-semibold text-emerald-400">{features.length}</div>
            </div>
            {/* Active Features, Avg Activation Freq */}
          </div>

          {/* Search & Filters */}
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search features..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              />
            </div>
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}
              className="px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500">
              <option value="activation_freq">Activation Frequency</option>
              <option value="interpretability">Interpretability Score</option>
            </select>
            <button onClick={() => setFilterFavorites(!filterFavorites)}
              className={`px-4 py-2 rounded-lg ${filterFavorites ? 'bg-emerald-600' : 'bg-slate-800'}`}>
              <Star className="w-4 h-4 inline" />
            </button>
          </div>

          {/* Feature Table */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-800/50 border-b border-slate-700">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-slate-300">Feature</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-slate-300">Label</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-slate-300">Activation Freq</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-slate-300">Example</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-slate-300">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredFeatures.map(feature => (
                  <tr key={feature.id} className="border-b border-slate-800 hover:bg-slate-800/30 cursor-pointer">
                    <td className="px-4 py-3 font-mono text-sm text-slate-400">#{feature.neuron_index}</td>
                    <td className="px-4 py-3">{feature.name}</td>
                    <td className="px-4 py-3">{(feature.activation_frequency * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3">
                      {/* Token highlighting preview (first 10 tokens) */}
                    </td>
                    <td className="px-4 py-3">
                      <button className="text-yellow-400 hover:text-yellow-300">
                        <Star className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
```

---

## 5. Analysis Tools

### Logit Lens Analysis

**File:** `backend/src/services/analysis_service.py`

```python
import torch

async def compute_logit_lens(feature_id: int, model_id: str, top_k: int = 10):
    """
    Logit lens: Project feature activation to vocabulary.

    Steps:
    1. Get feature's typical activation vector
    2. Pass through model's unembedding layer
    3. Return top-K predicted tokens
    """
    # Load feature
    feature = await get_feature(feature_id)

    # Load model
    model, tokenizer = load_model(model_id)

    # Get typical activation (average of top-10 max-activating examples)
    examples = await get_top_examples(feature_id, limit=10)
    typical_activation = torch.stack([
        torch.tensor(ex.activations).max()  # Max activation in each example
        for ex in examples
    ]).mean()

    # Reconstruct activation vector (simplified - use SAE decoder)
    sae = load_sae_from_training(feature.training_id)
    feature_vector = torch.zeros(sae.latent_dim)
    feature_vector[feature.neuron_index] = typical_activation
    reconstructed = sae.decoder(feature_vector.unsqueeze(0))  # [1, hidden_dim]

    # Pass through unembedding
    logits = model.lm_head(reconstructed)  # [1, vocab_size]
    probs = torch.softmax(logits, dim=-1)[0]

    # Get top-K
    top_probs, top_indices = torch.topk(probs, k=top_k)
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

    return {
        "tokens": top_tokens,
        "probabilities": top_probs.tolist()
    }
```

---

**Document End**
**Status:** Ready for Task Generation
**Estimated Size:** ~25KB
**Next:** 005_FTID|Model_Steering.md (final TID)
