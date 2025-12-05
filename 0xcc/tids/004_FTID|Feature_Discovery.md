# Technical Implementation Document: Feature Discovery

**Document ID:** 004_FTID|Feature_Discovery
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related TDD:** [004_FTDD|Feature_Discovery](../tdds/004_FTDD|Feature_Discovery.md)

---

## 1. Implementation Order

### Phase 1: Extraction Infrastructure
1. Database migrations (features, extraction_jobs)
2. Feature model and schemas
3. Extraction service
4. Celery extraction task

### Phase 2: Feature Analysis
1. Token aggregation service
2. Feature statistics computation
3. Correlation analysis
4. Feature browser backend

### Phase 3: Labeling System
1. Labeling job model
2. OpenAI/Anthropic labeling service
3. Labeling prompt templates
4. Batch labeling worker

### Phase 4: Frontend
1. Feature browser component
2. Feature detail modal
3. Token highlighting
4. Labeling interface

---

## 2. File-by-File Implementation

### 2.1 Backend - Extraction

#### `backend/src/models/feature.py`
```python
from sqlalchemy import Column, String, Integer, Float, ForeignKey, TIMESTAMP, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
import uuid
from src.db.base import Base

class Feature(Base):
    __tablename__ = "features"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sae_id = Column(UUID(as_uuid=True), nullable=False)  # training_id or external_sae_id
    feature_index = Column(Integer, nullable=False)

    # Labels (dual system)
    semantic_label = Column(String(500))
    category_label = Column(String(255))

    # Statistics
    activation_frequency = Column(Float)
    max_activation = Column(Float)
    mean_activation = Column(Float)
    std_activation = Column(Float)

    # Top tokens (denormalized for fast access)
    top_tokens = Column(JSONB)  # [{token, count, mean_activation}, ...]

    # Metadata
    extraction_job_id = Column(UUID(as_uuid=True), ForeignKey("extraction_jobs.id"))
    labeling_job_id = Column(UUID(as_uuid=True), ForeignKey("labeling_jobs.id"))

    created_at = Column(TIMESTAMP, server_default="now()")
    updated_at = Column(TIMESTAMP, onupdate="now()")

    # Relationships
    activations = relationship("FeatureActivation", back_populates="feature")


class FeatureActivation(Base):
    """Individual activation instances for a feature."""
    __tablename__ = "feature_activations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    feature_id = Column(UUID(as_uuid=True), ForeignKey("features.id", ondelete="CASCADE"))

    token = Column(String(100))
    token_id = Column(Integer)
    activation_value = Column(Float, nullable=False)

    # Context
    context_before = Column(Text)
    context_after = Column(Text)

    # Source tracking
    sample_index = Column(Integer)
    token_position = Column(Integer)

    created_at = Column(TIMESTAMP, server_default="now()")

    feature = relationship("Feature", back_populates="activations")
```

**Key Implementation Notes:**
- Dual label system: `semantic_label` (AI-generated) and `category_label` (hierarchical)
- `top_tokens` denormalized in JSONB for fast feature browser queries
- Separate `FeatureActivation` table for detailed activation data

#### `backend/src/services/extraction_service.py`
```python
import torch
import numpy as np
from typing import List, Dict, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from collections import defaultdict

from src.models.feature import Feature, FeatureActivation
from src.models.extraction_job import ExtractionJob
from src.ml.sparse_autoencoder import BaseSAE
from src.ml.activation_extraction import ActivationExtractor

class ExtractionService:
    def __init__(self, db: Session):
        self.db = db

    def create_job(
        self,
        sae_id: UUID,
        dataset_id: UUID,
        config: dict
    ) -> ExtractionJob:
        job = ExtractionJob(
            sae_id=sae_id,
            dataset_id=dataset_id,
            config=config,
            status="pending"
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def extract_features(
        self,
        sae: BaseSAE,
        activations: torch.Tensor,
        tokenizer,
        input_ids: torch.Tensor,
        job_id: UUID,
        top_k: int = 50,
        threshold: float = 0.0,
        progress_callback=None
    ) -> List[Feature]:
        """Extract features from activations through SAE."""
        device = next(sae.parameters()).device
        activations = activations.to(device)

        # Encode through SAE
        with torch.no_grad():
            # Shape: [batch, seq_len, d_sae]
            z = sae.encode(activations)

        d_sae = z.shape[-1]
        features = []

        # Process each feature
        for feat_idx in range(d_sae):
            if progress_callback and feat_idx % 100 == 0:
                progress_callback(feat_idx / d_sae * 100)

            # Get activations for this feature
            feat_acts = z[:, :, feat_idx]  # [batch, seq_len]

            # Find non-zero activations
            mask = feat_acts > threshold
            if not mask.any():
                continue

            # Compute statistics
            non_zero = feat_acts[mask]
            stats = {
                "activation_frequency": mask.float().mean().item(),
                "max_activation": non_zero.max().item(),
                "mean_activation": non_zero.mean().item(),
                "std_activation": non_zero.std().item() if len(non_zero) > 1 else 0
            }

            # Get top activating tokens
            top_tokens = self._get_top_tokens(
                feat_acts, input_ids, tokenizer, top_k
            )

            # Create feature record
            feature = Feature(
                sae_id=str(job_id),  # Will be corrected in task
                feature_index=feat_idx,
                extraction_job_id=job_id,
                top_tokens=top_tokens,
                **stats
            )
            features.append(feature)

        return features

    def _get_top_tokens(
        self,
        feat_acts: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer,
        top_k: int
    ) -> List[Dict]:
        """Get top activating tokens with aggregation."""
        # Flatten
        flat_acts = feat_acts.flatten()
        flat_ids = input_ids.flatten()

        # Get top k activations
        top_values, top_indices = torch.topk(flat_acts, min(top_k * 10, len(flat_acts)))
        top_token_ids = flat_ids[top_indices]

        # Aggregate by token
        token_stats = defaultdict(lambda: {"values": [], "count": 0})
        for token_id, value in zip(top_token_ids.tolist(), top_values.tolist()):
            if value > 0:
                token = tokenizer.decode([token_id])
                token_stats[token]["values"].append(value)
                token_stats[token]["count"] += 1

        # Sort by count and compute means
        result = []
        for token, data in sorted(
            token_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:top_k]:
            result.append({
                "token": token,
                "count": data["count"],
                "mean_activation": np.mean(data["values"]),
                "max_activation": max(data["values"])
            })

        return result

    def save_features(self, features: List[Feature]) -> None:
        """Batch save features."""
        self.db.bulk_save_objects(features)
        self.db.commit()

    def get_feature(self, feature_id: UUID) -> Optional[Feature]:
        return self.db.query(Feature).filter(Feature.id == feature_id).first()

    def list_features(
        self,
        sae_id: UUID,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "activation_frequency"
    ) -> List[Feature]:
        query = self.db.query(Feature).filter(Feature.sae_id == str(sae_id))

        if sort_by == "activation_frequency":
            query = query.order_by(Feature.activation_frequency.desc())
        elif sort_by == "max_activation":
            query = query.order_by(Feature.max_activation.desc())
        elif sort_by == "feature_index":
            query = query.order_by(Feature.feature_index)

        return query.offset(offset).limit(limit).all()
```

### 2.2 Backend - Labeling

#### `backend/src/services/openai_labeling_service.py`
```python
import openai
from typing import List, Dict, Optional
import json
import asyncio
from src.core.config import settings

class OpenAILabelingService:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o-mini"

    async def label_feature(
        self,
        top_tokens: List[Dict],
        example_contexts: List[str],
        prompt_template: str
    ) -> Dict:
        """Generate label for a single feature."""
        # Format tokens for prompt
        tokens_str = ", ".join([
            f"'{t['token']}' ({t['count']}x)"
            for t in top_tokens[:20]
        ])

        # Format contexts
        contexts_str = "\n".join([
            f"- {ctx}" for ctx in example_contexts[:10]
        ])

        prompt = prompt_template.format(
            tokens=tokens_str,
            contexts=contexts_str
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at interpreting neural network features."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "semantic_label": result.get("label", ""),
            "category_label": result.get("category", ""),
            "confidence": result.get("confidence", 0.5)
        }

    async def label_features_batch(
        self,
        features_data: List[Dict],
        prompt_template: str,
        concurrency: int = 5
    ) -> List[Dict]:
        """Label multiple features with rate limiting."""
        semaphore = asyncio.Semaphore(concurrency)

        async def label_with_semaphore(data):
            async with semaphore:
                return await self.label_feature(
                    data["top_tokens"],
                    data["contexts"],
                    prompt_template
                )

        tasks = [label_with_semaphore(d) for d in features_data]
        return await asyncio.gather(*tasks)
```

**Key Implementation Notes:**
- Use `gpt-4o-mini` for cost efficiency
- JSON response format for structured output
- Semaphore for rate limiting concurrent requests
- Async/await for efficient batch processing

#### `backend/src/services/labeling_context_formatter.py`
```python
from typing import List, Dict
from src.models.feature import Feature, FeatureActivation

class LabelingContextFormatter:
    """Format feature data for labeling prompts."""

    @staticmethod
    def format_top_tokens(feature: Feature, max_tokens: int = 20) -> str:
        """Format top tokens as readable string."""
        if not feature.top_tokens:
            return "No tokens found"

        tokens = feature.top_tokens[:max_tokens]
        return ", ".join([
            f"'{t['token']}' (count: {t['count']}, mean: {t['mean_activation']:.2f})"
            for t in tokens
        ])

    @staticmethod
    def format_example_contexts(
        activations: List[FeatureActivation],
        max_examples: int = 10
    ) -> List[str]:
        """Format activation contexts for labeling."""
        contexts = []
        for act in activations[:max_examples]:
            before = act.context_before or ""
            after = act.context_after or ""
            token = f"[{act.token}]"
            contexts.append(f"{before}{token}{after}")
        return contexts

    @staticmethod
    def format_statistics(feature: Feature) -> str:
        """Format feature statistics."""
        return (
            f"Activation frequency: {feature.activation_frequency:.4f}\n"
            f"Max activation: {feature.max_activation:.2f}\n"
            f"Mean activation: {feature.mean_activation:.2f}"
        )
```

#### `backend/src/models/labeling_prompt_template.py`
```python
from sqlalchemy import Column, String, Text, Boolean, Integer, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
import uuid
from src.db.base import Base

class LabelingPromptTemplate(Base):
    __tablename__ = "labeling_prompt_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)

    # Template metadata
    description = Column(Text)
    is_default = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)

    created_at = Column(TIMESTAMP, server_default="now()")
    updated_at = Column(TIMESTAMP, onupdate="now()")


# Default template
DEFAULT_LABELING_PROMPT = """
Analyze this neural network feature based on its activation patterns.

Top activating tokens (with frequency):
{tokens}

Example contexts where feature activates:
{contexts}

Feature statistics:
{statistics}

Provide a JSON response with:
1. "label": A concise semantic description (2-5 words) of what this feature represents
2. "category": A hierarchical category path (e.g., "Language > Syntax > Punctuation")
3. "confidence": Your confidence score (0-1) in this interpretation

Focus on the underlying concept, not just listing tokens.
"""
```

### 2.3 Frontend Components

#### `frontend/src/components/features/FeatureBrowser.tsx`
```typescript
import React, { useState, useEffect } from 'react';
import { useFeaturesStore } from '../../stores/featuresStore';
import { Feature } from '../../types/features';
import { Search, Filter, ChevronDown } from 'lucide-react';
import { FeatureCard } from './FeatureCard';
import { FeatureDetailModal } from './FeatureDetailModal';

interface FeatureBrowserProps {
  saeId: string;
}

export function FeatureBrowser({ saeId }: FeatureBrowserProps) {
  const { features, loading, fetchFeatures } = useFeaturesStore();
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<'activation_frequency' | 'max_activation' | 'feature_index'>('activation_frequency');
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);

  useEffect(() => {
    fetchFeatures(saeId, { sortBy });
  }, [saeId, sortBy]);

  const filteredFeatures = features.filter(f => {
    if (!search) return true;
    const searchLower = search.toLowerCase();
    return (
      f.semantic_label?.toLowerCase().includes(searchLower) ||
      f.category_label?.toLowerCase().includes(searchLower) ||
      f.top_tokens?.some(t => t.token.toLowerCase().includes(searchLower))
    );
  });

  return (
    <div className="h-full flex flex-col">
      {/* Search and filters */}
      <div className="flex gap-4 mb-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search features by label or token..."
            className="w-full bg-slate-800 border border-slate-700 rounded pl-10 pr-4 py-2"
          />
        </div>

        <select
          value={sortBy}
          onChange={e => setSortBy(e.target.value as any)}
          className="bg-slate-800 border border-slate-700 rounded px-4 py-2"
        >
          <option value="activation_frequency">Most Frequent</option>
          <option value="max_activation">Highest Activation</option>
          <option value="feature_index">Feature Index</option>
        </select>
      </div>

      {/* Features grid */}
      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full" />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredFeatures.map(feature => (
              <FeatureCard
                key={feature.id}
                feature={feature}
                onClick={() => setSelectedFeature(feature)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Detail modal */}
      {selectedFeature && (
        <FeatureDetailModal
          feature={selectedFeature}
          onClose={() => setSelectedFeature(null)}
        />
      )}
    </div>
  );
}
```

#### `frontend/src/components/features/TokenHighlight.tsx`
```typescript
import React from 'react';

interface TokenHighlightProps {
  context: string;
  token: string;
  activation: number;
  maxActivation: number;
}

export function TokenHighlight({ context, token, activation, maxActivation }: TokenHighlightProps) {
  // Parse context to find token position
  const parts = context.split(new RegExp(`(\\[${escapeRegex(token)}\\])`, 'g'));

  // Calculate intensity based on activation
  const intensity = Math.min(activation / maxActivation, 1);
  const bgOpacity = Math.round(intensity * 100);

  return (
    <span className="font-mono text-sm">
      {parts.map((part, i) => {
        if (part === `[${token}]`) {
          return (
            <span
              key={i}
              className="px-1 rounded"
              style={{
                backgroundColor: `rgba(16, 185, 129, ${intensity * 0.5})`,
                borderBottom: `2px solid rgba(16, 185, 129, ${intensity})`
              }}
              title={`Activation: ${activation.toFixed(2)}`}
            >
              {token}
            </span>
          );
        }
        return <span key={i} className="text-slate-400">{part}</span>;
      })}
    </span>
  );
}

function escapeRegex(string: string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
```

#### `frontend/src/components/features/FeatureDetailModal.tsx`
```typescript
import React, { useEffect, useState } from 'react';
import { Feature } from '../../types/features';
import { X, Tag, BarChart2, Zap } from 'lucide-react';
import { featuresApi } from '../../api/features';
import { TokenHighlight } from './TokenHighlight';

interface FeatureDetailModalProps {
  feature: Feature;
  onClose: () => void;
}

export function FeatureDetailModal({ feature, onClose }: FeatureDetailModalProps) {
  const [activations, setActivations] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadActivations() {
      try {
        const data = await featuresApi.getActivations(feature.id);
        setActivations(data);
      } finally {
        setLoading(false);
      }
    }
    loadActivations();
  }, [feature.id]);

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-slate-900 rounded-lg w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex justify-between items-center p-4 border-b border-slate-800">
          <div>
            <h2 className="text-lg font-medium">Feature #{feature.feature_index}</h2>
            {feature.semantic_label && (
              <p className="text-emerald-400">{feature.semantic_label}</p>
            )}
          </div>
          <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4 space-y-6">
          {/* Statistics */}
          <div className="grid grid-cols-4 gap-4">
            <StatCard
              icon={<BarChart2 className="w-4 h-4" />}
              label="Frequency"
              value={`${(feature.activation_frequency * 100).toFixed(2)}%`}
            />
            <StatCard
              icon={<Zap className="w-4 h-4" />}
              label="Max Activation"
              value={feature.max_activation?.toFixed(2)}
            />
            <StatCard
              icon={<Zap className="w-4 h-4" />}
              label="Mean Activation"
              value={feature.mean_activation?.toFixed(2)}
            />
            <StatCard
              icon={<Tag className="w-4 h-4" />}
              label="Category"
              value={feature.category_label || 'Unlabeled'}
            />
          </div>

          {/* Top tokens */}
          <div>
            <h3 className="text-sm font-medium text-slate-400 mb-2">Top Activating Tokens</h3>
            <div className="flex flex-wrap gap-2">
              {feature.top_tokens?.slice(0, 30).map((t, i) => (
                <span
                  key={i}
                  className="px-2 py-1 bg-slate-800 rounded text-sm"
                  title={`Count: ${t.count}, Mean: ${t.mean_activation.toFixed(2)}`}
                >
                  {t.token} <span className="text-slate-500">×{t.count}</span>
                </span>
              ))}
            </div>
          </div>

          {/* Example activations */}
          <div>
            <h3 className="text-sm font-medium text-slate-400 mb-2">Example Activations</h3>
            {loading ? (
              <div className="animate-pulse h-32 bg-slate-800 rounded" />
            ) : (
              <div className="space-y-2">
                {activations.slice(0, 20).map((act, i) => (
                  <div key={i} className="bg-slate-800 rounded p-3">
                    <TokenHighlight
                      context={`${act.context_before}[${act.token}]${act.context_after}`}
                      token={act.token}
                      activation={act.activation_value}
                      maxActivation={feature.max_activation}
                    />
                    <div className="text-xs text-slate-500 mt-1">
                      Activation: {act.activation_value.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="bg-slate-800 rounded p-3">
      <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
        {icon}
        {label}
      </div>
      <div className="font-medium truncate" title={value}>{value}</div>
    </div>
  );
}
```

---

## 3. Common Patterns

### 3.1 Vectorized Feature Extraction
```python
# backend/src/services/extraction_vectorized.py
import torch
from typing import List, Dict

def extract_features_vectorized(
    sae: BaseSAE,
    activations: torch.Tensor,
    threshold: float = 0.0
) -> Dict[str, torch.Tensor]:
    """Vectorized feature extraction for speed."""
    with torch.no_grad():
        z = sae.encode(activations)  # [batch, seq, d_sae]

    # Compute statistics across all positions
    mask = z > threshold

    # Activation frequency per feature
    freq = mask.float().mean(dim=(0, 1))  # [d_sae]

    # Max, mean, std per feature
    z_masked = z.clone()
    z_masked[~mask] = float('nan')

    max_act = torch.nanmax(z_masked.view(-1, z.shape[-1]), dim=0).values
    mean_act = torch.nanmean(z_masked.view(-1, z.shape[-1]), dim=0)

    return {
        "frequencies": freq,
        "max_activations": max_act,
        "mean_activations": mean_act,
        "active_features": (freq > 0).nonzero().flatten()
    }
```

### 3.2 Feature Correlation Analysis
```python
# backend/src/services/correlation_service.py
import torch
import numpy as np
from scipy.stats import spearmanr

def compute_feature_correlations(
    z: torch.Tensor,
    feature_indices: List[int],
    method: str = "cosine"
) -> np.ndarray:
    """Compute pairwise feature correlations."""
    # Extract relevant features
    features = z[:, :, feature_indices].view(-1, len(feature_indices))
    features = features.cpu().numpy()

    if method == "cosine":
        # Normalize and compute dot product
        norms = np.linalg.norm(features, axis=0, keepdims=True)
        normalized = features / (norms + 1e-8)
        correlations = normalized.T @ normalized / features.shape[0]

    elif method == "spearman":
        correlations = np.zeros((len(feature_indices), len(feature_indices)))
        for i in range(len(feature_indices)):
            for j in range(i, len(feature_indices)):
                corr, _ = spearmanr(features[:, i], features[:, j])
                correlations[i, j] = correlations[j, i] = corr

    return correlations
```

---

## 4. Testing Strategy

### 4.1 Extraction Tests
```python
# backend/tests/test_extraction_service.py
import torch
import pytest
from src.services.extraction_service import ExtractionService
from src.ml.sparse_autoencoder import StandardSAE

def test_extract_features():
    sae = StandardSAE(d_in=256, d_sae=1024)
    activations = torch.randn(10, 32, 256)  # batch, seq, d_in

    # Mock tokenizer
    class MockTokenizer:
        def decode(self, ids):
            return f"token_{ids[0]}"

    service = ExtractionService(mock_db)
    features = service.extract_features(
        sae, activations, MockTokenizer(),
        input_ids=torch.randint(0, 1000, (10, 32)),
        job_id="test-job"
    )

    assert len(features) > 0
    assert all(f.feature_index >= 0 for f in features)

def test_top_tokens_aggregation():
    # Test that tokens are properly aggregated
    pass
```

### 4.2 Labeling Tests
```python
# backend/tests/test_labeling_service.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_label_feature():
    service = OpenAILabelingService()

    with patch.object(service.client.chat.completions, 'create', new_callable=AsyncMock) as mock:
        mock.return_value.choices[0].message.content = '{"label": "test", "category": "Test", "confidence": 0.8}'

        result = await service.label_feature(
            top_tokens=[{"token": "hello", "count": 10}],
            example_contexts=["Hello world"],
            prompt_template="Label this: {tokens}"
        )

        assert result["semantic_label"] == "test"
```

---

## 5. Common Pitfalls

### Pitfall 1: Memory in Feature Extraction
```python
# WRONG - Loading all activations at once
activations = torch.load("huge_activations.pt")  # OOM!

# RIGHT - Process in batches
for batch in torch.utils.data.DataLoader(activations, batch_size=256):
    features = extract_batch(batch)
    save_features(features)
```

### Pitfall 2: Token Decoding Edge Cases
```python
# WRONG - Assuming clean token strings
token = tokenizer.decode([token_id])  # May have special chars

# RIGHT - Handle edge cases
token = tokenizer.decode([token_id], skip_special_tokens=True)
token = token.replace('\n', '\\n').replace('\t', '\\t')
```

### Pitfall 3: Label Caching
```python
# WRONG - Re-labeling already labeled features
await label_all_features(features)

# RIGHT - Only label unlabeled features
unlabeled = [f for f in features if not f.semantic_label]
await label_features(unlabeled)
```

---

## 6. Performance Tips

1. **Batch Database Inserts**
   ```python
   # Use bulk_save_objects for large batches
   db.bulk_save_objects(features)
   db.commit()
   ```

2. **Async Labeling with Rate Limiting**
   ```python
   # Use asyncio.Semaphore for API rate limits
   semaphore = asyncio.Semaphore(5)  # Max 5 concurrent
   ```

3. **Index for Fast Queries**
   ```sql
   CREATE INDEX idx_features_sae_freq ON features(sae_id, activation_frequency DESC);
   ```

---

*Related: [PRD](../prds/004_FPRD|Feature_Discovery.md) | [TDD](../tdds/004_FTDD|Feature_Discovery.md) | [FTASKS](../tasks/004_FTASKS|Feature_Discovery.md)*
