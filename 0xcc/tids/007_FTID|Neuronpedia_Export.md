# Technical Implementation Document: Neuronpedia Export

**Document ID:** 007_FTID|Neuronpedia_Export
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related TDD:** [007_FTDD|Neuronpedia_Export](../tdds/007_FTDD|Neuronpedia_Export.md)

---

## 1. Implementation Order

### Phase 1: Data Computation Services
1. Logit lens service
2. Histogram service
3. Token aggregator service
4. Feature dashboard data model

### Phase 2: Export Infrastructure
1. Export job model
2. Export service
3. Celery export task
4. WebSocket progress emission

### Phase 3: Package Generation
1. Feature JSON generator
2. SAELens config converter
3. ZIP archive builder
4. Download endpoint

### Phase 4: Frontend
1. Export modal component
2. Progress tracking
3. Export history list
4. Download handler

---

## 2. File-by-File Implementation

### 2.1 Backend - Data Computation

#### `backend/src/services/logit_lens_service.py`
```python
import torch
from typing import List, Dict, Optional
from transformers import PreTrainedModel

class LogitLensService:
    """Compute logit lens data for SAE features."""

    def __init__(self, model: PreTrainedModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def compute_logit_lens(
        self,
        sae,
        feature_indices: List[int],
        top_k: int = 20
    ) -> Dict[int, Dict]:
        """Compute top promoted/suppressed tokens for each feature."""
        results = {}

        # Get decoder weights
        W_dec = sae.W_dec  # [d_sae, d_in]

        # Get model's unembedding matrix
        # For most models: lm_head.weight or embed_out.weight
        if hasattr(self.model, 'lm_head'):
            unembed = self.model.lm_head.weight  # [vocab, hidden]
        elif hasattr(self.model, 'embed_out'):
            unembed = self.model.embed_out.weight
        else:
            raise ValueError("Could not find unembedding matrix")

        for feat_idx in feature_indices:
            # Get feature direction in residual stream
            feature_direction = W_dec[feat_idx]  # [d_in]

            # Project to vocabulary
            logits = feature_direction @ unembed.T  # [vocab]

            # Get top positive (promoted)
            top_pos_values, top_pos_indices = torch.topk(logits, top_k)
            top_positive = [
                {
                    "token": self.tokenizer.decode([idx.item()]),
                    "token_id": idx.item(),
                    "value": val.item()
                }
                for idx, val in zip(top_pos_indices, top_pos_values)
            ]

            # Get top negative (suppressed)
            top_neg_values, top_neg_indices = torch.topk(-logits, top_k)
            top_negative = [
                {
                    "token": self.tokenizer.decode([idx.item()]),
                    "token_id": idx.item(),
                    "value": -val.item()
                }
                for idx, val in zip(top_neg_indices, top_neg_values)
            ]

            results[feat_idx] = {
                "top_positive": top_positive,
                "top_negative": top_negative
            }

        return results

    def compute_single_feature(
        self,
        sae,
        feature_index: int,
        top_k: int = 20
    ) -> Dict:
        """Compute logit lens for a single feature."""
        result = self.compute_logit_lens(sae, [feature_index], top_k)
        return result[feature_index]
```

**Key Implementation Notes:**
- Feature direction from decoder weights
- Project through model's unembedding matrix
- Top-k for both promoted (positive) and suppressed (negative) tokens

#### `backend/src/services/histogram_service.py`
```python
import torch
import numpy as np
from typing import List, Dict, Optional

class HistogramService:
    """Generate activation histograms for features."""

    def compute_histogram(
        self,
        activations: torch.Tensor,
        feature_index: int,
        num_bins: int = 50
    ) -> Dict:
        """Compute histogram for a feature's activations."""
        # Get activations for this feature
        feat_acts = activations[:, :, feature_index].flatten()

        # Filter non-zero activations
        non_zero = feat_acts[feat_acts > 0].cpu().numpy()

        if len(non_zero) == 0:
            return {
                "bin_edges": [0],
                "counts": [0],
                "total_activations": 0,
                "non_zero_count": 0
            }

        # Compute histogram
        counts, bin_edges = np.histogram(non_zero, bins=num_bins)

        return {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
            "total_activations": len(feat_acts),
            "non_zero_count": len(non_zero),
            "min": float(non_zero.min()),
            "max": float(non_zero.max()),
            "mean": float(non_zero.mean()),
            "std": float(non_zero.std())
        }

    def compute_batch_histograms(
        self,
        activations: torch.Tensor,
        feature_indices: List[int],
        num_bins: int = 50,
        progress_callback=None
    ) -> Dict[int, Dict]:
        """Compute histograms for multiple features."""
        results = {}

        for i, feat_idx in enumerate(feature_indices):
            results[feat_idx] = self.compute_histogram(
                activations, feat_idx, num_bins
            )

            if progress_callback and i % 100 == 0:
                progress_callback(i / len(feature_indices) * 100)

        return results
```

#### `backend/src/services/token_aggregator_service.py`
```python
import torch
from typing import List, Dict
from collections import defaultdict

class TokenAggregatorService:
    """Aggregate token statistics for features."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def aggregate_top_tokens(
        self,
        activations: torch.Tensor,
        input_ids: torch.Tensor,
        feature_index: int,
        top_k: int = 50
    ) -> List[Dict]:
        """Get top activating tokens with aggregated statistics."""
        # Get feature activations
        feat_acts = activations[:, :, feature_index]  # [batch, seq]

        # Flatten both
        flat_acts = feat_acts.flatten()
        flat_ids = input_ids.flatten()

        # Aggregate by token
        token_stats = defaultdict(lambda: {
            "count": 0,
            "activations": [],
            "positions": []
        })

        for pos, (token_id, act) in enumerate(zip(flat_ids, flat_acts)):
            if act.item() > 0:
                token_id = token_id.item()
                token_stats[token_id]["count"] += 1
                token_stats[token_id]["activations"].append(act.item())
                token_stats[token_id]["positions"].append(pos)

        # Sort by count and compute statistics
        sorted_tokens = sorted(
            token_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:top_k]

        results = []
        for token_id, stats in sorted_tokens:
            activations = stats["activations"]
            results.append({
                "token": self.tokenizer.decode([token_id]),
                "token_id": token_id,
                "count": stats["count"],
                "mean_activation": sum(activations) / len(activations),
                "max_activation": max(activations),
                "min_activation": min(activations)
            })

        return results
```

### 2.2 Backend - Export Service

#### `backend/src/models/neuronpedia_export.py`
```python
from sqlalchemy import Column, String, Integer, Float, BigInteger, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from src.db.base import Base

class NeuronpediaExportJob(Base):
    """Track Neuronpedia export jobs."""
    __tablename__ = "neuronpedia_export_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sae_id = Column(UUID(as_uuid=True), nullable=False)

    # Configuration
    config = Column(JSONB, nullable=False)
    feature_selection = Column(String(50))  # all, extracted, custom
    feature_range_start = Column(Integer)
    feature_range_end = Column(Integer)

    # Options
    include_logit_lens = Column(Boolean, default=True)
    include_histograms = Column(Boolean, default=True)
    include_top_tokens = Column(Boolean, default=True)
    include_explanations = Column(Boolean, default=True)
    include_saelens_weights = Column(Boolean, default=True)

    # Progress
    status = Column(String(50), default="pending")
    progress = Column(Float, default=0)
    current_stage = Column(String(100))
    feature_count = Column(Integer)

    # Output
    output_path = Column(String(500))
    file_size_bytes = Column(BigInteger)
    error_message = Column(String)

    created_at = Column(TIMESTAMP, server_default="now()")
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
```

#### `backend/src/services/neuronpedia_export_service.py`
```python
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session

from src.models.neuronpedia_export import NeuronpediaExportJob
from src.models.feature import Feature
from src.services.logit_lens_service import LogitLensService
from src.services.histogram_service import HistogramService
from src.services.token_aggregator_service import TokenAggregatorService
from src.core.config import settings

class NeuronpediaExportService:
    """Orchestrate Neuronpedia export process."""

    def __init__(self, db: Session):
        self.db = db
        self.exports_dir = Path(settings.data_dir) / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def create_job(
        self,
        sae_id: UUID,
        config: dict
    ) -> NeuronpediaExportJob:
        """Create export job record."""
        job = NeuronpediaExportJob(
            sae_id=sae_id,
            config=config,
            feature_selection=config.get("feature_selection", "all"),
            include_logit_lens=config.get("include_logit_lens", True),
            include_histograms=config.get("include_histograms", True),
            include_top_tokens=config.get("include_top_tokens", True),
            include_explanations=config.get("include_explanations", True),
            include_saelens_weights=config.get("include_saelens_weights", True),
            status="pending"
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get_features_to_export(
        self,
        sae_id: UUID,
        selection: str,
        range_start: int = None,
        range_end: int = None
    ) -> List[Feature]:
        """Get features based on selection criteria."""
        query = self.db.query(Feature).filter(Feature.sae_id == str(sae_id))

        if selection == "extracted":
            # Only features with semantic labels
            query = query.filter(Feature.semantic_label.isnot(None))
        elif selection == "custom" and range_start is not None:
            query = query.filter(
                Feature.feature_index >= range_start,
                Feature.feature_index <= range_end
            )

        return query.order_by(Feature.feature_index).all()

    def generate_feature_json(
        self,
        feature: Feature,
        logit_lens_data: Optional[Dict] = None,
        histogram_data: Optional[Dict] = None
    ) -> Dict:
        """Generate Neuronpedia-compatible feature JSON."""
        data = {
            "feature_index": feature.feature_index,
            "label": feature.semantic_label,
            "category": feature.category_label,
            "statistics": {
                "activation_frequency": feature.activation_frequency,
                "max_activation": feature.max_activation,
                "mean_activation": feature.mean_activation
            }
        }

        if logit_lens_data:
            data["logit_lens"] = logit_lens_data

        if histogram_data:
            data["histogram"] = histogram_data

        if feature.top_tokens:
            data["top_tokens"] = feature.top_tokens

        return data

    def generate_metadata(
        self,
        sae_info: Dict,
        feature_count: int,
        config: Dict
    ) -> Dict:
        """Generate export metadata."""
        return {
            "sae_id": sae_info["id"],
            "sae_name": sae_info["name"],
            "model_name": sae_info["model_name"],
            "layer": sae_info["layer"],
            "d_in": sae_info["d_in"],
            "d_sae": sae_info["d_sae"],
            "architecture": sae_info.get("architecture", "standard"),
            "export_date": datetime.utcnow().isoformat() + "Z",
            "feature_count": feature_count,
            "includes": {
                "logit_lens": config.get("include_logit_lens", True),
                "histograms": config.get("include_histograms", True),
                "top_tokens": config.get("include_top_tokens", True),
                "explanations": config.get("include_explanations", True),
                "saelens_weights": config.get("include_saelens_weights", True)
            }
        }

    def create_export_archive(
        self,
        job_id: UUID,
        metadata: Dict,
        feature_jsons: List[Dict],
        saelens_path: Optional[str] = None
    ) -> str:
        """Create ZIP archive with export data."""
        archive_path = self.exports_dir / f"{job_id}.zip"

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add metadata
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

            # Add README
            readme = self._generate_readme(metadata)
            zf.writestr("README.md", readme)

            # Add feature JSONs
            for feat_json in feature_jsons:
                idx = feat_json["feature_index"]
                zf.writestr(
                    f"features/{idx}.json",
                    json.dumps(feat_json, indent=2)
                )

            # Add explanations summary
            explanations = {
                str(f["feature_index"]): f.get("label", "")
                for f in feature_jsons
            }
            zf.writestr(
                "explanations/explanations.json",
                json.dumps(explanations, indent=2)
            )

            # Add SAELens weights if requested
            if saelens_path:
                saelens_dir = Path(saelens_path)
                if (saelens_dir / "cfg.json").exists():
                    zf.write(saelens_dir / "cfg.json", "saelens/cfg.json")
                if (saelens_dir / "sae_weights.safetensors").exists():
                    zf.write(
                        saelens_dir / "sae_weights.safetensors",
                        "saelens/sae_weights.safetensors"
                    )

        return str(archive_path)

    def _generate_readme(self, metadata: Dict) -> str:
        """Generate README for export."""
        return f"""# Neuronpedia Export

## SAE Information
- **Name:** {metadata['sae_name']}
- **Model:** {metadata['model_name']}
- **Layer:** {metadata['layer']}
- **Dimensions:** {metadata['d_in']} → {metadata['d_sae']}
- **Architecture:** {metadata['architecture']}

## Export Details
- **Date:** {metadata['export_date']}
- **Feature Count:** {metadata['feature_count']}

## Contents
- `metadata.json` - Export configuration and SAE info
- `features/` - Individual feature JSON files
- `explanations/` - Feature labels/explanations
- `saelens/` - SAELens-compatible weights (if included)

## Usage
This export is compatible with [Neuronpedia](https://neuronpedia.org).

Generated by miStudio.
"""
```

#### `backend/src/workers/neuronpedia_tasks.py`
```python
from celery import shared_task
from datetime import datetime
from src.db.session import SessionLocal
from src.services.neuronpedia_export_service import NeuronpediaExportService
from src.services.sae_manager_service import SAEManagerService
from src.services.logit_lens_service import LogitLensService
from src.services.histogram_service import HistogramService
from src.ml.sae_loader import load_sae
from src.ml.model_loader import ModelLoader
from src.workers.websocket_emitter import emit_export_progress

@shared_task(bind=True, queue='export')
def neuronpedia_export_task(self, job_id: str, config: dict):
    """Background task for Neuronpedia export."""
    db = SessionLocal()

    try:
        export_service = NeuronpediaExportService(db)
        sae_manager = SAEManagerService(db)

        # Get job and SAE info
        job = db.query(NeuronpediaExportJob).get(job_id)
        sae_info = sae_manager.get_sae_by_id(job.sae_id)

        if not sae_info:
            raise ValueError("SAE not found")

        # Update status
        job.status = "computing"
        job.started_at = datetime.utcnow()
        db.commit()

        emit_export_progress(job_id, 0, "Loading models...")

        # Load model and SAE
        model_loader = ModelLoader.get_instance()
        model = model_loader.load_model(sae_info["model_name"])
        tokenizer = model_loader.load_tokenizer(sae_info["model_name"])
        sae = load_sae(sae_info["local_path"])

        # Get features
        features = export_service.get_features_to_export(
            job.sae_id,
            job.feature_selection,
            job.feature_range_start,
            job.feature_range_end
        )
        job.feature_count = len(features)
        db.commit()

        emit_export_progress(job_id, 5, f"Processing {len(features)} features...")

        # Initialize services
        logit_lens_service = LogitLensService(model, tokenizer) if job.include_logit_lens else None
        histogram_service = HistogramService() if job.include_histograms else None

        feature_jsons = []
        total = len(features)

        for i, feature in enumerate(features):
            # Compute logit lens
            logit_lens_data = None
            if logit_lens_service:
                logit_lens_data = logit_lens_service.compute_single_feature(
                    sae, feature.feature_index
                )

            # Generate feature JSON
            feat_json = export_service.generate_feature_json(
                feature,
                logit_lens_data=logit_lens_data
            )
            feature_jsons.append(feat_json)

            # Update progress
            if i % 10 == 0:
                progress = 5 + (i / total * 80)
                emit_export_progress(
                    job_id, progress,
                    f"Processing feature {i}/{total}"
                )
                job.progress = progress
                job.current_stage = f"Feature {i}/{total}"
                db.commit()

        emit_export_progress(job_id, 85, "Creating archive...")

        # Generate metadata
        metadata = export_service.generate_metadata(
            sae_info, len(features), config
        )

        # Create archive
        saelens_path = sae_info["local_path"] if job.include_saelens_weights else None
        archive_path = export_service.create_export_archive(
            job_id, metadata, feature_jsons, saelens_path
        )

        # Update job
        import os
        job.status = "completed"
        job.progress = 100
        job.output_path = archive_path
        job.file_size_bytes = os.path.getsize(archive_path)
        job.completed_at = datetime.utcnow()
        db.commit()

        emit_export_progress(job_id, 100, "Complete", completed=True)

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        emit_export_progress(job_id, 0, str(e), failed=True)

    finally:
        db.close()
```

### 2.3 Frontend Components

#### `frontend/src/components/saes/ExportToNeuronpedia.tsx`
```typescript
import React, { useState, useEffect } from 'react';
import { neuronpediaApi } from '../../api/neuronpedia';
import { SAE } from '../../types/sae';
import { Download, X } from 'lucide-react';

interface ExportToNeuronpediaProps {
  sae: SAE;
  onClose: () => void;
}

export function ExportToNeuronpedia({ sae, onClose }: ExportToNeuronpediaProps) {
  const [config, setConfig] = useState({
    feature_selection: 'all',
    range_start: 0,
    range_end: sae.d_sae,
    include_logit_lens: true,
    include_histograms: true,
    include_top_tokens: true,
    include_explanations: true,
    include_saelens_weights: true,
    logit_lens_k: 20,
    histogram_bins: 50,
    top_tokens_k: 50
  });

  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('');
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');

  // WebSocket subscription for progress
  useEffect(() => {
    if (!jobId) return;

    const channel = `export/${jobId}`;
    // Subscribe to WebSocket channel...

    return () => {
      // Unsubscribe
    };
  }, [jobId]);

  const startExport = async () => {
    try {
      const job = await neuronpediaApi.startExport(sae.id, config);
      setJobId(job.id);
      setStatus('running');
    } catch (error) {
      console.error('Export failed:', error);
      setStatus('failed');
    }
  };

  const downloadExport = async () => {
    if (jobId) {
      window.location.href = `/api/v1/neuronpedia/export/${jobId}/download`;
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-slate-900 rounded-lg w-full max-w-2xl p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-medium">Export to Neuronpedia</h2>
          <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded">
            <X className="w-5 h-5" />
          </button>
        </div>

        {status === 'idle' && (
          <form className="space-y-4">
            {/* SAE Info */}
            <div className="bg-slate-800 rounded p-3 text-sm">
              <div><strong>SAE:</strong> {sae.name}</div>
              <div><strong>Features:</strong> {sae.d_sae}</div>
            </div>

            {/* Feature Selection */}
            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Feature Selection
              </label>
              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    value="all"
                    checked={config.feature_selection === 'all'}
                    onChange={e => setConfig({ ...config, feature_selection: e.target.value })}
                  />
                  All features ({sae.d_sae})
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    value="extracted"
                    checked={config.feature_selection === 'extracted'}
                    onChange={e => setConfig({ ...config, feature_selection: e.target.value })}
                  />
                  Extracted features only
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    value="custom"
                    checked={config.feature_selection === 'custom'}
                    onChange={e => setConfig({ ...config, feature_selection: e.target.value })}
                  />
                  Custom range
                </label>
              </div>

              {config.feature_selection === 'custom' && (
                <div className="grid grid-cols-2 gap-4 mt-2">
                  <input
                    type="number"
                    placeholder="Start"
                    value={config.range_start}
                    onChange={e => setConfig({ ...config, range_start: parseInt(e.target.value) })}
                    className="bg-slate-800 rounded px-3 py-2"
                  />
                  <input
                    type="number"
                    placeholder="End"
                    value={config.range_end}
                    onChange={e => setConfig({ ...config, range_end: parseInt(e.target.value) })}
                    className="bg-slate-800 rounded px-3 py-2"
                  />
                </div>
              )}
            </div>

            {/* Include Options */}
            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Include Data
              </label>
              <div className="space-y-2">
                {[
                  { key: 'include_logit_lens', label: 'Logit lens data' },
                  { key: 'include_histograms', label: 'Activation histograms' },
                  { key: 'include_top_tokens', label: 'Top activating tokens' },
                  { key: 'include_explanations', label: 'Feature explanations' },
                  { key: 'include_saelens_weights', label: 'SAELens-compatible weights' }
                ].map(({ key, label }) => (
                  <label key={key} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={config[key as keyof typeof config] as boolean}
                      onChange={e => setConfig({ ...config, [key]: e.target.checked })}
                    />
                    {label}
                  </label>
                ))}
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-slate-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={startExport}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded"
              >
                Start Export
              </button>
            </div>
          </form>
        )}

        {status === 'running' && (
          <div className="space-y-4">
            <div className="text-center">
              <div className="text-4xl font-bold text-emerald-400">
                {progress.toFixed(0)}%
              </div>
              <div className="text-slate-400 mt-2">{stage}</div>
            </div>

            <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {status === 'completed' && (
          <div className="text-center space-y-4">
            <div className="text-emerald-400 text-lg">Export Complete!</div>
            <button
              onClick={downloadExport}
              className="flex items-center gap-2 mx-auto px-6 py-3 bg-emerald-600 hover:bg-emerald-500 rounded"
            >
              <Download className="w-5 h-5" />
              Download Archive
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
```

---

## 3. Common Patterns

### 3.1 Chunked Processing for Large Exports
```python
def process_features_chunked(features, chunk_size=100):
    """Process features in chunks to manage memory."""
    for i in range(0, len(features), chunk_size):
        chunk = features[i:i+chunk_size]
        yield from process_chunk(chunk)
        gc.collect()  # Free memory between chunks
```

### 3.2 Streaming Archive Creation
```python
def stream_export_archive(job_id):
    """Stream ZIP archive creation for large exports."""
    def generate():
        with zipfile.ZipFile(io.BytesIO(), 'w') as zf:
            # Add files incrementally
            for feat_json in feature_generator():
                zf.writestr(f"features/{feat_json['index']}.json", json.dumps(feat_json))
                yield zf.read()  # Stream bytes

    return StreamingResponse(generate(), media_type="application/zip")
```

---

## 4. Testing Strategy

### 4.1 Export Service Tests
```python
def test_generate_feature_json():
    service = NeuronpediaExportService(mock_db)
    feature = Feature(
        feature_index=0,
        semantic_label="test",
        activation_frequency=0.1
    )

    json_data = service.generate_feature_json(feature)

    assert json_data["feature_index"] == 0
    assert json_data["label"] == "test"

def test_create_archive():
    service = NeuronpediaExportService(mock_db)

    archive_path = service.create_export_archive(
        "test-job",
        {"sae_name": "test"},
        [{"feature_index": 0}]
    )

    assert Path(archive_path).exists()
    with zipfile.ZipFile(archive_path) as zf:
        assert "metadata.json" in zf.namelist()
```

---

## 5. Common Pitfalls

### Pitfall 1: Memory in Logit Lens
```python
# WRONG - Compute all at once
logits = feature_directions @ unembed.T  # [d_sae, vocab] - huge!

# RIGHT - Process per feature
for feat_idx in feature_indices:
    direction = W_dec[feat_idx]
    logits = direction @ unembed.T  # [vocab] - manageable
```

### Pitfall 2: Archive Path Security
```python
# WRONG - User-controlled path
zf.write(user_path, filename)  # Path traversal risk!

# RIGHT - Sanitize filenames
safe_name = Path(filename).name  # Remove directory components
zf.writestr(f"features/{safe_name}", data)
```

---

## 6. Performance Tips

1. **Parallel Logit Lens Computation**
   ```python
   from concurrent.futures import ThreadPoolExecutor

   with ThreadPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(compute_logit_lens, features))
   ```

2. **Compress Archive Efficiently**
   ```python
   # Use maximum compression for text data
   zf = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9)
   ```

---

*Related: [PRD](../prds/007_FPRD|Neuronpedia_Export.md) | [TDD](../tdds/007_FTDD|Neuronpedia_Export.md) | [FTASKS](../tasks/007_FTASKS|Neuronpedia_Export.md)*
