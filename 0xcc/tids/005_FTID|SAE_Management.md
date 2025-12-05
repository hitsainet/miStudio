# Technical Implementation Document: SAE Management

**Document ID:** 005_FTID|SAE_Management
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related TDD:** [005_FTDD|SAE_Management](../tdds/005_FTDD|SAE_Management.md)

---

## 1. Implementation Order

### Phase 1: Database & Models
1. Database migration (external_saes table)
2. SQLAlchemy model
3. Pydantic schemas
4. SAE Manager service

### Phase 2: HuggingFace Integration
1. HuggingFace SAE download service
2. Format detection utilities
3. Celery download task

### Phase 3: Format Support
1. SAELens Community format loader
2. miStudio native format loader
3. Format conversion utilities

### Phase 4: Frontend
1. SAEs store
2. SAE card component
3. Download from HF modal
4. SAEs panel

---

## 2. File-by-File Implementation

### 2.1 Backend Files

#### `backend/src/models/external_sae.py`
```python
from sqlalchemy import Column, String, Integer, Boolean, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from src.db.base import Base

class ExternalSAE(Base):
    """SAE downloaded from external sources (HuggingFace, Gemma Scope)."""
    __tablename__ = "external_saes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)

    # Model association
    model_name = Column(String(255), nullable=False)
    layer = Column(Integer, nullable=False)
    hook_name = Column(String(255))

    # Dimensions
    d_in = Column(Integer, nullable=False)
    d_sae = Column(Integer, nullable=False)

    # Architecture
    architecture = Column(String(50))  # standard, jumprelu, skip, transcoder

    # Source tracking
    source = Column(String(50), nullable=False)  # huggingface, gemma_scope, upload
    repo_id = Column(String(255))
    local_path = Column(String(500))
    format = Column(String(50))  # community, mistudio

    # Metadata
    metadata = Column(JSONB)
    status = Column(String(50), default="pending")
    error_message = Column(String)

    created_at = Column(TIMESTAMP, server_default="now()")
    updated_at = Column(TIMESTAMP, onupdate="now()")
```

#### `backend/src/services/sae_manager_service.py`
```python
from typing import List, Optional, Union
from uuid import UUID
from pathlib import Path
from sqlalchemy.orm import Session

from src.models.external_sae import ExternalSAE
from src.models.training import Training
from src.core.config import settings

class SAEManagerService:
    """Unified service for managing all SAE sources."""

    def __init__(self, db: Session):
        self.db = db
        self.saes_dir = Path(settings.data_dir) / "saes"
        self.saes_dir.mkdir(parents=True, exist_ok=True)

    def list_all_saes(
        self,
        model_name: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[dict]:
        """List all SAEs from both internal training and external sources."""
        results = []

        # Get trained SAEs
        trained_query = self.db.query(Training).filter(
            Training.status == "completed"
        )
        if model_name:
            trained_query = trained_query.filter(Training.model_name == model_name)

        for training in trained_query.all():
            results.append({
                "id": str(training.id),
                "name": training.name,
                "source": "trained",
                "model_name": training.model_name,
                "layer": training.layer,
                "d_in": training.d_in,
                "d_sae": training.d_sae,
                "architecture": training.architecture,
                "local_path": training.output_path,
                "feature_count": self._get_feature_count(training.id),
                "created_at": training.created_at
            })

        # Get external SAEs
        external_query = self.db.query(ExternalSAE).filter(
            ExternalSAE.status == "ready"
        )
        if model_name:
            external_query = external_query.filter(ExternalSAE.model_name == model_name)
        if source:
            external_query = external_query.filter(ExternalSAE.source == source)

        for sae in external_query.all():
            results.append({
                "id": str(sae.id),
                "name": sae.name,
                "source": sae.source,
                "model_name": sae.model_name,
                "layer": sae.layer,
                "d_in": sae.d_in,
                "d_sae": sae.d_sae,
                "architecture": sae.architecture,
                "local_path": sae.local_path,
                "repo_id": sae.repo_id,
                "created_at": sae.created_at
            })

        return sorted(results, key=lambda x: x["created_at"], reverse=True)

    def get_sae_by_id(self, sae_id: UUID) -> Optional[dict]:
        """Get SAE details by ID (checks both sources)."""
        # Check trained SAEs
        training = self.db.query(Training).filter(Training.id == sae_id).first()
        if training:
            return self._training_to_dict(training)

        # Check external SAEs
        external = self.db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
        if external:
            return self._external_to_dict(external)

        return None

    def create_external_sae(self, data: dict) -> ExternalSAE:
        """Create external SAE record."""
        sae = ExternalSAE(**data)
        self.db.add(sae)
        self.db.commit()
        self.db.refresh(sae)
        return sae

    def delete_sae(self, sae_id: UUID) -> bool:
        """Delete SAE and its files."""
        # Check external SAEs first
        external = self.db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
        if external:
            if external.local_path:
                import shutil
                shutil.rmtree(external.local_path, ignore_errors=True)
            self.db.delete(external)
            self.db.commit()
            return True

        return False

    def _get_feature_count(self, training_id: UUID) -> int:
        """Get count of extracted features for a training."""
        from src.models.feature import Feature
        return self.db.query(Feature).filter(
            Feature.sae_id == str(training_id)
        ).count()

    def _training_to_dict(self, training: Training) -> dict:
        return {
            "id": str(training.id),
            "name": training.name,
            "source": "trained",
            "model_name": training.model_name,
            "layer": training.layer,
            "d_in": training.d_in,
            "d_sae": training.d_sae,
            "architecture": training.architecture,
            "local_path": training.output_path
        }

    def _external_to_dict(self, sae: ExternalSAE) -> dict:
        return {
            "id": str(sae.id),
            "name": sae.name,
            "source": sae.source,
            "model_name": sae.model_name,
            "layer": sae.layer,
            "d_in": sae.d_in,
            "d_sae": sae.d_sae,
            "architecture": sae.architecture,
            "local_path": sae.local_path,
            "repo_id": sae.repo_id
        }
```

#### `backend/src/services/huggingface_sae_service.py`
```python
import json
from pathlib import Path
from typing import Optional, Dict
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from safetensors import safe_open
from src.core.config import settings

class HuggingFaceSAEService:
    """Service for downloading SAEs from HuggingFace."""

    def __init__(self):
        self.cache_dir = Path(settings.data_dir) / "saes" / "external"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api = HfApi()

    def get_repo_info(self, repo_id: str) -> Dict:
        """Get repository metadata."""
        info = self.api.repo_info(repo_id, repo_type="model")
        files = self.api.list_repo_files(repo_id)

        return {
            "repo_id": repo_id,
            "files": files,
            "is_gemma_scope": self._is_gemma_scope(files),
            "available_layers": self._parse_layers(files) if self._is_gemma_scope(files) else []
        }

    def _is_gemma_scope(self, files: list) -> bool:
        """Check if repo follows Gemma Scope structure."""
        return any("layer_" in f for f in files)

    def _parse_layers(self, files: list) -> list:
        """Parse available layers from Gemma Scope structure."""
        layers = set()
        for f in files:
            if "layer_" in f:
                parts = f.split("/")
                for part in parts:
                    if part.startswith("layer_"):
                        layers.add(int(part.replace("layer_", "")))
        return sorted(layers)

    def download_sae(
        self,
        repo_id: str,
        subfolder: Optional[str] = None,
        progress_callback=None
    ) -> str:
        """Download SAE from HuggingFace."""
        local_dir = self.cache_dir / repo_id.replace("/", "_")
        if subfolder:
            local_dir = local_dir / subfolder.replace("/", "_")

        path = snapshot_download(
            repo_id,
            local_dir=str(local_dir),
            allow_patterns=["*.json", "*.safetensors", "*.pt"],
            revision="main",
            local_dir_use_symlinks=False
        )

        return path

    def load_sae_config(self, local_path: str) -> Dict:
        """Load SAE configuration from downloaded files."""
        path = Path(local_path)

        # Try community format (cfg.json)
        cfg_path = path / "cfg.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f)

        # Try miStudio format (config.json)
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

        raise ValueError(f"No config found in {local_path}")

    def detect_format(self, local_path: str) -> str:
        """Detect SAE storage format."""
        path = Path(local_path)

        if (path / "cfg.json").exists() and (path / "sae_weights.safetensors").exists():
            return "community"
        elif (path / "config.json").exists() and (path / "model.safetensors").exists():
            return "mistudio"
        elif (path / "model.pt").exists():
            return "pytorch"
        else:
            raise ValueError(f"Unknown format in {local_path}")


class GemmaScopeService(HuggingFaceSAEService):
    """Specialized service for Gemma Scope SAEs."""

    GEMMA_SCOPE_REPOS = {
        "2b-res": "google/gemma-scope-2b-pt-res",
        "2b-mlp": "google/gemma-scope-2b-pt-mlp",
        "9b-res": "google/gemma-scope-9b-pt-res",
        "27b-res": "google/gemma-scope-27b-pt-res"
    }

    def list_available_saes(self, repo_key: str, layer: int) -> list:
        """List available SAE variants for a layer."""
        repo_id = self.GEMMA_SCOPE_REPOS.get(repo_key)
        if not repo_id:
            raise ValueError(f"Unknown Gemma Scope repo: {repo_key}")

        files = self.api.list_repo_files(repo_id)
        layer_prefix = f"layer_{layer}/"

        variants = []
        for f in files:
            if f.startswith(layer_prefix) and f.endswith("cfg.json"):
                parts = f.replace(layer_prefix, "").split("/")
                if len(parts) >= 2:
                    width = parts[0]  # e.g., "width_16k"
                    l0 = parts[1]  # e.g., "average_l0_82"
                    variants.append({
                        "width": width,
                        "l0": l0,
                        "subfolder": f.replace("/cfg.json", "")
                    })

        return variants

    def download_gemma_scope_sae(
        self,
        repo_key: str,
        layer: int,
        width: str,
        l0: str,
        progress_callback=None
    ) -> str:
        """Download specific Gemma Scope SAE variant."""
        repo_id = self.GEMMA_SCOPE_REPOS[repo_key]
        subfolder = f"layer_{layer}/{width}/{l0}"

        return self.download_sae(repo_id, subfolder, progress_callback)
```

#### `backend/src/ml/community_format.py`
```python
import json
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from typing import Dict, Any

class CommunityFormatLoader:
    """Load SAEs in SAELens Community Standard format."""

    @staticmethod
    def load(path: str) -> tuple:
        """Load SAE from community format.

        Returns:
            (state_dict, config)
        """
        path = Path(path)

        # Load config
        with open(path / "cfg.json") as f:
            config = json.load(f)

        # Load weights
        weights_path = path / "sae_weights.safetensors"
        if weights_path.exists():
            state_dict = load_file(str(weights_path))
        else:
            # Fallback to PyTorch format
            weights_path = path / "sae_weights.pt"
            state_dict = torch.load(str(weights_path))

        return state_dict, config

    @staticmethod
    def save(
        state_dict: Dict[str, torch.Tensor],
        config: Dict[str, Any],
        path: str
    ):
        """Save SAE in community format."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "cfg.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save weights
        save_file(state_dict, str(path / "sae_weights.safetensors"))

    @staticmethod
    def validate(path: str) -> bool:
        """Validate community format structure."""
        path = Path(path)
        required = ["cfg.json", "sae_weights.safetensors"]
        return all((path / f).exists() for f in required)


class MiStudioFormatLoader:
    """Load SAEs in miStudio native format."""

    @staticmethod
    def load(path: str) -> tuple:
        """Load SAE from miStudio format."""
        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        weights_path = path / "model.safetensors"
        if weights_path.exists():
            state_dict = load_file(str(weights_path))
        else:
            state_dict = torch.load(str(path / "model.pt"))

        return state_dict, config

    @staticmethod
    def to_community_format(state_dict: dict, config: dict) -> tuple:
        """Convert miStudio format to community format."""
        # Rename keys if needed
        key_mapping = {
            "encoder.weight": "W_enc",
            "encoder.bias": "b_enc",
            "decoder.weight": "W_dec",
            "decoder.bias": "b_dec"
        }

        new_state_dict = {}
        for old_key, tensor in state_dict.items():
            new_key = key_mapping.get(old_key, old_key)
            new_state_dict[new_key] = tensor

        # Convert config
        community_config = {
            "d_in": config.get("d_in") or config.get("input_dim"),
            "d_sae": config.get("d_sae") or config.get("hidden_dim"),
            "model_name": config.get("model_name") or config.get("model"),
            "hook_name": config.get("hook_name") or config.get("hook_point"),
            "architecture": config.get("architecture", "standard")
        }

        return new_state_dict, community_config
```

### 2.2 Frontend Files

#### `frontend/src/types/sae.ts`
```typescript
export interface SAE {
  id: string;
  name: string;
  source: 'trained' | 'huggingface' | 'gemma_scope' | 'upload';
  model_name: string;
  layer: number;
  hook_name?: string;
  d_in: number;
  d_sae: number;
  architecture: 'standard' | 'jumprelu' | 'skip' | 'transcoder';
  local_path?: string;
  repo_id?: string;
  feature_count?: number;
  created_at: string;
}

export interface SAEDownloadRequest {
  repo_id: string;
  name?: string;
  subfolder?: string;  // For Gemma Scope
}

export interface GemmaScopeVariant {
  width: string;
  l0: string;
  subfolder: string;
}
```

#### `frontend/src/stores/saesStore.ts`
```typescript
import { create } from 'zustand';
import { SAE, SAEDownloadRequest } from '../types/sae';
import { saesApi } from '../api/saes';

interface SAEsState {
  saes: SAE[];
  loading: boolean;
  error: string | null;

  // Filters
  sourceFilter: string | null;
  modelFilter: string | null;

  // Actions
  fetchSAEs: () => Promise<void>;
  downloadSAE: (request: SAEDownloadRequest) => Promise<SAE>;
  deleteSAE: (id: string) => Promise<void>;
  setSourceFilter: (source: string | null) => void;
  setModelFilter: (model: string | null) => void;

  // Selectors
  getFilteredSAEs: () => SAE[];
  getReadySAEs: () => SAE[];
  getSAEById: (id: string) => SAE | undefined;
}

export const useSAEsStore = create<SAEsState>((set, get) => ({
  saes: [],
  loading: false,
  error: null,
  sourceFilter: null,
  modelFilter: null,

  fetchSAEs: async () => {
    set({ loading: true, error: null });
    try {
      const saes = await saesApi.list();
      set({ saes, loading: false });
    } catch (error) {
      set({ error: 'Failed to fetch SAEs', loading: false });
    }
  },

  downloadSAE: async (request) => {
    const sae = await saesApi.download(request);
    set(state => ({
      saes: [sae, ...state.saes]
    }));
    return sae;
  },

  deleteSAE: async (id) => {
    await saesApi.delete(id);
    set(state => ({
      saes: state.saes.filter(s => s.id !== id)
    }));
  },

  setSourceFilter: (source) => set({ sourceFilter: source }),
  setModelFilter: (model) => set({ modelFilter: model }),

  getFilteredSAEs: () => {
    const { saes, sourceFilter, modelFilter } = get();
    return saes.filter(s => {
      if (sourceFilter && s.source !== sourceFilter) return false;
      if (modelFilter && s.model_name !== modelFilter) return false;
      return true;
    });
  },

  getReadySAEs: () => get().saes,

  getSAEById: (id) => get().saes.find(s => s.id === id)
}));
```

#### `frontend/src/components/saes/DownloadFromHF.tsx`
```typescript
import React, { useState, useEffect } from 'react';
import { useSAEsStore } from '../../stores/saesStore';
import { saesApi } from '../../api/saes';
import { GemmaScopeVariant } from '../../types/sae';

interface DownloadFromHFProps {
  onClose: () => void;
}

export function DownloadFromHF({ onClose }: DownloadFromHFProps) {
  const downloadSAE = useSAEsStore(s => s.downloadSAE);
  const [mode, setMode] = useState<'custom' | 'gemma_scope'>('custom');
  const [repoId, setRepoId] = useState('');
  const [loading, setLoading] = useState(false);

  // Gemma Scope specific
  const [gemmaScopeRepo, setGemmaScopeRepo] = useState('2b-res');
  const [layer, setLayer] = useState(12);
  const [variants, setVariants] = useState<GemmaScopeVariant[]>([]);
  const [selectedVariant, setSelectedVariant] = useState<GemmaScopeVariant | null>(null);

  // Load variants when Gemma Scope options change
  useEffect(() => {
    if (mode === 'gemma_scope') {
      loadVariants();
    }
  }, [mode, gemmaScopeRepo, layer]);

  const loadVariants = async () => {
    try {
      const data = await saesApi.getGemmaScopeVariants(gemmaScopeRepo, layer);
      setVariants(data);
      setSelectedVariant(data[0] || null);
    } catch (error) {
      console.error('Failed to load variants:', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      if (mode === 'custom') {
        await downloadSAE({ repo_id: repoId });
      } else if (selectedVariant) {
        await downloadSAE({
          repo_id: `google/gemma-scope-${gemmaScopeRepo}`,
          subfolder: selectedVariant.subfolder
        });
      }
      onClose();
    } catch (error) {
      console.error('Download failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Mode toggle */}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setMode('custom')}
          className={`px-4 py-2 rounded ${
            mode === 'custom' ? 'bg-emerald-600' : 'bg-slate-700'
          }`}
        >
          Custom Repository
        </button>
        <button
          type="button"
          onClick={() => setMode('gemma_scope')}
          className={`px-4 py-2 rounded ${
            mode === 'gemma_scope' ? 'bg-emerald-600' : 'bg-slate-700'
          }`}
        >
          Gemma Scope
        </button>
      </div>

      {mode === 'custom' ? (
        <div>
          <label className="block text-sm text-slate-400 mb-1">
            HuggingFace Repository ID
          </label>
          <input
            type="text"
            value={repoId}
            onChange={e => setRepoId(e.target.value)}
            placeholder="username/sae-repo"
            className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2"
            required
          />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">Model</label>
              <select
                value={gemmaScopeRepo}
                onChange={e => setGemmaScopeRepo(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2"
              >
                <option value="2b-res">Gemma 2B Residual</option>
                <option value="2b-mlp">Gemma 2B MLP</option>
                <option value="9b-res">Gemma 9B Residual</option>
                <option value="27b-res">Gemma 27B Residual</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">Layer</label>
              <input
                type="number"
                value={layer}
                onChange={e => setLayer(parseInt(e.target.value))}
                min={0}
                max={26}
                className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2"
              />
            </div>
          </div>

          {variants.length > 0 && (
            <div>
              <label className="block text-sm text-slate-400 mb-1">Variant</label>
              <select
                value={selectedVariant?.subfolder || ''}
                onChange={e => setSelectedVariant(
                  variants.find(v => v.subfolder === e.target.value) || null
                )}
                className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2"
              >
                {variants.map(v => (
                  <option key={v.subfolder} value={v.subfolder}>
                    {v.width} / {v.l0}
                  </option>
                ))}
              </select>
            </div>
          )}
        </>
      )}

      <div className="flex justify-end gap-2 pt-4">
        <button
          type="button"
          onClick={onClose}
          className="px-4 py-2 text-slate-400 hover:text-white"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={loading || (mode === 'custom' && !repoId)}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded disabled:opacity-50"
        >
          {loading ? 'Downloading...' : 'Download'}
        </button>
      </div>
    </form>
  );
}
```

---

## 3. Common Patterns

### 3.1 SAE Loading Abstraction
```python
# backend/src/ml/sae_loader.py
from typing import Union
from pathlib import Path
from src.ml.sparse_autoencoder import StandardSAE, JumpReLUSAE, SkipSAE
from src.ml.community_format import CommunityFormatLoader, MiStudioFormatLoader

def load_sae(path: str, device: str = "cuda") -> Union[StandardSAE, JumpReLUSAE, SkipSAE]:
    """Load SAE from any supported format."""
    path = Path(path)

    # Detect and load format
    if (path / "cfg.json").exists():
        state_dict, config = CommunityFormatLoader.load(str(path))
    elif (path / "config.json").exists():
        state_dict, config = MiStudioFormatLoader.load(str(path))
    else:
        raise ValueError(f"Unknown format at {path}")

    # Create appropriate SAE class
    arch = config.get("architecture", "standard")
    d_in = config["d_in"]
    d_sae = config["d_sae"]

    if arch == "standard":
        sae = StandardSAE(d_in, d_sae)
    elif arch == "jumprelu":
        sae = JumpReLUSAE(d_in, d_sae)
    elif arch == "skip":
        sae = SkipSAE(d_in, d_sae)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    sae.load_state_dict(state_dict)
    return sae.to(device)
```

### 3.2 Source-Agnostic SAE Selection
```typescript
// frontend/src/components/saes/SAESelector.tsx
import React from 'react';
import { useSAEsStore } from '../../stores/saesStore';

interface SAESelectorProps {
  value: string | null;
  onChange: (saeId: string) => void;
  modelFilter?: string;
}

export function SAESelector({ value, onChange, modelFilter }: SAESelectorProps) {
  const saes = useSAEsStore(s => s.saes);

  const filteredSAEs = modelFilter
    ? saes.filter(s => s.model_name === modelFilter)
    : saes;

  // Group by source
  const grouped = {
    trained: filteredSAEs.filter(s => s.source === 'trained'),
    external: filteredSAEs.filter(s => s.source !== 'trained')
  };

  return (
    <select
      value={value || ''}
      onChange={e => onChange(e.target.value)}
      className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2"
    >
      <option value="">Select SAE...</option>

      {grouped.trained.length > 0 && (
        <optgroup label="Trained SAEs">
          {grouped.trained.map(sae => (
            <option key={sae.id} value={sae.id}>
              {sae.name} (Layer {sae.layer})
            </option>
          ))}
        </optgroup>
      )}

      {grouped.external.length > 0 && (
        <optgroup label="External SAEs">
          {grouped.external.map(sae => (
            <option key={sae.id} value={sae.id}>
              {sae.name} ({sae.source})
            </option>
          ))}
        </optgroup>
      )}
    </select>
  );
}
```

---

## 4. Testing Strategy

### 4.1 Backend Tests
```python
# backend/tests/test_sae_manager.py
def test_list_all_saes(db_session):
    service = SAEManagerService(db_session)

    # Create test data
    # ...

    saes = service.list_all_saes()
    assert len(saes) > 0
    assert all('source' in s for s in saes)

def test_community_format_load():
    state_dict, config = CommunityFormatLoader.load("test_data/community_sae")
    assert "W_enc" in state_dict
    assert "d_in" in config

def test_format_conversion():
    state_dict, config = MiStudioFormatLoader.load("test_data/mistudio_sae")
    new_sd, new_cfg = MiStudioFormatLoader.to_community_format(state_dict, config)
    assert "W_enc" in new_sd
```

---

## 5. Common Pitfalls

### Pitfall 1: Weight Key Mismatches
```python
# WRONG - Assuming consistent key names
sae.load_state_dict(state_dict)  # KeyError!

# RIGHT - Map keys as needed
mapped = {key_mapping.get(k, k): v for k, v in state_dict.items()}
sae.load_state_dict(mapped)
```

### Pitfall 2: Dimension Inference
```python
# WRONG - Trusting config blindly
d_in = config["d_in"]

# RIGHT - Verify from weights
d_in = state_dict["W_enc"].shape[0]
d_sae = state_dict["W_enc"].shape[1]
```

---

## 6. Performance Tips

1. **Lazy Loading**
   ```python
   # Don't load SAE until needed
   class SAEWrapper:
       def __init__(self, path):
           self.path = path
           self._sae = None

       @property
       def sae(self):
           if self._sae is None:
               self._sae = load_sae(self.path)
           return self._sae
   ```

2. **Config Caching**
   ```python
   @lru_cache(maxsize=100)
   def get_sae_config(path: str) -> dict:
       # Only reads config, not weights
       with open(Path(path) / "cfg.json") as f:
           return json.load(f)
   ```

---

*Related: [PRD](../prds/005_FPRD|SAE_Management.md) | [TDD](../tdds/005_FTDD|SAE_Management.md) | [FTASKS](../tasks/005_FTASKS|SAE_Management.md)*
