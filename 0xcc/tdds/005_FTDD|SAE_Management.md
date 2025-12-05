# Technical Design Document: SAE Management

**Document ID:** 005_FTDD|SAE_Management
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related PRD:** [005_FPRD|SAE_Management](../prds/005_FPRD|SAE_Management.md)

---

## 1. System Architecture

### 1.1 SAE Sources
```
┌─────────────────────────────────────────────────────────────────┐
│                      SAE Sources                                 │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Trained    │  │  HuggingFace │  │     Gemma Scope      │  │
│  │  (internal)  │  │  (external)  │  │     (special)        │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         └─────────────────┼──────────────────────┘              │
│                           ▼                                     │
│                 ┌──────────────────┐                           │
│                 │  SAE Manager     │                           │
│                 │   Service        │                           │
│                 └─────────┬────────┘                           │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         ▼                 ▼                 ▼                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────────┐         │
│  │  Format    │   │  Storage   │   │   Database     │         │
│  │  Converter │   │  Manager   │   │   Records      │         │
│  └────────────┘   └────────────┘   └────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Format Handling
```
┌─────────────────────────────────────────────────────────────────┐
│                    Format Detection Flow                         │
│                                                                  │
│  Input SAE Directory                                            │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────┐                                │
│  │  Check for cfg.json         │                                │
│  │  + sae_weights.safetensors  │                                │
│  └─────────────┬───────────────┘                                │
│                │                                                 │
│         ┌──────┴──────┐                                         │
│         ▼             ▼                                         │
│  ┌────────────┐  ┌────────────┐                                │
│  │ Community  │  │  miStudio  │                                │
│  │ Standard   │  │  Native    │                                │
│  │ (SAELens)  │  │  Format    │                                │
│  └────────────┘  └────────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Database Schema

### 2.1 External SAE Table
```sql
CREATE TABLE external_saes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    layer INTEGER NOT NULL,
    hook_name VARCHAR(255),
    d_in INTEGER NOT NULL,
    d_sae INTEGER NOT NULL,
    architecture VARCHAR(50),        -- standard, jumprelu, skip, transcoder
    source VARCHAR(50) NOT NULL,     -- huggingface, gemma_scope, upload, trained
    repo_id VARCHAR(255),            -- HuggingFace repo if downloaded
    local_path VARCHAR(500) NOT NULL,
    format VARCHAR(50) NOT NULL,     -- community, mistudio
    training_id UUID REFERENCES trainings(id),  -- Link if from training
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT external_saes_source_check CHECK (
        source IN ('huggingface', 'gemma_scope', 'upload', 'trained')
    ),
    CONSTRAINT external_saes_format_check CHECK (
        format IN ('community', 'mistudio')
    )
);

CREATE INDEX idx_external_saes_model ON external_saes(model_name);
CREATE INDEX idx_external_saes_source ON external_saes(source);
```

---

## 3. SAE Format Specifications

### 3.1 Community Standard (SAELens Compatible)
```
sae_directory/
├── cfg.json                    # Configuration
└── sae_weights.safetensors     # Weights

# cfg.json
{
  "d_in": 2304,
  "d_sae": 18432,
  "dtype": "float32",
  "model_name": "google/gemma-2-2b",
  "hook_name": "blocks.12.hook_resid_post",
  "architecture": "standard",
  "normalize_activations": "constant_norm_rescale",
  "activation_fn": "relu",
  "b_dec_init": "zeros"
}
```

### 3.2 miStudio Native Format
```
sae_directory/
├── config.json                 # Extended config
├── model.safetensors           # Weights
└── training_metadata.json      # Training history

# config.json (superset of cfg.json)
{
  "d_in": 2304,
  "d_sae": 18432,
  "dtype": "float32",
  "model_name": "google/gemma-2-2b",
  "hook_name": "blocks.12.hook_resid_post",
  "architecture": "standard",
  "training_id": "uuid",
  "final_loss": 0.0234,
  "training_steps": 100000,
  "hyperparameters": {...}
}
```

---

## 4. Service Layer

### 4.1 SAEManagerService
```python
class SAEManagerService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.hf_service = HuggingFaceSAEService()
        self.converter = SAEConverter()

    async def list_saes(self, filters: SAEFilters) -> List[ExternalSAE]:
        """List all SAEs with optional filtering."""
        query = select(ExternalSAE)

        if filters.source:
            query = query.where(ExternalSAE.source == filters.source)
        if filters.model_name:
            query = query.where(ExternalSAE.model_name == filters.model_name)
        if filters.layer is not None:
            query = query.where(ExternalSAE.layer == filters.layer)

        return await self.db.execute(query)

    async def download_from_hf(self, repo_id: str, config: dict) -> ExternalSAE:
        """Download SAE from HuggingFace."""
        # Download files
        local_path = await self.hf_service.download(repo_id, config)

        # Detect format
        format_type = self.converter.detect_format(local_path)

        # Parse config
        sae_config = self.converter.load_config(local_path, format_type)

        # Create database record
        sae = ExternalSAE(
            name=config.get('name', repo_id.split('/')[-1]),
            model_name=sae_config['model_name'],
            layer=sae_config['layer'],
            d_in=sae_config['d_in'],
            d_sae=sae_config['d_sae'],
            source='huggingface',
            repo_id=repo_id,
            local_path=str(local_path),
            format=format_type
        )

        self.db.add(sae)
        await self.db.commit()
        return sae

    async def link_to_training(self, sae_id: UUID, training_id: UUID):
        """Link external SAE to its training record."""
        sae = await self.get_sae(sae_id)
        sae.training_id = training_id
        sae.source = 'trained'
        await self.db.commit()
```

### 4.2 HuggingFaceSAEService
```python
class HuggingFaceSAEService:
    async def download(self, repo_id: str, config: dict) -> Path:
        """Download SAE from HuggingFace Hub."""
        target_dir = Path(settings.data_dir) / 'saes' / 'external' / repo_id.replace('/', '_')
        target_dir.mkdir(parents=True, exist_ok=True)

        # Download all SAE files
        files = ['cfg.json', 'sae_weights.safetensors']
        for file in files:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=target_dir,
                token=config.get('hf_token')
            )

        return target_dir

    async def download_gemma_scope(self, layer: int, width: str, l0: str) -> Path:
        """Download specific Gemma Scope SAE."""
        repo_id = "google/gemma-scope-2b-pt-res"
        subfolder = f"layer_{layer}/width_{width}/average_l0_{l0}"

        # Gemma Scope has nested structure
        target_dir = Path(settings.data_dir) / 'saes' / 'gemma_scope' / f"L{layer}_{width}_{l0}"

        hf_hub_download(
            repo_id=repo_id,
            filename="cfg.json",
            subfolder=subfolder,
            local_dir=target_dir
        )
        hf_hub_download(
            repo_id=repo_id,
            filename="sae_weights.safetensors",
            subfolder=subfolder,
            local_dir=target_dir
        )

        return target_dir
```

### 4.3 SAEConverter
```python
class SAEConverter:
    def detect_format(self, path: Path) -> str:
        """Detect SAE format from directory contents."""
        if (path / 'cfg.json').exists() and (path / 'sae_weights.safetensors').exists():
            return 'community'
        elif (path / 'config.json').exists() and (path / 'model.safetensors').exists():
            return 'mistudio'
        else:
            raise ValueError(f"Unknown SAE format in {path}")

    def load_config(self, path: Path, format: str) -> dict:
        """Load and normalize SAE config."""
        if format == 'community':
            with open(path / 'cfg.json') as f:
                config = json.load(f)
            return self._normalize_community_config(config)
        else:
            with open(path / 'config.json') as f:
                return json.load(f)

    def convert_to_community(self, path: Path) -> Path:
        """Convert miStudio format to community standard."""
        config = self.load_config(path, 'mistudio')

        # Write cfg.json
        community_config = {
            'd_in': config['d_in'],
            'd_sae': config['d_sae'],
            'dtype': config.get('dtype', 'float32'),
            'model_name': config['model_name'],
            'hook_name': config['hook_name'],
            'architecture': config.get('architecture', 'standard')
        }

        with open(path / 'cfg.json', 'w') as f:
            json.dump(community_config, f, indent=2)

        # Rename weights if needed
        if (path / 'model.safetensors').exists():
            shutil.copy(path / 'model.safetensors', path / 'sae_weights.safetensors')

        return path

    def _normalize_community_config(self, config: dict) -> dict:
        """Normalize community config to internal format."""
        # Extract layer from hook_name: "blocks.12.hook_resid_post" -> 12
        hook = config.get('hook_name', '')
        layer_match = re.search(r'blocks\.(\d+)', hook)
        layer = int(layer_match.group(1)) if layer_match else 0

        return {
            'd_in': config['d_in'],
            'd_sae': config['d_sae'],
            'model_name': config.get('model_name', 'unknown'),
            'hook_name': hook,
            'layer': layer,
            'architecture': config.get('architecture', 'standard'),
            'dtype': config.get('dtype', 'float32')
        }
```

---

## 5. API Design

### 5.1 Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/saes` | List all SAEs |
| POST | `/saes` | Create SAE record (manual) |
| GET | `/saes/{id}` | Get SAE details |
| DELETE | `/saes/{id}` | Delete SAE + files |
| POST | `/saes/download-hf` | Download from HuggingFace |
| POST | `/saes/{id}/convert` | Convert to community format |
| GET | `/saes/{id}/config` | Get SAE configuration |

### 5.2 Download Request
```python
class SAEDownloadRequest(BaseModel):
    repo_id: str
    name: Optional[str] = None
    hf_token: Optional[str] = None

    # For Gemma Scope
    layer: Optional[int] = None
    width: Optional[str] = None
    l0: Optional[str] = None
```

---

## 6. Frontend Components

### 6.1 SAEsPanel
```typescript
// Main SAEs panel with filtering
<SAEsPanel>
  <FilterBar>
    <SourceFilter options={['all', 'trained', 'huggingface', 'gemma_scope']} />
    <ModelFilter options={models} />
    <SearchInput />
  </FilterBar>

  <SAEGrid>
    {saes.map(sae => <SAECard key={sae.id} sae={sae} />)}
  </SAEGrid>

  <DownloadFromHFButton onClick={openModal} />
</SAEsPanel>
```

### 6.2 SAECard
```typescript
interface SAECardProps {
  sae: ExternalSAE;
  onExtract: () => void;
  onSteering: () => void;
  onExport: () => void;
  onDelete: () => void;
}

// Displays:
// - Source badge (Trained, HuggingFace, Gemma Scope)
// - Model name and layer
// - Dimensions (d_in → d_sae)
// - Architecture type
// - Feature count if extracted
// - Action buttons
```

---

## 7. File Storage

```
DATA_DIR/
└── saes/
    ├── trained/
    │   └── {training_id}/
    │       ├── cfg.json
    │       └── sae_weights.safetensors
    ├── external/
    │   └── {repo_id_normalized}/
    │       ├── cfg.json
    │       └── sae_weights.safetensors
    └── gemma_scope/
        └── L{layer}_{width}_{l0}/
            ├── cfg.json
            └── sae_weights.safetensors
```

---

## 8. Gemma Scope Integration

### 8.1 Repository Structure
```
google/gemma-scope-2b-pt-res/
├── layer_0/
│   ├── width_16k/
│   │   ├── average_l0_82/
│   │   │   ├── cfg.json
│   │   │   └── sae_weights.safetensors
│   │   ├── average_l0_105/
│   │   └── ...
│   └── width_32k/
├── layer_12/
└── layer_25/
```

### 8.2 Selection UI
```typescript
// GemmaScopeSelector component
<GemmaScopeSelector>
  <LayerSelect
    options={[0, 6, 12, 18, 25]}
    onChange={setLayer}
  />
  <WidthSelect
    options={['16k', '32k', '65k']}
    onChange={setWidth}
  />
  <L0Select
    options={['82', '105', '210', '420']}
    onChange={setL0}
  />
  <DownloadButton onClick={download} />
</GemmaScopeSelector>
```

---

*Related: [PRD](../prds/005_FPRD|SAE_Management.md) | [TID](../tids/005_FTID|SAE_Management.md) | [FTASKS](../tasks/005_FTASKS|SAE_Management.md)*
