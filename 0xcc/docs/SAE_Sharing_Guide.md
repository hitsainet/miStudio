# SAE Sharing Integration Guide

This guide covers the SAE (Sparse Autoencoder) sharing features in miStudio, including downloading SAEs from HuggingFace, importing from training, and uploading to HuggingFace.

## Overview

miStudio supports multiple SAE formats and sources:

- **HuggingFace Downloads**: Download community SAEs from HuggingFace Hub
- **Training Imports**: Import SAEs from completed miStudio training jobs
- **Local File Imports**: Import SAEs from local files
- **HuggingFace Uploads**: Share your trained SAEs with the community

## Supported Formats

### Community Standard (SAELens)
The de-facto standard format used by SAELens, Neuronpedia, and HuggingFace community SAEs.

**Directory Structure:**
```
sae_folder/
  cfg.json           # Configuration file
  sae_weights.safetensors  # Model weights
```

**Weight Naming Convention:**
- `W_enc` - Encoder weights [d_in, d_sae]
- `b_enc` - Encoder bias [d_sae]
- `W_dec` - Decoder weights [d_sae, d_in]
- `b_dec` - Decoder bias [d_in]

### miStudio Format
Native checkpoint format from miStudio training.

**Weight Naming Convention:**
- `model.encoder.weight` - Encoder weights [d_sae, d_in]
- `model.encoder.bias` - Encoder bias [d_sae]
- `model.decoder.weight` - Decoder weights [d_in, d_sae]
- `model.decoder_bias` - Decoder bias [d_in]

Note: Weight dimensions are transposed between formats due to PyTorch's Linear layer convention.

## API Endpoints

### Preview HuggingFace Repository
```http
POST /api/v1/saes/hf/preview
Content-Type: application/json

{
  "repo_id": "jbloom/GPT2-Small-SAEs",
  "access_token": "hf_xxxxx"  // Optional, for private repos
}
```

**Response:**
```json
{
  "repo_id": "jbloom/GPT2-Small-SAEs",
  "repo_type": "model",
  "description": "SAEs trained on GPT-2 Small",
  "files": [...],
  "sae_files": [...],
  "sae_paths": ["layer_0", "layer_6", ...],
  "model_name": "gpt2",
  "total_size_bytes": 123456789
}
```

### Download SAE
```http
POST /api/v1/saes/download
Content-Type: application/json

{
  "repo_id": "jbloom/GPT2-Small-SAEs",
  "filepath": "layer_6",
  "name": "GPT2 Layer 6 SAE",
  "description": "SAE for layer 6 residual stream",
  "model_name": "gpt2"
}
```

The download runs as a background Celery task with progress updates via WebSocket.

### Upload SAE
```http
POST /api/v1/saes/upload
Content-Type: application/json

{
  "sae_id": "sae_abc123",
  "repo_id": "username/my-sae",
  "filepath": "layer_6",
  "access_token": "hf_xxxxx",
  "create_repo": true,
  "private": false,
  "commit_message": "Initial upload"
}
```

### Import from Training
```http
POST /api/v1/saes/import/training
Content-Type: application/json

{
  "training_id": "train_abc123",
  "name": "My Trained SAE",
  "description": "SAE trained on custom dataset"
}
```

### Import from File
```http
POST /api/v1/saes/import/file
Content-Type: application/json

{
  "file_path": "/data/saes/my_sae",
  "name": "Local SAE",
  "format": "saelens",
  "model_name": "gpt2",
  "layer": 6
}
```

## WebSocket Progress Updates

Subscribe to SAE download/upload progress via WebSocket channels:

**Download Progress:**
- Channel: `sae/{sae_id}/download`
- Event: `sae:download`

**Upload Progress:**
- Channel: `sae/{sae_id}/upload`
- Event: `sae:upload`

**Event Data:**
```json
{
  "sae_id": "sae_abc123",
  "progress": 45.5,
  "status": "downloading",
  "message": "Downloading: 45.5%",
  "stage": "download"
}
```

**Status Values:**
- `pending` - Queued for processing
- `downloading` - Downloading from HuggingFace
- `converting` - Converting format
- `ready` - Complete and ready to use
- `error` - Failed with error

## Format Conversion

### SAELens to miStudio
Used automatically when downloading from HuggingFace:

```python
from src.services.sae_converter import SAEConverterService

checkpoint_path, metadata = SAEConverterService.saelens_to_mistudio(
    source_dir="/path/to/saelens/sae",
    target_dir="/path/to/output"
)
```

### miStudio to SAELens
Used automatically when uploading to HuggingFace:

```python
output_dir = SAEConverterService.mistudio_to_saelens(
    source_path="/path/to/mistudio/checkpoint",
    target_dir="/path/to/output",
    model_name="gpt2",
    layer=6,
    hyperparams={"l1_coefficient": 0.001}
)
```

## Model Inference from Dimensions

The converter can infer the model name from hidden dimensions:

| Dimension | Model |
|-----------|-------|
| 768 | gpt2 |
| 1024 | gpt2-medium |
| 1280 | gpt2-large |
| 1600 | gpt2-xl |
| 2048 | google/gemma-2b |
| 4096 | meta-llama/Llama-2-7b |

## Celery Task Configuration

SAE tasks run in the dedicated `sae` queue:

```bash
celery -A src.core.celery_app worker -Q sae --loglevel=info
```

Or include in the main worker:
```bash
celery -A src.core.celery_app worker -Q high_priority,datasets,processing,training,extraction,sae,low_priority -c 8 --loglevel=info
```

## Error Handling

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `Repository not found` | Invalid repo_id or private repo | Check repo_id, provide access_token |
| `Config file not found` | Not Community Standard format | Check SAE format compatibility |
| `Dimension mismatch` | Incompatible model | Verify model_name matches SAE |
| `Rate limit exceeded` | Too many HuggingFace requests | Wait and retry, use access token |

## Frontend Usage

### SAEsPanel Component
The SAEs panel provides:
- List of all SAEs with status indicators
- Download from HuggingFace with repository preview
- Import from training jobs
- Upload to HuggingFace
- Delete with file cleanup option

### useSAEWebSocket Hook
Subscribe to progress updates:

```typescript
import { useSAEWebSocket } from '../hooks/useSAEWebSocket';

// In component
useSAEWebSocket(saeIds);

// Updates automatically sync to saesStore
```

### saesStore Actions
```typescript
const { downloadSAE, uploadSAE, importFromTraining } = useSAEsStore();

// Download from HuggingFace
await downloadSAE({
  repo_id: 'jbloom/GPT2-Small-SAEs',
  filepath: 'layer_6',
  name: 'GPT2 Layer 6 SAE'
});
```
