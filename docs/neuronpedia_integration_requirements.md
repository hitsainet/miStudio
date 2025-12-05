# miStudio → Neuronpedia Integration: Technical Requirements

## Executive Summary

This document outlines the technical requirements for miStudio to produce SAE outputs (weights, feature activations, and metadata) that are compatible with the Neuronpedia platform. Neuronpedia is the leading open-source interpretability platform for hosting and exploring SAE features, and integration would allow miStudio users to share their trained SAEs with the broader mechanistic interpretability community.

---

## 1. Integration Architecture Overview

### 1.1 The Neuronpedia Ecosystem

Neuronpedia consists of several interconnected components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neuronpedia Platform                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Webapp     │  │  Inference   │  │    Autointerp        │  │
│  │  (Frontend + │  │   Server     │  │     Server           │  │
│  │     API)     │  │              │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         │                │                     │                 │
│         └────────────────┼─────────────────────┘                 │
│                          │                                       │
│                  ┌───────▼───────┐                              │
│                  │   Database    │                              │
│                  │  (PostgreSQL) │                              │
│                  └───────────────┘                              │
│                          │                                       │
│                  ┌───────▼───────┐                              │
│                  │  S3 Storage   │                              │
│                  │  (Datasets)   │                              │
│                  └───────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Integration Pathways

miStudio has **two primary pathways** to Neuronpedia:

#### Pathway A: SAELens → SAEDashboard → Neuronpedia (Recommended)
1. Export SAE in SAELens-compatible format to HuggingFace
2. Register SAE in SAELens `pretrained_saes.yaml`
3. Generate dashboard data using SAEDashboard's `NeuronpediaRunner`
4. Convert via `convert-saedashboard-to-neuronpedia.py`
5. Upload to Neuronpedia S3 bucket (requires coordination with Neuronpedia team)
6. Import via Neuronpedia admin panel

#### Pathway B: Direct API Upload (Future)
- Neuronpedia is working on automated upload processes
- Currently requires manual coordination with Neuronpedia team
- Fill out the SAE upload form: https://docs.neuronpedia.org/upload-saes

---

## 2. SAE Weight Format Requirements

### 2.1 SAELens-Compatible Format (Required)

miStudio SAEs must be exportable in SAELens format to work with the Neuronpedia pipeline.

#### Required Files Structure
```
your_sae_directory/
├── cfg.json              # Configuration file
├── sae_weights.safetensors  # Model weights (or .pt)
└── sparsity.safetensors  # (Optional) Sparsity statistics
```

#### cfg.json Schema
```json
{
  "architecture": "standard",
  "d_in": 768,
  "d_sae": 24576,
  "dtype": "float32",
  "device": "cuda",
  "model_name": "gpt2-small",
  "hook_name": "blocks.6.hook_resid_pre",
  "hook_layer": 6,
  "hook_head_index": null,
  "activation_fn_str": "relu",
  "activation_fn_kwargs": {},
  "apply_b_dec_to_input": true,
  "finetuning_scaling_factor": false,
  "sae_lens_training_version": "2.0.0",
  "prepend_bos": true,
  "dataset_path": "Skylion007/openwebtext",
  "dataset_trust_remote_code": true,
  "context_size": 128,
  "normalize_activations": "none",
  "model_from_pretrained_kwargs": {}
}
```

#### Key Configuration Fields

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `architecture` | string | SAE architecture type ("standard", "topk", "jumprelu", etc.) | Yes |
| `d_in` | int | Input dimension (model's d_model) | Yes |
| `d_sae` | int | SAE hidden dimension (number of features) | Yes |
| `dtype` | string | Data type ("float32", "bfloat16") | Yes |
| `model_name` | string | TransformerLens model identifier | Yes |
| `hook_name` | string | TransformerLens hook point | Yes |
| `hook_layer` | int | Layer index | Yes |
| `activation_fn_str` | string | Activation function name | Yes |
| `prepend_bos` | bool | Whether BOS token was prepended during training | Yes |
| `context_size` | int | Training context length | Yes |
| `normalize_activations` | string | Normalization method ("none", "expected_average_only_in", etc.) | Yes |

#### sae_weights.safetensors Contents
```python
# Required weight tensors
{
    "W_enc": tensor,     # Shape: (d_in, d_sae) - Encoder weights
    "W_dec": tensor,     # Shape: (d_sae, d_in) - Decoder weights
    "b_enc": tensor,     # Shape: (d_sae,) - Encoder bias
    "b_dec": tensor,     # Shape: (d_in,) - Decoder bias
}

# Optional for specific architectures
{
    "threshold": tensor,  # For JumpReLU SAEs
    "scaling_factor": tensor,  # For finetuning
}
```

### 2.2 Model Compatibility Requirements

Neuronpedia relies on **TransformerLens** for model loading. Your SAE must be compatible with:
- TransformerLens model identifiers (see: https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html)
- Standard hook points (e.g., `blocks.{layer}.hook_resid_pre`, `blocks.{layer}.hook_resid_post`)

#### Supported Hook Points
```python
# Residual stream
"blocks.{layer}.hook_resid_pre"
"blocks.{layer}.hook_resid_post"

# MLP
"blocks.{layer}.hook_mlp_out"
"blocks.{layer}.mlp.hook_pre"
"blocks.{layer}.mlp.hook_post"

# Attention
"blocks.{layer}.attn.hook_z"
"blocks.{layer}.attn.hook_q"
"blocks.{layer}.attn.hook_k"
"blocks.{layer}.attn.hook_v"
```

---

## 3. Feature Activation Data Requirements

### 3.1 Dashboard Generation Process

Neuronpedia displays feature dashboards that require pre-computed activation data. This data is generated using **SAEDashboard's NeuronpediaRunner**.

#### Generation Command
```bash
poetry run neuronpedia-runner \
    --sae-set="your-sae-set-name" \
    --sae-path="blocks.6.hook_resid_pre" \
    --np-set-name="your-neuronpedia-set-name" \
    --dataset-path="monology/pile-uncopyrighted" \
    --output-dir="neuronpedia_outputs/" \
    --sae_dtype="float32" \
    --model_dtype="bfloat16" \
    --sparsity-threshold=1 \
    --n-prompts=24576 \
    --n-tokens-in-prompt=128 \
    --n-features-per-batch=128 \
    --n-prompts-in-forward-pass=128
```

#### Key Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `n-prompts` | Total prompts to sample from dataset | 24576 |
| `n-tokens-in-prompt` | Context length per prompt | 128 |
| `n-features-per-batch` | Features processed simultaneously | 128 |
| `sparsity-threshold` | Minimum activation frequency | 1 |

### 3.2 Feature Data Structure

Each feature requires the following data:

#### Top Activating Examples
```json
{
  "feature_index": 7650,
  "activations": [
    {
      "tokens": ["In", " the", " beginning", ",", " God", " created"],
      "values": [0.0, 0.1, 0.5, 0.2, 3.8, 2.1],
      "max_value": 3.8,
      "max_token_index": 4
    },
    // ... more activation examples (typically 20+)
  ]
}
```

#### Logit Lens Data
```json
{
  "feature_index": 7650,
  "top_positive_logits": [
    {"token": "Bible", "value": 2.45},
    {"token": "God", "value": 2.31},
    {"token": "prayer", "value": 2.18},
    // ... typically top 10-20
  ],
  "top_negative_logits": [
    {"token": "scientific", "value": -1.82},
    {"token": "research", "value": -1.65},
    // ... typically top 10-20
  ]
}
```

#### Activation Density Histogram
```json
{
  "feature_index": 7650,
  "histogram_bins": [0.0, 0.5, 1.0, 1.5, 2.0, ...],
  "histogram_counts": [15234, 892, 234, 89, 45, ...],
  "frequency": 0.023,  // Fraction of tokens where feature activates
  "max_activation": 12.5
}
```

### 3.3 Neuronpedia-Specific Output Format

After running SAEDashboard, you must convert to Neuronpedia's import format using:
```bash
cd neuronpedia/utils/neuronpedia-utils
python convert-saedashboard-to-neuronpedia.py
```

The conversion produces JSON files structured for Neuronpedia's database import.

---

## 4. Metadata and Identification Requirements

### 4.1 Neuronpedia ID Structure

Every SAE on Neuronpedia has a unique identifier:
```
[MODEL_ID]@[SAE_ID]
```

Example: `gpt2-small@6-res-jb`

#### Feature ID Structure
```
[MODEL_ID]@[SAE_ID]:[FEATURE_INDEX]
```

Example: `gpt2-small@6-res-jb:7650`

### 4.2 SAELens pretrained_saes.yaml Entry

To integrate with the standard pipeline, register your SAE in SAELens:

```yaml
your-unique-sae-set-id:
  # Unique ID for your set of SAEs
  conversion_func: null  # null if already SAELens-compatible
  
  links:
    model: https://huggingface.co/google/gemma-2-2b  # Optional
  
  model: gemma-2-2b  # TransformerLens model ID
  repo_id: your-username/your-sae-repo  # HuggingFace repo
  
  saes:
    - id: blocks.0.hook_resid_post
      path: standard/blocks.0.hook_resid_post
      l0: 45.0  # Average L0 sparsity
      neuronpedia: gemma-2-2b/0-your-sae-name  # Expected Neuronpedia URL path
    
    - id: blocks.1.hook_resid_post
      path: standard/blocks.1.hook_resid_post
      l0: 47.0
      neuronpedia: gemma-2-2b/1-your-sae-name
    
    # ... additional layers
```

### 4.3 Required Metadata Fields

| Field | Description | Example |
|-------|-------------|---------|
| `model` | TransformerLens model ID | `gemma-2-2b` |
| `repo_id` | HuggingFace repository path | `your-org/your-sae` |
| `l0` | Average L0 sparsity (features active per token) | `45.0` |
| `neuronpedia` | Expected Neuronpedia URL path | `gemma-2-2b/12-your-sae` |
| `path` | Path within HF repo to SAE weights | `layer_12/canonical` |

---

## 5. Automatic Interpretability (AutoInterp) Integration

### 5.1 Explanation Data Structure

Neuronpedia stores auto-generated explanations for each feature:

```json
{
  "feature_index": 7650,
  "explanations": [
    {
      "description": "religious and biblical text",
      "method": "oai_token-act-pair",
      "model": "gpt-4",
      "score": 0.85,
      "created_at": "2024-01-15T10:30:00Z"
    },
    {
      "description": "Bible references",
      "method": "np_max-act-logits", 
      "model": "claude-3.5-sonnet",
      "score": 0.92,
      "created_at": "2024-03-20T14:22:00Z"
    }
  ]
}
```

### 5.2 Scoring API Format

Neuronpedia's scorer expects:

```json
{
  "explanation": "fast animals",
  "secret": "",
  "activations": [
    {
      "tokens": ["the ", " quick", " brown", " fox"],
      "values": [0, 2, 0, 2.5]
    },
    {
      "tokens": ["spotted ", " leopard", " sprints", " at"],
      "values": [0.5, 2.5, 1.5, 0]
    }
  ]
}
```

Response:
```json
{
  "score": 0.8896,
  "simulations": [...]
}
```

### 5.3 AutoInterp Methods Supported

| Method | Description |
|--------|-------------|
| `oai_token-act-pair` | OpenAI's original method (Bills et al.) |
| `np_max-act-logits` | Neuronpedia's method using top logits |
| `eleuther_delphi` | EleutherAI's Delphi library |

---

## 6. API Integration Points

### 6.1 Feature API (Read)

```http
GET /api/feature/{model_id}/{sae_id}/{feature_index}
```

Response contains:
- Feature activations
- Logit data
- Explanations
- Statistics

### 6.2 Steering API

```http
POST /api/steer
Content-Type: application/json

{
  "prompt": "The most iconic structure on Earth is",
  "modelId": "gemma-2b",
  "features": [
    {
      "modelId": "gemma-2b",
      "layer": "6-res-jb",
      "index": 10200,
      "strength": 5
    }
  ],
  "temperature": 0.2,
  "n_tokens": 16,
  "freq_penalty": 1.0,
  "seed": 16,
  "strength_multiplier": 4
}
```

### 6.3 Search API

```http
GET /api/explanation/export?modelId={model}&saeId={sae_id}
```

Returns all explanations for an SAE set.

---

## 7. Implementation Checklist for miStudio

### Phase 1: SAE Export Format
- [ ] Implement SAELens-compatible `cfg.json` export
- [ ] Export weights in `.safetensors` format with correct tensor names
- [ ] Support all required configuration fields
- [ ] Validate hook point naming matches TransformerLens conventions

### Phase 2: HuggingFace Integration
- [ ] Implement HuggingFace upload functionality for SAEs
- [ ] Generate proper repository structure
- [ ] Create `pretrained_saes.yaml` entries automatically
- [ ] Support PR creation to SAELens repository (optional)

### Phase 3: Activation Data Generation
- [ ] Generate top activating examples per feature
- [ ] Compute logit lens data (top positive/negative logits)
- [ ] Calculate activation density histograms
- [ ] Compute feature statistics (frequency, max activation, L0)

### Phase 4: Dashboard Export
- [ ] Integrate with SAEDashboard's `NeuronpediaRunner`
- [ ] OR implement equivalent data generation natively
- [ ] Support Neuronpedia's JSON import format
- [ ] Handle large-scale feature counts (16k-131k features)

### Phase 5: AutoInterp Integration (Optional)
- [ ] Generate explanations compatible with Neuronpedia format
- [ ] Support explanation scoring API format
- [ ] Integrate with EleutherAI's Delphi library

### Phase 6: Direct Upload (Future)
- [ ] Implement Neuronpedia's upload API (when available)
- [ ] Support incremental updates
- [ ] Handle authentication and rate limiting

---

## 8. Technical Dependencies

### Required Libraries
```python
# Core
sae-lens>=6.0.0
sae-dashboard>=1.0.0
transformer-lens>=2.0.0
safetensors>=0.4.0

# For dashboard generation
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0

# For HuggingFace integration
huggingface-hub>=0.20.0
datasets>=2.0.0

# For autointerp (optional)
delphi  # EleutherAI's library
openai>=1.0.0  # For OpenAI-based explanations
```

### Infrastructure Requirements
- **GPU**: Recommended for activation generation (CUDA or MPS)
- **Memory**: 16GB+ RAM for large models
- **Storage**: Significant disk space for activation caching
- **Network**: HuggingFace Hub access

---

## 9. Coordination with Neuronpedia Team

### Current Upload Process
1. Fill out SAE upload form: https://docs.neuronpedia.org/upload-saes
2. Neuronpedia team reviews and coordinates upload
3. Dashboard data uploaded to Neuronpedia S3 bucket
4. Data imported via admin panel

### Contact
- Slack: #neuronpedia channel on OSMI Slack
- Email: johnny@neuronpedia.org
- GitHub Issues: https://github.com/hijohnnylin/neuronpedia/issues

---

## 10. Example: Complete Export Workflow

```python
# Step 1: Export SAE from miStudio in SAELens format
from mistudio.export import export_sae_for_neuronpedia

export_sae_for_neuronpedia(
    sae=trained_sae,
    output_dir="./exported_sae",
    model_name="gpt2-small",
    hook_name="blocks.6.hook_resid_pre",
    dataset_path="Skylion007/openwebtext"
)

# Step 2: Upload to HuggingFace
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./exported_sae",
    repo_id="your-org/your-sae-gpt2-small",
    repo_type="model"
)

# Step 3: Generate Neuronpedia dashboard data
from sae_dashboard.neuronpedia.neuronpedia_runner import NeuronpediaRunner
from sae_dashboard.neuronpedia.neuronpedia_runner_config import NeuronpediaRunnerConfig

config = NeuronpediaRunnerConfig(
    sae_set="your-sae-set",
    sae_path="blocks.6.hook_resid_pre",
    np_set_name="your-np-set",
    huggingface_dataset_path="Skylion007/openwebtext",
    n_prompts_total=24576,
    n_features_at_a_time=128
)

runner = NeuronpediaRunner(config)
runner.run()

# Step 4: Convert to Neuronpedia format
# Run: python convert-saedashboard-to-neuronpedia.py

# Step 5: Contact Neuronpedia team for upload
```

---

## Appendix A: Neuronpedia Data Export URLs

- **Public datasets**: https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/
- **Legacy exports**: https://neuronpedia-exports.s3.amazonaws.com/index.html
- **API documentation**: https://neuronpedia.org/api-doc

## Appendix B: TransformerLens Model IDs

Common model IDs for Neuronpedia integration:
- `gpt2-small`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- `pythia-70m`, `pythia-160m`, `pythia-410m`, etc.
- `gemma-2b`, `gemma-2-2b`, `gemma-2-2b-it`
- `llama-3.1-8b`, `llama-3.1-8b-it`

Full list: https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html

## Appendix C: Feature Dashboard Components

A complete Neuronpedia feature dashboard includes:
1. **Feature Index**: Unique identifier within SAE
2. **Explanations**: Auto-generated and user-submitted
3. **Activation Density Plot**: Histogram of activation values
4. **Top Activations**: Text samples with highest feature activation
5. **Logit Lens**: Top positive and negative output logits
6. **Live Testing**: Interactive activation testing
7. **Lists**: Curated feature collections
8. **Comments**: User discussions

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Based on: Neuronpedia open-source codebase (March 2025 release)*
