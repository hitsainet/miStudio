# Download Gemma Scope SAEs for miStudio
## Exact SAEs Available on Neuronpedia - Direct Download Guide

**Purpose:** Download the exact same SAEs that Neuronpedia uses to enable direct comparison of steering results between miStudio and Neuronpedia.

**Date:** November 26, 2025

---

## Quick Reference: Neuronpedia SAE Identifiers

### Gemma 2 2B Residual Stream SAEs

| Neuronpedia ID | HuggingFace Path | Layer | Width | L0 |
|----------------|------------------|-------|-------|-----|
| `20-gemmascope-res-16k` | `google/gemma-scope-2b-pt-res/layer_20/width_16k/average_l0_71` | 20 | 16,384 | 71 |
| `12-gemmascope-res-16k` | `google/gemma-scope-2b-pt-res/layer_12/width_16k/average_l0_71` | 12 | 16,384 | 71 |
| `2-gemmascope-res-16k` | `google/gemma-scope-2b-pt-res/layer_2/width_16k/average_l0_71` | 2 | 16,384 | 71 |

**Key Details from Neuronpedia:**
- **Configuration:** `google/gemma-scope-2b-pt-res/layer_20/width_16k/average_l0_71`
- **Hook Name:** `blocks.20.hook_resid_post`
- **Architecture:** JumpReLU
- **Features:** 16,384
- **Data Type:** float32
- **Context Size:** 1,024 tokens
- **Dataset:** `monology/pile-uncopyrighted`
- **Prompts:** 36,864 prompts, 128 tokens each

---

## Download Method: HuggingFace Hub

### Step 1: Install Required Packages

```bash
# Install HuggingFace Hub
pip install huggingface-hub

# Optional: For direct SAE loading
pip install safetensors torch
```

### Step 2: Download Specific SAE

```python
from huggingface_hub import hf_hub_download
import os

def download_gemma_scope_sae(
    layer: int = 20,
    width: str = "16k",
    l0: str = "71",
    save_dir: str = "./saes/gemma-2-2b"
):
    """
    Download a specific Gemma Scope SAE that matches Neuronpedia.
    
    Args:
        layer: Layer number (0-25 for Gemma 2 2B)
        width: SAE width ('16k', '65k', '131k', etc.)
        l0: Average L0 sparsity ('71', '50', etc.)
        save_dir: Local directory to save SAE
    
    Returns:
        Path to downloaded SAE checkpoint
    """
    # Construct the path
    repo_id = "google/gemma-scope-2b-pt-res"
    subfolder = f"layer_{layer}/width_{width}/average_l0_{l0}"
    
    # Download all files in the subfolder
    files_to_download = [
        "params.npz",           # SAE weights
        "config.json",          # Configuration
        "sparsity.safetensors", # Sparsity info (if available)
    ]
    
    downloaded_paths = {}
    
    for filename in files_to_download:
        try:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}",
                local_dir=save_dir,
                local_dir_use_symlinks=False
            )
            downloaded_paths[filename] = file_path
            print(f"Downloaded: {filename} -> {file_path}")
        except Exception as e:
            print(f"Note: {filename} not found (may not exist): {e}")
    
    return downloaded_paths

# Example: Download the exact SAE used in Neuronpedia
# This is the "20-gemmascope-res-16k" SAE
paths = download_gemma_scope_sae(
    layer=20,
    width="16k",
    l0="71"
)

print("\nDownloaded SAE files:")
for name, path in paths.items():
    print(f"  {name}: {path}")
```

### Step 3: Verify Download

```python
import numpy as np
import json

def verify_sae_download(sae_dir: str, layer: int = 20):
    """Verify the downloaded SAE matches Neuronpedia specs."""
    
    # Load params
    params_path = f"{sae_dir}/layer_{layer}/width_16k/average_l0_71/params.npz"
    params = np.load(params_path)
    
    print("SAE Verification:")
    print(f"  Available arrays: {list(params.keys())}")
    
    # Check dimensions
    if 'W_dec' in params:
        W_dec = params['W_dec']
        print(f"  W_dec shape: {W_dec.shape}")
        print(f"  Expected: (2304, 16384) for Gemma 2 2B")
        
    if 'W_enc' in params:
        W_enc = params['W_enc']
        print(f"  W_enc shape: {W_enc.shape}")
        print(f"  Expected: (16384, 2304)")
    
    # Load config if available
    config_path = f"{sae_dir}/layer_{layer}/width_16k/average_l0_71/config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\nConfiguration:")
        print(f"  {json.dumps(config, indent=2)}")
    except FileNotFoundError:
        print("\nConfig file not found (may not be included)")
    
    return params

# Verify the download
params = verify_sae_download("./saes/gemma-2-2b", layer=20)
```

---

## Loading the SAE in miStudio (Without SAELens)

### Custom Loader Implementation

```python
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class SAEConfig:
    """Configuration for a Gemma Scope SAE."""
    d_model: int = 2304  # Gemma 2 2B hidden size
    d_sae: int = 16384   # SAE width
    hook_name: str = "blocks.20.hook_resid_post"
    hook_layer: int = 20
    architecture: str = "jumprelu"
    dtype: torch.dtype = torch.float32

class JumpReLUSAE(torch.nn.Module):
    """
    JumpReLU SAE implementation compatible with Gemma Scope.
    """
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        # Initialize parameters (will be loaded from checkpoint)
        self.W_enc = torch.nn.Parameter(
            torch.zeros(config.d_sae, config.d_model, dtype=config.dtype)
        )
        self.W_dec = torch.nn.Parameter(
            torch.zeros(config.d_model, config.d_sae, dtype=config.dtype)
        )
        self.b_enc = torch.nn.Parameter(
            torch.zeros(config.d_sae, dtype=config.dtype)
        )
        self.b_dec = torch.nn.Parameter(
            torch.zeros(config.d_model, dtype=config.dtype)
        )
        self.threshold = torch.nn.Parameter(
            torch.zeros(config.d_sae, dtype=config.dtype)
        )
    
    def load_from_gemma_scope(self, params_path: str):
        """
        Load weights from Gemma Scope .npz file.
        
        Args:
            params_path: Path to params.npz file
        """
        params = np.load(params_path)
        
        # Load weights (adjust keys based on actual file structure)
        if 'W_enc' in params:
            self.W_enc.data = torch.from_numpy(params['W_enc']).to(self.config.dtype)
        if 'W_dec' in params:
            self.W_dec.data = torch.from_numpy(params['W_dec']).to(self.config.dtype)
        if 'b_enc' in params:
            self.b_enc.data = torch.from_numpy(params['b_enc']).to(self.config.dtype)
        if 'b_dec' in params:
            self.b_dec.data = torch.from_numpy(params['b_dec']).to(self.config.dtype)
        if 'threshold' in params:
            self.threshold.data = torch.from_numpy(params['threshold']).to(self.config.dtype)
        
        print(f"Loaded SAE from {params_path}")
        print(f"  W_enc: {self.W_enc.shape}")
        print(f"  W_dec: {self.W_dec.shape}")
        print(f"  b_enc: {self.b_enc.shape}")
        print(f"  b_dec: {self.b_dec.shape}")
        print(f"  threshold: {self.threshold.shape}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse features.
        
        Args:
            x: Input activations (batch, seq, d_model)
        
        Returns:
            Sparse features (batch, seq, d_sae)
        """
        # Pre-activation
        pre_acts = torch.matmul(x, self.W_enc.T) + self.b_enc
        
        # JumpReLU: z * H(z - threshold)
        acts = pre_acts * (pre_acts > self.threshold).float()
        
        return acts
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.
        
        Args:
            f: Sparse features (batch, seq, d_sae)
        
        Returns:
            Reconstructed activations (batch, seq, d_model)
        """
        return torch.matmul(f, self.W_dec.T) + self.b_dec
    
    def forward(self, x: torch.Tensor):
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input activations
        
        Returns:
            Tuple of (reconstructed, features)
        """
        f = self.encode(x)
        x_reconstructed = self.decode(f)
        return x_reconstructed, f

# Usage Example
config = SAEConfig(
    d_model=2304,
    d_sae=16384,
    hook_name="blocks.20.hook_resid_post",
    hook_layer=20
)

sae = JumpReLUSAE(config)
sae.load_from_gemma_scope("./saes/layer_20_16k_l0_71.npz")

# Now you can use this SAE in miStudio!
```

---

## Complete Download Script for All Neuronpedia SAEs

```python
"""
Download all Gemma Scope SAEs available on Neuronpedia.
"""

from huggingface_hub import hf_hub_download
import os
from typing import List, Dict

# Define all SAEs available on Neuronpedia
NEURONPEDIA_SAES = [
    # Gemma 2 2B - Residual Stream - 16K width
    {
        "neuronpedia_id": "20-gemmascope-res-16k",
        "repo_id": "google/gemma-scope-2b-pt-res",
        "layer": 20,
        "width": "16k",
        "l0": "71",
        "site": "residual"
    },
    {
        "neuronpedia_id": "12-gemmascope-res-16k",
        "repo_id": "google/gemma-scope-2b-pt-res",
        "layer": 12,
        "width": "16k",
        "l0": "71",
        "site": "residual"
    },
    {
        "neuronpedia_id": "2-gemmascope-res-16k",
        "repo_id": "google/gemma-scope-2b-pt-res",
        "layer": 2,
        "width": "16k",
        "l0": "71",
        "site": "residual"
    },
    # Add more as needed...
]

def download_all_neuronpedia_saes(
    save_dir: str = "./saes/gemma-2-2b",
    saes: List[Dict] = NEURONPEDIA_SAES
):
    """
    Download all SAEs used by Neuronpedia.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    downloaded = []
    
    for sae_info in saes:
        print(f"\n{'='*60}")
        print(f"Downloading: {sae_info['neuronpedia_id']}")
        print(f"{'='*60}")
        
        subfolder = f"layer_{sae_info['layer']}/width_{sae_info['width']}/average_l0_{sae_info['l0']}"
        
        try:
            file_path = hf_hub_download(
                repo_id=sae_info['repo_id'],
                filename=f"{subfolder}/params.npz",
                local_dir=save_dir,
                local_dir_use_symlinks=False
            )
            
            downloaded.append({
                "neuronpedia_id": sae_info['neuronpedia_id'],
                "file_path": file_path,
                "config": sae_info
            })
            
            print(f"✓ Downloaded to: {file_path}")
            
        except Exception as e:
            print(f"✗ Failed to download: {e}")
    
    print(f"\n{'='*60}")
    print(f"Downloaded {len(downloaded)} / {len(saes)} SAEs")
    print(f"{'='*60}")
    
    return downloaded

# Run the download
if __name__ == "__main__":
    downloaded_saes = download_all_neuronpedia_saes()
    
    # Print summary
    print("\nDownloaded SAEs:")
    for sae in downloaded_saes:
        print(f"  {sae['neuronpedia_id']}: {sae['file_path']}")
```

---

## Mapping Neuronpedia IDs to HuggingFace Paths

### Complete Reference Table

| Model | Neuronpedia ID | HuggingFace Repo | Path | Features |
|-------|----------------|------------------|------|----------|
| Gemma-2-2B | `0-gemmascope-res-16k` | `google/gemma-scope-2b-pt-res` | `layer_0/width_16k/average_l0_71` | 16,384 |
| Gemma-2-2B | `2-gemmascope-res-16k` | `google/gemma-scope-2b-pt-res` | `layer_2/width_16k/average_l0_71` | 16,384 |
| Gemma-2-2B | `12-gemmascope-res-16k` | `google/gemma-scope-2b-pt-res` | `layer_12/width_16k/average_l0_71` | 16,384 |
| Gemma-2-2B | `20-gemmascope-res-16k` | `google/gemma-scope-2b-pt-res` | `layer_20/width_16k/average_l0_71` | 16,384 |
| Gemma-2-2B | `25-gemmascope-res-16k` | `google/gemma-scope-2b-pt-res` | `layer_25/width_16k/average_l0_71` | 16,384 |

**Pattern:**
```
Neuronpedia: {layer}-gemmascope-{site}-{width}
HuggingFace: layer_{layer}/width_{width}/average_l0_{l0}
```

---

## Testing Your Download

### Verification Script

```python
import torch
from mistudio.sae import JumpReLUSAE, SAEConfig

def test_sae_loading():
    """Test that the SAE loads correctly and matches Neuronpedia specs."""
    
    # Configure for Neuronpedia's "20-gemmascope-res-16k"
    config = SAEConfig(
        d_model=2304,      # Gemma 2 2B
        d_sae=16384,       # 16K width
        hook_name="blocks.20.hook_resid_post",
        hook_layer=20
    )
    
    # Load SAE
    sae = JumpReLUSAE(config)
    sae.load_from_gemma_scope("./saes/layer_20_16k_l0_71.npz")
    
    # Test with dummy input
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randn(batch_size, seq_len, config.d_model)
    
    # Forward pass
    reconstructed, features = sae(dummy_input)
    
    # Verify dimensions
    assert reconstructed.shape == dummy_input.shape, "Reconstruction shape mismatch!"
    assert features.shape == (batch_size, seq_len, config.d_sae), "Features shape mismatch!"
    
    # Check sparsity
    l0 = (features != 0).float().sum(dim=-1).mean()
    print(f"Average L0 sparsity: {l0:.1f}")
    print(f"Expected L0: ~71")
    
    # Check reconstruction error
    mse = ((reconstructed - dummy_input) ** 2).mean()
    print(f"Reconstruction MSE: {mse:.4f}")
    
    print("\n✓ SAE loaded and verified successfully!")
    
    return sae

# Run test
sae = test_sae_loading()
```

---

## Next Steps: Compare with Neuronpedia

### Steering Comparison Workflow

```python
# 1. Load the same SAE that Neuronpedia uses
sae = load_neuronpedia_sae("20-gemmascope-res-16k")

# 2. Select the same feature
feature_idx = 7650  # Example from Neuronpedia

# 3. Apply steering with same parameters
steering_strength = 50.0
prompt = "Once upon a time"

# 4. Generate in miStudio
mistudio_output = mistudio_generate(
    prompt=prompt,
    sae=sae,
    feature_idx=feature_idx,
    strength=steering_strength
)

# 5. Generate in Neuronpedia (manually via web UI)
# Compare results!

# 6. Verify they're using the same SAE
print(f"miStudio SAE path: {sae.config.hook_name}")
print(f"Neuronpedia SAE: blocks.20.hook_resid_post")
print(f"Match: {sae.config.hook_name == 'blocks.20.hook_resid_post'}")
```

---

## Available SAE Widths and Sparsities

### Gemma 2 2B Options

From the Gemma Scope paper, these widths are available:

| Width | Features | Typical L0 Options |
|-------|----------|-------------------|
| 16K | 16,384 | 30, 50, 71, 100 |
| 65K | 65,536 | 30, 50, 71, 100 |
| 131K | 131,072 | 30, 50, 71, 100 |
| 262K | 262,144 | 50, 71, 100 |
| 524K | 524,288 | 50, 71, 100 |
| 1M | 1,048,576 | 71, 100 |

**Neuronpedia primarily uses:**
- Width: 16K (most common)
- L0: 71 (canonical sparsity)

---

## Troubleshooting

### Issue 1: File Not Found

```python
# If you get a 404 error, the file might be in a different location
# Try listing available files first

from huggingface_hub import list_repo_files

files = list_repo_files(
    repo_id="google/gemma-scope-2b-pt-res",
    repo_type="model"
)

# Filter for layer 20
layer_20_files = [f for f in files if "layer_20" in f]
print("Available files for layer 20:")
for f in layer_20_files:
    print(f"  {f}")
```

### Issue 2: Large Download

```python
# params.npz files are large (100MB - 1GB)
# Use streaming or show progress

from tqdm import tqdm
import requests

def download_with_progress(url: str, save_path: str):
    """Download large file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=save_path
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
```

### Issue 3: Wrong Parameter Names

```python
# If parameter names don't match, inspect the file

import numpy as np

params = np.load("params.npz")
print("Available parameters:")
for key in params.keys():
    print(f"  {key}: {params[key].shape}")

# Map to your expected names
# Might be: W_encoder, decoder_weight, etc.
```

---

## Summary

**To download SAEs that match Neuronpedia:**

1. **Use HuggingFace Hub** - Easiest method
2. **Target Repository:** `google/gemma-scope-2b-pt-res`
3. **Path Pattern:** `layer_{layer}/width_{width}/average_l0_{l0}`
4. **Recommended Starting Point:** Layer 20, 16K width, L0=71
   - This is `20-gemmascope-res-16k` in Neuronpedia
   - Path: `layer_20/width_16k/average_l0_71/params.npz`

5. **Load in miStudio** without SAELens dependency
6. **Compare results** with identical steering parameters

This ensures you're using **exactly the same SAE** as Neuronpedia for valid comparisons!

---

**Quick Start Command:**

```bash
# Download the most popular Neuronpedia SAE
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='google/gemma-scope-2b-pt-res',
    filename='layer_20/width_16k/average_l0_71/params.npz',
    local_dir='./saes'
)
print(f'Downloaded to: {path}')
"
```

---

**Created:** November 26, 2025  
**For:** miStudio Development Team  
**Purpose:** Enable exact comparison with Neuronpedia steering results
