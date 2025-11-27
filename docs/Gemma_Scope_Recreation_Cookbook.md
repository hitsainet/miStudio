# Gemma Scope Recreation Cookbook for miStudio
## A Complete Guide to Reproducing Google DeepMind's Sparse Autoencoder Research

**Version:** 1.0  
**Last Updated:** November 26, 2025  
**Based on:** Gemma Scope Technical Report (arXiv:2408.05147v2)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Prerequisites & Setup](#prerequisites--setup)
4. [Phase 1: Data Pipeline](#phase-1-data-pipeline)
5. [Phase 2: SAE Architecture Implementation](#phase-2-sae-architecture-implementation)
6. [Phase 3: Training Infrastructure](#phase-3-training-infrastructure)
7. [Phase 4: Training Execution](#phase-4-training-execution)
8. [Phase 5: Evaluation & Validation](#phase-5-evaluation--validation)
9. [Phase 6: Feature Steering Integration](#phase-6-feature-steering-integration)
10. [Resource Requirements](#resource-requirements)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [References & Citations](#references--citations)

---

## Executive Summary

This cookbook provides a step-by-step guide to recreate Google DeepMind's **Gemma Scope** project using miStudio. Gemma Scope is a comprehensive suite of 400+ Sparse Autoencoders (SAEs) trained on Gemma 2 models (2B, 9B, and 27B parameters) using the state-of-the-art **JumpReLU** architecture.

### What You'll Build

- **JumpReLU Sparse Autoencoders** at multiple widths (16K to 1M features)
- **Multi-layer coverage** (every layer and sublayer of target models)
- **Evaluation framework** for sparsity-fidelity tradeoffs
- **Feature steering capabilities** for behavior modification
- **Comprehensive metrics** (FVU, delta loss, L0 sparsity)

### Key Achievements

- 30+ million learned features across all SAEs
- Training on 4-16B tokens per SAE
- Used ~15% of Gemma 2 9B training compute
- Stored ~20 Pebibytes of activation data

---

## Project Overview

### Background: What is Gemma Scope?

Gemma Scope uses **Sparse Autoencoders (SAEs)** to decompose neural network activations into interpretable features. Think of it as a "microscope" for AI models—breaking down complex internal representations into understandable components.

### Why JumpReLU Architecture?

Traditional SAEs struggle to balance:
1. **Detecting which features are present** (binary on/off)
2. **Estimating their strength** (magnitude)

JumpReLU solves this with a **learnable threshold per feature**:

```
JumpReLU_θ(z) = z ⊙ H(z - θ)
```

Where:
- `θ` = learned threshold vector (positive values)
- `H` = Heaviside step function (1 if input > 0, else 0)
- `⊙` = element-wise multiplication

### Core SAE Equations

**Encoder:**
```
f(x) = JumpReLU_θ(W_enc × x + b_enc)
```

**Decoder:**
```
x̂(f) = W_dec × f + b_dec
```

**Loss Function:**
```
L = ||x - x̂||² + λ||f(x)||₀
```

Where:
- `λ` = sparsity penalty coefficient
- `||·||₀` = L0 norm (count of non-zero elements)

---

## Prerequisites & Setup

### 1. Hardware Requirements

#### Minimum Configuration (for Gemma 2 2B)
- **GPU:** 1x A100 (40GB) or 2x V100 (32GB each)
- **Storage:** 2TB SSD for activations
- **RAM:** 128GB system memory
- **Network:** High-bandwidth for data transfer

#### Recommended Configuration (for Gemma 2 9B)
- **GPU:** 4-8x A100 (80GB) or TPUv3 pods
- **Storage:** 20TB+ NVMe SSD array
- **RAM:** 512GB+ system memory
- **Network:** 10Gbps+ backbone

#### Full-Scale Configuration (for complete suite)
- **Compute:** TPUv5p in 2x2x4 configuration
- **Storage:** 20+ Pebibytes disk array
- **Data throughput:** 1+ GiB/s sustained read speed

### 2. Software Stack

```bash
# Core ML frameworks
pip install torch>=2.0.0
pip install jax[cuda11_cudnn86]>=0.4.0
pip install flax>=0.7.0

# Transformer utilities
pip install transformer-lens>=1.0.0
pip install transformers>=4.35.0

# SAE-specific libraries
pip install sae-lens>=1.0.0

# Data handling
pip install datasets>=2.14.0
pip install huggingface-hub>=0.17.0

# Monitoring & visualization
pip install wandb
pip install tensorboard
pip install plotly
```

### 3. Model Access

```python
# Setup Hugging Face authentication
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")

# Load base Gemma 2 model
from transformers import AutoModel, AutoTokenizer

model_name = "google/gemma-2-2b"  # or gemma-2-9b
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32
)
```

### 4. Data Access

**Training Data Distribution:**
- Same distribution as Gemma 1 pretraining data
- Predominantly English text
- Filtered for quality and safety
- Excludes CSAM, sensitive PII

**Alternative Public Datasets:**
```python
from datasets import load_dataset

# Option 1: C4 (Common Crawl)
dataset = load_dataset("c4", "en", streaming=True)

# Option 2: The Pile
dataset = load_dataset("EleutherAI/pile", streaming=True)

# Option 3: RedPajama
dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", streaming=True)
```

---

## Phase 1: Data Pipeline

### Step 1.1: Activation Collection

**Objective:** Extract and store model activations from all layers.

#### Code Template: Activation Extraction

```python
import torch
import transformer_lens
from pathlib import Path
import numpy as np

class ActivationExtractor:
    def __init__(self, model_name, hook_points, save_dir):
        """
        Args:
            model_name: HuggingFace model identifier
            hook_points: List of layer names to hook
            save_dir: Directory to save activations
        """
        self.model = transformer_lens.HookedTransformer.from_pretrained(
            model_name,
            device="cuda"
        )
        self.hook_points = hook_points
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_batch(self, tokens, batch_idx):
        """Extract activations for a batch of tokens."""
        activations = {}
        
        def hook_fn(act, hook_name):
            """Store activations during forward pass."""
            # Store on CPU to avoid OOM
            activations[hook_name] = act.detach().cpu().numpy()
        
        # Register hooks
        for hook_point in self.hook_points:
            self.model.add_hook(hook_point, hook_fn)
        
        # Forward pass
        with torch.no_grad():
            self.model(tokens)
        
        # Save to disk
        save_path = self.save_dir / f"batch_{batch_idx}.npz"
        np.savez_compressed(save_path, **activations)
        
        # Clear hooks
        self.model.reset_hooks()
        
        return save_path

# Example usage
hook_points = [
    "blocks.12.hook_resid_post",  # Residual stream after layer 12
    "blocks.12.attn.hook_result",  # Attention output
    "blocks.12.mlp.hook_post",     # MLP output
]

extractor = ActivationExtractor(
    model_name="google/gemma-2-2b",
    hook_points=hook_points,
    save_dir="./activations/gemma-2b-layer12"
)
```

#### Key Parameters from Gemma Scope

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Sequence Length** | 1,024 tokens | Standard context window |
| **Batch Size** | 4,096 | For activation collection |
| **Total Tokens** | 4B - 16B | Depending on SAE width |
| **Excluded Tokens** | BOS, EOS, PAD | Filter these out |
| **Storage Format** | Raw bytes (.npz) | 32-bit float precision |
| **Shard Size** | 10-20 GiB | Per disk shard |

### Step 1.2: Activation Normalization

**Critical:** Normalize activations to have unit mean squared norm.

```python
def normalize_activations(activations):
    """
    Normalize activations to have unit mean squared norm.
    
    This allows hyperparameter transfer across layers and sites.
    """
    # Compute RMS norm
    rms_norm = np.sqrt(np.mean(activations ** 2))
    
    # Normalize
    normalized = activations / rms_norm
    
    return normalized, rms_norm

# Store normalization constant for later rescaling
normalization_stats = {
    "rms_norm": rms_norm,
    "layer": layer_idx,
    "site": site_name
}
```

### Step 1.3: Data Shuffling

**Shuffle activations in buckets of ~10⁶ samples.**

```python
import random

def shuffle_in_buckets(activation_files, bucket_size=1_000_000):
    """
    Shuffle activations in manageable buckets.
    
    Args:
        activation_files: List of paths to activation shards
        bucket_size: Number of activations per bucket
    """
    for i in range(0, len(activation_files), bucket_size):
        bucket = activation_files[i:i + bucket_size]
        random.shuffle(bucket)
        
        # Save shuffled bucket
        yield bucket
```

### Step 1.4: Distributed Data Loading

**Implement shared server system for parallel training.**

```python
import zmq
import threading

class ActivationServer:
    """
    Shared data buffer server for multiple training jobs.
    """
    def __init__(self, data_dir, buffer_size=100, port=5555):
        self.data_dir = Path(data_dir)
        self.buffer_size = buffer_size
        self.port = port
        self.buffer = []
        self.lock = threading.Lock()
        
    def start(self):
        """Start server and listen for batch requests."""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")
        
        # Pre-load buffer
        self._refill_buffer()
        
        while True:
            # Wait for batch request
            message = socket.recv_json()
            
            # Send batch
            with self.lock:
                batch = self.buffer.pop(0)
                socket.send_json({"batch": batch})
                
            # Refill if needed
            if len(self.buffer) < self.buffer_size // 2:
                threading.Thread(target=self._refill_buffer).start()
    
    def _refill_buffer(self):
        """Load batches from disk into buffer."""
        activation_files = sorted(self.data_dir.glob("*.npz"))
        
        with self.lock:
            for file in activation_files:
                if len(self.buffer) >= self.buffer_size:
                    break
                    
                data = np.load(file)
                self.buffer.append(data)
```

---

## Phase 2: SAE Architecture Implementation

### Step 2.1: JumpReLU Activation Function

```python
import torch
import torch.nn as nn

class JumpReLU(nn.Module):
    """
    JumpReLU activation with learnable thresholds.
    
    JumpReLU_θ(z) = z ⊙ H(z - θ)
    """
    def __init__(self, num_features):
        super().__init__()
        # Initialize thresholds to small positive value
        self.threshold = nn.Parameter(
            torch.full((num_features,), 0.001, dtype=torch.float32)
        )
        
    def forward(self, z):
        """
        Args:
            z: Pre-activations, shape (batch, seq, num_features)
        Returns:
            Activated features with threshold gating
        """
        # Heaviside step function (no gradient)
        gate = (z > self.threshold).float()
        
        # Apply gate to z
        return z * gate
    
    def backward(self, grad_output):
        """
        Use straight-through estimator (STE) for threshold gradients.
        """
        # Standard gradient for z
        grad_z = grad_output * (self.pre_activations > self.threshold).float()
        
        # STE gradient for threshold using KDE
        # See Rajamanoharan et al. 2024 for details
        return grad_z
```

### Step 2.2: Complete SAE Implementation

```python
class JumpReLUSAE(nn.Module):
    """
    Sparse Autoencoder with JumpReLU activation.
    """
    def __init__(self, d_model, d_sae, tied_weights=False):
        """
        Args:
            d_model: Input dimension (model hidden size)
            d_sae: SAE hidden dimension (num features)
            tied_weights: Whether to tie encoder/decoder weights
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_sae = d_sae
        
        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        # Decoder
        if tied_weights:
            self.W_dec = self.W_enc.T
        else:
            self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # JumpReLU activation
        self.activation = JumpReLU(d_sae)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize following Gemma Scope methodology."""
        # He-uniform initialization for decoder
        nn.init.kaiming_uniform_(self.W_dec, mode='fan_in', nonlinearity='relu')
        
        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = nn.functional.normalize(
                self.W_dec.data, dim=0, p=2
            )
        
        # Initialize encoder as transpose of decoder
        if not hasattr(self, 'tied_weights') or not self.tied_weights:
            self.W_enc.data = self.W_dec.data.T.clone()
    
    def encode(self, x):
        """
        Encode input to sparse feature representation.
        
        Args:
            x: Input activations, shape (batch, seq, d_model)
        Returns:
            f: Sparse features, shape (batch, seq, d_sae)
        """
        # Pre-activations
        z = torch.matmul(x, self.W_enc.T) + self.b_enc
        
        # Apply JumpReLU
        f = self.activation(z)
        
        return f
    
    def decode(self, f):
        """
        Decode sparse features back to input space.
        
        Args:
            f: Sparse features, shape (batch, seq, d_sae)
        Returns:
            x_hat: Reconstructed activations, shape (batch, seq, d_model)
        """
        x_hat = torch.matmul(f, self.W_dec.T) + self.b_dec
        return x_hat
    
    def forward(self, x):
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input activations
        Returns:
            x_hat: Reconstructed activations
            f: Sparse feature activations
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f
    
    def normalize_decoder(self):
        """
        Project decoder columns to unit norm.
        Call after each optimizer step.
        """
        with torch.no_grad():
            self.W_dec.data = nn.functional.normalize(
                self.W_dec.data, dim=0, p=2
            )
    
    def get_l0(self, f):
        """
        Compute L0 norm (sparsity) of features.
        
        Args:
            f: Feature activations
        Returns:
            Mean number of active features per token
        """
        return (f != 0).float().sum(dim=-1).mean()
```

### Step 2.3: Loss Function Implementation

```python
class SAELoss(nn.Module):
    """
    Loss function for JumpReLU SAE training.
    
    L = ||x - x̂||² + λ||f||₀
    """
    def __init__(self, sparsity_coeff, warmup_steps=10_000):
        super().__init__()
        self.sparsity_coeff = sparsity_coeff
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def forward(self, x, x_hat, f):
        """
        Compute total loss.
        
        Args:
            x: Original activations
            x_hat: Reconstructed activations
            f: Sparse features
        Returns:
            loss: Total loss
            metrics: Dict of loss components
        """
        # Reconstruction loss
        recon_loss = torch.mean((x - x_hat) ** 2)
        
        # L0 sparsity penalty
        l0 = (f != 0).float().sum(dim=-1).mean()
        sparsity_loss = self.sparsity_coeff * l0
        
        # Apply warmup to sparsity coefficient
        if self.current_step < self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            sparsity_loss = warmup_factor * sparsity_loss
        
        # Total loss
        total_loss = recon_loss + sparsity_loss
        
        # Increment step counter
        self.current_step += 1
        
        metrics = {
            "loss/total": total_loss.item(),
            "loss/reconstruction": recon_loss.item(),
            "loss/sparsity": sparsity_loss.item(),
            "metrics/l0": l0.item(),
        }
        
        return total_loss, metrics
```

---

## Phase 3: Training Infrastructure

### Step 3.1: Hyperparameter Configuration

**Gemma Scope Training Hyperparameters:**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SAEConfig:
    """Configuration for SAE training."""
    
    # Model architecture
    d_model: int = 2048          # Gemma 2 2B hidden size
    d_sae: int = 65_536          # 2^16 features (65K)
    tied_weights: bool = False
    
    # Training data
    num_tokens: int = 8_000_000_000  # 8B tokens
    batch_size: int = 4_096
    sequence_length: int = 1_024
    
    # Optimization
    learning_rate: float = 7e-5
    lr_warmup_steps: int = 1_000
    adam_beta1: float = 0.0
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    
    # Sparsity control
    sparsity_coeff: float = 6e-4  # λ, tune this for desired L0
    sparsity_warmup_steps: int = 10_000
    
    # JumpReLU specific
    kde_bandwidth: float = 0.001  # ε for threshold gradient estimation
    initial_threshold: float = 0.001
    
    # Decoder normalization
    normalize_decoder: bool = True
    project_gradients: bool = True
    
    # Logging & checkpointing
    log_every: int = 100
    eval_every: int = 1_000
    save_every: int = 10_000
    
    # Hardware
    device: str = "cuda"
    num_gpus: int = 4
    mixed_precision: bool = True  # Use fp16/bf16

# Create configuration for different SAE widths
configs = {
    "16k": SAEConfig(d_sae=2**14, num_tokens=4_000_000_000),
    "32k": SAEConfig(d_sae=2**15, num_tokens=8_000_000_000),
    "65k": SAEConfig(d_sae=2**16, num_tokens=8_000_000_000),
    "131k": SAEConfig(d_sae=2**17, num_tokens=8_000_000_000),
    "262k": SAEConfig(d_sae=2**18, num_tokens=8_000_000_000),
    "524k": SAEConfig(d_sae=2**19, num_tokens=8_000_000_000),
    "1m": SAEConfig(d_sae=2**20, num_tokens=16_000_000_000),
}
```

### Step 3.2: Optimizer Setup

```python
def create_optimizer(model, config):
    """
    Create Adam optimizer with gradient projection.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay
    )
    
    return optimizer

def create_lr_scheduler(optimizer, config):
    """
    Create learning rate scheduler with cosine warmup.
    """
    def lr_lambda(step):
        if step < config.lr_warmup_steps:
            # Linear warmup from 0.1 * lr to lr
            return 0.1 + 0.9 * (step / config.lr_warmup_steps)
        else:
            return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )
    
    return scheduler
```

### Step 3.3: Gradient Projection

**Critical:** Project out gradients parallel to decoder columns.

```python
def project_gradients(model):
    """
    Project decoder gradients to be orthogonal to decoder columns.
    
    This maintains unit norm constraint on decoder vectors.
    """
    if model.W_dec.grad is not None:
        with torch.no_grad():
            # Get decoder weights (d_model, d_sae)
            W = model.W_dec.data
            G = model.W_dec.grad
            
            # Project out component parallel to W
            # G_perp = G - W * (W^T G)
            parallel_component = (W * G).sum(dim=0, keepdim=True)
            G_perp = G - W * parallel_component
            
            # Replace gradient
            model.W_dec.grad = G_perp
```

---

## Phase 4: Training Execution

### Step 4.1: Training Loop

```python
import wandb
from tqdm import tqdm

def train_sae(
    sae_model,
    train_loader,
    config,
    checkpoint_dir="./checkpoints"
):
    """
    Main training loop for SAE.
    """
    # Setup
    device = torch.device(config.device)
    sae_model = sae_model.to(device)
    
    optimizer = create_optimizer(sae_model, config)
    scheduler = create_lr_scheduler(optimizer, config)
    loss_fn = SAELoss(config.sparsity_coeff, config.sparsity_warmup_steps)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
    
    # Initialize wandb
    wandb.init(
        project="gemma-scope-recreation",
        config=config.__dict__
    )
    
    # Training loop
    global_step = 0
    total_steps = config.num_tokens // (config.batch_size * config.sequence_length)
    
    pbar = tqdm(total=total_steps, desc="Training SAE")
    
    for batch in train_loader:
        # Move to device
        x = batch["activations"].to(device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            x_hat, f = sae_model(x)
            loss, metrics = loss_fn(x, x_hat, f)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Project gradients before step
        if config.project_gradients:
            project_gradients(sae_model)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Normalize decoder columns
        if config.normalize_decoder:
            sae_model.normalize_decoder()
        
        # Logging
        if global_step % config.log_every == 0:
            wandb.log(metrics, step=global_step)
        
        # Evaluation
        if global_step % config.eval_every == 0:
            eval_metrics = evaluate_sae(sae_model, val_loader, device)
            wandb.log(eval_metrics, step=global_step)
        
        # Checkpointing
        if global_step % config.save_every == 0:
            save_checkpoint(sae_model, optimizer, global_step, checkpoint_dir)
        
        global_step += 1
        pbar.update(1)
        
        if global_step >= total_steps:
            break
    
    pbar.close()
    wandb.finish()
    
    return sae_model
```

### Step 4.2: Data Loading

```python
from torch.utils.data import Dataset, DataLoader

class ActivationDataset(Dataset):
    """
    Dataset for loading pre-computed activations.
    """
    def __init__(self, data_dir, layer, site, normalization_constant):
        self.data_dir = Path(data_dir)
        self.layer = layer
        self.site = site
        self.norm_constant = normalization_constant
        
        # Get all activation files
        self.files = sorted(self.data_dir.glob(f"layer{layer}_{site}_*.npz"))
        
        # Compute total samples
        self.total_samples = len(self.files) * 4096  # batch size
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Load shard
        file_idx = idx // 4096
        sample_idx = idx % 4096
        
        data = np.load(self.files[file_idx])
        activation = data[self.site][sample_idx]
        
        # Normalize
        activation = activation / self.norm_constant
        
        return {
            "activations": torch.from_numpy(activation).float()
        }

def create_dataloader(config, data_dir, layer, site):
    """Create DataLoader with distributed support."""
    
    # Load normalization constant
    norm_stats = np.load(data_dir / f"norm_stats_layer{layer}_{site}.npz")
    norm_constant = norm_stats["rms_norm"]
    
    dataset = ActivationDataset(data_dir, layer, site, norm_constant)
    
    # Use DistributedSampler for multi-GPU training
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        shuffle=True
    ) if config.num_gpus > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
```

### Step 4.3: Checkpointing

```python
def save_checkpoint(model, optimizer, step, checkpoint_dir):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["step"]
```

---

## Phase 5: Evaluation & Validation

### Step 5.1: Key Metrics

**Gemma Scope uses three primary metrics:**

1. **L0 Sparsity:** Average number of active features
2. **FVU (Fraction of Variance Unexplained):** Reconstruction quality
3. **Delta Loss:** Impact on language model's next-token prediction

```python
def compute_fvu(x, x_hat):
    """
    Compute Fraction of Variance Unexplained.
    
    FVU = MSE(x, x_hat) / Var(x)
    
    Lower is better (0 = perfect reconstruction).
    """
    mse = torch.mean((x - x_hat) ** 2)
    variance = torch.var(x)
    fvu = mse / variance
    return fvu.item()

def compute_l0_sparsity(f):
    """
    Compute L0 sparsity: mean number of active features.
    """
    l0 = (f != 0).float().sum(dim=-1).mean()
    return l0.item()

def compute_delta_loss(model, sae, tokens, layer_name):
    """
    Compute increase in LM loss when SAE is spliced in.
    
    Delta Loss = Loss(with SAE) - Loss(without SAE)
    
    Lower is better (less impact on model).
    """
    # Original forward pass
    with torch.no_grad():
        logits_original = model(tokens).logits
        loss_original = compute_ce_loss(logits_original, tokens)
    
    # Forward pass with SAE reconstruction
    def sae_hook(activations, hook):
        x_hat, _ = sae(activations)
        return x_hat
    
    with torch.no_grad():
        model.add_hook(layer_name, sae_hook)
        logits_sae = model(tokens).logits
        loss_sae = compute_ce_loss(logits_sae, tokens)
        model.reset_hooks()
    
    delta_loss = loss_sae - loss_original
    return delta_loss.item()
```

### Step 5.2: Evaluation Loop

```python
def evaluate_sae(sae_model, eval_loader, device, lm_model=None):
    """
    Comprehensive SAE evaluation.
    
    Returns dict of metrics:
    - l0: Average sparsity
    - fvu: Reconstruction quality
    - delta_loss: Impact on LM (if lm_model provided)
    """
    sae_model.eval()
    
    metrics = {
        "l0": [],
        "fvu": [],
        "delta_loss": []
    }
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            x = batch["activations"].to(device)
            
            # Forward pass
            x_hat, f = sae_model(x)
            
            # Compute metrics
            l0 = compute_l0_sparsity(f)
            fvu = compute_fvu(x, x_hat)
            
            metrics["l0"].append(l0)
            metrics["fvu"].append(fvu)
            
            # Optionally compute delta loss
            if lm_model is not None and "tokens" in batch:
                delta = compute_delta_loss(
                    lm_model,
                    sae_model,
                    batch["tokens"].to(device),
                    batch["layer_name"]
                )
                metrics["delta_loss"].append(delta)
    
    # Aggregate
    results = {
        "eval/l0_mean": np.mean(metrics["l0"]),
        "eval/l0_std": np.std(metrics["l0"]),
        "eval/fvu_mean": np.mean(metrics["fvu"]),
        "eval/fvu_std": np.std(metrics["fvu"]),
    }
    
    if metrics["delta_loss"]:
        results.update({
            "eval/delta_loss_mean": np.mean(metrics["delta_loss"]),
            "eval/delta_loss_std": np.std(metrics["delta_loss"]),
        })
    
    sae_model.train()
    return results
```

### Step 5.3: Sparsity-Fidelity Curve

**Generate Pareto frontier by sweeping sparsity coefficient.**

```python
def plot_sparsity_fidelity_curve(
    model_name,
    layer,
    site,
    sparsity_coeffs=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    sae_width=65_536
):
    """
    Train SAEs with different sparsity coefficients and plot trade-off.
    """
    import matplotlib.pyplot as plt
    
    results = []
    
    for lambda_val in sparsity_coeffs:
        print(f"\nTraining with λ = {lambda_val}")
        
        # Update config
        config = SAEConfig(
            d_sae=sae_width,
            sparsity_coeff=lambda_val
        )
        
        # Train SAE
        sae = JumpReLUSAE(config.d_model, config.d_sae)
        train_sae(sae, train_loader, config)
        
        # Evaluate
        metrics = evaluate_sae(sae, val_loader, device)
        
        results.append({
            "lambda": lambda_val,
            "l0": metrics["eval/l0_mean"],
            "fvu": metrics["eval/fvu_mean"],
            "delta_loss": metrics.get("eval/delta_loss_mean", None)
        })
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # FVU vs L0
    ax1.scatter([r["l0"] for r in results], 
                [r["fvu"] for r in results])
    ax1.set_xlabel("L0 (Mean Active Features)")
    ax1.set_ylabel("FVU (Fraction Variance Unexplained)")
    ax1.set_title(f"{model_name} - Layer {layer} - {site}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    
    # Delta Loss vs L0
    if results[0]["delta_loss"] is not None:
        ax2.scatter([r["l0"] for r in results],
                    [r["delta_loss"] for r in results])
        ax2.set_xlabel("L0 (Mean Active Features)")
        ax2.set_ylabel("Delta Loss")
        ax2.set_title("Impact on Language Model")
        ax2.set_xscale("log")
    
    plt.tight_layout()
    plt.savefig(f"sparsity_fidelity_{model_name}_layer{layer}_{site}.png")
    
    return results
```

---

## Phase 6: Feature Steering Integration

### Step 6.1: Load Trained SAEs

```python
def load_gemma_scope_sae(model_name, layer, site, width):
    """
    Load a trained Gemma Scope SAE.
    
    Args:
        model_name: "gemma-2-2b" or "gemma-2-9b"
        layer: Layer index (0-25 for 2B, 0-41 for 9B)
        site: "residual", "mlp", or "attention"
        width: "16k", "65k", "131k", "1m", etc.
    
    Returns:
        Loaded SAE model
    """
    # Map to config
    d_model = 2048 if "2b" in model_name else 3584
    d_sae = {
        "16k": 2**14,
        "65k": 2**16,
        "131k": 2**17,
        "1m": 2**20
    }[width]
    
    # Initialize model
    sae = JumpReLUSAE(d_model, d_sae)
    
    # Load checkpoint
    checkpoint_path = f"./checkpoints/{model_name}/layer_{layer}/{site}/width_{width}/best.pt"
    checkpoint = torch.load(checkpoint_path)
    sae.load_state_dict(checkpoint["model_state_dict"])
    
    return sae
```

### Step 6.2: Feature Steering Implementation

**Implement the feature steering workflow from the PRD.**

```python
class FeatureSteering:
    """
    Feature steering using SAE decoder vectors.
    """
    def __init__(self, model, sae, layer_name):
        self.model = model
        self.sae = sae
        self.layer_name = layer_name
        
    def steer(self, tokens, feature_idx, strength=50.0):
        """
        Generate text with feature steering.
        
        Args:
            tokens: Input token IDs
            feature_idx: Index of feature to steer on
            strength: Steering coefficient α
        
        Returns:
            Generated tokens
        """
        # Get steering vector from SAE decoder
        steering_vector = self.sae.W_dec[:, feature_idx]  # (d_model,)
        
        # Scale by strength
        scaled_vector = strength * steering_vector
        
        # Define steering hook
        def steering_hook(activations, hook):
            # Add steering vector to all positions
            return activations + scaled_vector.unsqueeze(0).unsqueeze(0)
        
        # Generate with hook
        with self.model.hooks([(self.layer_name, steering_hook)]):
            outputs = self.model.generate(
                tokens,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
        
        return outputs
    
    def compare_steering(self, prompt, feature_configs):
        """
        Compare multiple steering configurations side-by-side.
        
        Args:
            prompt: Text prompt
            feature_configs: List of (feature_idx, strength) tuples
        
        Returns:
            Dict mapping config name to generated text
        """
        tokenizer = self.model.tokenizer
        tokens = tokenizer.encode(prompt, return_tensors="pt")
        
        results = {}
        
        # Unsteered baseline
        with torch.no_grad():
            baseline_ids = self.model.generate(tokens, max_new_tokens=100)
            results["unsteered"] = tokenizer.decode(baseline_ids[0])
        
        # Steered versions
        for i, (feat_idx, strength) in enumerate(feature_configs):
            output_ids = self.steer(tokens, feat_idx, strength)
            results[f"feature_{feat_idx}_alpha_{strength}"] = tokenizer.decode(output_ids[0])
        
        return results
```

### Step 6.3: Feature Browser

```python
class FeatureBrowser:
    """
    Browse and search SAE features.
    """
    def __init__(self, sae, feature_descriptions=None):
        self.sae = sae
        self.descriptions = feature_descriptions or {}
        
    def get_feature_info(self, feature_idx):
        """Get information about a specific feature."""
        return {
            "index": feature_idx,
            "description": self.descriptions.get(feature_idx, "Unknown"),
            "decoder_norm": torch.norm(self.sae.W_dec[:, feature_idx]).item(),
            "threshold": self.sae.activation.threshold[feature_idx].item()
        }
    
    def search_features(self, query):
        """Search features by description."""
        results = []
        for idx, desc in self.descriptions.items():
            if query.lower() in desc.lower():
                results.append(self.get_feature_info(idx))
        return results
    
    def top_activating_features(self, tokens, top_k=10):
        """Find features that activate most strongly on given tokens."""
        # Get activations
        with torch.no_grad():
            x = self.sae.encode(tokens)
            
        # Get top-k features by max activation
        max_acts, _ = x.max(dim=1)  # Max over sequence
        top_k_vals, top_k_idxs = torch.topk(max_acts[0], top_k)
        
        results = []
        for idx, val in zip(top_k_idxs, top_k_vals):
            info = self.get_feature_info(idx.item())
            info["max_activation"] = val.item()
            results.append(info)
        
        return results
```

---

## Resource Requirements

### Compute Budget

**For Full Gemma Scope Recreation:**

| Component | Value | Notes |
|-----------|-------|-------|
| **Total Compute** | ~20% of GPT-3 training | ~60 PFlop/s-days |
| **Per SAE (65K)** | ~100 GPU-days | A100 80GB equivalent |
| **Storage** | 20 PiB | For all activations |
| **Training Time** | 2-4 weeks | With 100+ GPUs |

**Scaled-Down Version (Single Layer):**

| Component | Value | Notes |
|-----------|-------|-------|
| **Compute** | ~8 GPU-days | Per SAE width |
| **Storage** | ~100 TB | For one layer |
| **Training Time** | 1-2 days | With 4-8 GPUs |

### Cost Estimation

**Cloud GPU Pricing (Approximate):**

```
A100 80GB: $3.00/hr
V100 32GB: $2.50/hr
T4 16GB:   $0.35/hr

Full Recreation: ~$150,000
Single Layer:    ~$600
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms:** CUDA OOM error during training

**Solutions:**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (fp16/bf16)
- Reduce SAE width
- Use model/data parallelism

```python
# Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = train_step(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. Dead Features

**Symptoms:** Many SAE features never activate

**Solutions:**
- Use feature resampling (from Anthropic)
- Lower initial threshold
- Adjust sparsity coefficient
- Increase learning rate temporarily

```python
def resample_dead_features(sae, activations, threshold=1e-6):
    """Resample features that rarely activate."""
    with torch.no_grad():
        # Compute activation frequency
        f = sae.encode(activations)
        freq = (f > 0).float().mean(dim=(0, 1))
        
        # Find dead features
        dead_mask = freq < threshold
        num_dead = dead_mask.sum().item()
        
        if num_dead > 0:
            print(f"Resampling {num_dead} dead features")
            
            # Reinitialize dead features
            sae.W_dec[:, dead_mask] = torch.randn_like(
                sae.W_dec[:, dead_mask]
            )
            sae.W_enc[dead_mask, :] = sae.W_dec[:, dead_mask].T
```

#### 3. High FVU / Poor Reconstruction

**Symptoms:** FVU > 0.5, poor reconstruction quality

**Solutions:**
- Increase SAE width (more features)
- Lower sparsity coefficient
- Train for more steps
- Check activation normalization

#### 4. Training Instability

**Symptoms:** Loss spikes, NaN gradients

**Solutions:**
- Lower learning rate
- Increase warmup steps
- Enable gradient clipping
- Check for numerical instability in JumpReLU STE

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(
    sae.parameters(),
    max_norm=1.0
)
```

---

## References & Citations

### Primary Sources

1. **Gemma Scope Paper**
   - Lieberum, T., et al. (2024). "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2"
   - arXiv:2408.05147v2
   - https://arxiv.org/abs/2408.05147

2. **JumpReLU Architecture**
   - Rajamanoharan, S., et al. (2024). "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders"
   - arXiv:2407.14435

3. **Gemma Scope Resources**
   - HuggingFace: https://huggingface.co/google/gemma-scope
   - Tutorial: https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
   - Blog Post: https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/

### Additional Reading

4. **Dictionary Learning Foundations**
   - Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning"
   - Templeton, A., et al. (2024). "Scaling Monosemanticity"

5. **SAE Applications**
   - Marks, S., et al. (2024). "Sparse Feature Circuits"
   - Conmy, A., & Nanda, N. (2024). "Activation Steering with SAEs"

### Tools & Libraries

6. **SAELens:** https://github.com/jbloomAus/SAELens
7. **TransformerLens:** https://github.com/neelnanda-io/TransformerLens
8. **Neuronpedia:** https://www.neuronpedia.org/gemma-scope

---

## Appendix A: Quick Start Checklist

- [ ] Install required packages
- [ ] Download Gemma 2 model
- [ ] Collect activations for target layer
- [ ] Normalize activations
- [ ] Implement JumpReLU SAE
- [ ] Configure training hyperparameters
- [ ] Train SAE
- [ ] Evaluate FVU and L0 metrics
- [ ] Generate sparsity-fidelity curve
- [ ] Implement feature steering
- [ ] Integrate with miStudio PRD

## Appendix B: miStudio Integration Points

Based on the SAE_Feature_Steering_PRD.md:

1. **Feature Selection (FR-1.x)**
   - Load SAEs using `load_gemma_scope_sae()`
   - Implement feature browser with `FeatureBrowser`

2. **Steering Configuration (FR-2.x)**
   - Use `FeatureSteering.steer()` for basic steering
   - Support multi-feature via `compare_steering()`

3. **Evaluation (FR-4.x)**
   - Compute metrics with provided functions
   - Generate comparison tables

4. **Visualization (FR-5.x)**
   - Plot sparsity-fidelity curves
   - Track feature activations during generation

---

**End of Cookbook**

*For questions or issues, please refer to the Gemma Scope GitHub repository or the miStudio documentation.*
