# MechInterp Studio (miStudio): Complete System Specification

**Version:** 1.0
**Last Updated:** 2025-10-05
**Status:** Active Development

---

## Executive Summary

MechInterp Studio (miStudio) is an **edge-deployed mechanistic interpretability platform** that brings advanced neural network analysis capabilities to resource-constrained hardware. It enables researchers, developers, and AI enthusiasts to discover, understand, and manipulate the learned features in language models without requiring cloud infrastructure or expensive GPU clusters.

---

## Vision & Mission

### Vision
**Democratize mechanistic interpretability** by making it accessible on edge devices, enabling anyone with a Jetson board, consumer GPU, or AI accelerator to understand how their language models work.

### Mission
Provide a **complete, production-ready toolkit** for:
- Training sparse autoencoders on model activations
- Discovering interpretable features in neural networks
- Visualizing and analyzing learned representations
- Steering model behavior through feature manipulation
- Experimenting with interpretability techniques locally and privately

### Why Edge AI Interpretability Matters

**Privacy-First Research**: Keep sensitive model analysis and experimentation entirely local—no data leaves your device.

**Accessibility**: Eliminate cloud costs and GPU rental fees. Run interpretability research on hardware you already own ($500-$2000 devices vs. $10,000+ cloud bills).

**Real-Time Interaction**: Low-latency steering and experimentation without network round trips.

**Educational Tool**: Teach mechanistic interpretability concepts with hands-on experimentation using modest hardware.

**Production Deployment**: Analyze models where they're deployed—on edge devices, not in the cloud.

---

## Problem Statement

### Current Challenges in Mechanistic Interpretability

1. **Infrastructure Barrier**: Existing tools (Neuronpedia, TransformerLens) assume access to powerful cloud GPUs and high-bandwidth connections.

2. **Cost Prohibitive**: Training sparse autoencoders on even small models can cost hundreds of dollars in GPU time on cloud platforms.

3. **Privacy Concerns**: Researchers working with proprietary or sensitive models cannot use cloud-based interpretability services.

4. **Deployment Gap**: Models deployed on edge devices (robotics, IoT, edge AI) cannot be analyzed in their deployment environment.

5. **Learning Curve**: Existing tools require deep Python/PyTorch expertise and manual scripting for basic analysis tasks.

### What miStudio Solves

✅ **Single-User Desktop Application**: Complete UI for all interpretability workflows—no coding required for basic tasks.

✅ **Edge-Optimized**: Runs on Jetson Orin Nano (8GB), consumer GPUs (RTX 3060+), or AI accelerators (Hailo-8, Coral TPU).

✅ **End-to-End Workflow**: From dataset ingestion to model steering in one integrated platform.

✅ **Cost-Effective**: One-time hardware purchase ($500-$2000) vs. recurring cloud costs.

✅ **Privacy-Preserving**: All computation happens locally—models, datasets, and discoveries never leave your device.

---

## Target Audience

### Primary Users

**1. ML Researchers & Students**
- **Need**: Hands-on learning environment for mechanistic interpretability concepts
- **Use Case**: Train SAEs on small models (GPT-2, TinyLlama) to understand feature discovery
- **Constraint**: Limited budget, no access to institutional GPU clusters

**2. Independent AI Researchers**
- **Need**: Analyze custom-trained models without cloud dependency
- **Use Case**: Discover features in fine-tuned models, experiment with steering techniques
- **Constraint**: Privacy concerns, cost-sensitive, need reproducible experiments

**3. Edge AI Developers**
- **Need**: Understand model behavior in deployment environment
- **Use Case**: Debug unexpected model outputs, analyze edge-optimized quantized models
- **Constraint**: Models deployed on Jetson/Hailo devices, cannot offload to cloud

**4. AI Safety Researchers**
- **Need**: Discover potentially harmful features in models
- **Use Case**: Identify and ablate dangerous capabilities, test steering interventions
- **Constraint**: Need full control over model, cannot use third-party services

### Secondary Users

**5. Educators & Workshop Leaders**
- **Need**: Teaching tool for interpretability courses
- **Use Case**: Live demonstrations of feature discovery and steering
- **Benefit**: Visual, interactive UI suitable for classroom settings

**6. AI Ethics Auditors**
- **Need**: Analyze deployed models for bias and unintended behaviors
- **Use Case**: Feature discovery on production models, comparative analysis of model versions
- **Benefit**: Works on air-gapped networks, full audit trail

---

## Core Use Cases

### Use Case 1: Feature Discovery Workflow
**Actor**: ML Researcher
**Goal**: Discover interpretable features in a GPT-2 model

**Steps**:
1. Download GPT-2-small from HuggingFace
2. Load OpenWebText dataset (10K samples)
3. Extract activations from layer 6 (residual stream)
4. Train sparse autoencoder (8x expansion, 10K steps)
5. Extract features and view max-activating examples
6. Generate natural language descriptions for top features

**Outcome**: Discover 50-100 interpretable features (e.g., "quotes", "negation", "proper nouns")

---

### Use Case 2: Model Steering Experiment
**Actor**: AI Safety Researcher
**Goal**: Reduce toxicity in model outputs

**Steps**:
1. Use pre-trained SAE on Llama-2-7B
2. Identify features associated with toxic language
3. Create steering preset: negative coefficient on toxicity features
4. Generate text with/without steering
5. Compare outputs and measure toxicity scores

**Outcome**: Reduce toxicity by 60% while maintaining coherence

---

### Use Case 3: Educational Demonstration
**Actor**: University Professor
**Goal**: Teach students about sparse autoencoders

**Steps**:
1. Project miStudio UI on classroom screen
2. Live-train a small SAE (5-minute training time)
3. Show real-time loss curves and sparsity metrics
4. Explore discovered features interactively
5. Demonstrate steering with audience-suggested prompts

**Outcome**: Students understand SAE training and feature discovery through hands-on demo

---

### Use Case 4: Edge Model Debugging
**Actor**: Robotics Engineer
**Goal**: Understand why navigation model makes unexpected turns

**Steps**:
1. Extract activations from deployed Jetson model
2. Train SAE to discover navigation-related features
3. Analyze features active during unexpected turns
4. Identify spurious feature (e.g., "bright light" feature)
5. Ablate feature and test improved behavior

**Outcome**: Fix navigation bug by removing spurious feature

---

## System Architecture Overview

MechInterp Studio implements mechanistic interpretability techniques for edge-deployed language models, enabling real-time feature discovery and manipulation on resource-constrained hardware (Jetson boards, Hailo processors, Coral Edge TPU, consumer GPUs).

### System Context

- **Deployment**: Single-user desktop application (local web UI)
- **Authentication**: None (single-user system)
- **Network**: Optional (only for HuggingFace downloads)
- **Storage**: Local filesystem (models, datasets, activations)
- **Compute**: Local GPU/accelerator (Jetson, NVIDIA consumer GPUs, Hailo, Coral)

### Design Principles

1. **Offline-First**: Core functionality works without internet connection
2. **Privacy-Preserving**: No telemetry, no cloud sync, no external API calls
3. **Resource-Aware**: Automatic memory management, quantization support
4. **User-Friendly**: Complete UI for all workflows, minimal configuration
5. **Extensible**: Plugin architecture for custom encoders and analysis tools

---

## Success Criteria

### Technical Metrics
- Train SAE on GPT-2-small (124M params) in < 30 minutes on Jetson Orin Nano
- Discover > 50 interpretable features per layer with < 20% dead neurons
- Real-time steering latency < 100ms for text generation
- Support models up to 3B parameters on 16GB consumer GPUs

### User Experience Metrics
- First-time user can complete feature discovery workflow in < 10 minutes
- Training configuration requires < 5 parameter changes from defaults
- Feature browser loads and filters 10,000+ features in < 1 second

### Adoption Metrics
- Used in > 10 university courses within first year
- > 1,000 GitHub stars within 6 months
- Active community creating and sharing SAE checkpoints

---

## Differentiation

### vs. Cloud-Based Tools (Neuronpedia, etc.)
| Feature | miStudio | Cloud Tools |
|---------|----------|-------------|
| **Privacy** | 100% local | Data uploaded |
| **Cost** | One-time hardware | Recurring GPU fees |
| **Latency** | < 100ms | Network dependent |
| **Offline** | Full functionality | Requires internet |
| **Customization** | Full access | Limited to API |

### vs. Research Code (TransformerLens, SAELens)
| Feature | miStudio | Research Code |
|---------|----------|---------------|
| **User Interface** | Complete GUI | Python scripts |
| **Learning Curve** | Hours | Weeks |
| **Deployment** | Single installer | Manual setup |
| **Edge Support** | Optimized | Manual porting |
| **Integration** | End-to-end | Modular libraries |

---

## Technical Workflow Overview

The following sections describe the **complete technical implementation** across six phases:

1. **Dataset Management**: HuggingFace integration, tokenization, storage
2. **Model Loading**: Quantization, memory mapping, activation extraction
3. **Autoencoder Training**: SAE/Skip/Transcoder architectures, optimization
4. **Feature Discovery**: Activation mapping, interpretation, analysis
5. **Visualization**: Interactive UI, real-time monitoring, dashboards
6. **Steering Tool**: Feature intervention, comparative generation, metrics

---

## Deep Technical Workflow

### Phase 1: Dataset Management Pipeline

**1.1 HuggingFace Integration Layer**
- **Dataset Discovery**: Query HF datasets API using `datasets` library with streaming support for memory-constrained environments
- **Metadata Extraction**: Pull dataset cards, token counts, splits, and feature schemas
- **Streaming Download**: Implement chunked download with `datasets.load_dataset(streaming=True)` to avoid memory overflow on edge devices
- **Checkpointing**: Resume interrupted downloads using Apache Arrow memory-mapped files

**1.2 Data Curation & Transformation**
- **Tokenization Pipeline**: 
  - Load model-specific tokenizer from HF
  - Batch tokenize with padding/truncation strategies
  - Generate attention masks and token type IDs
  - Store tokenized sequences in memory-mapped arrow format for zero-copy access
  
- **Activation Collection Preparation**:
  - Identify hook points (residual stream, MLP outputs, attention outputs per layer)
  - Pre-compute sequence length distributions for batch optimization
  - Generate stratified samples if full dataset exceeds memory constraints
  
- **Data Augmentation** (optional):
  - Implement dropout-based augmentation for robustness
  - Generate synthetic contrastive pairs for feature discovery

**1.3 Storage Architecture**
```
/data
  /raw                    # Original HF datasets
  /tokenized             # Arrow-format tokenized sequences
  /activations           # Cached activation tensors per layer
  /metadata              # JSON manifests with statistics
```

---

### Phase 2: Model Loading & Activation Extraction

**2.1 Model Selection & Quantization**
- **Model Registry**: Support for edge-optimized models (Phi-3-mini, Qwen2-1.5B, TinyLlama, Gemma-2B)
- **Quantization Pipeline**:
  - INT8/INT4 quantization using `bitsandbytes` or `quanto`
  - ONNX Runtime quantization for cross-platform inference
  - TensorRT optimization for Jetson (FP16/INT8 mixed precision)
  
- **Memory Mapping**: Load models with `device_map="auto"` and offloading strategies

**2.2 Activation Extraction Infrastructure**
- **Forward Hooks Registration**:
  - Attach hooks to transformer blocks at specified layers
  - Extract pre/post-MLP activations, attention outputs, residual stream
  - Handle different architectures (LLaMA, GPT-2, Pythia families)

- **Batch Processing**:
  - Stream inference with configurable batch sizes (1-8 for edge devices)
  - Accumulate activations in memory-mapped tensors
  - Shape: `[n_samples, seq_len, d_model]` where `d_model` is hidden dimension

- **Compression**: 
  - Optional FP16 storage to halve memory footprint
  - Chunk large activation matrices across files

---

### Phase 3: Autoencoder Training

**3.1 Encoder Architecture Selection**

**Sparse Autoencoder (SAE)**:
- **Architecture**: 
  - Encoder: Linear(`d_model → d_sae`) + ReLU
  - Decoder: Linear(`d_sae → d_model`)
  - Typical expansion factor: 4x-32x (e.g., 768 → 24,576)
  
- **Loss Function**:
  ```
  L = ||x - x̂||² + λ * ||f||₁ + λ_ghost * ghost_grad_penalty
  ```
  - MSE reconstruction loss
  - L1 sparsity penalty on activations `f`
  - Ghost gradient for dead neurons (neurons that never activate)

**Skip Autoencoder**:
- **Architecture**: SAE with residual connection from input to decoder
  ```
  x̂ = W_dec(ReLU(W_enc(x))) + α * x
  ```
- **Purpose**: Better preserve input information while extracting features
- **Hyperparameter**: `α` (residual scaling factor)

**Transcoder**:
- **Architecture**: Cross-layer feature translation
  ```
  h_layer_n → [Encoder] → latents → [Decoder] → h_layer_n+1
  ```
- **Use Case**: Model how features transform between layers
- **Training**: Predict next layer's activations from current layer

**3.2 Training Infrastructure**

- **Framework**: PyTorch with mixed precision (`torch.cuda.amp`)
- **Optimizer**: AdamW with cosine annealing schedule
- **Batch Strategy**: 
  - Sample random activation vectors from stored tensors
  - Batch size: 256-2048 depending on GPU memory
  
- **Sparsity Scheduling**:
  - Gradually increase L1 coefficient to target sparsity (L0 ≈ 10-100 active features)
  - Monitor activation frequency distributions

- **Checkpointing**:
  - Save encoder/decoder weights every N steps
  - Log loss curves, sparsity metrics, dead neuron counts
  - Early stopping on reconstruction loss plateau

- **Edge Optimization**:
  - Gradient accumulation for effective large batch sizes
  - Automatic mixed precision (AMP) for Jetson GPUs
  - Distributed training across multiple edge nodes (optional)

---

### Phase 4: Feature Discovery & Analysis

**4.1 Feature Extraction**
- **Activation Mapping**:
  - Pass evaluation dataset through trained autoencoder
  - Record which latent neurons activate for each token
  - Build sparse activation matrix: `[n_tokens, d_sae]`

- **Feature Statistics**:
  - Compute activation frequency per neuron
  - Calculate max activating examples (top-k tokens/sequences)
  - Measure feature correlation matrix to identify redundant features

**4.2 Feature Interpretation**

- **Max Activating Dataset Examples**:
  - For each feature, retrieve top-100 contexts where it activates strongest
  - Display tokens with gradient-based saliency highlighting
  - Cluster similar contexts using embedding similarity

- **Automated Description Generation**:
  - Feed max-activating examples to a separate LLM (can be same model)
  - Prompt: "What concept do these examples have in common?"
  - Generate natural language feature labels

- **Logit Lens Analysis**:
  - Decode feature vectors through unembedding matrix
  - Show top predicted tokens for each feature
  - Reveals what the model "thinks" the feature represents

- **Feature Ablation**:
  - Zero out specific features and measure perplexity change
  - Identify critical vs. redundant features

---

### Phase 5: Visualization Interface

**5.1 React Frontend Components**

- **Dataset Explorer**:
  - Table view of available datasets with metadata
  - Preview tokenized samples
  - Statistics dashboard (vocab distribution, sequence lengths)

- **Training Dashboard**:
  - Real-time loss curves (WebSocket streaming from backend)
  - Sparsity histogram (active features per sample)
  - GPU utilization and memory usage (via `nvidia-smi` polling)
  - Dead neuron tracking

- **Feature Browser**:
  - Grid/list view of discovered features
  - Search/filter by activation frequency, labels
  - Click to view max-activating examples with highlighting
  - Interactive correlation heatmap

**5.2 Visualization Techniques**

- **Activation Heatmaps**:
  - Token-level heatmap showing feature activation intensities
  - Color scale: white (inactive) → red (max activation)
  - Hoverable tokens with activation scores

- **Dimensionality Reduction**:
  - UMAP/t-SNE projection of feature vectors
  - Interactive scatter plot with hover tooltips
  - Color by activation frequency or semantic clusters

- **Feature Attribution**:
  - Integrated gradients to show token importance
  - Attention pattern overlay for context

---

### Phase 6: Steering Tool

**6.1 Feature Intervention Mechanism**

- **Steering Vector Construction**:
  - User selects features and scaling coefficients
  - Build steering vector: `s = Σ(αᵢ * fᵢ)` where `αᵢ` are scales, `fᵢ` are feature directions
  
- **Activation Addition**:
  - During inference, add steering vector to residual stream at target layer
  - `h_layer = h_layer + s`
  - Continue forward pass normally

**6.2 Comparative Generation Interface**

- **Dual-Pane View**:
  - Left: Unsteered generation
  - Right: Steered generation
  - Same prompt, temperature, top-p settings

- **Interactive Controls**:
  - Multi-select features with slider controls for coefficients
  - Layer selection dropdown (which layer to intervene)
  - Real-time regeneration button

- **Difference Highlighting**:
  - Token-level diff view showing divergence
  - Probability distribution shifts (logit comparisons)
  - Semantic similarity scores between outputs

**6.3 Steering Evaluation Metrics**

- **Quantitative**:
  - KL divergence between steered/unsteered distributions
  - Perplexity change
  - Feature activation change in downstream layers

- **Qualitative**:
  - User ratings of steering effectiveness
  - A/B testing framework for feature interpretability

---

## Technical Stack Recommendations

### Backend
- **Framework**: FastAPI (async Python) or Flask
- **Inference**: PyTorch, TensorRT, ONNX Runtime
- **Data**: Apache Arrow, HuggingFace `datasets`
- **Task Queue**: Celery with Redis (for async training jobs)

### Frontend
- **Framework**: React + TypeScript
- **Visualization**: D3.js, Plotly.js, Recharts
- **State Management**: Redux Toolkit or Zustand
- **Real-time**: Socket.io or WebSockets

### Edge Deployment
- **Containerization**: Docker with CUDA/TensorRT support
- **Orchestration**: Docker Compose for local, K3s for multi-node
- **Model Serving**: TorchServe or custom FastAPI endpoints
- **Monitoring**: Prometheus + Grafana for GPU metrics

---

## Key Challenges for Embedded Systems

1. **Memory Constraints**: 
   - Use streaming datasets, memory-mapped files
   - Quantize models and activations
   - Implement activation caching strategies

2. **Compute Limitations**:
   - Train smaller autoencoders (4x-8x expansion instead of 32x)
   - Use mixed precision training
   - Distribute training across multiple edge nodes

3. **Latency Requirements**:
   - Pre-cache common activations
   - Optimize steering vector addition with fused kernels
   - Use TensorRT for inference acceleration

4. **Storage**:
   - Compress activation datasets with FP16 or quantization
   - Implement LRU caching for frequently accessed features
   - Use sparse tensor storage formats

This workflow enables mechanistic interpretability research on edge devices, democratizing access to model analysis tools for developers without cloud-scale infrastructure.