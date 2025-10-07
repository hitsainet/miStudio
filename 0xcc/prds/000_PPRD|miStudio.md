# Project PRD: MechInterp Studio (miStudio)

**Version:** 1.0
**Created:** 2025-10-05
**Status:** Active Development
**Document Type:** Project-Level Product Requirements Document

---

## 1. Project Overview

### Project Name
**MechInterp Studio (miStudio)**

### Brief Description
MechInterp Studio is an edge-deployed mechanistic interpretability platform that brings advanced neural network analysis capabilities to resource-constrained hardware. It enables researchers, developers, and AI enthusiasts to discover, understand, and manipulate learned features in language models without requiring cloud infrastructure or expensive GPU clusters.

### Vision Statement
**Democratize mechanistic interpretability** by making it accessible on edge devices, enabling anyone with a Jetson board, consumer GPU, or AI accelerator to understand how their language models work.

### Primary Objectives
1. Enable **local-first, privacy-preserving** mechanistic interpretability research
2. Provide a **complete desktop application** with integrated UI for all interpretability workflows
3. Optimize performance for **edge hardware** (Jetson Orin Nano, consumer GPUs, AI accelerators)
4. Deliver an **end-to-end workflow** from dataset ingestion to model steering
5. Make interpretability research **accessible and affordable** (eliminate cloud costs)

### Problem Statement
Current mechanistic interpretability tools (Neuronpedia, TransformerLens) assume access to powerful cloud GPUs and high-bandwidth connections, creating barriers:

- **Infrastructure Barrier**: Require expensive cloud GPU access
- **Cost Prohibitive**: Training sparse autoencoders costs hundreds of dollars in GPU time
- **Privacy Concerns**: Cannot analyze proprietary or sensitive models locally
- **Deployment Gap**: Cannot analyze models in their edge deployment environment
- **Learning Curve**: Require deep Python/PyTorch expertise and manual scripting

### Opportunity
miStudio solves these problems by providing:
- **Single-user desktop application** with complete UI (no coding required)
- **Edge-optimized** for Jetson Orin Nano (8GB), consumer GPUs (RTX 3060+), AI accelerators
- **Privacy-preserving** - all computation happens locally
- **Cost-effective** - one-time hardware purchase vs. recurring cloud costs
- **Complete workflow** - dataset management → training → feature discovery → model steering

### Success Definition
miStudio is successful when:
1. Users can train sparse autoencoders on GPT-2-small in < 30 minutes on Jetson Orin Nano
2. First-time users complete feature discovery workflow in < 10 minutes
3. The platform is used in > 10 university courses within first year
4. Active community creates and shares SAE checkpoints
5. > 1,000 GitHub stars within 6 months

---

## 2. Project Goals & Objectives

### Primary Business Goals
1. **Educational Impact**: Become the standard teaching tool for mechanistic interpretability courses
2. **Research Enablement**: Enable independent researchers to conduct interpretability research without institutional resources
3. **Edge AI Advancement**: Demonstrate practical edge AI applications for ML interpretability
4. **Community Building**: Foster open-source community around edge interpretability

### Secondary Objectives
1. Support AI safety research by enabling local analysis of potentially harmful model features
2. Provide debugging tools for edge AI developers working with deployed models
3. Create reproducible interpretability experiments through checkpoint sharing
4. Demonstrate privacy-preserving ML analysis workflows

### Success Metrics and KPIs

**Technical Performance:**
- Train SAE on GPT-2-small (124M params) in < 30 minutes on Jetson Orin Nano
- Discover > 50 interpretable features per layer with < 20% dead neurons
- Real-time steering latency < 100ms for text generation
- Support models up to 3B parameters on 16GB consumer GPUs

**User Experience:**
- First-time user completes feature discovery workflow in < 10 minutes
- Training configuration requires < 5 parameter changes from defaults
- Feature browser loads and filters 10,000+ features in < 1 second

**Adoption:**
- Used in > 10 university courses within first year
- > 1,000 GitHub stars within 6 months
- Active community creating and sharing SAE checkpoints
- 100+ research papers cite or use miStudio

### Timeline and Milestone Expectations

**Phase 1 (Months 1-3): MVP Development**
- Core UI implementation (5 main tabs per Mock UI)
- Dataset management and HuggingFace integration
- Model loading with quantization support
- Basic SAE training with real-time monitoring

**Phase 2 (Months 4-6): Feature Complete**
- Feature discovery and analysis tools
- Model steering interface
- Advanced visualization (heatmaps, UMAP)
- Template/preset management system

**Phase 3 (Months 7-9): Polish & Optimization**
- Edge optimization (Jetson-specific improvements)
- Performance tuning and memory optimization
- Documentation and educational materials
- Community features (checkpoint sharing)

**Phase 4 (Months 10-12): Community Growth**
- University partnerships and course integration
- Workshop materials and tutorials
- Research collaborations
- Production deployment support

---

## 3. Target Users & Stakeholders

### Primary User Personas

#### 1. ML Researcher & Student
**Profile:**
- Graduate student or independent researcher
- Limited budget, no institutional GPU cluster access
- Learning mechanistic interpretability concepts
- Needs hands-on experimentation environment

**Needs:**
- Train SAEs on small models (GPT-2, TinyLlama) for learning
- Understand feature discovery process through visualization
- Low-cost experimentation (no cloud fees)
- Clear documentation and examples

**Use Cases:**
- Train SAE on GPT-2 to understand sparse autoencoders
- Discover interpretable features in small language models
- Experiment with different hyperparameters
- Complete course assignments and research projects

#### 2. Independent AI Researcher
**Profile:**
- Professional researcher without institutional affiliation
- Privacy-sensitive work (proprietary models)
- Needs reproducible experiments
- Cost-conscious

**Needs:**
- Analyze custom-trained models locally
- Full control over models and data
- Reproducible training with checkpoints
- Export results for papers/presentations

**Use Cases:**
- Discover features in fine-tuned models
- Experiment with steering techniques
- Conduct ablation studies on model features
- Generate publication-quality visualizations

#### 3. Edge AI Developer
**Profile:**
- Building edge AI applications (robotics, IoT)
- Models deployed on Jetson or similar hardware
- Debugging unexpected model behavior
- Performance-constrained environment

**Needs:**
- Analyze models in deployment environment
- Understand edge-optimized quantized models
- Debug unexpected outputs
- Optimize model behavior for edge

**Use Cases:**
- Extract activations from deployed Jetson model
- Identify spurious features causing bugs
- Test steering interventions for behavior correction
- Analyze quantized model differences

#### 4. AI Safety Researcher
**Profile:**
- Focused on identifying harmful model capabilities
- Requires complete model control
- Cannot use third-party cloud services
- Needs rigorous testing and ablation

**Needs:**
- Discover potentially dangerous features
- Test steering interventions safely
- Ablate capabilities systematically
- Maintain complete audit trail

**Use Cases:**
- Identify features associated with toxic content
- Measure effectiveness of safety interventions
- Compare model versions for capability changes
- Document safety analysis for audits

### Secondary Users

#### 5. University Professor / Educator
**Profile:**
- Teaching ML interpretability courses
- Needs classroom demonstration tools
- Limited time for technical setup
- Wants engaging visual teaching aids

**Needs:**
- Live demonstration capabilities
- Quick setup for classroom use
- Interactive exploration tools
- Student-friendly interface

**Use Cases:**
- Live-train SAE during lecture (5-minute training)
- Demonstrate feature discovery interactively
- Show steering effects with audience prompts
- Assign hands-on labs with clear workflows

#### 6. AI Ethics Auditor
**Profile:**
- Evaluating deployed models for bias/safety
- Air-gapped or restricted network environments
- Regulatory compliance requirements
- Needs comprehensive audit trails

**Needs:**
- Analyze production models offline
- Document all analysis steps
- Compare model versions systematically
- Generate compliance reports

**Use Cases:**
- Feature discovery on production models
- Bias detection through feature analysis
- Comparative analysis of model updates
- Generate audit documentation

### Key Stakeholders

**Internal Stakeholders:**
- Development team (full-stack, ML engineers)
- Product manager (roadmap and prioritization)
- Designer (UI/UX implementation per Mock UI)
- QA team (testing on edge hardware)

**External Stakeholders:**
- Open-source community (contributors, users)
- University partners (course integration)
- Edge AI hardware vendors (Jetson, Hailo partnerships)
- AI safety organizations (research collaborations)

### User Journey Overview

**Typical First-Time User Journey:**
1. **Installation**: Download and install miStudio (< 5 minutes)
2. **Model Download**: Download TinyLlama or GPT-2 from HuggingFace (< 10 minutes)
3. **Dataset Download**: Download TinyStories dataset (< 5 minutes)
4. **Training**: Train sparse autoencoder with default settings (< 30 minutes on Jetson)
5. **Discovery**: Browse discovered features, view max-activating examples
6. **Steering**: Select interesting features, test steering on custom prompts
7. **Share**: Export results or save checkpoint for later

**Power User Journey:**
- Import custom fine-tuned models
- Configure advanced hyperparameters
- Run multiple training experiments with templates
- Analyze features across multiple layers
- Create steering presets for specific behaviors
- Share checkpoints with research community

---

## 4. Project Scope

### What is Included in This Project

**Core Functionality:**
1. **Dataset Management System**
   - HuggingFace dataset browser and downloader
   - Local dataset ingestion and tokenization
   - Dataset statistics and preview
   - Streaming support for large datasets

2. **Model Management System**
   - HuggingFace model browser and downloader
   - Model quantization (FP16, Q8, Q4, Q2)
   - Model architecture viewer
   - Activation extraction pipeline

3. **SAE Training System**
   - Sparse Autoencoder (SAE) training
   - Skip Autoencoder support
   - Transcoder architecture support
   - Real-time training metrics dashboard
   - Checkpoint management (save/load/delete)
   - Training templates and presets

4. **Feature Discovery System**
   - Activation extraction and analysis
   - Max-activating example finder
   - Feature interpretation tools
   - Feature search and filtering
   - Logit lens analysis
   - Feature correlation analysis

5. **Model Steering System**
   - Feature-based intervention interface
   - Comparative generation (steered vs. unsteered)
   - Steering presets management
   - Real-time generation with steering
   - Metrics for steering effectiveness

**User Interface (per Mock UI specification):**
- Main navigation with 5 tabs: Datasets, Models, Training, Features, Steering
- Dark theme (slate color palette)
- Real-time progress indicators
- Modal dialogs for detailed views
- Responsive layouts
- Interactive visualizations

**Infrastructure:**
- REST API backend (FastAPI/NestJS)
- PostgreSQL database for metadata
- Redis for caching and real-time updates
- Job queue for long-running tasks
- WebSocket for real-time progress updates

**Documentation:**
- User guide and tutorials
- API documentation
- Developer setup guide
- Edge deployment guide (Jetson-specific)

### What is Explicitly Out of Scope

**Not Included in Initial Release:**
1. **Multi-user Features**
   - User authentication/authorization
   - Multi-tenancy
   - Role-based access control
   - User management

2. **Cloud Features**
   - Cloud sync/backup
   - Remote collaboration
   - Cloud training offload
   - Distributed training across cloud nodes

3. **Advanced Analysis**
   - Automated circuit discovery
   - Causal tracing
   - Mechanistic anomaly detection
   - Adversarial feature analysis

4. **Production Deployment Tools**
   - Model serving API
   - Production monitoring
   - A/B testing framework
   - Automated steering in production

5. **Data Management**
   - Version control for datasets
   - Data provenance tracking
   - Automated data cleaning
   - Data augmentation pipelines

### Future Roadmap Considerations

**Version 2.0 (Future):**
- Multi-model comparison interface
- Automated feature labeling with LLMs
- Cross-layer feature tracking
- Circuit discovery tools
- Advanced visualizations (3D feature spaces)

**Version 3.0 (Future):**
- Optional cloud sync for backups
- Collaborative research features
- Integration with external interpretability tools
- Automated research workflows

**Long-term Vision:**
- Mobile/tablet version for monitoring
- Web-based version (optional)
- Enterprise features (teams, permissions)
- Integration with MLOps platforms

### Dependencies and Assumptions

**Technical Dependencies:**
- PyTorch 2.0+ for ML operations
- HuggingFace Transformers and Datasets libraries
- CUDA/TensorRT for GPU acceleration
- PostgreSQL for data persistence
- Redis for caching
- Node.js/React for frontend

**Hardware Assumptions:**
- Target: Jetson Orin Nano (8GB) or equivalent
- Minimum: 16GB RAM, 4GB GPU VRAM
- Storage: 256GB SSD minimum
- Network: Internet for initial downloads (optional for core features)

**User Assumptions:**
- Basic understanding of language models
- Familiarity with ML concepts (training, inference)
- Comfortable with desktop applications
- Has access to edge hardware or consumer GPU

**Business Assumptions:**
- Open-source project with community contributions
- Funding through grants/sponsorships
- Hardware vendor partnerships (NVIDIA, Hailo)
- Academic partnerships for validation

---

## 5. High-Level Requirements

### Core Functional Requirements

**FR-1: Dataset Management**
- System shall support downloading datasets from HuggingFace
- System shall support local dataset upload and ingestion
- System shall tokenize datasets using model-specific tokenizers
- System shall display dataset statistics and previews
- System shall support streaming for large datasets
- System shall cache tokenized datasets for reuse

**FR-2: Model Management**
- System shall support downloading models from HuggingFace
- System shall support INT8, INT4, FP16 quantization
- System shall display model architecture and layer details
- System shall extract activations from specified layers
- System shall support models up to 3B parameters on 16GB GPU

**FR-3: SAE Training**
- System shall train Sparse Autoencoders on model activations
- System shall support Skip Autoencoder and Transcoder architectures
- System shall display real-time training metrics (loss, sparsity, dead neurons)
- System shall support training pause/resume/stop operations
- System shall save checkpoints automatically and on-demand
- System shall support training templates for common configurations

**FR-4: Feature Discovery**
- System shall extract features from trained SAEs
- System shall identify max-activating examples for each feature
- System shall compute activation frequency and interpretability scores
- System shall support feature search and filtering
- System shall display feature correlations and relationships
- System shall generate automatic feature descriptions (optional)

**FR-5: Model Steering**
- System shall support feature-based activation interventions
- System shall generate text with and without steering simultaneously
- System shall display comparative outputs with difference highlighting
- System shall support steering presets for common interventions
- System shall measure steering effectiveness (perplexity, KL divergence)

### Non-Functional Requirements

**Performance:**
- Training: SAE training on GPT-2-small < 30 min on Jetson Orin Nano
- Latency: API response times < 100ms (p95)
- Steering: Real-time generation with < 100ms steering overhead
- UI: Feature browser renders 10,000+ features in < 1 second

**Scalability:**
- Support models up to 3B parameters on 16GB consumer GPUs
- Support datasets up to 50GB on local storage
- Support 10,000+ features per trained SAE
- Support 100+ concurrent WebSocket connections (future multi-user)

**Reliability:**
- System shall recover from crashes without data loss
- Training shall support automatic checkpoint recovery
- System shall validate all user inputs before processing
- System shall handle GPU out-of-memory gracefully

**Usability:**
- First-time user completes workflow in < 10 minutes
- Training configuration requires < 5 parameter changes
- All workflows accessible via UI (no CLI/scripting required)
- Clear error messages with actionable suggestions

**Maintainability:**
- Modular architecture for easy component updates
- Comprehensive logging for debugging
- Automated tests for critical paths
- Clear documentation for developers

### Compliance and Regulatory Considerations

**Privacy:**
- All data processing happens locally (no cloud transmission)
- No telemetry or analytics collection
- No external API calls except HuggingFace downloads
- User has complete control over all data

**Open Source:**
- Released under permissive open-source license (Apache 2.0 / MIT)
- All dependencies compatible with open-source distribution
- No proprietary components required

**Accessibility:**
- Follow WCAG 2.1 AA guidelines where applicable
- Keyboard navigation support
- Screen reader compatibility (future enhancement)
- High-contrast dark theme for low vision users

### Integration and Compatibility Requirements

**Platform Support:**
- Primary: Linux (Ubuntu 20.04+, Jetson Linux)
- Secondary: Windows 10+, macOS (Intel/Apple Silicon)
- Architectures: x86_64, ARM64 (Jetson)

**Hardware Compatibility:**
- NVIDIA GPUs with CUDA support
- Jetson Orin Nano, Orin NX, AGX Xavier
- Consumer GPUs: RTX 3060+, AMD equivalents (future)
- AI accelerators: Hailo-8, Coral TPU (future)

**Software Integration:**
- HuggingFace Hub for model/dataset downloads
- Local filesystem for storage
- No external service dependencies for core features

---

## 6. Feature Breakdown

### Core Features (MVP / Essential Functionality)

#### Feature 1: Dataset Management Panel
**Priority:** P0 (Blocker for MVP)
**Description:** Complete dataset lifecycle management including HuggingFace downloads, local ingestion, tokenization, and preview. Implements the "Datasets" tab from Mock UI specification.

**User Value:**
- Researchers can easily acquire training data without manual downloads
- Support for both public datasets and private/custom data
- Visual feedback during downloads and processing
- Dataset quality verification before training

**Dependencies:** None
**Estimated Complexity:** Medium

---

#### Feature 2: Model Management Panel
**Priority:** P0 (Blocker for MVP)
**Description:** Model download, quantization, architecture inspection, and activation extraction setup. Implements the "Models" tab from Mock UI specification.

**User Value:**
- Easy model acquisition from HuggingFace
- Automatic quantization for edge hardware
- Understand model structure before training
- Configure activation extraction points

**Dependencies:** None
**Estimated Complexity:** Medium-High

---

#### Feature 3: SAE Training System
**Priority:** P0 (Blocker for MVP)
**Description:** Complete sparse autoencoder training pipeline with real-time monitoring, checkpoint management, and training control (pause/resume/stop). Implements the "Training" tab from Mock UI specification.

**User Value:**
- Core interpretability capability (train SAEs locally)
- Real-time feedback on training progress
- Experiment with different architectures and hyperparameters
- Save/resume training for long experiments

**Dependencies:** Dataset Management, Model Management
**Estimated Complexity:** High

---

#### Feature 4: Feature Discovery & Browser
**Priority:** P0 (Blocker for MVP)
**Description:** Extract features from trained SAEs, compute statistics, find max-activating examples, and provide searchable/filterable feature browser. Implements the "Features" tab from Mock UI specification.

**User Value:**
- Primary research output (interpretable features)
- Understand what the model has learned
- Identify interesting or unexpected features
- Export findings for papers/presentations

**Dependencies:** SAE Training System
**Estimated Complexity:** High

---

#### Feature 5: Model Steering Interface
**Priority:** P0 (Blocker for MVP)
**Description:** Feature-based model intervention with comparative generation (steered vs. unsteered) and real-time control. Implements the "Steering" tab from Mock UI specification.

**User Value:**
- Test understanding of features by manipulating them
- Control model behavior in specific ways
- Validate feature interpretations
- Demonstrate interpretability concepts

**Dependencies:** Feature Discovery
**Estimated Complexity:** Medium-High

---

### Secondary Features (Important but Not Critical)

#### Feature 6: Training Templates & Presets
**Priority:** P0 (MVP - Integrated into Training System)
**Description:** Complete template management system for saving, loading, deleting, favoriting, exporting, and importing training configurations. Auto-generated descriptive names with timestamp uniqueness. Fully integrated into Training tab UI with collapsible management section.

**User Value:**
- Faster experiment iteration with one-click template loading
- Share reproducible configurations with community via export/import
- Learn from community best practices
- Never lose successful hyperparameter configurations
- Compare training results across different configurations

**Key Features:**
- Save current configuration as template (with auto-generated name)
- Load template to restore all hyperparameters + model + dataset selections
- Delete unwanted templates
- Favorite important templates for quick access
- Export all templates to JSON file for backup/sharing
- Import templates from JSON file (with validation)
- Auto-naming: `{encoder}_{expansion}x_{steps}steps_{HHMM}` (e.g., `sparse_8x_10000steps_1430`)
- Collapsible "Saved Templates" section with template count

**Dependencies:** SAE Training System
**Estimated Complexity:** Medium (full CRUD + export/import)

---

#### Feature 7: Extraction Templates
**Priority:** P0 (MVP - Integrated into Model Management)
**Description:** Complete template management system for activation extraction configurations. Save/load layer selections, hook types, and sampling strategies. Auto-generated descriptive names. Fully integrated into Activation Extraction modal.

**User Value:**
- Quick setup for common extraction patterns (e.g., "all residual streams")
- Standardized extraction for reproducibility across experiments
- Share extraction strategies with collaborators
- Avoid reconfiguring complex multi-layer extractions

**Key Features:**
- Save extraction configuration as template
- Load template to restore layer selections + hook types + sampling settings
- Delete unwanted templates
- Favorite frequently-used templates
- Auto-naming: `{type}_layers{min}-{max}_{samples}samples_{HHMM}` (e.g., `residual_layers0-11_1000samples_1430`)
- Collapsible "Saved Templates" section within Activation Extraction modal
- Included in combined export/import with training templates and steering presets

**Dependencies:** Model Management, Activation Extraction
**Estimated Complexity:** Medium

---

#### Feature 8: Steering Presets
**Priority:** P0 (MVP - Integrated into Steering System)
**Description:** Complete preset management system for steering configurations. Save/load feature selections, coefficients, intervention layers, and temperature settings. Linked to training jobs for context. Auto-generated descriptive names. Fully integrated into Steering tab UI.

**User Value:**
- Reuse effective steering interventions instantly
- Compare different steering strategies systematically
- Share successful interventions with community
- Document steering experiments for reproducibility
- Quick access to favorite presets

**Key Features:**
- Save current steering configuration as preset (includes training_id reference)
- Load preset to restore feature selections + coefficients + intervention layers + temperature
- Delete unwanted presets
- Favorite powerful or interesting presets
- Export/import presets alongside templates
- Auto-naming: `steering_{count}features_layer{N}_{HHMM}` or `steering_{count}features_layers{min}-{max}_{HHMM}`
- Shows training job context: `{encoderType} SAE • {modelName} • {datasetName}`
- Collapsible "Saved Presets" section with preset count

**Dependencies:** Model Steering Interface, Feature Discovery
**Estimated Complexity:** Medium

---

#### Feature 8a: Multi-Layer Training Support
**Priority:** P0 (MVP - Core Enhancement to Training System)
**Description:** Train sparse autoencoders on multiple transformer layers simultaneously instead of single-layer training. Dynamic UI generation based on model architecture. Enables efficient multi-layer analysis in a single training job.

**User Value:**
- Faster iteration for multi-layer feature analysis
- Unified training context (same data, same hyperparameters across layers)
- Compare feature emergence patterns across layers
- More efficient than running separate single-layer training jobs
- Analyze layer interactions and feature evolution

**Key Features:**
- Change from `trainingLayer: number` to `trainingLayers: number[]` in hyperparameters
- 8-column checkbox grid for layer selection (dynamically generated from model architecture)
- Select All / Clear All buttons for convenience
- Visual feedback: selected layers highlighted in emerald green
- Shows "Training Layers (N selected)" label with count
- Model architecture metadata (num_layers, hidden_dim, num_heads) stored in database
- Checkpoint format: sub-directories per layer for independent SAE states
- Memory-aware: recommend ≤4 layers on 8GB Jetson

**UI Impact:**
- Training Panel: Dynamic layer selector grid adapts to model (TinyLlama: 22, Phi-2: 32)
- ModelArchitectureViewer: Shows actual layer count, hidden dim, attention heads
- Training templates: Include trainingLayers array

**Dependencies:** SAE Training System, Model Management (architecture metadata)
**Estimated Complexity:** High (training pipeline changes, checkpoint management)

---

#### Feature 8b: Multi-Layer Steering Support
**Priority:** P0 (MVP - Core Enhancement to Steering System)
**Description:** Apply steering interventions across multiple transformer layers simultaneously instead of single-layer interventions. Same UI pattern as multi-layer training. Enables more powerful interventions and layer interaction studies.

**User Value:**
- More powerful steering interventions (cascading effects across layers)
- Explore layer-wise steering effects systematically
- Support cascading interventions for complex behavior modifications
- Facilitate ablation studies (which layers matter most)
- Research layer interaction effects

**Key Features:**
- Change from `interventionLayer: number` to `interventionLayers: number[]`
- 8-column checkbox grid matching training layer selector pattern
- Select All / Clear All buttons
- Dynamic generation based on selected model's architecture
- Shows "Intervention Layers (N selected)" label with count
- Hook registration: Register forward hooks at each specified layer
- Same steering vector (features + coefficients) applied at all layers
- Preset naming includes layer range: `steering_3features_layers6-12_1430`

**UI Impact:**
- Steering Panel: Dynamic layer selector grid below feature selector
- Steering presets: Include intervention_layers array
- Preset display: Shows single or range format for layers

**Dependencies:** Model Steering Interface, Model Management (architecture metadata)
**Estimated Complexity:** Medium-High (hook management, multi-layer intervention logic)

---

#### Feature 8c: Training Job Selector (Steering Context)
**Priority:** P0 (MVP - Core Enhancement to Steering System)
**Description:** Dropdown selector in Steering tab to choose which completed training job's features to use for steering. Provides clear context about SAE source and enables switching between different trained SAEs.

**User Value:**
- Clear context: Know which SAE's features are being used
- Easy switching between different trained SAEs for comparison
- Prevents confusion about feature source
- Enables systematic comparison of steering effectiveness across different SAEs
- Training job context preserved in steering presets

**Key Features:**
- "Training Job (Source of Features)" label at top of Steering configuration
- Dropdown showing only completed trainings (status = 'completed')
- Display format: `{encoderType} SAE • {modelName} • {datasetName} • Started {date}`
- Example: "sparse SAE • TinyLlama-1.1B • OpenWebText-10K • Started 1/15/2025"
- Dynamic lookup of model and dataset names from IDs
- Empty state message when no completed trainings available
- Selected training_id saved with steering presets for reproducibility
- Preset loading restores correct training job selection

**UI Impact:**
- Steering Panel: New dropdown selector at top of configuration section
- Matches display format from Features tab for consistency
- Steering presets: Include training_id reference

**Dependencies:** Model Steering Interface, Feature Discovery, SAE Training System
**Estimated Complexity:** Low-Medium (mostly UI + state management)

---

#### Feature 9: Advanced Visualizations
**Priority:** P1
**Description:** UMAP/t-SNE feature projections, correlation heatmaps, activation patterns across samples.

**User Value:**
- Deeper understanding of feature spaces
- Identify feature clusters and relationships
- Publication-quality visualizations

**Dependencies:** Feature Discovery
**Estimated Complexity:** Medium

---

#### Feature 10: Feature Analysis Tools
**Priority:** P1
**Description:** Logit lens analysis, feature ablation studies, automated feature descriptions using LLMs.

**User Value:**
- Richer feature interpretations
- Quantitative feature importance
- Automated documentation of findings

**Dependencies:** Feature Discovery
**Estimated Complexity:** Medium

---

#### Feature 11: Checkpoint Auto-Save
**Priority:** P1
**Description:** Automatic checkpoint saving at configurable intervals during training.

**User Value:**
- Protection against crashes or interruptions
- Ability to rollback to earlier training states
- Less manual checkpoint management

**Dependencies:** SAE Training System
**Estimated Complexity:** Low

---

#### Feature 12: Dataset Statistics Dashboard
**Priority:** P2
**Description:** Detailed statistics, distributions, and quality metrics for datasets.

**User Value:**
- Understand dataset characteristics before training
- Identify potential data quality issues
- Select appropriate datasets for research questions

**Dependencies:** Dataset Management
**Estimated Complexity:** Low-Medium

---

### Future Features (Nice-to-Have / Roadmap Items)

#### Feature 13: Multi-Model Comparison
**Priority:** P3 (Future)
**Description:** Side-by-side comparison of features across different models or training runs.

**User Value:**
- Understand model differences systematically
- Track feature evolution during training
- Compare effects of different architectures

**Dependencies:** Feature Discovery
**Estimated Complexity:** Medium

---

#### Feature 14: Export & Reporting
**Priority:** P3 (Future)
**Description:** Export features, visualizations, and analysis results in multiple formats (PDF, HTML, JSON).

**User Value:**
- Generate publication-ready figures
- Share findings with collaborators
- Archive research results

**Dependencies:** Feature Discovery, Advanced Visualizations
**Estimated Complexity:** Medium

---

#### Feature 15: Collaborative Features
**Priority:** P3 (Future)
**Description:** Share checkpoints, presets, and findings with community (optional cloud sync).

**User Value:**
- Learn from community research
- Contribute to shared knowledge base
- Reproducible research

**Dependencies:** All core features
**Estimated Complexity:** High

---

#### Feature 16: Advanced Circuit Analysis
**Priority:** P3 (Future)
**Description:** Automated circuit discovery, causal tracing, mechanistic anomaly detection.

**User Value:**
- Deeper mechanistic understanding
- Automated discovery of computational patterns
- Advanced research capabilities

**Dependencies:** Feature Discovery, Analysis Tools
**Estimated Complexity:** Very High

---

## 7. User Experience Goals

### Overall UX Principles and Guidelines

**Design Principles (from Mock UI):**
1. **Dark-First Design**: Slate dark theme optimized for long research sessions
2. **Information Density**: Show relevant data without overwhelming users
3. **Progressive Disclosure**: Advanced features hidden behind toggles
4. **Real-Time Feedback**: Immediate visual feedback for all actions
5. **Accessibility**: Clear labels, keyboard navigation, high contrast
6. **Consistency**: Uniform component patterns across all tabs

**Visual Design Language:**
- **Color Palette**: Slate grays (bg-slate-900, 950), emerald accents for success, red for errors
- **Typography**: Clear hierarchy with semibold headings, medium body text
- **Spacing**: Generous padding and margins for readability
- **Icons**: Lucide icons for consistent visual language
- **Animations**: Subtle transitions (progress bars, loading spinners)

**Interaction Patterns:**
- **Primary Actions**: Prominent emerald buttons (Start Training, Download, etc.)
- **Secondary Actions**: Gray buttons for auxiliary functions
- **Destructive Actions**: Red text/icons with confirmation dialogs
- **Status Indicators**: Color-coded badges and icons (green=ready, yellow=processing, red=error)

### Accessibility Requirements

**Keyboard Navigation:**
- All interactive elements accessible via Tab key
- Clear focus indicators on all focusable elements
- Keyboard shortcuts for common actions (future enhancement)

**Visual Accessibility:**
- High-contrast dark theme (meets WCAG AA standards)
- Text size minimum 14px (16px for body text)
- Clear visual hierarchy
- Adequate spacing between interactive elements

**Assistive Technology:**
- Semantic HTML elements for screen readers
- ARIA labels for complex UI components
- Alt text for icons and images
- Form labels associated with inputs

### Performance Expectations

**UI Responsiveness:**
- Initial page load: < 2 seconds
- Tab switching: < 100ms
- Modal opening: < 50ms
- Feature browser filtering: < 500ms for 10,000 features
- Real-time chart updates: 60fps

**Perceived Performance:**
- Optimistic UI updates (show changes immediately, sync in background)
- Skeleton screens during loading
- Progressive loading for large datasets
- Smooth animations and transitions

**Feedback Timing:**
- Button click feedback: Immediate (<16ms)
- Form validation: Real-time as user types
- Long operations: Progress indicators with time estimates
- Success/error messages: Visible for 3-5 seconds

### Cross-Platform Considerations

**Primary Platform: Linux Desktop**
- Native performance on Ubuntu 20.04+
- Optimized for Jetson Linux environment
- Full feature support

**Secondary Platforms:**
- **Windows 10+**: Full feature parity, minor UI adjustments
- **macOS**: Full feature parity, test on both Intel and Apple Silicon
- **Future**: Web-based version for remote monitoring

**Responsive Design:**
- Minimum resolution: 1280x720 (720p)
- Optimal resolution: 1920x1080 (1080p)
- Support ultra-wide monitors (3440x1440)
- Future: Tablet support for monitoring training

**Browser Requirements (if web-based):**
- Modern browsers: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- WebSocket support required
- IndexedDB for client-side caching

---

## 8. Business Considerations

### Budget and Resource Constraints

**Development Budget:**
- **Funding Model**: Open-source project with grant funding or sponsorships
- **Initial Phase**: Bootstrap with volunteer contributors
- **Growth Phase**: Seek academic grants, hardware vendor sponsorships
- **Target Funding**: $50K-$100K for first year (salaries, hardware, hosting)

**Resource Allocation:**
- **Core Team**: 2-3 full-time developers (1 backend/ML, 1 frontend, 1 DevOps)
- **Part-Time**: Product manager, designer (can be shared roles)
- **Community**: Open-source contributors for features, testing, documentation

**Hardware Costs:**
- Development hardware: 2-3 Jetson Orin Nano boards ($499 each)
- Testing: Various consumer GPUs (RTX 3060, 4070, etc.)
- CI/CD: Cloud GPU runners for automated testing (~$200/month)

### Risk Assessment and Mitigation

**Technical Risks:**

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Edge hardware performance insufficient | High | Medium | Early performance testing, fallback optimizations |
| PyTorch edge support issues | High | Low | Test on target hardware early, use TensorRT |
| GPU memory constraints | High | Medium | Implement aggressive memory management, streaming |
| HuggingFace API changes | Medium | Medium | Abstract integration layer, version pinning |

**Business Risks:**

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Limited adoption | High | Medium | Focus on university partnerships, tutorials |
| Competition from cloud tools | Medium | High | Emphasize privacy, cost, offline capabilities |
| Insufficient funding | High | Medium | Multiple funding sources, bootstrap approach |
| Loss of key contributors | Medium | Low | Documentation, knowledge sharing, bus factor > 1 |

**Community Risks:**

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Low contributor interest | Medium | Medium | Clear contribution guidelines, good onboarding |
| Poor code quality from contributors | Low | High | Code review process, automated testing |
| Community fragmentation | Low | Low | Clear governance model, active maintainers |

### Competitive Landscape Awareness

**Direct Competitors:**
- **Neuronpedia**: Cloud-based, feature browser, limited to public SAEs
- **TransformerLens**: Python library, requires coding, no UI
- **SAELens**: Research library, no production features

**Competitive Advantages:**
1. **Privacy-First**: Only solution for local, private interpretability
2. **Complete UI**: No coding required for basic workflows
3. **Edge-Optimized**: Only tool designed for edge hardware
4. **Cost**: One-time hardware vs. recurring cloud costs
5. **Offline**: Works without internet after initial setup

**Differentiation Strategy:**
- Focus on education and accessibility
- Hardware vendor partnerships (NVIDIA, Hailo)
- Academic partnerships for validation and adoption
- Community-driven checkpoint and preset sharing

### Monetization or Value Creation Model

**Primary Model: Open Source**
- Free and open-source software (Apache 2.0 / MIT license)
- No direct monetization of software
- Value creation through community and ecosystem

**Revenue Streams (Optional):**
1. **Support Contracts**: Enterprise support for custom deployments
2. **Training/Workshops**: Paid workshops and courses
3. **Consulting**: Custom interpretability research projects
4. **Grants**: Academic and research grants
5. **Sponsorships**: Hardware vendors, AI safety organizations

**Value Proposition:**
- **For Users**: Free, powerful interpretability tool
- **For Education**: Teaching platform and research enabler
- **For Industry**: Debugging tool for edge AI products
- **For Research**: Reproducible interpretability experiments

---

## 9. Technical Considerations (High-Level)

### Deployment Environment Preferences

**Primary Deployment: Edge Devices**
- Jetson Orin Nano (8GB RAM, 1024 CUDA cores)
- Jetson Orin NX (8-16GB RAM, up to 1024 CUDA cores)
- Consumer GPUs: RTX 3060 (12GB), 4060 Ti (16GB), 4070 (12GB)

**Deployment Architecture:**
- **Single-User Desktop Application**
- Backend API + Frontend UI running locally
- Local PostgreSQL and Redis instances
- Local filesystem storage
- Optional: Docker containers for easier deployment

**Installation Methods:**
1. Native installers (AppImage for Linux, MSI for Windows, DMG for macOS)
2. Docker Compose for containerized deployment
3. Source installation for developers

### Security and Privacy Requirements

**Data Security:**
- All data stays local (models, datasets, activations)
- No telemetry or usage tracking
- No external API calls except HuggingFace downloads (user-initiated)
- Local authentication optional (for shared workstations)

**Code Security:**
- Input validation on all API endpoints
- Sanitization of file paths to prevent traversal attacks
- Rate limiting on expensive operations
- GPU memory monitoring to prevent OOM crashes

**Privacy by Design:**
- No cloud sync by default
- No user data collection
- Open-source code for transparency
- Optional telemetry with explicit opt-in (future)

### Performance and Scalability Needs

**Performance Targets:**
- SAE training: < 30 min for GPT-2-small on Jetson
- UI responsiveness: < 100ms for all interactions
- Feature browser: < 1s for 10,000 features
- Real-time steering: < 100ms overhead

**Scalability Considerations:**
- Support up to 3B parameter models on 16GB GPU
- Handle datasets up to 50GB
- Process 10,000+ features per SAE
- Efficient memory management for edge devices

**Optimization Strategies:**
- Mixed precision training (FP16)
- Gradient accumulation for larger effective batch sizes
- Memory-mapped files for large datasets
- Streaming inference for activation extraction
- TensorRT optimization for Jetson
- Lazy loading for UI components

### Integration and API Requirements

**HuggingFace Hub Integration:**
- Use `datasets` library for dataset downloads
- Use `transformers` library for model downloads
- Support private/gated models with access tokens
- Handle rate limiting and download resume

**Internal API Design:**
- RESTful API for CRUD operations
- WebSocket for real-time updates
- Job queue for long-running tasks
- Clear API versioning (v1)

**External Integration Points:**
- HuggingFace Hub API (read-only)
- Local filesystem (models, datasets, checkpoints)
- GPU drivers (CUDA, TensorRT)

### Technology Stack (High-Level)

**Backend:**
- Framework: FastAPI (Python) or NestJS (TypeScript)
- ML: PyTorch 2.0+, HuggingFace Transformers/Datasets
- Database: PostgreSQL 14+
- Cache: Redis 7+
- Queue: Celery (Python) or BullMQ (Node.js)

**Frontend:**
- Framework: React 18+ with TypeScript
- State: Redux Toolkit or Zustand
- UI: Custom components per Mock UI specification
- Charts: D3.js, Recharts, or Plotly.js
- Real-time: Socket.io or native WebSockets

**Infrastructure:**
- Containerization: Docker + Docker Compose
- Edge Runtime: TensorRT for inference optimization
- Storage: Local filesystem + PostgreSQL
- Monitoring: Prometheus + Grafana (optional)

**Note:** Detailed technology decisions will be made in the Architecture Decision Record (ADR).

---

## 10. Project Constraints

### Timeline Constraints

**Hard Deadlines:**
- MVP (Core Features 1-5): 3 months from project start
- Beta Release: 6 months (includes secondary features)
- V1.0 Release: 9 months (polished, documented, tested)

**External Dependencies:**
- Hardware availability (Jetson boards for testing)
- HuggingFace API stability
- PyTorch release schedule (dependency updates)

**Critical Path:**
1. Backend API + Database setup (2 weeks)
2. Dataset Management (3 weeks)
3. Model Management (3 weeks)
4. SAE Training (6 weeks) ← Critical path bottleneck
5. Feature Discovery (4 weeks)
6. Model Steering (3 weeks)
7. Polish + Testing (4 weeks)

### Budget Limitations

**Development Costs:**
- Limited to open-source/grant funding
- Cannot afford large-scale cloud infrastructure
- Must optimize for commodity hardware

**Hardware Costs:**
- Target users have limited budget ($500-$2000 for hardware)
- Cannot assume access to high-end GPUs
- Must work on 8GB Jetson Orin Nano

**Operational Costs:**
- Minimal hosting costs (documentation site, CI/CD)
- No ongoing service costs (everything local)
- Community-driven support model

### Resource Availability

**Development Team:**
- Small core team (2-3 developers)
- Part-time contributors from community
- Limited design/UX resources (use Mock UI as spec)

**Hardware Resources:**
- Limited test hardware (2-3 Jetson boards)
- Community testing for diverse hardware
- CI/CD on cloud GPU runners (limited hours)

**Time Constraints:**
- Developers may have other commitments
- Contributors are volunteers
- Seasonal availability (students during semester)

### Technical or Regulatory Constraints

**Technical Constraints:**
- Must run on edge hardware (limited compute)
- Cannot require cloud connectivity
- Must support air-gapped environments
- Limited GPU memory (8-16GB)

**Platform Constraints:**
- Linux primary, Windows/macOS secondary
- CUDA/TensorRT dependencies (NVIDIA GPUs)
- Python/Node.js runtime requirements

**Regulatory Constraints:**
- Open-source license compatibility (all dependencies)
- No GDPR concerns (no user data collection)
- Export control compliance (research software)
- Academic use compliance (HuggingFace models)

**Community Constraints:**
- Must accept community contributions
- Open governance model required
- Code of conduct enforcement
- Inclusive and welcoming environment

---

## 11. Success Metrics

### Quantitative Success Measures

**Technical Performance:**
- ✅ Train SAE on GPT-2-small in < 30 minutes on Jetson Orin Nano
- ✅ Discover > 50 interpretable features per layer with < 20% dead neurons
- ✅ Real-time steering latency < 100ms for text generation
- ✅ Support models up to 3B parameters on 16GB consumer GPUs
- ✅ API response times < 100ms (p95)
- ✅ Feature browser handles 10,000+ features with < 1s load time

**Adoption Metrics:**
- 🎯 1,000 GitHub stars within 6 months
- 🎯 Used in 10+ university courses within first year
- 🎯 100+ active users per month (measured by GitHub discussions/issues)
- 🎯 50+ community-contributed checkpoints/presets
- 🎯 10+ research papers cite or use miStudio

**User Engagement:**
- 🎯 First-time user completes workflow in < 10 minutes (usability testing)
- 🎯 Training configuration requires < 5 parameter changes from defaults
- 🎯 50% of users return for second session within a week
- 🎯 Average session length > 30 minutes (indicates productive use)

**Community Health:**
- 🎯 20+ active contributors (>1 commit per month)
- 🎯 100+ community members (GitHub discussions, Discord)
- 🎯 Average issue response time < 48 hours
- 🎯 90%+ of PRs reviewed within 1 week

### Qualitative Success Indicators

**User Satisfaction:**
- Positive feedback in GitHub issues and discussions
- High-quality bug reports (indicates investment)
- Feature requests aligned with roadmap (indicates understanding)
- Community members help each other (self-sustaining support)

**Research Impact:**
- Used in published research papers
- Cited in mechanistic interpretability courses
- Referenced in blog posts and tutorials
- Invited talks at ML conferences

**Educational Impact:**
- Adopted by university courses
- Used in workshops and tutorials
- Students contribute to codebase
- Professors recommend to students

**Industry Recognition:**
- Hardware vendor partnerships (NVIDIA, Hailo)
- Featured in AI/ML newsletters
- Conference presentations accepted
- Media coverage in tech publications

### User Satisfaction Metrics

**Usability Metrics:**
- System Usability Scale (SUS) score > 75 (Good)
- Task completion rate > 90% for core workflows
- Error rate < 5% for typical tasks
- User reports minimal learning curve

**Support Metrics:**
- 80%+ of questions answered by community
- Common issues documented in FAQ
- < 5% of users abandon during first session
- Positive sentiment in community discussions

**Feature Satisfaction:**
- 90%+ of users use core features (datasets, training, discovery)
- 50%+ of users use advanced features (steering, visualization)
- 70%+ of users utilize template/preset management systems
- 30%+ of users leverage multi-layer training/steering capabilities
- Feature requests indicate understanding of capabilities
- Users share workflows and best practices
- Community shares templates and presets via export/import

### Business Impact Measurements

**Cost Savings (for users):**
- Average user saves $500+ in cloud GPU costs
- Eliminates recurring cloud subscription fees
- Reduces time-to-research for students/researchers

**Ecosystem Value:**
- Growing library of shared checkpoints and presets
- Community contributions reduce maintenance burden
- Hardware partnerships reduce costs for users
- Academic partnerships provide validation

**Project Sustainability:**
- Sufficient funding for core team (grants, sponsorships)
- Active contributor pipeline (new contributors each quarter)
- Low operational costs (< $1000/month)
- Positive ROI for sponsors (visibility, ecosystem value)

**Strategic Goals:**
- Establish miStudio as standard teaching tool
- Recognized as leading edge interpretability platform
- Influence development of edge ML hardware
- Contribute to AI safety research community

---

## 12. Next Steps

### Immediate Next Actions

**Week 1-2: Project Setup**
1. ✅ Create XCC framework directory structure (`0xcc/`)
2. ✅ Initialize git repository and set up version control
3. ⏳ **Create Architecture Decision Record (ADR)** - Next document in workflow
4. ❌ Update CLAUDE.md with Project Standards from ADR
5. ❌ Set up development environment (Python, Node.js, Docker)
6. ❌ Create initial project README and CONTRIBUTING.md

**Week 3-4: Technical Foundation**
1. Set up backend API skeleton (FastAPI + PostgreSQL + Redis)
2. Set up frontend React application (per Mock UI)
3. Implement basic REST API structure
4. Implement WebSocket connection for real-time updates
5. Create Docker Compose setup for local development

**Week 5-8: Core Feature Development**
1. Implement Dataset Management (Feature 1)
2. Implement Model Management (Feature 2)
3. Begin SAE Training System (Feature 3)

### Architecture and Tech Stack Evaluation Needs

**Backend Framework Decision:**
- Evaluate FastAPI (Python) vs. NestJS (TypeScript)
- Consider: ML integration, async support, documentation
- Decision factors: Team expertise, ecosystem, edge performance

**Database Schema Design:**
- Design comprehensive schema for all entities
- Plan for time-series data (training metrics)
- Consider: indexing strategy, query performance

**ML Pipeline Architecture:**
- Design activation extraction pipeline
- Plan training job queue and worker pool
- Consider: memory management, GPU utilization

**Frontend Architecture:**
- Implement component structure per Mock UI
- Design state management strategy
- Plan real-time update handling

**Edge Optimization Strategy:**
- Evaluate TensorRT integration
- Plan memory management for 8GB devices
- Consider: quantization strategies, caching

### Feature Prioritization Approach

**Priority Levels:**
- **P0 (Blocker)**: Core features 1-5 (Dataset, Model, Training, Discovery, Steering)
- **P1 (Important)**: Secondary features 6-12 (Templates, Advanced Viz)
- **P2 (Nice-to-have)**: Future enhancements
- **P3 (Backlog)**: Long-term vision features

**Prioritization Criteria:**
1. **User Value**: Impact on primary use cases
2. **Technical Dependency**: Required by other features
3. **Complexity**: Development effort vs. value
4. **Risk**: Technical or business risk mitigation
5. **Community Interest**: Requested by users

**Development Sequence:**
1. Infrastructure and technical foundation (Weeks 1-4)
2. Core features in dependency order (Weeks 5-16)
3. Secondary features based on feedback (Weeks 17-24)
4. Polish, testing, documentation (Weeks 25-36)

### Resource and Timeline Planning

**Development Timeline:**
- **Months 1-3**: MVP (Core Features 1-5)
- **Months 4-6**: Feature Complete (Secondary Features)
- **Months 7-9**: Polish, optimization, documentation
- **Months 10-12**: Community growth, partnerships, v1.0 release

**Team Allocation:**
- **Backend/ML Engineer**: Dataset, Model, Training, Discovery systems
- **Frontend Engineer**: UI implementation per Mock UI specification
- **DevOps Engineer**: Infrastructure, deployment, CI/CD, edge optimization

**Hardware Needs:**
- Jetson Orin Nano boards for testing (2-3 units)
- Consumer GPU for CI/CD (cloud or local)
- Diverse hardware for compatibility testing (community)

**Documentation Needs:**
- User guide (getting started, tutorials, reference)
- API documentation (OpenAPI/Swagger)
- Developer guide (setup, contribution, architecture)
- Deployment guide (Jetson-specific, Docker, native)

**Community Building:**
- Set up GitHub Discussions for Q&A
- Create Discord/Slack for real-time chat
- Write contribution guidelines
- Plan initial workshops/tutorials

---

## Document Control

**Document Version:** 1.0
**Created By:** AI Dev Tasks Framework (XCC)
**Created Date:** 2025-10-05
**Last Updated:** 2025-10-05
**Status:** Active - Approved for Development

**Review and Approval:**
- [ ] Product Owner Review
- [ ] Technical Lead Review
- [ ] Stakeholder Approval
- [ ] Development Team Acknowledgment

**Related Documents:**
- Architecture Decision Record: `0xcc/adrs/000_PADR|miStudio.md` (To be created)
- UI/UX Reference: `0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
- Technical Specification: `0xcc/project-specs/core/miStudio_Specification.md`
- Framework Guide: `0xcc/instruct/000_README.md`

**Next Document in Workflow:**
- **ADR Creation**: Use `@0xcc/instruct/002_create-adr.md` to create Architecture Decision Record
- **CLAUDE.md Update**: Copy Project Standards from ADR to CLAUDE.md
- **Feature PRD**: Begin with Dataset Management feature using `@0xcc/instruct/003_create-feature-prd.md`

---

*This document follows the AI Dev Tasks Framework structure for project-level Product Requirements Documents. All subsequent feature development should reference this PRD for context and alignment with project goals.*
