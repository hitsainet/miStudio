# miStudio — MechInterp Studio

miStudio is an end-to-end mechanistic interpretability platform. It provides a single, professional, database-backed workbench for the complete SAE research pipeline: ingesting datasets, downloading and quantizing models, training Sparse Autoencoders, extracting and labeling interpretable features, and proving causal claims through feature-based steering interventions.

The goal is to replace the fragmented tooling that characterizes most interpretability research — Jupyter notebooks with hidden state, ad hoc Python scripts, manually tracked experiment parameters — with a reproducible environment where every training run, extraction job, and steering result is recorded with exact hyperparameters and queryable after the fact.

---

## Why miStudio Exists

The mechanistic interpretability field has a tooling problem. The core science — train a Sparse Autoencoder on a model's internal activations, identify which features correspond to human-understandable concepts, verify causal claims by steering those features during inference — is well-established. But the practical workflow to execute it is not. Most researchers assemble a bespoke pipeline from SAELens training scripts, TransformerLens hooks, Neuronpedia uploads, and manual inspection. Results are difficult to reproduce. Iteration is slow. Scaling to larger models or multiple experiments requires significant engineering effort that has nothing to do with the research questions.

miStudio treats this as an infrastructure problem worth solving properly. It provides every stage of the pipeline in one application: download a dataset from HuggingFace, tokenize it, download a model and choose your quantization level based on available VRAM, configure an SAE training job with one of six paper-grounded architectures, watch training metrics in real time, run a feature extraction pass to record what each feature responds to, auto-label features using any OpenAI-compatible LLM endpoint, then build a steering experiment to prove that a specific feature causally influences model behavior.

None of this requires writing code or managing a Python environment. All of it is reproducible, because every configuration is stored in PostgreSQL and every long-running operation emits progress via WebSocket so you can close the browser and return to completed results.

---

## The Research Pipeline

The mechanistic interpretability workflow follows five stages, and miStudio has a dedicated panel for each:

**1. Data & Model Setup** — Download datasets directly from HuggingFace and tokenize them with configurable context length and stride. Download models with optional 4-bit (Q4), 8-bit (Q8), or full-precision loading. miStudio's dynamic layer discovery automatically maps every layer and hook point in the model — residual stream, MLP output, attention output — without requiring architecture-specific configuration. Any transformer model on HuggingFace works without code changes, including LFM2 and other non-standard architectures.

**2. SAE Training** — Configure and launch SAE training jobs through a guided UI. Training runs asynchronously on the GPU via Celery workers — configure a job, close the browser, return to results. Real-time loss curves and sparsity metrics stream via WebSocket while training is running. Six SAE architectures are supported, each grounded in published research:

- **Standard SAELens** (Bricken et al., 2023) — ReLU activation with L1 sparsity penalty
- **Standard Anthropic** (Templeton et al., 2024) — Anthropic's normalization variant
- **JumpReLU** (Rajamanoharan et al., 2024) — Learnable threshold per feature, Gemma Scope approach
- **TopK** (OpenAI, 2024) — Exactly K features active per token, no sparsity penalty needed
- **Skip SAE** — Skip connections for improved reconstruction
- **Transcoder** — Predicts MLP output from MLP input for interpretable MLP decomposition

Training on multiple layers and hook types simultaneously is supported: selecting layers [6, 12] with residual and MLP hooks trains four SAEs in a single job.

**3. Feature Extraction** — Run extraction jobs that scan the dataset through the trained SAE, recording the top-K activating examples per feature. The extraction pass is what gives each feature its "evidence" — the tokens and surrounding context that the feature responds to most strongly. Token filtering controls keep activation examples clean by removing special tokens, fragments, punctuation, and stop words, so the labeling signal is focused on semantically meaningful content.

**4. Auto-Labeling** — With 8,000 to 131,000 features in a wide SAE, manual labeling is not practical. miStudio's labeling system sends each feature's top activation examples to an LLM and asks it to generate a semantic label and category. Four labeling backends are supported: OpenAI's API, any OpenAI-compatible endpoint (Ollama, miLLM, vLLM, or hosted providers), locally loaded HuggingFace models, or manual human entry. Labels record their provenance — which model generated them — enabling quality comparison across methods. Reasoning models that emit `<think>` tags are handled automatically.

**5. Steering** — The definitive proof of a research claim. Select features, configure strengths (positive to amplify, negative to suppress), write a prompt, and generate. miStudio's steering system is built around a grid testing approach rather than a single slider: select up to 20 features and either **compare** them side-by-side across a range of strengths, or **blend** them — applying all selected features simultaneously in one generation pass to discover synergistic effects. A frequency-derived auto-baseline suggests a starting strength per feature so you are not tuning from zero. Steering runs asynchronously against a dedicated GPU partition so the interactive steering UI stays responsive while generation is happening.

---

## Architecture

miStudio runs as a coordinated stack of six services. Each has a specific responsibility and is isolated in its own container:

**FastAPI Backend** — The API orchestrator. It exposes a REST API for all operations (dataset management, model loading, training configuration, extraction, labeling, steering) and a WebSocket layer via Socket.IO for real-time progress updates. The backend coordinates Celery task dispatch, manages database state, and handles HuggingFace authentication for gated model downloads.

**React Frontend** — A single-page application providing the full research UI. Every panel — Datasets, Models, Training, Extractions, Labeling, SAEs, Steering, Templates, Monitor, Settings — is a dedicated view with real-time updates from the WebSocket connection. The frontend does not require a separate development server in production; it is served as static files by Nginx.

**Celery Worker** — Executes all GPU-intensive and long-running tasks off the main API thread: SAE training, activation extraction, feature labeling, model downloads, Neuronpedia pushes. Multiple workers can run in parallel and are GPU-aware — jobs are routed to specific CUDA devices based on available VRAM. The queue is durable: restarting the application does not lose queued or in-progress tasks.

**Celery Beat** — Handles scheduled and periodic tasks: the stuck-job janitors, checkpoint pruning, the GPU watchdog and the steering reconciler. *(System monitoring is **not** among them — it is an asyncio task inside the FastAPI process, `services/background_monitor.py`. MIS-E2E-156.)*

**PostgreSQL** — The source of truth for all experiment state. Every dataset, model, training run, extraction job, SAE, label, steering result, template, and settings value is stored here with exact configuration. This is what makes miStudio's research reproducible by default: you can return to any experiment from any point in the past and know exactly what parameters produced those results.

**Nginx** — Reverse proxy that routes `/api` requests to the backend, `/socket.io` to the WebSocket layer, and serves the frontend static files at the root. The result is a single access point at `http://mistudio.hitsai.local` (or your configured domain) for the entire application.

### The Task Queue Architecture

The Celery/Redis backbone is what makes multi-hour research workflows practical. GPU jobs are dispatched as tasks to a durable queue. Workers pick up tasks, execute them with full GPU access, emit progress via WebSocket, and write results to PostgreSQL. You can queue five training runs with different L1 penalty settings, close the browser, and check results the next morning. The research continues on the backend regardless of whether the UI is open.

For multi-GPU setups, miStudio partitions work: one GPU stays dedicated to the base model for interactive steering, while remaining GPUs handle training and extraction via separate Celery workers. GPU utilization for each device streams live to the Dashboard.

---

## Key Features

**Six SAE Architectures** — Each architecture implements a different sparsity mechanism with paper-grounded default hyperparameters. The UI surfaces the parameters most likely to need tuning (L1 penalty, sparsity target, dead neuron threshold) with guidance on what each controls.

**Dynamic Architecture Support** — Any transformer model that can be loaded by HuggingFace Transformers is supported without code changes. Layer discovery is fully dynamic — there are no hardcoded architecture whitelists.

**External SAE Integration** — Download pre-trained SAEs from any HuggingFace repository (Gemma Scope, EleutherAI's SAE suite, custom uploads) and use them directly for extraction and steering, bypassing training entirely. Multi-select downloads with a grouped preview browser make it practical to select the right layer/width combination.

**Template Ecosystem** — Four template types (extraction, training, labeling prompt, steering prompt) store configurations as JSON for export, import, and sharing. Templates enable scientific reproducibility: a colleague can import your training template and replicate your exact SAE training conditions.

**Neuronpedia Integration** — Export SAEs and their labeled features to a local Neuronpedia instance with direct push (async via Celery with WebSocket progress) or export as a ZIP for manual upload. This is the integration path for contributing SAEs to the public interpretability research commons.

**Settings & Encryption** — Application settings including API keys are stored in PostgreSQL with AES-256-GCM encryption. The Settings panel provides a tabbed interface for configuring LLM endpoints (OpenAI, OpenAI-compatible, local), API keys, labeling defaults, and display preferences — all without editing config files.

**System Monitor** — Real-time dashboard showing per-GPU utilization, memory, temperature, and power; CPU utilization; RAM and swap; disk I/O; and network I/O. All metrics stream via WebSocket every 2 seconds from an asyncio task inside the backend process — so collection stops and resumes with a backend restart, and duplicates if the backend is scaled out.

---

## Hardware Requirements

miStudio requires an NVIDIA GPU. The practical minimum is 8 GB VRAM, which supports 1B–3B parameter models (TinyLlama, Phi-2, Phi-4-mini) at Q4 quantization. For models up to 9B with wide SAEs (16k–131k features), 16–24 GB VRAM (RTX 3090, RTX 4090) is the practical target. Multi-GPU setups with 2×24 GB or more allow dedicated partitioning of inference and training workloads.

VRAM is the binding constraint — system RAM cannot substitute for it. Model weights, SAE weights, and the activation buffer during extraction all must reside on GPU. The Dashboard's real-time VRAM display makes it straightforward to plan what fits before launching a job.

---

## Installation

miStudio ships with two production deployment paths. Both are covered in detail in the manual, including hardware configuration, first-run setup, and domain/hosts file configuration.

The **Docker Compose** path is the recommended starting point for single-machine deployments. Copy `.env.example` to `.env`, then `docker compose up -d` brings up the stack — Postgres, Redis, nginx, the backend, the Celery workers and the frontend — in the correct order. The NVIDIA Container Toolkit is the only prerequisite.

> This previously pointed at `./start-mistudio.sh`, which **cannot work for a fresh clone**: it hardcodes `PROJECT_ROOT=/home/x-sean/app/miStudio` under `set -e`, runs the backend, Celery and frontend *on the host* from a `backend/venv/` it never creates, and serves a different domain than the one documented beside it. That script is a development convenience for one machine; the Compose path is the one that is verified to work. (MIS-E2E-162)

→ [Docker Compose Installation Guide](https://onegaishimas.github.io/miStudio/getting-started/install-guide-compose)

The **Kubernetes** path is designed for shared lab infrastructure. The kustomize base at `k8s/base/` deploys the full stack into a dedicated `mistudio` namespace, with hostPath volumes for persistent data, ingress rules for both local (`hitsai.local`) and tunneled (`hitsai.net`) access, and GPU resource requests. This is the deployment mode used in the hitsai.local research cluster.

→ [Kubernetes Installation Guide](https://onegaishimas.github.io/miStudio/getting-started/install-guide-k8s)

If you are new to miStudio, the [Introduction](https://onegaishimas.github.io/miStudio/getting-started/introduction) explains the architecture, the superposition hypothesis that motivates SAE research, and why a database-backed workbench is preferable to notebook-based workflows.

---

## Documentation

The full user manual is hosted at **[onegaishimas.github.io/miStudio](https://onegaishimas.github.io/miStudio/)**.

| Section | Description |
|---------|-------------|
| [Getting Started](https://onegaishimas.github.io/miStudio/getting-started/introduction) | Introduction, hardware requirements, installation |
| [The Researcher's Journey](https://onegaishimas.github.io/miStudio/core-workflow/researcher-journey) | The five-stage pipeline end-to-end |
| [Data & Model Management](https://onegaishimas.github.io/miStudio/core-workflow/data-model-management) | HuggingFace download, quantization, hook points, tokenization |
| [SAE Training](https://onegaishimas.github.io/miStudio/core-workflow/sae-training) | All six architectures, hyperparameters, training controls |
| [Feature Extraction](https://onegaishimas.github.io/miStudio/core-workflow/feature-extraction) | Extraction configuration, token filtering, context windows |
| [Auto-Labeling](https://onegaishimas.github.io/miStudio/core-workflow/auto-labeling) | Four labeling backends, dual-label system, batch configuration |
| [Model Steering](https://onegaishimas.github.io/miStudio/core-workflow/steering) | Steering modes, strength calibration, grid testing |
| [Template Ecosystem](https://onegaishimas.github.io/miStudio/advanced/templates) | Training, extraction, labeling, and steering templates |
| [External SAEs](https://onegaishimas.github.io/miStudio/advanced/external-saes) | Downloading and using pre-trained SAEs from HuggingFace |
| [Exporting & Neuronpedia](https://onegaishimas.github.io/miStudio/advanced/exporting) | Export formats, direct Neuronpedia push |
| [Multi-GPU Scalability](https://onegaishimas.github.io/miStudio/advanced/multi-gpu) | GPU partitioning, Celery worker configuration |
| [Troubleshooting](https://onegaishimas.github.io/miStudio/troubleshooting) | OOM errors, dead features, common configuration issues |

---

## Relation to miLLM

miStudio and [miLLM](https://github.com/Onegaishimas/miLLM) are complementary tools in the same interpretability research stack.

miStudio is the research environment: it trains SAEs from scratch, extracts and labels features, and runs steering experiments using a model loaded directly into its own GPU partition. miLLM is the inference server: it takes a trained SAE and deploys it for steering and activation monitoring behind an OpenAI-compatible API, making it accessible to any downstream tool.

The most direct integration point is miStudio's auto-labeling system. When configured to use the "OpenAI-Compatible" labeling method, miStudio can call miLLM's `/v1/chat/completions` endpoint to label features using whatever model is loaded in miLLM — entirely on local hardware, without external API calls. SAEs trained in miStudio can be exported and loaded into miLLM for production steering use.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
