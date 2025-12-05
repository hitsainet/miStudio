# miStudio Video Demonstration Script
## Complete Mechanistic Interpretability Workflow

**Target Duration:** 15-20 minutes
**Target Audience:** ML researchers, interpretability practitioners, AI safety researchers
**Goal:** Demonstrate the complete end-to-end workflow from raw data to interpretable features

---

## Pre-Recording Checklist

### System Preparation
- [ ] All services running: `docker ps` (postgres, redis, nginx)
- [ ] Backend running on port 8000: `lsof -i :8000`
- [ ] Frontend running on port 3000: `lsof -i :3000`
- [ ] Celery worker running: `pgrep -f celery`
- [ ] Access application at: http://mistudio.mcslab.io
- [ ] Clear browser cache and localStorage (fresh start)
- [ ] Close unnecessary browser tabs
- [ ] Set browser zoom to 100%
- [ ] Disable browser notifications

### Database Preparation
- [ ] Clean database (optional, for fresh demo):
  ```bash
  # Backup first!
  PGPASSWORD=mistudio psql -h localhost -U mistudio -d mistudio -c "
    TRUNCATE datasets, models, trainings, extraction_jobs, labeling_jobs CASCADE;
  "
  ```
- [ ] OR use existing data and clean up incomplete/failed jobs

### Screen Recording Setup
- [ ] Screen recording software configured (OBS, QuickTime, etc.)
- [ ] Resolution: 1920x1080 (Full HD)
- [ ] Frame rate: 30 FPS minimum
- [ ] Audio: Clear microphone (test audio levels)
- [ ] Recording area: Full browser window (or full screen)
- [ ] Cursor highlighting enabled (if available)
- [ ] Hide desktop clutter, bookmarks bar, OS notifications

### Content Preparation
- [ ] Practice run-through (at least once)
- [ ] Prepare HuggingFace datasets to demonstrate:
  - Small dataset for quick demo: "wikitext-2-raw-v1" (~4MB, 36K samples)
  - Alternative: "openai_humaneval" (very small, code-focused)
- [ ] Prepare model to demonstrate:
  - "gpt2" (124M parameters, quick download)
  - Alternative: "distilgpt2" (smaller, faster)
- [ ] Have HuggingFace access token ready (if using gated models)
- [ ] Have OpenAI API key ready (if demonstrating OpenAI labeling)
- [ ] Prepare browser bookmarks:
  - HuggingFace Datasets: https://huggingface.co/datasets
  - HuggingFace Models: https://huggingface.co/models

### Time Management Notes
- **Fast Path (10-12 minutes):** Use small dataset, skip activation caching, use pattern labeling
- **Standard Path (15-18 minutes):** Medium dataset, show activation caching, use local LLM labeling
- **Comprehensive Path (20-25 minutes):** Show all features, compare labeling methods, demonstrate templates

---

## Script Structure

### Opening (30 seconds)
**[SCENE: Desktop or splash screen]**

**NARRATION:**
> "Welcome to miStudio - a complete platform for mechanistic interpretability research. In this demo, I'll show you the full workflow: from downloading a dataset and model from HuggingFace, to training sparse autoencoders, discovering interpretable features, and understanding what language models really learn."

**VISUAL CUES:**
- Quick montage of key UI panels (1-2 seconds each):
  - Datasets panel with downloads
  - Models panel with architecture viewer
  - Training panel with real-time metrics
  - Feature browser with token highlighting
  - System monitor with GPU metrics

**ACTIONS:**
- Open browser to http://mistudio.mcslab.io

**EDITING NOTES:**
- Consider adding background music (subtle, professional)
- Add text overlay: "miStudio - Mechanistic Interpretability Platform"
- Speed up transitions between panels (2x speed)

---

## Act 1: Data Preparation (3-4 minutes)

### Scene 1.1: Dataset Download (1.5 minutes)

**[SCENE: Navigate to Datasets panel]**

**NARRATION:**
> "Let's start by downloading a dataset. miStudio integrates directly with HuggingFace Hub, making it easy to access thousands of datasets."

**ACTIONS:**
1. Click "Datasets" in left sidebar navigation
2. Show empty datasets list (if clean database)
3. Click "Download Dataset" button

**VISUAL CUES:**
- Zoom in on "Download Dataset" button
- Cursor hover highlights the button

---

**[SCENE: Dataset Download Modal]**

**NARRATION:**
> "I'll download the WikiText-2 dataset - a popular benchmark for language modeling. It's small enough for a quick demo but representative of real-world text."

**ACTIONS:**
1. Type in HuggingFace Repo ID: `wikitext`
2. Show autocomplete suggestions appearing
3. Select `wikitext-2-raw-v1` from suggestions (or type full ID)
4. Leave "Split" as default: `train`
5. Leave "Configuration" empty (use default)
6. Leave "Access Token" empty (public dataset)
7. Click "Download Dataset" button

**VISUAL CUES:**
- Highlight "HuggingFace Repo ID" field
- Show tooltip on hover: "Enter a dataset repository ID from HuggingFace Hub"
- Arrow annotation pointing to "Split" dropdown (explain train/test/validation)

**NARRATION (continued):**
> "For large or private datasets, you can provide a HuggingFace access token. We'll stick with the default 'train' split for now."

**EDITING NOTES:**
- Speed up typing (2x speed) or use jump cut
- Keep real-time download progress (don't speed up)

---

**[SCENE: Real-time Download Progress]**

**NARRATION:**
> "Notice the real-time progress tracking - miStudio uses WebSockets to stream updates as the dataset downloads and processes."

**ACTIONS:**
- Watch progress bar fill (10-30 seconds depending on connection)
- Point out progress percentage: "Downloading... 45%"
- Show "Samples" count incrementing: "~36,000 samples"
- Wait for "Status: completed" and green checkmark

**VISUAL CUES:**
- Zoom in on progress bar
- Highlight incrementing percentage
- Add annotation: "WebSocket real-time updates"

**NARRATION (continued):**
> "And we're done! The dataset has 36,000 training samples - perfect for our demo."

**EDITING NOTES:**
- Consider time-lapse effect for long downloads (30 seconds → 5 seconds)
- Add speedometer icon overlay to show "real-time" concept

---

### Scene 1.2: Dataset Tokenization (1.5 minutes)

**[SCENE: Dataset Card with Tokenization Options]**

**NARRATION:**
> "Before we can train a sparse autoencoder, we need to tokenize the dataset. miStudio supports multiple tokenizers - let's use GPT-2's tokenizer since we'll be working with the GPT-2 model."

**ACTIONS:**
1. Click "Tokenize" button on the dataset card
2. Show tokenization modal

**VISUAL CUES:**
- Highlight "Tokenize" button
- Zoom in on modal

---

**[SCENE: Tokenization Configuration Modal]**

**NARRATION:**
> "We need to specify which tokenizer to use. This should match the model we'll train on, or the dataset will be incompatible."

**ACTIONS:**
1. Click "Tokenizer" dropdown
2. Show available tokenizers (if any models are already downloaded)
   - If no tokenizers: "We'll download GPT-2 first, then come back to tokenize"
   - If GPT-2 tokenizer available: Select "gpt2"
3. Leave other settings as default:
   - Max Sequence Length: 512
   - Stride: 256
4. Click "Start Tokenization" button

**VISUAL CUES:**
- Highlight "Tokenizer" dropdown
- Add annotation: "Must match your target model!"
- Show warning icon if no tokenizers available

**NARRATION (conditional):**
> **[IF NO TOKENIZERS]:** "Ah, we need to download a model first to get its tokenizer. Let's do that now, then we'll come back to tokenization."

**EDITING NOTES:**
- If skipping tokenization for now, add jump cut with text overlay: "Coming back to tokenization after model download"

---

**[SCENE: Tokenization Progress]**

**ACTIONS:**
- Watch real-time tokenization progress (if proceeding now)
- Show incrementing token count
- Wait for completion

**NARRATION:**
> "Tokenization processes the raw text into token IDs that the model understands. This typically takes a few seconds to a few minutes depending on dataset size."

**VISUAL CUES:**
- Zoom in on progress bar
- Highlight "Tokens Processed" counter

**EDITING NOTES:**
- Time-lapse effect (30 seconds → 5 seconds)

---

### Scene 1.3: Model Download (1.5 minutes)

**[SCENE: Navigate to Models Panel]**

**NARRATION:**
> "Now let's download our model. We'll use GPT-2, a classic transformer model with 124 million parameters."

**ACTIONS:**
1. Click "Models" in left sidebar navigation
2. Click "Download Model" button

**VISUAL CUES:**
- Smooth transition from Datasets to Models panel
- Highlight "Download Model" button

---

**[SCENE: Model Download Modal]**

**NARRATION:**
> "miStudio supports downloading any transformer model from HuggingFace. You can also apply quantization to reduce memory usage - essential for edge devices like NVIDIA Jetson."

**ACTIONS:**
1. Type in Model Repo ID: `gpt2`
2. Show autocomplete suggestions
3. Leave "Quantization" as "None" (or select FP16 for faster demo)
4. Leave "Trust Remote Code" unchecked
5. Leave "Access Token" empty
6. Click "Download Model" button

**VISUAL CUES:**
- Highlight "Quantization" dropdown
- Add tooltip annotation: "FP16/INT8/INT4 for edge devices"
- Show "Trust Remote Code" warning icon on hover

**NARRATION (continued):**
> "For this demo, we'll download the full precision model. In production, you might use FP16 or INT8 quantization to fit larger models on constrained hardware."

**EDITING NOTES:**
- Speed up typing
- Highlight quantization dropdown briefly (2 seconds)

---

**[SCENE: Model Download Progress]**

**ACTIONS:**
- Watch real-time download progress
- Show "Downloading model files... 45%"
- Show estimated time remaining
- Wait for "Status: completed"

**NARRATION:**
> "GPT-2 is about 500MB, so this takes a minute or two. You'll see real-time progress with estimated time remaining."

**VISUAL CUES:**
- Zoom in on progress bar
- Highlight ETA (Estimated Time Remaining)
- Add annotation: "~500MB download"

**EDITING NOTES:**
- Time-lapse effect (2 minutes → 10 seconds)
- Add download speed overlay (optional)

---

**[SCENE: Model Card with Architecture Info]**

**ACTIONS:**
1. Click on the downloaded model card
2. Show model architecture details:
   - 12 layers
   - 768 hidden dimensions
   - 50,257 vocabulary size
   - 124M parameters

**NARRATION:**
> "Once downloaded, we can view the model architecture. GPT-2 has 12 transformer layers, each with 768-dimensional hidden states. We'll train sparse autoencoders on several of these layers."

**VISUAL CUES:**
- Zoom in on architecture section
- Highlight key metrics: layers, hidden_dim, vocab_size
- Add annotations to explain each metric

**EDITING NOTES:**
- Hold on architecture view for 3-5 seconds
- Add subtle highlight animation on key metrics

---

**[SCENE: Return to Tokenization (if skipped earlier)]**

**NARRATION:**
> "Now that we have our model and its tokenizer, let's go back and tokenize our dataset."

**ACTIONS:**
1. Navigate back to Datasets panel
2. Click "Tokenize" on wikitext-2-raw-v1 card
3. Select "gpt2" tokenizer
4. Start tokenization
5. Watch progress (time-lapse)

**VISUAL CUES:**
- Quick transition back to Datasets panel
- Highlight "gpt2" tokenizer selection

**EDITING NOTES:**
- Speed up this section (already explained tokenization earlier)
- Use 4x time-lapse for tokenization progress

---

## Act 2: SAE Training (4-5 minutes)

### Scene 2.1: Training Configuration - Basic Settings (2 minutes)

**[SCENE: Navigate to Training Panel]**

**NARRATION:**
> "With our data and model ready, it's time to train sparse autoencoders. SAEs learn to decompose model activations into interpretable features - the core technique in mechanistic interpretability."

**ACTIONS:**
1. Click "Training" in left sidebar navigation
2. Show empty training jobs list (if clean database)
3. Scroll down to "New Training Configuration" section

**VISUAL CUES:**
- Smooth transition to Training panel
- Highlight "New Training Configuration" header
- Add text overlay: "Sparse Autoencoder Training"

---

**[SCENE: Training Configuration Form - Dataset Selection]**

**NARRATION:**
> "Let's configure our training. First, we select the tokenized dataset."

**ACTIONS:**
1. Click "Dataset" dropdown
2. Select "wikitext-2-raw-v1 (gpt2 tokenizer)"
3. Show checkmark and tokenizer info below dropdown

**VISUAL CUES:**
- Zoom in on Dataset dropdown
- Highlight tokenizer compatibility indicator
- Add annotation: "Must use tokenized dataset!"

**NARRATION (continued):**
> "miStudio automatically shows which tokenizer was used, and will warn if there's a mismatch with the model."

---

**[SCENE: Training Configuration Form - Model Selection]**

**NARRATION:**
> "Next, select our model. Notice how miStudio validates tokenizer compatibility."

**ACTIONS:**
1. Click "Model" dropdown
2. Select "gpt2"
3. Show vocabulary size compatibility check:
   - Dataset vocab: 50,257
   - Model vocab: 50,257
   - ✓ Compatible

**VISUAL CUES:**
- Zoom in on Model dropdown
- Highlight vocabulary compatibility indicator (green checkmark)
- Add annotation: "Automatic compatibility validation"

**NARRATION (continued):**
> "Perfect - both use the same 50,257 token vocabulary. If these didn't match, miStudio would show a warning."

---

**[SCENE: Training Configuration Form - Architecture & Layers]**

**NARRATION:**
> "Now we choose the SAE architecture and which layers to train on. We'll use the standard SAE architecture and train on three layers: early, middle, and late."

**ACTIONS:**
1. Click "SAE Architecture" dropdown
2. Show options: Standard, Standard with Skip Connection, Transcoder
3. Select "Standard"
4. Show layer selection grid (12 layers numbered 0-11)
5. Click layers: 0, 6, 11
6. Show selected layers highlighted in emerald green

**VISUAL CUES:**
- Zoom in on Architecture dropdown
- Add tooltip on "Skip Connection" and "Transcoder" (hover briefly)
- Zoom out to show full layer grid
- Highlight selected layers (0, 6, 11) with emerald glow
- Add annotation: "Multi-layer training in parallel"

**NARRATION (continued):**
> "miStudio supports multi-layer training - we can train SAEs on multiple layers simultaneously. Layer 0 captures low-level features, layer 6 captures mid-level features, and layer 11 captures high-level features near the model's output."

**EDITING NOTES:**
- Hold on layer grid for 3-4 seconds to let viewers understand
- Add subtle animation highlighting the three selected layers

---

**[SCENE: Training Configuration Form - Dimensions & Sparsity]**

**NARRATION:**
> "Next, we configure the autoencoder dimensions and sparsity. The hidden dimension is automatically set to match the model's layer dimension - 768 for GPT-2."

**ACTIONS:**
1. Show "Hidden Dimension" field: 768 (auto-filled, disabled)
2. Show "Latent Dimension" field with multiplier:
   - Default: 8192 (768 × 10.67)
3. Click "Auto-calculate L1 Alpha" button
4. Enter "Target L0 Sparsity": 50 (out of 8192 latents)
5. Show calculated L1 Alpha: ~0.015

**VISUAL CUES:**
- Highlight "Hidden Dimension" (auto-filled, locked icon)
- Zoom in on "Latent Dimension" field and multiplier indicator
- Highlight "Auto-calculate L1 Alpha" button with glow
- Add annotation: "L1 alpha controls sparsity penalty"

**NARRATION (continued):**
> "The latent dimension is typically 4-16 times larger than the hidden dimension - we're using about 11x here. miStudio can auto-calculate the L1 sparsity penalty to achieve a target sparsity level. Let's aim for 50 active features out of 8192 total."

**EDITING NOTES:**
- Pause on "Auto-calculate" button (2 seconds)
- Show before/after L1 Alpha value change
- Add subtle math equation overlay: "L1 α ≈ 0.015 for L0 = 50"

---

**[SCENE: Training Configuration Form - Training Settings]**

**NARRATION:**
> "Finally, we set the training hyperparameters. For this demo, we'll use a short training run."

**ACTIONS:**
1. Show "Learning Rate": 3e-4 (default)
2. Show "Batch Size": 512 (default)
3. Change "Training Steps": 1000 (short demo, default is 10,000)
4. Show "Warmup Steps": 100 (10% of total steps)
5. Leave other settings as default

**VISUAL CUES:**
- Highlight "Training Steps" field
- Add annotation: "Shortened for demo (normally 10K+)"
- Show estimated training time: "~2-3 minutes"

**NARRATION (continued):**
> "Normally you'd train for 10,000 to 100,000 steps, but for this demo we'll use just 1,000 steps. That's enough to see the system in action."

---

### Scene 2.2: Memory Estimation & Advanced Settings (1 minute)

**[SCENE: Memory Estimation Panel]**

**NARRATION:**
> "Before we start training, let's check the memory requirements. miStudio estimates GPU memory usage per layer and total."

**ACTIONS:**
1. Scroll down to "Memory Estimation" section
2. Show per-layer estimates:
   - Layer 0: ~1.2 GB
   - Layer 6: ~1.2 GB
   - Layer 11: ~1.2 GB
3. Show total estimate: ~3.6 GB
4. Show available GPU memory: e.g., "Available: 8 GB / 24 GB"

**VISUAL CUES:**
- Zoom in on memory estimation panel
- Highlight "Total Estimated Memory" with emerald glow
- Add annotation: "Fits comfortably in GPU memory"
- Show GPU icon with checkmark (sufficient memory)

**NARRATION (continued):**
> "Training three layers in parallel will use about 3.6 GB of GPU memory - well within our budget. If memory is tight, you can train layers separately or reduce the batch size."

**EDITING NOTES:**
- Hold on memory panel for 2-3 seconds
- Add subtle warning icon animation if memory is close to limit

---

**[SCENE: Advanced Settings (Brief Overview)]**

**NARRATION:**
> "miStudio also offers advanced settings like activation normalization and dead neuron resampling. We'll stick with the defaults for now."

**ACTIONS:**
1. Click "Advanced Settings" collapsible section (expand briefly)
2. Show options:
   - Activation Normalization: Enabled
   - Dead Neuron Resampling: Enabled
   - Resampling Threshold: 1e-5
   - Gradient Clipping: 1.0
3. Collapse section (don't change anything)

**VISUAL CUES:**
- Quick pan over advanced settings
- Add annotation: "Advanced: Fine-tune training dynamics"
- Don't spend too long here (30 seconds max)

**EDITING NOTES:**
- Speed through this section (1.5x speed)
- Optional: Skip entirely for faster demo

---

### Scene 2.3: Starting Training & Real-time Monitoring (2 minutes)

**[SCENE: Start Training]**

**NARRATION:**
> "Everything looks good. Let's start training!"

**ACTIONS:**
1. Scroll to top of training form
2. Click "Start Training" button
3. Show training job card appear at top of "Training Jobs" list
4. Show status change: "Queued" → "Running"

**VISUAL CUES:**
- Highlight "Start Training" button with glow
- Add animation: Form → Training card (smooth transition)
- Zoom in on new training card

**NARRATION (continued):**
> "The training job is now queued and will start in a few seconds. Notice how the form persists after starting - you can easily iterate on configurations."

---

**[SCENE: Real-time Training Metrics]**

**NARRATION:**
> "Now we get real-time progress updates via WebSocket. Let's watch the metrics evolve."

**ACTIONS:**
1. Show training card with live metrics:
   - Progress bar: "Step 1 / 1000"
   - Loss (Total): Starting at ~150, decreasing
   - Loss (Reconstruction): ~145
   - Loss (Sparsity): ~5
   - L0 Sparsity: Starting at ~80, moving toward target of 50
   - Dead Neurons: Starting at 0, may increase slightly
   - Learning Rate: Ramping up (warmup phase)
   - Gradient Norm: ~1.5

2. Let training run for 30-60 seconds (show ~100-200 steps)

3. Point out key observations:
   - Total loss decreasing steadily
   - L0 sparsity converging toward target (50)
   - Learning rate warming up
   - Dead neuron count stable

**VISUAL CUES:**
- Zoom in on metrics section of training card
- Highlight loss curve (line chart animation)
- Add annotations:
  - "Reconstruction loss: How well we rebuild activations"
  - "Sparsity loss: Penalty for using too many features"
  - "L0: Number of active features (target: 50)"
- Show GPU status in top-right corner (compact view)

**NARRATION:**
> "The reconstruction loss shows how well the autoencoder rebuilds the original activations. The sparsity loss encourages the model to use fewer features. Notice the L0 sparsity converging toward our target of 50 active features. During the warmup phase, the learning rate gradually increases to stabilize training."

**EDITING NOTES:**
- Time-lapse training (60 seconds → 15 seconds)
- Keep metric updates smooth (don't jump cut)
- Add speedometer overlay: "Real-time updates"
- Add background music (subtle, rhythmic)

---

**[SCENE: Training Completion]**

**NARRATION:**
> "After a few minutes, training completes successfully."

**ACTIONS:**
1. Fast-forward to final steps (step 950+)
2. Show final metrics:
   - Total Loss: ~85 (reduced from ~150)
   - L0 Sparsity: ~52 (close to target of 50)
   - Dead Neurons: ~5-10 (acceptable)
3. Show status change: "Running" → "Completed" (green checkmark)
4. Show "Extract Features" button appear on training card

**VISUAL CUES:**
- Zoom in on progress bar reaching 100%
- Add green checkmark animation
- Highlight "Extract Features" button with glow
- Add text overlay: "Training Complete! ✓"

**NARRATION (continued):**
> "Training finished! Final loss is down to about 85 from 150, and we achieved our target sparsity of 50 active features. The SAE learned to decompose activations into sparse, interpretable features. Now it's time to discover what those features are."

**EDITING NOTES:**
- Celebratory sound effect (subtle chime)
- Add green success animation (checkmark, confetti)

---

## Act 3: Feature Discovery (4-5 minutes)

### Scene 3.1: Feature Extraction Configuration (1 minute)

**[SCENE: Start Feature Extraction]**

**NARRATION:**
> "To discover what features the SAE learned, we need to run feature extraction. This analyzes the trained SAE on evaluation data and finds the tokens that maximally activate each feature."

**ACTIONS:**
1. Click "Extract Features" button on completed training card
2. Show feature extraction configuration modal

**VISUAL CUES:**
- Highlight "Extract Features" button
- Smooth transition to modal
- Add text overlay: "Feature Extraction & Labeling"

---

**[SCENE: Feature Extraction Configuration Modal]**

**NARRATION:**
> "We'll configure the extraction to use 10,000 evaluation samples and capture the top 100 activating examples per feature. We can also choose a labeling method - let's use pattern matching for speed."

**ACTIONS:**
1. Show "Evaluation Samples" field: 10,000 (default)
2. Show "Top-K Examples per Feature" field: 100 (default)
3. Click "Labeling Method" dropdown
4. Show options:
   - Pattern Matching (Fast, rule-based)
   - Local LLM (Phi-3, Llama, Qwen)
   - OpenAI API (GPT-4o-mini, GPT-4)
5. Select "Pattern Matching"
6. Show explanation tooltip: "Fast rule-based labeling using 8 predefined patterns"
7. Click "Start Extraction" button

**VISUAL CUES:**
- Highlight "Labeling Method" dropdown
- Add annotations for each method:
  - Pattern: "Fast (seconds)"
  - Local LLM: "High quality, zero cost (minutes)"
  - OpenAI: "Fast, high quality (costs money)"
- Zoom in on "Start Extraction" button

**NARRATION (continued):**
> "Pattern matching is the fastest option - it uses predefined rules to identify common feature types like punctuation, numbers, and common words. For higher quality labels, you can use a local LLM like Phi-3, or OpenAI's GPT-4. We'll re-label with a better method shortly."

**EDITING NOTES:**
- Pause on labeling method dropdown (2-3 seconds)
- Show tooltip/comparison table (optional overlay)

---

### Scene 3.2: Feature Extraction Progress (30 seconds)

**[SCENE: Extraction Progress]**

**NARRATION:**
> "Extraction is now running. It processes batches of data through the SAE, collects activations, and generates initial labels."

**ACTIONS:**
1. Show extraction job card appear in "Feature Extractions" list
2. Show real-time progress:
   - Progress bar: "Processing... 45%"
   - Features Found: ~8192 (total latent dimensions)
   - Estimated Time Remaining: ~2 minutes
3. Wait for completion (time-lapse)
4. Show status: "Completed" (green checkmark)
5. Show "View Features" button appear

**VISUAL CUES:**
- Zoom in on extraction progress card
- Highlight features found counter (incrementing)
- Add annotation: "Analyzing 10K samples"
- Show GPU utilization in corner

**NARRATION (continued):**
> "Extraction analyzes 10,000 samples from the dataset, tracking which tokens activate each feature most strongly. This takes a couple of minutes."

**EDITING NOTES:**
- Time-lapse (2 minutes → 10 seconds)
- Keep progress bar smooth
- Add processing animation overlay (optional)

---

### Scene 3.3: Feature Browser Overview (1.5 minutes)

**[SCENE: Navigate to Feature Browser]**

**NARRATION:**
> "Let's explore the discovered features. miStudio provides a powerful feature browser with search, sort, and filter capabilities."

**ACTIONS:**
1. Click "View Features" button (or navigate to Features panel)
2. Show feature browser interface:
   - Search bar at top
   - Sort dropdown (Activation Frequency / Interpretability / Feature ID)
   - Filter: "Show Only Favorites" toggle
   - Statistics summary:
     - Total Features: 8,192
     - Interpretable: ~6,500 (79%)
     - Avg Activation Rate: 0.8%
   - Feature table with 8 columns:
     - ID (neuron index)
     - Label (human-readable name)
     - Category (semantic grouping)
     - Description (detailed explanation)
     - Example Context (token highlighting)
     - Activation Frequency (%)
     - Interpretability Score (%)
     - Of Interest (favorite star)
3. Show pagination controls:
   - "Showing 1-50 of 8,192"
   - "Go to Feature" and "Go to Page" input fields

**VISUAL CUES:**
- Zoom out to show full feature browser layout
- Highlight key UI elements:
  - Search bar (top)
  - Statistics (top-right)
  - Feature table (center)
  - Pagination (bottom)
- Add annotations pointing to each section

**NARRATION:**
> "The feature browser shows all 8,192 features in a searchable, sortable table. We can see that about 79% are considered interpretable - meaning they have clear, understandable behavior. Each feature has a semantic label, category, description, and example tokens with activation highlighting."

**EDITING NOTES:**
- Hold on full browser view for 3-4 seconds
- Pan over key elements slowly
- Add subtle highlight animations

---

### Scene 3.4: Exploring Individual Features (2 minutes)

**[SCENE: Browse Features]**

**NARRATION:**
> "Let's browse some interesting features. I'll search for punctuation-related features first."

**ACTIONS:**
1. Click in search bar
2. Type: "punctuation"
3. Show filtered results:
   - Feature 1547: "Punctuation - Period"
   - Feature 2891: "Punctuation - Comma"
   - Feature 4201: "Punctuation - Question Mark"
4. Click on "Punctuation - Period" (Feature 1547)

**VISUAL CUES:**
- Zoom in on search bar
- Speed up typing (2x)
- Highlight filtered results
- Add annotation: "Instant search filtering"

---

**[SCENE: Feature Detail Modal - Examples Tab]**

**NARRATION:**
> "This opens the feature detail modal. The Examples tab shows the top 100 token sequences that most strongly activated this feature."

**ACTIONS:**
1. Show feature detail modal:
   - Header: "Feature 1547: Punctuation - Period"
   - Four tabs: Examples | Logit Lens | Correlations | Ablation
   - "Examples" tab is active (default)
2. Show max-activating examples list:
   - Example 1: "...end of sentence<PERIOD_TOKEN>"
   - Tokens highlighted with color intensity (activation strength)
   - Period token has bright emerald highlight (max activation)
   - Surrounding tokens have faint highlights
3. Scroll through 3-5 examples
4. Point out activation intensity colors:
   - Dark emerald = highest activation
   - Light emerald = moderate activation
   - No highlight = low/zero activation

**VISUAL CUES:**
- Zoom in on modal header
- Highlight token highlighting mechanism
- Add color scale legend:
  - Dark emerald: "Max Activation"
  - Light emerald: "Moderate Activation"
  - Gray: "Low Activation"
- Add annotation: "Token activation intensity"

**NARRATION:**
> "Notice how the period token is highlighted in bright emerald - this is where the feature activates most strongly. Surrounding tokens have lighter highlights, showing moderate activation. This feature clearly responds to periods at the end of sentences."

**EDITING NOTES:**
- Scroll slowly through examples (2-3 seconds per example)
- Add subtle glow effect on max-activating tokens
- Zoom in on color intensity differences

---

**[SCENE: Feature Detail Modal - Logit Lens Tab]**

**NARRATION:**
> "The Logit Lens tab shows what this feature predicts. If we project this feature's direction back through the model's output layer, what tokens does it predict?"

**ACTIONS:**
1. Click "Logit Lens" tab
2. Show predicted token distribution:
   - Top predicted token: "." (period) - 95% probability
   - Second: "!" (exclamation) - 3%
   - Third: "?" (question mark) - 1.5%
3. Show bar chart visualization of token probabilities

**VISUAL CUES:**
- Smooth tab transition animation
- Zoom in on bar chart
- Highlight top predicted token (period)
- Add annotation: "Feature → Token Predictions"

**NARRATION (continued):**
> "As expected, this feature strongly predicts the period token with 95% probability. It also weakly predicts other sentence-ending punctuation. This confirms our interpretation - the feature represents 'end of sentence' concept."

**EDITING NOTES:**
- Hold on logit lens for 2-3 seconds
- Add subtle animation on bar chart (bars growing)

---

**[SCENE: Feature Detail Modal - Correlations Tab]**

**NARRATION:**
> "The Correlations tab shows other features that tend to activate together with this one."

**ACTIONS:**
1. Click "Correlations" tab
2. Show correlated features list:
   - Feature 2891 (Comma): 0.65 correlation
   - Feature 1203 (Capitalization - Start of Sentence): 0.58 correlation
   - Feature 4201 (Question Mark): 0.42 correlation
3. Click on "Feature 2891" link to open that feature in a new modal

**VISUAL CUES:**
- Tab transition animation
- Highlight correlation scores
- Add annotation: "Co-occurring features"
- Show nested modal opening (Feature 2891)

**NARRATION (continued):**
> "This period feature correlates with comma and start-of-sentence features - makes sense, as these are all related to sentence structure. We can click any correlated feature to explore it further."

**EDITING NOTES:**
- Hold on correlations for 2 seconds
- Don't spend too long on nested modal (close after 2 seconds)

---

**[SCENE: Feature Detail Modal - Ablation Tab]**

**NARRATION:**
> "Finally, the Ablation tab shows what happens when we remove this feature from the model."

**ACTIONS:**
1. Click "Ablation" tab
2. Show ablation impact metrics:
   - Perplexity increase: +2.3% (removing this feature hurts performance)
   - Most affected tokens: ".", "!", "?"
   - Example completions with/without feature:
     - With feature: "The cat sat on the mat."
     - Without feature: "The cat sat on the mat" (no period)

**VISUAL CUES:**
- Tab transition animation
- Highlight perplexity delta (+2.3%)
- Show side-by-side comparison of completions
- Add annotation: "Impact of removing feature"

**NARRATION (continued):**
> "Removing this feature increases perplexity by 2.3% - the model struggles more with predicting periods. The ablation analysis confirms this feature is functionally important for punctuation."

**EDITING NOTES:**
- Hold on ablation tab for 3 seconds
- Add visual comparison (split-screen effect for completions)

---

**[SCENE: Close Modal and Browse More Features]**

**NARRATION:**
> "Let's look at a few more features quickly to show the diversity of what the SAE learned."

**ACTIONS:**
1. Close feature modal
2. Clear search bar
3. Change sort to "Interpretability Score" (descending)
4. Scroll through top features:
   - Feature 4521: "Names - Person (First Name)" - 98% interpretability
   - Feature 1892: "Numbers - Digits" - 96% interpretability
   - Feature 3401: "Pronouns - Third Person Singular (he/she)" - 94% interpretability
5. Click on "Names - Person" feature briefly
6. Show examples: "John", "Mary", "Alice", "Bob" highly highlighted
7. Close modal

**VISUAL CUES:**
- Smooth scrolling through table (2x speed)
- Highlight interpretability scores column
- Quick pan through "Names" feature examples
- Add text overlays for each feature category:
  - "Punctuation"
  - "Numbers"
  - "Names"
  - "Pronouns"
  - "Syntax"

**NARRATION:**
> "The high interpretability features are incredibly diverse - we see features for names, numbers, pronouns, and various grammatical structures. The SAE successfully decomposed the model into hundreds of interpretable concepts."

**EDITING NOTES:**
- Montage-style editing (quick cuts between features)
- Add upbeat background music
- Keep this section fast-paced (30 seconds total)

---

## Act 4: Feature Interpretation & Labeling (2-3 minutes)

### Scene 4.1: Starting Labeling Job (1 minute)

**[SCENE: Navigate to Labeling Panel]**

**NARRATION:**
> "The pattern matching labels we used earlier are helpful, but we can get much better semantic labels using a local language model. Let's re-label our features with Phi-3."

**ACTIONS:**
1. Click "Labeling" in left sidebar navigation
2. Show empty labeling jobs list (if clean database)
3. Click "Start Labeling" button

**VISUAL CUES:**
- Smooth transition to Labeling panel
- Highlight "Start Labeling" button
- Add text overlay: "Semantic Feature Labeling"

---

**[SCENE: Labeling Configuration Modal]**

**NARRATION:**
> "We'll select our completed extraction job and configure local LLM labeling with Phi-3-mini."

**ACTIONS:**
1. Click "Extraction Job" dropdown
2. Select the extraction we just completed: "Training #1 - Layer 0, 6, 11 - Extraction"
3. Click "Labeling Method" dropdown
4. Select "Local LLM"
5. Show additional options appear:
   - Model: Phi-3-mini-4k-instruct (default, recommended)
   - Alternatives: Llama 3.2, Qwen 2.5
6. Leave "Model" as "Phi-3-mini-4k-instruct"
7. Leave "Feature Filter" empty (label all features)
8. Show estimated time: "~10-15 minutes for 8,192 features"
9. Click "Start Labeling" button

**VISUAL CUES:**
- Zoom in on "Labeling Method" dropdown
- Highlight "Local LLM" option
- Add annotations:
  - "Zero cost, high quality"
  - "Runs on your hardware"
  - "No API keys needed"
- Show comparison tooltip:
  - Pattern: ~1 sec (fast, low quality)
  - Local LLM: ~10 min (moderate speed, high quality)
  - OpenAI: ~2 min (fast, high quality, $$$)

**NARRATION:**
> "Local LLM labeling uses Phi-3, a small but capable language model running on your own hardware. It's much higher quality than pattern matching and costs nothing - just takes a bit longer. For even faster results, you could use OpenAI's API, but that costs money."

**EDITING NOTES:**
- Pause on model selection (2 seconds)
- Show comparison overlay (speed vs quality vs cost)

---

### Scene 4.2: Labeling Progress & Completion (1 minute)

**[SCENE: Labeling Progress]**

**NARRATION:**
> "Labeling is now running. The local LLM analyzes each feature's activating examples and generates semantic labels."

**ACTIONS:**
1. Show labeling job card appear in list
2. Show real-time progress:
   - Progress bar: "Labeling features... 1,245 / 8,192"
   - Percentage: 15%
   - Current rate: ~14 features/second
   - Estimated Time Remaining: ~8 minutes
3. Wait for completion (time-lapse)
4. Show status: "Completed" (green checkmark)

**VISUAL CUES:**
- Zoom in on labeling progress card
- Highlight features processed counter (incrementing rapidly)
- Add annotation: "Phi-3 analyzing features"
- Show GPU/CPU utilization in corner (Phi-3 is using resources)

**NARRATION (continued):**
> "The local LLM processes about 10-15 features per second. This is much slower than pattern matching but produces significantly better semantic labels. Let's fast-forward..."

**EDITING NOTES:**
- Time-lapse (10 minutes → 15 seconds)
- Keep progress bar smooth
- Add loading animation overlay (brain icon processing)

---

**[SCENE: Labeling Complete]**

**NARRATION:**
> "Labeling complete! Now all our features have high-quality semantic labels."

**ACTIONS:**
1. Show final labeling statistics:
   - Total Features Labeled: 8,192
   - High Interpretability (>80%): 6,845 (84%)
   - Medium Interpretability (50-80%): 1,120 (14%)
   - Low Interpretability (<50%): 227 (3%)
2. Click "View Features" button to return to feature browser

**VISUAL CUES:**
- Zoom in on completion stats
- Highlight interpretability breakdown (pie chart or bar chart)
- Add green checkmark animation
- Add text overlay: "Labeling Complete! ✓"

**NARRATION (continued):**
> "84% of features are now highly interpretable with detailed semantic labels. Only 3% remain difficult to interpret - these might be polysemantic features that respond to multiple unrelated concepts."

**EDITING NOTES:**
- Hold on statistics for 2-3 seconds
- Add celebratory animation (subtle)

---

### Scene 4.3: Comparing Labels (1 minute)

**[SCENE: Feature Browser with Updated Labels]**

**NARRATION:**
> "Let's see how the labels improved. I'll search for a feature we looked at earlier."

**ACTIONS:**
1. Return to feature browser
2. Search for Feature 1547 (the period feature)
3. Show updated label:
   - Old (Pattern): "Punctuation - Period"
   - New (Phi-3): "Sentence-ending punctuation mark (declarative statements)"
4. Show updated description:
   - Old: "Detects period characters"
   - New: "Activates on periods that conclude declarative sentences, particularly in formal or narrative text. Distinguishes sentence-ending periods from abbreviations."
5. Click on feature to open detailed modal
6. Show updated category: "Syntax - Sentence Boundaries"

**VISUAL CUES:**
- Highlight label differences (before/after comparison)
- Add annotation: "More nuanced interpretation"
- Show side-by-side comparison overlay (optional)

**NARRATION:**
> "The new label is much more descriptive - it distinguishes sentence-ending periods from abbreviations and provides context about when this feature activates. This level of detail makes it much easier to understand what the model learned."

**EDITING NOTES:**
- Hold on comparison for 2-3 seconds
- Add subtle before/after animation

---

## Act 5: Workflow Optimization (2-3 minutes)

### Scene 5.1: Saving Templates (1 minute)

**[SCENE: Training Templates]**

**NARRATION:**
> "To speed up future experiments, we can save successful configurations as templates. Let's save our training configuration."

**ACTIONS:**
1. Navigate to "Templates" panel in sidebar
2. Show three sub-tabs: Extraction Templates | Training Templates | Labeling Prompt Templates
3. Click "Training Templates" tab
4. Click "New Template" button
5. Fill in template form:
   - Name: "GPT-2 Multi-Layer Standard SAE"
   - Description: "Standard SAE trained on layers 0, 6, 11 with sparsity target of 50"
   - Architecture: Standard
   - Layers: 0, 6, 11
   - Latent Dimension Multiplier: 10.67
   - L1 Alpha: 0.015
   - Target L0: 50
   - Training Steps: 10,000 (not 1,000 - save for real training)
6. Click "Save Template" button
7. Show template appear in list

**VISUAL CUES:**
- Smooth navigation to Templates panel
- Highlight three template types (tabs)
- Zoom in on template form
- Add annotation: "Reusable configurations"

**NARRATION:**
> "Templates let you save and reuse configurations. This is especially useful when you're iterating on experiments - just load a template and tweak one parameter at a time."

**EDITING NOTES:**
- Speed up form filling (2x speed)
- Show template card appearing with smooth animation

---

**[SCENE: Extraction & Labeling Templates]**

**NARRATION:**
> "You can also save extraction and labeling prompt templates. miStudio even tracks usage statistics to help you identify your most successful configurations."

**ACTIONS:**
1. Click "Extraction Templates" tab
2. Show a few pre-existing templates (if any):
   - "Standard Extraction - 10K samples"
   - "Large Extraction - 100K samples"
3. Click "Labeling Prompt Templates" tab
4. Show pre-existing prompt templates:
   - "Standard Feature Labeling Prompt"
   - "Custom Prompt: Focus on Semantics"
5. Point out usage statistics: "Used 3 times"

**VISUAL CUES:**
- Quick tab transitions
- Highlight template cards
- Add annotation: "Track what works best"

**NARRATION (continued):**
> "The usage statistics help you identify which templates produce the best results. You can export templates as JSON files to share with colleagues or backup your configurations."

**EDITING NOTES:**
- Keep this section fast (30 seconds)
- Quick pan through template types

---

### Scene 5.2: System Monitoring (1 minute)

**[SCENE: Compact GPU Status (Navbar)]**

**NARRATION:**
> "Throughout this demo, you may have noticed the compact GPU status indicator in the top-right corner. This provides always-visible monitoring of your system resources."

**ACTIONS:**
1. Pan to top-right corner of screen
2. Show compact GPU status:
   - GPU 0: 45% utilization | 65°C | 4.2 / 8.0 GB
3. Hover over status (show tooltip with more details)

**VISUAL CUES:**
- Zoom in on compact GPU status
- Highlight utilization percentage, temperature, and memory
- Add annotation: "Always-visible monitoring"

---

**[SCENE: Full System Monitor Panel]**

**NARRATION:**
> "For detailed monitoring, we can open the full System Monitor panel."

**ACTIONS:**
1. Click "System Monitor" in left sidebar navigation
2. Show full system monitoring dashboard:
   - GPU Card(s): Individual cards for each GPU
   - Utilization & Temperature Chart: 1-hour time series
   - Memory Usage Chart: 1-hour time series
3. Point out key features:
   - Real-time updates (WebSocket, 2-second intervals)
   - GPU utilization overlay with temperature
   - Dual Y-axis chart (percentage vs temperature)
   - Memory usage (used/total)
   - Power consumption

**VISUAL CUES:**
- Zoom out to show full dashboard
- Highlight time series charts
- Add annotations:
  - "Real-time WebSocket updates"
  - "1-hour historical view"
  - "Multi-GPU support"

**NARRATION:**
> "The system monitor provides real-time GPU utilization, temperature, memory usage, and power consumption. The charts show the last hour of history, so you can see how your system performed during training. This is especially useful on edge devices like NVIDIA Jetson where resources are limited."

**EDITING NOTES:**
- Hold on dashboard for 3-4 seconds
- Add subtle chart animation (line drawing)

---

### Scene 5.3: Cached Activations (Optional, 1 minute)

**[SCENE: Activation Extraction from Models]**

**NARRATION:**
> "One powerful feature we didn't use earlier is cached activation extraction. For large-scale experiments, you can pre-extract activations from your model and cache them on disk. This speeds up SAE training by 10-20x."

**ACTIONS:**
1. Navigate to Models panel
2. Click on the GPT-2 model card
3. Click "Extract Activations" button (or similar)
4. Show activation extraction configuration modal:
   - Select dataset: wikitext-2-raw-v1
   - Select layers: 0, 6, 11
   - Activation type: Residual Stream
   - Batch size: 128
   - Sample limit: 100,000 (or unlimited)
5. Show resource estimates:
   - Disk space required: ~2.5 GB
   - Extraction time: ~5 minutes
6. Click "Start Extraction" button (but don't actually run it for demo)

**VISUAL CUES:**
- Highlight "Extract Activations" button on model card
- Zoom in on configuration modal
- Highlight resource estimates
- Add annotation: "10-20x training speedup"

**NARRATION:**
> "Activation extraction saves the model's intermediate representations to disk. Then, during SAE training, you reference this cached extraction instead of running the model forward pass every time. This is a huge speedup for large models or long training runs."

**EDITING NOTES:**
- This section is optional (skip for shorter demo)
- Keep it brief (30-60 seconds if included)

---

## Conclusion (1-2 minutes)

### Scene: Recap & Future Directions

**[SCENE: Pan through key panels one last time]**

**NARRATION:**
> "Let's recap what we've accomplished in this demo."

**ACTIONS:**
1. Quick montage through panels (2 seconds each):
   - Datasets panel: "Downloaded and tokenized WikiText-2"
   - Models panel: "Downloaded GPT-2 model"
   - Training panel: "Trained sparse autoencoders on 3 layers"
   - Features panel: "Discovered 8,192 interpretable features"
   - Labeling panel: "Generated semantic labels with local LLM"
   - Templates panel: "Saved configurations for future experiments"
   - System Monitor: "Monitored resources in real-time"

**VISUAL CUES:**
- Fast transitions between panels (2x speed)
- Add text overlays for each step:
  - "✓ Data Preparation"
  - "✓ SAE Training"
  - "✓ Feature Discovery"
  - "✓ Semantic Labeling"
  - "✓ Workflow Optimization"

**NARRATION (continued):**
> "We went from raw data to interpretable features in just a few steps. miStudio handled the entire mechanistic interpretability workflow: downloading datasets and models from HuggingFace, training sparse autoencoders, extracting features, and generating semantic labels."

**EDITING NOTES:**
- Montage-style editing (upbeat music)
- Keep pace fast (30 seconds total)

---

**[SCENE: Feature Browser (Final View)]**

**NARRATION:**
> "The result is a comprehensive library of interpretable features - we can now understand what GPT-2 learned at different layers. Early layers capture syntax and low-level patterns. Middle layers represent semantic concepts. Late layers encode high-level abstractions."

**ACTIONS:**
1. Return to feature browser
2. Show a few diverse features:
   - Layer 0: "Punctuation - Comma"
   - Layer 6: "Names - Person"
   - Layer 11: "Sentiment - Positive Valence"
3. Pan over feature table

**VISUAL CUES:**
- Slow pan over feature table (cinematic)
- Add text overlays highlighting layer progression:
  - "Layer 0: Syntax"
  - "Layer 6: Semantics"
  - "Layer 11: Abstractions"

---

**[SCENE: Desktop or Closing Screen]**

**NARRATION:**
> "miStudio is an open-source platform designed for mechanistic interpretability research, optimized for both edge devices and data center GPUs. It's built with modern web technologies - React, FastAPI, PostgreSQL - and supports real-time progress tracking via WebSockets."

**ACTIONS:**
1. Close browser (or minimize to show desktop)
2. Show terminal with running services (optional)

**VISUAL CUES:**
- Add text overlays:
  - "Open Source"
  - "Edge-Optimized (NVIDIA Jetson)"
  - "Real-Time WebSocket Updates"
  - "Comprehensive Feature Browser"
  - "Local & Cloud LLM Support"
- Add GitHub link: "github.com/[your-repo]"

**NARRATION (continued):**
> "Key features include multi-layer SAE training, comprehensive feature analysis with logit lens and ablation studies, multiple labeling methods, and a powerful templates system for workflow automation."

---

**[SCENE: Call to Action]**

**NARRATION:**
> "To get started with miStudio, check out the GitHub repository linked in the description. The platform is under active development - we're currently working on model steering and intervention features, which will let you use discovered features to control model behavior. Thanks for watching, and happy interpreting!"

**VISUAL CUES:**
- Add text overlay: "Get Started: github.com/[your-repo]"
- Add text overlay: "Coming Soon: Model Steering & Interventions"
- Add social media handles (optional)
- Fade to black with logo

**EDITING NOTES:**
- Add call-to-action music (inspirational)
- End card with links (YouTube end screen)
- Add subscribe button animation

---

## Post-Recording Checklist

### Video Editing
- [ ] Trim dead air and long pauses
- [ ] Speed up typing and form filling (2x speed)
- [ ] Time-lapse long operations (downloads, training, labeling)
- [ ] Add text overlays and annotations
- [ ] Add background music (subtle, non-intrusive)
- [ ] Normalize audio levels
- [ ] Add intro/outro animations
- [ ] Color grade for consistency (if needed)
- [ ] Add captions/subtitles (accessibility)
- [ ] Export at 1080p, 30 FPS (or 4K if recorded at higher resolution)

### YouTube Preparation
- [ ] Create compelling thumbnail:
  - Feature browser screenshot with token highlighting
  - Bold text: "miStudio - Mechanistic Interpretability"
  - Visually appealing (emerald accents, dark theme)
- [ ] Write video description:
  - Brief overview (2-3 sentences)
  - Timestamps for each section
  - Link to GitHub repository
  - Link to documentation
  - Credits and acknowledgments
- [ ] Add tags:
  - Mechanistic Interpretability
  - Sparse Autoencoders
  - Machine Learning
  - AI Safety
  - Interpretability
  - Deep Learning
  - GPT-2
  - PyTorch
- [ ] Set up YouTube end screen:
  - Subscribe button
  - Related videos
  - External link (GitHub)
- [ ] Enable comments and community features

### Distribution
- [ ] Share on relevant subreddits: r/MachineLearning, r/LanguageTechnology
- [ ] Share on Twitter/X with relevant hashtags
- [ ] Share on LinkedIn
- [ ] Post in AI Safety / Interpretability communities
- [ ] Share in research group Slack/Discord channels

---

## Alternative Script Variations

### Short Version (8-10 minutes)
- Skip: Cached activations, detailed ablation analysis, templates
- Fast-forward: Tokenization, training progress, labeling progress
- Focus: Core workflow (dataset → model → training → features)

### Long Version (25-30 minutes)
- Add: Detailed explanation of SAE theory
- Add: Comparison of different labeling methods (run multiple labeling jobs)
- Add: Demonstrate training templates (load template and start new training)
- Add: Show extraction templates in action
- Add: Demonstrate export/import functionality
- Add: Show failed job recovery (retry button)

### Technical Deep-Dive Version (40-60 minutes)
- Add: Detailed architecture explanation
- Add: Walk through backend code (FastAPI, Celery, WebSocket emission)
- Add: Walk through frontend code (React, Zustand, WebSocket hooks)
- Add: Database schema explanation
- Add: Deployment walkthrough (Docker Compose, systemd)
- Add: Performance optimization tips
- Add: Edge deployment (Jetson setup)

---

## Technical Notes

### Performance Considerations
- **Fast Demo Setup:**
  - Use small dataset: wikitext-2-raw-v1 (4MB, 36K samples)
  - Use small model: distilgpt2 (82M params, ~300MB)
  - Train for 500-1000 steps (2-3 minutes)
  - Use pattern labeling (instant)

- **Realistic Demo Setup:**
  - Use medium dataset: openwebtext (8GB subset)
  - Use standard model: gpt2 (124M params, ~500MB)
  - Train for 10,000 steps (10-15 minutes)
  - Use local LLM labeling (Phi-3, 10-15 minutes)

- **Impressive Demo Setup:**
  - Use large dataset: pile-deduped (50GB subset)
  - Use large model: gpt2-medium (355M params, ~1.5GB)
  - Train for 50,000 steps (1-2 hours)
  - Use OpenAI labeling (GPT-4o-mini, 5-10 minutes)

### Troubleshooting
- **If training is too slow:** Reduce batch size or training steps
- **If download is too slow:** Use smaller model (distilgpt2) or dataset
- **If labeling is too slow:** Use pattern matching or reduce feature count
- **If GPU memory is insufficient:** Train one layer at a time, reduce batch size
- **If database is too cluttered:** Clean up old jobs before recording

### Screen Recording Tips
- **Resolution:** 1920x1080 (Full HD) - readable text, manageable file size
- **Frame Rate:** 30 FPS minimum (60 FPS for smoother motion)
- **Bitrate:** 8-12 Mbps for high quality
- **Audio:** 48 kHz, 192 kbps (clear voice)
- **Cursor:** Enable cursor highlighting (helps viewers follow along)
- **Lighting:** Good lighting for face cam (if using)
- **Noise:** Use noise cancellation or quiet environment

---

## Appendix: Key Talking Points

### What is Mechanistic Interpretability?
> "Mechanistic interpretability aims to reverse-engineer neural networks - to understand not just what they do, but how they do it. Sparse autoencoders are a powerful technique that decomposes model activations into interpretable features, revealing the internal representations that drive model behavior."

### Why Sparse Autoencoders?
> "Neural networks learn distributed representations - each neuron responds to many different concepts, and each concept activates many neurons. This makes interpretation difficult. Sparse autoencoders learn an overcomplete basis of features where each feature is highly sparse and interpretable. This 'unmixing' reveals the true building blocks of the model's representations."

### Why miStudio?
> "Existing interpretability tools are fragmented - you need separate tools for data preparation, training, analysis, and visualization. miStudio unifies the entire workflow in a modern, user-friendly platform. It's optimized for edge devices like NVIDIA Jetson, making interpretability research accessible even without access to data center GPUs."

### Key Differentiators
- **End-to-end workflow:** Datasets → Models → Training → Features → Labels
- **Real-time progress tracking:** WebSocket-powered updates for all async operations
- **Edge-optimized:** Runs on NVIDIA Jetson with mixed precision and quantization
- **Multiple labeling methods:** Pattern matching, local LLMs (Phi-3/Llama/Qwen), OpenAI API
- **Comprehensive feature analysis:** Examples, logit lens, correlations, ablation
- **Workflow automation:** Templates for training, extraction, and labeling configurations
- **Production-ready:** Robust error handling, validation, progress recovery

---

## Final Production Checklist

Before publishing:
- [ ] All sections recorded and edited
- [ ] Audio quality verified (no background noise, clear voice)
- [ ] Video quality verified (readable text, smooth playback)
- [ ] Annotations and overlays added
- [ ] Intro and outro included
- [ ] Captions/subtitles added (if applicable)
- [ ] Thumbnail created and compelling
- [ ] Description written with timestamps
- [ ] Tags added
- [ ] End screen configured
- [ ] Video reviewed by colleague (optional but recommended)
- [ ] GitHub repository link verified
- [ ] Documentation link verified
- [ ] Published and shared!

---

**Script Version:** 1.0
**Created:** 2025-11-15
**Last Updated:** 2025-11-15
**Estimated Total Duration:** 15-20 minutes (standard path)
**Target Audience:** ML researchers, interpretability practitioners, AI safety researchers
**Recording Status:** Ready for production
