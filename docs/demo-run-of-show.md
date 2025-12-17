# miStudio Demo - Run of Show

## Overview
This document outlines the screen recording workflow for demonstrating miStudio's SAE training and feature discovery capabilities, using Neuronpedia's GemmaScope as a reference point.

---

## Part 1: Neuronpedia - Understanding SAE Features (3-5 min)

### Scene 1.1: Navigate to Neuronpedia GemmaScope
**URL:** https://www.neuronpedia.org/gemma-scope

**Actions:**
1. Open browser, navigate to neuronpedia.org
2. Click on "Gemma Scope" or navigate directly to the URL above
3. **Narration:** "Neuronpedia hosts pre-trained Sparse Autoencoders from Google's GemmaScope project. Let's explore how features work before training our own."

### Scene 1.2: Select Gemma-2-2B Model
**Actions:**
1. From the GemmaScope page, select **Gemma-2-2B** model
2. Choose the **16K width** SAE (res-16k)
3. **Narration:** "We're looking at the Gemma 2 2B model with a 16,000 feature SAE trained on the residual stream."

### Scene 1.3: Explore the "Yelling" Feature
**Navigation:** Advanced Steering → Search for uppercase/yelling feature

**Actions:**
1. Click on "Advanced Steering" or "Steer Gemma"
2. In the feature search, look for features related to:
   - "uppercase"
   - "caps"
   - "emphasis"
   - "exclamation"
3. Select a feature that activates on ALL CAPS or emphatic text
4. **Note the feature number** (e.g., feature #XXXX at layer 20)

**Narration:** "This feature activates when the model sees text written in all capitals - commonly associated with yelling or emphasis online."

### Scene 1.4: Demonstrate Steering
**Actions:**
1. Enter a prompt like: "Write a short message about the weather"
2. Show baseline response (normal)
3. Increase the feature strength slider (e.g., +2.0 or +3.0)
4. Generate again - response should be in ALL CAPS
5. **Narration:** "By amplifying this feature, we can steer the model to respond as if it's yelling!"

### Scene 1.5: Note the SAE Details
**Record these details for Part 2:**
- Model: `gemma-2-2b`
- SAE: `20-gemmascope-res-16k` (Layer 20, Residual Stream, 16K features)
- Feature Number: [note from exploration]
- HuggingFace path: `google/gemma-scope-2b-pt-res`

---

## Part 2: Download SAE from HuggingFace (2-3 min)

### Scene 2.1: Navigate to HuggingFace
**URL:** https://huggingface.co/google/gemma-scope-2b-pt-res

**Actions:**
1. Open new tab, navigate to the HuggingFace URL
2. **Narration:** "Google publishes all GemmaScope SAEs on HuggingFace. Let's download the Layer 20 SAE we just explored."

### Scene 2.2: Browse to Layer 20 SAE
**Path:** `layer_20/width_16k/average_l0_71`

**Actions:**
1. Click "Files and versions" tab
2. Navigate folder structure:
   - `layer_20/` → Click to expand
   - `width_16k/` → Click to expand
   - `average_l0_71/` → This is the SAE with L0 ≈ 71
3. **Narration:** "The 'average_l0_71' refers to the sparsity level - on average, 71 features activate per token."

### Scene 2.3: Show SAE Files
**Files in the directory:**
- `cfg.json` - Configuration file
- `sae_weights.safetensors` - The actual SAE weights

**Actions:**
1. Click on `cfg.json` to show the configuration
2. Point out key parameters:
   - `d_in`: Input dimension (model hidden size)
   - `d_sae`: SAE dimension (16384 features)
   - `hook_name`: Where it attaches (`blocks.20.hook_resid_post`)
3. **Narration:** "These files are in SAELens format, which miStudio fully supports."

---

## Part 3: miStudio - Training Your Own SAE (10-15 min)

### Scene 3.1: Open miStudio
**URL:** http://mistudio.mcslab.io

**Actions:**
1. Navigate to miStudio
2. Show the dashboard overview
3. **Narration:** "Now let's train our own SAE using miStudio. We'll replicate what Google did with GemmaScope, but on a smaller scale."

### Scene 3.2: Dataset Management
**Panel:** Datasets

**Actions:**
1. Click "Datasets" in sidebar
2. Click "Add Dataset" → "From HuggingFace"
3. Enter: `monology/pile-uncopyrighted` (or similar)
4. Select split and sample size
5. Start download
6. **Narration:** "First, we need training data. miStudio can pull datasets directly from HuggingFace."

### Scene 3.3: Model Management
**Panel:** Models

**Actions:**
1. Click "Models" in sidebar
2. Click "Add Model" → "From HuggingFace"
3. Enter: `google/gemma-2-2b`
4. Show quantization options (4-bit for memory efficiency)
5. Start download
6. **Narration:** "We'll use the same Gemma 2 2B model. miStudio supports quantization for running on consumer GPUs."

### Scene 3.4: Configure SAE Training
**Panel:** Training

**Actions:**
1. Click "Training" in sidebar
2. Click "New Training"
3. Configure parameters:
   - **Dataset:** Select the downloaded dataset
   - **Model:** Select Gemma-2-2B
   - **Target Layer:** 20 (to match what we saw on Neuronpedia)
   - **Hook Point:** `resid_post` (residual stream)
   - **SAE Width:** 16384 (16K features)
   - **Expansion Factor:** 8x (if shown)
   - **Learning Rate:** 3e-4
   - **Batch Size:** 4096 tokens
   - **Training Steps:** 10000 (for demo)
4. **Narration:** "We configure the SAE to match the GemmaScope setup - Layer 20, residual stream, 16K features."

### Scene 3.5: Start Training
**Actions:**
1. Click "Start Training"
2. Show real-time training metrics:
   - Loss curve
   - L0 sparsity (should approach ~70-100)
   - Reconstruction loss
   - GPU utilization
3. **Narration:** "Training begins! miStudio shows real-time metrics. Watch the L0 sparsity - it measures how many features activate per token."

### Scene 3.6: Monitor Training Progress
**Actions:**
1. Point out the live-updating charts
2. Show WebSocket status (connected)
3. Explain key metrics:
   - "L0 around 70-100 means good sparsity"
   - "Reconstruction loss should decrease steadily"
4. **Fast-forward or cut** to show completed training

---

## Part 4: Feature Discovery (5-7 min)

### Scene 4.1: Run Feature Extraction
**Panel:** Features / Extraction

**Actions:**
1. After training completes, go to "Features" panel
2. Select the trained SAE
3. Click "Extract Features"
4. Configure:
   - Top-K per feature: 100
   - Activation threshold: 0.5
5. Start extraction
6. **Narration:** "Now we extract what each feature has learned by finding text that activates it."

### Scene 4.2: Browse Features
**Actions:**
1. Show the feature browser
2. Click through several features
3. For each feature, show:
   - Top activating examples (highlighted tokens)
   - Activation histogram
   - Feature statistics
4. **Narration:** "Each feature captures a specific pattern. Let's see if we can find features similar to the 'yelling' feature from Neuronpedia."

### Scene 4.3: Search for Uppercase Feature
**Actions:**
1. Use the search/filter functionality
2. Look for features that activate on:
   - Capital letters
   - Exclamation marks
   - Emphatic text
3. When found, compare to Neuronpedia
4. **Narration:** "Here's a feature that activates on capitalized text - our own 'yelling' detector!"

### Scene 4.4: Auto-Label Features
**Actions:**
1. Select multiple features
2. Click "Auto-Label"
3. Show the labeling progress (if using OpenAI or Ollama)
4. View generated labels
5. **Narration:** "miStudio can automatically label features using GPT-4 or local LLMs via Ollama."

---

## Part 5: Model Steering (3-5 min)

### Scene 5.1: Navigate to Steering
**Panel:** Steering

**Actions:**
1. Click "Steering" in sidebar
2. Select the trained SAE
3. Choose a feature to steer with

### Scene 5.2: Demonstrate Steering
**Actions:**
1. Enter a test prompt
2. Generate baseline response
3. Increase feature strength
4. Generate steered response
5. Compare the outputs
6. **Narration:** "Just like Neuronpedia, we can now steer the model using our own trained features!"

### Scene 5.3: Export SAE
**Actions:**
1. Show export options
2. Export in Neuronpedia format
3. **Narration:** "You can export your SAE in Neuronpedia-compatible format to share with the community."

---

## Closing (1 min)

**Actions:**
1. Return to dashboard
2. Show System Monitor (GPU utilization, etc.)
3. Summarize what was demonstrated

**Narration:**
"In this demo, we:
1. Explored pre-trained SAE features on Neuronpedia
2. Downloaded a GemmaScope SAE from HuggingFace
3. Trained our own SAE using miStudio
4. Discovered and labeled interpretable features
5. Steered the model using our trained SAE

miStudio makes mechanistic interpretability accessible - train SAEs, discover features, and steer models, all from your browser."

---

## Technical Notes

### URLs Referenced
- Neuronpedia GemmaScope: https://www.neuronpedia.org/gemma-scope
- Neuronpedia Steering Docs: https://docs.neuronpedia.org/steering
- HuggingFace Gemma Scope: https://huggingface.co/google/gemma-scope-2b-pt-res
- SAE Path: `layer_20/width_16k/average_l0_71`

### SAE Naming Convention
- `L20` = Layer 20
- `71` = Average L0 sparsity (~71 features activate per token)
- `res` = Residual stream (vs `mlp` or `att`)
- `16k` = 16,384 features

### Recommended Screen Recording Settings
- Resolution: 1920x1080 or 2560x1440
- Frame rate: 30fps
- Microphone: External mic recommended
- Browser: Chrome (DevTools can be shown for technical audiences)

### Timing Summary
| Part | Duration |
|------|----------|
| 1. Neuronpedia Exploration | 3-5 min |
| 2. HuggingFace Download | 2-3 min |
| 3. miStudio Training | 10-15 min |
| 4. Feature Discovery | 5-7 min |
| 5. Model Steering | 3-5 min |
| 6. Closing | 1 min |
| **Total** | **24-36 min** |

---

*Generated: 2025-12-17*
