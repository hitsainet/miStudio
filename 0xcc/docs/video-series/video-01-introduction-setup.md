# Video 1: Getting Started with miStudio

**Series:** miStudio SAE Development Tutorial
**Video Title:** "Getting Started with miStudio - Your Local SAE Training Studio"
**Target Duration:** 5-8 minutes
**Difficulty:** Beginner

---

## Pre-Recording Checklist

### Environment Setup
- [ ] miStudio running and accessible at http://mistudio.mcslab.io
- [ ] All panels visible and responsive
- [ ] No active training jobs (clean slate for demo)
- [ ] System Monitor showing GPU available
- [ ] Browser at 100% zoom, dark mode enabled
- [ ] Screen resolution: 1920x1080 or 2560x1440 recommended
- [ ] Close unnecessary browser tabs and applications

### Content Preparation
- [ ] Have 1-2 models already downloaded (for showing the Models panel isn't empty)
- [ ] Have 1 dataset already downloaded (for showing Datasets panel)
- [ ] Optional: Have 1 completed training (to show what success looks like)

### Recording Setup
- [ ] Microphone tested and levels set
- [ ] Screen recording software ready (OBS, ScreenFlow, etc.)
- [ ] Mouse cursor visible and appropriately sized
- [ ] Disable notifications (Do Not Disturb mode)

---

## Script & Storyboard

### Scene 1: Opening Hook (0:00 - 0:30)

**[SCREEN: miStudio landing page with all panels visible]**

**VOICEOVER:**
> "What if you could train, analyze, and steer sparse autoencoders right from your own machine? No cloud dependencies, no API costs eating into your research budget, and complete control over your interpretability experiments.
>
> I'm going to show you miStudio - an open-source application that brings the full SAE development workflow to your local environment, whether that's a powerful workstation or even a Jetson edge device."

**ACTIONS:**
- Show the full miStudio interface
- Slow pan across the navigation tabs (Datasets, Models, SAEs, Training, Features, Steering)

---

### Scene 2: What is miStudio? (0:30 - 1:30)

**[SCREEN: Still on main interface, highlight different sections as mentioned]**

**VOICEOVER:**
> "miStudio is a complete mechanistic interpretability workbench. It handles everything from downloading models and datasets from HuggingFace, to training sparse autoencoders, discovering interpretable features, and even steering model behavior based on what you find.
>
> The entire pipeline runs locally on your hardware. You can use it air-gapped, with no internet required after you've downloaded your initial models and data.
>
> Let me give you a quick tour of what's here."

**ACTIONS:**
- Hover over each main panel tab as you mention it
- Keep movements slow and deliberate for viewers to follow

---

### Scene 3: Interface Tour - Navigation (1:30 - 2:30)

**[SCREEN: Click through each panel tab]**

**VOICEOVER:**
> "The interface is organized into six main panels, arranged in the order you'll typically use them:
>
> **Datasets** - where you download and prepare your training data from HuggingFace. You can tokenize the same dataset with different models.
>
> **Models** - where you download transformer models. miStudio supports quantization to fit larger models in limited GPU memory.
>
> **SAEs** - your library of trained sparse autoencoders, both ones you train yourself and ones you download from the community.
>
> **Training** - where you configure and monitor SAE training jobs with real-time metrics.
>
> **Features** - where you extract, browse, and label the interpretable features your SAE has learned.
>
> **Steering** - where you test your features by amplifying or suppressing them during text generation."

**ACTIONS:**
1. Click "Datasets" tab - pause 2 seconds
2. Click "Models" tab - pause 2 seconds
3. Click "SAEs" tab - pause 2 seconds
4. Click "Training" tab - pause 2 seconds
5. Click "Features" tab - pause 2 seconds
6. Click "Steering" tab - pause 2 seconds

---

### Scene 4: System Monitor Overview (2:30 - 3:30)

**[SCREEN: Click on System Monitor in the header or navigate to it]**

**VOICEOVER:**
> "Before we dive into training, let's check our system resources. The System Monitor shows you real-time GPU utilization, memory usage, and temperature.
>
> This is crucial for SAE training because you need to know how much GPU memory you have available. That determines how large a model you can load and what batch sizes you can use.
>
> Here you can see my GPU is currently at [X]% utilization with [Y] GB of memory available. The temperature is healthy at [Z] degrees.
>
> During training, you'll see these metrics update in real-time, which helps you tune your hyperparameters without running out of memory."

**ACTIONS:**
- Open System Monitor panel/modal
- Point out GPU utilization chart
- Point out memory usage
- Point out temperature
- Show how metrics update in real-time (wait a few seconds)

---

### Scene 5: The SAE Workflow Concept (3:30 - 5:00)

**[SCREEN: Could use a simple diagram overlay, or just talk over the interface]**

**VOICEOVER:**
> "Now let me explain the workflow we'll follow in this video series.
>
> **Step 1: Data Preparation**
> We'll download a dataset and a model from HuggingFace, then tokenize the dataset using the model's tokenizer. This prepares our text for training.
>
> **Step 2: SAE Training**
> We'll configure and train a sparse autoencoder on the model's internal activations. The SAE learns to decompose the model's representations into interpretable features.
>
> **Step 3: Feature Discovery**
> Once trained, we'll extract features by running the dataset through our SAE and collecting examples of what makes each feature activate.
>
> **Step 4: Feature Labeling**
> We'll use an LLM - either GPT-4 or a local model through Ollama - to automatically generate human-readable labels for our features.
>
> **Step 5: Steering**
> Finally, we'll test our understanding by steering the model's behavior - amplifying features we want more of, or suppressing ones we want less of.
>
> **Step 6: Sharing**
> We can export our SAE in Neuronpedia-compatible format and upload it to HuggingFace for the community to use."

**ACTIONS:**
- As each step is mentioned, briefly hover over the corresponding panel tab
- Keep a steady pace - this is the conceptual foundation

---

### Scene 6: What You'll Need (5:00 - 6:00)

**[SCREEN: Back to main interface or show terminal/requirements]**

**VOICEOVER:**
> "To follow along with this series, you'll need:
>
> - A machine with an NVIDIA GPU - I recommend at least 8GB of VRAM for comfortable training
> - miStudio installed and running - check the GitHub repository for installation instructions
> - A HuggingFace account for downloading models and datasets
> - Optionally, an OpenAI API key if you want to use GPT-4 for auto-labeling, though we'll also show how to use local models with Ollama
>
> In the next video, we'll download TinyLlama - a small but capable model that's perfect for learning the SAE workflow - along with a subset of The Pile dataset for training."

**ACTIONS:**
- Could show the GitHub repo briefly
- Show the HuggingFace login prompt in the download modal (don't actually enter credentials)

---

### Scene 7: Closing & Next Steps (6:00 - 6:30)

**[SCREEN: Main interface with all panels visible]**

**VOICEOVER:**
> "That's your introduction to miStudio. You now understand the interface layout and the end-to-end workflow we'll be following.
>
> In the next video, we'll get hands-on by downloading our model and dataset, and preparing everything for SAE training.
>
> If you found this helpful, subscribe and hit the notification bell so you don't miss the rest of the series. Drop any questions in the comments below.
>
> See you in the next one."

**ACTIONS:**
- Return to the main Datasets panel
- End on a clean, professional shot of the interface

---

## B-Roll Suggestions

If you want to add visual variety, consider capturing:

1. **Terminal showing miStudio starting up** (./start-mistudio.sh output)
2. **GPU status from nvidia-smi** in terminal
3. **Quick cuts of different panels** to add energy
4. **Zoom-in on specific UI elements** when discussing them

---

## Key Terms to Define

| Term | Definition for Viewers |
|------|------------------------|
| **Sparse Autoencoder (SAE)** | A neural network that learns to represent model activations as combinations of interpretable features |
| **Activation** | The numerical output of a neuron or layer in a neural network |
| **Feature** | A direction in activation space that corresponds to a human-interpretable concept |
| **Tokenization** | Converting text into numerical tokens that the model can process |
| **Steering** | Modifying model behavior by artificially boosting or suppressing specific features |

---

## Common Mistakes to Avoid

1. **Moving too fast** - Viewers need time to see UI elements
2. **Clicking without explaining** - Always say what you're about to do
3. **Assuming knowledge** - Define terms as you introduce them
4. **Not showing the mouse cursor** - Viewers follow the cursor to understand navigation
5. **Recording with notifications on** - Very distracting

---

## Thumbnail Ideas

- miStudio interface with text overlay: "Local SAE Training"
- Split image: code terminal + miStudio UI
- GPU/brain imagery with "Interpretability on Your Machine"

---

## Video Description Template

```
Learn how to train, analyze, and steer Sparse Autoencoders (SAEs) locally with miStudio - an open-source mechanistic interpretability workbench.

In this first video of the series, we tour the miStudio interface and understand the complete SAE development workflow.

TIMESTAMPS:
0:00 - Introduction
0:30 - What is miStudio?
1:30 - Interface Tour
2:30 - System Monitor
3:30 - The SAE Workflow
5:00 - Requirements
6:00 - Next Steps

LINKS:
- miStudio GitHub: [your-repo-url]
- HuggingFace: https://huggingface.co
- Series Playlist: [playlist-url]

SERIES:
1. Getting Started (this video)
2. Data Preparation
3. Training Your First SAE
4. Feature Discovery
5. Auto-Labeling Features
6. Model Steering
7. Export & Sharing

#MechanisticInterpretability #SAE #MachineLearning #AIResearch
```

---

## Next Video Preview

**Video 2: Data Preparation** will cover:
- Downloading TinyLlama model
- Viewing model architecture
- Downloading a dataset subset
- Tokenizing with filtering options
- Reviewing tokenization statistics

Prepare by ensuring you have:
- HuggingFace token ready (for gated models if needed)
- Sufficient disk space (~5GB for model + dataset)
- Stable internet connection
