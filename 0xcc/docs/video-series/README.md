# miStudio YouTube Video Series

**Channel:** HITSAI (@miStudio-hitsai)
**Tagline:** "Democratizing Mechanistic Interpretability"

---

## Video Series Overview

A complete tutorial series demonstrating end-to-end SAE development with miStudio. Short-format videos (2-4 minutes each) designed for practical, hands-on learning.

---

## Video Index

| # | Title | Duration | Status | Script |
|---|-------|----------|--------|--------|
| - | Steer Like Neuronpedia (Conceptual Intro to MI) | 3:50 | **Published** | - |
| 1 | Download and Tokenize Dataset from HF | 2:25 | **Published** | - |
| 2 | Download Models from HF | 4:14 | **Published** | - |
| 3 | Train Your First SAE | 3-4 min | Script Ready | [video-02-train-sae.md](video-02-train-sae.md) |
| 4 | Extract and Browse Features | 3-4 min | Script Ready | [video-03-feature-discovery.md](video-03-feature-discovery.md) |
| 5 | Auto-Label Features with LLM | 3-4 min | Script Ready | [video-04-auto-labeling.md](video-04-auto-labeling.md) |
| 6 | Steer Model Behavior with Features | 3-4 min | Script Ready | [video-05-steering-demo.md](video-05-steering-demo.md) |
| 7 | Export SAE to HuggingFace | 2-3 min | Script Ready | [video-06-export-huggingface.md](video-06-export-huggingface.md) |

**Total Series Duration:** ~22-27 minutes (including published videos)

---

## Workflow Sequence

```
┌─────────────────────────────────────────────────────────────────┐
│  "Steer Like Neuronpedia" - What is MI? Why does it matter?    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Dataset    │ →  │   Model     │ →  │  Train SAE  │
│  Download   │    │  Download   │    │             │
│  (Published)│    │ (Published) │    │  (Script)   │
└─────────────┘    └─────────────┘    └─────────────┘
                                              ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Export    │ ←  │  Steering   │ ←  │  Features   │
│   to HF     │    │    Demo     │    │  + Labels   │
│  (Script)   │    │  (Script)   │    │  (Scripts)  │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## Published Videos

### Steer Like Neuronpedia
- **Purpose:** Conceptual introduction to mechanistic interpretability
- **Content:** What is MI, why steering matters, how miStudio fits
- **Not a demo** - separate from the hands-on steering tutorial

### Download and Tokenize Dataset from HF (2:25)
- Download datasets from HuggingFace
- Configure tokenization with model's tokenizer
- View tokenization statistics

### Download Models from HF (4:14)
- Search and download transformer models
- View model architecture
- Configure quantization options

---

## Scripts to Record

### 3. Train Your First SAE
- Select tokenized dataset and model
- Configure hook point and hyperparameters
- Monitor training in real-time
- View completed SAE in library

### 4. Extract and Browse Features
- Run feature extraction
- Browse top activating examples
- Understand feature patterns
- View activation statistics

### 5. Auto-Label Features with LLM
- Configure labeling backend (OpenAI/Ollama)
- Run batch labeling
- Review and edit labels
- Search features by label

### 6. Steer Model Behavior (Hands-on Demo)
- Search for features to steer
- Generate baseline output
- Amplify/suppress features
- Compare steered vs baseline

### 7. Export SAE to HuggingFace
- Package in SAELens format
- Configure HuggingFace upload
- Verify published repository

---

## Recording Checklist (All Videos)

### Technical Setup
- [ ] Screen resolution: 1920x1080 or 2560x1440
- [ ] Browser zoom: 100%
- [ ] Dark mode enabled
- [ ] Mouse cursor visible
- [ ] Microphone tested

### Environment
- [ ] miStudio running at http://mistudio.mcslab.io
- [ ] Close unnecessary applications
- [ ] Disable notifications
- [ ] GPU available (check System Monitor)

### Content
- [ ] Review script before recording
- [ ] Practice walkthrough once
- [ ] Prepare any needed data/models/SAEs

---

## Channel Branding

### Tone
- Professional but approachable
- Educational, not salesy
- Assume intelligence, not prior knowledge
- Short and practical (2-4 minutes)

### Visual Style
- Clean, dark interface (miStudio dark mode)
- Slow, deliberate mouse movements
- Highlight areas being discussed
- Jump cuts OK for long operations

### Audio
- Clear, well-paced narration
- No background music (keeps videos concise)
- Consistent volume levels

---

## Playlist Description

```
miStudio: Complete SAE Development Workflow

Learn to train, analyze, and steer Sparse Autoencoders locally with miStudio - an open-source mechanistic interpretability workbench.

This series covers:
1. Data preparation - Download & tokenize from HuggingFace
2. Model setup - Download & configure transformer models
3. SAE training - Train with real-time monitoring
4. Feature discovery - Extract and browse interpretable features
5. Auto-labeling - Generate human-readable feature descriptions
6. Steering - Control model behavior through features
7. Sharing - Export to HuggingFace for the community

No cloud required. Run everything on your own hardware.

GitHub: [repo-url]
Documentation: [docs-url]

#MechanisticInterpretability #SAE #LocalAI
```

---

## File Structure

```
0xcc/docs/video-series/
├── README.md                    # This file
├── video-01-introduction-setup.md  # (Deferred - introduction video)
├── video-01-quick-reference.md     # (Deferred - quick reference)
├── video-02-train-sae.md        # Script: Train Your First SAE
├── video-03-feature-discovery.md # Script: Extract and Browse Features
├── video-04-auto-labeling.md    # Script: Auto-Label Features
├── video-05-steering-demo.md    # Script: Steering Demo
└── video-06-export-huggingface.md # Script: Export to HuggingFace
```

---

## Next Steps

1. Record **Train Your First SAE** (natural next step after existing videos)
2. Record **Feature Discovery**
3. Record **Auto-Labeling**
4. Record **Steering Demo**
5. Record **Export to HuggingFace**
6. (Later) Record **Introduction/Overview** video if needed
