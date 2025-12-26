# Video: Train Your First SAE

**Channel:** HITSAI / miStudio
**Title:** "miStudio: Train Your First SAE"
**Target Duration:** 3-4 minutes
**Prerequisites:** Dataset tokenized, Model downloaded

---

## Pre-Recording Checklist

- [ ] TinyLlama model downloaded and ready
- [ ] Dataset tokenized (e.g., The Pile subset, ~10k samples)
- [ ] No active training jobs
- [ ] GPU available (check System Monitor)
- [ ] Browser at 100% zoom, dark mode

---

## Script

### Scene 1: Opening (0:00 - 0:15)

**[SCREEN: Training panel, empty state]**

**VOICEOVER:**
> "You've got your model and dataset ready. Now let's train a sparse autoencoder to discover what features your model has learned."

**ACTION:** Show Training panel

---

### Scene 2: Create New Training (0:15 - 1:00)

**[SCREEN: Click "New Training" button]**

**VOICEOVER:**
> "Click New Training. First, select your tokenized dataset - I'm using a Pile subset tokenized with TinyLlama.
>
> Select your model - TinyLlama in my case.
>
> For architecture, we'll target the residual stream at layer 8. This is where the model builds up its representations before the final output."

**ACTIONS:**
1. Click "New Training" button
2. Select dataset from dropdown (pause)
3. Select model from dropdown (pause)
4. Set hook point: `blocks.8.hook_resid_post`
5. Show expansion factor (default 8x)

---

### Scene 3: Hyperparameters (1:00 - 1:45)

**[SCREEN: Scroll through training config]**

**VOICEOVER:**
> "The key hyperparameters:
>
> **Dictionary size** - how many features to learn. 8x expansion of the model's hidden dimension is a good starting point.
>
> **L1 coefficient** - controls sparsity. Higher means fewer features activate per input. Start around 0.001.
>
> **Learning rate** - 3e-4 works well for most cases.
>
> **Training steps** - for a quick test, 1000 steps. For production, 50,000 or more."

**ACTIONS:**
1. Point to dictionary size field
2. Point to L1 coefficient
3. Point to learning rate
4. Set steps to 1000 for demo

---

### Scene 4: Start Training (1:45 - 2:30)

**[SCREEN: Click Start Training]**

**VOICEOVER:**
> "Click Start Training. The job appears in the queue and starts processing.
>
> Watch the real-time metrics - you'll see the reconstruction loss drop as the SAE learns to compress and decompress the model's activations.
>
> The L0 sparsity shows how many features activate on average. Lower is sparser."

**ACTIONS:**
1. Click "Start Training"
2. Watch training card appear
3. Point to loss chart as it updates
4. Point to L0 metric
5. Show GPU utilization in System Monitor (quick glance)

---

### Scene 5: Training Complete (2:30 - 3:15)

**[SCREEN: Training completes or shows good progress]**

**VOICEOVER:**
> "When training completes, your SAE is saved to the SAEs library. You can see the final metrics here - reconstruction loss and sparsity.
>
> From here, you can train another SAE with different hyperparameters, or move on to feature discovery to see what your SAE actually learned."

**ACTIONS:**
1. Show completed training card (or skip ahead to completed one)
2. Point to final loss value
3. Navigate to SAEs panel to show the new SAE
4. Hover over "Features" tab to tease next video

---

### Scene 6: Close (3:15 - 3:30)

**[SCREEN: SAEs panel showing the trained SAE]**

**VOICEOVER:**
> "That's SAE training in miStudio. Next, we'll extract features and see what interpretable concepts the SAE discovered."

**ACTION:** End on SAEs panel

---

## Key Points to Hit

- Dataset + Model must match (same tokenizer)
- Hook point determines what activations you're decomposing
- L1 coefficient is the main sparsity lever
- More steps = better quality (but slower)
- Real-time metrics help you catch problems early

---

## If Training Takes Too Long

For recording, you can either:
1. Use a pre-trained SAE and just show the config + start
2. Cut to a completed training (jump cut)
3. Speed up the middle section in editing

---

## YouTube Description

```
Train a Sparse Autoencoder (SAE) locally with miStudio!

In this video:
- Configure SAE architecture and hook point
- Set hyperparameters (dictionary size, L1, learning rate)
- Monitor training in real-time
- View your trained SAE in the library

Prerequisites:
- Model downloaded (see: [link to model video])
- Dataset tokenized (see: [link to dataset video])

Next: Extract and explore features from your trained SAE

#MechanisticInterpretability #SAE #miStudio
```
