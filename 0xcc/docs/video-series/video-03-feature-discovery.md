# Video: Extract Features from SAE

**Channel:** HITSAI / miStudio
**Title:** "miStudio: Extract and Browse Features"
**Target Duration:** 3-4 minutes
**Prerequisites:** Trained SAE

---

## Pre-Recording Checklist

- [ ] At least one trained SAE available
- [ ] Dataset used for training still available
- [ ] No active extraction jobs
- [ ] GPU available

---

## Script

### Scene 1: Opening (0:00 - 0:15)

**[SCREEN: Features panel]**

**VOICEOVER:**
> "Your SAE is trained. Now let's find out what features it actually learned. We'll extract activations and browse through what the SAE discovered."

**ACTION:** Show Features panel

---

### Scene 2: Start Extraction (0:15 - 0:50)

**[SCREEN: Click to start extraction]**

**VOICEOVER:**
> "Select your trained SAE and click Extract Features.
>
> This runs your dataset through the model and SAE, recording which features activate for each input. We collect the top activating examples for each feature."

**ACTIONS:**
1. Select SAE from dropdown
2. Click "Extract Features" or equivalent button
3. Show extraction progress bar

---

### Scene 3: Extraction Progress (0:50 - 1:20)

**[SCREEN: Extraction running]**

**VOICEOVER:**
> "The extraction processes your dataset batch by batch. For each feature, we're finding the inputs that activate it most strongly.
>
> This gives us concrete examples of what each feature responds to."

**ACTIONS:**
1. Show progress updating
2. Point to samples processed count
3. Brief GPU utilization check

---

### Scene 4: Browse Features (1:20 - 2:30)

**[SCREEN: Feature browser with results]**

**VOICEOVER:**
> "Extraction complete. Now we can browse features.
>
> Each feature shows its top activating examples. Look at this one - see how the highlighted tokens share a common theme? That's the concept this feature has learned to detect.
>
> You can click through features, see activation strengths, and start to understand what your model represents internally."

**ACTIONS:**
1. Show feature list/grid
2. Click on a feature with clear pattern
3. Point to highlighted tokens in examples
4. Show activation strength values
5. Click through 2-3 more features
6. Show search/filter if available

---

### Scene 5: Feature Statistics (2:30 - 3:00)

**[SCREEN: Feature details or statistics view]**

**VOICEOVER:**
> "Each feature has statistics - how often it activates, its average activation strength, and the distribution across your dataset.
>
> Dead features that never activate might indicate your L1 was too high. Features that activate on everything might be too general."

**ACTIONS:**
1. Show feature activation frequency
2. Point to any dead features (if visible)
3. Show highly active features

---

### Scene 6: Close (3:00 - 3:20)

**[SCREEN: Feature browser]**

**VOICEOVER:**
> "You've now got interpretable features from your SAE. Browsing manually works, but with thousands of features, you'll want automatic labeling. That's next."

**ACTION:** End on feature browser

---

## Key Points to Hit

- Extraction finds top activating examples per feature
- Highlighted tokens show what triggers the feature
- Look for patterns across examples
- Dead features = too sparse, always-on = too dense
- Manual browsing doesn't scale - need auto-labeling

---

## Feature Examples to Look For

Good features to demo (if you can find them):
- Code-related (brackets, keywords)
- Punctuation patterns
- Number formats
- Common words/phrases
- Domain-specific terms

---

## YouTube Description

```
Extract and browse interpretable features from your trained SAE!

In this video:
- Run feature extraction on your dataset
- Browse top activating examples per feature
- Understand what patterns each feature detects
- View feature statistics and activation distributions

Prerequisites:
- Trained SAE (see: [link to training video])

Next: Auto-label features with an LLM

#MechanisticInterpretability #SAE #miStudio #FeatureDiscovery
```
