# Video: Export SAE to HuggingFace

**Channel:** HITSAI / miStudio
**Title:** "miStudio: Export SAE to HuggingFace"
**Target Duration:** 2-3 minutes
**Prerequisites:** Trained SAE with labels

---

## Pre-Recording Checklist

- [ ] Trained SAE with labeled features
- [ ] HuggingFace account and write token ready
- [ ] Target repository name decided
- [ ] Good feature examples to show in export preview

---

## Script

### Scene 1: Opening (0:00 - 0:10)

**[SCREEN: SAEs panel]**

**VOICEOVER:**
> "You've trained and validated your SAE. Now let's share it with the community by uploading to HuggingFace."

**ACTION:** Show SAEs panel with your trained SAE

---

### Scene 2: Start Export (0:10 - 0:40)

**[SCREEN: Export dialog/options]**

**VOICEOVER:**
> "Select your SAE and click Export. miStudio packages everything in SAELens-compatible format - the community standard.
>
> This includes the model weights, configuration, and optionally your feature labels."

**ACTIONS:**
1. Select SAE
2. Click Export button
3. Show export options (include labels, repo name, etc.)
4. Enter repository name

---

### Scene 3: Configure Upload (0:40 - 1:15)

**[SCREEN: HuggingFace configuration]**

**VOICEOVER:**
> "Enter your HuggingFace token with write access. Choose your repository name - something descriptive like 'tinyllama-pile-sae-8x'.
>
> You can set visibility to public or private. Public SAEs help the research community build on your work."

**ACTIONS:**
1. Show token input (or show it's already configured)
2. Enter repository name
3. Select public/private
4. Show any metadata options (description, tags)

---

### Scene 4: Upload (1:15 - 1:45)

**[SCREEN: Upload progress]**

**VOICEOVER:**
> "Click Upload. The SAE weights, config, and labels are packaged and pushed to HuggingFace.
>
> Larger SAEs take longer - a typical 8x expansion for a small model is just a few hundred megabytes."

**ACTIONS:**
1. Click Upload/Export button
2. Show progress indicator
3. Wait for completion (or cut if slow)

---

### Scene 5: Verify on HuggingFace (1:45 - 2:20)

**[SCREEN: HuggingFace repo page]**

**VOICEOVER:**
> "Done. Let's verify on HuggingFace. Here's our new repository with the SAE files.
>
> Anyone can now download this SAE using SAELens or miStudio and explore the features you discovered."

**ACTIONS:**
1. Open HuggingFace in browser
2. Navigate to the new repo
3. Show files (cfg.json, model weights)
4. Show README if generated

---

### Scene 6: Close (2:20 - 2:40)

**[SCREEN: HuggingFace repo or miStudio]**

**VOICEOVER:**
> "That's the complete miStudio workflow - from raw data to a published, interpretable SAE. Train locally, discover features, and share with the world."

**ACTION:** End on HuggingFace repo or miStudio interface

---

## Key Points to Hit

- SAELens format is the community standard
- Includes weights, config, and labels
- HuggingFace token needs write permission
- Public repos help the MI community
- Others can load your SAE in SAELens or miStudio

---

## Export Includes

- `cfg.json` - SAE configuration
- `sae_weights.safetensors` - Model weights
- `sparsity.safetensors` - Feature sparsity statistics
- `feature_labels.json` - Your generated labels (optional)
- `README.md` - Auto-generated model card

---

## YouTube Description

```
Export your trained SAE to HuggingFace with miStudio!

In this video:
- Package SAE in SAELens-compatible format
- Upload to HuggingFace Hub
- Share your interpretability research with the community

Prerequisites:
- Trained SAE (see: [link to training video])
- HuggingFace account with write token

This completes the miStudio workflow:
1. Download dataset & model
2. Train SAE
3. Extract features
4. Auto-label features
5. Steer model behavior
6. Export & share

#MechanisticInterpretability #SAE #miStudio #HuggingFace
```
