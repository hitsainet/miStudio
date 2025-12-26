# Video: Steering Demo

**Channel:** HITSAI / miStudio
**Title:** "miStudio: Steer Model Behavior with Features"
**Target Duration:** 3-4 minutes
**Prerequisites:** Labeled features

---

## Pre-Recording Checklist

- [ ] SAE with labeled features ready
- [ ] Model loaded
- [ ] Found 2-3 good steering candidates (clear effect features)
- [ ] GPU available for generation

---

## Script

### Scene 1: Opening (0:00 - 0:15)

**[SCREEN: Steering panel]**

**VOICEOVER:**
> "You've trained an SAE, extracted features, and labeled them. Now the fun part - let's actually change how the model behaves by steering specific features."

**ACTION:** Show Steering panel

---

### Scene 2: Setup Steering (0:15 - 0:50)

**[SCREEN: Steering configuration]**

**VOICEOVER:**
> "Select your model and SAE. Then search for a feature to steer. I'll search for something obvious - let's try 'code' or 'python'.
>
> Found it - a feature that activates on Python code patterns. Now I can amplify or suppress this feature during generation."

**ACTIONS:**
1. Select model
2. Select SAE
3. Search for feature by label
4. Select a clear feature (code-related works well)
5. Show the feature's examples as reminder of what it does

---

### Scene 3: Baseline Generation (0:50 - 1:30)

**[SCREEN: Text generation area]**

**VOICEOVER:**
> "First, let's see baseline behavior. I'll prompt the model with something neutral that could go either direction.
>
> 'Write a tutorial about...' - without steering, the model generates normal prose."

**ACTIONS:**
1. Enter a neutral prompt
2. Generate without steering (coefficient = 0)
3. Show the baseline output
4. Point out the style/content

---

### Scene 4: Amplify Feature (1:30 - 2:15)

**[SCREEN: Adjust steering coefficient positive]**

**VOICEOVER:**
> "Now let's amplify the code feature. I'll set the coefficient to positive - say 2 or 3.
>
> Same prompt, but now the model is biased toward activating this feature. Watch what happens."

**ACTIONS:**
1. Set steering coefficient to positive (e.g., +2.0)
2. Generate with same prompt
3. Show output - should have more code-like elements
4. Highlight the difference from baseline

---

### Scene 5: Suppress Feature (2:15 - 2:50)

**[SCREEN: Adjust steering coefficient negative]**

**VOICEOVER:**
> "What about suppression? Set the coefficient negative.
>
> Now the model actively avoids activating this feature. The output should have less of whatever concept this feature represents."

**ACTIONS:**
1. Set coefficient to negative (e.g., -2.0)
2. Generate again
3. Show output - should avoid the concept
4. Compare all three: baseline, amplified, suppressed

---

### Scene 6: Close (2:50 - 3:15)

**[SCREEN: Side-by-side comparison or steering panel]**

**VOICEOVER:**
> "That's steering in action. You've gone from training an SAE to actually controlling model behavior through interpretable features.
>
> The last step is sharing your work - uploading your SAE to HuggingFace for others to use."

**ACTION:** End on steering panel or tease export

---

## Key Points to Hit

- Positive coefficient = amplify feature
- Negative coefficient = suppress feature
- Same prompt, different outputs based on steering
- This proves features are causally relevant, not just correlational
- Start with small coefficients (1-3), increase for stronger effect

---

## Good Steering Demos

Features that show clear effects:
- **Code features** - steering toward/away from code formatting
- **Question features** - making outputs more/less interrogative
- **Formal language** - shifting register
- **List/bullet features** - structured vs prose output
- **Number features** - more/fewer numerical references

---

## Troubleshooting

If steering effect is weak:
- Increase coefficient magnitude
- Try a different feature with clearer semantics
- Use prompts that are ambiguous (could go either way)
- Check feature activation examples to understand what it detects

---

## YouTube Description

```
Steer model behavior using SAE features in miStudio!

In this video:
- Select features to amplify or suppress
- Compare baseline vs steered generation
- See how positive/negative coefficients change output
- Prove causal relevance of interpretable features

Prerequisites:
- Labeled features (see: [link to labeling video])

Next: Export your SAE to HuggingFace

#MechanisticInterpretability #SAE #miStudio #ModelSteering
```
