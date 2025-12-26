# Video: Auto-Label Features with LLM

**Channel:** HITSAI / miStudio
**Title:** "miStudio: Auto-Label Features with LLM"
**Target Duration:** 3-4 minutes
**Prerequisites:** Extracted features

---

## Pre-Recording Checklist

- [ ] Features extracted from SAE
- [ ] OpenAI API key configured (or Ollama running locally)
- [ ] Some unlabeled features available
- [ ] Good examples visible in feature browser

---

## Script

### Scene 1: Opening (0:00 - 0:15)

**[SCREEN: Feature browser with unlabeled features]**

**VOICEOVER:**
> "You've got thousands of features. Labeling them manually would take forever. Let's use an LLM to automatically generate human-readable descriptions."

**ACTION:** Show features panel with unlabeled features

---

### Scene 2: Configure Labeling (0:15 - 0:50)

**[SCREEN: Auto-labeling configuration]**

**VOICEOVER:**
> "miStudio supports two labeling backends: OpenAI's GPT-4 for best quality, or Ollama for fully local labeling with no API costs.
>
> Select your features - you can label all unlabeled features, or select specific ones. The LLM sees the top activating examples and generates a concise description."

**ACTIONS:**
1. Navigate to labeling interface
2. Show LLM provider selection (OpenAI vs Ollama)
3. Select features to label (or "all unlabeled")
4. Show any configuration options

---

### Scene 3: Start Labeling (0:50 - 1:30)

**[SCREEN: Labeling in progress]**

**VOICEOVER:**
> "Click Start Labeling. The system sends batches of examples to the LLM, which analyzes the patterns and generates labels.
>
> Each label captures what concept or pattern the feature detects - things like 'code comments', 'question marks', or 'scientific terminology'."

**ACTIONS:**
1. Click "Start Labeling" button
2. Show progress indicator
3. Watch as labels appear on features

---

### Scene 4: Review Labels (1:30 - 2:30)

**[SCREEN: Features with generated labels]**

**VOICEOVER:**
> "Labels are now attached to your features. Let's browse a few.
>
> This one is labeled 'opening parentheses in function calls' - and look at the examples, that's exactly right.
>
> Here's another: 'Python import statements'. The LLM identified the pattern across all these activating examples.
>
> You can edit labels if the LLM got it wrong, or regenerate with different examples."

**ACTIONS:**
1. Show feature with good label
2. Point to label text
3. Show corresponding examples match the label
4. Browse 2-3 more labeled features
5. Show edit label option (if available)

---

### Scene 5: Search by Label (2:30 - 3:00)

**[SCREEN: Search/filter features]**

**VOICEOVER:**
> "Now that features have labels, you can search. Looking for math-related features? Just search 'math' or 'number'. Want code features? Search 'code' or 'function'.
>
> This transforms your SAE from a black box into a searchable dictionary of concepts."

**ACTIONS:**
1. Show search box
2. Type a search term
3. Show filtered results
4. Click on a result to verify relevance

---

### Scene 6: Close (3:00 - 3:20)

**[SCREEN: Labeled feature browser]**

**VOICEOVER:**
> "Your features are now labeled and searchable. Next, let's use these features to actually steer model behavior."

**ACTION:** End on labeled features

---

## Key Points to Hit

- LLM sees examples, generates pattern description
- OpenAI = better quality, Ollama = free & local
- Labels make features searchable
- Can edit/regenerate labels
- Foundation for steering experiments

---

## Good Label Examples to Show

- Code patterns (brackets, keywords, comments)
- Punctuation (periods, question marks)
- Number formats (dates, prices, percentages)
- Domain terms (scientific, legal, medical)
- Structural patterns (list items, headers)

---

## If Using Ollama

Mention:
> "I'm using Ollama with Llama 3 running locally - completely free and private. Quality is slightly lower than GPT-4, but it works great for most features."

---

## YouTube Description

```
Automatically label SAE features using GPT-4 or local Ollama!

In this video:
- Configure LLM backend (OpenAI or Ollama)
- Run auto-labeling on extracted features
- Review and edit generated labels
- Search features by label

Prerequisites:
- Features extracted (see: [link to extraction video])
- OpenAI API key OR Ollama installed locally

Next: Steer model behavior using labeled features

#MechanisticInterpretability #SAE #miStudio #AutoLabeling
```
