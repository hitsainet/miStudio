# Real Example: Steering-Aware Prompt for "Political Terms" Feature

## Current Prompt (What Was Used)

```
Analyze this sparse autoencoder neuron's activation pattern.

Top tokens that activate this neuron:

TOKEN                | COUNT | AVG_ACT | MAX_ACT
---------------------|-------|---------|--------
'▁Trump'             |    45 |   0.023 |   0.089
'▁Clinton'           |    32 |   0.019 |   0.076
'▁Republican'        |    28 |   0.018 |   0.071
'▁Democratic'        |    26 |   0.017 |   0.068
'▁Congress'          |    24 |   0.016 |   0.065
'▁Senate'            |    22 |   0.015 |   0.062
'▁President'         |    20 |   0.014 |   0.058
'▁election'          |    18 |   0.013 |   0.055
...

What single concept does this neuron represent?
Respond with ONLY one word or short phrase (max 3 words).

Concept:
```

**GPT Response:** `political_terms`

---

## Improved Steering-Aware Prompt

```
You are analyzing a sparse autoencoder feature for mechanistic interpretability and model steering.

FEATURE ACTIVATION STATISTICS:
================================

Top tokens that activate this feature (sorted by total activation strength):

TOKEN                | COUNT | AVG_ACT | MAX_ACT | PERCENTILE
---------------------|-------|---------|---------|------------
'▁Trump'             |    45 |   0.023 |   0.089 |   100.0%
'▁Clinton'           |    32 |   0.019 |   0.076 |    95.0%
'▁Republican'        |    28 |   0.018 |   0.071 |    90.0%
'▁Democratic'        |    26 |   0.017 |   0.068 |    85.0%
'▁Congress'          |    24 |   0.016 |   0.065 |    80.0%
'▁Senate'            |    22 |   0.015 |   0.062 |    75.0%
'▁President'         |    20 |   0.014 |   0.058 |    70.0%
'▁election'          |    18 |   0.013 |   0.055 |    65.0%
'▁party'             |    16 |   0.012 |   0.052 |    60.0%
'▁vote'              |    15 |   0.011 |   0.048 |    55.0%
'▁policy'            |    14 |   0.011 |   0.046 |    50.0%
'▁House'             |    13 |   0.010 |   0.043 |    45.0%
'▁bill'              |    12 |   0.010 |   0.041 |    40.0%
'▁political'         |    11 |   0.009 |   0.039 |    35.0%
'▁campaign'          |    10 |   0.009 |   0.037 |    30.0%

ACTIVATION CONTEXTS:
====================

Here are real text examples where this feature activates strongly:
(The token triggering the activation is shown in [BRACKETS])

1. (activation=0.089) "The [Trump] administration announced new tariff policies on Chinese imports yesterday."

2. (activation=0.076) "Senator [Clinton] voted against the healthcare reform bill in a narrow decision."

3. (activation=0.071) "The [Republican] party's stance on immigration has evolved significantly over the past decade."

4. (activation=0.068) "[Democratic] lawmakers introduced legislation aimed at expanding voting rights nationwide."

5. (activation=0.065) "[Congress] passed the infrastructure bill with bipartisan support after months of negotiation."

6. (activation=0.062) "The [Senate] held hearings on Supreme Court nominee confirmation this week."

7. (activation=0.058) "The [President] addressed the nation regarding foreign policy decisions in the Middle East."

8. (activation=0.055) "The upcoming [election] will determine control of both chambers and shape policy for years."

STEERING ANALYSIS QUESTIONS:
==============================

Analyze this feature from a model steering perspective:

1. CONCEPT LABEL: What single concept or pattern does this feature represent?
   (1-3 words, e.g., "negation", "past_tense", "political_entities")

2. AMPLIFICATION EFFECT: If you INCREASE this feature's activation by 3-5x:
   - What behaviors or outputs would become MORE likely?
   - What semantic/syntactic patterns would be emphasized?

3. SUPPRESSION EFFECT: If you DECREASE this feature's activation to near-zero:
   - What behaviors or outputs would become LESS likely?
   - What would be removed or diminished from the model's outputs?

4. STEERING DESCRIPTION: Write a 1-sentence description suitable for a steering interface:
   "Amplifying this feature will make the model..."

Respond in this EXACT JSON format:
{
  "label": "your_concept_label",
  "amplify_effect": "concise description of amplification effects",
  "suppress_effect": "concise description of suppression effects",
  "steering_description": "one sentence for UI display"
}

IMPORTANT: Respond with ONLY valid JSON, no other text.
```

---

## Expected GPT-4o-mini Response

```json
{
  "label": "political_entities",
  "amplify_effect": "Dramatically increase mentions of politicians (Trump, Clinton), political parties (Republican, Democratic), governmental bodies (Congress, Senate, President), and political processes (elections, bills, campaigns). Text will become heavily focused on political topics, governance, policy debates, and partisan framing. May insert political references even in non-political contexts.",
  "suppress_effect": "Eliminate or drastically reduce references to political figures, parties, and governmental institutions. Text will avoid political framing, use more neutral descriptors for actors and events, and shift focus away from governance and policy topics toward other domains. May struggle to discuss inherently political topics.",
  "steering_description": "Amplifying this feature will make the model focus heavily on political entities, governmental processes, and partisan topics while suppressing it creates more apolitical, neutral content"
}
```

---

## Steering Interface Mockup

```
┌────────────────────────────────────────────────────────────────┐
│ Feature #31188: "political_entities"                          │
├────────────────────────────────────────────────────────────────┤
│ Current Activation: ▓▓▓▓▓▓░░░░ 58%                           │
│                                                                │
│ 🎚️  Steering Control                                          │
│ ├─ Suppress  [0.0] [0.5] [●1.0] [2.0] [5.0]  Amplify        │
│ └─ Recommended safe range: 0.2x - 3.0x                        │
│                                                                │
│ 📊 Steering Effects Preview                                   │
│                                                                │
│ ✨ Amplify to 3.0x:                                           │
│    • Heavy focus on politicians & political parties           │
│    • More mentions: Trump, Clinton, Republican, Democratic    │
│    • Increased political framing of topics                    │
│    • More governance & policy discussion                      │
│    ⚠️  May inject politics into non-political contexts        │
│                                                                │
│ 🔇 Suppress to 0.2x:                                          │
│    • Remove political figures & party references              │
│    • More neutral, apolitical language                        │
│    • Shift away from governance topics                        │
│    • Less partisan framing                                    │
│    ⚠️  May struggle with inherently political topics          │
│                                                                │
│ 💡 Use Cases:                                                 │
│    • Bias analysis (amplify to expose political bias)         │
│    • Content neutralization (suppress for apolitical tone)    │
│    • Domain adaptation (suppress for non-political writing)   │
└────────────────────────────────────────────────────────────────┘
```

---

## Key Improvements Summary

### 1. **Richer Input Context**
- Real text examples (not just token stats)
- Shows WHERE and HOW feature activates
- Activation magnitudes for each example

### 2. **Causal Reasoning**
- Explicit questions about amplification effects
- Explicit questions about suppression effects
- Forces model to think causally about interventions

### 3. **Actionable Output**
- Structured JSON (machine-parseable)
- Steering descriptions ready for UI
- Warning/guidance information included

### 4. **Better for Research**
- Documents causal effects
- Enables hypothesis generation
- Supports intervention planning
- Validates interpretability claims

---

## Implementation Roadmap

### Phase 1: Data Collection Enhancement
```python
# Modify feature_activations to store text contexts
ALTER TABLE feature_activations ADD COLUMN text_context TEXT;
ALTER TABLE feature_activations ADD COLUMN token_position INT;
```

### Phase 2: New Labeling Service
```python
class OpenAISteeringLabelingService(OpenAILabelingService):
    """Enhanced labeling with steering awareness."""

    def _build_prompt(self, sorted_tokens, text_examples):
        return build_steering_aware_prompt(
            token_stats=dict(sorted_tokens),
            text_examples=text_examples,
            top_k=50
        )

    def _parse_response(self, response):
        # Parse JSON instead of single label
        data = json.loads(response)
        return {
            "label": data["label"],
            "steering_metadata": {
                "amplify_effect": data["amplify_effect"],
                "suppress_effect": data["suppress_effect"],
                "description": data["steering_description"]
            }
        }
```

### Phase 3: Database Schema Update
```python
# Add steering metadata to features table
class Feature(Base):
    # ... existing fields ...
    steering_amplify_effect = Column(Text, nullable=True)
    steering_suppress_effect = Column(Text, nullable=True)
    steering_description = Column(Text, nullable=True)
    steering_confidence = Column(Float, nullable=True)
```

### Phase 4: UI Integration
- Steering panel with slider controls
- Effect previews based on metadata
- Warning system for risky interventions
- Save/load steering configurations
