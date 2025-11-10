# Steering-Aware Prompting: Comparison & Examples

## Problem with Current Approach

The current prompting strategy shows only **aggregated token statistics** without context:

```
TOKEN                | COUNT | AVG_ACT | MAX_ACT
---------------------|-------|---------|--------
'▁not'               |     3 |    0.00 |    0.00
'▁never'             |     2 |    0.00 |    0.00
```

**Limitations:**
- ❌ No context about WHERE/WHEN feature activates
- ❌ Doesn't explain CAUSAL effects of steering
- ❌ Only provides a label, not steering guidance
- ❌ Loses important activation magnitude information

## Improved Steering-Aware Approach

### Key Enhancements

1. **Activation Context Examples**
   - Show 5-8 real text snippets where feature fires strongly
   - Highlight the specific token that triggered activation
   - Include activation magnitude for each example

2. **Steering Questions**
   - Explicitly ask: "What happens if we AMPLIFY this 3-5x?"
   - Explicitly ask: "What happens if we SUPPRESS to ~0?"
   - Request causal/behavioral descriptions

3. **Structured Output**
   - JSON format with label + steering descriptions
   - Machine-parseable for UI display
   - Includes both amplification and suppression effects

---

## Example 1: Negation Feature

### Current Prompt Output
```
Label: "negation"
```

### Improved Prompt Output
```json
{
  "label": "negation",
  "amplify_effect": "Model will insert more negative constructions (not, never, no), increase contradictions and refutations, more likely to express disagreement or denial",
  "suppress_effect": "Model will avoid negative statements, reduce use of 'not/never/no', produce more affirmative/positive language, may struggle with expressing disagreement",
  "steering_description": "Amplifying this feature will make the model more likely to use negative constructions and express denial or disagreement"
}
```

**Steering UI Display:**
```
Feature: negation
├─ Amplify (3x): → More negative constructions, contradictions, refutations
└─ Suppress (0.1x): → Avoid negations, more affirmative language
```

---

## Example 2: Political Terms Feature

### With Context Examples

**Activation Contexts:**
```
1. (activation=0.89) "The [Trump] administration announced new policies..."
2. (activation=0.76) "Senator [Clinton] voted against the bill..."
3. (activation=0.71) "The [Republican] party's stance on..."
4. (activation=0.68) "[Congress] passed legislation regarding..."
5. (activation=0.61) "The [Democratic] candidate emphasized..."
```

**Steering Analysis:**
```json
{
  "label": "political_entities",
  "amplify_effect": "Increase references to politicians, parties, and political institutions; more likely to discuss political topics, governance, and policy",
  "suppress_effect": "Reduce or eliminate mentions of political figures and parties; avoid political framing of topics; more neutral/apolitical language",
  "steering_description": "Amplifying this feature will make the model focus more on political figures, parties, and governmental topics"
}
```

---

## Example 3: Dialogue Feature

**Activation Contexts:**
```
1. (activation=0.92) "She [said], 'I don't think that's right.'"
2. (activation=0.88) "He [replied], 'That makes sense to me.'"
3. (activation=0.81) "'What do you mean?' she [asked]."
4. (activation=0.76) "The teacher [explained], 'Let me show you...'"
5. (activation=0.73) "'I can't believe it,' John [whispered]."
```

**Steering Analysis:**
```json
{
  "label": "dialogue_markers",
  "amplify_effect": "Generate more quoted speech, dialogue tags (said/asked/replied), conversational exchanges, narrative dialogue structure",
  "suppress_effect": "Reduce or eliminate direct quotes and dialogue; produce more exposition and description; fewer conversational exchanges",
  "steering_description": "Amplifying this feature will make the model generate more direct dialogue and quoted speech"
}
```

---

## Implementation Strategy

### Phase 1: Enhanced Data Collection
```python
# Instead of just token stats, collect:
{
    "token_stats": {...},
    "text_examples": [
        {
            "text": "The Trump administration announced...",
            "activation": 0.89,
            "highlighted_token": "Trump",
            "position": 4,
            "sequence_length": 20
        },
        # ... more examples
    ],
    "activation_distribution": {
        "mean": 0.12,
        "std": 0.08,
        "max": 0.95,
        "percentile_90": 0.28
    }
}
```

### Phase 2: Improved Prompt Construction
```python
prompt = build_steering_aware_prompt(
    token_stats=aggregated_stats,
    text_examples=top_activation_contexts,
    top_k=50
)

# Prompt now includes:
# - Token statistics (current)
# - Real text examples with context
# - Steering-focused questions
# - Structured JSON output request
```

### Phase 3: Store Steering Metadata
```python
# Database schema addition
class Feature:
    # ... existing fields ...
    steering_amplify_effect: str  # What happens when amplified
    steering_suppress_effect: str # What happens when suppressed
    steering_description: str     # UI-friendly steering guidance
    steering_safe_range: dict     # {"min": 0.1, "max": 5.0}
```

---

## Benefits for Steering Applications

### 1. **Better Interpretability**
- Users understand what feature controls
- Clear cause-effect relationships
- Context examples validate the label

### 2. **Safer Steering**
- Explicit warnings about suppression effects
- Understanding of amplification consequences
- Helps avoid unintended side effects

### 3. **UI Integration**
```
┌─────────────────────────────────────────┐
│ Feature #1234: "negation"              │
├─────────────────────────────────────────┤
│ Activation: ▓▓▓▓▓░░░░░ 45%            │
│                                         │
│ 🎚️  Steering Control                   │
│ ├─ [0.0] [0.5] [1.0] [2.0] [5.0]      │
│ └─ Current: 1.0x (baseline)            │
│                                         │
│ 📊 Effects                              │
│ ├─ Amplify (3x):                       │
│ │   More negative constructions        │
│ │   Increased contradictions           │
│ │                                       │
│ └─ Suppress (0.1x):                    │
│     Avoid negations                    │
│     More affirmative language          │
└─────────────────────────────────────────┘
```

### 4. **Research Applications**
- Better feature documentation
- Causal analysis of features
- Hypothesis generation for interventions
- Comparative steering studies

---

## Cost Analysis

**Current Prompt:**
- ~500 tokens per feature
- Output: 1-3 tokens (label only)

**Improved Prompt:**
- ~1,200 tokens per feature (+140% input)
- Output: ~100 tokens (JSON with descriptions)

**Cost Impact:**
- Input: $0.150/1M tokens → ~$0.00018 per feature
- Output: $0.600/1M tokens → ~$0.00006 per feature
- **Total: ~$0.00024 per feature** (vs $0.00010 current)

For 44,745 features: **$10.74** (vs $4.47 current)
**Extra cost: $6.27 for steering metadata**

---

## Recommendation

**Implement as optional labeling mode:**

```python
class LabelingMethod(str, Enum):
    OPENAI = "openai"
    OPENAI_STEERING = "openai_steering"  # New mode
    LOCAL = "local"
```

**UI Selection:**
```
┌─────────────────────────────────────┐
│ Labeling Method:                    │
│ ⚪ OpenAI (Standard)                │
│    Fast, cost-effective             │
│    Label only                       │
│                                     │
│ ⚫ OpenAI (Steering-Aware)          │
│    More detailed analysis           │
│    Includes steering guidance       │
│    +2.4x cost                       │
│                                     │
│ ⚪ Local LLM                        │
└─────────────────────────────────────┘
```

This gives users choice based on their needs:
- **Standard:** Quick labels for exploration
- **Steering-Aware:** Detailed analysis for intervention research
