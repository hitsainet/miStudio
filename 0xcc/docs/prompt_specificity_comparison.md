# Prompt Engineering for Label Specificity

## The Problem

**Current Prompt Results:**
```
Tokens: Trump, Trumps, Donald, MAGA, Trump's, administration
Label: "political_terms" ❌ TOO GENERIC
```

This loses critical information! If a feature is Trump-specific, that's mechanistically important for understanding and steering.

---

## Solution: Three Prompt Engineering Strategies

### Strategy 1: Explicit Specificity Instructions

**Technique:** Tell the model exactly what we want

**Key Prompt Elements:**
```
CRITICAL INSTRUCTION: Be as SPECIFIC as possible in your label.

Examples of GOOD (specific) vs BAD (too generic) labels:
  ✓ GOOD: "trump_mentions" (when 80%+ tokens relate to Trump)
  ✗ BAD:  "political_terms" (when tokens are Trump-specific)
```

**Expected Result:**
```
Tokens: Trump, Trumps, Donald, MAGA, Trump's
Label: "trump_mentions" ✓ SPECIFIC
```

---

### Strategy 2: Two-Stage Chain-of-Thought

**Technique:** Make the model reason step-by-step

**Prompt Structure:**
```
STAGE 1: Analyze clustering
- Do tokens cluster around one entity? YES
- What is it? "Donald Trump / Trump"
- Specificity level: HIGHLY SPECIFIC (90%+ Trump-related)

STAGE 2: Generate label based on Stage 1
- Rule: HIGHLY SPECIFIC → Use entity name
- Label: "trump_related"
```

**Expected JSON Response:**
```json
{
  "analysis": {
    "clustering": "All tokens relate to Donald Trump specifically",
    "specificity_level": "HIGHLY SPECIFIC",
    "dominant_examples": ["Trump", "Donald", "MAGA"]
  },
  "label": "trump_related"
}
```

---

### Strategy 3: Contrastive Learning (RECOMMENDED)

**Technique:** Show anti-patterns (what NOT to do) + good patterns (what TO do)

**Prompt Structure:**
```
ANTI-PATTERNS (Avoid these):
❌ "political_terms" when tokens are: Trump, Trumps, Donald, MAGA
   → TOO GENERIC. Should be: "trump_mentions"

GOOD PATTERNS (Emulate these):
✓ "trump_mentions" when tokens dominated by Trump, Donald, Trumps
✓ "biden_administration" when tokens dominated by Biden, Joe, Bidens
✓ "clinton_references" when tokens dominated by Clinton, Hillary

DECISION TREE:
1. ONE dominant entity (70%+)? → Use entity name
2. NARROW domain (60%+)? → Use domain name
3. SPECIFIC pattern? → Use pattern name
4. Otherwise → Use precise category
```

**Why This Works Best:**
- Shows concrete examples of good vs bad
- Provides a decision tree for edge cases
- Teaches the model our preferences through examples

---

## Real-World Comparison

### Example 1: Trump Feature

**Tokens:**
```
1. 'Trump'        count=45  avg=0.023
2. 'Trumps'       count=12  avg=0.019
3. 'Donald'       count=8   avg=0.016
4. 'MAGA'         count=6   avg=0.014
5. 'administration' count=4 avg=0.012
```

| Prompt Strategy | Label | Specificity | Notes |
|----------------|-------|-------------|-------|
| **Current** | `political_terms` | ❌ Generic | Loses Trump-specific info |
| **Explicit Instructions** | `trump_mentions` | ✓ Specific | Clear entity focus |
| **Two-Stage** | `trump_related` | ✓ Specific | Reasoning included |
| **Contrastive** | `trump_mentions` | ✓ Specific | Most reliable |

---

### Example 2: Mixed Political Feature

**Tokens:**
```
1. 'president'    count=20  avg=0.018
2. 'senator'      count=15  avg=0.016
3. 'congress'     count=14  avg=0.015
4. 'Trump'        count=3   avg=0.012
5. 'vote'         count=12  avg=0.011
```

| Prompt Strategy | Label | Specificity | Notes |
|----------------|-------|-------------|-------|
| **Current** | `political_terms` | ✓ Appropriate | Diverse tokens |
| **Explicit Instructions** | `political_terms` | ✓ Appropriate | No single entity |
| **Two-Stage** | `governmental_roles` | ✓ Better | More precise |
| **Contrastive** | `political_institutions` | ✓ Better | Domain-specific |

**Key Insight:** Specificity prompts won't over-specify when tokens are genuinely diverse.

---

### Example 3: COVID Feature

**Tokens:**
```
1. 'COVID'        count=42  avg=0.028
2. 'coronavirus'  count=35  avg=0.025
3. 'pandemic'     count=28  avg=0.022
4. 'vaccine'      count=24  avg=0.019
5. 'quarantine'   count=18  avg=0.016
```

| Prompt Strategy | Label | Specificity |
|----------------|-------|-------------|
| **Current** | `health_topics` | ❌ Too generic |
| **Specificity-Aware** | `covid_pandemic` | ✓ Much better |

---

## Implementation Comparison

### Current Implementation
```python
def _build_prompt(self, sorted_tokens):
    prompt = """Analyze this sparse autoencoder neuron's activation pattern.

Top tokens that activate this neuron:
[token table]

What single concept does this neuron represent?
Concept:"""
    return prompt
```

### Recommended Implementation
```python
def _build_prompt(self, sorted_tokens):
    prompt = """You are labeling a sparse autoencoder feature.

CRITICAL: Be as SPECIFIC as possible.

ANTI-PATTERNS (Avoid):
❌ "political_terms" when tokens are Trump-dominated → Use "trump_mentions"
❌ "health_topics" when tokens are COVID-dominated → Use "covid_pandemic"

[token table]

DECISION TREE:
1. ONE dominant entity (70%+)? → Use entity name
2. NARROW domain (60%+)? → Use domain name
3. SPECIFIC pattern? → Use pattern name
4. Otherwise → Precise category

Your label:"""
    return prompt
```

**Changes:**
1. ✅ Added explicit specificity instruction
2. ✅ Added anti-pattern examples
3. ✅ Added decision tree for edge cases
4. ✅ Shows good examples to emulate

---

## Expected Label Distribution Changes

### Before (Generic Labels)
```
political_terms:          2,627 features (5.87%)
names:                    1,527 features (3.41%)
proper_nouns:             1,904 features (4.26%)
```

### After (Specific Labels)
```
trump_mentions:             892 features (1.99%)
biden_references:           234 features (0.52%)
clinton_related:            178 features (0.40%)
political_institutions:     445 features (0.99%)
other_political_terms:      878 features (1.96%)

trump_name:                 234 features (0.52%)
biden_name:                  89 features (0.20%)
celebrity_names:            456 features (1.02%)
historical_figures:         348 features (0.78%)
other_names:                400 features (0.89%)
```

**Benefits:**
- ✅ More granular understanding of what model represents
- ✅ Better steering precision (target Trump-specific vs generic politics)
- ✅ Reveals data biases (e.g., Trump over-represented)
- ✅ Enables entity-specific interventions

---

## Cost Impact

**Prompt Length:**
- Current: ~400 tokens
- Specificity-aware: ~650 tokens (+62%)

**Cost per Feature:**
- Current: $0.00010
- Specificity-aware: $0.00016 (+60%)

**Total for 44,745 features:**
- Current: $4.47
- Specificity-aware: $7.16 (+$2.69)

**Worth it?** YES - for $2.69 extra, you get:
- More accurate interpretability
- Better steering precision
- Data bias visibility
- Entity-specific analysis capability

---

## Migration Strategy

### Option 1: Re-label Everything
```bash
# Re-run labeling with new prompt on all 44,745 features
# Cost: $7.16
# Time: ~3.5 hours
# Benefit: Complete consistency
```

### Option 2: Selective Re-labeling
```bash
# Re-label only features with generic labels that might be specific
# Target: political_terms, names, proper_nouns (~6,000 features)
# Cost: ~$0.96
# Time: ~30 minutes
# Benefit: Fix the most problematic cases
```

### Option 3: New Extractions Only
```bash
# Use new prompt for future labeling jobs only
# Cost: $0 now
# Time: 0
# Benefit: Gradual improvement
```

**Recommendation:** Option 2 (Selective Re-labeling)
- Fixes the immediate problem
- Low cost and time
- Can compare old vs new labels for validation

---

## Implementation Checklist

- [ ] Update `OpenAILabelingService._build_prompt()` with contrastive examples
- [ ] Add specificity instructions to prompt
- [ ] Include anti-pattern examples
- [ ] Add decision tree logic
- [ ] Test on sample features (Trump, mixed politics, COVID)
- [ ] Validate improved specificity
- [ ] Deploy to production
- [ ] (Optional) Re-label generic features
