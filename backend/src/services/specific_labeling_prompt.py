"""
Specificity-aware prompting for feature labeling.

Uses prompt engineering to encourage the LLM to provide maximally specific labels
rather than generic category labels.

Key principle: If tokens cluster around a specific entity/concept, the label should
reflect that specificity (e.g., "trump_mentions" not "political_terms").
"""

from typing import Dict, List


def build_specificity_aware_prompt(
    sorted_tokens: List[tuple],  # (token, stats_dict)
    top_k: int = 50
) -> str:
    """
    Build prompt that explicitly encourages specificity in labeling.

    Prompt engineering techniques used:
    1. Explicit instruction to be as specific as possible
    2. Contrasting examples (specific vs generic)
    3. Multi-step reasoning (identify pattern, then assess specificity)
    4. Self-verification questions
    """

    prompt = """You are labeling a sparse autoencoder feature for mechanistic interpretability.

CRITICAL INSTRUCTION: Be as SPECIFIC as possible in your label. If tokens cluster around
a particular entity, person, concept, or narrow domain, use a SPECIFIC label rather than
a generic category label.

Examples of GOOD (specific) vs BAD (too generic) labels:
  ✓ GOOD: "trump_mentions" (when 80%+ tokens relate to Trump specifically)
  ✗ BAD:  "political_terms" (when tokens are actually Trump-specific)

  ✓ GOOD: "covid_pandemic" (when tokens are pandemic-specific)
  ✗ BAD:  "health_topics" (when tokens are actually COVID-specific)

  ✓ GOOD: "shakespeare_plays" (when tokens are Shakespeare-specific)
  ✗ BAD:  "literature" (when tokens are actually Shakespeare-specific)

  ✓ GOOD: "python_syntax" (when tokens are Python code keywords)
  ✗ BAD:  "programming" (when tokens are Python-specific)

TOP ACTIVATING TOKENS:
=======================

"""

    # Add token statistics
    for rank, (token, stats) in enumerate(sorted_tokens[:30], 1):
        avg_act = stats["total_activation"] / stats["count"] if stats["count"] > 0 else 0
        token_display = repr(token)[:20].ljust(20)
        prompt += f"{rank:2d}. {token_display} | count={stats['count']:4d} | avg={avg_act:6.3f} | max={stats['max_activation']:6.3f}\n"

    prompt += """

LABELING INSTRUCTIONS:
======================

Step 1: PATTERN IDENTIFICATION
Examine the top tokens carefully. Ask yourself:
- Do these tokens cluster around a SPECIFIC entity/person/concept?
- Or do they represent a BROAD category?

Step 2: SPECIFICITY CHECK
Answer these questions:
a) Is there a dominant entity/person? (e.g., 70%+ tokens relate to one thing)
b) Is there a narrow domain? (e.g., COVID pandemic, not general health)
c) Is there a specific style/format? (e.g., Python code, not general programming)

Step 3: LABEL SELECTION
- If YES to (a): Label should name the specific entity (e.g., "trump_mentions")
- If YES to (b): Label should name the narrow domain (e.g., "covid_pandemic")
- If YES to (c): Label should name the specific format (e.g., "python_keywords")
- If NO to all: Use a broader category label (but still be precise)

Step 4: VERIFICATION
Ask yourself: "If I showed these tokens to someone else, would they arrive at the
SAME SPECIFIC label I chose, or would they use a more generic one?"
- If they'd use the same specific label → your label is good
- If they'd use a more generic label → you need to be MORE specific

RESPONSE FORMAT:
================

Respond with ONLY a single label using lowercase_with_underscores.

DO NOT use overly generic labels like:
- "political_terms" (unless truly diverse political topics)
- "names" (unless truly diverse names)
- "function_words" (unless truly generic function words)

DO use specific labels like:
- "trump_related" (if dominated by Trump)
- "shakespeare_character_names" (if specific to Shakespeare)
- "covid_terminology" (if pandemic-specific)

Your label:"""

    return prompt


def build_two_stage_specificity_prompt(
    sorted_tokens: List[tuple],
    top_k: int = 50
) -> str:
    """
    Two-stage prompt: First identify pattern, then rate specificity.

    This uses chain-of-thought reasoning to improve label quality.
    """

    prompt = """You are analyzing a sparse autoencoder feature for interpretability.

Your task has TWO STAGES:

STAGE 1: PATTERN ANALYSIS
==========================

Here are the top activating tokens:

"""

    # Add token statistics
    for rank, (token, stats) in enumerate(sorted_tokens[:25], 1):
        avg_act = stats["total_activation"] / stats["count"] if stats["count"] > 0 else 0
        token_display = repr(token)[:20].ljust(20)
        prompt += f"{rank:2d}. {token_display} | count={stats['count']:3d} | avg={avg_act:5.3f}\n"

    prompt += """

First, analyze these tokens and answer:

1. CLUSTERING: Do tokens cluster around a specific entity/person/concept?
   - If yes, what is it? Name it specifically.
   - If no, what broader pattern do they share?

2. SPECIFICITY LEVEL:
   - HIGHLY SPECIFIC (90%+ tokens relate to one narrow thing)
   - MODERATELY SPECIFIC (60-90% tokens relate to a specific domain)
   - SOMEWHAT GENERIC (40-60% tokens share a broad category)
   - VERY GENERIC (<40% tokens share any clear pattern)

3. DOMINANT EXAMPLES:
   - List the 3-5 most representative tokens that define this feature

STAGE 2: LABEL GENERATION
==========================

Based on your Stage 1 analysis, generate a label with these rules:

SPECIFICITY RULES:
- HIGHLY SPECIFIC → Use a named entity label (e.g., "trump_mentions", "covid_pandemic")
- MODERATELY SPECIFIC → Use a narrow domain label (e.g., "astronomy_terms", "legal_jargon")
- SOMEWHAT GENERIC → Use a category label (e.g., "proper_nouns", "technical_terms")
- VERY GENERIC → Use a broad label (e.g., "function_words", "punctuation")

EXAMPLES:
- If tokens are: Trump, Donald, MAGA, Trumps → Label: "trump_related" (HIGHLY SPECIFIC)
- If tokens are: Trump, Biden, Clinton, Obama → Label: "us_politicians" (MODERATELY SPECIFIC)
- If tokens are: president, senator, congress, vote → Label: "political_terms" (SOMEWHAT GENERIC)
- If tokens are: the, and, of, to, in → Label: "function_words" (VERY GENERIC)

YOUR RESPONSE FORMAT:
====================

{
  "analysis": {
    "clustering": "brief description of what tokens cluster around",
    "specificity_level": "HIGHLY SPECIFIC | MODERATELY SPECIFIC | SOMEWHAT GENERIC | VERY GENERIC",
    "dominant_examples": ["token1", "token2", "token3"]
  },
  "label": "your_specific_label_here"
}

Respond with ONLY valid JSON.
"""

    return prompt


def build_contrastive_examples_prompt(
    sorted_tokens: List[tuple],
    top_k: int = 50
) -> str:
    """
    Use contrastive examples to teach specificity.

    Shows the LLM what we want vs what we don't want.
    """

    prompt = """You are labeling a sparse autoencoder feature. Your goal: MAXIMUM SPECIFICITY.

ANTI-PATTERNS (Labels we want to AVOID):
=========================================

❌ "political_terms" when tokens are: Trump, Trumps, Donald, MAGA, Trump's
   → This is TOO GENERIC. Should be: "trump_mentions"

❌ "names" when tokens are: Elizabeth, Lizzie, Liz, Beth, Betty
   → This is TOO GENERIC. Should be: "elizabeth_variations"

❌ "punctuation" when tokens are: ", ', ", ", ', '
   → This could be MORE SPECIFIC. Should be: "quotation_marks"

❌ "function_words" when tokens are: don, didn, wouldn, couldn, shouldn
   → This could be MORE SPECIFIC. Should be: "negative_contractions"

GOOD PATTERNS (Labels we want to EMULATE):
===========================================

✓ "trump_mentions" when tokens are dominated by Trump, Donald, Trumps
✓ "covid_pandemic" when tokens are: COVID, coronavirus, pandemic, vaccine
✓ "python_keywords" when tokens are: def, class, import, return
✓ "shakespeare_plays" when tokens are: Hamlet, Romeo, Juliet, Macbeth
✓ "json_syntax" when tokens are: {, }, [, ], ":", ","

YOUR TOKEN DATA:
================

"""

    # Add token statistics
    for rank, (token, stats) in enumerate(sorted_tokens[:20], 1):
        avg_act = stats["total_activation"] / stats["count"] if stats["count"] > 0 else 0
        token_display = repr(token)[:20].ljust(20)
        prompt += f"{token_display} | count={stats['count']:3d} | avg={avg_act:5.3f}\n"

    prompt += """

LABELING DECISION TREE:
========================

1. Is there ONE dominant entity/person/topic that 70%+ tokens relate to?
   → YES: Use that entity's name (e.g., "trump_related", "bitcoin_crypto")
   → NO: Continue to step 2

2. Is there a NARROW domain that 60%+ tokens belong to?
   → YES: Use domain name (e.g., "medical_terminology", "legal_language")
   → NO: Continue to step 3

3. Is there a SPECIFIC grammatical pattern or format?
   → YES: Use pattern name (e.g., "negative_contractions", "plural_nouns")
   → NO: Continue to step 4

4. Only now use a broader category:
   → Use precise category (e.g., "proper_nouns", "prepositions")

YOUR LABEL (lowercase_with_underscores):"""

    return prompt


# Recommended: Use the contrastive examples approach
# It teaches by showing what we want vs don't want

RECOMMENDED_PROMPT_BUILDER = build_contrastive_examples_prompt
