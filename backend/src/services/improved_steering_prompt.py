"""
Improved prompting for steering-aware feature labeling.

This module provides an enhanced prompt structure that includes:
1. Top activating tokens with statistics
2. Example text contexts where feature activates strongly
3. Steering interpretation questions
4. Causal effect descriptions
"""

from typing import Dict, List, Tuple


def build_steering_aware_prompt(
    token_stats: Dict[str, Dict[str, float]],
    text_examples: List[Tuple[str, float, str]],  # (text, activation, highlighted_token)
    top_k: int = 50
) -> str:
    """
    Build prompt that emphasizes steering/intervention interpretation.

    Args:
        token_stats: Dict mapping token to {count, total_activation, max_activation}
        text_examples: List of (text_snippet, activation_value, highlighted_token) tuples
                       showing actual contexts where feature activates
        top_k: Number of top tokens to include

    Returns:
        Formatted prompt for OpenAI API
    """

    # Sort tokens by total activation strength
    sorted_tokens = sorted(
        token_stats.items(),
        key=lambda x: x[1]["total_activation"],
        reverse=True
    )[:top_k]

    prompt = """You are analyzing a sparse autoencoder feature for mechanistic interpretability and model steering.

FEATURE ACTIVATION STATISTICS:
================================

Top tokens that activate this feature (sorted by total activation strength):

TOKEN                | COUNT | AVG_ACT | MAX_ACT | PERCENTILE
---------------------|-------|---------|---------|------------
"""

    # Add token statistics with percentile ranking
    for rank, (token, stats) in enumerate(sorted_tokens[:20], 1):
        avg_act = stats["total_activation"] / stats["count"] if stats["count"] > 0 else 0
        percentile = ((top_k - rank) / top_k) * 100
        token_display = repr(token)[:20].ljust(20)
        prompt += f"{token_display} | {stats['count']:5} | {avg_act:7.3f} | {stats['max_activation']:7.3f} | {percentile:6.1f}%\n"

    prompt += """

ACTIVATION CONTEXTS:
====================

Here are real text examples where this feature activates strongly:
(The token triggering the activation is shown in [BRACKETS])

"""

    # Add text examples showing context
    for i, (text, activation, highlighted_token) in enumerate(text_examples[:8], 1):
        # Highlight the activating token in context
        highlighted_text = text.replace(highlighted_token, f"[{highlighted_token}]")
        prompt += f"{i}. (activation={activation:.3f}) {highlighted_text}\n\n"

    prompt += """

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
"""

    return prompt


def build_simplified_steering_prompt(
    token_stats: Dict[str, Dict[str, float]],
    text_examples: List[Tuple[str, float, str]],
    top_k: int = 30
) -> str:
    """
    Simpler steering-aware prompt that's more token-efficient.

    Focuses on examples + steering interpretation without heavy token statistics.
    """

    # Sort tokens by total activation
    sorted_tokens = sorted(
        token_stats.items(),
        key=lambda x: x[1]["total_activation"],
        reverse=True
    )[:top_k]

    prompt = """Analyze this sparse autoencoder feature for model steering.

TOP ACTIVATING TOKENS:
"""

    # Compact token list
    top_tokens_str = ", ".join([repr(token) for token, _ in sorted_tokens[:15]])
    prompt += f"{top_tokens_str}\n\n"

    prompt += """EXAMPLE CONTEXTS (token in [brackets]):
"""

    # Add examples
    for i, (text, act, token) in enumerate(text_examples[:6], 1):
        highlighted = text.replace(token, f"[{token}]")
        prompt += f"{i}. (act={act:.2f}) {highlighted}\n"

    prompt += """

Provide:
1. LABEL: One concept word/phrase (e.g., "negation", "dialogue_markers")
2. STEER UP: What increases if you amplify this feature 3-5x?
3. STEER DOWN: What decreases if you suppress this feature to ~0?

JSON response only:
{"label": "...", "steer_up": "...", "steer_down": "..."}
"""

    return prompt


# Example usage patterns for different steering scenarios

STEERING_USE_CASES = {
    "content_control": {
        "description": "Control semantic content (topics, themes, domains)",
        "example_features": ["political_terms", "astronomy_terms", "medical_jargon"],
        "amplify_goal": "Increase mentions of specific topics",
        "suppress_goal": "Reduce or eliminate topic mentions"
    },

    "style_control": {
        "description": "Control writing style and tone",
        "example_features": ["formal_language", "dialogue", "technical_terminology"],
        "amplify_goal": "Make output more stylistically aligned",
        "suppress_goal": "Remove style characteristics"
    },

    "structural_control": {
        "description": "Control syntactic structure",
        "example_features": ["subordinate_clauses", "passive_voice", "list_markers"],
        "amplify_goal": "Increase structural complexity/patterns",
        "suppress_goal": "Simplify structure"
    },

    "safety_control": {
        "description": "Control safety-relevant features",
        "example_features": ["refusal_language", "caveats", "uncertainty_markers"],
        "amplify_goal": "Make model more cautious/conservative",
        "suppress_goal": "Remove safety hedging (use carefully!)"
    }
}


def get_steering_guidance(feature_label: str) -> Dict[str, str]:
    """
    Provide steering guidance based on feature label.

    Args:
        feature_label: The assigned label for the feature

    Returns:
        Dict with steering recommendations
    """

    # Pattern matching for common feature types
    guidance = {
        "safe_amplification": 1.5,  # Safe multiplier for amplification
        "safe_suppression": 0.5,    # Safe multiplier for suppression
        "caution_level": "low",
        "recommended_use": ""
    }

    # Adjust based on feature type
    if "negation" in feature_label or "refusal" in feature_label:
        guidance["caution_level"] = "high"
        guidance["recommended_use"] = "Amplify for safety, suppress with extreme caution"

    elif "political" in feature_label or "controversial" in feature_label:
        guidance["caution_level"] = "medium"
        guidance["recommended_use"] = "Use for bias analysis, careful with generation"

    elif "punctuation" in feature_label or "empty" in feature_label:
        guidance["caution_level"] = "low"
        guidance["recommended_use"] = "Safe to manipulate for formatting control"
        guidance["safe_amplification"] = 2.0

    return guidance
