"""
Auto-labeling heuristics for SAE features.

This module provides pattern-matching heuristics to automatically generate
descriptive labels for discovered SAE features based on their max-activating examples.
"""

from typing import List, Dict, Any
import string


# Pattern matching dictionaries
QUESTION_WORDS = {
    "what", "how", "why", "when", "where", "who", "which",
    "What", "How", "Why", "When", "Where", "Who", "Which"
}

CODE_TOKENS = {
    "def", "function", "class", "import", "return", "const", "let", "var",
    "if", "else", "for", "while", "try", "catch", "async", "await",
    "public", "private", "static", "void", "int", "str", "bool"
}

POSITIVE_SENTIMENT_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "happy", "love", "best", "awesome", "perfect", "beautiful",
    "Good", "Great", "Excellent", "Amazing", "Wonderful", "Fantastic",
    "Happy", "Love", "Best", "Awesome", "Perfect", "Beautiful"
}

NEGATIVE_SENTIMENT_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate",
    "sad", "angry", "disappointed", "poor", "disgust", "disgusting",
    "Bad", "Terrible", "Awful", "Horrible", "Worst", "Hate",
    "Sad", "Angry", "Disappointed", "Poor", "Disgust", "Disgusting"
}

NEGATION_WORDS = {
    "not", "no", "never", "n't", "nothing", "nobody", "nowhere",
    "Not", "No", "Never", "Nothing", "Nobody", "Nowhere",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't"
}

FIRST_PERSON_PRONOUNS = {
    "I", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"
}


def extract_high_activation_tokens(
    top_examples: List[Dict[str, Any]],
    intensity_threshold: float = 0.7,
    max_examples: int = 5
) -> List[str]:
    """
    Extract tokens with high activation intensity from top examples.

    Args:
        top_examples: List of example dicts with 'tokens' and 'activations' arrays
        intensity_threshold: Minimum activation intensity (0-1) to consider
        max_examples: Maximum number of examples to analyze

    Returns:
        List of high-activation token strings
    """
    high_activation_tokens = []

    for example in top_examples[:max_examples]:
        tokens = example.get("tokens", [])
        activations = example.get("activations", [])

        if not tokens or not activations:
            continue

        # Find max activation to normalize
        max_activation = max(activations) if activations else 1.0

        # Extract tokens with high intensity
        for token, activation in zip(tokens, activations):
            if max_activation > 0:
                intensity = activation / max_activation
                if intensity > intensity_threshold:
                    high_activation_tokens.append(token)

    return high_activation_tokens


def auto_label_feature(
    top_examples: List[Dict[str, Any]],
    neuron_index: int
) -> str:
    """
    Automatically generate a descriptive label for a feature.

    Uses pattern matching heuristics on high-activation tokens from
    the feature's max-activating examples.

    Args:
        top_examples: List of dicts with 'tokens' (list of str) and
                     'activations' (list of float) arrays
        neuron_index: SAE neuron index for fallback label

    Returns:
        Descriptive label string (e.g., "Punctuation", "Question Pattern")
    """
    # Extract high-activation tokens from top 5 examples
    high_tokens = extract_high_activation_tokens(top_examples, intensity_threshold=0.7, max_examples=5)

    if not high_tokens:
        # No tokens found, return fallback
        return f"Feature {neuron_index}"

    # Convert to set for faster lookups
    token_set = set(high_tokens)

    # Pattern 1: All punctuation
    if all(token in string.punctuation for token in high_tokens):
        return "Punctuation"

    # Pattern 2: Question words (what/how/why/when/where/who)
    if any(token in QUESTION_WORDS for token in high_tokens):
        return "Question Pattern"

    # Pattern 3: Code syntax tokens
    if any(token in CODE_TOKENS for token in high_tokens):
        return "Code Syntax"

    # Pattern 4: Positive sentiment words
    if any(token in POSITIVE_SENTIMENT_WORDS for token in high_tokens):
        return "Sentiment Positive"

    # Pattern 5: Negative sentiment words
    if any(token in NEGATIVE_SENTIMENT_WORDS for token in high_tokens):
        return "Sentiment Negative"

    # Pattern 6: Negation logic
    if any(token in NEGATION_WORDS for token in high_tokens):
        return "Negation Logic"

    # Pattern 7: First-person pronouns
    if any(token in FIRST_PERSON_PRONOUNS for token in high_tokens):
        return "Pronouns First Person"

    # Fallback: No pattern matched
    return f"Feature {neuron_index}"
