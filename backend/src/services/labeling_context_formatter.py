"""
Labeling Context Formatter

This module provides formatting utilities for converting activation examples
into prompt-ready text for different labeling template styles.

Supports three template formats:
1. miStudio Internal: Full-context examples with << >> markers
2. Anthropic-Style: Examples + logit effects (promoted/suppressed tokens)
3. EleutherAI Detection: Numbered test examples for binary classification
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class LabelingContextFormatter:
    """
    Static methods for formatting activation examples into template-specific prompt text.
    """

    @staticmethod
    def format_mistudio_context(
        examples: List[Dict[str, Any]],
        template_config: Dict[str, Any],
        feature_id: str,
        negative_examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Format examples in miStudio Internal style.

        SIMPLIFIED APPROACH (Layer 1 Fix):
        - Cleanly separate prefix, prime, and suffix
        - No complex word reconstruction - just wrap the prime token
        - Handle invisible/space tokens visibly
        - Robust duplicate detection and removal

        Format:
            Example 1 (activation: 0.007): commercial and residential issuance of <<sorts>> . The last of four
            Example 2 (activation: 0.007): talk about women and women's issues <<lately>> , a nod to the emergence

            NEGATIVE EXAMPLES (Low Activation):
            Example 1 (activation: 0.001): ...text where feature does NOT activate strongly...

        Args:
            examples: List of activation examples with prefix_tokens, prime_token, suffix_tokens, max_activation
            template_config: Template configuration including prime_token_marker, include_prefix, include_suffix
            feature_id: Feature identifier for reference
            negative_examples: Optional list of low-activation examples for contrastive learning

        Returns:
            Formatted examples block as string
        """
        marker_left, marker_right = LabelingContextFormatter._split_marker(
            template_config.get('prime_token_marker', '<<>>')
        )
        include_prefix = template_config.get('include_prefix', True)
        include_suffix = template_config.get('include_suffix', True)

        # Format positive (high-activation) examples
        formatted_lines = []
        for idx, example in enumerate(examples, start=1):
            context = LabelingContextFormatter._format_single_example(
                example=example,
                marker_left=marker_left,
                marker_right=marker_right,
                include_prefix=include_prefix,
                include_suffix=include_suffix,
                example_idx=idx
            )
            max_activation = example.get('max_activation', 0.0)
            formatted_lines.append(
                f"Example {idx} (activation: {max_activation:.3f}): {context}"
            )

        # Format negative (low-activation) examples if provided
        if negative_examples:
            formatted_lines.append("")  # Blank line separator
            formatted_lines.append("NEGATIVE EXAMPLES (Low Activation):")

            for idx, example in enumerate(negative_examples, start=1):
                context = LabelingContextFormatter._format_single_example(
                    example=example,
                    marker_left=marker_left,
                    marker_right=marker_right,
                    include_prefix=include_prefix,
                    include_suffix=include_suffix,
                    example_idx=idx,
                    is_negative=True
                )
                max_activation = example.get('max_activation', 0.0)
                formatted_lines.append(
                    f"Example {idx} (activation: {max_activation:.3f}): {context}"
                )

        return '\n'.join(formatted_lines)

    @staticmethod
    def _format_single_example(
        example: Dict[str, Any],
        marker_left: str,
        marker_right: str,
        include_prefix: bool,
        include_suffix: bool,
        example_idx: int,
        is_negative: bool = False
    ) -> str:
        """
        Format a single activation example with the simplified approach.

        This method:
        1. Extracts and validates prefix_tokens, prime_token, suffix_tokens
        2. Removes any duplicates of prime_token from prefix/suffix
        3. Cleans the prime token and handles invisible tokens visibly
        4. Joins everything naturally with proper spacing

        Args:
            example: Single example dict with prefix_tokens, prime_token, suffix_tokens
            marker_left: Left marker (e.g., '<<')
            marker_right: Right marker (e.g., '>>')
            include_prefix: Whether to include prefix tokens
            include_suffix: Whether to include suffix tokens
            example_idx: Example index for logging
            is_negative: Whether this is a negative example (for logging)

        Returns:
            Formatted context string like "prefix <<prime>> suffix"
        """
        log_prefix = f"[NEG {example_idx}]" if is_negative else f"[EX {example_idx}]"

        # Extract example data
        prefix_tokens = list(example.get('prefix_tokens', [])) if include_prefix else []
        prime_token = example.get('prime_token', '')
        suffix_tokens = list(example.get('suffix_tokens', [])) if include_suffix else []

        logger.debug(f"{log_prefix} Input: prime='{prime_token}', prefix={prefix_tokens}, suffix={suffix_tokens}")

        # STEP 1: Validate and clean prime_token
        if not prime_token:
            logger.warning(f"{log_prefix} prime_token is empty/None!")
            # Fall back to showing context without markers
            prefix_str = LabelingContextFormatter._join_tokens_naturally(prefix_tokens) if prefix_tokens else ""
            suffix_str = LabelingContextFormatter._join_tokens_naturally(suffix_tokens) if suffix_tokens else ""
            return f"{prefix_str} {suffix_str}".strip()

        # STEP 2: Get cleaned prime for comparison and display
        cleaned_prime = LabelingContextFormatter._clean_token(prime_token)

        # STEP 3: Remove duplicates - prime_token should NOT appear in prefix or suffix
        # Check using both exact match and cleaned match
        original_prefix_len = len(prefix_tokens)
        original_suffix_len = len(suffix_tokens)

        # Filter prefix tokens - remove any that match prime_token
        prefix_tokens = [
            t for t in prefix_tokens
            if t != prime_token and LabelingContextFormatter._clean_token(t) != cleaned_prime
        ]

        # Filter suffix tokens - remove any that match prime_token
        suffix_tokens = [
            t for t in suffix_tokens
            if t != prime_token and LabelingContextFormatter._clean_token(t) != cleaned_prime
        ]

        # Log if we removed duplicates
        if len(prefix_tokens) < original_prefix_len:
            removed = original_prefix_len - len(prefix_tokens)
            logger.error(f"{log_prefix} 🚨 DUPLICATE BUG: Removed {removed} prime_token duplicate(s) from prefix!")
        if len(suffix_tokens) < original_suffix_len:
            removed = original_suffix_len - len(suffix_tokens)
            logger.error(f"{log_prefix} 🚨 DUPLICATE BUG: Removed {removed} prime_token duplicate(s) from suffix!")

        # STEP 4: Reassemble complete word if prime_token is a fragment
        # This handles cases like "Jim L <<ites>>" → "Jim <<Lites>>"
        prefix_tokens, prime_display, suffix_tokens = LabelingContextFormatter._reassemble_word_around_prime(
            prefix_tokens, prime_token, suffix_tokens, log_prefix
        )

        # STEP 5: Handle edge cases for prime display (space tokens, etc.)
        # Only apply special handling if reassembly didn't produce a meaningful word
        if not prime_display or prime_display.isspace():
            prime_display = LabelingContextFormatter._get_prime_display(prime_token, cleaned_prime, log_prefix)

        # STEP 6: Build the context string
        # Join prefix tokens naturally
        prefix_str = LabelingContextFormatter._join_tokens_naturally(prefix_tokens) if prefix_tokens else ""

        # Join suffix tokens naturally
        suffix_str = LabelingContextFormatter._join_tokens_naturally(suffix_tokens) if suffix_tokens else ""

        # Build the marked prime token
        marked_prime = f"{marker_left}{prime_display}{marker_right}"

        # Combine parts with proper spacing
        parts = []
        if prefix_str:
            parts.append(prefix_str)
        parts.append(marked_prime)
        if suffix_str:
            parts.append(suffix_str)

        context = ' '.join(parts)

        logger.debug(f"{log_prefix} Output: '{context}'")
        return context

    @staticmethod
    def _get_prime_display(prime_token: str, cleaned_prime: str, log_prefix: str = "") -> str:
        """
        Determine the display text for the prime token.

        Handles edge cases:
        - Normal tokens: Use cleaned version (e.g., "Ġword" -> "word")
        - Invisible/space tokens: Make them visible (e.g., "Ġ" -> "[SPACE]")
        - Empty after cleaning: Use original token repr

        Args:
            prime_token: Original prime token (e.g., "Ġword" or "Ġ")
            cleaned_prime: Cleaned prime token (e.g., "word" or "")
            log_prefix: Logging prefix for context

        Returns:
            Display string for the prime token
        """
        # Case 1: Cleaned prime is non-empty and visible - use it
        if cleaned_prime and not cleaned_prime.isspace():
            return cleaned_prime

        # Case 2: Prime token is a pure BPE space marker (Ġ, ▁, etc.)
        # These are important for SAE activation but invisible - make them visible
        if prime_token in ('Ġ', '▁', ' ', '\t', '\n', '##'):
            logger.warning(f"{log_prefix} Prime token is a space/marker character: {prime_token!r}")
            return "[SPACE]"

        # Case 3: Prime token starts with marker but has no content after cleaning
        if prime_token.startswith(('Ġ', '▁', '##', ' ')) and not cleaned_prime:
            logger.warning(f"{log_prefix} Prime token '{prime_token!r}' became empty after cleaning")
            return "[SPACE]"

        # Case 4: Cleaned prime is whitespace-only
        if cleaned_prime and cleaned_prime.isspace():
            logger.warning(f"{log_prefix} Cleaned prime is whitespace: {cleaned_prime!r}")
            return "[SPACE]"

        # Case 5: Both cleaned and original are problematic - use original with BPE markers
        # This preserves the token for debugging while still showing something
        if not cleaned_prime:
            logger.error(f"{log_prefix} 🚨 Unexpected empty prime! Using original: {prime_token!r}")
            # Strip BPE markers but keep the rest
            fallback = prime_token.lstrip('Ġ▁## ')
            if fallback:
                return fallback
            # Absolute fallback - show the raw token
            return f"[{prime_token!r}]"

        return cleaned_prime

    @staticmethod
    def format_anthropic_logit(
        examples: List[Dict[str, Any]],
        logit_effects: Dict[str, Any],
        template_config: Dict[str, Any],
        feature_id: str,
        negative_examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Format examples in Anthropic-Style with logit effects.

        Format:
            TOP ACTIVATING EXAMPLES:
            Example 1 (activation: 0.007): ...context...
            ...

            NEGATIVE EXAMPLES (Low Activation):
            Example 1 (activation: 0.001): ...context...
            ...

            LOGIT EFFECTS:
            Top promoted tokens: token1, token2, token3, ...
            Top suppressed tokens: token4, token5, token6, ...

        Args:
            examples: List of activation examples
            logit_effects: Dict with 'promoted' and 'suppressed' token lists
            template_config: Template configuration
            feature_id: Feature identifier
            negative_examples: Optional list of low-activation examples for contrastive learning

        Returns:
            Formatted examples block with logit effects
        """
        # Format examples section (reuse miStudio formatter)
        examples_text = LabelingContextFormatter.format_mistudio_context(
            examples, template_config, feature_id, negative_examples
        )

        # Format logit effects section
        logit_text_parts = []
        if logit_effects:
            promoted = logit_effects.get('promoted', [])
            suppressed = logit_effects.get('suppressed', [])

            if promoted:
                top_count = template_config.get('top_promoted_tokens_count', 10)
                promoted_tokens = ', '.join([f'"{token}"' for token in promoted[:top_count]])
                logit_text_parts.append(f"Top promoted tokens: {promoted_tokens}")

            if suppressed:
                top_count = template_config.get('top_suppressed_tokens_count', 10)
                suppressed_tokens = ', '.join([f'"{token}"' for token in suppressed[:top_count]])
                logit_text_parts.append(f"Top suppressed tokens: {suppressed_tokens}")

        logit_text = '\n'.join(logit_text_parts) if logit_text_parts else "No logit effects available."

        # Combine sections
        return (
            f"TOP ACTIVATING EXAMPLES:\n"
            f"{examples_text}\n\n"
            f"LOGIT EFFECTS:\n"
            f"{logit_text}"
        )

    @staticmethod
    def format_eleutherai_detection(
        feature_explanation: Dict[str, Any],
        test_examples: List[str]
    ) -> str:
        """
        Format test examples for EleutherAI Detection scoring.

        Format:
            1. example text 1
            2. example text 2
            3. example text 3
            ...

        Args:
            feature_explanation: Feature explanation with name, category, description
            test_examples: List of test example strings to score

        Returns:
            Numbered list of test examples
        """
        formatted_lines = []
        for idx, example_text in enumerate(test_examples, start=1):
            formatted_lines.append(f"{idx}. {example_text}")

        return '\n'.join(formatted_lines)

    @staticmethod
    def _split_marker(marker: str) -> tuple[str, str]:
        """
        Split prime token marker into left and right parts.

        Examples:
            '<<>>' -> ('<<', '>>')
            '**' -> ('*', '*')
            '<< >>' -> ('<<', '>>')
            '[[]]' -> ('[[', ']]')

        Args:
            marker: Marker string like '<<>>' or '**'

        Returns:
            Tuple of (left_marker, right_marker)
        """
        if not marker:
            return ('', '')

        # If marker contains space, split on space first
        if ' ' in marker:
            parts = marker.split(' ', 1)
            return (parts[0], parts[1] if len(parts) > 1 else '')

        # For even-length markers, split in the middle
        # This handles <<>>, **, [[]], etc.
        if len(marker) >= 2 and len(marker) % 2 == 0:
            mid = len(marker) // 2
            return (marker[:mid], marker[mid:])

        # For odd-length markers, put the extra char on the right
        if len(marker) >= 2:
            mid = len(marker) // 2
            return (marker[:mid], marker[mid:])

        # Single character - use as both left and right
        return (marker, marker)

    @staticmethod
    def _join_tokens_naturally(tokens: List[str]) -> str:
        """
        Join tokens with proper spacing, reuniting subword tokens.

        Many tokenizers (GPT-2, GPT-3, etc.) use special characters to mark word boundaries:
        - 'Ġ' (U+0120) indicates the start of a new word (GPT-2)
        - '▁' indicates word start (SentencePiece)
        - '##' indicates word continuation (BERT)
        - Tokens without these markers continue the previous word

        Examples:
            ["Ġtoken", "ization"] → "tokenization"
            ["Ġhello", "Ġworld"] → "hello world"
            ["Ġun", "der", "stand", "ing"] → "understanding"

        Args:
            tokens: List of token strings

        Returns:
            Human-readable text with proper spacing
        """
        if not tokens:
            return ''

        result_parts = []

        for token in tokens:
            # Check if this token starts a new word
            if token.startswith('Ġ'):
                # GPT-2 style: Remove marker and add space before (except at start)
                clean_token = token[1:]
                if result_parts:
                    result_parts.append(' ')
                result_parts.append(clean_token)
            elif token.startswith('▁'):
                # SentencePiece style: Remove marker and add space before (except at start)
                clean_token = token[1:]
                if result_parts:
                    result_parts.append(' ')
                result_parts.append(clean_token)
            elif token.startswith('##'):
                # BERT style: Remove marker and append directly (continuation)
                clean_token = token[2:]
                result_parts.append(clean_token)
            elif token.startswith(' '):
                # Some tokenizers use actual space
                clean_token = token[1:]
                if result_parts:
                    result_parts.append(' ')
                result_parts.append(clean_token)
            else:
                # No marker - continuation of previous word
                result_parts.append(token)

        return ''.join(result_parts)

    @staticmethod
    def _clean_token(token: str) -> str:
        """
        Clean token by removing BPE markers.

        Removes:
        - "Ġ" (GPT-2 space marker)
        - "▁" (SentencePiece space marker)
        - "##" (BERT continuation marker)
        - Surrounding quotes

        Args:
            token: Raw token string

        Returns:
            Cleaned token string
        """
        # Remove surrounding quotes
        cleaned = token.strip('"\'')

        # Remove BPE markers
        cleaned = cleaned.lstrip('Ġ▁##_')

        return cleaned

    @staticmethod
    def _is_word_start_token(token: str) -> bool:
        """
        Check if a token represents the start of a new word.

        In BPE tokenization:
        - GPT-2: 'Ġ' prefix means start of new word (space before)
        - SentencePiece: '▁' prefix means start of new word
        - BERT: '##' prefix means CONTINUATION (so no ## = potential word start)
        - Actual space prefix: start of new word

        Args:
            token: Raw token string

        Returns:
            True if token starts a new word, False if it continues previous word
        """
        if not token:
            return True  # Empty token treated as word boundary

        # Check for word-start markers
        if token.startswith(('Ġ', '▁', ' ')):
            return True

        # BERT's ## means continuation, so lack of ## could mean word start
        # But we can't assume - if no marker at all, it's likely a continuation
        # in GPT-2/SentencePiece style tokenizers

        return False

    @staticmethod
    def _is_continuation_token(token: str) -> bool:
        """
        Check if a token continues a previous word (is a word fragment).

        Args:
            token: Raw token string

        Returns:
            True if token is a continuation/fragment, False if it starts a new word
        """
        if not token:
            return False

        # BERT-style continuation marker
        if token.startswith('##'):
            return True

        # If no word-start marker, it's a continuation in GPT-2/SentencePiece
        return not token.startswith(('Ġ', '▁', ' '))

    @staticmethod
    def _reassemble_word_around_prime(
        prefix_tokens: List[str],
        prime_token: str,
        suffix_tokens: List[str],
        log_prefix: str = ""
    ) -> Tuple[List[str], str, List[str]]:
        """
        Reassemble a complete word when prime_token is a word fragment.

        When the prime token is a continuation (e.g., "ites" from "Lites"),
        this method finds the word start in prefix and/or continuation in suffix,
        and returns the reassembled word along with the modified token lists.

        Examples:
            prefix=["Jim", "ĠL"], prime="ites", suffix=["Ġsaid"]
            → (["Jim"], "Lites", ["Ġsaid"])

            prefix=["Hillary", "ĠCl"], prime="inton", suffix=["Ġas"]
            → (["Hillary"], "Clinton", ["Ġas"])

        Args:
            prefix_tokens: List of prefix tokens
            prime_token: The prime token (may be a fragment)
            suffix_tokens: List of suffix tokens
            log_prefix: Logging prefix for context

        Returns:
            Tuple of (new_prefix_tokens, reassembled_word, new_suffix_tokens)
        """
        # Check if prime_token is a continuation (word fragment)
        prime_is_continuation = LabelingContextFormatter._is_continuation_token(prime_token)

        # If prime starts a new word, check only if suffix continues it
        if not prime_is_continuation:
            # Prime starts a new word - check if suffix continues it
            word_parts = [prime_token]
            new_suffix = list(suffix_tokens)

            # Collect suffix continuations
            while new_suffix and LabelingContextFormatter._is_continuation_token(new_suffix[0]):
                word_parts.append(new_suffix.pop(0))

            if len(word_parts) > 1:
                # Word extends into suffix
                reassembled = LabelingContextFormatter._join_tokens_naturally(word_parts)
                logger.debug(f"{log_prefix} Reassembled word (suffix only): '{reassembled}' from {word_parts}")
                return list(prefix_tokens), reassembled, new_suffix
            else:
                # No reassembly needed
                return list(prefix_tokens), LabelingContextFormatter._clean_token(prime_token), list(suffix_tokens)

        # Prime is a continuation - find word start in prefix
        word_parts = []
        new_prefix = list(prefix_tokens)

        # Walk backwards through prefix to find word start
        while new_prefix:
            last_token = new_prefix[-1]

            if LabelingContextFormatter._is_continuation_token(last_token):
                # This is also a continuation - add to word parts and keep going
                word_parts.insert(0, new_prefix.pop())
            else:
                # This token STARTS the word - include it and stop
                word_parts.insert(0, new_prefix.pop())
                break

        # Add the prime token
        word_parts.append(prime_token)

        # Check if suffix continues the word
        new_suffix = list(suffix_tokens)
        while new_suffix and LabelingContextFormatter._is_continuation_token(new_suffix[0]):
            word_parts.append(new_suffix.pop(0))

        # Reassemble the word
        reassembled = LabelingContextFormatter._join_tokens_naturally(word_parts)

        logger.debug(f"{log_prefix} Reassembled word: '{reassembled}' from {word_parts}")

        return new_prefix, reassembled, new_suffix

    @staticmethod
    def format_token_list(
        tokens: List[str],
        max_display: Optional[int] = None,
        clean: bool = True
    ) -> str:
        """
        Format a list of tokens into a comma-separated string.

        Args:
            tokens: List of token strings
            max_display: Maximum number of tokens to display (None = all)
            clean: Whether to clean BPE markers

        Returns:
            Formatted string like: "token1, token2, token3, ..."
        """
        if clean:
            tokens = [LabelingContextFormatter._clean_token(t) for t in tokens]

        if max_display and len(tokens) > max_display:
            displayed = tokens[:max_display]
            remaining = len(tokens) - max_display
            return ', '.join([f'"{t}"' for t in displayed]) + f', ... (+{remaining} more)'

        return ', '.join([f'"{t}"' for t in tokens])
