"""
Labeling Context Formatter

This module provides formatting utilities for converting activation examples
into prompt-ready text for different labeling template styles.

Supports three template formats:
1. miStudio Internal: Full-context examples with << >> markers
2. Anthropic-Style: Examples + logit effects (promoted/suppressed tokens)
3. EleutherAI Detection: Numbered test examples for binary classification
"""

from typing import List, Dict, Any, Optional


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

        Format:
            Example 1 (activation: 0.007): commercial and residential r ena issuance of <<sorts>> . The last of four
            Example 2 (activation: 0.007): talk about women and women 's issues l <<ately>> , a nod to the emergence

            NEGATIVE EXAMPLES (Low Activation):
            Example 1 (activation: 0.001): ...text where feature does NOT activate strongly...
            Example 2 (activation: 0.001): ...another low-activation example...

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
            # Extract example data
            prefix_tokens = example.get('prefix_tokens', []) if include_prefix else []
            prime_token = example.get('prime_token', '')
            suffix_tokens = example.get('suffix_tokens', []) if include_suffix else []
            max_activation = example.get('max_activation', 0.0)

            # Build context string: "prefix <<prime>> suffix"
            context_parts = []
            if prefix_tokens:
                context_parts.append(LabelingContextFormatter._join_tokens_naturally(prefix_tokens))
            if prime_token:
                # Clean the prime token for display
                clean_prime = LabelingContextFormatter._clean_token(prime_token)
                context_parts.append(f'{marker_left}{clean_prime}{marker_right}')
            if suffix_tokens:
                context_parts.append(LabelingContextFormatter._join_tokens_naturally(suffix_tokens))

            context = ' '.join(context_parts)

            # Format: Example N (activation: X.XXX): context
            formatted_lines.append(
                f"Example {idx} (activation: {max_activation:.3f}): {context}"
            )

        # Format negative (low-activation) examples if provided
        if negative_examples:
            formatted_lines.append("")  # Blank line separator
            formatted_lines.append("NEGATIVE EXAMPLES (Low Activation):")

            for idx, example in enumerate(negative_examples, start=1):
                # Extract example data
                prefix_tokens = example.get('prefix_tokens', []) if include_prefix else []
                prime_token = example.get('prime_token', '')
                suffix_tokens = example.get('suffix_tokens', []) if include_suffix else []
                max_activation = example.get('max_activation', 0.0)

                # Build context string: "prefix <<prime>> suffix"
                context_parts = []
                if prefix_tokens:
                    context_parts.append(LabelingContextFormatter._join_tokens_naturally(prefix_tokens))
                if prime_token:
                    # Clean the prime token for display
                    clean_prime = LabelingContextFormatter._clean_token(prime_token)
                    context_parts.append(f'{marker_left}{clean_prime}{marker_right}')
                if suffix_tokens:
                    context_parts.append(LabelingContextFormatter._join_tokens_naturally(suffix_tokens))

                context = ' '.join(context_parts)

                # Format: Example N (activation: X.XXX): context
                formatted_lines.append(
                    f"Example {idx} (activation: {max_activation:.3f}): {context}"
                )

        return '\n'.join(formatted_lines)

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
            '**' -> ('', '**')
            '<< >>' -> ('<<', '>>')

        Args:
            marker: Marker string like '<<>>' or '**'

        Returns:
            Tuple of (left_marker, right_marker)
        """
        if not marker:
            return ('', '')

        # If marker has even length and is symmetric, split in half
        if len(marker) >= 2 and marker[:len(marker)//2] == marker[len(marker)//2:]:
            mid = len(marker) // 2
            return (marker[:mid], marker[mid:])

        # If marker contains space, split on space
        if ' ' in marker:
            parts = marker.split(' ', 1)
            return (parts[0], parts[1] if len(parts) > 1 else '')

        # Otherwise, treat entire marker as suffix only
        return ('', marker)

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
