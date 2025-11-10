"""
Token and feature filtering utilities for dataset tokenization and feature labeling.

Provides two levels of filtering:
1. TokenFilter: Conservative filtering during dataset tokenization (permanent)
2. FeatureFilter: Aggressive filtering before LLM labeling (reversible)
"""

import re
import string
from typing import Dict, List, Set, Optional
from enum import Enum


class FilterMode(str, Enum):
    """Filter aggressiveness modes."""
    MINIMAL = "minimal"      # Only control chars and nulls
    CONSERVATIVE = "conservative"  # + obvious junk (for tokenization)
    STANDARD = "standard"    # + punctuation, single chars (for labeling)
    AGGRESSIVE = "aggressive"  # + short tokens, low entropy


class TokenFilter:
    """
    Filter tokens during dataset tokenization or before sending to LLM.

    Conservative mode (for tokenization): Only removes obvious junk
    Standard/Aggressive mode (for labeling): More aggressive filtering
    """

    def __init__(
        self,
        mode: FilterMode = FilterMode.STANDARD,
        keep_patterns: Optional[List[str]] = None,
        custom_junk_tokens: Optional[Set[str]] = None
    ):
        """
        Initialize token filter.

        Args:
            mode: Filter aggressiveness level
            keep_patterns: Regex patterns for tokens to always keep (e.g., r'C\+\+', r'\.NET')
            custom_junk_tokens: Additional tokens to always filter
        """
        self.mode = mode
        self.keep_patterns = keep_patterns or [
            r'C\+\+',  # Programming languages
            r'F#',
            r'C#',
            r'\.NET',
        ]
        self.custom_junk_tokens = custom_junk_tokens or set()

        # BPE marker patterns
        self.bpe_markers = ['Ġ', '▁', '##']

    def _clean_bpe_markers(self, token: str) -> str:
        """Remove BPE markers for analysis."""
        cleaned = token
        for marker in self.bpe_markers:
            cleaned = cleaned.replace(marker, '')
        return cleaned

    def _is_control_char(self, token: str) -> bool:
        """Check if token contains control characters."""
        cleaned = self._clean_bpe_markers(token)
        return any(ord(c) < 32 for c in cleaned)

    def _is_whitespace_only(self, token: str) -> bool:
        """Check if token is only whitespace."""
        cleaned = self._clean_bpe_markers(token)
        return not cleaned or cleaned.isspace()

    def _is_pure_punctuation(self, token: str) -> bool:
        """Check if token is only punctuation with no alphanumeric."""
        cleaned = self._clean_bpe_markers(token)
        if not cleaned:
            return False
        return not any(c.isalnum() for c in cleaned)

    def _is_single_char(self, token: str) -> bool:
        """Check if token is a single character (after BPE removal)."""
        cleaned = self._clean_bpe_markers(token)
        return len(cleaned) == 1

    def _is_short_token(self, token: str) -> bool:
        """Check if token is very short (1-2 chars)."""
        cleaned = self._clean_bpe_markers(token)
        return len(cleaned) <= 2

    def _matches_keep_pattern(self, token: str) -> bool:
        """Check if token matches any keep pattern."""
        for pattern in self.keep_patterns:
            if re.match(pattern, token):
                return True
        return False

    def is_junk_token(self, token: str) -> bool:
        """
        Determine if token should be filtered based on mode.

        Returns True if token should be filtered (is junk).
        """
        # Always keep tokens matching keep patterns
        if self._matches_keep_pattern(token):
            return False

        # Always filter custom junk tokens
        if token in self.custom_junk_tokens:
            return True

        # MINIMAL mode: Only control chars and nulls
        if self.mode == FilterMode.MINIMAL:
            return self._is_control_char(token)

        # CONSERVATIVE mode (for tokenization): Control chars + whitespace-only
        if self.mode == FilterMode.CONSERVATIVE:
            return (
                self._is_control_char(token) or
                self._is_whitespace_only(token)
            )

        # STANDARD mode (for labeling): + pure punctuation + single non-alnum chars
        if self.mode == FilterMode.STANDARD:
            if self._is_control_char(token) or self._is_whitespace_only(token):
                return True

            # Filter single character punctuation
            if self._is_single_char(token):
                cleaned = self._clean_bpe_markers(token)
                if not cleaned.isalnum():
                    return True

            # Filter pure punctuation strings
            if self._is_pure_punctuation(token):
                return True

            return False

        # AGGRESSIVE mode (for labeling): + short tokens
        if self.mode == FilterMode.AGGRESSIVE:
            if self._is_control_char(token) or self._is_whitespace_only(token):
                return True

            # Filter all single chars except alphanumeric
            if self._is_single_char(token):
                cleaned = self._clean_bpe_markers(token)
                if not cleaned.isalnum():
                    return True

            # Filter pure punctuation
            if self._is_pure_punctuation(token):
                return True

            # Filter very short tokens (1-2 chars) that are mostly non-alnum
            if self._is_short_token(token):
                cleaned = self._clean_bpe_markers(token)
                alnum_ratio = sum(c.isalnum() for c in cleaned) / len(cleaned)
                if alnum_ratio < 0.5:
                    return True

            return False

        return False

    def filter_token_list(self, tokens: List[str]) -> List[str]:
        """Filter a list of tokens, returning only meaningful ones."""
        return [token for token in tokens if not self.is_junk_token(token)]

    def filter_token_stats(
        self,
        token_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Filter token statistics dictionary."""
        return {
            token: stats
            for token, stats in token_stats.items()
            if not self.is_junk_token(token)
        }

    def get_filter_stats(
        self,
        original: Dict[str, Dict[str, float]],
        filtered: Dict[str, Dict[str, float]]
    ) -> Dict[str, any]:
        """Get statistics about filtering operation."""
        original_count = len(original)
        filtered_count = len(filtered)
        removed_count = original_count - filtered_count
        removal_pct = (removed_count / original_count * 100) if original_count > 0 else 0

        return {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "removed_count": removed_count,
            "removal_percentage": removal_pct,
            "filter_mode": self.mode.value
        }


class FeatureFilter:
    """
    Filter features before LLM labeling based on token activation patterns.

    Identifies features that are likely to be labeled as junk categories
    (punctuation, whitespace, symbols) to save API costs.
    """

    def __init__(
        self,
        junk_ratio_threshold: float = 0.8,
        single_char_ratio_threshold: float = 0.7,
        min_tokens_for_decision: int = 5
    ):
        """
        Initialize feature filter.

        Args:
            junk_ratio_threshold: If >X% of top tokens are junk, skip feature (0.0-1.0)
            single_char_ratio_threshold: If >X% of top tokens are single char, skip feature (0.0-1.0)
            min_tokens_for_decision: Minimum tokens needed to make filtering decision
        """
        self.junk_ratio_threshold = junk_ratio_threshold
        self.single_char_ratio_threshold = single_char_ratio_threshold
        self.min_tokens_for_decision = min_tokens_for_decision

        # Use standard token filter for individual token checks
        self.token_filter = TokenFilter(mode=FilterMode.STANDARD)

    def is_junk_feature(
        self,
        token_stats: Dict[str, Dict[str, float]],
        top_k: int = 10
    ) -> bool:
        """
        Determine if feature is likely junk based on token statistics.

        Args:
            token_stats: Token statistics dict {token: {max, mean, count}}
            top_k: Number of top tokens to analyze

        Returns:
            True if feature should be skipped (is likely junk)
        """
        if not token_stats:
            return True  # No tokens = definitely junk

        # Get top K tokens by max activation
        top_tokens = sorted(
            token_stats.items(),
            key=lambda x: x[1].get('max', 0),
            reverse=True
        )[:top_k]

        if len(top_tokens) < self.min_tokens_for_decision:
            return False  # Not enough data, be conservative

        # Analyze top tokens
        junk_count = 0
        single_char_count = 0
        whitespace_count = 0

        for token, stats in top_tokens:
            # Clean BPE markers for analysis
            cleaned = token.replace("Ġ", "").replace("▁", "").replace("##", "")

            # Count whitespace tokens
            if not cleaned or cleaned.isspace():
                whitespace_count += 1
                junk_count += 1
                continue

            # Count single character tokens
            if len(cleaned) == 1:
                single_char_count += 1
                # If single char is not alphanumeric, it's junk
                if not cleaned.isalnum():
                    junk_count += 1
            # Count pure punctuation tokens
            elif not any(c.isalnum() for c in cleaned):
                junk_count += 1

        # Calculate ratios
        total_analyzed = len(top_tokens)
        junk_ratio = junk_count / total_analyzed
        single_char_ratio = single_char_count / total_analyzed

        # Decision rules
        is_junk = (
            whitespace_count == total_analyzed or  # All whitespace
            junk_ratio >= self.junk_ratio_threshold or  # Too much junk
            single_char_ratio >= self.single_char_ratio_threshold  # Too many single chars
        )

        return is_junk

    def filter_features(
        self,
        features_with_stats: List[tuple]
    ) -> tuple[List, List, Dict[str, any]]:
        """
        Filter a list of (feature, token_stats) tuples.

        Args:
            features_with_stats: List of (feature_object, token_stats_dict) tuples

        Returns:
            Tuple of (features_to_label, skipped_features, stats_dict)
        """
        features_to_label = []
        skipped_features = []

        for feature, token_stats in features_with_stats:
            if self.is_junk_feature(token_stats):
                skipped_features.append(feature)
            else:
                features_to_label.append(feature)

        stats = {
            "total_features": len(features_with_stats),
            "features_to_label": len(features_to_label),
            "features_skipped": len(skipped_features),
            "skip_percentage": (len(skipped_features) / len(features_with_stats) * 100)
                             if features_with_stats else 0,
            "junk_ratio_threshold": self.junk_ratio_threshold,
            "single_char_ratio_threshold": self.single_char_ratio_threshold
        }

        return features_to_label, skipped_features, stats


# Pre-configured filter factories
def get_tokenization_filter() -> TokenFilter:
    """Get conservative filter for dataset tokenization (permanent filtering)."""
    return TokenFilter(mode=FilterMode.CONSERVATIVE)


def get_labeling_token_filter() -> TokenFilter:
    """Get standard filter for token stats before sending to LLM."""
    return TokenFilter(mode=FilterMode.STANDARD)


def get_feature_filter() -> FeatureFilter:
    """Get feature filter for pre-labeling filtering."""
    return FeatureFilter(
        junk_ratio_threshold=0.8,
        single_char_ratio_threshold=0.7,
        min_tokens_for_decision=5
    )
