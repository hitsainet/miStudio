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
    STRICT = "strict"        # + ALL punctuation (even within words)


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
        custom_junk_tokens: Optional[Set[str]] = None,
        remove_all_punctuation: bool = False,
        custom_filter_chars: Optional[str] = None
    ):
        """
        Initialize token filter.

        Args:
            mode: Filter aggressiveness level
            keep_patterns: Regex patterns for tokens to always keep (e.g., r'C\\+\\+', r'\\.NET')
            custom_junk_tokens: Additional tokens to always filter
            remove_all_punctuation: If True, removes ALL punctuation characters (overrides mode)
            custom_filter_chars: Additional characters to filter (e.g., "~@#$%")
        """
        self.mode = mode
        self.keep_patterns = keep_patterns or [
            r'C\+\+',  # Programming languages
            r'F#',
            r'C#',
            r'\.NET',
        ]
        self.custom_junk_tokens = custom_junk_tokens or set()
        self.remove_all_punctuation = remove_all_punctuation
        self.custom_filter_chars = set(custom_filter_chars) if custom_filter_chars else set()

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

    def _contains_any_punctuation(self, token: str) -> bool:
        """Check if token contains any punctuation characters."""
        cleaned = self._clean_bpe_markers(token)
        return any(c in string.punctuation for c in cleaned)

    def _contains_custom_chars(self, token: str) -> bool:
        """Check if token contains any custom filter characters."""
        if not self.custom_filter_chars:
            return False
        cleaned = self._clean_bpe_markers(token)
        return any(c in self.custom_filter_chars for c in cleaned)

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

        # Filter tokens containing custom filter characters
        if self._contains_custom_chars(token):
            return True

        # If remove_all_punctuation is enabled, filter any token with punctuation
        if self.remove_all_punctuation and self._contains_any_punctuation(token):
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

        # STRICT mode: Filter ANY token containing punctuation
        if self.mode == FilterMode.STRICT:
            if self._is_control_char(token) or self._is_whitespace_only(token):
                return True

            # Filter any token containing ANY punctuation
            if self._contains_any_punctuation(token):
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

    def is_junk_sequence(
        self,
        token_ids: List[int],
        tokenizer,
        junk_ratio_threshold: float = 0.7
    ) -> bool:
        """
        Analyze if a token sequence is too junky (for sample-level filtering).

        Args:
            token_ids: List of token IDs
            tokenizer: Tokenizer to convert IDs to tokens
            junk_ratio_threshold: Reject if >X% tokens are junk (default: 0.7)

        Returns:
            True if sequence should be rejected (too junky)
        """
        if not token_ids or len(token_ids) == 0:
            return True  # Empty sequence is junk

        # Convert token IDs to tokens using decode() for proper Unicode handling
        # across all tokenizer types (byte-level BPE like GPT-2/Phi and SentencePiece like Gemma/LLaMA)
        tokens = [tokenizer.decode([tid]) if isinstance(tid, int) else tid
                  for tid in token_ids]

        # Count junk tokens
        junk_count = sum(1 for token in tokens if self.is_junk_token(token))
        junk_ratio = junk_count / len(tokens)

        return junk_ratio >= junk_ratio_threshold

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

    def is_junk_feature_from_examples(self, examples: List[Dict]) -> bool:
        """
        Determine if feature is likely junk based on activation examples.

        Extracts prime tokens from context examples and applies the same
        junk/single-char heuristics as is_junk_feature.

        Args:
            examples: List of context example dicts containing 'prime_token' key

        Returns:
            True if feature should be skipped (is likely junk)
        """
        prime_tokens = [ex.get('prime_token', '') for ex in examples if ex.get('prime_token')]

        if len(prime_tokens) < self.min_tokens_for_decision:
            return False  # Not enough data, be conservative

        junk_count = 0
        single_char_count = 0
        whitespace_count = 0

        # JUDGE ON EVERY EXAMPLE, NOT THE FIRST TEN OF A SHUFFLED LIST.
        #
        # The caller shuffles `examples` before passing them here (to break
        # primacy bias in the PROMPT, which is a different concern), so `[:10]`
        # was a random sample — and this decision writes `label_status='skipped'`,
        # which `ADJUDICATED_STATUSES` never redoes and `STALEABLE_STATUSES`
        # deliberately excludes on the reasoning that "no judge was ever asked".
        #
        # So a feature near the threshold was a coin flip whose loser was
        # permanently removed from the labelable estate, with an error message
        # asserting a definite fact. At the 0.8 threshold over 10 of 25 drawn
        # examples, a feature whose true junk ratio is 0.76 had a 54% chance of
        # being deleted; anything from ~0.56 to ~0.88 was materially uncertain.
        # Only hand-written SQL could recover it.
        #
        # Using the whole set makes the verdict a property of the FEATURE rather
        # than of the draw — and, since `example_sampling` changes which
        # examples arrive, stops a template setting from silently changing which
        # features exist.
        for token in prime_tokens:
            cleaned = token.replace("Ġ", "").replace("▁", "").replace("##", "")

            if not cleaned or cleaned.isspace():
                whitespace_count += 1
                junk_count += 1
                continue

            if len(cleaned) == 1:
                single_char_count += 1
                if not cleaned.isalnum():
                    junk_count += 1
            elif not any(c.isalnum() for c in cleaned):
                junk_count += 1

        # Denominator matches the loop above, or the ratio is nonsense: counting
        # over every token and dividing by ten made a 25-example feature's ratio
        # exceed 1.0 and cleared the threshold unconditionally.
        total_analyzed = len(prime_tokens)
        junk_ratio = junk_count / total_analyzed
        single_char_ratio = single_char_count / total_analyzed

        return (
            whitespace_count == total_analyzed
            or junk_ratio >= self.junk_ratio_threshold
            or single_char_ratio >= self.single_char_ratio_threshold
        )

    def filter_features_from_examples(
        self,
        features: List,
        features_examples: List[List[Dict]],
        all_features_examples: List[List[Dict]],
        verdict_examples: Optional[List[List[Dict]]] = None,
    ) -> tuple:
        """
        Filter features using activation examples (context-based approach).

        Args:
            features: List of feature objects
            features_examples: Parallel list of LLM-display example lists
            all_features_examples: Parallel list of all-example lists (for NLP analysis)

        Returns:
            Tuple of (filtered_features, filtered_examples, filtered_all_examples, stats)
        """
        filtered_features = []
        filtered_examples = []
        filtered_all_examples = []
        skipped_count = 0

        # THE VERDICT IS JUDGED ON A FIXED SET, NOT ON WHAT THE PROMPT SHOWS.
        #
        # `features_examples` is the DISPLAY set, chosen by the template's
        # `example_sampling`. Judging junkiness on it made a template setting
        # decide which features exist: stratified shows ranks 1,11,21…91 while
        # top_k shows 1–10, and a register feature that fires strongly on a word
        # and weakly at sentence boundaries has a punctuation-heavy mid-range.
        #
        # That would be survivable if the decision were revisable. It is not.
        # `_persist_filtered_out` writes `label_status='skipped'`, which is in
        # ADJUDICATED_STATUSES (never redone) and carries no fingerprint, so the
        # staleness path cannot reach it either. Flipping the default template
        # to stratified would have permanently retired a set of features with a
        # message asserting a definite fact about them — and nothing would ever
        # re-offer them.
        #
        # The filter is a data-quality gate on the FEATURE. It must not move
        # with the arm under test.
        verdicts = verdict_examples if verdict_examples is not None else features_examples
        if len(verdicts) != len(features):
            raise ValueError(
                f"verdict_examples has {len(verdicts)} entries for "
                f"{len(features)} features; a positional mismatch would judge "
                f"one feature on another feature's tokens"
            )

        for feature, examples, all_ex, verdict_ex in zip(
            features, features_examples, all_features_examples, verdicts
        ):
            if self.is_junk_feature_from_examples(verdict_ex):
                skipped_count += 1
            else:
                filtered_features.append(feature)
                filtered_examples.append(examples)
                filtered_all_examples.append(all_ex)

        total = len(features)
        stats = {
            "total_features": total,
            "features_to_label": len(filtered_features),
            "features_skipped": skipped_count,
            "skip_percentage": (skipped_count / total * 100) if total else 0,
            "junk_ratio_threshold": self.junk_ratio_threshold,
            "single_char_ratio_threshold": self.single_char_ratio_threshold,
        }

        return filtered_features, filtered_examples, filtered_all_examples, stats

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
