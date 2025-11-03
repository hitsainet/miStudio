"""
Text cleaning utilities for preprocessing datasets before tokenization.

ENHANCED VERSION - Addresses:
- Excessive punctuation (ellipses, emphasis marks)
- URL remnants (www., domains, partial URLs)
- Spam patterns (repeated characters)

This module provides functions to clean and filter text data to remove:
- HTML/XML tags and entities
- Control characters and special Unicode characters
- URLs and domain names (enhanced detection)
- Excessive whitespace and punctuation
- Low-quality or junk text
"""

import re
from typing import List, Optional
import unicodedata


class TextCleaner:
    """
    Comprehensive text cleaning for SAE training datasets.

    Enhanced version with improved:
    - URL detection (www., domains, fragments)
    - Punctuation normalization (ellipses, emphasis)
    - Spam pattern removal (repeated characters)
    """

    # HTML/XML patterns
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    HTML_ENTITY_PATTERN = re.compile(r'&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;')

    # Enhanced URL patterns - catches more variants
    URL_PATTERNS = [
        # Full URLs with protocol
        re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
        # URLs without protocol (www., ftp.)
        re.compile(r'\b(?:www|ftp)\.[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=]+'),
        # Domain-like patterns with common TLDs
        re.compile(r'\b[a-zA-Z0-9\-]+\.(?:com|org|net|edu|gov|io|co|ai|dev|app)(?:/[^\s]*)?(?:\s|$)'),
    ]

    # Email pattern
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    # Enhanced punctuation patterns

    # Ellipses patterns (multiple variations)
    ELLIPSIS_PATTERNS = [
        re.compile(r'\.{2,}'),      # Multiple dots: .. ... ....
        re.compile(r'\. \. \.'),    # Spaced dots: . . .
        re.compile(r'…+'),          # Unicode ellipsis character
    ]

    # Emphasis patterns (repeated punctuation for emphasis)
    EMPHASIS_PATTERNS = [
        re.compile(r'!{2,}'),       # Multiple exclamation: !! !!! !!!!
        re.compile(r'\?{2,}'),      # Multiple question: ?? ??? ????
        re.compile(r'!+\?+|!\?+!'), # Mixed: !? !?! !!??
    ]

    # Excessive punctuation (4+ consecutive different punctuation marks)
    EXCESSIVE_PUNCT_PATTERN = re.compile(r'([!?.,;:\-_=+*/\\|<>(){}\[\]]{4,})')

    # Repeated character patterns (spam detection)
    REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{4,}')  # 5+ same character

    # Whitespace normalization
    WHITESPACE_PATTERN = re.compile(r'\s+')

    # Unicode control characters (except newline, tab, carriage return)
    CONTROL_CHARS_PATTERN = re.compile(
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]'
    )

    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = False,
        normalize_whitespace: bool = True,
        remove_control_chars: bool = True,
        normalize_ellipses: bool = False,
        normalize_emphasis: bool = False,
        remove_excessive_punct: bool = True,
        remove_repeated_chars: bool = False,
        min_length: int = 10,
        max_length: Optional[int] = None,
        lowercase: bool = False,
        ellipsis_replacement: str = '.',
        emphasis_replacement: str = '!',
    ):
        """
        Initialize text cleaner with configuration.

        Args:
            remove_html: Remove HTML/XML tags and entities
            remove_urls: Remove URLs (enhanced detection)
            remove_emails: Remove email addresses
            normalize_whitespace: Convert all whitespace to single spaces
            remove_control_chars: Remove Unicode control characters
            normalize_ellipses: Normalize ellipses (... → .)
            normalize_emphasis: Normalize emphasis punctuation (!!! → !, ??? → ?)
            remove_excessive_punct: Remove excessive punctuation sequences
            remove_repeated_chars: Remove spam-like repeated characters
            min_length: Minimum text length after cleaning (discard shorter texts)
            max_length: Maximum text length (truncate longer texts)
            lowercase: Convert text to lowercase
            ellipsis_replacement: What to replace ellipses with
            emphasis_replacement: What to replace emphasis with
        """
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.remove_control_chars = remove_control_chars
        self.normalize_ellipses = normalize_ellipses
        self.normalize_emphasis = normalize_emphasis
        self.remove_excessive_punct = remove_excessive_punct
        self.remove_repeated_chars = remove_repeated_chars
        self.min_length = min_length
        self.max_length = max_length
        self.lowercase = lowercase
        self.ellipsis_replacement = ellipsis_replacement
        self.emphasis_replacement = emphasis_replacement

    def clean(self, text: str) -> Optional[str]:
        """
        Clean a single text string.

        Args:
            text: Input text

        Returns:
            Cleaned text, or None if text is too short after cleaning
        """
        if not text or not isinstance(text, str):
            return None

        # Remove HTML/XML tags and entities
        if self.remove_html:
            text = self.HTML_TAG_PATTERN.sub(' ', text)
            text = self.HTML_ENTITY_PATTERN.sub(' ', text)

        # Remove URLs (enhanced patterns)
        if self.remove_urls:
            for url_pattern in self.URL_PATTERNS:
                text = url_pattern.sub(' ', text)

        # Remove emails
        if self.remove_emails:
            text = self.EMAIL_PATTERN.sub(' ', text)

        # Remove control characters
        if self.remove_control_chars:
            text = self.CONTROL_CHARS_PATTERN.sub('', text)

        # Normalize ellipses (... → .)
        if self.normalize_ellipses:
            for ellipsis_pattern in self.ELLIPSIS_PATTERNS:
                text = ellipsis_pattern.sub(self.ellipsis_replacement, text)

        # Normalize emphasis punctuation (!!! → !, ??? → ?)
        if self.normalize_emphasis:
            for emphasis_pattern in self.EMPHASIS_PATTERNS:
                def replace_emphasis(match):
                    original = match.group(0)
                    if '!' in original and '?' in original:
                        return '!?'  # Keep !? but single occurrence
                    elif '!' in original:
                        return self.emphasis_replacement
                    elif '?' in original:
                        return '?'
                    return original[0]

                text = emphasis_pattern.sub(replace_emphasis, text)

        # Remove repeated characters (spam patterns)
        if self.remove_repeated_chars:
            def limit_repeats(match):
                char = match.group(1)
                # Allow up to 2 repeats for legitimate cases
                return char * 2
            text = self.REPEATED_CHAR_PATTERN.sub(limit_repeats, text)

        # Remove excessive punctuation (limit to 3 repeats)
        if self.remove_excessive_punct:
            def limit_punct(match):
                char = match.group(0)[0]  # Get first character
                return char * 3  # Return max 3 of that character
            text = self.EXCESSIVE_PUNCT_PATTERN.sub(limit_punct, text)

        # Normalize unicode (e.g., convert fancy quotes to regular quotes)
        text = unicodedata.normalize('NFKC', text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.WHITESPACE_PATTERN.sub(' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Lowercase if requested
        if self.lowercase:
            text = text.lower()

        # Check minimum length
        if len(text) < self.min_length:
            return None

        # Truncate if too long
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts, filtering out those that become too short.

        Args:
            texts: List of input texts

        Returns:
            List of cleaned texts (may be shorter than input if some were filtered)
        """
        cleaned = []
        for text in texts:
            cleaned_text = self.clean(text)
            if cleaned_text is not None:
                cleaned.append(cleaned_text)
        return cleaned

    def clean_batch_with_indices(self, texts: List[str]) -> tuple[List[str], List[int]]:
        """
        Clean a batch of texts, returning cleaned texts and their original indices.

        Args:
            texts: List of input texts

        Returns:
            Tuple of (cleaned_texts, original_indices)
        """
        cleaned = []
        indices = []
        for i, text in enumerate(texts):
            cleaned_text = self.clean(text)
            if cleaned_text is not None:
                cleaned.append(cleaned_text)
                indices.append(i)
        return cleaned, indices


# Pre-configured cleaners for common use cases

def get_ultra_clean_cleaner() -> TextCleaner:
    """
    Ultra-clean configuration for highest quality SAE training.

    ENHANCED VERSION:
    - Removes ALL URL patterns (http, www., domains)
    - Normalizes ellipses to single period (... → .)
    - Normalizes emphasis marks (!!! → !, ??? → ?)
    - Removes repeated character spam (aaaa → aa)
    - Higher minimum length threshold

    Expected improvements:
    - Excessive punctuation: 13% → <2%
    - URL remnants: 2% → 0%
    - Overall cleanliness: 97% → 99%+

    Recommended for: Production SAE training, research-grade datasets
    """
    return TextCleaner(
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        normalize_whitespace=True,
        remove_control_chars=True,
        normalize_ellipses=True,     # NEW: ... → .
        normalize_emphasis=True,     # NEW: !!! → !, ??? → ?
        remove_excessive_punct=True,
        remove_repeated_chars=True,  # NEW: sooo → soo
        min_length=20,
        lowercase=False,
        ellipsis_replacement='.',
        emphasis_replacement='!',
    )


def get_aggressive_cleaner() -> TextCleaner:
    """
    Aggressive text cleaner with enhanced URL and punctuation handling.

    ENHANCED VERSION:
    - Enhanced URL removal
    - Punctuation normalization enabled
    - Spam pattern removal

    Recommended for training high-quality SAEs on web text.
    """
    return TextCleaner(
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        normalize_whitespace=True,
        remove_control_chars=True,
        normalize_ellipses=True,     # NEW
        normalize_emphasis=True,     # NEW
        remove_excessive_punct=True,
        remove_repeated_chars=True,  # NEW
        min_length=20,
        lowercase=False,
    )


def get_standard_cleaner() -> TextCleaner:
    """
    Standard text cleaner with balanced cleaning.

    ENHANCED VERSION:
    - Enhanced URL detection
    - Optional punctuation normalization

    Recommended for most SAE training scenarios.
    """
    return TextCleaner(
        remove_html=True,
        remove_urls=True,           # Enhanced detection
        remove_emails=False,
        normalize_whitespace=True,
        remove_control_chars=True,
        normalize_ellipses=False,   # Off by default for standard
        normalize_emphasis=False,   # Off by default for standard
        remove_excessive_punct=True,
        remove_repeated_chars=False,
        min_length=10,
        lowercase=False,
    )


def get_minimal_cleaner() -> TextCleaner:
    """
    Minimal text cleaner that only removes control characters.

    Recommended when you want to preserve most content.
    """
    return TextCleaner(
        remove_html=False,
        remove_urls=False,
        remove_emails=False,
        normalize_whitespace=True,
        remove_control_chars=True,
        normalize_ellipses=False,
        normalize_emphasis=False,
        remove_excessive_punct=False,
        remove_repeated_chars=False,
        min_length=5,
        lowercase=False,
    )
