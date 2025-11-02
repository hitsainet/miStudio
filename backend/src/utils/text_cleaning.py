"""
Text cleaning utilities for preprocessing datasets before tokenization.

This module provides functions to clean and filter text data to remove:
- HTML/XML tags and entities
- Control characters and special Unicode characters
- Excessive whitespace
- Low-quality or junk text
"""

import re
from typing import List, Optional
import unicodedata


class TextCleaner:
    """
    Comprehensive text cleaning for SAE training datasets.

    Removes HTML/XML, control characters, normalizes whitespace,
    and filters low-quality text to improve feature learning.
    """

    # HTML/XML patterns
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    HTML_ENTITY_PATTERN = re.compile(r'&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;')

    # URL patterns
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )

    # Email pattern
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    # Excessive punctuation or special characters (more than 3 in a row)
    EXCESSIVE_PUNCT_PATTERN = re.compile(r'([!?.,;:\-_=+*/\\|<>(){}\[\]]{4,})')

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
        remove_excessive_punct: bool = True,
        min_length: int = 10,
        max_length: Optional[int] = None,
        lowercase: bool = False,
    ):
        """
        Initialize text cleaner with configuration.

        Args:
            remove_html: Remove HTML/XML tags and entities
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            normalize_whitespace: Convert all whitespace to single spaces
            remove_control_chars: Remove Unicode control characters
            remove_excessive_punct: Remove excessive punctuation sequences
            min_length: Minimum text length after cleaning (discard shorter texts)
            max_length: Maximum text length (truncate longer texts)
            lowercase: Convert text to lowercase
        """
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.remove_control_chars = remove_control_chars
        self.remove_excessive_punct = remove_excessive_punct
        self.min_length = min_length
        self.max_length = max_length
        self.lowercase = lowercase

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

        # Remove URLs
        if self.remove_urls:
            text = self.URL_PATTERN.sub(' ', text)

        # Remove emails
        if self.remove_emails:
            text = self.EMAIL_PATTERN.sub(' ', text)

        # Remove control characters
        if self.remove_control_chars:
            text = self.CONTROL_CHARS_PATTERN.sub('', text)

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

def get_aggressive_cleaner() -> TextCleaner:
    """
    Get an aggressive text cleaner that removes most special content.

    Recommended for training high-quality SAEs on web text.
    """
    return TextCleaner(
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        normalize_whitespace=True,
        remove_control_chars=True,
        remove_excessive_punct=True,
        min_length=20,  # Longer minimum for better quality
        lowercase=False,
    )


def get_standard_cleaner() -> TextCleaner:
    """
    Get a standard text cleaner with balanced cleaning.

    Recommended for most SAE training scenarios.
    """
    return TextCleaner(
        remove_html=True,
        remove_urls=False,  # Keep URLs (might be meaningful)
        remove_emails=False,
        normalize_whitespace=True,
        remove_control_chars=True,
        remove_excessive_punct=True,
        min_length=10,
        lowercase=False,
    )


def get_minimal_cleaner() -> TextCleaner:
    """
    Get a minimal text cleaner that only removes control characters.

    Recommended when you want to preserve most content.
    """
    return TextCleaner(
        remove_html=False,
        remove_urls=False,
        remove_emails=False,
        normalize_whitespace=True,
        remove_control_chars=True,
        remove_excessive_punct=False,
        min_length=5,
        lowercase=False,
    )
