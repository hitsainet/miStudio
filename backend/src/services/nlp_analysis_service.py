"""
NLP Analysis Service for Feature Interpretability.

This service provides advanced NLP analysis of feature activation examples:
- Prime token analysis: POS tagging, NER, frequency statistics
- Context pattern extraction: N-grams, common structures
- Semantic clustering: Group examples by semantic similarity

Analysis results are cached in FeatureAnalysisCache for reuse in labeling prompts.
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select

logger = logging.getLogger(__name__)

# Lazy imports for NLP libraries (may not be installed)
_spacy_nlp = None
_nltk_initialized = False


def _get_spacy_nlp():
    """Lazy load spaCy model."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            # Try to load the model, download if not available
            try:
                _spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.info("Downloading spaCy model en_core_web_sm...")
                from spacy.cli import download
                download("en_core_web_sm")
                _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except ImportError:
            logger.warning("spaCy not installed. Some NLP features will be limited.")
            _spacy_nlp = False  # Mark as unavailable
    return _spacy_nlp if _spacy_nlp else None


def _init_nltk():
    """Initialize NLTK data."""
    global _nltk_initialized
    if not _nltk_initialized:
        try:
            import nltk
            # Download required NLTK data
            for package in ['punkt', 'averaged_perceptron_tagger', 'stopwords']:
                try:
                    nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'taggers/{package}' if 'tagger' in package else f'corpora/{package}')
                except LookupError:
                    nltk.download(package, quiet=True)
            _nltk_initialized = True
        except ImportError:
            logger.warning("NLTK not installed. Some NLP features will be limited.")
    return _nltk_initialized


class NLPAnalysisService:
    """
    Service for NLP analysis of feature activation examples.

    Analyzes all 100 max-activating examples per feature to extract:
    - Prime token statistics (POS, NER, frequency)
    - Context patterns (n-grams, common structures)
    - Semantic clusters (groupings of similar examples)
    - Activation statistics

    Includes BPE subword reconstruction to analyze full words instead of fragments.
    """

    # BPE markers that indicate the START of a new word
    # These markers appear at the beginning of tokens that start new words
    BPE_WORD_START_MARKERS = ('Ġ', '▁')  # GPT-2/Llama, SentencePiece

    # BPE markers that indicate CONTINUATION of previous word
    BPE_CONTINUATION_MARKERS = ('##',)  # BERT style

    # All BPE markers for cleaning
    BPE_MARKERS = ('Ġ', '▁', '##', ' ')

    # Common stop words for filtering
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'that', 'which', 'who',
        'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
        'he', 'she', 'him', 'her', 'his', 'hers', 'we', 'us', 'our', 'you',
        'your', 'i', 'me', 'my', 'not', 'no', 'yes', 'so', 'if', 'then'
    }

    def __init__(self, db: Optional[Session] = None):
        """Initialize NLP analysis service."""
        self.db = db
        self._spacy_nlp = None

    def _clean_token(self, token: str) -> str:
        """Clean BPE markers from token."""
        if not token:
            return ''
        cleaned = token.strip('"\'')
        for marker in self.BPE_MARKERS:
            if cleaned.startswith(marker):
                cleaned = cleaned[len(marker):]
        return cleaned.strip()

    def _is_word_start(self, token: str) -> bool:
        """
        Determine if a token starts a new word based on BPE markers.

        Token starts a new word if:
        - It starts with Ġ (GPT-2/Llama) or ▁ (SentencePiece)
        - It starts with a space
        - It does NOT start with ## (BERT continuation marker)

        Tokens without any marker are continuations of the previous word.
        """
        if not token:
            return False

        # Check for explicit continuation markers first (BERT style)
        for marker in self.BPE_CONTINUATION_MARKERS:
            if token.startswith(marker):
                return False

        # Check for word start markers
        for marker in self.BPE_WORD_START_MARKERS:
            if token.startswith(marker):
                return True

        # Space at start indicates new word
        if token.startswith(' '):
            return True

        # No marker = continuation of previous word
        return False

    def _reconstruct_words(
        self,
        prefix_tokens: List[str],
        prime_token: str,
        suffix_tokens: List[str]
    ) -> Dict[str, Any]:
        """
        Reconstruct full words from BPE subword tokens.

        This method combines subword tokens back into complete words based on
        BPE marker conventions. It tracks which word contains the prime token
        (the "prime word") for more accurate NLP analysis.

        Examples:
        - Input: ['Ġnormal', 'cy'] with prime='cy' → prime_word='normalcy'
        - Input: ['ĠMoh', 'awk'] with prime='awk' → prime_word='Mohawk'
        - Input: ['Ġlad', 'bro', 'kes'] with prime='lad' → prime_word='ladbrokes'
        - Input: ['.', 'first', '_name'] with prime='.first' → prime_word='.first_name'

        Args:
            prefix_tokens: Tokens before the prime token
            prime_token: The token with maximum activation
            suffix_tokens: Tokens after the prime token

        Returns:
            Dict containing:
                - words: List of reconstructed words
                - prime_word: The complete word containing the prime token
                - prime_word_index: Index of prime word in words list
                - prime_is_fragment: Whether the prime token was a word fragment
                - context_string: Full reconstructed context as string
        """
        # Combine all tokens with markers indicating position
        all_tokens = []
        prime_token_index = len(prefix_tokens)  # Index of prime token

        for token in prefix_tokens:
            all_tokens.append(token)
        all_tokens.append(prime_token)
        for token in suffix_tokens:
            all_tokens.append(token)

        # Reconstruct words
        words = []
        current_word_parts = []
        prime_word_index = -1
        prime_in_current_word = False

        for i, token in enumerate(all_tokens):
            # Check if this token starts a new word
            starts_new_word = self._is_word_start(token)

            # Handle special case: first token always starts a word
            if i == 0:
                starts_new_word = True

            if starts_new_word and current_word_parts:
                # Save the current word before starting new one
                completed_word = ''.join(current_word_parts)
                words.append(completed_word)

                # Track if prime was in the word we just completed
                if prime_in_current_word:
                    prime_word_index = len(words) - 1
                    prime_in_current_word = False

                current_word_parts = []

            # Clean the token and add to current word
            cleaned = self._clean_token(token)
            if cleaned:
                current_word_parts.append(cleaned)

            # Track if this is the prime token
            if i == prime_token_index:
                prime_in_current_word = True

        # Don't forget the last word
        if current_word_parts:
            completed_word = ''.join(current_word_parts)
            words.append(completed_word)
            if prime_in_current_word:
                prime_word_index = len(words) - 1

        # Get the prime word
        prime_word = words[prime_word_index] if 0 <= prime_word_index < len(words) else self._clean_token(prime_token)

        # Determine if prime was a fragment
        cleaned_prime = self._clean_token(prime_token)
        prime_is_fragment = prime_word != cleaned_prime and len(prime_word) > len(cleaned_prime)

        # Build context string with proper spacing
        context_string = ' '.join(words)

        return {
            "words": words,
            "prime_word": prime_word,
            "prime_word_index": prime_word_index,
            "prime_is_fragment": prime_is_fragment,
            "context_string": context_string,
            "original_prime_token": cleaned_prime
        }

    def _get_prime_words_from_examples(
        self,
        examples: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract reconstructed prime words from all examples.

        Args:
            examples: List of feature activation examples

        Returns:
            Tuple of (prime_words, reconstruction_details)
            - prime_words: List of reconstructed prime words
            - reconstruction_details: List of full reconstruction info per example
        """
        prime_words = []
        reconstruction_details = []

        for ex in examples:
            prime = ex.get('prime_token', '')
            prefix = ex.get('prefix_tokens', [])
            suffix = ex.get('suffix_tokens', [])

            if prime:
                recon = self._reconstruct_words(prefix, prime, suffix)
                prime_words.append(recon['prime_word'])
                reconstruction_details.append(recon)
            else:
                prime_words.append('')
                reconstruction_details.append({
                    "words": [],
                    "prime_word": '',
                    "prime_word_index": -1,
                    "prime_is_fragment": False,
                    "context_string": '',
                    "original_prime_token": ''
                })

        return prime_words, reconstruction_details

    def _join_context(self, prefix_tokens: List[str], prime: str, suffix_tokens: List[str]) -> str:
        """Join tokens into readable context string."""
        parts = []

        # Join prefix
        for token in prefix_tokens:
            cleaned = self._clean_token(token)
            if cleaned:
                # Check if token starts a new word
                if token.startswith(self.BPE_MARKERS):
                    parts.append(' ')
                parts.append(cleaned)

        # Add prime
        if parts:
            parts.append(' ')
        parts.append(self._clean_token(prime))

        # Join suffix
        for token in suffix_tokens:
            cleaned = self._clean_token(token)
            if cleaned:
                if token.startswith(self.BPE_MARKERS):
                    parts.append(' ')
                parts.append(cleaned)

        return ''.join(parts).strip()

    def analyze_feature(
        self,
        examples: List[Dict[str, Any]],
        feature_id: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive NLP analysis on feature activation examples.

        This method reconstructs full words from BPE subword tokens before analysis,
        providing more accurate POS tagging and NER than analyzing fragments.

        Args:
            examples: List of activation examples with prefix_tokens, prime_token,
                     suffix_tokens, max_activation
            feature_id: Feature identifier for logging

        Returns:
            Dict containing all analysis results
        """
        logger.info(f"Starting NLP analysis for feature {feature_id} with {len(examples)} examples")

        if not examples:
            logger.warning(f"No examples provided for feature {feature_id}")
            return self._empty_analysis()

        # Extract components from examples
        prime_tokens = []  # Original BPE tokens
        prefix_contexts = []
        suffix_contexts = []
        activations = []
        full_contexts = []

        for ex in examples:
            prime = ex.get('prime_token', '')
            prefix = ex.get('prefix_tokens', [])
            suffix = ex.get('suffix_tokens', [])
            activation = ex.get('max_activation', 0.0)

            if prime:
                prime_tokens.append(prime)
                prefix_contexts.append(prefix)
                suffix_contexts.append(suffix)
                activations.append(activation)
                full_contexts.append(self._join_context(prefix, prime, suffix))

        # Reconstruct words from BPE tokens to get full words instead of fragments
        prime_words, reconstruction_details = self._get_prime_words_from_examples(examples)

        # Filter to only examples that had valid primes
        valid_prime_words = [w for w in prime_words if w]

        # Count how many primes were fragments that got reconstructed
        fragment_count = sum(1 for d in reconstruction_details if d.get('prime_is_fragment', False))
        logger.info(f"Feature {feature_id}: {fragment_count}/{len(reconstruction_details)} prime tokens were word fragments")

        # Perform analyses using RECONSTRUCTED prime words for NLP
        # This gives better POS tagging and NER than analyzing fragments
        prime_analysis = self._analyze_prime_tokens_with_words(prime_tokens, valid_prime_words, reconstruction_details)
        context_analysis = self._analyze_context_patterns(prefix_contexts, suffix_contexts, full_contexts)
        activation_analysis = self._analyze_activation_patterns(activations, prime_tokens)
        semantic_clusters = self._cluster_examples(full_contexts, valid_prime_words, activations)

        # Generate summary for prompt
        summary = self._generate_analysis_summary(
            prime_analysis, context_analysis, activation_analysis, semantic_clusters
        )

        result = {
            "prime_token_analysis": prime_analysis,
            "context_patterns": context_analysis,
            "activation_stats": activation_analysis,
            "semantic_clusters": semantic_clusters,
            "summary_for_prompt": summary,
            "num_examples_analyzed": len(examples),
            "word_reconstruction_stats": {
                "total_examples": len(reconstruction_details),
                "fragment_count": fragment_count,
                "fragment_percentage": (fragment_count / len(reconstruction_details) * 100) if reconstruction_details else 0
            },
            "computed_at": datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"Completed NLP analysis for feature {feature_id}")
        return result

    def _analyze_prime_tokens(self, prime_tokens: List[str]) -> Dict[str, Any]:
        """
        Analyze prime tokens: frequency, POS tags, NER.

        Args:
            prime_tokens: List of prime tokens from all examples

        Returns:
            Dict with token statistics
        """
        # Clean tokens
        cleaned_tokens = [self._clean_token(t) for t in prime_tokens]
        cleaned_tokens = [t for t in cleaned_tokens if t]  # Remove empty

        # Frequency distribution
        freq_counter = Counter(cleaned_tokens)
        unique_tokens = list(freq_counter.keys())

        # Lowercase for grouping
        lowercase_counter = Counter(t.lower() for t in cleaned_tokens)

        # POS tagging and NER with spaCy if available
        pos_distribution = Counter()
        ner_entities = []

        nlp = _get_spacy_nlp()
        if nlp:
            # Process unique tokens for POS and NER
            for token in unique_tokens:
                doc = nlp(token)
                for tok in doc:
                    pos_distribution[tok.pos_] += freq_counter[token]
                for ent in doc.ents:
                    ner_entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "count": freq_counter.get(token, 1)
                    })

            # Also process tokens in sentence context for better NER
            sample_contexts = cleaned_tokens[:50]  # Limit for performance
            joined_text = ". ".join(sample_contexts)
            if joined_text:
                doc = nlp(joined_text[:10000])  # Limit text length
                ner_counter = Counter()
                for ent in doc.ents:
                    if ent.text.lower() in [t.lower() for t in unique_tokens[:20]]:
                        ner_counter[(ent.text, ent.label_)] += 1

                # Deduplicate NER results
                ner_entities = [
                    {"text": text, "label": label, "count": count}
                    for (text, label), count in ner_counter.most_common(10)
                ]
        else:
            # Fallback: simple heuristics for POS
            for token in cleaned_tokens:
                if token[0].isupper() and len(token) > 1:
                    pos_distribution["PROPN"] += 1  # Proper noun
                elif token.isdigit():
                    pos_distribution["NUM"] += 1
                elif token in self.STOP_WORDS:
                    pos_distribution["STOP"] += 1
                else:
                    pos_distribution["WORD"] += 1

        # Token type categorization
        token_types = self._categorize_tokens(unique_tokens)

        return {
            "unique_count": len(unique_tokens),
            "total_count": len(cleaned_tokens),
            "unique_tokens": unique_tokens[:30],  # Top 30 for display
            "frequency_distribution": dict(freq_counter.most_common(20)),
            "lowercase_distribution": dict(lowercase_counter.most_common(15)),
            "pos_distribution": dict(pos_distribution),
            "ner_entities": ner_entities[:10],
            "token_types": token_types,
            "most_common_token": freq_counter.most_common(1)[0] if freq_counter else ("", 0),
            "concentration_ratio": freq_counter.most_common(1)[0][1] / len(cleaned_tokens) if freq_counter and cleaned_tokens else 0
        }

    def _analyze_prime_tokens_with_words(
        self,
        prime_tokens: List[str],
        prime_words: List[str],
        reconstruction_details: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze prime tokens with word reconstruction for better NLP accuracy.

        This method uses the reconstructed full words (instead of BPE fragments)
        for POS tagging and NER, which gives significantly better results for
        partial word tokens like 'cy' (from 'normalcy') or 'awk' (from 'Mohawk').

        Args:
            prime_tokens: List of original BPE prime tokens
            prime_words: List of reconstructed full words containing the prime tokens
            reconstruction_details: Full reconstruction info including fragment status

        Returns:
            Dict with token and word statistics, including both original and reconstructed analysis
        """
        # Clean original tokens for frequency analysis
        cleaned_tokens = [self._clean_token(t) for t in prime_tokens]
        cleaned_tokens = [t for t in cleaned_tokens if t]

        # Frequency distribution of original tokens
        token_freq_counter = Counter(cleaned_tokens)
        unique_tokens = list(token_freq_counter.keys())

        # Frequency distribution of reconstructed words
        word_freq_counter = Counter(prime_words)
        unique_words = list(word_freq_counter.keys())

        # Lowercase distributions
        token_lowercase_counter = Counter(t.lower() for t in cleaned_tokens)
        word_lowercase_counter = Counter(w.lower() for w in prime_words if w)

        # POS tagging and NER using RECONSTRUCTED WORDS for better accuracy
        pos_distribution = Counter()
        ner_entities = []

        nlp = _get_spacy_nlp()
        if nlp:
            # Process reconstructed words for POS and NER
            # This is the key improvement - analyzing "normalcy" instead of "cy"
            for word in unique_words:
                if word:
                    doc = nlp(word)
                    for tok in doc:
                        pos_distribution[tok.pos_] += word_freq_counter[word]
                    for ent in doc.ents:
                        ner_entities.append({
                            "text": ent.text,
                            "label": ent.label_,
                            "count": word_freq_counter.get(word, 1)
                        })

            # Also process words in sentence context for better NER
            sample_words = prime_words[:50]  # Limit for performance
            joined_text = ". ".join(w for w in sample_words if w)
            if joined_text:
                doc = nlp(joined_text[:10000])  # Limit text length
                ner_counter = Counter()
                for ent in doc.ents:
                    if ent.text.lower() in [w.lower() for w in unique_words[:20] if w]:
                        ner_counter[(ent.text, ent.label_)] += 1

                # Deduplicate NER results
                ner_entities = [
                    {"text": text, "label": label, "count": count}
                    for (text, label), count in ner_counter.most_common(10)
                ]
        else:
            # Fallback: simple heuristics for POS using reconstructed words
            for word in prime_words:
                if not word:
                    continue
                elif word[0].isupper() and len(word) > 1:
                    pos_distribution["PROPN"] += 1  # Proper noun
                elif word.isdigit():
                    pos_distribution["NUM"] += 1
                elif word.lower() in self.STOP_WORDS:
                    pos_distribution["STOP"] += 1
                else:
                    pos_distribution["WORD"] += 1

        # Token type categorization using reconstructed words
        token_types = self._categorize_tokens(unique_words)

        # Count fragments for reporting
        fragment_count = sum(1 for d in reconstruction_details if d.get('prime_is_fragment', False))

        # Get most common word (not token)
        most_common_word = word_freq_counter.most_common(1)[0] if word_freq_counter else ("", 0)

        return {
            # Original token statistics
            "unique_count": len(unique_tokens),
            "total_count": len(cleaned_tokens),
            "unique_tokens": unique_tokens[:30],
            "frequency_distribution": dict(token_freq_counter.most_common(20)),
            "lowercase_distribution": dict(token_lowercase_counter.most_common(15)),
            # Reconstructed word statistics (for better NLP)
            "unique_words": unique_words[:30],
            "word_frequency_distribution": dict(word_freq_counter.most_common(20)),
            "word_lowercase_distribution": dict(word_lowercase_counter.most_common(15)),
            # NLP analysis (based on reconstructed words)
            "pos_distribution": dict(pos_distribution),
            "ner_entities": ner_entities[:10],
            "token_types": token_types,
            # Summary statistics
            "most_common_token": token_freq_counter.most_common(1)[0] if token_freq_counter else ("", 0),
            "most_common_word": most_common_word,
            "concentration_ratio": token_freq_counter.most_common(1)[0][1] / len(cleaned_tokens) if token_freq_counter and cleaned_tokens else 0,
            # Word reconstruction metadata
            "fragment_count": fragment_count,
            "fragment_percentage": (fragment_count / len(reconstruction_details) * 100) if reconstruction_details else 0,
            "reconstruction_enabled": True
        }

    def _categorize_tokens(self, tokens: List[str]) -> Dict[str, int]:
        """Categorize tokens by type."""
        categories = Counter()

        for token in tokens:
            if not token:
                continue
            elif token.isdigit():
                categories["numbers"] += 1
            elif token[0].isupper() and len(token) > 1:
                categories["capitalized"] += 1
            elif token.lower() in self.STOP_WORDS:
                categories["stop_words"] += 1
            elif len(token) <= 2:
                categories["short_tokens"] += 1
            elif re.match(r'^[^\w\s]', token):
                categories["punctuation"] += 1
            else:
                categories["content_words"] += 1

        return dict(categories)

    def _analyze_context_patterns(
        self,
        prefix_contexts: List[List[str]],
        suffix_contexts: List[List[str]],
        full_contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze context patterns: n-grams, common structures.

        Args:
            prefix_contexts: List of prefix token lists
            suffix_contexts: List of suffix token lists
            full_contexts: List of full context strings

        Returns:
            Dict with context pattern analysis
        """
        # Extract n-grams from prefix and suffix
        prefix_bigrams = Counter()
        prefix_trigrams = Counter()
        suffix_bigrams = Counter()
        suffix_trigrams = Counter()

        for prefix in prefix_contexts:
            cleaned = [self._clean_token(t) for t in prefix if self._clean_token(t)]
            # Get last few tokens (most relevant to prime)
            relevant = cleaned[-5:] if len(cleaned) > 5 else cleaned

            for i in range(len(relevant) - 1):
                bigram = (relevant[i].lower(), relevant[i+1].lower())
                prefix_bigrams[bigram] += 1

            for i in range(len(relevant) - 2):
                trigram = (relevant[i].lower(), relevant[i+1].lower(), relevant[i+2].lower())
                prefix_trigrams[trigram] += 1

        for suffix in suffix_contexts:
            cleaned = [self._clean_token(t) for t in suffix if self._clean_token(t)]
            # Get first few tokens (most relevant to prime)
            relevant = cleaned[:5] if len(cleaned) > 5 else cleaned

            for i in range(len(relevant) - 1):
                bigram = (relevant[i].lower(), relevant[i+1].lower())
                suffix_bigrams[bigram] += 1

            for i in range(len(relevant) - 2):
                trigram = (relevant[i].lower(), relevant[i+1].lower(), relevant[i+2].lower())
                suffix_trigrams[trigram] += 1

        # Extract word patterns around prime
        immediately_before = Counter()
        immediately_after = Counter()

        for prefix in prefix_contexts:
            if prefix:
                last_token = self._clean_token(prefix[-1])
                if last_token and last_token.lower() not in self.STOP_WORDS:
                    immediately_before[last_token.lower()] += 1

        for suffix in suffix_contexts:
            if suffix:
                first_token = self._clean_token(suffix[0])
                if first_token and first_token.lower() not in self.STOP_WORDS:
                    immediately_after[first_token.lower()] += 1

        # Detect common syntactic structures using spaCy if available
        syntactic_patterns = []
        nlp = _get_spacy_nlp()
        if nlp and full_contexts:
            pattern_counter = Counter()
            for context in full_contexts[:30]:  # Limit for performance
                doc = nlp(context[:500])  # Limit context length

                # Extract dependency patterns around root
                for token in doc:
                    if token.dep_ == "ROOT":
                        pattern = f"{token.pos_}-ROOT"
                        children_deps = sorted([child.dep_ for child in token.children])[:3]
                        if children_deps:
                            pattern += f"({','.join(children_deps)})"
                        pattern_counter[pattern] += 1

            syntactic_patterns = [{"pattern": p, "count": c} for p, c in pattern_counter.most_common(5)]

        return {
            "prefix_bigrams": [{"tokens": list(bg), "count": c} for bg, c in prefix_bigrams.most_common(10)],
            "prefix_trigrams": [{"tokens": list(tg), "count": c} for tg, c in prefix_trigrams.most_common(5)],
            "suffix_bigrams": [{"tokens": list(bg), "count": c} for bg, c in suffix_bigrams.most_common(10)],
            "suffix_trigrams": [{"tokens": list(tg), "count": c} for tg, c in suffix_trigrams.most_common(5)],
            "immediately_before": dict(immediately_before.most_common(10)),
            "immediately_after": dict(immediately_after.most_common(10)),
            "syntactic_patterns": syntactic_patterns
        }

    def _analyze_activation_patterns(
        self,
        activations: List[float],
        prime_tokens: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze activation value patterns.

        Args:
            activations: List of max activation values
            prime_tokens: Corresponding prime tokens

        Returns:
            Dict with activation statistics
        """
        if not activations:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "distribution_type": "unknown",
                "high_activation_tokens": [],
                "activation_range_buckets": {}
            }

        arr = np.array(activations)

        # Basic statistics
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        median_val = float(np.median(arr))

        # Determine distribution type
        skewness = 0.0
        if std_val > 0:
            skewness = float(np.mean(((arr - mean_val) / std_val) ** 3))

        if abs(skewness) < 0.5:
            dist_type = "symmetric"
        elif skewness > 0.5:
            dist_type = "right-skewed"
        else:
            dist_type = "left-skewed"

        # Find tokens with highest activations
        sorted_indices = np.argsort(arr)[::-1]
        high_activation_tokens = []
        for idx in sorted_indices[:10]:
            if idx < len(prime_tokens):
                high_activation_tokens.append({
                    "token": self._clean_token(prime_tokens[idx]),
                    "activation": float(arr[idx])
                })

        # Create activation buckets
        buckets = {}
        if max_val > min_val:
            bucket_edges = np.linspace(min_val, max_val, 5)
            for i in range(len(bucket_edges) - 1):
                low, high = bucket_edges[i], bucket_edges[i+1]
                count = np.sum((arr >= low) & (arr < high if i < len(bucket_edges) - 2 else arr <= high))
                buckets[f"{low:.3f}-{high:.3f}"] = int(count)

        return {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "median": median_val,
            "skewness": skewness,
            "distribution_type": dist_type,
            "high_activation_tokens": high_activation_tokens,
            "activation_range_buckets": buckets,
            "coefficient_of_variation": std_val / mean_val if mean_val > 0 else 0.0
        }

    def _cluster_examples(
        self,
        full_contexts: List[str],
        prime_tokens: List[str],
        activations: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Cluster examples by semantic similarity.

        Uses simple token overlap and keyword-based clustering as a fallback
        when sentence transformers are not available.

        Args:
            full_contexts: List of full context strings
            prime_tokens: List of prime tokens
            activations: List of activation values

        Returns:
            List of cluster dicts with label, examples, and representative tokens
        """
        if len(full_contexts) < 3:
            return []

        # Clean prime tokens
        cleaned_primes = [self._clean_token(t).lower() for t in prime_tokens]

        # Group by prime token (simple clustering)
        token_groups = defaultdict(list)
        for idx, token in enumerate(cleaned_primes):
            if token:
                token_groups[token].append(idx)

        # Create clusters from groups with multiple examples
        clusters = []
        used_indices = set()

        # Sort groups by size
        sorted_groups = sorted(token_groups.items(), key=lambda x: len(x[1]), reverse=True)

        for token, indices in sorted_groups[:5]:  # Top 5 clusters
            if len(indices) >= 2:  # At least 2 examples
                cluster = {
                    "label": token,
                    "example_indices": indices[:10],  # Limit to 10 examples
                    "size": len(indices),
                    "representative_tokens": [token],
                    "avg_activation": float(np.mean([activations[i] for i in indices if i < len(activations)]))
                }
                clusters.append(cluster)
                used_indices.update(indices)

        # Try semantic clustering with spaCy vectors if available
        nlp = _get_spacy_nlp()
        if nlp and len(full_contexts) >= 10:
            try:
                # Get context vectors
                vectors = []
                valid_indices = []

                for idx, context in enumerate(full_contexts[:50]):  # Limit for performance
                    if idx not in used_indices and context:
                        doc = nlp(context[:200])  # Limit context length
                        if doc.vector.any():
                            vectors.append(doc.vector)
                            valid_indices.append(idx)

                if len(vectors) >= 5:
                    # Simple K-means clustering
                    from sklearn.cluster import KMeans

                    n_clusters = min(3, len(vectors) // 3)
                    if n_clusters >= 2:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(vectors)

                        # Extract cluster info
                        for cluster_id in range(n_clusters):
                            cluster_indices = [valid_indices[i] for i, l in enumerate(labels) if l == cluster_id]

                            if len(cluster_indices) >= 2:
                                # Find representative tokens for this cluster
                                cluster_primes = [cleaned_primes[i] for i in cluster_indices if i < len(cleaned_primes)]
                                rep_tokens = [t for t, _ in Counter(cluster_primes).most_common(3)]

                                cluster = {
                                    "label": f"semantic_group_{cluster_id + 1}" if not rep_tokens else rep_tokens[0],
                                    "example_indices": cluster_indices[:10],
                                    "size": len(cluster_indices),
                                    "representative_tokens": rep_tokens,
                                    "avg_activation": float(np.mean([activations[i] for i in cluster_indices if i < len(activations)]))
                                }
                                clusters.append(cluster)
            except ImportError:
                logger.debug("sklearn not available for clustering")
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")

        # Sort clusters by size
        clusters.sort(key=lambda x: x["size"], reverse=True)

        return clusters[:5]  # Return top 5 clusters

    def _generate_analysis_summary(
        self,
        prime_analysis: Dict[str, Any],
        context_analysis: Dict[str, Any],
        activation_analysis: Dict[str, Any],
        semantic_clusters: List[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable summary for LLM prompt.

        Args:
            prime_analysis: Prime token analysis results
            context_analysis: Context pattern analysis results
            activation_analysis: Activation statistics
            semantic_clusters: Semantic cluster information

        Returns:
            Formatted summary string for inclusion in labeling prompt
        """
        lines = []

        # Check if word reconstruction was used
        reconstruction_enabled = prime_analysis.get("reconstruction_enabled", False)

        # Prime token/word statistics
        if reconstruction_enabled:
            lines.append("**Prime Word Statistics (reconstructed from BPE tokens):**")

            # Show most common word first (reconstructed)
            most_common_word = prime_analysis.get("most_common_word", ("", 0))
            if most_common_word[0]:
                lines.append(f"- Most common word: \"{most_common_word[0]}\" ({most_common_word[1]} occurrences)")

            # Show word frequency distribution
            word_freq_dist = prime_analysis.get("word_frequency_distribution", {})
            if word_freq_dist:
                top_words = list(word_freq_dist.items())[:5]
                words_str = ", ".join([f'"{w}" ({c}x)' for w, c in top_words])
                lines.append(f"- Top words: {words_str}")

            # Fragment statistics
            fragment_pct = prime_analysis.get("fragment_percentage", 0)
            if fragment_pct > 0:
                lines.append(f"- {fragment_pct:.0f}% of prime tokens were word fragments (reconstructed to full words)")
        else:
            lines.append("**Prime Token Statistics:**")

            most_common = prime_analysis.get("most_common_token", ("", 0))
            if most_common[0]:
                lines.append(f"- Most common token: \"{most_common[0]}\" ({most_common[1]} occurrences)")

            freq_dist = prime_analysis.get("frequency_distribution", {})
            if freq_dist:
                top_tokens = list(freq_dist.items())[:5]
                tokens_str = ", ".join([f'"{t}" ({c}x)' for t, c in top_tokens])
                lines.append(f"- Top tokens: {tokens_str}")

        lines.append(f"- Unique: {prime_analysis.get('unique_count', 0)} / {prime_analysis.get('total_count', 0)} total")

        concentration = prime_analysis.get('concentration_ratio', 0)
        if concentration > 0.3:
            lines.append(f"- High concentration: top item appears in {concentration*100:.0f}% of examples")

        # POS distribution
        pos_dist = prime_analysis.get("pos_distribution", {})
        if pos_dist:
            pos_sorted = sorted(pos_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            pos_str = ", ".join([f"{pos} ({count})" for pos, count in pos_sorted])
            lines.append(f"- POS distribution: {pos_str}")

        # NER entities
        ner_entities = prime_analysis.get("ner_entities", [])
        if ner_entities:
            ner_str = ", ".join([f'{e["text"]} ({e["label"]})' for e in ner_entities[:3]])
            lines.append(f"- Named entities: {ner_str}")

        lines.append("")

        # Context patterns
        lines.append("**Context Patterns:**")

        # Immediately before/after
        before = context_analysis.get("immediately_before", {})
        if before:
            before_str = ", ".join([f'"{w}"' for w in list(before.keys())[:5]])
            lines.append(f"- Words before prime: {before_str}")

        after = context_analysis.get("immediately_after", {})
        if after:
            after_str = ", ".join([f'"{w}"' for w in list(after.keys())[:5]])
            lines.append(f"- Words after prime: {after_str}")

        # N-grams
        prefix_bigrams = context_analysis.get("prefix_bigrams", [])
        if prefix_bigrams:
            bigram_str = ", ".join([f'"{" ".join(bg["tokens"])}"' for bg in prefix_bigrams[:3]])
            lines.append(f"- Common prefix patterns: {bigram_str}")

        suffix_bigrams = context_analysis.get("suffix_bigrams", [])
        if suffix_bigrams:
            bigram_str = ", ".join([f'"{" ".join(bg["tokens"])}"' for bg in suffix_bigrams[:3]])
            lines.append(f"- Common suffix patterns: {bigram_str}")

        lines.append("")

        # Semantic clusters
        if semantic_clusters:
            lines.append("**Semantic Groupings:**")
            for i, cluster in enumerate(semantic_clusters[:3], 1):
                rep_tokens = cluster.get("representative_tokens", [])
                tokens_str = ", ".join([f'"{t}"' for t in rep_tokens[:3]]) if rep_tokens else cluster.get("label", "unknown")
                lines.append(f"- Group {i} ({cluster.get('size', 0)} examples): {tokens_str}")
            lines.append("")

        # Activation statistics
        lines.append("**Activation Pattern:**")
        mean_act = activation_analysis.get("mean", 0)
        max_act = activation_analysis.get("max", 0)
        dist_type = activation_analysis.get("distribution_type", "unknown")
        lines.append(f"- Mean: {mean_act:.4f}, Max: {max_act:.4f}")
        lines.append(f"- Distribution: {dist_type}")

        cv = activation_analysis.get("coefficient_of_variation", 0)
        if cv < 0.3:
            lines.append("- Activations are consistent across examples")
        elif cv > 0.7:
            lines.append("- Activations vary significantly across examples")

        return "\n".join(lines)

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "prime_token_analysis": {
                "unique_count": 0,
                "total_count": 0,
                "unique_tokens": [],
                "frequency_distribution": {},
                "pos_distribution": {},
                "ner_entities": [],
                "token_types": {}
            },
            "context_patterns": {
                "prefix_bigrams": [],
                "suffix_bigrams": [],
                "immediately_before": {},
                "immediately_after": {}
            },
            "activation_stats": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "distribution_type": "unknown"
            },
            "semantic_clusters": [],
            "summary_for_prompt": "No examples available for analysis.",
            "num_examples_analyzed": 0,
            "computed_at": datetime.now(timezone.utc).isoformat()
        }

    def format_for_labeling_prompt(self, analysis: Dict[str, Any]) -> str:
        """
        Format analysis results for inclusion in labeling prompt.

        Args:
            analysis: Full analysis dict from analyze_feature()

        Returns:
            Formatted string ready for prompt insertion
        """
        return analysis.get("summary_for_prompt", "No analysis available.")
