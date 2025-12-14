/**
 * Token Utilities
 *
 * Shared utilities for handling BPE tokens and word reconstruction.
 * Used by ExampleRow, TokenHighlight, and featureExampleFormatter components.
 *
 * This module provides a single source of truth for token handling across
 * the frontend, matching the backend logic in:
 * - nlp_analysis_service.py: _reconstruct_words()
 * - labeling_context_formatter.py: _reassemble_word_around_prime()
 */

/**
 * BPE markers that indicate the START of a new word.
 * These markers are typically prepended to tokens that begin a new word.
 */
export const WORD_START_MARKERS = ['Ġ', '▁', '\u2581', ' '];

/**
 * BPE markers that indicate CONTINUATION of previous word (BERT style).
 * These markers are prepended to tokens that continue the previous word.
 */
export const CONTINUATION_MARKERS = ['##'];

/**
 * Clean a token by removing BPE markers from various tokenizers.
 * Handles GPT-2, SentencePiece, BERT, and other tokenizer markers.
 *
 * @param token - Raw token string from tokenizer
 * @returns Cleaned token without BPE markers
 */
export const cleanToken = (token: string): string => {
  // Remove surrounding quotes if present
  let cleaned = token.replace(/^["']|["']$/g, '');

  // Remove BPE markers from various tokenizers
  cleaned = cleaned
    .replace(/^Ġ/g, '')       // GPT-2/Llama space marker → start of word
    .replace(/^▁/g, '')       // SentencePiece/T5 space marker → start of word
    .replace(/^\u2581/g, '')  // Escaped SentencePiece marker
    .replace(/^##/g, '')      // BERT continuation marker
    .replace(/^_/g, '');      // Underscore fallback

  return cleaned;
};

/**
 * Check if a token starts a new word based on BPE markers.
 *
 * @param token - Raw token string
 * @returns True if token starts a new word, false if it continues
 */
export const isWordStart = (token: string): boolean => {
  if (!token) return false;

  // Check for explicit continuation markers first (BERT style)
  for (const marker of CONTINUATION_MARKERS) {
    if (token.startsWith(marker)) return false;
  }

  // Check for word start markers
  for (const marker of WORD_START_MARKERS) {
    if (token.startsWith(marker)) return true;
  }

  // No marker = continuation of previous word (GPT-2 style)
  return false;
};

/**
 * Result of word reconstruction.
 */
export interface WordReconstructionResult {
  /** Reconstructed words array */
  words: string[];
  /** The complete word containing the prime token */
  primeWord: string;
  /** Index of the prime word in the words array */
  primeWordIndex: number;
  /** Whether the prime token was a fragment of a larger word */
  primeIsFragment: boolean;
  /** Mapping from token index to word index (for activation aggregation) */
  tokenToWordMap: number[];
  /** Start and end token indices for each word */
  wordTokenRanges: Array<{ start: number; end: number }>;
}

/**
 * Reconstruct full words from BPE subword tokens.
 * This method combines subword tokens back into complete words based on
 * BPE marker conventions. It tracks which word contains the prime token.
 *
 * @param prefixTokens - Tokens before the prime token
 * @param primeToken - The token with maximum activation
 * @param suffixTokens - Tokens after the prime token
 * @returns Object with reconstructed words and prime word info
 */
export const reconstructWords = (
  prefixTokens: string[],
  primeToken: string,
  suffixTokens: string[]
): WordReconstructionResult => {
  // Combine all tokens
  const allTokens = [...prefixTokens, primeToken, ...suffixTokens];
  const primeTokenIndex = prefixTokens.length;

  // Reconstruct words
  const words: string[] = [];
  const tokenToWordMap: number[] = [];
  const wordTokenRanges: Array<{ start: number; end: number }> = [];
  let currentWordParts: string[] = [];
  let currentWordStartIdx = 0;
  let primeWordIndex = -1;
  let primeInCurrentWord = false;

  for (let i = 0; i < allTokens.length; i++) {
    const token = allTokens[i];

    // Check if this token starts a new word
    let startsNewWord = isWordStart(token);

    // First token always starts a word
    if (i === 0) startsNewWord = true;

    if (startsNewWord && currentWordParts.length > 0) {
      // Save the current word before starting new one
      const completedWord = currentWordParts.join('');
      words.push(completedWord);
      wordTokenRanges.push({ start: currentWordStartIdx, end: i - 1 });

      // Track if prime was in the word we just completed
      if (primeInCurrentWord) {
        primeWordIndex = words.length - 1;
        primeInCurrentWord = false;
      }

      currentWordParts = [];
      currentWordStartIdx = i;
    }

    // Clean the token and add to current word
    const cleaned = cleanToken(token);
    if (cleaned) {
      currentWordParts.push(cleaned);
    }

    // Track token to word mapping
    tokenToWordMap.push(words.length);

    // Track if this is the prime token
    if (i === primeTokenIndex) {
      primeInCurrentWord = true;
    }
  }

  // Don't forget the last word
  if (currentWordParts.length > 0) {
    const completedWord = currentWordParts.join('');
    words.push(completedWord);
    wordTokenRanges.push({ start: currentWordStartIdx, end: allTokens.length - 1 });
    if (primeInCurrentWord) {
      primeWordIndex = words.length - 1;
    }
  }

  // Update token to word mapping for last word
  for (let i = currentWordStartIdx; i < allTokens.length; i++) {
    tokenToWordMap[i] = words.length - 1;
  }

  // Get the prime word
  const primeWord =
    primeWordIndex >= 0 && primeWordIndex < words.length
      ? words[primeWordIndex]
      : cleanToken(primeToken);

  // Determine if prime was a fragment
  const cleanedPrime = cleanToken(primeToken);
  const primeIsFragment = primeWord !== cleanedPrime && primeWord.length > cleanedPrime.length;

  return {
    words,
    primeWord,
    primeWordIndex,
    primeIsFragment,
    tokenToWordMap,
    wordTokenRanges,
  };
};

/**
 * Get just the prime word from tokens without full reconstruction.
 * Faster when you only need the prime word, not all reconstructed words.
 *
 * @param prefixTokens - Tokens before the prime token
 * @param primeToken - The token with maximum activation
 * @param suffixTokens - Tokens after the prime token
 * @returns The reconstructed prime word
 */
export const getPrimeWord = (
  prefixTokens: string[],
  primeToken: string,
  suffixTokens: string[]
): string => {
  const { primeWord } = reconstructWords(prefixTokens, primeToken, suffixTokens);
  return primeWord;
};

/**
 * Aggregate activations from tokens to words.
 * Takes the maximum activation among all tokens that form each word.
 *
 * @param activations - Per-token activation values
 * @param wordTokenRanges - Token index ranges for each word
 * @returns Per-word activation values (max of constituent tokens)
 */
export const aggregateActivationsToWords = (
  activations: number[],
  wordTokenRanges: Array<{ start: number; end: number }>
): number[] => {
  return wordTokenRanges.map(({ start, end }) => {
    const wordActivations = activations.slice(start, end + 1);
    return Math.max(...wordActivations, 0);
  });
};

/**
 * Format tokens into a readable string with proper spacing.
 * Uses word reconstruction to combine BPE fragments.
 *
 * @param prefixTokens - Tokens before the prime token
 * @param primeToken - The token with maximum activation
 * @param suffixTokens - Tokens after the prime token
 * @param highlightPrime - Whether to highlight the prime word with brackets
 * @returns Formatted string with words properly spaced
 */
export const formatTokensAsText = (
  prefixTokens: string[],
  primeToken: string,
  suffixTokens: string[],
  highlightPrime: boolean = true
): string => {
  const { words, primeWordIndex } = reconstructWords(prefixTokens, primeToken, suffixTokens);

  const displayParts = words.map((word, idx) => {
    if (highlightPrime && idx === primeWordIndex) {
      return `[${word}]`;
    }
    return word;
  });

  return displayParts.join(' ').trim();
};
