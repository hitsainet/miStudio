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
 * Common suffixes that typically indicate continuation of a word.
 * Used for heuristic word boundary detection when markers are absent.
 */
const COMMON_SUFFIXES = [
  'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ment', 'ness', 'able', 'ible',
  'ful', 'less', 'ous', 'ive', 'al', 'ial', 'ical', 'ence', 'ance', 'ty', 'ity',
  's', 'es', 'ies', 'ating', 'tion', 'ness', 'ment', 'ously', 'ously', 'ized',
  'ization', 'ified', 'ification', 'ered', 'ering', 'ular', 'ularly',
];

/**
 * Common short words that are NOT suffixes (standalone words).
 * These should be treated as word starts even though they're short.
 */
const COMMON_SHORT_WORDS = new Set([
  'a', 'i', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is',
  'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we',
  'am', 'are', 'was', 'for', 'the', 'and', 'but', 'not', 'you', 'all', 'can',
  'had', 'her', 'him', 'his', 'how', 'its', 'let', 'may', 'new', 'now', 'old',
  'see', 'way', 'who', 'boy', 'did', 'get', 'has', 'man', 'out', 'put', 'say',
  'she', 'too', 'use', 'one', 'two', 'ten', 'tom', 'bob', 'joe', 'max', 'day',
]);

/**
 * Check if a token array contains BPE markers.
 * Used to determine whether to use marker-based or heuristic word reconstruction.
 *
 * @param tokens - Array of token strings
 * @returns True if any token has a BPE marker, false otherwise
 */
export const hasMarkers = (tokens: string[]): boolean => {
  if (!tokens || tokens.length === 0) return false;

  for (const token of tokens) {
    if (!token) continue;

    // Check for word start markers
    for (const marker of WORD_START_MARKERS) {
      if (token.startsWith(marker)) return true;
    }

    // Check for continuation markers
    for (const marker of CONTINUATION_MARKERS) {
      if (token.startsWith(marker)) return true;
    }
  }

  return false;
};

/**
 * Infer word boundary using heuristics when BPE markers are not present.
 * This is a fallback for data extracted before the marker preservation fix.
 *
 * Heuristics used:
 * 1. First token always starts a word
 * 2. Common suffixes (-ing, -ed, -tion, etc.) = continuation
 * 3. After punctuation = new word
 * 4. Capitalized after lowercase = new word
 * 5. Short tokens that aren't common words after longer tokens = likely continuation
 *
 * @param token - Current token to analyze
 * @param index - Index of the token in the sequence
 * @param allTokens - Complete array of tokens for context
 * @returns True if token likely starts a new word
 */
export const inferWordStartHeuristic = (
  token: string,
  index: number,
  allTokens: string[]
): boolean => {
  // First token always starts a word
  if (index === 0) return true;

  if (!token) return true;

  const prevToken = allTokens[index - 1] || '';
  const tokenLower = token.toLowerCase();

  // After punctuation = new word (except apostrophe)
  if (prevToken && /[.!?,;:\-\(\)\[\]{}]$/.test(prevToken)) {
    return true;
  }

  // Capitalized after lowercase = new word (e.g., "the United")
  if (token[0] === token[0]?.toUpperCase() && /[a-z]/.test(prevToken.slice(-1))) {
    return true;
  }

  // Common suffixes = likely continuation
  for (const suffix of COMMON_SUFFIXES) {
    if (tokenLower === suffix || tokenLower.endsWith(suffix)) {
      // But check if it's also a common word
      if (!COMMON_SHORT_WORDS.has(tokenLower)) {
        return false; // Continuation
      }
    }
  }

  // Short token (1-3 chars) after longer token might be continuation
  // unless it's a common short word
  if (token.length <= 3 && prevToken.length > 3) {
    if (!COMMON_SHORT_WORDS.has(tokenLower)) {
      // Likely a suffix/continuation
      return false;
    }
  }

  // Default: treat as new word (safer for readability)
  return true;
};

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
 * @param index - Optional token index (required for heuristic mode)
 * @param allTokens - Optional complete token array (required for heuristic mode)
 * @param useHeuristics - If true, use heuristics when no markers are found
 * @returns True if token starts a new word, false if it continues
 */
export const isWordStart = (
  token: string,
  index?: number,
  allTokens?: string[],
  useHeuristics?: boolean
): boolean => {
  if (!token) return false;

  // Check for explicit continuation markers first (BERT style)
  for (const marker of CONTINUATION_MARKERS) {
    if (token.startsWith(marker)) return false;
  }

  // Check for word start markers
  for (const marker of WORD_START_MARKERS) {
    if (token.startsWith(marker)) return true;
  }

  // If heuristics are enabled and we have context, use them
  if (useHeuristics && index !== undefined && allTokens) {
    return inferWordStartHeuristic(token, index, allTokens);
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
 * Automatically detects whether tokens have BPE markers and uses appropriate
 * strategy: marker-based (accurate) or heuristic-based (fallback for legacy data).
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

  // Detect if tokens have BPE markers - if not, use heuristics
  const useHeuristics = !hasMarkers(allTokens);

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

    // Check if this token starts a new word (with heuristic fallback if needed)
    let startsNewWord = isWordStart(token, i, allTokens, useHeuristics);

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
