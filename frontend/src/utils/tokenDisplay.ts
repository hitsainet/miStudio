/**
 * Utility functions for displaying tokens in a human-readable format.
 *
 * Handles subword tokenization (BPE) by intelligently merging adjacent tokens
 * that are parts of the same word.
 */

/**
 * Join tokens with proper spacing, reuniting subword tokens.
 *
 * Many tokenizers (GPT-2, GPT-3, etc.) use special characters to mark word boundaries:
 * - 'Ġ' (U+0120) indicates the start of a new word
 * - Tokens without this character are continuations of the previous word
 *
 * Examples:
 * - ["Ġtoken", "ization"] → "tokenization"
 * - ["Ġhello", "Ġworld"] → "hello world"
 * - ["Ġun", "der", "stand", "ing"] → "understanding"
 *
 * @param tokens - Array of token strings
 * @returns Human-readable text with proper spacing
 */
export function joinTokensWithProperSpacing(tokens: string[]): string {
  if (!tokens || tokens.length === 0) {
    return '';
  }

  const result: string[] = [];

  for (const token of tokens) {
    // Check if this token starts a new word (has leading space marker 'Ġ')
    if (token.startsWith('Ġ')) {
      // Remove the marker and add as new word with space
      const cleanToken = token.slice(1);
      if (result.length > 0) {
        result.push(' ');
      }
      result.push(cleanToken);
    } else if (token.startsWith(' ')) {
      // Some tokenizers use actual space character
      const cleanToken = token.slice(1);
      if (result.length > 0) {
        result.push(' ');
      }
      result.push(cleanToken);
    } else {
      // Token is continuation of previous word - append directly
      result.push(token);
    }
  }

  return result.join('');
}

/**
 * Alternative: Simple join with space fallback for tokenizers without markers.
 * Use this if the tokenizer doesn't use space markers (rare for modern LLMs).
 */
export function joinTokensSimple(tokens: string[]): string {
  return tokens.join(' ');
}

/**
 * Format tokens for display with optional highlighting.
 *
 * @param prefixTokens - Tokens before the highlighted section
 * @param primeToken - The highlighted token
 * @param suffixTokens - Tokens after the highlighted section
 * @returns Object with formatted text sections
 */
export function formatTokensForDisplay(
  prefixTokens: string[],
  primeToken: string,
  suffixTokens: string[]
): {
  prefix: string;
  prime: string;
  suffix: string;
  full: string;
} {
  const prefix = joinTokensWithProperSpacing(prefixTokens);
  const prime = joinTokensWithProperSpacing([primeToken]);
  const suffix = joinTokensWithProperSpacing(suffixTokens);

  // Full text needs special handling to ensure proper spacing around prime token
  let full = prefix;

  // Add space before prime if needed (prime starts a new word)
  if (primeToken.startsWith('Ġ') || primeToken.startsWith(' ')) {
    if (full.length > 0) {
      full += ' ';
    }
    full += prime;
  } else {
    // Prime continues previous word
    full += prime;
  }

  // Add space before suffix if needed
  if (suffixTokens.length > 0) {
    const firstSuffix = suffixTokens[0];
    if (firstSuffix.startsWith('Ġ') || firstSuffix.startsWith(' ')) {
      full += ' ';
    }
    full += suffix;
  }

  return { prefix, prime, suffix, full };
}

/**
 * Check if a token string contains the GPT-2 style space marker.
 */
export function hasSpaceMarker(token: string): boolean {
  return token.startsWith('Ġ') || token.startsWith(' ');
}

/**
 * Remove space markers from a token for clean display.
 */
export function cleanToken(token: string): string {
  if (token.startsWith('Ġ')) {
    return token.slice(1);
  }
  if (token.startsWith(' ')) {
    return token.trim();
  }
  return token;
}
