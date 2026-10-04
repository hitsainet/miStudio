/**
 * Byte-level BPE tokens must be DECODED for display, not just de-markered.
 *
 * THE DEFECT. `cleanToken` stripped 'Ġ', '▁', '##' and '_' and stopped there.
 * That handles the space marker and nothing else. Byte-level BPE — GPT-2,
 * Llama 3, LFM2 — maps every raw byte to a printable character, so any token
 * containing a non-ASCII byte arrives as mojibake and stayed that way:
 *
 *     'âĢĻs'  is the byte-level spelling of  "’s"   (U+2019 + 's')
 *
 * On the 16k L12 extraction that was the SECOND most common prime token at
 * 3,015 rows. Every reader of the feature browser saw 'âĢĻs' wherever the
 * corpus used a typographic apostrophe — which in web text is most possessives
 * and most contractions.
 *
 * WHY THE EXISTING TESTS MISSED IT. Every fixture used ASCII: 'Ġhello',
 * '▁world', '##ing'. Marker-stripping is sufficient for ASCII, so the suite was
 * green over a function that mangled a fifth of real evidence.
 *
 * MUTATION CONTROLS (each alone; suite must go red):
 *   T1  make decodeByteLevelToken return its input unchanged
 *         -> every test in 'decodes real corpus tokens' fails
 *   T2  remove the decode call from cleanToken
 *         -> `keeps the browser readable` fails
 *   T3  remove the leading-whitespace trim
 *         -> `strips the decoded space marker` fails
 *   T4  make the decoder throw instead of returning the input on an unknown char
 *         -> `leaves non-byte-level tokens alone` fails
 *   T5  trim BOTH ends instead of leading only
 *         -> `keeps a trailing space, which is content` fails
 */

import { describe, it, expect } from 'vitest';

import { cleanToken, decodeByteLevelToken } from './tokenUtils';

describe('decodes real corpus tokens', () => {
  // Every one of these is a prime_token taken from the live extraction
  // `extr_20260920_070000_sae_sae_7c5b`, not invented.
  it.each([
    ['âĢĻs', '’s'],
    ['âĢĻt', '’t'],
    ['Ġthe', ' the'],
    ['Ġanswer', ' answer'],
    ['ĠLicense', ' License'],
    ['assistant', 'assistant'],
    ['-step', '-step'],
  ])('decodes %s to %s', (raw, expected) => {
    expect(decodeByteLevelToken(raw)).toBe(expected);
  });

  it('decodes the newline marker to an actual newline', () => {
    expect(decodeByteLevelToken('Ċ')).toBe('\n');
  });

  it('leaves non-byte-level tokens alone', () => {
    // SentencePiece and special tokens contain characters outside the table.
    // Mangling them would be worse than returning them untouched.
    expect(decodeByteLevelToken('▁the')).toBe('▁the');
    expect(decodeByteLevelToken('<|endoftext|>')).toBe('<|endoftext|>');
  });
});

describe('cleanToken', () => {
  it('keeps the browser readable', () => {
    // The headline: this is what a reader saw before.
    expect(cleanToken('âĢĻs')).toBe('’s');
  });

  it('strips the decoded space marker', () => {
    // Decoding turns 'Ġ' into a real space, so the old regex cannot remove it.
    expect(cleanToken('Ġthe')).toBe('the');
    expect(cleanToken('Ġanswer')).toBe('answer');
  });

  it('keeps a trailing space, which is content', () => {
    // 'theĠ' decodes to 'the ' — the space belongs to the token and dropping it
    // silently joins words when examples are reassembled.
    expect(cleanToken('theĠ')).toBe('the ');
  });

  it('still handles the SentencePiece form it always handled', () => {
    expect(cleanToken('▁world')).toBe('world');
  });

  it('still strips BERT continuation markers', () => {
    expect(cleanToken('##ing')).toBe('ing');
  });

  it('does not mangle a plain word', () => {
    expect(cleanToken('neural')).toBe('neural');
  });
});
