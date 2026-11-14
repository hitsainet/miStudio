import { describe, expect, it } from 'vitest';
import { cleanDisplayText } from './tokenDisplay';

describe('cleanDisplayText', () => {
  it('strips GPT-2 (Ġ) word-boundary markers into spaces', () => {
    expect(cleanDisplayText('Ġhave Ġmore Ġnews letters')).toBe('have more news letters');
  });

  it('strips SentencePiece (▁) markers', () => {
    expect(cleanDisplayText('▁fear')).toBe('fear');
  });

  it('strips WordPiece (##) continuation markers', () => {
    expect(cleanDisplayText('token ##ization')).toBe('token ization');
  });

  it('preserves emphasis markers around the prime token', () => {
    expect(cleanDisplayText('Ġthanks Ġfor *Ġfear* Ġyou')).toBe('thanks for *fear* you');
  });

  it('collapses the spaces markers introduce and trims', () => {
    expect(cleanDisplayText('Ġa  Ġb')).toBe('a b');
  });

  it('leaves plain text unchanged', () => {
    expect(cleanDisplayText('article_subscription')).toBe('article_subscription');
  });

  it('handles null/undefined/empty safely', () => {
    expect(cleanDisplayText(null)).toBe('');
    expect(cleanDisplayText(undefined)).toBe('');
    expect(cleanDisplayText('')).toBe('');
  });
});
