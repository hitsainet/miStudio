/**
 * Permalinks carry SETUP, never results.
 *
 * A link carrying a readout would show its recipient a grid computed on
 * someone else's machine, at some earlier time, possibly against an artifact
 * that has since been refitted. The link says what to read out; the reader's
 * own server produces it.
 *
 * MUTATION CONTROLS:
 *   * encode the readout too                -> "setup only" fails
 *   * carry modelId instead of repo          -> "portable" fails
 *   * trust an unrecognised lens value       -> "falls back" fails
 */

import { describe, expect, it } from 'vitest';
import { decodePermalink, encodePermalink, MAX_PINNED } from './jlensStore';

describe('permalinks', () => {
  it('round-trips model, prompt, lens and pins', () => {
    const link = encodePermalink({
      repo: 'google/gemma-2-2b-it',
      prompt: 'The capital of France is',
      mode: 'DIFF',
      pins: [' Paris', ' France'],
    });
    const back = decodePermalink(link);
    expect(back).toEqual({
      repo: 'google/gemma-2-2b-it',
      prompt: 'The capital of France is',
      mode: 'DIFF',
      pins: [' Paris', ' France'],
    });
  });

  it('carries the REPO id, which is portable, not the local model id', () => {
    const link = encodePermalink({
      repo: 'LiquidAI/LFM2.5-1.2B-Instruct',
      prompt: 'x',
      mode: 'LOGIT_LENS',
      pins: [],
    });
    expect(link).toContain('LiquidAI');
    // `m_xxxxxxxx` ids are local to one installation; a link built from one
    // resolves to nothing, or to a DIFFERENT model, on any other.
    expect(link).not.toMatch(/m_[0-9a-f]{8}/);
  });

  it('encodes setup only — no readout, no artifact contents', () => {
    const link = encodePermalink({
      repo: 'a/b',
      prompt: 'hello',
      mode: 'LOGIT_LENS',
      pins: [],
    });
    for (const leaked of ['top_tokens', 'top_probs', 'meta', 'tokens']) {
      expect(link).not.toContain(leaked);
    }
  });

  it('falls back to the logit lens on an unrecognised mode', () => {
    // The logit lens needs no artifact, so it is the one mode guaranteed to
    // work for whoever opens the link.
    const back = decodePermalink('#jlens?repo=a%2Fb&lens=WISHFUL_LENS');
    expect(back?.mode).toBe('LOGIT_LENS');
  });

  it('caps pins at the same limit the store enforces', () => {
    const many = Array.from({ length: MAX_PINNED + 5 }, (_, i) => `t${i}`);
    const back = decodePermalink(
      encodePermalink({ repo: 'a/b', prompt: 'x', mode: 'LOGIT_LENS', pins: many })
    );
    expect(back?.pins).toHaveLength(MAX_PINNED);
  });

  it('returns null for a fragment that is not a J-Lens link', () => {
    expect(decodePermalink('')).toBeNull();
    expect(decodePermalink('#steering?x=1')).toBeNull();
  });
});
