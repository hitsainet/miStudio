/**
 * MIS-E2E-129 — the Diff view shaded agreement as disagreement.
 *
 * `rankOf` returns `i + 1` — one-based, and therefore NEVER 0. `diffColor` and
 * the tooltip were written for a zero-based rank. Four visible consequences:
 *
 *   * rank 1 — the two lenses agreeing on the top token, which is precisely
 *     what the Diff view exists to locate — fell through to the amber ramp and
 *     was shaded as DISAGREEMENT;
 *   * every tooltip rank was reported one too high (`#2` for rank 1);
 *   * the "same top token" legend swatch was UNREACHABLE — the cheap tell that
 *     an index base is wrong;
 *   * the shading contradicted the `first diverge at L…` badge beside it,
 *     which `firstDisagreement` computes correctly.
 *
 * `rankColor` cannot move to zero-based (it takes `Math.log(rank)`, so rank 0
 * is -Infinity), and "rank 1 = best" is what the word means — so the base is 1
 * everywhere and `diffColor` moved, not `rankOf`.
 *
 * MUTATION CONTROLS: restore `rank === 0` in diffColor, or `rank / span` in the
 * ramp, and these go red.
 */

import { describe, it, expect } from 'vitest';
import { diffColor, rankColor } from './utils';
import { rankOf } from '../../stores/jlensStore';

const AGREE = 'rgba(100,116,139,.16)';
const ABSENT = 'rgba(244,63,94,.34)';

function slice(rows: string[][]) {
  return { top_tokens: rows } as any;
}

describe('diffColor and rankOf agree on one index base', () => {
  it('rankOf is one-based and never returns 0', () => {
    const s = slice([[' Paris', ' Rome', ' Berlin']]);
    expect(rankOf(s, 0, ' Paris')).toBe(1);
    expect(rankOf(s, 0, ' Rome')).toBe(2);
    expect(rankOf(s, 0, ' Nowhere')).toBeNull();
  });

  it('shades the top token as AGREEMENT, not disagreement', () => {
    // The defect in one line: rankOf returns 1 for the top token, and
    // diffColor(1) used to fall through to the amber ramp.
    const s = slice([[' Paris', ' Rome']]);
    const r = rankOf(s, 0, ' Paris');
    expect(diffColor(r, 8)).toBe(AGREE);
  });

  it('the "same top token" swatch is reachable from a real rank', () => {
    // An unreachable legend entry is the tell. The legend renders
    // diffColor(1, topN); some cell must be able to receive it.
    const s = slice([[' Paris']]);
    expect(diffColor(rankOf(s, 0, ' Paris'), 8)).toBe(diffColor(1, 8));
  });

  it('a token outside the top N is the strongest disagreement', () => {
    expect(diffColor(null, 8)).toBe(ABSENT);
  });

  it('ramps from rank 2 upward, monotonically', () => {
    const two = diffColor(2, 8);
    const eight = diffColor(8, 8);
    expect(two).not.toBe(AGREE);
    expect(two).not.toBe(eight);
    const alpha = (c: string) => parseFloat(c.split(',')[3]);
    expect(alpha(eight)).toBeGreaterThan(alpha(two));
  });

  it('rank 2 does not start the ramp mid-way', () => {
    // With the old zero-based ramp, rank 2 landed at t = 2/7 instead of 1/7 —
    // every disagreement was reported as worse than it was.
    const span = Math.max(8 - 1, 1);
    const expected = (0.18 + 0.16 * (1 / span)).toFixed(3);
    expect(diffColor(2, 8)).toBe(`rgba(245,158,11,${expected})`);
  });

  it('a top-1 readout does not divide by zero', () => {
    expect(diffColor(1, 1)).toBe(AGREE);
    expect(() => diffColor(2, 1)).not.toThrow();
  });

  it('rankColor still requires one-based — it is why rankOf stays as it is', () => {
    // Negative control for the DIRECTION of the fix. Moving `rankOf` to
    // zero-based instead of moving `diffColor` would take `Math.log(0)` here:
    // alpha becomes `Math.max(0.06, 1 - (-Infinity)/k)` = Infinity, an invalid
    // alpha the browser discards — so every top-ranked cell would silently
    // lose its shading. Not NaN, which is what this test first asserted;
    // checked rather than assumed.
    expect(rankColor(1, 8)).toBe('rgba(52, 211, 153, 1.000)');
    const zero = rankColor(0, 8);
    expect(Number.isFinite(parseFloat(zero.split(',')[3]))).toBe(false);
  });
});
