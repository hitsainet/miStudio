/**
 * The pin palette must cover the pin cap.
 *
 * Both consumers index with `% PIN_COLORS.length`, so a palette shorter than
 * `MAX_PINNED` does NOT throw — it recycles, and two different pinned tokens
 * are drawn in the same colour in the trajectory chart and the chip list. The
 * user then compares two lines believing they are one token.
 *
 * This is the same shape as the steering cap: raising the feature limit there
 * needed the colour Literal widened with it, and missing that produced a 422.
 * Here it produces no error at all, which is worse.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * raise MAX_PINNED without extending PIN_COLORS -> "covers the cap" fails
 *   * duplicate a colour in the palette             -> "distinct" fails
 */

import { describe, expect, it } from 'vitest';
import { PIN_COLORS } from './utils';
import { MAX_PINNED } from '../../stores/jlensStore';

describe('PIN_COLORS', () => {
  it('covers the pin cap, so no two pins share a colour', () => {
    expect(PIN_COLORS.length).toBeGreaterThanOrEqual(MAX_PINNED);
  });

  it('has no duplicate entries', () => {
    expect(new Set(PIN_COLORS).size).toBe(PIN_COLORS.length);
  });

  it('assigns a distinct colour to every pin index up to the cap', () => {
    // Exercises the actual indexing both consumers use, rather than the
    // length alone — `% length` is where recycling happens.
    const assigned = Array.from(
      { length: MAX_PINNED },
      (_, i) => PIN_COLORS[i % PIN_COLORS.length]
    );
    expect(new Set(assigned).size).toBe(MAX_PINNED);
  });
});
