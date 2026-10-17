import { describe, expect, it } from 'vitest';
import {
  BASELINE_INTERCEPT,
  BASELINE_MAX,
  BASELINE_MIN,
  BASELINE_SLOPE,
  DEFAULT_STRENGTH,
  computeBaselineStrength,
} from './steeringStrength';

describe('computeBaselineStrength', () => {
  it('matches the measured optima across the tested frequency range', () => {
    // From experiment c4a273f1 (rounded to 0.1)
    expect(computeBaselineStrength(0.037)).toEqual({ value: 2.8, source: 'auto' });
    expect(computeBaselineStrength(0.214)).toEqual({ value: 2.3, source: 'auto' });
    expect(computeBaselineStrength(0.368)).toEqual({ value: 1.9, source: 'auto' });
    expect(computeBaselineStrength(0.484)).toEqual({ value: 1.6, source: 'auto' });
  });

  it('falls back to the default strength when frequency is unavailable', () => {
    expect(computeBaselineStrength(null)).toEqual({ value: DEFAULT_STRENGTH, source: 'default' });
    expect(computeBaselineStrength(undefined)).toEqual({ value: DEFAULT_STRENGTH, source: 'default' });
    expect(computeBaselineStrength(NaN)).toEqual({ value: DEFAULT_STRENGTH, source: 'default' });
  });

  it('clamps to the [1.0, 3.0] band at the extremes', () => {
    // freq 0 → intercept 2.9, below the 3.0 ceiling
    expect(computeBaselineStrength(0)).toEqual({ value: 2.9, source: 'auto' });
    // very low freq stays under the max
    expect(computeBaselineStrength(0.001).value).toBeLessThanOrEqual(BASELINE_MAX);
    // high freq floors at 1.0 (2.9 - 2.6*1 = 0.3 → clamp)
    expect(computeBaselineStrength(1)).toEqual({ value: BASELINE_MIN, source: 'auto' });
    expect(computeBaselineStrength(0.9).value).toBeGreaterThanOrEqual(BASELINE_MIN);
  });

  it('rounds to one decimal place', () => {
    const { value } = computeBaselineStrength(0.123456);
    expect(Number.isInteger(value * 10)).toBe(true);
  });
});

describe('the slope is load-bearing (MIS-E2E-127)', () => {
  /**
   * Mutation control M22 changed BASELINE_SLOPE from 2.6 to 2.4 and the audit
   * recorded 75 tests green. Re-run during remediation, it is now KILLED by
   * two tests — `matches the measured optima across the tested frequency
   * range` here, and `applyAutoBaseline recomputes every tile` in the store.
   *
   * Those two catch it incidentally, as a side effect of asserting exact
   * values. This block states the invariant directly, so deleting either of
   * them leaves the slope pinned rather than silently unguarded again.
   *
   * Why it matters: the formula sets the default steering strength for every
   * feature the user has not tuned. A silent change to it changes what the
   * product does by default across the whole Steering panel.
   *
   * The three points the ORIGINAL tests sampled cannot distinguish any slope:
   *   freq 0    — the slope term is multiplied by zero
   *   freq 0.9  — clamps to the floor
   *   freq 1.0  — clamps to the floor
   */
  it('produces different values in the mid-range for a different slope', () => {
    // Computed independently of the implementation, from IDL-27's formula.
    const atSlope = (slope: number, freq: number) =>
      Math.round(
        Math.min(BASELINE_MAX, Math.max(BASELINE_MIN, BASELINE_INTERCEPT - slope * freq)) * 10,
      ) / 10;

    expect(atSlope(2.6, 0.5)).toBe(1.6);
    expect(atSlope(2.4, 0.5)).toBe(1.7);
    // ...and the shipped constant is the measured one.
    expect(computeBaselineStrength(0.5).value).toBe(atSlope(2.6, 0.5));
  });

  it('pins the constant itself, so a change must be deliberate', () => {
    expect(BASELINE_SLOPE).toBe(2.6);
    expect(BASELINE_INTERCEPT).toBe(2.9);
  });

  it('the clamp endpoints cannot distinguish slopes — which is why they are not enough', () => {
    // Documents the gap rather than asserting behaviour: at these points every
    // plausible slope agrees, so a test sampling only here proves nothing.
    for (const freq of [0, 0.9, 1]) {
      const a = Math.min(BASELINE_MAX, Math.max(BASELINE_MIN, 2.9 - 2.6 * freq));
      const b = Math.min(BASELINE_MAX, Math.max(BASELINE_MIN, 2.9 - 2.4 * freq));
      expect(a).toBeCloseTo(b, 5);
    }
  });
});
