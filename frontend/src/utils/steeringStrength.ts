/**
 * Steering baseline-strength calculation (Feature 011).
 *
 * A feature's optimal starting steering strength is predictable from its
 * activation frequency: denser features (active on more tokens) need weaker
 * steering. Measured on the LFM2.5-1.2B L12 SAE (experiment c4a273f1) across
 * frequency 0.037–0.484, the optima fit:
 *
 *     S ≈ clamp(2.9 − 2.6 · activation_frequency, 1.0, 3.0)
 *
 * `max_activation` is deliberately NOT used: the SAE's decoder columns are
 * unit-norm, so the injected steering vector's magnitude equals the raw
 * strength coefficient — a feature's activation ceiling doesn't change how
 * hard steering hits. (See PADR IDL-27.)
 *
 * The 2.9/2.6 constants are SAE-local but treated as global this phase; the
 * manual strength control and the default fallback always remain available.
 */

/** Strength assigned when a feature is added without a frequency to derive from. */
export const DEFAULT_STRENGTH = 10;

/** Coefficients of the frequency→strength line (SAE-local; see IDL-27). */
export const BASELINE_INTERCEPT = 2.9;
export const BASELINE_SLOPE = 2.6;
export const BASELINE_MIN = 1.0;
export const BASELINE_MAX = 3.0;

export type StrengthSource = 'auto' | 'default' | 'manual' | 'cluster';

export interface BaselineStrength {
  value: number;
  source: StrengthSource;
}

/**
 * Compute a starting steering strength for a feature from its activation
 * frequency. Falls back to {@link DEFAULT_STRENGTH} when frequency is unknown.
 *
 * @param freq activation_frequency in [0, 1], or null/undefined if unavailable
 */
export function computeBaselineStrength(
  freq: number | null | undefined,
): BaselineStrength {
  if (freq == null || Number.isNaN(freq)) {
    return { value: DEFAULT_STRENGTH, source: 'default' };
  }
  const raw = BASELINE_INTERCEPT - BASELINE_SLOPE * freq;
  const clamped = Math.min(BASELINE_MAX, Math.max(BASELINE_MIN, raw));
  // Round to 0.1 — steering strengths are entered at that granularity.
  return { value: Math.round(clamped * 10) / 10, source: 'auto' };
}
