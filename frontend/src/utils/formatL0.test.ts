/**
 * L0 is a count of active features. Never show a live SAE as zero.
 *
 * Reported 2026-08-26 on train_9355afa6: a healthy JumpReLU SAE (FVU 0.092,
 * 0 dead neurons, ~7 of 30,720 latents active) displayed "L0: 0.000" in the
 * checkpoint list and "0.0%" in the metrics summary. Both rendered the RAW
 * FRACTION the trainer stores -- `(z > 0).float().mean()`, 0.000228 here --
 * so a good result read as a collapsed dictionary. That is the kind of number
 * a run gets discarded over.
 *
 * The main card row was already correct via formatL0Absolute; two other
 * surfaces were not.
 */

import { describe, it, expect } from 'vitest';
import { formatL0Absolute, formatL0Percent } from './formatters';

describe('formatL0Absolute', () => {
  const LATENTS = 30720;

  it('reports the real production value as a count, not a fraction', () => {
    // train_9355afa6: current_l0_sparsity = 0.00022773744422011077
    expect(formatL0Absolute(0.00022773744422011077, LATENTS)).toBe('~7.0');
  });

  it('never rounds a live SAE down to zero', () => {
    // 0.4 features active on average: sparse, not dead.
    const fraction = 0.4 / LATENTS;
    expect(formatL0Absolute(fraction, LATENTS)).toBe('<1');
    expect(formatL0Absolute(fraction, LATENTS)).not.toBe('0');
  });

  it('keeps a decimal in the single digits, where it matters', () => {
    expect(formatL0Absolute(7.4 / LATENTS, LATENTS)).toBe('~7.4');
    expect(formatL0Absolute(9.9 / LATENTS, LATENTS)).toBe('~9.9');
  });

  it('rounds once the count is large enough for a decimal to be noise', () => {
    expect(formatL0Absolute(71 / LATENTS, LATENTS)).toBe('~71');
    expect(formatL0Absolute(2458 / LATENTS, LATENTS)).toBe('~2458');
  });

  it('still says zero when the SAE really is dead', () => {
    expect(formatL0Absolute(0, LATENTS)).toBe('0');
  });

  it('does not invent a number when inputs are unusable', () => {
    expect(formatL0Absolute(NaN, LATENTS)).toBe('—');
    expect(formatL0Absolute(0.0002, NaN)).toBe('—');
  });

  it('percent remains available for when the latent count is unknown', () => {
    expect(formatL0Percent(0.0002)).toBe('0.0%');
  });
});
