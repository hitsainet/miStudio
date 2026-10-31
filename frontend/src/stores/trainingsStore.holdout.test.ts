/**
 * A new training form holds out 2% of documents by default.
 *
 * At 0 every number a run reports is in-sample, which is how a per-layer FVU
 * gradient went unverified until a separate post-run evaluation.
 *
 * MUTATION CONTROL: restore `holdout_fraction: 0` in the store's default
 * config -> this test fails.
 */
import { describe, it, expect } from 'vitest';
import { useTrainingsStore } from './trainingsStore';
import { DEFAULT_HOLDOUT_FRACTION } from '../utils/trainingHyperparameters';

describe('trainingsStore default config', () => {
  it('holds out the shared default fraction', () => {
    expect(useTrainingsStore.getState().config.holdout_fraction).toBe(DEFAULT_HOLDOUT_FRACTION);
    expect(DEFAULT_HOLDOUT_FRACTION).toBeGreaterThan(0);
  });
});
