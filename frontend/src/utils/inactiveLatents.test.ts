/**
 * The training card's inactive-latent window (review round 1, R1-D L2).
 *
 * The windows are the training loop's own arithmetic: `ema * (1 - batch / 50,000) + fired`,
 * inactive below 0.01. The numbers asserted here were computed independently by running
 * that recurrence from its steady state until it crossed 0.01.
 */

import { describe, it, expect } from 'vitest';
import { inactiveLatentsExplanation, inactivityWindowSteps } from './inactiveLatents';

function simulatedWindow(batchSize: number): number {
  const decay = Math.max(0, 1 - batchSize / 50_000);
  let ema = 1 / (1 - decay); // a latent that fired on every step
  let steps = 0;
  while (ema >= 0.01) {
    ema *= decay;
    steps += 1;
  }
  return steps;
}

describe('inactivityWindowSteps', () => {
  it('is 84 silent steps at batch 4,096, not the resampler threshold', () => {
    expect(inactivityWindowSteps(4096)).toBe(84);
  });

  it('matches the recurrence at every batch size the form offers', () => {
    for (const batch of [64, 256, 1024, 2048, 4096, 8192, 16384, 49999]) {
      expect(inactivityWindowSteps(batch), String(batch)).toBe(simulatedWindow(batch));
    }
  });

  it('is one step when a batch covers the whole estimate window, and unknown without a batch', () => {
    expect(inactivityWindowSteps(50_000)).toBe(1);
    expect(inactivityWindowSteps(undefined)).toBeNull();
    expect(inactivityWindowSteps(0)).toBeNull();
  });
});

describe('inactiveLatentsExplanation', () => {
  it('names both windows', () => {
    const text = inactiveLatentsExplanation(4096, 1000);
    expect(text).toContain('about the last 84 steps');
    expect(text).toContain('1,000 consecutive silent steps');
  });
});
