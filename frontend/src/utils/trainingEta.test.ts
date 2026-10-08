/**
 * The training card's ETA.
 *
 * The rates used here are the real ones observed on `train_c9db4def`: ~362
 * steps/min at steady state, and the 212 -> 362 ramp over its first 1,000 steps
 * that makes a since-start average misleading early on.
 *
 * An ETA that is confidently wrong is worse than no ETA, so most of these tests
 * are about the cases that must return NULL rather than a number.
 */

import { describe, it, expect } from 'vitest';
import {
  STALE_AFTER_MS,
  estimateCompletion,
  etaTooltip,
  formatEtaClock,
  formatRemaining,
  recentRate,
} from './trainingEta';

const NOW = Date.parse('2026-09-18T03:44:00.000Z');

/** `count` metric points at `stepsPerMin`, ending `endedMsAgo` before NOW. */
function window(count: number, stepsPerMin: number, startStep: number, endedMsAgo = 0) {
  const steps: number[] = [];
  const timestamps: string[] = [];
  const intervalMs = 60_000 / stepsPerMin * 250; // one point per 250 steps
  for (let i = 0; i < count; i += 1) {
    steps.push(startStep + i * 250);
    timestamps.push(
      new Date(NOW - endedMsAgo - (count - 1 - i) * intervalMs).toISOString()
    );
  }
  return { steps, timestamps };
}

describe('recentRate', () => {
  it('measures steps per ms across the window', () => {
    const { steps, timestamps } = window(5, 362, 10_000);
    const rate = recentRate(steps, timestamps);
    expect(rate).not.toBeNull();
    expect((rate as number) * 60_000).toBeCloseTo(362, 0);
  });

  it('is null with fewer than two points', () => {
    expect(recentRate([100], ['2026-09-18T03:00:00Z'])).toBeNull();
    expect(recentRate([], [])).toBeNull();
    expect(recentRate(undefined, undefined)).toBeNull();
  });

  it('is null when the run has not advanced, rather than dividing by zero', () => {
    const t = ['2026-09-18T03:00:00Z', '2026-09-18T03:01:00Z'];
    expect(recentRate([500, 500], t)).toBeNull();
  });

  it('is null when time did not advance, rather than returning Infinity', () => {
    const t = ['2026-09-18T03:00:00Z', '2026-09-18T03:00:00Z'];
    expect(recentRate([250, 500], t)).toBeNull();
  });

  it('is null on unparseable timestamps', () => {
    expect(recentRate([250, 500], ['not a date', 'also not'])).toBeNull();
  });
});

describe('estimateCompletion', () => {
  const base = {
    status: 'running',
    currentStep: 20_000,
    totalSteps: 60_000,
    startedAt: new Date(NOW - 55 * 60_000).toISOString(),
    updatedAt: new Date(NOW - 20_000).toISOString(),
    now: NOW,
  };

  it('uses the recent window when it has one', () => {
    const { steps, timestamps } = window(5, 362, 19_000);
    const eta = estimateCompletion({ ...base, recentSteps: steps, recentTimestamps: timestamps });
    expect(eta).not.toBeNull();
    expect(eta!.basis).toBe('recent');
    // 40,000 steps left at 362/min = ~110.5 min
    expect(eta!.remainingMs / 60_000).toBeCloseTo(110.5, 0);
  });

  it('falls back to the since-start average before any metrics arrive', () => {
    const eta = estimateCompletion(base);
    expect(eta).not.toBeNull();
    expect(eta!.basis).toBe('since-start');
    // 20,000 steps in 55 min = 363.6/min; 40,000 left = ~110 min
    expect(eta!.remainingMs / 60_000).toBeCloseTo(110, 0);
  });

  it('THE REASON FOR THE WINDOW: since-start is pessimistic while throughput ramps', () => {
    // Real shape of train_c9db4def: 1,000 steps done, but the average since start
    // is dragged down by the buffer warm-up while the CURRENT rate is already 362.
    const rampAverage = estimateCompletion({
      ...base,
      currentStep: 1_000,
      startedAt: new Date(NOW - 4 * 60_000).toISOString(), // 250 steps/min average
    });
    const { steps, timestamps } = window(4, 362, 250);
    const current = estimateCompletion({
      ...base,
      currentStep: 1_000,
      startedAt: new Date(NOW - 4 * 60_000).toISOString(),
      recentSteps: steps,
      recentTimestamps: timestamps,
    });
    expect(rampAverage!.basis).toBe('since-start');
    expect(current!.basis).toBe('recent');
    // The windowed estimate is the shorter, and by a wide margin (~236 vs ~163 min).
    expect(current!.remainingMs).toBeLessThan(rampAverage!.remainingMs);
    expect(rampAverage!.remainingMs - current!.remainingMs).toBeGreaterThan(30 * 60_000);
  });

  it('is null for a finished run — completed_at already shows the duration', () => {
    for (const status of ['completed', 'failed', 'cancelled', 'stopped', 'COMPLETED']) {
      expect(estimateCompletion({ ...base, status })).toBeNull();
    }
  });

  it('is null for a PAUSED run, immediately — not five minutes later', () => {
    // A paused run keeps a recent updated_at, so the staleness guard would not
    // fire for minutes. Without an explicit check the card would show a
    // completion time for a run that is not advancing.
    const justPaused = { ...base, status: 'paused', updatedAt: new Date(NOW - 5_000).toISOString() };
    expect(estimateCompletion(justPaused)).toBeNull();
    expect(estimateCompletion({ ...justPaused, status: 'PAUSED' })).toBeNull();
  });

  it('is null before the run begins', () => {
    for (const status of ['pending', 'initializing']) {
      expect(estimateCompletion({ ...base, status })).toBeNull();
    }
  });

  it('is null at step 0, when there is no rate to measure', () => {
    expect(estimateCompletion({ ...base, currentStep: 0 })).toBeNull();
  });

  it('is null once the run is at or past its total', () => {
    expect(estimateCompletion({ ...base, currentStep: 60_000 })).toBeNull();
    expect(estimateCompletion({ ...base, currentStep: 60_001 })).toBeNull();
  });

  it('is null when the run has not started', () => {
    expect(estimateCompletion({ ...base, startedAt: null, updatedAt: null })).toBeNull();
  });

  it('is null when the heartbeat is stale — a paused run must not show a receding ETA', () => {
    const stale = new Date(NOW - STALE_AFTER_MS - 1000).toISOString();
    expect(estimateCompletion({ ...base, updatedAt: stale })).toBeNull();
  });

  it('still estimates when the heartbeat is recent', () => {
    const fresh = new Date(NOW - STALE_AFTER_MS + 60_000).toISOString();
    expect(estimateCompletion({ ...base, updatedAt: fresh })).not.toBeNull();
  });

  it('is null rather than Infinity on a zero rate', () => {
    expect(
      estimateCompletion({ ...base, startedAt: new Date(NOW).toISOString() })
    ).toBeNull();
  });

  it('is null on non-finite inputs', () => {
    expect(estimateCompletion({ ...base, totalSteps: NaN })).toBeNull();
    expect(estimateCompletion({ ...base, currentStep: NaN })).toBeNull();
  });
});

describe('formatRemaining', () => {
  it('uses the coarsest useful unit', () => {
    expect(formatRemaining(2 * 3600_000 + 45 * 60_000)).toBe('2h 45m');
    expect(formatRemaining(45 * 60_000)).toBe('45m');
    expect(formatRemaining(30_000)).toBe('30s');
  });

  it('never goes negative', () => {
    expect(formatRemaining(-5000)).toBe('0s');
  });
});

describe('formatEtaClock', () => {
  it('shows a time only when the ETA is today', () => {
    const at = NOW + 60 * 60_000;
    expect(formatEtaClock(at, NOW)).toBe(
      new Date(at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    );
  });

  it('includes the date when the run crosses midnight — 2h45m from 11:44 PM does', () => {
    const lateNight = Date.parse('2026-09-18T03:44:00.000Z');
    const at = lateNight + 165 * 60_000;
    const shown = formatEtaClock(at, lateNight);
    const sameDay = new Date(lateNight).toDateString() === new Date(at).toDateString();
    if (!sameDay) {
      expect(shown).toBe(
        new Date(at).toLocaleString([], {
          month: 'short',
          day: 'numeric',
          hour: 'numeric',
          minute: '2-digit',
        })
      );
    } else {
      expect(shown).not.toContain(',');
    }
  });

  it('renders in the host timezone, whatever it is', () => {
    // Not asserting a literal string: the suite must pass in any TZ. What matters
    // is that the value comes from toLocale*, which uses the host zone.
    const at = NOW + 3600_000;
    expect(formatEtaClock(at, NOW)).toBe(
      new Date(at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    );
  });
});

describe('etaTooltip', () => {
  it('names the basis and the rate, so the estimate can be judged', () => {
    const { steps, timestamps } = window(5, 362, 19_000);
    const eta = estimateCompletion({
      status: 'running',
      currentStep: 20_000,
      totalSteps: 60_000,
      startedAt: new Date(NOW - 55 * 60_000).toISOString(),
      updatedAt: new Date(NOW - 20_000).toISOString(),
      recentSteps: steps,
      recentTimestamps: timestamps,
      now: NOW,
    })!;
    const tip = etaTooltip(eta);
    expect(tip).toContain('362 steps/min');
    expect(tip).toContain('recent step rate');
    expect(tip).toContain('local timezone');
  });

  it('says so when it fell back to the since-start average', () => {
    const eta = estimateCompletion({
      status: 'running',
      currentStep: 20_000,
      totalSteps: 60_000,
      startedAt: new Date(NOW - 55 * 60_000).toISOString(),
      updatedAt: new Date(NOW - 20_000).toISOString(),
      now: NOW,
    })!;
    expect(etaTooltip(eta)).toContain('no recent metrics yet');
  });
});
