/** ENH-006 D84: the report states the catch rate and the nonsense-text result in the server's plain words. */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { CatchRateLine } from '../CatchRateLine';
import { NonsenseTextLine } from '../EverydayTextLine';
import type { CatchRateCheck, EverydayTextCheck } from '../../../types/probeMonitor';

const catchRate = (over: Partial<CatchRateCheck>): CatchRateCheck => ({
  status: 'too_few_caught', passed: false, min_recall: 0.5, next_step: null,
  reason: 'catches 10% of positives on unseen tasks at the shipped bar (needs ≥ 50%) on carriers: 41 of 410 positives',
  sets: [{ name: 'carriers (pmd_c)', positives: 410, caught: 41, threshold: 4.3, recall: 0.1, needed: 205,
    meets: false, note: null,
    sentence: 'On unseen tasks (carriers (pmd_c)): catches 10% of positives at the shipped bar (41 of 410; needs ≥ 50%)' }],
  ...over,
});
const nonsense = (over: Partial<EverydayTextCheck>): EverydayTextCheck => ({
  kind: 'nonsense_text', status: 'not_checked', passed: false, next_step: null, target_fpr: 0.01, factor: 2,
  min_rows: 1000, allowed_rate: 0.02, sets: [],
  reason: 'not yet checked on nonsense text — evaluate on a nonsense-text set', ...over,
});

describe('CatchRateLine', () => {
  it('states the catch rate at the shipped bar and why rung 2 is not reached', () => {
    render(<CatchRateLine check={catchRate({})} />);
    expect(screen.getByTestId('catch-rate-set')).toHaveTextContent(
      'catches 10% of positives at the shipped bar (41 of 410; needs ≥ 50%)');
    expect(screen.getByTestId('catch-rate')).toHaveTextContent(
      'Rung 2 not reached: catches 10% of positives on unseen tasks at the shipped bar (needs ≥ 50%)');
  });

  it('says nothing without an out-of-distribution set or a verdict', () => {
    const { rerender } = render(<CatchRateLine check={catchRate({ status: 'no_sets', sets: [] })} />);
    expect(screen.queryByTestId('catch-rate')).toBeNull();
    rerender(<CatchRateLine check={null} />);
    expect(screen.queryByTestId('catch-rate')).toBeNull();
  });
});

describe('NonsenseTextLine', () => {
  it('says when it was never checked', () => {
    render(<NonsenseTextLine check={nonsense({})} />);
    expect(screen.getByTestId('nonsense-text')).toHaveTextContent(
      'Nonsense text: not yet checked on nonsense text — evaluate on a nonsense-text set');
    expect(screen.getByTestId('nonsense-text')).toHaveTextContent('at least 1,000 rows');
  });

  it('states a failing set with its numbers', () => {
    render(<NonsenseTextLine check={nonsense({
      status: 'too_many_false_alarms',
      reason: 'false alarms on nonsense text 2.05% at the shipped bar (allowed ≤ 2%)',
      sets: [{ name: 'salad', rows: 2000, false_alarms: 41, threshold: 1, counts: true, allowed: 40, fpr: 0.0205,
        within_allowance: false, note: null,
        sentence: 'On nonsense text: 41 false alarms in 2,000 (allowed up to 40)' }],
    })} />);
    expect(screen.getByTestId('nonsense-text-set')).toHaveTextContent(
      'On nonsense text: 41 false alarms in 2,000 (allowed up to 40)');
    expect(screen.getByTestId('nonsense-text')).toHaveTextContent('Rung 2 not reached: false alarms on nonsense text 2.05%');
  });
});
