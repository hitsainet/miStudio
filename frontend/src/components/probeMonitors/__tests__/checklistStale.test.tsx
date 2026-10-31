/**
 * Feature 039 (FR-19, D-T7): an exported step whose definition states an old decision reads STALE —
 * in words as well as colour — and names the tool that rebuilds it. A `stale` drawn in the
 * `not_done` style with no word would tell a reader "never exported", which is false.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { BehaviourChecklist } from '../BehaviourChecklist';
import type { BehaviourChecklist as Checklist } from '../../../types/behaviour';

const KEY = `b-${Math.random().toString(16).slice(2, 10)}`;

const CHECKLIST: Checklist = {
  key: KEY,
  status: 'validated',
  steps: [
    {
      id: 'exported', label: 'Primary probe exported', state: 'stale',
      reason: 'primary probe pm_1 has a definition that states an old decision (threshold recalibrated)',
      evidence: ['pm_1'], link: { tab: 'probes' },
      next: [{ tool: 'build_probe_definition', arguments: { probe_id: 'pm_1' }, why: 'rebuild' }],
    },
  ],
};

describe('BehaviourChecklist stale step (039 FR-19)', () => {
  it('says "stale" in words and names the rebuild tool', () => {
    render(<BehaviourChecklist checklist={CHECKLIST} />);
    const item = screen.getByText('Primary probe exported').closest('li') as HTMLElement;
    expect(item.getAttribute('data-state')).toBe('stale');
    expect(item.textContent).toContain('(stale)');
    expect(item.textContent).not.toContain('(not done)');
    expect(screen.getByTestId('checklist-next-exported').textContent).toContain('build_probe_definition');
  });
});
