/**
 * Feature 041 (FTASKS 3.2): the family section, rendered from a random-key fixture with only the network
 * mocked. The request is asserted by its arguments and call count; every word on screen is the fixture's.
 */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { APIError } from '../../../api/client';
import type { ProbeFamily } from '../../../types/probeFamily';

const api = vi.hoisted(() => ({
  getFamily: vi.fn(),
  listReports: vi.fn().mockResolvedValue({ behaviour_key: 'k', reports: [], offset: 0, limit: 25, next_offset: null }),
  getSuggestedBlendHistory: vi.fn().mockResolvedValue({ key: 'k', model: 'm', entries: [], next_offset: null }),
  exportUrl: vi.fn(() => '/api/v1/x'),
}));
vi.mock('../../../api/probeFamilies', () => ({ probeFamiliesApi: api }));

import { useProbeFamiliesStore } from '../../../stores/probeFamiliesStore';
import { FamilySection, flagWords } from '../FamilySection';

const slug = () => {
  const part = () => Array.from({ length: 5 }, () => 'abcdefghijklmnopqrstuvwxyz'[Math.floor(Math.random() * 26)]).join('');
  return `${part()}-${part()}`;
};

function member(id: string, overrides: Record<string, unknown> = {}) {
  return {
    probe_id: id, run_id: `pmr_${id}`, model_id: 'm_x', role: 'candidate', role_recorded: true, precision: 'bfloat16',
    precision_refusal: null, render_form: 'served', scope: 'all', layer: 3, rule: 'mean', variant: 'dense',
    sub_question: null, behaviour_version: 1, rung: 2, rung_language: 'detects on unseen tasks', threshold: 0.5,
    decision_revision: 1, decision_digest: null, detector_digest: null, identity_current: null, training_view: 'pmd_t',
    training_dataset_version: 'dv', definition_built: true, default_blend: true, flags: [], ...overrides,
  };
}

function family(key: string, sub: string, missing: string): ProbeFamily {
  return {
    key, version: 2, model: 'm_x', models: [{ model_id: 'm_x', members: 2 }], declared_sub_questions: [missing, sub],
    coverage: [
      { sub_question: missing, state: 'missing', members: [], next: [{ tool: 'start_probe_training_run',
        arguments: { model_id: 'm_x', train_dataset_id: 'pmd_s', behaviour: key }, why: 'train the specialist' }] },
      { sub_question: sub, state: 'covered', members: ['pm_b'] },
      { sub_question: null, group: 'main_question', state: 'covered', members: ['pm_a'] },
    ],
    default_blend_set: ['pm_a', 'pm_b'], flags_basis: 'majority_of_all_members',
    members: [
      member('pm_a', { role: 'primary' }),
      member('pm_b', { sub_question: sub, precision: 'float32',
        flags: [{ flag: 'differs_precision', field: 'precision', value: 'float32', majority: 'bfloat16' }] }),
    ] as ProbeFamily['members'],
    total: 2, offset: 0, limit: 50, next_offset: null,
  };
}

beforeEach(() => {
  api.getFamily.mockReset();
  api.listReports.mockClear();
  useProbeFamiliesStore.setState({ key: null, model: null, family: null, models: [], includeRetired: false,
    loading: false, refusal: null });
});

describe('FamilySection', () => {
  it('shows coverage first with the missing sub-question and its call, then the members with flags in words', async () => {
    const [key, sub, missing] = [slug(), slug(), slug()];
    api.getFamily.mockResolvedValue(family(key, sub, missing));
    render(<FamilySection behaviourKey={key} />);
    await screen.findByText(/differs in precision: float32 vs bfloat16/);
    expect(api.getFamily).toHaveBeenCalledTimes(1);
    expect(api.getFamily).toHaveBeenCalledWith(key, { model: null, offset: 0, include_retired: false });
    const coverage = screen.getByLabelText('Sub-question coverage');
    const items = within(coverage).getAllByRole('listitem');
    expect(items[0]).toHaveTextContent(`Sub-question ${missing}: missing`);
    expect(items[0]).toHaveTextContent('start_probe_training_run');
    expect(within(screen.getByTestId('member-pm_a')).getByText('detects on unseen tasks')).toBeInTheDocument();
    // S2/S3 (FR-28): the reports, the suggested blend and the export are mounted for the family's model.
    expect(await screen.findByLabelText('Blend reports')).toBeInTheDocument();
    await waitFor(() => expect(api.listReports).toHaveBeenCalledTimes(1));
    expect(api.listReports).toHaveBeenCalledWith(key, 'm_x');
    expect(screen.getByLabelText('Suggested blend')).toBeInTheDocument();
    expect(screen.getByLabelText('Family export')).toBeInTheDocument();
  });

  it('offers the models a model_required refusal lists, and loads the chosen one', async () => {
    const key = slug();
    api.getFamily.mockRejectedValueOnce(new APIError(422, 'model required', {
      code: 'model_required', message: 'name one', next: [],
      models: [{ model_id: 'm_one', members: 2 }, { model_id: 'm_two', members: 1 }] }));
    api.getFamily.mockResolvedValueOnce(family(key, slug(), slug()));
    render(<FamilySection behaviourKey={key} />);
    const picker = await screen.findByLabelText('Family model');
    await userEvent.selectOptions(picker, 'm_two');
    await waitFor(() => expect(api.getFamily).toHaveBeenCalledTimes(2));
    expect(api.getFamily).toHaveBeenLastCalledWith(key, { model: 'm_two', offset: 0, include_retired: false });
  });

  it('words every flag kind', () => {
    expect(flagWords({ flag: 'definition_stale', reason: 'not_built' })).toBe('definition not current (not built)');
    expect(flagWords({ flag: 'sub_question_not_declared', sub_question: 'x-y', version: 3 }))
      .toBe('sub-question x-y is not declared by version 3; left out of the default blend');
    expect(flagWords({ flag: 'differs_dataset_version', field: 'training_dataset_version', value: 'a', majority: 'b',
      majority_tie: true })).toBe('differs in dataset version: a vs b (no clear majority)');
  });
});
