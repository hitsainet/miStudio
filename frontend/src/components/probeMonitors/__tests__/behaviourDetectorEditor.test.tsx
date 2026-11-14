/**
 * Feature 040 FR-1, FR-3, FR-6, FR-43 (FTASKS 4.4): the structured detector editor, from random-key fixtures.
 *
 * Each save is asserted by its EXACT request body and call count: an edit is a change against the version
 * it was opened from (`base_version` + only what differs, an emptied field as null); a create sends every
 * declared field. A `stale_base_version` refusal shows the reload notice with what changed.
 */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { APIError } from '../../../api/client';
import type { BehaviourDefinition, BehaviourDetail as Detail } from '../../../types/behaviour';

const api = vi.hoisted(() => ({
  list: vi.fn(), get: vi.fn(), create: vi.fn(), versions: vi.fn(), newVersion: vi.fn(), setRole: vi.fn(),
  unlinkView: vi.fn(), confirmPositive: vi.fn(), archive: vi.fn(), unarchive: vi.fn(), remove: vi.fn(),
  checklist: vi.fn(), history: vi.fn(),
}));
vi.mock('../../../api/behaviours', () => ({ behavioursApi: api }));
vi.mock('../../../api/spotChecks', () => ({ spotChecksApi: {
  availability: vi.fn().mockResolvedValue({ available: false, reason: 'not connected', url: null }),
  list: vi.fn().mockResolvedValue({ spot_checks: [] }),
} }));

import { BehaviourDetail } from '../BehaviourDetail';
import { BehavioursTab } from '../BehavioursTab';

const slug = () => {
  const part = () => Array.from({ length: 5 }, () => 'abcdefghijklmnopqrstuvwxyz'[Math.floor(Math.random() * 26)]).join('');
  return `${part()}-${part()}`;
};

function definition(overrides: Partial<BehaviourDefinition> = {}): BehaviourDefinition {
  return {
    version: 3, name: slug(), description: null, positive_meaning: slug(), negative_meaning: null, target_fpr: 0.02,
    created_by: 'operator', created_at: null, ...overrides,
  };
}

function detail(key: string, def: BehaviourDefinition): Detail {
  return {
    key, catalogued: true, current_version: def.version, archived: false, archived_at: null, archived_by: null,
    created_by: 'operator', created_at: null, updated_at: null, definition: def,
    counts: { prompts: 0, runs: 0, probes: 0, views: 0 },
    adopted: { prompts: [], runs: [], probes: [], views: [] },
  };
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset());
  api.checklist.mockResolvedValue({ key: 'x', status: 'draft', steps: [] });
  api.versions.mockResolvedValue({ key: 'x', versions: [] });
  api.history.mockResolvedValue({ key: 'x', total: 0, entries: [] });
});

describe('editing the detector data is a change against the version it was opened from', () => {
  it('sends base_version and only the changed fields; an emptied field is null', async () => {
    const key = slug();
    const column = slug();
    const criterionId = slug();
    const def = definition({
      held_out_columns: [column],
      models: [{ model_id: slug() }],
      success_criteria: [{ id: criterionId, detector: 'best_single', subset: 'all', held_out_test: column, min_recall: 0.8 }],
    });
    api.get.mockResolvedValue(detail(key, def));
    api.newVersion.mockResolvedValue({ ...detail(key, { ...def, version: 4 }), current_version: 4, warnings: [],
      diff: [{ path: 'calibration_floor', before: null, after: 4321 }, { path: 'models', before: def.models, after: null }] });
    render(<BehaviourDetail behaviourKey={key} onChanged={vi.fn()} />);
    await userEvent.click(await screen.findByRole('button', { name: 'Edit meaning' }));
    const form = screen.getByRole('form', { name: 'Behaviour form' });
    await userEvent.type(within(form).getByLabelText('Calibration floor'), '4321');
    await userEvent.clear(within(form).getByLabelText('Models'));
    await userEvent.clear(within(form).getByLabelText('Criterion 1 minimum recall'));
    await userEvent.type(within(form).getByLabelText('Criterion 1 minimum recall'), '0.9');
    await userEvent.click(within(form).getByRole('button', { name: 'Save as new version' }));
    await waitFor(() => expect(api.newVersion).toHaveBeenCalledTimes(1));
    expect(api.newVersion).toHaveBeenCalledWith(key, {
      base_version: 3,
      success_criteria: [{ id: criterionId, detector: 'best_single', subset: 'all', held_out_test: column, min_recall: 0.9 }],
      models: null,
      calibration_floor: 4321,
    });
    expect(await screen.findByTestId('saved-diff')).toHaveTextContent('changed: calibration_floor, models');
  });

  it('adds a criterion with a column subset and a named probe', async () => {
    const key = slug();
    const def = definition({ target_fpr: null });
    api.get.mockResolvedValue(detail(key, def));
    api.newVersion.mockResolvedValue({ ...detail(key, def), warnings: [], diff: [] });
    render(<BehaviourDetail behaviourKey={key} onChanged={vi.fn()} />);
    await userEvent.click(await screen.findByRole('button', { name: 'Edit meaning' }));
    const form = screen.getByRole('form', { name: 'Behaviour form' });
    await userEvent.click(within(form).getByRole('button', { name: 'Add criterion' }));
    const [id, column, value, probe] = [slug(), slug(), slug(), `pm_${slug()}`];
    await userEvent.type(within(form).getByLabelText('Criterion 1 id'), id);
    await userEvent.selectOptions(within(form).getByLabelText('Criterion 1 detector'), 'named_probe');
    await userEvent.selectOptions(within(form).getByLabelText('Criterion 1 subset'), 'column');
    await userEvent.type(within(form).getByLabelText('Criterion 1 subset column'), column);
    await userEvent.type(within(form).getByLabelText('Criterion 1 subset value'), value);
    await userEvent.type(within(form).getByLabelText('Criterion 1 target false-positive rate'), '0.005');
    await userEvent.type(within(form).getByLabelText('Criterion 1 minimum recall'), '0.5');
    await userEvent.type(within(form).getByLabelText('Criterion 1 probe'), probe);
    await userEvent.click(within(form).getByRole('button', { name: 'Save as new version' }));
    await waitFor(() => expect(api.newVersion).toHaveBeenCalledTimes(1));
    expect(api.newVersion).toHaveBeenCalledWith(key, {
      base_version: 3,
      success_criteria: [{ id, detector: 'named_probe', subset: { column, value }, target_fpr: 0.005, min_recall: 0.5, probe_id: probe }],
    });
  });

  it('a stale base shows the reload notice naming what changed', async () => {
    const key = slug();
    const opened = definition();
    const moved = { ...opened, version: 4, calibration_floor: 9 };
    api.get.mockResolvedValueOnce(detail(key, opened)).mockResolvedValue(detail(key, moved));
    api.newVersion.mockRejectedValue(new APIError(409, 'changed', { code: 'stale_base_version', current_version: 4, base_version: 3, next: [] }));
    render(<BehaviourDetail behaviourKey={key} onChanged={vi.fn()} />);
    await userEvent.click(await screen.findByRole('button', { name: 'Edit meaning' }));
    const form = screen.getByRole('form', { name: 'Behaviour form' });
    await userEvent.type(within(form).getByLabelText('Calibration floor'), '5');
    await userEvent.click(within(form).getByRole('button', { name: 'Save as new version' }));
    const notice = await screen.findByTestId('stale-base');
    expect(notice).toHaveTextContent('you edited version 3; it is now version 4');
    expect(notice).toHaveTextContent('changed: calibration_floor');
    expect(api.newVersion).toHaveBeenCalledTimes(1);
  });

  it('shows the rule set read-only with its validation state', async () => {
    const key = slug();
    const digest = 'ab'.repeat(32);
    api.get.mockResolvedValue(detail(key, definition({ rule_set: {
      digest, digest_scheme: 'midataworks_rule_set_identity', source: 'inline', source_ref: null,
      validation: { status: 'invalid', code: 'rule_set_cases_failed' } } })));
    render(<BehaviourDetail behaviourKey={key} onChanged={vi.fn()} />);
    expect(await screen.findByTestId('rule-set')).toHaveTextContent(`inline (${digest.slice(0, 12)}…): invalid — rule set cases failed`);
  });
});

describe('creating a behaviour sends every declared field', () => {
  it('posts the full body with the detector fields', async () => {
    api.list.mockResolvedValue({ behaviours: [], uncatalogued: [] });
    const [key, name, positive, column, sub] = [slug(), slug(), slug(), slug(), slug()];
    api.create.mockResolvedValue({ ...detail(key, definition()), warnings: [] });
    render(<BehavioursTab />);
    await userEvent.click(await screen.findByRole('button', { name: 'New behaviour' }));
    const form = screen.getByRole('form', { name: 'Behaviour form' });
    await userEvent.type(within(form).getByLabelText('Key'), key);
    await userEvent.type(within(form).getByLabelText('Name'), name);
    await userEvent.type(within(form).getByLabelText('Positive meaning'), positive);
    await userEvent.type(within(form).getByLabelText('Target false-positive rate'), '0.01');
    await userEvent.type(within(form).getByLabelText('Held-out columns'), column);
    await userEvent.click(within(form).getByRole('button', { name: 'Add sub-question' }));
    await userEvent.type(within(form).getByLabelText('Sub-question 1 key'), sub);
    await userEvent.type(within(form).getByLabelText('Sub-question 1 question'), 'q');
    await userEvent.type(within(form).getByLabelText('Sub-question 1 positive meaning'), 'p');
    await userEvent.click(within(form).getByRole('button', { name: 'Create behaviour' }));
    await waitFor(() => expect(api.create).toHaveBeenCalledTimes(1));
    expect(api.create).toHaveBeenCalledWith({
      key, name, positive_meaning: positive, description: null, negative_meaning: null, target_fpr: 0.01,
      sub_questions: [{ key: sub, question: 'q', positive_meaning: 'p' }], held_out_columns: [column],
    });
  });

  it('the catalogue pages: "Show more" asks for the next page with its cursor', async () => {
    const a = { key: slug(), catalogued: true as const, current_version: 1, archived: false, archived_at: null,
      archived_by: null, created_by: 'o', created_at: null, updated_at: null, definition: definition({ version: 1 }),
      counts: { prompts: 0, runs: 0, probes: 0, views: 0 }, status: 'draft' as const };
    const b = { ...a, key: slug(), definition: definition({ version: 1 }) };
    api.list.mockResolvedValueOnce({ behaviours: [a], uncatalogued: [], next_cursor: 'c-1', total: 2 })
      .mockResolvedValueOnce({ behaviours: [b], uncatalogued: [], next_cursor: null, total: 2 });
    render(<BehavioursTab />);
    await userEvent.click(await screen.findByRole('button', { name: /Show more \(1 of 2\)/ }));
    await waitFor(() => expect(api.list).toHaveBeenCalledTimes(2));
    expect(api.list).toHaveBeenLastCalledWith(false, { cursor: 'c-1' });
    const listed = within(screen.getByRole('list', { name: 'Behaviours' })).getAllByRole('listitem');
    expect(listed).toHaveLength(2);
    expect(screen.queryByRole('button', { name: /Show more/ })).toBeNull();
  });
});
