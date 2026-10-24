/**
 * Feature 040 FR-9 (FTASKS 1.6): the judge run form labels its "default" choice from the library row the
 * server marks `is_default`, never from text in the component. Random keys and titles, so a component
 * that spelled one behaviour's prompt would show the wrong label here.
 */
import { render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { JudgePrompt, ProbeDataset } from '../../../types/probeMonitor';

const api = vi.hoisted(() => ({ listJudgePrompts: vi.fn() }));
vi.mock('../../../api/probeMonitors', () => ({ probeMonitorsApi: api }));

import { JudgeRunForm } from '../JudgeRunForm';

function slug(): string {
  const part = () => Array.from({ length: 5 }, () => 'abcdefghijklmnopqrstuvwxyz'[Math.floor(Math.random() * 26)]).join('');
  return `${part()}-${part()}`;
}

function row(overrides: Partial<JudgePrompt>): JudgePrompt {
  const name = slug();
  return {
    id: `jp_${name}`, name, version: 1, key: `${name}/v1`, title: `${slug()} ${slug()}`, description: null,
    behaviour: slug(), behaviour_label: null, system_prompt: 's', user_template: '{text}',
    answer_form: { kind: 'binary', yes_token: 'yes', no_token: 'no', positive_answer: 'yes', json_key: 'answer', reply: 'json', case_sensitive: false },
    answer_form_summary: '', max_interaction_chars: 8000, content_hash: 'a'.repeat(64), source: 'seed',
    source_ref: null, parent_id: null, created_by: 'seed', created_at: '', updated_at: '', locked: false,
    locked_at: null, archived: false, archived_at: null, used_by_runs: 0, warnings: [],
    ...overrides,
  } as JudgePrompt;
}

const SETS = [{ id: 'pmd_a', name: 'set-a', role: 'eval' }] as unknown as ProbeDataset[];

beforeEach(() => {
  api.listJudgePrompts.mockReset();
  window.localStorage.clear();
});

describe('the default judge prompt is labelled from the library row', () => {
  it('names the row the server marks as the default', async () => {
    const other = row({});
    const theDefault = row({ version: 3, is_default: true });
    api.listJudgePrompts.mockResolvedValue({ prompts: [other, theDefault], placeholders: {} });
    render(<JudgeRunForm datasets={SETS} probes={[]} onSubmit={vi.fn()} />);
    const picker = screen.getByTestId('judge-prompt-picker');
    await waitFor(() =>
      expect(within(picker).getByRole('option', { name: /the default/ })).toHaveTextContent(
        `${theDefault.title} (${theDefault.name} v3) — the default`
      )
    );
    expect(within(picker).getByRole('option', { name: /the default/ })).toHaveValue('');
    expect(api.listJudgePrompts).toHaveBeenCalledTimes(1);
  });

  it('says only "the library default" when no row is marked', async () => {
    api.listJudgePrompts.mockResolvedValue({ prompts: [row({})], placeholders: {} });
    render(<JudgeRunForm datasets={SETS} probes={[]} onSubmit={vi.fn()} />);
    await waitFor(() => expect(api.listJudgePrompts).toHaveBeenCalledTimes(1));
    const first = within(screen.getByTestId('judge-prompt-picker')).getAllByRole('option')[0];
    expect(first).toHaveTextContent("The library's default prompt");
    expect(first).toHaveValue('');
  });
});
