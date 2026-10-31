/**
 * The judge-prompt library in the UI (operator decisions D1/D2, 2026-10-21; PADR IDL-57).
 *
 * Rendered, not scraped: the panel itself is mounted with only the network mocked, so removing the
 * tab from `TABS`, the tab's render branch, or a button's handler turns a test here red. Each
 * request is asserted by its BODY and CALL COUNT. Controls are recorded in
 * 0xcc/reviews/judge_prompt_library_2026-10-21.md.
 */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { JudgePrompt, ProbeDataset, ProbeJudgeRun, ProbeMonitorSummary } from '../../../types/probeMonitor';

const api = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  listRuns: vi.fn(),
  listProbes: vi.fn(),
  listJudgeRuns: vi.fn(),
  getReport: vi.fn(),
  listJudgePrompts: vi.fn(),
  getJudgePrompt: vi.fn(),
  createJudgePrompt: vi.fn(),
  updateJudgePrompt: vi.fn(),
  newJudgePromptVersion: vi.fn(),
  cloneJudgePrompt: vi.fn(),
  archiveJudgePrompt: vi.fn(),
  deleteJudgePrompt: vi.fn(),
  previewJudgePrompt: vi.fn(),
  tryJudgePrompt: vi.fn(),
  exportJudgePrompt: vi.fn(),
  importJudgePrompt: vi.fn(),
  listMiDataworksRubrics: vi.fn(),
  importRubric: vi.fn(),
  setProbeBehaviour: vi.fn(),
  submitJudgeRun: vi.fn(),
}));

vi.mock('../../../api/probeMonitors', () => ({ probeMonitorsApi: api }));
vi.mock('../../../hooks/useProbeMonitorWebSocket', () => ({ useProbeMonitorWebSocket: () => undefined }));

import { ProbeMonitorsPanel } from '../../panels/ProbeMonitorsPanel';
import { JudgeRunForm, parseLimitFraction } from '../JudgeRunForm';
import { JudgeRunResults } from '../JudgeRunResults';
import { JudgePromptsTab } from '../JudgePromptsTab';
import { useProbeMonitorsStore } from '../../../stores/probeMonitorsStore';

const PLACEHOLDERS = { interaction: 'the conversation', text: 'the text' };

function prompt(overrides: Partial<JudgePrompt> = {}): JudgePrompt {
  return {
    id: 'jp_pii_presence_v1', name: 'pii-presence', version: 1, key: 'pii-presence/v1',
    title: 'Personal data present (PII)', description: null, behaviour: 'pii',
    behaviour_label: 'Personal data present (PII)',
    system_prompt: 'Answer {"answer": "yes" or "no"}.', user_template: 'Text:\n{text}',
    answer_form: { kind: 'binary', yes_token: 'yes', no_token: 'no', positive_answer: 'yes', json_key: 'answer', reply: 'json', case_sensitive: false },
    answer_form_summary: '"yes" or "no", read from a JSON object with "answer"; "yes" counts as positive.',
    max_interaction_chars: 8000, content_hash: 'a'.repeat(64), source: 'seed', source_ref: null,
    parent_id: null, created_by: 'seed', created_at: '2026-10-21T00:00:00Z', updated_at: '2026-10-21T00:00:00Z',
    locked: false, locked_at: null, archived: false, archived_at: null, used_by_runs: 0, warnings: [],
    versions: [{ id: 'jp_pii_presence_v1', version: 1, key: 'pii-presence/v1', content_hash: 'a'.repeat(64),
      locked: false, archived: false, used_by_runs: 0, created_at: '2026-10-21T00:00:00Z', created_by: 'seed' }],
    ...overrides,
  };
}

const STAKES = prompt({
  id: 'jp_stakes_rating_v1', name: 'stakes-rating', key: 'stakes-rating/v1', title: 'High-stakes interaction',
  behaviour: 'high-stakes', behaviour_label: 'High-stakes interaction',
  answer_form: { kind: 'rating', min: 1, max: 10, positive_from: 6, json_key: 'rating', reply: 'json' },
});

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset());
  window.localStorage.clear();
  api.listDatasets.mockResolvedValue([]);
  api.listRuns.mockResolvedValue([]);
  api.listProbes.mockResolvedValue([]);
  api.listJudgeRuns.mockResolvedValue([]);
  api.listJudgePrompts.mockResolvedValue({ prompts: [prompt(), STAKES], placeholders: PLACEHOLDERS });
  api.getJudgePrompt.mockImplementation(async (id: string) => (id === STAKES.id ? STAKES : prompt()));
});

describe('the Judge prompts sub-tab is reachable from the panel', () => {
  it('a "Judge prompts" tab exists and opens the library', async () => {
    useProbeMonitorsStore.setState({ activeTab: 'runs' });
    render(<ProbeMonitorsPanel />);
    await userEvent.click(screen.getByRole('tab', { name: 'Judge prompts' }));
    expect(await screen.findByTestId('tab-judge-prompts')).toBeInTheDocument();
    await waitFor(() => expect(api.listJudgePrompts).toHaveBeenCalledTimes(1));
    expect(api.listJudgePrompts).toHaveBeenCalledWith({ includeArchived: false });
    expect(await screen.findAllByTestId('judge-prompt-row')).toHaveLength(2);
  });
});

describe('the library', () => {
  it('a LOCKED version says "used by N runs" and its content cannot be typed into', async () => {
    api.getJudgePrompt.mockResolvedValue(prompt({ locked: true, used_by_runs: 3 }));
    render(<JudgePromptsTab datasets={[]} />);
    await userEvent.click((await screen.findAllByTestId('judge-prompt-row'))[0]);
    expect(await screen.findByTestId('judge-prompt-lock')).toHaveTextContent('Used by 3 runs — locked');
    expect(screen.getByTestId('judge-prompt-system')).toBeDisabled();
    expect(screen.getByTestId('judge-prompt-behaviour')).toBeDisabled();
  });

  it('"Start a new version" unlocks the text and saves version N+1 with exactly the content', async () => {
    api.getJudgePrompt.mockResolvedValue(prompt({ locked: true, used_by_runs: 1 }));
    api.newJudgePromptVersion.mockResolvedValue(prompt({ id: 'jp_2', version: 2, key: 'pii-presence/v2' }));
    render(<JudgePromptsTab datasets={[]} />);
    await userEvent.click((await screen.findAllByTestId('judge-prompt-row'))[0]);
    await userEvent.click(await screen.findByTestId('judge-prompt-start-version'));
    const system = screen.getByTestId('judge-prompt-system');
    expect(system).toBeEnabled();
    await userEvent.clear(system);
    await userEvent.type(system, 'Stricter. Answer with answer yes or no.');
    await userEvent.click(screen.getByTestId('judge-prompt-save-version'));
    await waitFor(() => expect(api.newJudgePromptVersion).toHaveBeenCalledTimes(1));
    const [id, body] = api.newJudgePromptVersion.mock.calls[0];
    expect(id).toBe('jp_pii_presence_v1');
    expect(body).toMatchObject({
      system_prompt: 'Stricter. Answer with answer yes or no.',
      user_template: 'Text:\n{text}',
      answer_form: { kind: 'binary', positive_answer: 'yes' },
      max_interaction_chars: 8000,
    });
    expect(api.updateJudgePrompt).not.toHaveBeenCalled();
  });

  it('a new prompt is created with its name, behaviour and answer form', async () => {
    api.createJudgePrompt.mockResolvedValue(prompt({ id: 'jp_new', name: 'refusal-presence' }));
    render(<JudgePromptsTab datasets={[]} />);
    await userEvent.click(await screen.findByTestId('judge-prompt-new'));
    await userEvent.type(screen.getByTestId('judge-prompt-title'), 'Refusal');
    await userEvent.type(screen.getByTestId('judge-prompt-name'), 'refusal-presence');
    await userEvent.type(screen.getByTestId('judge-prompt-behaviour'), 'refusal');
    await userEvent.selectOptions(screen.getByTestId('answer-kind'), 'labels');
    await userEvent.click(screen.getByTestId('judge-prompt-create'));
    await waitFor(() => expect(api.createJudgePrompt).toHaveBeenCalledTimes(1));
    expect(api.createJudgePrompt.mock.calls[0][0]).toMatchObject({
      name: 'refusal-presence', title: 'Refusal', behaviour: 'refusal',
      answer_form: { kind: 'labels', labels: ['present', 'absent'], positive_labels: ['present'] },
    });
  });

  it('a server refusal is shown in its own words', async () => {
    api.createJudgePrompt.mockRejectedValue(new Error('a judge prompt called \'x\' already exists'));
    render(<JudgePromptsTab datasets={[]} />);
    await userEvent.click(await screen.findByTestId('judge-prompt-new'));
    await userEvent.click(screen.getByTestId('judge-prompt-create'));
    expect(await screen.findByTestId('judge-prompt-error')).toHaveTextContent('already exists');
  });

  it('an unused version can be deleted; a used one offers no delete', async () => {
    api.deleteJudgePrompt.mockResolvedValue(undefined);
    render(<JudgePromptsTab datasets={[]} />);
    await userEvent.click((await screen.findAllByTestId('judge-prompt-row'))[0]);
    await userEvent.click(await screen.findByTestId('judge-prompt-delete'));
    await waitFor(() => expect(api.deleteJudgePrompt).toHaveBeenCalledWith('jp_pii_presence_v1'));

    api.getJudgePrompt.mockResolvedValue(prompt({ locked: true, used_by_runs: 1 }));
    await userEvent.click((await screen.findAllByTestId('judge-prompt-row'))[0]);
    await screen.findByTestId('judge-prompt-lock');
    expect(screen.queryByTestId('judge-prompt-delete')).toBeNull();
  });

  it('archiving calls the archive route for that version', async () => {
    api.archiveJudgePrompt.mockResolvedValue(prompt({ archived: true }));
    render(<JudgePromptsTab datasets={[]} />);
    await userEvent.click((await screen.findAllByTestId('judge-prompt-row'))[0]);
    await userEvent.click(await screen.findByTestId('judge-prompt-archive'));
    await waitFor(() => expect(api.archiveJudgePrompt).toHaveBeenCalledWith('jp_pii_presence_v1', true));
  });

  it('the live preview renders the DRAFT against the chosen view', async () => {
    api.previewJudgePrompt.mockResolvedValue({
      prompt_key: 'draft', content_hash: 'b', dataset_name: 'pii-us-test', system_prompt: 's',
      warnings: [], rows: [{ label: 'positive', user_message: 'Text:\nmy email is a@b.c' }],
    });
    const views = [{ id: 'pmd_1', name: 'pii-us-test', role: 'eval' }] as unknown as ProbeDataset[];
    render(<JudgePromptsTab datasets={views} />);
    await userEvent.click((await screen.findAllByTestId('judge-prompt-row'))[0]);
    await userEvent.selectOptions(await screen.findByTestId('judge-prompt-preview-view'), 'pmd_1');
    expect(await screen.findByTestId('judge-prompt-preview', {}, { timeout: 3000 })).toHaveTextContent('my email is a@b.c');
    expect(api.previewJudgePrompt).toHaveBeenLastCalledWith({
      draft: {
        system_prompt: 'Answer {"answer": "yes" or "no"}.', user_template: 'Text:\n{text}',
        answer_form: prompt().answer_form, max_interaction_chars: 8000,
      },
      dataset_id: 'pmd_1',
      rows: 4,
    });
  });

  it('an ARCHIVED view is not offered for the preview', async () => {
    const views = [
      { id: 'pmd_1', name: 'pii-us-test', role: 'eval', archived_at: null },
      { id: 'pmd_old', name: 'old-view', role: 'eval', archived_at: '2026-10-20T00:00:00Z' },
    ] as unknown as ProbeDataset[];
    render(<JudgePromptsTab datasets={views} />);
    await userEvent.click((await screen.findAllByTestId('judge-prompt-row'))[0]);
    const picker = await screen.findByTestId('judge-prompt-preview-view');
    expect(within(picker).queryByRole('option', { name: 'old-view' })).toBeNull();
    expect(within(picker).getByRole('option', { name: 'pii-us-test' })).toBeInTheDocument();
  });

  it('a miDataworks rubric FILE is imported through the rubric route', async () => {
    api.importRubric.mockResolvedValue(prompt({ id: 'jp_dw' }));
    render(<JudgePromptsTab datasets={[]} />);
    await userEvent.click(await screen.findByTestId('judge-prompt-import-toggle'));
    const rubric = { format: 'midataworks.rubric/v1', name: 'pii-check', version: 3, body: {} };
    const file = new File([JSON.stringify(rubric)], 'rubric.json', { type: 'application/json' });
    await userEvent.upload(screen.getByTestId('judge-prompt-import-file'), file);
    const form = await screen.findByTestId('judge-prompt-rubric-form');
    const inputs = within(form).getAllByRole('textbox');
    await userEvent.type(inputs[0], 'dw-pii');
    await userEvent.type(inputs[1], 'PII from rubric');
    await userEvent.type(inputs[2], 'pii');
    await userEvent.type(inputs[3], 'passage=text');
    await userEvent.type(inputs[5], 'yes');
    await userEvent.click(screen.getByTestId('judge-prompt-import-rubric'));
    await waitFor(() => expect(api.importRubric).toHaveBeenCalledTimes(1));
    expect(api.importRubric.mock.calls[0][0]).toMatchObject({
      document: rubric, name: 'dw-pii', behaviour: 'pii', field_map: { passage: 'text' },
      positive_verdicts: ['yes'],
    });
  });
});

const PII_PROBE = { id: 'pm_pii', layer: 11, rule: 'mean', variant: 'dense', behaviour: 'pii' } as unknown as ProbeMonitorSummary;
const SETS = [{ id: 'pmd_a', name: 'pii-intl-ood', role: 'eval' }] as unknown as ProbeDataset[];

describe('the judge run form picks a prompt by the probe’s behaviour', () => {
  it('sends the chosen prompt and the parse limit as a FRACTION', async () => {
    const onSubmit = vi.fn();
    render(<JudgeRunForm datasets={SETS} probes={[PII_PROBE]} onSubmit={onSubmit} />);
    await userEvent.type(screen.getByTestId('judge-endpoint'), 'http://millm/v1');
    await userEvent.type(screen.getByTestId('judge-model'), 'qwen');
    await userEvent.click(screen.getByTestId('judge-set-pmd_a'));
    await userEvent.selectOptions(screen.getByTestId('judge-probe'), 'pm_pii');
    await waitFor(() => expect(screen.getAllByRole('option', { name: /Personal data present/ }).length).toBeGreaterThan(0));
    await userEvent.selectOptions(screen.getByTestId('judge-prompt-picker'), 'jp_pii_presence_v1');
    await userEvent.click(screen.getByTestId('judge-submit'));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      prompt_id: 'jp_pii_presence_v1', probe_id: 'pm_pii', parse_failure_limit: 0.05,
    });
  });

  it('the matching prompts come first; another behaviour is labelled as not raising the level', async () => {
    render(<JudgeRunForm datasets={SETS} probes={[PII_PROBE]} onSubmit={vi.fn()} />);
    await userEvent.selectOptions(screen.getByTestId('judge-probe'), 'pm_pii');
    const picker = screen.getByTestId('judge-prompt-picker');
    await waitFor(() => expect(within(picker).getAllByRole('group')).toHaveLength(2));
    const [mine, others] = within(picker).getAllByRole('group');
    expect(mine).toHaveAttribute('label', 'Judges "pii", like this probe');
    expect(within(mine).getByRole('option')).toHaveValue('jp_pii_presence_v1');
    expect(others.getAttribute('label')).toMatch(/will not raise/);
    await userEvent.selectOptions(picker, 'jp_stakes_rating_v1');
    expect(screen.getByTestId('judge-prompt-mismatch')).toHaveTextContent('baseline only');
  });

  it('a probe with no behaviour is called out', async () => {
    render(<JudgeRunForm datasets={SETS} probes={[{ ...PII_PROBE, behaviour: null }]} onSubmit={vi.fn()} />);
    await userEvent.selectOptions(screen.getByTestId('judge-probe'), 'pm_pii');
    expect(screen.getByTestId('judge-probe-no-behaviour')).toBeInTheDocument();
  });

  it('the percentage converts exactly', () => {
    expect(parseLimitFraction(5)).toBe(0.05);
    expect(parseLimitFraction(50)).toBe(0.5);
  });
});

describe('a judge run shows what it measured per set', () => {
  const RUN = {
    id: 'pmj_1', model: 'qwen', status: 'completed', prompt_version: 'pii-presence/v1',
    prompt_title: 'Personal data present (PII)', counts_toward_rung: true, rung_note: 'matches',
    max_rows_per_set: 200, dataset_ids: ['pmd_a', 'pmd_b'],
    metrics: {
      per_set: {
        pmd_a: { set_name: 'pii-intl-ood', out_of_distribution: true, scored: true, auroc: 0.912,
          ci: { low: 0.88, high: 0.94 }, n_positive: 100, n_negative: 100, n_dropped: 2,
          decision: { rule: 'yes counts', recall: 0.84, false_positive_rate: 0.03, precision: 0.9, judged_positive: 90, judged: 198 },
          at_probe_fpr: { target_fpr: 0.01, recall: 0.5, realised_fpr: 0.01 } },
        pmd_b: { scored: false, reason: 'fewer than 20 rows per class' },
      },
    },
  } as unknown as ProbeJudgeRun;

  it('AUROC with its interval, the judge’s own recall, and recall at the probe’s rate', () => {
    render(<JudgeRunResults run={RUN} datasetNames={{ pmd_b: 'tiny-set' }} />);
    const [scored, refused] = screen.getAllByTestId('judge-set-row');
    expect(scored).toHaveTextContent('pii-intl-ood');
    expect(scored).toHaveTextContent('0.912 (0.880–0.940)');
    expect(scored).toHaveTextContent('84.0%');
    expect(scored).toHaveTextContent('50.0% at 1.0%');
    expect(scored).toHaveTextContent('2 unreadable');
    expect(refused).toHaveTextContent('tiny-set');
    expect(refused).toHaveTextContent('fewer than 20 rows per class');
    expect(screen.getByTestId('judge-run-results')).toHaveTextContent('Up to 200 rows per set');
  });
});

describe('the panel wires the judge results and the probe behaviour', () => {
  const RUN = {
    id: 'pmj_9', endpoint: 'http://j/v1', model: 'qwen', status: 'completed',
    prompt_version: 'pii-presence/v1', prompt_id: 'jp_pii_presence_v1', prompt_hash: 'a',
    prompt_title: 'Personal data present (PII)', prompt_behaviour: 'pii',
    counts_toward_rung: false, rung_note: 'the probe has no behaviour recorded',
    dataset_ids: ['pmd_a'], probe_id: 'pm_1', progress: 100, parse_failures: 0,
    max_rows_per_set: 200, error_message: null,
    metrics: { per_set: { pmd_a: { set_name: 'pii-intl-ood', scored: true, auroc: 0.9, ci: { low: 0.85, high: 0.95 }, n_positive: 50, n_negative: 50 } } },
  } as unknown as ProbeJudgeRun;

  it('the Judge tab shows each run’s prompt, whether it counts, and its per-set results', async () => {
    api.listJudgeRuns.mockResolvedValue([RUN]);
    useProbeMonitorsStore.setState({ activeTab: 'judge' });
    render(<ProbeMonitorsPanel />);
    const row = await screen.findByTestId('judge-row');
    expect(row).toHaveTextContent('Personal data present (PII)');
    expect(within(row).getByTestId('judge-run-counts')).toHaveTextContent('Does not raise');
    expect(within(row).getByTestId('judge-run-results')).toHaveTextContent('0.900 (0.850–0.950)');
  });

  it('an open probe report shows what it detects, and saving it calls the behaviour route', async () => {
    const probe = {
      id: 'pm_1', run_id: 'pmr_1', layer: 11, rule: 'mean', rule_params: {}, variant: 'dense',
      sae_id: null, sae_feature_indices: null, val_metrics: {}, selected: true, threshold: 1,
      target_fpr: 0.01, realised_fpr: 0.01, threshold_source: 'calibration_set', streamable: true,
      rung: 2, rung_reasons: [], rung_language: 'detects on unseen tasks', rung_next_step: 'judge it',
      behaviour: null, behaviour_label: null, published: [], created_at: '2026-10-21T00:00:00Z',
    } as unknown as ProbeMonitorSummary;
    api.listProbes.mockResolvedValue([probe]);
    api.setProbeBehaviour.mockResolvedValue({ probe_id: 'pm_1', behaviour: 'pii', rung: { rung: 3 } });
    useProbeMonitorsStore.setState({
      activeTab: 'probes',
      openProbeId: 'pm_1',
      report: {
        probe, rung_language: 'detects on unseen tasks', rung_next_step: 'judge it',
        evaluations: [], paired_probe_id: null, judge_runs: [RUN], threshold_transfer: null,
      } as never,
    });
    render(<ProbeMonitorsPanel />);
    const behaviour = await screen.findByTestId('probe-behaviour');
    expect(behaviour).toHaveTextContent('not recorded');
    expect(screen.getByTestId('report-judge-runs')).toHaveTextContent('pii-intl-ood');
    await userEvent.click(screen.getByTestId('probe-behaviour-edit'));
    await userEvent.type(screen.getByTestId('probe-behaviour-key'), 'pii');
    await userEvent.click(screen.getByTestId('probe-behaviour-save'));
    await waitFor(() => expect(api.setProbeBehaviour).toHaveBeenCalledTimes(1));
    expect(api.setProbeBehaviour).toHaveBeenCalledWith('pm_1', 'pii', null);
  });
});
