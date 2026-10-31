/**
 * The behaviour catalogue in the UI (Feature 037 FR-16). The panel is mounted with only the network
 * mocked, so removing the tab from `TABS`, its render branch, or the role handler turns a test red.
 * Requests are asserted by their arguments AND call count.
 */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { BehaviourChecklist, BehaviourDetail, BehaviourList } from '../../../types/behaviour';

const probeApi = vi.hoisted(() => ({
  listDatasets: vi.fn(), listRuns: vi.fn(), listProbes: vi.fn(), listJudgeRuns: vi.fn(), getReport: vi.fn(),
}));
const api = vi.hoisted(() => ({
  list: vi.fn(), get: vi.fn(), create: vi.fn(), versions: vi.fn(), newVersion: vi.fn(), setRole: vi.fn(),
  unlinkView: vi.fn(), confirmPositive: vi.fn(), archive: vi.fn(), unarchive: vi.fn(), remove: vi.fn(),
  checklist: vi.fn(), history: vi.fn(),
}));
vi.mock('../../../api/probeMonitors', () => ({ probeMonitorsApi: probeApi }));
vi.mock('../../../api/behaviours', () => ({ behavioursApi: api }));
vi.mock('../../../api/spotChecks', () => ({ spotChecksApi: {
  availability: vi.fn().mockResolvedValue({ available: false, reason: 'miLLM is not connected', url: null }),
  list: vi.fn().mockResolvedValue({ spot_checks: [] }),
} }));
vi.mock('../../../hooks/useProbeMonitorWebSocket', () => ({ useProbeMonitorWebSocket: () => undefined }));
// Feature 041 (FTASKS 3.2): the family section is mounted on the detail.
const familyApi = vi.hoisted(() => ({
  getFamily: vi.fn(),
  listReports: vi.fn().mockResolvedValue({ behaviour_key: 'pii', reports: [], offset: 0, limit: 25, next_offset: null }),
  getSuggestedBlendHistory: vi.fn().mockResolvedValue({ key: 'pii', model: 'm_1', entries: [], next_offset: null }),
  exportUrl: vi.fn(() => '/api/v1/x'),
}));
vi.mock('../../../api/probeFamilies', () => ({ probeFamiliesApi: familyApi }));

import { ProbeMonitorsPanel } from '../../panels/ProbeMonitorsPanel';
import { useProbeMonitorsStore } from '../../../stores/probeMonitorsStore';

const DEF = { version: 1, name: 'Personal data present', description: null, positive_meaning: 'Identifies a person.',
  negative_meaning: null, target_fpr: 0.01, created_by: 'operator', created_at: '2026-10-21T00:00:00Z' };
const LIST: BehaviourList = {
  behaviours: [{ key: 'pii', catalogued: true, current_version: 1, archived: false, archived_at: null, archived_by: null,
    created_by: 'operator', created_at: null, updated_at: null, definition: DEF,
    counts: { prompts: 1, runs: 1, probes: 2, views: 1 }, status: 'draft' }],
  uncatalogued: [{ key: 'refusal', catalogued: false, prompts: 1, runs: 0, probes: 0, views: 0 }],
};
const DETAIL: BehaviourDetail = {
  ...LIST.behaviours[0],
  adopted: {
    prompts: [{ id: 'jp_pii_presence_v1', key: 'pii-presence/v1', title: 'Personal data present (PII)', locked: false, archived: false }],
    runs: [],
    probes: [{ id: 'pm_1', run_id: 'pmr_1', layer: 12, rule: 'mean', variant: 'dense', rung: 2, behaviour_label: null,
      definition_sha256: null, role: 'candidate', training_view_confirmed: true, created_at: null }],
    views: [],
  },
};
const CHECKLIST: BehaviourChecklist = {
  key: 'pii', status: 'unknown',
  steps: [
    { id: 'defined', label: 'Behaviour defined', state: 'done', reason: "'pii' is catalogued", evidence: ['pii'], link: { tab: 'behaviours' } },
    { id: 'armed', label: 'Armed in miLLM', state: 'unknown', reason: 'miLLM is not connected', evidence: [], link: { app: 'millm' } },
  ],
};

beforeEach(() => {
  Object.values(probeApi).forEach((fn) => fn.mockReset());
  Object.values(api).forEach((fn) => fn.mockReset());
  window.localStorage.clear();
  probeApi.listDatasets.mockResolvedValue([]);
  probeApi.listRuns.mockResolvedValue([]);
  probeApi.listProbes.mockResolvedValue([]);
  probeApi.listJudgeRuns.mockResolvedValue([]);
  api.list.mockResolvedValue(LIST);
  api.get.mockResolvedValue(DETAIL);
  api.checklist.mockResolvedValue(CHECKLIST);
  api.versions.mockResolvedValue({ key: 'pii', versions: [DEF] });
  api.history.mockResolvedValue({ key: 'pii', total: 0, entries: [] });
  api.setRole.mockResolvedValue(DETAIL);
  familyApi.getFamily.mockReset();
  familyApi.getFamily.mockResolvedValue({
    key: 'pii', version: 1, model: 'm_1', models: [{ model_id: 'm_1', members: 0 }], declared_sub_questions: [],
    coverage: [{ sub_question: null, group: 'main_question', state: 'empty', members: [] }], default_blend_set: [],
    flags_basis: 'majority_of_all_members', members: [], total: 0, offset: 0, limit: 50, next_offset: null,
  });
});

async function openTab() {
  useProbeMonitorsStore.setState({ activeTab: 'runs' });
  render(<ProbeMonitorsPanel />);
  await userEvent.click(screen.getByRole('tab', { name: 'Behaviours' }));
  return screen.findByTestId('tab-behaviours');
}

describe('the Behaviours tab', () => {
  it('is reachable from the panel and lists one catalogued and one uncatalogued key', async () => {
    await openTab();
    await waitFor(() => expect(api.list).toHaveBeenCalledTimes(1));
    expect(api.list).toHaveBeenCalledWith(false);
    const catalogued = await screen.findByRole('list', { name: 'Behaviours' });
    expect(within(catalogued).getByText(/Personal data present/)).toBeInTheDocument();
    const uncatalogued = screen.getByRole('list', { name: 'Uncatalogued keys' });
    expect(within(uncatalogued).getByText('refusal')).toBeInTheDocument();
    expect(within(uncatalogued).getByText('uncatalogued')).toBeInTheDocument();
  });

  it('shows the checklist with "unknown" as unknown, and changes a role with one exact request', async () => {
    await openTab();
    await userEvent.click(await screen.findByText(/Personal data present/));
    const checklist = await screen.findByRole('list', { name: 'Checklist' });
    expect(within(checklist).getByText(/\(unknown\)/)).toBeInTheDocument();
    // Feature 038: the spot-check section is mounted on the detail (it reports why it cannot start).
    expect(await screen.findByLabelText('Spot-checks')).toBeInTheDocument();
    expect(await screen.findByText('miLLM is not connected')).toBeInTheDocument();
    // Feature 041: the family section is mounted and reads the family once, for this key.
    expect(await screen.findByLabelText('Probe family')).toBeInTheDocument();
    await waitFor(() => expect(familyApi.getFamily).toHaveBeenCalledTimes(1));
    expect(familyApi.getFamily).toHaveBeenCalledWith('pii', { model: null, offset: 0, include_retired: false });
    await userEvent.selectOptions(screen.getByLabelText('Role of pm_1'), 'primary');
    await waitFor(() => expect(api.setRole).toHaveBeenCalledTimes(1));
    expect(api.setRole).toHaveBeenCalledWith('pii', 'pm_1', 'primary');
  });

  it('"Catalogue this" prefills the key and creates with exactly the form', async () => {
    api.create.mockResolvedValue({ ...DETAIL, key: 'refusal' });
    await openTab();
    await userEvent.click(await screen.findByRole('button', { name: 'Catalogue this' }));
    expect(screen.getByLabelText('Key')).toHaveValue('refusal');
    await userEvent.type(screen.getByLabelText('Name'), 'Refusal');
    await userEvent.type(screen.getByLabelText('Positive meaning'), 'The assistant refused.');
    await userEvent.click(screen.getByRole('button', { name: 'Create behaviour' }));
    await waitFor(() => expect(api.create).toHaveBeenCalledTimes(1));
    expect(api.create).toHaveBeenCalledWith({
      key: 'refusal', name: 'Refusal', positive_meaning: 'The assistant refused.',
      description: null, negative_meaning: null, target_fpr: null,
    });
  });
});
