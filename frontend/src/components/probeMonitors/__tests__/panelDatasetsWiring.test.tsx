/**
 * The Probes panel's Datasets tab IS the new tab (2026-10-21): rendered through the real panel,
 * so removing `<DatasetsTab />` from ProbeMonitorsPanel turns this red. (The evaluate and judge
 * pickers filter archived views themselves — `probeGaps.test.tsx`.)
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';

const api = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  listRuns: vi.fn(),
  listProbes: vi.fn(),
  listJudgeRuns: vi.fn(),
  planDatasetPurge: vi.fn(),
}));

vi.mock('../../../api/probeMonitors', () => ({ probeMonitorsApi: api }));
vi.mock('../ProbeDatasetForm', () => ({ ProbeDatasetForm: () => <div /> }));
vi.mock('../ProbeRunForm', () => ({ ProbeRunForm: () => <div /> }));
vi.mock('../../../hooks/useProbeMonitorWebSocket', () => ({ useProbeMonitorWebSocket: () => {} }));
vi.mock('../../../stores/modelsStore', () => {
  const state = { models: [], fetchModels: vi.fn() };
  return { useModelsStore: (selector?: (s: typeof state) => unknown) => (selector ? selector(state) : state) };
});

import { ProbeMonitorsPanel } from '../../panels/ProbeMonitorsPanel';
import { useProbeMonitorsStore } from '../../../stores/probeMonitorsStore';

const VIEWS = [
  { id: 'pmd_a', name: 'live eval', role: 'eval', counts: {}, archived_at: null },
  { id: 'pmd_b', name: 'old eval', role: 'eval', counts: {}, archived_at: '2026-10-21T14:00:00Z' },
];

beforeEach(() => {
  api.listDatasets.mockResolvedValue(VIEWS);
  api.listRuns.mockResolvedValue([]);
  api.listProbes.mockResolvedValue([]);
  api.listJudgeRuns.mockResolvedValue([]);
  api.planDatasetPurge.mockResolvedValue([]);
});

describe('ProbeMonitorsPanel → Datasets tab', () => {
  it('renders the archive-aware Datasets tab', async () => {
    act(() => useProbeMonitorsStore.setState({ activeTab: 'datasets', datasets: VIEWS as never }));
    render(<ProbeMonitorsPanel />);
    expect(await screen.findByTestId('datasets-show-archived')).toBeInTheDocument();
    expect(screen.getByTestId('dataset-browse-pmd_a')).toBeInTheDocument();
    expect(screen.queryByTestId('dataset-browse-pmd_b')).not.toBeInTheDocument();
  });
});
