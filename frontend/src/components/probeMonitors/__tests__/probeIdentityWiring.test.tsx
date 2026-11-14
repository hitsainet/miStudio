/**
 * Feature 039 (FR-20) wiring: the identity rows reach the export section THROUGH THE REAL PANEL, so
 * deleting the `identity={identityOf(report.probe)}` prop from ProbeMonitorsPanel turns this red
 * (the `panelDatasetsWiring.test.tsx` pattern — rendered, not a source scrape).
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';

const api = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  listRuns: vi.fn(),
  listProbes: vi.fn(),
  listJudgeRuns: vi.fn(),
  getReport: vi.fn(),
  exportUrl: (id: string) => `/x/${id}`,
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

const D = 'd'.repeat(64);
const PROBE = {
  id: 'pm_1', run_id: 'pmr_1', layer: 1, rule: 'mean', rule_params: {}, variant: 'dense',
  sae_id: null, sae_feature_indices: null, val_metrics: {}, selected: true, threshold: 1,
  target_fpr: 0.01, realised_fpr: 0.01, threshold_source: 'calibration_set', streamable: true,
  rung: 2, rung_reasons: [], created_at: '2026-10-23T00:00:00Z',
  definition_built_at: '2026-10-23T01:00:00Z', definition_sha256: 'a'.repeat(64),
  definition_build: { resolved: { vectors: 16, bytes: 1 } }, published: [],
  detector_digest: D, decision_digest: 'e'.repeat(64), decision_revision: 2,
  identity_current: false, identity_stale_reason: 'decision_recalibrated', identity_source: 'built',
};
const REPORT = {
  probe: PROBE, rung_language: 'x', rung_next_step: 'y', evaluations: [], judge_runs: [],
  threshold_transfer: null, paired_probe_id: null, sae_hf_repo: null,
};

beforeEach(() => {
  api.listDatasets.mockResolvedValue([]);
  api.listRuns.mockResolvedValue([]);
  api.listProbes.mockResolvedValue([PROBE]);
  api.listJudgeRuns.mockResolvedValue([]);
  api.getReport.mockResolvedValue(REPORT);
});

describe('ProbeMonitorsPanel → export identity (039 FR-20)', () => {
  it('passes the summary identity to the export section', async () => {
    act(() =>
      useProbeMonitorsStore.setState({
        activeTab: 'probes', probes: [PROBE] as never, report: REPORT as never, openProbeId: 'pm_1',
      } as never)
    );
    render(<ProbeMonitorsPanel />);
    expect(await screen.findByTestId('probe-export-identity')).toBeInTheDocument();
    expect(screen.getByTestId('probe-export-detector-digest').textContent).toContain(D.slice(0, 16));
    expect(screen.getByTestId('probe-export-identity-stale')).toBeInTheDocument();
  });
});

describe('ProbeMonitorsPanel → calibration size (040 FR-19)', () => {
  it('renders the report\'s calibration size line through the real panel', async () => {
    const report = { ...REPORT, calibration_size: { view_id: 'pmd_c', threshold_source: 'calibration_set',
      n_negatives: 540, n_negatives_source: 'recorded', floor: 2000, floor_source: 'setting', few_negatives: true,
      target_fpr: 0.01, allowed_false_positives: 5, sentence: '1% of 540 = 5 rows' } };
    act(() =>
      useProbeMonitorsStore.setState({
        activeTab: 'probes', probes: [PROBE] as never, report: report as never, openProbeId: 'pm_1',
      } as never)
    );
    render(<ProbeMonitorsPanel />);
    expect(await screen.findByTestId('calibration-size')).toHaveTextContent('1% of 540 = 5 rows');
  });
});
