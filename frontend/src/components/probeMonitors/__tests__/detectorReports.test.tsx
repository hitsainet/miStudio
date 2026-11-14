/**
 * Feature 040 S2 (FTASKS 13.2 – 13.4): the report surfaces, rendered from random-key fixtures with only the
 * network mocked. Each request is asserted by its arguments and call count; every figure on screen is the
 * fixture's; a refusal renders its message and its `next` call.
 */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { APIError } from '../../../api/client';
import type { ProbeBreakdown, SuccessReportEnvelope } from '../../../types/probeMonitor';

const api = vi.hoisted(() => ({
  updateViewColumns: vi.fn(), getBreakdown: vi.fn(), createSuccessReport: vi.fn(), listSuccessReports: vi.fn(),
  readSuccessReport: vi.fn(), compareModels: vi.fn(),
}));
const probeApi = vi.hoisted(() => ({ getReport: vi.fn(), listDatasets: vi.fn() }));
vi.mock('../../../api/behaviourDetectors', () => ({ behaviourDetectorsApi: api }));
vi.mock('../../../api/probeMonitors', () => ({ probeMonitorsApi: probeApi }));

import { useBehaviourDetectorsStore } from '../../../stores/behaviourDetectorsStore';
import { DetectorSuccessReport, ModelComparison, ProbeBreakdown as Breakdown } from '../DetectorReports';

const slug = () => {
  const part = () => Array.from({ length: 5 }, () => 'abcdefghijklmnopqrstuvwxyz'[Math.floor(Math.random() * 26)]).join('');
  return `${part()}-${part()}`;
};

function envelope(key: string, overrides: Partial<SuccessReportEnvelope> = {}): SuccessReportEnvelope {
  const [c1, c2, column] = [slug(), slug(), slug()];
  return {
    id: `dsr_${slug()}`, created_at: null, created_by: 'operator', body_sha256: 'a'.repeat(64),
    report: {
      format: 'mistudio.detector-success-report/1', behaviour_key: key, behaviour_version: 2, model_id: 'm_x',
      dataset_identity: { kind: 'view_set' }, probe: { id: 'pm_1', chosen_by: 'primary', run_id: 'pmr_1', decision_revision: 1 },
      criteria: [
        { id: c1, detector: 'best_single', subset: 'all', held_out_test: column, in_distribution: false, target_fpr: 0.02,
          min_recall: 0.7, verdict: 'met', reasons: [], next: [],
          figures: { positives: 40, flagged_positives: 31, negatives: 100, flagged_negatives: 1, recall: 0.775,
                     recall_ci: [0.625, 0.877], test_fpr: 0.01 },
          calibration: { view_id: 'pmd_c', threshold_source: 'calibration_set', n_negatives: 540, n_negatives_source: 'recorded',
                         floor: 2000, floor_source: 'setting', few_negatives: true, target_fpr: 0.02,
                         allowed_false_positives: 10, sentence: '2% of 540 = 10 rows', decision_revision: 1, realised_fpr: 0.0185 } },
        { id: c2, detector: 'rules_alone', subset: 'all', held_out_test: null, in_distribution: true, target_fpr: 0.02,
          min_recall: 0.5, verdict: 'not_assessable', reasons: [{ code: 'engine_not_available' }], next: [] },
      ],
      overall_verdict: 'not_assessable',
    },
    stale: { stale: false, moved: [], inputs_removed: {}, next: [] },
    ...overrides,
  };
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset());
  Object.values(probeApi).forEach((fn) => fn.mockReset());
  useBehaviourDetectorsStore.setState({ breakdown: null, report: null, reports: null, comparison: null, busy: false, refusal: null });
  api.listSuccessReports.mockResolvedValue({ key: 'x', reports: [], total: 0, next_cursor: null });
});

describe('the success report', () => {
  it('builds one for the behaviour and renders every criterion from the body', async () => {
    const key = slug();
    const built = envelope(key);
    api.createSuccessReport.mockResolvedValue(built);
    render(<DetectorSuccessReport behaviourKey={key} />);
    await waitFor(() => expect(api.listSuccessReports).toHaveBeenCalledWith(key));
    await userEvent.click(screen.getByRole('button', { name: 'Build success report' }));
    expect(api.createSuccessReport).toHaveBeenCalledTimes(1);
    expect(api.createSuccessReport).toHaveBeenCalledWith(key, {});
    expect(await screen.findByTestId('overall-verdict')).toHaveTextContent('not assessable');
    const [first, second] = built.report.criteria;
    const row1 = screen.getByTestId(`criterion-${first.id}`);
    expect(row1).toHaveTextContent('77.5%');
    expect(row1).toHaveTextContent(first.held_out_test as string);
    expect(row1).toHaveTextContent('2% of 540 = 10 rows (few negatives)');
    expect(screen.getByTestId(`verdict-${first.id}`)).toHaveTextContent('met');
    expect(screen.getByTestId(`criterion-${second.id}`)).toHaveTextContent('in distribution');
    expect(screen.getByTestId(`criterion-${second.id}`)).toHaveTextContent('engine not available');
  });

  it('a stale report says which input moved and the call that rebuilds it', async () => {
    const key = slug();
    api.createSuccessReport.mockResolvedValue(envelope(key, { stale: {
      stale: true, moved: [{ input: 'behaviour_version', was: 2, now: 3 }], inputs_removed: {},
      next: [{ tool: 'get_detector_success_report', arguments: { key }, why: 'rebuild' }] } }));
    render(<DetectorSuccessReport behaviourKey={key} />);
    await userEvent.click(screen.getByRole('button', { name: 'Build success report' }));
    expect(await screen.findByTestId('report-stale')).toHaveTextContent('behaviour version');
    expect(screen.getByTestId('report-stale')).toHaveTextContent('Next: get_detector_success_report');
  });

  it('a refusal renders its message and its next call', async () => {
    const key = slug();
    api.createSuccessReport.mockRejectedValue(new APIError(409, 'two models', {
      code: 'ambiguous_model', message: '2 models have a probe; name one',
      next: [{ tool: 'get_detector_success_report', arguments: { key, model_id: 'm_a' }, why: 'name it' }],
    }));
    render(<DetectorSuccessReport behaviourKey={key} />);
    await userEvent.click(screen.getByRole('button', { name: 'Build success report' }));
    const alert = await screen.findByTestId('detector-refusal');
    expect(alert).toHaveTextContent('2 models have a probe; name one');
    expect(alert).toHaveTextContent('Next: get_detector_success_report');
  });

  it('an earlier report is read by its id', async () => {
    const key = slug();
    const earlier = envelope(key);
    api.listSuccessReports.mockResolvedValue({ key, total: 1, next_cursor: null, reports: [
      { id: earlier.id, behaviour_version: 1, model_id: 'm_x', overall_verdict: 'met', created_by: 'o', created_at: null }] });
    api.readSuccessReport.mockResolvedValue(earlier);
    render(<DetectorSuccessReport behaviourKey={key} />);
    await userEvent.click(await screen.findByRole('button', { name: earlier.id }));
    expect(api.readSuccessReport).toHaveBeenCalledWith(earlier.id);
    expect(await screen.findByTestId('success-report')).toBeInTheDocument();
  });
});

describe('the model comparison', () => {
  it('renders one row per model with the blend column', async () => {
    const key = slug();
    const models = [slug(), slug(), slug()];
    api.compareModels.mockResolvedValue({ key, behaviour_version: 1, dataset_version: null, total: 3, next_cursor: null,
      models: models.map((m, i) => ({ model_id: m, overall_verdict: i === 2 ? 'not_met' : 'met',
        criteria: [{ id: 'c-one', verdict: 'met', recall: 0.8, reasons: [] }],
        blend: { verdict: 'not_assessable', reasons: ['no_blend_report'] } })) });
    render(<ModelComparison behaviourKey={key} />);
    await userEvent.click(screen.getByRole('button', { name: 'Compare models' }));
    expect(api.compareModels).toHaveBeenCalledWith(key);
    const table = await screen.findByTestId('model-comparison');
    expect(within(table).getAllByRole('row')).toHaveLength(3);
    for (const m of models) expect(table).toHaveTextContent(m);
    expect(table).toHaveTextContent('no blend report');
  });
});

describe('the breakdown', () => {
  it('loads evaluations, picks a declared column and shows too-small groups', async () => {
    const column = slug();
    probeApi.getReport.mockResolvedValue({ evaluations: [{ id: 'pme_1', probe_id: 'pm_1', dataset_id: 'pmd_t' }] });
    probeApi.listDatasets.mockResolvedValue([{ id: 'pmd_t', name: 'test view', group_columns: [column] }]);
    const body: ProbeBreakdown = {
      probe_id: 'pm_1', evaluation_id: 'pme_1', view_id: 'pmd_t',
      decision: { mode: 'recorded', revision: 2, scope: 'all', window: 'all', rule_counts: { global: 5 }, target_fpr: null, threshold: null },
      calibration: null, column, filters: [], min_rows: 30, total: 2, next_cursor: null, inputs_digest: 'd', settings: {},
      groups: [
        { value: 'a', positives: { n: 40, flagged: 30, recall: 0.75, ci: [0.6, 0.86] }, negatives: { n: 60, flagged: 1, fpr: 0.0167, ci: [0, 0.09] }, median_score: 1, rows: 100, too_small: false },
        { value: 'b', positives: { n: 12, flagged: 3, recall: 0.25, ci: [0.09, 0.53] }, negatives: { n: 0, flagged: 0, fpr: null, ci: null }, median_score: 0.1, rows: 12, too_small: true },
      ],
    };
    api.getBreakdown.mockResolvedValue(body);
    render(<Breakdown probes={[{ id: 'pm_1' } as never]} />);
    await userEvent.click(screen.getByRole('button', { name: 'Load evaluations' }));
    await userEvent.selectOptions(await screen.findByLabelText('Breakdown column'), column);
    await userEvent.click(screen.getByRole('button', { name: 'Show breakdown' }));
    expect(api.getBreakdown).toHaveBeenCalledTimes(1);
    expect(api.getBreakdown).toHaveBeenCalledWith('pm_1', 'pme_1', { by_column: column, mode: 'recorded' });
    const groups = await screen.findAllByTestId('breakdown-group');
    expect(groups).toHaveLength(2);
    expect(groups[1]).toHaveTextContent('too small (under 30 rows)');
    expect(groups[0]).toHaveTextContent('recall 75.0% (30/40)');
  });

  it('declares a new column additively on the evaluated view', async () => {
    const [declared, added] = [slug(), slug()];
    probeApi.getReport.mockResolvedValue({ evaluations: [{ id: 'pme_1', probe_id: 'pm_1', dataset_id: 'pmd_t' }] });
    probeApi.listDatasets.mockResolvedValue([{ id: 'pmd_t', name: 'test view', group_columns: [declared] }]);
    api.updateViewColumns.mockResolvedValue({ view_id: 'pmd_t', group_columns: [declared, added], added: [added] });
    render(<Breakdown probes={[{ id: 'pm_1' } as never]} />);
    await userEvent.click(screen.getByRole('button', { name: 'Load evaluations' }));
    await userEvent.type(await screen.findByLabelText('Column to declare'), added);
    await userEvent.click(screen.getByRole('button', { name: 'Declare column on test view' }));
    expect(api.updateViewColumns).toHaveBeenCalledTimes(1);
    expect(api.updateViewColumns).toHaveBeenCalledWith('pmd_t', [declared, added]);
    // the views are read again, so the new column can be chosen at once
    await waitFor(() => expect(probeApi.listDatasets).toHaveBeenCalledTimes(2));
  });
});
