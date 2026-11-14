/**
 * ENH-006 G15: the operator sees a probe run's TIME forecast, and the evaluation rows it assumes, on the card
 * they approve from — the banner and the Approvals page share it. A Llama run that scored a 1,000,000-row set
 * whole had an ETA of 347 hours, found only after it started; this is where that number must be read first.
 *
 * NEGATIVE CONTROL (2026-11-09, review record g15_time_forecast_2026-11-09.md): deleting the `<TimeForecast …/>`
 * line in ApprovalCard turns every test here red.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import type { ApprovalRequest } from '../../types/featureGroups';

vi.mock('../../api/featureGroups', () => ({
  listApprovals: vi.fn(),
  listDecidedApprovals: vi.fn(() => new Promise(() => {})),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
  getCodeRuleState: vi.fn(),
  HISTORY_PAGE: 200,
}));
vi.mock('../../api/models', () => ({ getModel: vi.fn(() => Promise.resolve({ id: 'm_1', name: 'Llama' })) }));
vi.mock('../../api/probeMonitors', () => ({ probeMonitorsApi: { listDatasets: vi.fn(() => Promise.resolve([])) } }));
vi.mock('../../api/system', () => ({ getGPUList: vi.fn(() => Promise.resolve({ gpus: [] })) }));

import { ApprovalsBanner } from '../layout/ApprovalsBanner';
import { ApprovalsPanel } from '../panels/ApprovalsPanel';
import { useApprovalsStore } from '../../stores/approvalsStore';
import { useUIStore } from '../../stores/uiStore';

const SUMMARY =
  'About 5.6 h (3.8 h to 7.7 h) by the measured rates of 5 completed run(s) of this model at scope all ' +
  '(newest pmr_55715c1c87e2). Evaluation scores 15,895 rows over 6 set(s), each capped at eval_max_rows 5,000.';

function run(id: string, forecast: ApprovalRequest['forecast']): ApprovalRequest {
  return {
    id, tool_name: 'start_probe_training_run', status: 'pending', reason: null, steering_task_id: null,
    created_at: '2026-11-09T08:00:00Z', resolved_at: null, requested_by: 'mcp',
    payload: { model_id: 'm_1', train_dataset_id: 'pmd_t', eval_dataset_ids: ['pmd_a', 'pmd_b'], gpu: 'auto' },
    forecast,
  };
}

const MEASURED = run('apr_measured', {
  time: {
    forecast: true, summary: SUMMARY, eval_rows: 15895, eval_max_rows: 5000,
    eval_rows_by_set: { pmd_a: 10895, pmd_b: 5000 }, seconds: { low: 13680, typical: 20253, high: 27684 },
    complete: true,
  },
});
const WHOLE = run('apr_whole', {
  time: {
    forecast: true, summary: 'About 95.4 h (61.0 h to 110.2 h) ...', eval_rows: 1_015_895, eval_max_rows: null,
    eval_rows_by_set: { pmd_a: 15895, pmd_b: 1_000_000 },
  },
});
const NOTHING_MEASURED = run('apr_none', {
  time: {
    forecast: false, eval_rows: 4000, eval_max_rows: 2000, eval_rows_by_set: { pmd_a: 2000, pmd_b: 2000 },
    summary: 'No time forecast: no completed run of model m_1 at scope all records stage timings, so there is ' +
      'nothing measured to forecast from.',
  },
});
const OLD_ROW = run('apr_old', null);

function card(id: string): HTMLElement {
  const found = document.querySelector(`[data-approval-id="${id}"]`);
  if (!found) throw new Error(`no card ${id}`);
  return found as HTMLElement;
}

beforeEach(() => {
  useUIStore.setState({ activePanel: 'datasets' });
  useApprovalsStore.setState({ pending: [MEASURED, WHOLE, NOTHING_MEASURED, OLD_ROW], loaded: true, error: null });
});

describe('the time forecast on the approval card', () => {
  it('shows the one-line summary and the evaluation rows it assumes', () => {
    render(<ApprovalsBanner />);
    const c = within(card('apr_measured'));
    expect(c.getByTestId('approval-time-forecast')).toHaveTextContent(`Time forecast: ${SUMMARY}`);
    expect(c.getByTestId('approval-time-forecast-eval-rows')).toHaveTextContent(
      'Evaluation rows it assumes: 15,895 rows over 2 sets, each set capped at 5,000 (eval_max_rows)',
    );
  });

  it('says when every set is scored whole — the G15 shape', () => {
    render(<ApprovalsBanner />);
    expect(within(card('apr_whole')).getByTestId('approval-time-forecast-eval-rows')).toHaveTextContent(
      '1,015,895 rows over 2 sets, every set scored whole (no eval_max_rows)',
    );
  });

  it('says plainly when nothing was measured, and still gives the rows', () => {
    render(<ApprovalsBanner />);
    const c = within(card('apr_none'));
    expect(c.getByTestId('approval-time-forecast')).toHaveTextContent('Time forecast: No time forecast: no completed run');
    expect(c.getByTestId('approval-time-forecast-eval-rows')).toHaveTextContent('4,000 rows over 2 sets');
  });

  it('says a probe-run request filed without one has none, rather than nothing', () => {
    // The banner shows the first three waiting requests; the page shows every one.
    render(<ApprovalsPanel />);
    expect(within(card('apr_old')).getByTestId('approval-time-forecast')).toHaveTextContent(
      'Time forecast: none was filed with this request.',
    );
  });

  it('is on the Approvals page too', () => {
    render(<ApprovalsPanel />);
    expect(within(card('apr_measured')).getByTestId('approval-time-forecast')).toHaveTextContent(SUMMARY);
  });

  it('appears once per card', () => {
    render(<ApprovalsPanel />);
    expect(screen.getAllByTestId('approval-time-forecast')).toHaveLength(4);
  });
});
