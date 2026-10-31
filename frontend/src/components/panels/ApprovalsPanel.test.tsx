/**
 * The Approvals page (ENH-008 D6, §2.5, S9): the waiting cards and the History, with the shared wording.
 * A DECIDED REQUEST APPEARS IN HISTORY: one waiting request is rejected with the reason "x"; the mocked API then
 * returns it as denied, and its row reads "Rejected", "operator", "x", with the local time and the exact instant as
 * its title. Reject posts to `/deny` once, with the reason.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import type { ApprovalRequest } from '../../types/featureGroups';

const api = vi.hoisted(() => ({
  listApprovals: vi.fn(),
  listDecidedApprovals: vi.fn(),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
}));
vi.mock('../../api/featureGroups', () => ({ ...api, getCodeRuleState: vi.fn(() => new Promise(() => {})), HISTORY_PAGE: 200 }));
vi.mock('../../api/models', () => ({ getModel: vi.fn() }));
vi.mock('../../api/probeMonitors', () => ({ probeMonitorsApi: { listDatasets: vi.fn() } }));
vi.mock('../../api/system', () => ({ getGPUList: vi.fn() }));

import { ApprovalsPanel } from './ApprovalsPanel';
import { useApprovalsStore } from '../../stores/approvalsStore';

const PENDING: ApprovalRequest = {
  id: 'req-1', tool_name: 'start_probe_judge_run', payload: { model: 'judge', dataset_ids: ['d1'] },
  status: 'pending', reason: null, steering_task_id: null, created_at: '2026-10-28T10:00:00Z', resolved_at: null,
  requested_by: 'mcp', expires_at: '2099-01-01T00:00:00Z',
};
const DECIDED_AT = '2026-10-28T18:30:00Z';

beforeEach(() => {
  Object.values(api).forEach((f) => f.mockReset());
  useApprovalsStore.setState({ pending: [], loaded: false, error: null, history: [], historyLoaded: false, historyHasMore: false });
});

describe('ApprovalsPanel', () => {
  it('shows the wording and both empty states', async () => {
    api.listDecidedApprovals.mockResolvedValue({ approvals: [], total: 0 });
    useApprovalsStore.setState({ pending: [], loaded: true });
    render(<ApprovalsPanel />);
    expect(screen.getByRole('heading', { name: 'Approvals' })).toBeInTheDocument();
    expect(screen.getByText('Requests from AI agents that wait for your decision, and the ones already decided.')).toBeInTheDocument();
    expect(screen.getByText('Waiting for you (0)')).toBeInTheDocument();
    expect(screen.getByText('An AI agent asked to do each of these. Nothing happens until you approve it; a request you do not decide expires.')).toBeInTheDocument();
    expect(screen.getByText('No agent requests are waiting.')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'History' })).toBeInTheDocument();
    expect(screen.getByText('Requests you decided, and the ones that expired.')).toBeInTheDocument();
    expect(await screen.findByText('No decided requests yet.')).toBeInTheDocument();
  });

  it('a decided request appears in History, with who, the reason and the local time', async () => {
    api.listDecidedApprovals.mockResolvedValueOnce({ approvals: [], total: 0 });
    useApprovalsStore.setState({ pending: [PENDING], loaded: true });
    render(<ApprovalsPanel />);
    await screen.findByText('No decided requests yet.');

    api.denyRequest.mockResolvedValue({});
    api.listApprovals.mockResolvedValue({ approvals: [], total: 0 });
    api.listDecidedApprovals.mockResolvedValue({
      approvals: [{ ...PENDING, status: 'denied', reason: 'x', decided_by: 'operator', resolved_at: DECIDED_AT, expires_at: null }],
      total: 1,
    });
    const card = screen.getByTestId('approval-request');
    fireEvent.click(within(card).getByRole('button', { name: /Reject/ }));
    fireEvent.change(within(card).getByPlaceholderText('Reason (optional)'), { target: { value: 'x' } });
    await act(async () => fireEvent.click(within(card).getByRole('button', { name: 'Confirm reject' })));

    expect(api.denyRequest).toHaveBeenCalledTimes(1);
    expect(api.denyRequest).toHaveBeenCalledWith('req-1', 'x');
    await waitFor(() => expect(screen.getByText('Waiting for you (0)')).toBeInTheDocument());
    const table = screen.getByTestId('approvals-history-table');
    const row = within(table).getByTestId('approval-history-row');
    const cells = within(row).getAllByRole('cell').map((c) => c.textContent);
    expect(cells[0]).toBe('Rejected');
    expect(cells[1]).toBe('Judge probe datasets with miLLM');
    expect(cells[2]).toBe('agent mcp');
    expect(cells[4]).toBe('operator');
    expect(cells[6]).toBe('x');
    expect(cells[8]).toBe('req-1');
    const decided = within(row).getAllByRole('cell')[5];
    expect(decided).toHaveAttribute('title', '2026-10-28T18:30:00.000Z');
    expect(decided.textContent).toMatch(/2026/);
    expect(decided.textContent).not.toMatch(/Invalid/);
  });

  it('labels each outcome in the shared words', async () => {
    const base = { ...PENDING, resolved_at: DECIDED_AT, expires_at: null };
    api.listDecidedApprovals.mockResolvedValue({
      approvals: [
        { ...base, id: 's', tool_name: 'steer_compare', status: 'approved', steering_task_id: 't9', decided_by: 'operator' },
        { ...base, id: 'w', tool_name: 'start_probe_training_run', status: 'approved', decided_by: 'operator' },
        { ...base, id: 'u', tool_name: 'start_probe_training_run', status: 'consumed', decided_by: 'operator' },
        { ...base, id: 'r', tool_name: 'code_rule_approve', status: 'approved', decided_by: 'operator' },
        { ...base, id: 'e', status: 'expired', decided_by: 'system', reason: 'expired: pending for more than 24 h' },
      ],
      total: 5,
    });
    render(<ApprovalsPanel />);
    await screen.findByTestId('approvals-history-table');
    const rows = within(screen.getByTestId('approvals-history-table')).getAllByTestId('approval-history-row');
    const outcome = rows.map((r) => within(r).getAllByRole('cell')[0].textContent);
    expect(outcome).toEqual(['Approved and run', 'Approved, waiting for the agent', 'Approved and used',
      'Approved (recorded)', 'Expired']);
    const expired = within(rows[4]).getAllByRole('cell').map((c) => c.textContent);
    expect(expired[4]).toBe('nobody (expired)');
    expect(expired[7]).toBe('expired without a decision');
    expect(within(rows[0]).getAllByRole('cell')[7].textContent).toBe('Steering task t9');
    expect(within(rows[2]).getAllByRole('cell')[7].textContent).toBe('used by the agent');
  });

  it('Show older loads the next 200', async () => {
    const page = Array.from({ length: 200 }, (_, i) => ({ ...PENDING, id: `r${i}`, status: 'denied' as const }));
    api.listDecidedApprovals.mockResolvedValueOnce({ approvals: page, total: 200 });
    render(<ApprovalsPanel />);
    const older = await screen.findByRole('button', { name: 'Show older' });
    api.listDecidedApprovals.mockResolvedValueOnce({ approvals: [{ ...PENDING, id: 'oldest', status: 'expired' }], total: 1 });
    await act(async () => fireEvent.click(older));
    expect(api.listDecidedApprovals).toHaveBeenLastCalledWith(200, 200);
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Show older' })).toBeNull());
  });
});
