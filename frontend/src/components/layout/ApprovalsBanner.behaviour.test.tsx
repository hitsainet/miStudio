/**
 * The banner's shared behaviour (ENH-008 §2.3, S7): a labelled region at the top of every panel, at most 3
 * cards (the soonest to expire first) and "and N more waiting", Collapse/Expand kept for the session and undone by
 * a NEW request, a failed read said out loud, not drawn on the Approvals panel, and Approve / Reject calling the
 * API once each, with the operator's reason (or the default).
 *
 * Mutation controls (0xcc/reviews/enh008_controls_2026-10-28.md): BANNER_MAX_CARDS → 4; drop the
 * `onApprovalsPage` early return; send a fixed reason in the store's reject.
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
vi.mock('../../contexts/WebSocketContext', () => ({
  useWebSocketContext: () => ({ on: vi.fn(), off: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn(), isConnected: false }),
}));

import { ApprovalsBanner } from './ApprovalsBanner';
import { useApprovalsStore } from '../../stores/approvalsStore';
import { useUIStore } from '../../stores/uiStore';

const HOUR = 3600_000;

function req(id: string, hoursLeft: number | null): ApprovalRequest {
  return {
    id, tool_name: 'evaluate_probe_monitor', payload: { probe_id: id }, status: 'pending', reason: null,
    steering_task_id: null, created_at: '2026-10-28T10:00:00Z', resolved_at: null, requested_by: 'mcp',
    expires_at: hoursLeft === null ? null : new Date(Date.now() + hoursLeft * HOUR).toISOString(),
  };
}

async function load(rows: ApprovalRequest[]) {
  api.listApprovals.mockResolvedValue({ approvals: rows, total: rows.length });
  await act(() => useApprovalsStore.getState().refresh());
}

beforeEach(() => {
  Object.values(api).forEach((f) => f.mockReset());
  sessionStorage.clear();
  useUIStore.setState({ activePanel: 'datasets' });
  useApprovalsStore.setState({ pending: [], loaded: false, error: null, history: [], historyLoaded: false });
});

describe('the approvals banner (§2.3)', () => {
  it('is a labelled region with the count heading and an Open Approvals link', async () => {
    await load([req('a', 2)]);
    render(<ApprovalsBanner />);
    const region = screen.getByRole('region', { name: 'Agent requests waiting for your approval' });
    expect(region).toHaveAttribute('data-testid', 'approvals-banner');
    expect(within(region).getByText('1 agent request waiting for your approval')).toHaveAttribute('aria-live', 'polite');
    fireEvent.click(within(region).getByRole('button', { name: 'Open Approvals' }));
    expect(useUIStore.getState().activePanel).toBe('approvals');
  });

  it('shows at most 3 cards, the soonest to expire first, then "and N more waiting"', async () => {
    await load([req('late', 20), req('never', null), req('soon', 1), req('mid', 5), req('next', 3)]);
    render(<ApprovalsBanner />);
    const ids = screen.getAllByTestId('approval-request').map((c) => c.getAttribute('data-approval-id'));
    expect(ids).toEqual(['soon', 'next', 'mid']);
    expect(screen.getByTestId('approvals-banner-more')).toHaveTextContent('and 2 more waiting — Open Approvals');
    expect(screen.getByText('5 agent requests waiting for your approval')).toBeInTheDocument();
  });

  it('collapses to the heading for the session, and a NEW request expands it again', async () => {
    await load([req('a', 2)]);
    const { unmount } = render(<ApprovalsBanner />);
    fireEvent.click(screen.getByRole('button', { name: 'Collapse' }));
    expect(screen.queryAllByTestId('approval-request')).toHaveLength(0);
    expect(screen.getByText('1 agent request waiting for your approval')).toBeInTheDocument();
    unmount();
    render(<ApprovalsBanner />);   // a remount (another panel) keeps the choice
    expect(screen.getByRole('button', { name: 'Expand' })).toBeInTheDocument();
    expect(screen.queryAllByTestId('approval-request')).toHaveLength(0);
    await load([req('a', 2), req('b', 3)]);
    await waitFor(() => expect(screen.getAllByTestId('approval-request')).toHaveLength(2));
    expect(screen.getByRole('button', { name: 'Collapse' })).toBeInTheDocument();
  });

  it('draws nothing on the Approvals panel, which shows the same cards', async () => {
    await load([req('a', 2)]);
    useUIStore.setState({ activePanel: 'approvals' });
    const { container } = render(<ApprovalsBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it('never fails silently: a failed read is said', async () => {
    api.listApprovals.mockRejectedValue(new Error('backend down'));
    await act(() => useApprovalsStore.getState().refresh());
    render(<ApprovalsBanner />);
    expect(screen.getByTestId('approvals-banner-error')).toHaveTextContent(
      'Could not read the approval queue: backend down');
  });

  it('Approve calls the API once with the id; both buttons are off while it runs', async () => {
    await load([req('a', 2)]);
    let finish: () => void = () => {};
    api.approveRequest.mockImplementation(() => new Promise<void>((r) => { finish = r; }));
    render(<ApprovalsBanner />);
    fireEvent.click(screen.getByRole('button', { name: /Approve evaluate a probe/ }));
    expect(screen.getByRole('button', { name: /Approve evaluate a probe/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Reject/ })).toBeDisabled();
    api.listApprovals.mockResolvedValue({ approvals: [], total: 0 });
    await act(async () => finish());
    expect(api.approveRequest).toHaveBeenCalledTimes(1);
    expect(api.approveRequest).toHaveBeenCalledWith('a');
    await waitFor(() => expect(screen.queryByTestId('approvals-banner')).toBeNull());
  });

  it('Reject asks for an optional reason and sends it; blank sends the default', async () => {
    await load([req('a', 2), req('b', 3)]);
    api.denyRequest.mockResolvedValue({});
    render(<ApprovalsBanner />);
    const [first, second] = screen.getAllByTestId('approval-request');
    fireEvent.click(within(first).getByRole('button', { name: /Reject/ }));
    fireEvent.change(within(first).getByPlaceholderText('Reason (optional)'), { target: { value: 'not now' } });
    await act(async () => fireEvent.click(within(first).getByRole('button', { name: 'Confirm reject' })));
    fireEvent.click(within(second).getByRole('button', { name: /Reject/ }));
    await act(async () => fireEvent.click(within(second).getByRole('button', { name: 'Confirm reject' })));
    expect(api.denyRequest.mock.calls).toEqual([['a', 'not now'], ['b', 'Rejected by the operator.']]);
  });

  it('an expired request says so inside the card', async () => {
    const { APIError } = await import('../../api/client');
    await load([req('a', 2)]);
    api.approveRequest.mockRejectedValue(new APIError(409, 'Request is expired, not pending'));
    render(<ApprovalsBanner />);
    await act(async () => fireEvent.click(screen.getByRole('button', { name: /Approve/ })));
    expect(await screen.findByTestId('approval-error')).toHaveTextContent(
      'This request expired. Ask the agent to try again.');
  });
});
