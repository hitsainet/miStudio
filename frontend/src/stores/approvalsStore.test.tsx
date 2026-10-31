/**
 * The single owner of the queue (ENH-008 D7): it polls every 15 seconds, refreshes on the
 * `mcp/approvals` events, sorts the waiting list soonest-to-expire first, and History is a separate,
 * paged read of decided requests.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, render } from '@testing-library/react';

const api = vi.hoisted(() => ({
  listApprovals: vi.fn(),
  listDecidedApprovals: vi.fn(),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
}));
vi.mock('../api/featureGroups', () => ({ ...api, HISTORY_PAGE: 200 }));
const ws = vi.hoisted(() => ({ handlers: {} as Record<string, () => void>, subscribe: vi.fn() }));
vi.mock('../contexts/WebSocketContext', () => ({
  useWebSocketContext: () => ({
    on: (e: string, h: () => void) => { ws.handlers[e] = h; },
    off: vi.fn(), subscribe: ws.subscribe, unsubscribe: vi.fn(), isConnected: true,
  }),
}));

import { APPROVALS_POLL_MS, useApprovalsLive, useApprovalsStore } from './approvalsStore';

function Live() {
  useApprovalsLive();
  return null;
}

beforeEach(() => {
  Object.values(api).forEach((f) => f.mockReset());
  api.listApprovals.mockResolvedValue({ approvals: [], total: 0 });
  useApprovalsStore.setState({ pending: [], loaded: false, error: null, history: [], historyLoaded: false });
});
afterEach(() => vi.useRealTimers());

describe('approvalsStore', () => {
  it('polls the waiting list every 15 seconds', async () => {
    vi.useFakeTimers();
    expect(APPROVALS_POLL_MS).toBe(15_000);
    render(<Live />);
    expect(api.listApprovals).toHaveBeenCalledTimes(1);
    await act(async () => { vi.advanceTimersByTime(15_000); });
    expect(api.listApprovals).toHaveBeenCalledTimes(2);
    expect(api.listApprovals).toHaveBeenLastCalledWith('pending');
  });

  it('refreshes on the mcp/approvals events', async () => {
    render(<Live />);
    expect(ws.subscribe).toHaveBeenCalledWith('mcp/approvals');
    const before = api.listApprovals.mock.calls.length;
    await act(async () => ws.handlers['approval:created']());
    await act(async () => ws.handlers['approval:resolved']());
    expect(api.listApprovals.mock.calls.length).toBe(before + 2);
  });

  it('History is the decided list, 200 at a time, and Show older asks for the next 200', async () => {
    const page = Array.from({ length: 200 }, (_, i) => ({ id: `r${i}`, status: 'denied' }));
    api.listDecidedApprovals.mockResolvedValueOnce({ approvals: page, total: 200 });
    await act(() => useApprovalsStore.getState().loadHistory());
    expect(api.listDecidedApprovals).toHaveBeenCalledWith(0, 200);
    expect(useApprovalsStore.getState().historyHasMore).toBe(true);
    api.listDecidedApprovals.mockResolvedValueOnce({ approvals: [{ id: 'old', status: 'expired' }], total: 1 });
    await act(() => useApprovalsStore.getState().loadOlder());
    expect(api.listDecidedApprovals).toHaveBeenLastCalledWith(200, 200);
    expect(useApprovalsStore.getState().history).toHaveLength(201);
    expect(useApprovalsStore.getState().historyHasMore).toBe(false);
  });

  it('a reject reason is trimmed and capped at 1000 characters', async () => {
    api.denyRequest.mockResolvedValue({});
    await act(() => useApprovalsStore.getState().reject('a', `  ${'r'.repeat(1200)}  `));
    expect(api.denyRequest).toHaveBeenCalledWith('a', 'r'.repeat(1000));
  });
});
