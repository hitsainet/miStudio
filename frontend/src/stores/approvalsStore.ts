/**
 * The ONE owner of miStudio's approval queue in the UI (ENH-008 D7, §3.2 frontend).
 *
 * `useApprovalsLive()` is mounted once, in the app shell (`App.tsx`): it polls the waiting list
 * every 15 seconds and refreshes on the `mcp/approvals` WebSocket events. The banner, the menu
 * badge and the Approvals page all read this store, so they always agree, and the poll keeps
 * running on the Approvals page where the banner is not drawn. One mount also means one
 * `mcp/approvals` subscription (the per-socket subscription lesson).
 *
 * Approve and reject are the same calls from the banner and from the page. Reject always sends a
 * reason: the operator's, or "Rejected by the operator." when the field is left blank (D8c).
 */
import { useEffect } from 'react';
import { create } from 'zustand';
import {
  approveBatch as postApproveBatch,
  approveRequest,
  denyRequest,
  HISTORY_PAGE,
  listApprovals,
  listDecidedApprovals,
} from '../api/featureGroups';
import { APIError } from '../api/client';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import type { ApproveBatchKind, ApproveBatchResult } from '../api/featureGroups';
import type { ApprovalRequest } from '../types/featureGroups';

export const APPROVALS_POLL_MS = 15_000;
export const DEFAULT_REJECT_REASON = 'Rejected by the operator.';
export const REJECT_REASON_MAX = 1000;

interface ApprovalsState {
  /** Waiting requests, the soonest to expire first. */
  pending: ApprovalRequest[];
  /** True once the waiting list has been read at least once. */
  loaded: boolean;
  /** Why the last read failed; null when it succeeded. */
  error: string | null;
  /** Decided requests, the most recent decision first. */
  history: ApprovalRequest[];
  historyLoaded: boolean;
  historyError: string | null;
  /** Another page of History may exist. */
  historyHasMore: boolean;
  refresh: () => Promise<void>;
  loadHistory: () => Promise<void>;
  loadOlder: () => Promise<void>;
  approve: (id: string) => Promise<void>;
  /** ENH-006 D81: one operator call approving every listed request of one kind; refreshes after. */
  approveBatch: (kind: ApproveBatchKind, ids: string[]) => Promise<ApproveBatchResult>;
  reject: (id: string, reason: string) => Promise<void>;
}

function expiryKey(r: ApprovalRequest): number {
  const t = r.expires_at ? Date.parse(r.expires_at) : NaN;
  return Number.isNaN(t) ? Number.POSITIVE_INFINITY : t;
}

/** Soonest to expire first; requests that never expire last, oldest first among equals. */
export function bySoonestExpiry(rows: ApprovalRequest[]): ApprovalRequest[] {
  return [...rows].sort((a, b) => {
    const d = expiryKey(a) - expiryKey(b);
    if (d !== 0 && !Number.isNaN(d)) return d;
    return (a.created_at ?? '').localeCompare(b.created_at ?? '');
  });
}

function message(e: unknown): string {
  if (e instanceof APIError) return e.detail;
  if (e instanceof Error) return e.message;
  return String(e);
}

export const useApprovalsStore = create<ApprovalsState>((set, get) => ({
  pending: [],
  loaded: false,
  error: null,
  history: [],
  historyLoaded: false,
  historyError: null,
  historyHasMore: false,

  refresh: async () => {
    try {
      const response = await listApprovals('pending');
      set({ pending: bySoonestExpiry(response.approvals), loaded: true, error: null });
    } catch (e) {
      // Never silent (§2.3): the banner says the queue could not be read.
      set({ error: message(e), loaded: true });
    }
  },

  loadHistory: async () => {
    try {
      // ⚠ THE HISTORY FILTER is the server's (`decided=true`: every status but pending).
      const response = await listDecidedApprovals(0, HISTORY_PAGE);
      set({
        history: response.approvals,
        historyLoaded: true,
        historyError: null,
        historyHasMore: response.approvals.length >= HISTORY_PAGE,
      });
    } catch (e) {
      set({ historyError: message(e), historyLoaded: true });
    }
  },

  loadOlder: async () => {
    try {
      const have = get().history;
      const response = await listDecidedApprovals(have.length, HISTORY_PAGE);
      const seen = new Set(have.map((r) => r.id));
      set({
        history: [...have, ...response.approvals.filter((r) => !seen.has(r.id))],
        historyHasMore: response.approvals.length >= HISTORY_PAGE,
        historyError: null,
      });
    } catch (e) {
      set({ historyError: message(e) });
    }
  },

  approve: async (id) => {
    try {
      await approveRequest(id);
    } finally {
      await Promise.all([get().refresh(), get().historyLoaded ? get().loadHistory() : Promise.resolve()]);
    }
  },

  approveBatch: async (kind, ids) => {
    try {
      return await postApproveBatch(kind, ids);
    } finally {
      await Promise.all([get().refresh(), get().historyLoaded ? get().loadHistory() : Promise.resolve()]);
    }
  },

  reject: async (id, reason) => {
    const sent = reason.trim().slice(0, REJECT_REASON_MAX) || DEFAULT_REJECT_REASON;
    try {
      await denyRequest(id, sent);
    } finally {
      await Promise.all([get().refresh(), get().historyLoaded ? get().loadHistory() : Promise.resolve()]);
    }
  },
}));

/** The single poll and WebSocket refresh. Mount ONCE, in the app shell. */
export function useApprovalsLive(): void {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();

  useEffect(() => {
    const { refresh } = useApprovalsStore.getState();
    void refresh();
    const timer = setInterval(() => void useApprovalsStore.getState().refresh(), APPROVALS_POLL_MS);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!isConnected) return;
    subscribe('mcp/approvals');
    const handler = () => {
      const state = useApprovalsStore.getState();
      void state.refresh();
      if (state.historyLoaded) void state.loadHistory();
    };
    on('approval:created', handler);
    on('approval:resolved', handler);
    return () => {
      off('approval:created', handler);
      off('approval:resolved', handler);
      unsubscribe('mcp/approvals');
    };
  }, [isConnected, on, off, subscribe, unsubscribe]);
}
