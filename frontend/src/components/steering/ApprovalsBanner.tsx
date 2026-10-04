/**
 * Pending MCP-agent approval requests (Feature 010, operator-approval mode).
 *
 * Shown at the top of the Steering panel. Live via the mcp/approvals channel
 * with a slow poll fallback; renders nothing when there are no pending
 * requests (i.e. approval mode off or queue empty).
 */

import { useCallback, useEffect, useState } from 'react';
import { Bot, Check, X } from 'lucide-react';
import { approveRequest, denyRequest, listApprovals } from '../../api/featureGroups';
import { useWebSocketContext } from '../../contexts/WebSocketContext';
import type { ApprovalRequest } from '../../types/featureGroups';

export function ApprovalsBanner() {
  const [pending, setPending] = useState<ApprovalRequest[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();

  const refresh = useCallback(async () => {
    try {
      const response = await listApprovals('pending');
      setPending(response.approvals);
    } catch {
      // Endpoint always exists; transient errors just skip a refresh cycle.
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), 20000);
    return () => clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!isConnected) return;
    subscribe('mcp/approvals');
    const handler = () => void refresh();
    on('approval:created', handler);
    on('approval:resolved', handler);
    return () => {
      off('approval:created', handler);
      off('approval:resolved', handler);
      unsubscribe('mcp/approvals');
    };
  }, [isConnected, on, off, subscribe, unsubscribe, refresh]);

  const resolve = async (id: string, action: 'approve' | 'deny') => {
    setBusy(id);
    setError(null);
    try {
      if (action === 'approve') await approveRequest(id);
      else await denyRequest(id, 'Denied by operator');
      await refresh();
    } catch (e: any) {
      setError(e.detail || e.message || `Failed to ${action}`);
    } finally {
      setBusy(null);
    }
  };

  if (pending.length === 0) return null;

  return (
    <div className="bg-amber-950/40 border border-amber-800 rounded-lg p-4 mb-4">
      <div className="flex items-center gap-2 mb-2">
        <Bot className="w-4 h-4 text-amber-400" />
        <span className="text-sm font-medium text-amber-300">
          {pending.length} agent steering request{pending.length > 1 ? 's' : ''} awaiting approval
        </span>
      </div>
      {error && <p className="text-xs text-red-400 mb-2">{error}</p>}
      <div className="space-y-2">
        {pending.map((request) => {
          const payload = request.payload as {
            prompt?: string;
            selected_features?: unknown[];
            strength_values?: number[];
          };
          const featureCount =
            payload.selected_features?.length ?? (payload.strength_values ? 1 : 0);
          return (
            <div
              key={request.id}
              className="flex items-center gap-3 bg-slate-100 dark:bg-slate-900/60 rounded-md px-3 py-2"
            >
              <div className="flex-1 min-w-0">
                <span className="text-xs font-mono text-amber-200">{request.tool_name}</span>
                <span className="text-xs text-slate-600 dark:text-slate-400 ml-2">
                  {featureCount} feature{featureCount === 1 ? '' : 's'} · "
                  {(payload.prompt ?? '').slice(0, 60)}"
                </span>
              </div>
              <button
                disabled={busy === request.id}
                onClick={() => void resolve(request.id, 'approve')}
                className="flex items-center gap-1 px-2.5 py-1 text-xs bg-emerald-700 hover:bg-emerald-600 text-white rounded disabled:opacity-50"
              >
                <Check className="w-3 h-3" /> Approve
              </button>
              <button
                disabled={busy === request.id}
                onClick={() => void resolve(request.id, 'deny')}
                className="flex items-center gap-1 px-2.5 py-1 text-xs bg-slate-100 dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 text-slate-800 dark:text-slate-200 rounded disabled:opacity-50"
              >
                <X className="w-3 h-3" /> Deny
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
