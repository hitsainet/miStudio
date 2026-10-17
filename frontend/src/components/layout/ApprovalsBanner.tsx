/**
 * Pending MCP-agent approval requests (Feature 010; app shell since Feature 034).
 *
 * Mounted ONCE, in the app shell above every panel (`App.tsx`, T-97). It used
 * to live in the Steering panel, which was right while only steering asked for
 * approval; an agent's miLLM model load (Feature 034, FR-20) evicts the served
 * model and must be visible wherever the operator is. One mount also means one
 * `mcp/approvals` subscription, so no other component's unsubscribe can
 * silence it (the per-socket subscription lesson).
 *
 * Live via the mcp/approvals channel with a slow poll fallback; renders
 * NOTHING when there are no pending requests — the padding is inside the root
 * below, so an empty queue leaves no gap above the panel.
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
    <div className="px-6 pt-4" data-testid="approvals-banner">
    <div className="bg-amber-950/40 border border-amber-800 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-2">
        <Bot className="w-4 h-4 text-amber-400" />
        <span className="text-sm font-medium text-amber-300">
          {pending.length} agent request{pending.length > 1 ? 's' : ''} awaiting approval
        </span>
      </div>
      {error && <p className="text-xs text-red-400 mb-2">{error}</p>}
      <div className="space-y-2">
        {pending.map((request) => {
          return (
            <div
              key={request.id}
              className="flex items-center gap-3 bg-slate-100 dark:bg-slate-900/60 rounded-md px-3 py-2"
            >
              <div className="flex-1 min-w-0">
                <span className="text-xs font-mono text-amber-200">{request.tool_name}</span>
                <span className="text-xs text-slate-600 dark:text-slate-400 ml-2">
                  {renderSummary(request)}
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
    </div>
  );
}

const STEERING_TOOLS = new Set(['steer_compare', 'steer_sweep', 'steer_combined']);

/**
 * One line per request, by tool. An UNKNOWN tool shows its name and a JSON
 * preview, so a future approvable tool is visible rather than blank — a
 * request the operator cannot read is one they approve blind.
 */
export function renderSummary(request: ApprovalRequest): string {
  if (request.tool_name === 'millm_load_model') {
    const modelId = (request.payload as { model_id?: unknown }).model_id;
    return `Load miLLM model #${String(modelId)} — evicts the resident model`;
  }
  if (STEERING_TOOLS.has(request.tool_name)) {
    const payload = request.payload as {
      prompt?: string;
      selected_features?: unknown[];
      strength_values?: number[];
    };
    const featureCount =
      payload.selected_features?.length ?? (payload.strength_values ? 1 : 0);
    return `${featureCount} feature${featureCount === 1 ? '' : 's'} · "${(payload.prompt ?? '').slice(0, 60)}"`;
  }
  return JSON.stringify(request.payload).slice(0, 120);
}
