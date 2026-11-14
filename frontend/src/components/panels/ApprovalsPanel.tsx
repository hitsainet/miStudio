/**
 * The Approvals page (ENH-008 D6, §2.5): **Waiting for you (n)** — the waiting cards, the soonest to
 * expire first — and **History** — decided requests, the most recent decision first, 200 at a time
 * with **Show older**. Same cards and the same calls as the banner, which is not drawn here.
 */
import { useEffect } from 'react';
import { ShieldCheck } from 'lucide-react';
import { ApprovalCardList } from '../approvals/ApprovalCardList';
import { ApproveAllCodeRules } from '../approvals/ApproveAllCodeRules';
import { ApprovalHistory } from '../approvals/ApprovalHistory';
import { useProbeRunContext } from '../approvals/ApprovalCard';
import { useApprovalsStore } from '../../stores/approvalsStore';

export function ApprovalsPanel() {
  const pending = useApprovalsStore((s) => s.pending);
  const error = useApprovalsStore((s) => s.error);
  const history = useApprovalsStore((s) => s.history);
  const historyLoaded = useApprovalsStore((s) => s.historyLoaded);
  const historyError = useApprovalsStore((s) => s.historyError);
  const historyHasMore = useApprovalsStore((s) => s.historyHasMore);
  const context = useProbeRunContext(pending);

  useEffect(() => {
    void useApprovalsStore.getState().loadHistory();
  }, []);

  return (
    <div className="p-6 space-y-6" data-testid="approvals-panel">
      <header>
        <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-500" aria-hidden="true" />
          Approvals
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Requests from AI agents that wait for your decision, and the ones already decided.
        </p>
      </header>

      <section aria-labelledby="approvals-waiting-heading" className="space-y-2">
        <h2 id="approvals-waiting-heading" className="text-base font-medium text-slate-900 dark:text-slate-100" aria-live="polite">
          Waiting for you ({pending.length})
        </h2>
        <p className="text-xs text-slate-600 dark:text-slate-400">
          An AI agent asked to do each of these. Nothing happens until you approve it; a request you do not decide expires.
        </p>
        {error && (
          <p className="text-xs text-red-600 dark:text-red-400" role="alert">
            Could not read the approval queue: {error}
          </p>
        )}
        {pending.length === 0 ? (
          <p className="text-sm text-slate-500" data-testid="approvals-waiting-empty">No agent requests are waiting.</p>
        ) : (
          <>
            <ApproveAllCodeRules pending={pending} />
            <ApprovalCardList requests={pending} context={context} />
          </>
        )}
      </section>

      <section aria-labelledby="approvals-history-heading" className="space-y-2">
        <h2 id="approvals-history-heading" className="text-base font-medium text-slate-900 dark:text-slate-100">
          History
        </h2>
        <p className="text-xs text-slate-600 dark:text-slate-400">Requests you decided, and the ones that expired.</p>
        {historyError && (
          <p className="text-xs text-red-600 dark:text-red-400" role="alert">
            Could not read the history: {historyError}
          </p>
        )}
        {historyLoaded && history.length === 0 && !historyError ? (
          <p className="text-sm text-slate-500" data-testid="approvals-history-empty">No decided requests yet.</p>
        ) : (
          <ApprovalHistory rows={history} />
        )}
        {historyHasMore && (
          <button
            type="button"
            onClick={() => void useApprovalsStore.getState().loadOlder()}
            className="px-3 py-1.5 text-xs bg-slate-200 dark:bg-slate-700 text-slate-800 dark:text-slate-200 rounded-control"
          >
            Show older
          </button>
        )}
      </section>
    </div>
  );
}
