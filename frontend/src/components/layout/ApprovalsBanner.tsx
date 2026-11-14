/**
 * Agent requests waiting for the operator, at the top of every panel (ENH-008 §2.3, D5).
 *
 * Mounted ONCE, in the app shell (`App.tsx`). It reads `approvalsStore`, which owns the single poll
 * and the `mcp/approvals` WebSocket refresh (`useApprovalsLive`, also in the shell), so the banner,
 * the menu badge and the Approvals page always agree.
 *
 * * A labelled region with the heading "{n} agent request(s) waiting for your approval" (announced
 *   politely) and the **Open Approvals** link.
 * * At most 3 cards, the soonest to expire first, then "and {n−3} more waiting — Open Approvals".
 * * Collapse / Expand reduces it to the heading line; the choice is kept in sessionStorage, and a
 *   NEW request expands it again.
 * * Nothing is drawn when nothing waits and the last read succeeded; a failed read says so.
 * * Not drawn on the Approvals panel, which shows the same cards.
 */
import { useEffect, useMemo, useState } from 'react';
import { Bot, ChevronDown, ChevronUp } from 'lucide-react';
import { ApprovalCardList } from '../approvals/ApprovalCardList';
import { ApproveAllCodeRules } from '../approvals/ApproveAllCodeRules';
import { useProbeRunContext } from '../approvals/ApprovalCard';
import { useApprovalsStore } from '../../stores/approvalsStore';
import { useUIStore } from '../../stores/uiStore';

export const BANNER_MAX_CARDS = 3;
/**
 * The banner floats (operator 2026-10-28): the window scrolls and the header is sticky at top-0
 * (h-14, z-30), so the dock sticks just under it at top-14, above page content (z-20) and under
 * the header, the sidebar (z-40) and dialogs (z-50). Opaque, because the dark tint is not. The
 * region caps its height at 40vh and scrolls inside, so it never hides the whole page.
 */
export const BANNER_DOCK_CLASS = 'sticky top-14 z-20 bg-slate-50 dark:bg-slate-950 px-6 pt-4';
const COLLAPSE_KEY = 'mistudio.approvalsBanner.collapsedFor';

function readCollapsed(): string[] | null {
  try {
    const raw = sessionStorage.getItem(COLLAPSE_KEY);
    if (raw === null) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(String) : null;
  } catch {
    return null;
  }
}

function writeCollapsed(ids: string[] | null): void {
  try {
    if (ids === null) sessionStorage.removeItem(COLLAPSE_KEY);
    else sessionStorage.setItem(COLLAPSE_KEY, JSON.stringify(ids));
  } catch {
    // Storage blocked: the choice lasts until reload, which is fine.
  }
}

export function ApprovalsBanner() {
  const pending = useApprovalsStore((s) => s.pending);
  const error = useApprovalsStore((s) => s.error);
  const onApprovalsPage = useUIStore((s) => s.activePanel === 'approvals');
  const openApprovals = () => useUIStore.getState().setActivePanel('approvals');
  // Collapsed for exactly the requests waiting when the operator collapsed it.
  const [collapsedFor, setCollapsedFor] = useState<string[] | null>(() => readCollapsed());
  const shown = useMemo(() => pending.slice(0, BANNER_MAX_CARDS), [pending]);
  const context = useProbeRunContext(onApprovalsPage ? [] : shown);

  const hasNew = collapsedFor !== null && pending.some((r) => !collapsedFor.includes(r.id));
  useEffect(() => {
    if (hasNew) {
      // A new request expands it again.
      setCollapsedFor(null);
      writeCollapsed(null);
    }
  }, [hasNew]);

  if (onApprovalsPage) return null;
  if (pending.length === 0 && !error) return null;

  const collapsed = collapsedFor !== null && !hasNew;
  const n = pending.length;
  const toggle = () => {
    const next = collapsed ? null : pending.map((r) => r.id);
    setCollapsedFor(next);
    writeCollapsed(next);
  };
  const linkClass = 'text-xs font-medium text-amber-700 dark:text-amber-300 underline hover:no-underline';

  return (
    <div className={BANNER_DOCK_CLASS} data-testid="approvals-banner-dock">
      <div
        role="region"
        aria-label="Agent requests waiting for your approval"
        data-testid="approvals-banner"
        className="max-h-[40vh] overflow-y-auto bg-amber-50 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800 rounded-card p-4"
      >
        {error && (
          <p className="text-xs text-red-600 dark:text-red-400 mb-2" role="alert" data-testid="approvals-banner-error">
            Could not read the approval queue: {error}
          </p>
        )}
        {n > 0 && (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Bot className="w-4 h-4 text-amber-600 dark:text-amber-400" aria-hidden="true" />
              <h2 className="text-sm font-medium text-amber-800 dark:text-amber-300" aria-live="polite">
                {n} agent request{n === 1 ? '' : 's'} waiting for your approval
              </h2>
              <button type="button" onClick={openApprovals} className={linkClass}>
                Open Approvals
              </button>
              <button
                type="button"
                onClick={toggle}
                aria-expanded={!collapsed}
                className="ml-auto flex items-center gap-1 text-xs text-amber-800 dark:text-amber-300"
              >
                {collapsed ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
                {collapsed ? 'Expand' : 'Collapse'}
              </button>
            </div>
            {!collapsed && (
              <div className="mt-2 space-y-2">
                {/* ENH-006 D81: counts EVERY waiting code rule, not only the cards shown here. */}
                <ApproveAllCodeRules pending={pending} />
                <ApprovalCardList requests={shown} context={context} />
                {n > BANNER_MAX_CARDS && (
                  <p className="text-xs text-amber-800 dark:text-amber-300" data-testid="approvals-banner-more">
                    and {n - BANNER_MAX_CARDS} more waiting —{' '}
                    <button type="button" onClick={openApprovals} className={linkClass}>
                      Open Approvals
                    </button>
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
