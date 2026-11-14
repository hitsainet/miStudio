/**
 * History (ENH-008 §2.2): decided requests, the most recent decision first, in the shared columns —
 * Outcome, Action, Asked by, Asked, Decided by, Decided, Reason, Result, Id. On a narrow screen each
 * row becomes a stacked card in the same order. Every value is a text node.
 */
import type { ApprovalRequest } from '../../types/featureGroups';
import { formatLocal } from '../../utils/formatLocal';
import { actionLabel, askedBy, decidedBy, outcomeLabel, resultText } from './approvalLabels';

const COLUMNS = ['Outcome', 'Action', 'Asked by', 'Asked', 'Decided by', 'Decided', 'Reason', 'Result', 'Id'] as const;

function cells(r: ApprovalRequest): { key: (typeof COLUMNS)[number]; text: string; title?: string; mono?: boolean }[] {
  const asked = formatLocal(r.created_at);
  // The decision time; an approval later used keeps its approval time.
  const decided = formatLocal(r.resolved_at);
  return [
    { key: 'Outcome', text: outcomeLabel(r) },
    { key: 'Action', text: actionLabel(r.tool_name) },
    { key: 'Asked by', text: askedBy(r.requested_by).replace(/^asked by /, '') },
    { key: 'Asked', text: asked.text, title: asked.title },
    { key: 'Decided by', text: decidedBy(r) },
    { key: 'Decided', text: decided.text, title: decided.title },
    { key: 'Reason', text: r.reason ?? '—' },
    { key: 'Result', text: resultText(r) },
    { key: 'Id', text: r.id, mono: true },
  ];
}

export function ApprovalHistory({ rows }: { rows: ApprovalRequest[] }) {
  return (
    <>
      {/* Wide screens: a table. */}
      <div className="hidden md:block overflow-x-auto">
        <table className="w-full text-xs text-left" data-testid="approvals-history-table">
          <thead className="text-slate-500 border-b border-slate-200 dark:border-slate-700">
            <tr>
              {COLUMNS.map((c) => (
                <th key={c} scope="col" className="py-1.5 pr-3 font-medium">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} data-testid="approval-history-row" className="border-b border-slate-100 dark:border-slate-800 align-top">
                {cells(r).map((c) => (
                  <td
                    key={c.key}
                    title={c.title || undefined}
                    className={`py-1.5 pr-3 break-words whitespace-pre-wrap ${c.mono ? 'font-mono text-[11px]' : ''}`}
                  >
                    {c.text}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Narrow screens: stacked cards, same order. */}
      <ul className="md:hidden space-y-2">
        {rows.map((r) => (
          <li key={r.id} className="border border-slate-200 dark:border-slate-700 rounded-control p-2">
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
              {cells(r).map((c) => (
                <div key={c.key} className="contents">
                  <dt className="text-slate-500">{c.key}</dt>
                  <dd title={c.title || undefined} className={`break-words whitespace-pre-wrap ${c.mono ? 'font-mono' : ''}`}>
                    {c.text}
                  </dd>
                </div>
              ))}
            </dl>
          </li>
        ))}
      </ul>
    </>
  );
}
