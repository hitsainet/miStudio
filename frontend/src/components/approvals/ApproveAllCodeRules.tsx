/**
 * "Approve all code rules (N)" (ENH-006 D81, operator 2026-10-29), on the banner and the Approvals page.
 *
 * Drawn only when TWO OR MORE code-rule requests wait; one waiting request is decided on its own card. The
 * button never approves directly: it opens a confirm that lists, for EVERY request, what the operator is
 * approving — the rule's name, the first 12 hex of its code hash, its functions, whether its own cases passed
 * when it was filed, and who asked. Confirming sends ONE call with exactly those ids
 * (`POST /mcp/approvals/approve-batch`, kind `code_rules`); the server approves each through the same path
 * as a single approve and refuses the whole batch if any id is no longer a waiting code-rule request.
 *
 * Code rules only: miStudio's queue has no settings-change request (D79), and the two kinds are never mixed.
 * There is no "reject all". Every field from a request is a text node.
 */
import { useState } from 'react';
import { CheckCheck } from 'lucide-react';
import { APIError } from '../../api/client';
import type { ApproveBatchResult } from '../../api/featureGroups';
import { useApprovalsStore } from '../../stores/approvalsStore';
import type { ApprovalRequest } from '../../types/featureGroups';
import { asText } from './ApprovalCard';
import { askedBy, CODE_RULE_TOOL } from './approvalLabels';

/** The batch button appears at this many waiting code-rule requests, not fewer. */
export const BATCH_MIN = 2;

/** Key-order-independent JSON, so `{a,b}` equals `{b,a}` as the server's dict comparison does. */
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value as Record<string, unknown>)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${canonical((value as Record<string, unknown>)[k])}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
}

export interface CodeRuleLine {
  id: string;
  name: string;
  sha12: string;
  functions: string;
  cases: string;
  casesPassed: boolean;
  askedBy: string;
}

/** What the confirm shows for one request. */
export function codeRuleLine(r: ApprovalRequest): CodeRuleLine {
  const p = r.payload ?? {};
  const doc = (p.document ?? {}) as { functions?: { name?: unknown }[]; derived_from?: { statement?: unknown } };
  const facts = (p.facts ?? {}) as {
    case_results?: { expect?: unknown; got?: unknown; same_twice?: unknown }[];
  };
  const results = Array.isArray(facts.case_results) ? facts.case_results : [];
  const passed = results.filter((c) => c.same_twice === true && canonical(c.got) === canonical(c.expect)).length;
  const statement = doc.derived_from?.statement;
  let cases: string;
  if (results.length === 0) cases = 'no case results recorded at filing';
  else if (passed === results.length) cases = `own cases passed at filing (${passed} of ${results.length})`;
  else cases = `own cases did NOT all pass at filing (${passed} of ${results.length})`;
  return {
    id: r.id,
    name: statement ? asText(statement) : 'unnamed code rule (its document carries no name)',
    sha12: asText(p.code_sha256 ?? '').slice(0, 12) || '—',
    functions: (doc.functions ?? []).map((f) => asText(f.name)).join(', ') || '—',
    cases,
    casesPassed: results.length > 0 && passed === results.length,
    askedBy: askedBy(r.requested_by),
  };
}

function failure(e: unknown): string {
  if (e instanceof APIError) return e.detail;
  if (e instanceof Error) return e.message;
  return String(e);
}

export function ApproveAllCodeRules({ pending }: { pending: ApprovalRequest[] }) {
  const codeRules = pending.filter((r) => r.tool_name === CODE_RULE_TOOL);
  const [confirming, setConfirming] = useState<ApprovalRequest[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState<ApproveBatchResult | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  const n = codeRules.length;
  if (n < BATCH_MIN && confirming === null && outcome === null && problem === null) return null;

  // The confirm lists a SNAPSHOT: exactly the requests shown are the ones sent, even if the poll changes the list.
  const open = () => {
    setOutcome(null);
    setProblem(null);
    setConfirming(codeRules);
  };
  const confirm = async () => {
    if (!confirming) return;
    setBusy(true);
    setProblem(null);
    try {
      const result = await useApprovalsStore.getState().approveBatch('code_rules', confirming.map((r) => r.id));
      setOutcome(result);
      setConfirming(null);
    } catch (e) {
      setProblem(failure(e));
    } finally {
      setBusy(false);
    }
  };

  const lines = (confirming ?? []).map(codeRuleLine);
  const failed = outcome?.results.filter((r) => r.status !== 'approved') ?? [];
  const approved = (outcome?.results.length ?? 0) - failed.length;

  return (
    <div className="space-y-2" data-testid="approve-all-code-rules">
      {n >= BATCH_MIN && confirming === null && (
        <button
          type="button"
          onClick={open}
          data-testid="approve-all-code-rules-button"
          className="flex items-center gap-1 px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-700 text-white rounded-control"
        >
          <CheckCheck className="w-3 h-3" aria-hidden="true" /> Approve all code rules ({n})
        </button>
      )}
      {confirming !== null && (
        <div
          role="dialog"
          aria-modal="false"
          aria-labelledby="approve-all-code-rules-title"
          data-testid="approve-all-code-rules-confirm"
          className="border border-amber-300 dark:border-amber-800 bg-white dark:bg-slate-900 rounded-control p-3 space-y-2"
        >
          <h3 id="approve-all-code-rules-title" className="text-sm font-medium text-slate-900 dark:text-slate-100">
            Approve these {lines.length} code rules?
          </h3>
          <p className="text-xs text-slate-600 dark:text-slate-400">
            Each lets this code run in miStudio&apos;s isolated runner, recorded against its code hash until you
            revoke it. Read each one&apos;s full source on its card before approving.
          </p>
          <ul className="space-y-1 text-xs text-slate-700 dark:text-slate-300" data-testid="approve-all-code-rules-list">
            {lines.map((l) => (
              <li key={l.id} data-testid="approve-all-code-rules-item" className="border-b border-slate-100 dark:border-slate-800 pb-1">
                <span className="font-medium">{l.name}</span> · code <span className="font-mono">{l.sha12}</span> ·
                functions: {l.functions} ·{' '}
                <span className={l.casesPassed ? '' : 'text-red-600 dark:text-red-400 font-medium'}>{l.cases}</span> ·{' '}
                {l.askedBy}
              </li>
            ))}
          </ul>
          {problem && (
            <p className="text-xs text-red-600 dark:text-red-400" role="alert" data-testid="approve-all-code-rules-problem">
              Nothing was approved: {problem}
            </p>
          )}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => void confirm()}
              disabled={busy}
              data-testid="approve-all-code-rules-confirm-button"
              className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white rounded-control"
            >
              {busy ? 'Approving…' : `Approve these ${lines.length} code rules`}
            </button>
            <button
              type="button"
              onClick={() => {
                setConfirming(null);
                setProblem(null);
              }}
              disabled={busy}
              className="px-3 py-1.5 text-xs bg-slate-200 dark:bg-slate-700 text-slate-800 dark:text-slate-200 rounded-control"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
      {outcome && (
        <div role="status" className="text-xs text-slate-700 dark:text-slate-300" data-testid="approve-all-code-rules-outcome">
          <p>
            Approved {approved} of {outcome.results.length} code rules together (batch{' '}
            <span className="font-mono">{outcome.batch_id}</span>).
          </p>
          {failed.length > 0 && (
            <ul className="text-red-600 dark:text-red-400">
              {failed.map((f) => (
                <li key={f.id}>
                  Not approved: <span className="font-mono">{f.id}</span> — {asText(f.error ?? 'failed')}
                </li>
              ))}
            </ul>
          )}
          <button type="button" onClick={() => setOutcome(null)} className="underline">
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}
