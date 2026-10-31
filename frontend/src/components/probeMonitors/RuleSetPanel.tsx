/**
 * A behaviour's rule set on the behaviour screen (Feature 040 S3, FR-32, FR-43): validate it with miStudio's own
 * engine and try it on one text, with the spans of what fired. Every word about a case, a signal or a verdict is
 * the server's; a refusal shows its message and the `next` call that unblocks it (FR-40). Nothing typed here is
 * stored or logged by the server. S4: operator code rules show their approval state, with the operator's Revoke and
 * Re-approve.
 */
import { useState } from 'react';

import { useBehaviourDetectorsStore } from '../../stores/behaviourDetectorsStore';
import type { CodeRuleState } from '../../types/probeMonitor';
import { RefusalNotice } from './DetectorReports';

function Highlighted({ text, spans }: { text: string; spans: { start: number; end: number }[] }) {
  // Spans are code points; Array.from splits by code point, so multi-byte text lines up.
  const chars = Array.from(text);
  const marked = new Array<boolean>(chars.length).fill(false);
  for (const s of spans) for (let i = s.start; i < s.end && i < chars.length; i += 1) marked[i] = true;
  return (
    <p className="whitespace-pre-wrap font-mono" data-testid="rule-spans">
      {chars.map((c, i) => (marked[i] ? <mark key={i} className="bg-amber-700 text-white">{c}</mark> : <span key={i}>{c}</span>))}
    </p>
  );
}

/**
 * Feature 040 S4 (FTDD Appendix C.6): each operator code rule the set names, with its approval state here, and the
 * operator's Revoke (an approved hash) or Re-approve (a revoked one). The reason is required; the server refuses an
 * agent's decision. The FIRST approval is the approvals banner's (an agent's request, with the source to read).
 */
function CodeRules({ behaviourKey, codeRules }: { behaviourKey: string; codeRules: Record<string, CodeRuleState> }) {
  const { busy, revokeCodeRule, reapproveCodeRule } = useBehaviourDetectorsStore();
  const [reason, setReason] = useState('');
  const entries = Object.entries(codeRules);
  if (entries.length === 0) return null;
  return (
    <div data-testid="code-rules" className="space-y-1">
      <p>Operator code rules</p>
      <label className="block">
        <span className="sr-only">Reason for the decision</span>
        <input aria-label="Reason for the decision" value={reason} onChange={(e) => setReason(e.target.value)}
          className="w-full bg-slate-900" placeholder="why (required for a revoke or a re-approve)" />
      </label>
      <ul>
        {entries.map(([sha, state]) => (
          <li key={sha} data-testid="code-rule" className="flex items-center gap-2">
            <span className="font-mono">{sha.slice(0, 12)}…</span>
            <span>{state.state}</span>
            {state.state === 'approved' ? (
              <button type="button" disabled={busy || !reason.trim()} className="rounded-control border border-red-700 px-2"
                onClick={() => void revokeCodeRule(sha, reason, { behaviour_key: behaviourKey })}>
                Revoke
              </button>
            ) : null}
            {state.state === 'revoked' ? (
              <button type="button" disabled={busy || !reason.trim()} className="rounded-control border border-emerald-700 px-2"
                onClick={() => void reapproveCodeRule(sha, reason, { behaviour_key: behaviourKey })}>
                Re-approve
              </button>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function RuleSetPanel({ behaviourKey }: { behaviourKey: string }) {
  const { ruleValidation, ruleVerdict, refusal, busy, validateRuleSet, evaluateRuleSignals } = useBehaviourDetectorsStore();
  const [text, setText] = useState('');
  const [tried, setTried] = useState('');
  return (
    <section aria-label="Rule set" className="space-y-1 text-xs text-slate-300">
      <h4 className="text-sm font-semibold text-slate-200">Rule set</h4>
      <button type="button" disabled={busy} className="rounded-control border border-slate-600 px-2 py-1"
        onClick={() => void validateRuleSet({ behaviour_key: behaviourKey })}>
        Validate rule set
      </button>
      <RefusalNotice refusal={refusal} />
      {ruleValidation ? (
        <div data-testid="rule-validation">
          {ruleValidation.valid ? (
            <p>
              Valid: {ruleValidation.cases?.passed ?? 0} case(s) pass; match {ruleValidation.match_expressions?.join(', ') || '—'};
              {ruleValidation.has_vetoes ? ' has vetoes' : ' no veto signals'}
              {ruleValidation.pending_expressions?.length
                ? `; pending operator code: ${ruleValidation.pending_expressions.join(', ')}` : ''}
              {ruleValidation.runner ? ` · operator-code runner ${ruleValidation.runner}` : ''}
            </p>
          ) : (
            <div role="alert" className="text-amber-300">
              <p>Not valid: {(ruleValidation.code ?? '').replace(/_/g, ' ')} — {ruleValidation.message}</p>
              <ul>
                {(ruleValidation.failing_cases ?? []).map((f) => (
                  <li key={f.case_index} data-testid="failing-case">case {f.case_index} ({f.case_id}): {f.where}</li>
                ))}
              </ul>
            </div>
          )}
          {ruleValidation.code_rules ? (
            <CodeRules behaviourKey={behaviourKey} codeRules={ruleValidation.code_rules} />
          ) : null}
        </div>
      ) : null}
      <label className="block">
        <span className="sr-only">Text to try</span>
        <textarea aria-label="Text to try" value={text} onChange={(e) => setText(e.target.value)} rows={3}
          className="w-full bg-slate-900" placeholder="a text to try the rules on (not stored)" />
      </label>
      <button type="button" disabled={busy || !text} className="rounded-control bg-sky-800 px-2 py-1 text-white"
        onClick={() => { setTried(text); void evaluateRuleSignals({ behaviour_key: behaviourKey, text }); }}>
        Try on this text
      </button>
      {ruleVerdict ? (
        <div data-testid="rule-verdict">
          <p>
            {ruleVerdict.unknown
              ? `Unknown: ${(ruleVerdict.reason ?? '').replace(/_/g, ' ')}`
              : `Match: ${ruleVerdict.rule_match ? 'yes' : 'no'} · Veto: ${ruleVerdict.rule_veto ? 'yes' : 'no'}`}
          </p>
          <ul>
            {ruleVerdict.fired.map((f) => (
              <li key={f.signal_id}>
                {f.signal_id} ({f.kind}){f.pending ? ' — pending operator code' : ''}
                {f.count != null ? ` — count ${f.count}, no positions (operator code)` : ''}
              </li>
            ))}
          </ul>
          <Highlighted text={tried} spans={ruleVerdict.fired.flatMap((f) => (f.spans ?? []).filter((s) => s.message_index === 0))} />
        </div>
      ) : null}
    </section>
  );
}
