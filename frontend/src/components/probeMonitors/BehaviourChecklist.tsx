/**
 * The guided checklist (Feature 037 FR-12, D26): eight steps computed by the server from real records.
 * Nothing here ticks a step. "Unknown" (miLLM could not say) is shown as unknown, never as not done.
 */
import type { BehaviourChecklist as Checklist } from '../../types/behaviour';

const MARK = { done: '✓', not_done: '○', unknown: '?', stale: '!', partly_done: '◐' } as const;
const WORD = { done: 'done', not_done: 'not done', unknown: 'unknown', stale: 'stale', partly_done: 'partly done' } as const;
/** Feature 039 (FR-19, D-T7): a definition that exists but states an old decision is STALE — said
 * in words as well as colour, and with the tool that rebuilds it. */
const TONE = { done: 'text-slate-400', not_done: 'text-slate-400', unknown: 'text-slate-400', stale: 'text-amber-300', partly_done: 'text-amber-200' } as const;
/* Feature 041 FR-5: `partly_done` — some family members are done; the reason names the rest. */

export function BehaviourChecklist({ checklist, onOpenTab }: { checklist: Checklist; onOpenTab?: (tab: string) => void }) {
  return (
    <ol className="space-y-1" aria-label="Checklist">
      {checklist.steps.map((step) => (
        <li key={step.id} className="flex items-start gap-2 text-sm" data-state={step.state}>
          <span aria-hidden className="w-4 text-center">{MARK[step.state]}</span>
          <div className="flex-1">
            <span className="text-slate-100">{step.label}</span>{' '}
            <span className={`text-xs ${TONE[step.state]}`}>({WORD[step.state]}) — {step.reason}</span>
            {step.next && step.next.length ? (
              <span className="ml-1 text-xs text-slate-500" data-testid={`checklist-next-${step.id}`}>
                next: <code className="font-mono">{step.next.map((call) => call.tool).join(', ')}</code>
              </span>
            ) : null}
          </div>
          {step.link.tab && onOpenTab ? (
            <button type="button" className="text-xs text-sky-400 underline" onClick={() => onOpenTab(step.link.tab as string)}>
              Open
            </button>
          ) : null}
          {step.link.app === 'millm' ? <span className="text-xs text-slate-500">in miLLM</span> : null}
        </li>
      ))}
    </ol>
  );
}
