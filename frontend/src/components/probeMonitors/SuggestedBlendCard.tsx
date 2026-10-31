/**
 * The suggested blend of a behaviour on a model (Feature 041 FR-21, FR-28): advice for an application, never
 * served — miLLM does not read it and no probe changes. Its history says who marked it, when and why.
 */
import { useEffect } from 'react';

import { useProbeFamiliesStore } from '../../stores/probeFamiliesStore';

export function SuggestedBlendCard({ behaviourKey, model }: { behaviourKey: string; model: string | null }) {
  const { suggested, history, loadSuggested } = useProbeFamiliesStore();

  useEffect(() => {
    if (model) void loadSuggested(behaviourKey, model);
  }, [behaviourKey, model, loadSuggested]);

  if (!model) return null;
  return (
    <section aria-label="Suggested blend" className="space-y-1 text-xs text-slate-300">
      <h5 className="font-semibold text-slate-200">
        Suggested blend{' '}
        <span className="rounded-full border border-sky-700 px-1 text-sky-300" data-testid="advisory-badge">
          Advisory — binds nothing; the application blends
        </span>
      </h5>
      {suggested ? (
        <p>
          {suggested.candidate} from report <span className="font-mono">{suggested.report_id}</span>
          {suggested.stale ? ' — stale: rerun the report and mark again' : ''}
        </p>
      ) : <p className="text-slate-500">None marked on this model.</p>}
      {history?.entries.length ? (
        <ul aria-label="Suggested blend history">
          {history.entries.map((h) => (
            <li key={h.id}>{h.created_at} — {h.actor}: {h.candidate}{h.reason ? ` (${h.reason})` : ''}</li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
