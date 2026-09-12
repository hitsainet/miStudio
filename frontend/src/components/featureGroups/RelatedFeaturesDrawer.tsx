/**
 * Slide-over showing features related to a seed feature (shared token /
 * context / correlation link types).
 */

import { X } from 'lucide-react';
import { useFeatureGroupsStore } from '../../stores/featureGroupsStore';

const LINK_BADGES: Record<string, { label: string; cls: string }> = {
  shared_token: { label: 'token', cls: 'bg-emerald-900/60 text-emerald-300' },
  context: { label: 'context', cls: 'bg-sky-900/60 text-sky-300' },
  correlation: { label: 'corr', cls: 'bg-purple-900/60 text-purple-300' },
};

interface RelatedFeaturesDrawerProps {
  onOpenFeature: (featureId: string) => void;
}

export function RelatedFeaturesDrawer({ onOpenFeature }: RelatedFeaturesDrawerProps) {
  const { relatedFor, related, clearRelated } = useFeatureGroupsStore();

  if (!relatedFor) return null;

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-white dark:bg-slate-900 border-l border-slate-300 dark:border-slate-700 shadow-2xl z-40 flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-800">
        <div>
          <h3 className="text-sm font-medium text-slate-800 dark:text-slate-200">Related features</h3>
          <p className="text-xs text-slate-500 font-mono">{relatedFor}</p>
        </div>
        <button onClick={clearRelated} className="text-slate-500 hover:text-slate-700 dark:hover:text-slate-300">
          <X className="w-4 h-4" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        {!related && <p className="text-xs text-slate-500 py-4 text-center">Loading…</p>}
        {related && related.related.length === 0 && (
          <p className="text-xs text-slate-500 py-4 text-center">No related features found.</p>
        )}
        {related?.related.map((r) => (
          <button
            key={r.feature_id}
            onClick={() => onOpenFeature(r.feature_id)}
            className="w-full text-left bg-slate-100 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-md px-3 py-2"
          >
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs text-slate-500">#{r.neuron_index}</span>
              <span className="text-sm text-slate-800 dark:text-slate-200 truncate flex-1">{r.name}</span>
              <span className="text-xs text-slate-500">{r.score.toFixed(2)}</span>
            </div>
            <div className="flex gap-1 mt-1">
              {r.link_types.map((lt) => (
                <span key={lt} className={`text-[10px] px-1.5 py-0.5 rounded ${LINK_BADGES[lt]?.cls}`}>
                  {LINK_BADGES[lt]?.label ?? lt}
                </span>
              ))}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
