/**
 * Grouping-index state banner: no index / computing (progress) / failed / stale hint.
 */

import { AlertCircle, Loader2, RefreshCw, Sparkles } from 'lucide-react';
import { useFeatureGroupsStore } from '../../stores/featureGroupsStore';

export function ComputeIndexBanner() {
  const { status, computeProgress, computeIndex, error } = useFeatureGroupsStore();

  const isComputing =
    computeProgress !== null || status?.status === 'computing' || status?.status === 'pending';

  if (isComputing) {
    const progress = computeProgress?.progress ?? 0;
    const stage = computeProgress?.stage ?? 'starting';
    return (
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg p-4 mb-4">
        <div className="flex items-center gap-3 mb-2">
          <Loader2 className="w-4 h-4 text-emerald-400 animate-spin" />
          <span className="text-sm text-slate-800 dark:text-slate-200">Building cluster index…</span>
          <span className="text-xs text-slate-500">{stage}</span>
        </div>
        <div className="w-full bg-white dark:bg-slate-800 rounded-full h-2">
          <div
            className="bg-emerald-500 h-2 rounded-full transition-all duration-500"
            style={{ width: `${Math.max(progress, 2)}%` }}
          />
        </div>
      </div>
    );
  }

  if (status?.status === 'failed') {
    return (
      <div className="bg-red-950/40 border border-red-800 rounded-lg p-4 mb-4 flex items-center gap-3">
        <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
        <div className="flex-1">
          <p className="text-sm text-red-300">Cluster index failed</p>
          <p className="text-xs text-red-400/70">{status.error_message}</p>
        </div>
        <button
          onClick={() => void computeIndex({}, true)}
          className="px-3 py-1.5 text-xs bg-red-900/50 hover:bg-red-900 text-red-200 rounded-md flex items-center gap-1.5"
        >
          <RefreshCw className="w-3 h-3" /> Retry
        </button>
      </div>
    );
  }

  if (!status || status.status === 'none') {
    return (
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg p-6 mb-4 text-center">
        <Sparkles className="w-6 h-6 text-emerald-400 mx-auto mb-2" />
        <p className="text-sm text-slate-800 dark:text-slate-200 mb-1">No cluster index yet</p>
        <p className="text-xs text-slate-500 mb-4 max-w-md mx-auto">
          Build a token→feature index to browse features that fire on the same top activating
          token with similar context. CPU-only background job; a few minutes for large extractions.
        </p>
        <button
          onClick={() => void computeIndex()}
          className="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded-md"
        >
          Compute Index
        </button>
        {error && <p className="text-xs text-red-400 mt-3">{error}</p>}
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between text-xs text-slate-500 mb-3">
      <span>
        Index: {status.feature_count?.toLocaleString()} features · {status.group_count?.toLocaleString()} clusters
        {status.computed_at && ` · computed ${new Date(status.computed_at).toLocaleString()}`}
      </span>
      <button
        onClick={() => void computeIndex({}, true)}
        className="flex items-center gap-1 text-slate-600 dark:text-slate-400 hover:text-emerald-400"
        title="Recompute the index"
      >
        <RefreshCw className="w-3 h-3" /> Recompute
      </button>
    </div>
  );
}
