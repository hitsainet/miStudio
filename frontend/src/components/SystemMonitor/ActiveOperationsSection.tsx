/**
 * Active Operations Section
 *
 * Displays currently running or queued background operations.
 * Part of the System Monitor for operation visibility.
 */

import { useEffect } from 'react';
import { Clock, RefreshCw } from 'lucide-react';
import { useTaskQueueStore } from '../../stores/taskQueueStore';
import { formatDuration } from '../../utils/formatters';

const TASK_TYPE_LABELS: Record<string, string> = {
  download: 'Download',
  training: 'Training',
  extraction: 'Extraction',
  tokenization: 'Tokenization',
  labeling: 'Labeling',
  neuronpedia_push: 'Neuronpedia Push',
  // J-space work rendered as the raw enum string — `jlens_fit` — because these
  // were never added. A fit is the longest-running job in the product and it
  // named itself worse than anything else in this list.
  jlens_fit: 'J-Lens fit',
  jlens_band_report: 'J-Lens band report',
  jlens_intervention: 'J-Lens intervention',
  jlens_readout: 'J-Lens readout',
  jlens_probe: 'J-Lens probe',
};

const ENTITY_TYPE_LABELS: Record<string, string> = {
  model: 'Model',
  dataset: 'Dataset',
  training: 'Training',
  extraction: 'Extraction',
  labeling: 'Labeling',
  neuronpedia: 'Neuronpedia',
};

/**
 * "Elapsed 2h 51m" for work that has STARTED, "Queued 2h 58m" for work that has
 * not, and a heartbeat age when one is known.
 *
 * THE TWO CLOCKS ARE DIFFERENT AND MUST NOT BE CONFLATED. `created_at` is when
 * the job was enqueued; `started_at` is when a worker picked it up. An LFM2 fit
 * waited three hours behind gemma, so measuring elapsed from `created_at` would
 * have reported a four-hour fit after one hour of work.
 *
 * Falls back to the raw creation timestamp when nothing better is known, rather
 * than inventing a duration from a missing field.
 */
export function elapsedLabel(task: {
  status: string;
  started_at?: string | null;
  created_at?: string | null;
  entity_info?: { seconds_since_heartbeat?: number } | null;
}): string {
  const beat = task.entity_info?.seconds_since_heartbeat;
  const beatText =
    typeof beat === 'number' ? ` · beat ${formatDuration(beat)} ago` : '';

  if (task.started_at) {
    const secs = (Date.now() - Date.parse(task.started_at)) / 1000;
    if (Number.isFinite(secs) && secs >= 0) {
      return `Elapsed ${formatDuration(secs)}${beatText}`;
    }
  }
  if (task.created_at) {
    const secs = (Date.now() - Date.parse(task.created_at)) / 1000;
    if (Number.isFinite(secs) && secs >= 0) {
      // NOT "elapsed": this job has not started, and calling its wait "elapsed"
      // is the conflation above.
      return `Queued ${formatDuration(secs)}`;
    }
    return `Created: ${new Date(task.created_at).toLocaleString()}`;
  }
  return '';
}

export function ActiveOperationsSection() {
  const { activeTasks, activeLoading, activeError, fetchActiveTasks } = useTaskQueueStore();

  // Fetch active tasks on mount and every 5 seconds (paused while tab hidden)
  useEffect(() => {
    fetchActiveTasks();
    const interval = setInterval(() => {
      if (document.visibilityState === 'visible') {
        fetchActiveTasks();
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [fetchActiveTasks]);

  const getTaskTypeLabel = (type: string): string => TASK_TYPE_LABELS[type] || type;
  const getEntityTypeLabel = (type: string): string => ENTITY_TYPE_LABELS[type] || type;

  const getStatusBadge = (status: string) => {
    // ORPHANED FIRST. This badge was binary — anything that was not 'running'
    // fell through to amber "Queued" — so a job whose worker had died rendered
    // as one merely waiting its turn. That is the exact confusion the backend
    // janitor removed, still on screen.
    if (status === 'orphaned') {
      return (
        <span className="px-2 py-0.5 bg-red-500/20 border border-red-500/30 rounded text-xs font-medium text-red-300">
          Stopped reporting
        </span>
      );
    }
    if (status === 'running') {
      return (
        <span className="px-2 py-0.5 bg-emerald-500/20 border border-emerald-500/30 rounded text-xs font-medium text-emerald-300 flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
          Running
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 bg-amber-500/20 border border-amber-500/30 rounded text-xs font-medium text-amber-300">
        Queued
      </span>
    );
  };

  if (activeTasks.length === 0 && !activeLoading) {
    return (
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <Clock className="w-5 h-5 text-emerald-400" />
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Active Operations</h2>
        </div>
        <p className="text-slate-500 text-center py-8">No active operations</p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Clock className="w-5 h-5 text-emerald-400" />
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Active Operations</h2>
          <span className="px-2 py-1 bg-emerald-500/10 border border-emerald-500/30 rounded text-xs text-emerald-300">
            {activeTasks.length}
          </span>
        </div>
        <button
          onClick={() => fetchActiveTasks()}
          disabled={activeLoading}
          className="text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors disabled:opacity-50"
          title="Refresh"
        >
          <RefreshCw className={`w-4 h-4 ${activeLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {activeError && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm">
          {activeError}
        </div>
      )}

      <div className="space-y-3">
        {activeTasks.map((task) => (
          <div
            key={task.id}
            className="bg-slate-100 dark:bg-slate-800/50 border border-slate-300 dark:border-slate-700 rounded-lg p-4"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  {getStatusBadge(task.status)}
                  <span className="px-2 py-0.5 bg-emerald-500/20 border border-emerald-500/30 rounded text-xs font-medium text-emerald-300">
                    {getTaskTypeLabel(task.task_type)}
                  </span>
                  <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-700 rounded text-xs text-slate-700 dark:text-slate-300">
                    {getEntityTypeLabel(task.entity_type)}
                  </span>
                </div>
                <div className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                  {task.entity_info?.name || task.entity_id}
                </div>
                {task.entity_info?.repo_id && (
                  <div className="text-xs text-slate-500 font-mono truncate mt-0.5">
                    {task.entity_info.repo_id}
                  </div>
                )}
                {task.entity_info?.details && (
                  <div className="text-xs text-slate-500 truncate mt-0.5">
                    {task.entity_info.details}
                  </div>
                )}
                <div className="text-xs text-slate-500 mt-1">
                  {elapsedLabel(task)}
                </div>
                {task.retry_count > 0 && (
                  <div className="text-xs text-amber-400 mt-1">
                    Retry attempt: {task.retry_count + 1}
                  </div>
                )}
              </div>

              {/* Progress indicator */}
              {task.progress !== null && task.progress !== undefined && (
                <div className="text-right">
                  <div className="text-lg font-semibold text-emerald-400">
                    {task.progress.toFixed(1)}%
                  </div>
                  <div className="text-xs text-slate-500">Progress</div>
                </div>
              )}
            </div>

            {/* Progress bar */}
            {task.progress !== null && task.progress !== undefined && (
              <div className="mt-3">
                <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-emerald-500 h-full transition-all duration-300 ease-out"
                    style={{ width: `${Math.min(100, Math.max(0, task.progress))}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
