/**
 * Active Operations Section
 *
 * Displays currently running or queued background operations.
 * Part of the System Monitor for operation visibility.
 */

import { useEffect } from 'react';
import { Clock, RefreshCw } from 'lucide-react';
import { useTaskQueueStore } from '../../stores/taskQueueStore';

export function ActiveOperationsSection() {
  const { activeTasks, loading, error, fetchActiveTasks } = useTaskQueueStore();

  // Fetch active tasks on mount and every 5 seconds
  useEffect(() => {
    fetchActiveTasks();
    const interval = setInterval(fetchActiveTasks, 5000);
    return () => clearInterval(interval);
  }, [fetchActiveTasks]);

  const getTaskTypeLabel = (type: string): string => {
    const labels: Record<string, string> = {
      download: 'Download',
      training: 'Training',
      extraction: 'Extraction',
      tokenization: 'Tokenization',
    };
    return labels[type] || type;
  };

  const getEntityTypeLabel = (type: string): string => {
    const labels: Record<string, string> = {
      model: 'Model',
      dataset: 'Dataset',
      training: 'Training',
      extraction: 'Extraction',
    };
    return labels[type] || type;
  };

  const getStatusBadge = (status: string) => {
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

  if (activeTasks.length === 0 && !loading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <Clock className="w-5 h-5 text-emerald-400" />
          <h2 className="text-lg font-semibold text-slate-100">Active Operations</h2>
        </div>
        <p className="text-slate-500 text-center py-8">No active operations</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Clock className="w-5 h-5 text-emerald-400" />
          <h2 className="text-lg font-semibold text-slate-100">Active Operations</h2>
          <span className="px-2 py-1 bg-emerald-500/10 border border-emerald-500/30 rounded text-xs text-emerald-300">
            {activeTasks.length}
          </span>
        </div>
        <button
          onClick={() => fetchActiveTasks()}
          disabled={loading}
          className="text-slate-400 hover:text-slate-300 transition-colors disabled:opacity-50"
          title="Refresh"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm">
          {error}
        </div>
      )}

      <div className="space-y-3">
        {activeTasks.map((task) => (
          <div
            key={task.id}
            className="bg-slate-800/50 border border-slate-700 rounded-lg p-4"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  {getStatusBadge(task.status)}
                  <span className="px-2 py-0.5 bg-emerald-500/20 border border-emerald-500/30 rounded text-xs font-medium text-emerald-300">
                    {getTaskTypeLabel(task.task_type)}
                  </span>
                  <span className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                    {getEntityTypeLabel(task.entity_type)}
                  </span>
                </div>
                <div className="text-sm font-medium text-slate-100 truncate">
                  {task.entity_info?.name || task.entity_id}
                </div>
                {task.entity_info?.repo_id && (
                  <div className="text-xs text-slate-500 font-mono truncate mt-0.5">
                    {task.entity_info.repo_id}
                  </div>
                )}
                <div className="text-xs text-slate-500 mt-1">
                  Started: {new Date(task.started_at || task.created_at || '').toLocaleString()}
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
                <div className="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden">
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
