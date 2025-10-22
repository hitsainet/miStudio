/**
 * Failed Operations Section
 *
 * Displays failed background operations with retry/delete controls.
 * Part of the System Monitor for manual operation management.
 */

import { useEffect, useState } from 'react';
import { AlertTriangle, RefreshCw, Trash2, ChevronDown, ChevronUp } from 'lucide-react';
import { useTaskQueueStore } from '../../stores/taskQueueStore';
import { TaskQueueEntry } from '../../types/taskQueue';
import { RetryConfirmDialog } from './RetryConfirmDialog';

export function FailedOperationsSection() {
  const { failedTasks, loading, error, fetchFailedTasks, deleteTask } = useTaskQueueStore();
  const [expandedTasks, setExpandedTasks] = useState<Set<string>>(new Set());
  const [retryingTask, setRetryingTask] = useState<TaskQueueEntry | null>(null);

  // Fetch failed tasks on mount and every 30 seconds
  useEffect(() => {
    fetchFailedTasks();
    const interval = setInterval(fetchFailedTasks, 30000);
    return () => clearInterval(interval);
  }, [fetchFailedTasks]);

  const toggleExpanded = (taskId: string) => {
    setExpandedTasks((prev) => {
      const next = new Set(prev);
      if (next.has(taskId)) {
        next.delete(taskId);
      } else {
        next.add(taskId);
      }
      return next;
    });
  };

  const handleRetryClick = (task: TaskQueueEntry) => {
    setRetryingTask(task);
  };

  const handleDeleteClick = async (taskId: string) => {
    if (!confirm('Are you sure you want to remove this failed operation from the list?')) {
      return;
    }

    try {
      await deleteTask(taskId);
    } catch (error) {
      console.error('Failed to delete task:', error);
      alert('Failed to delete task. Please try again.');
    }
  };

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

  if (failedTasks.length === 0 && !loading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <AlertTriangle className="w-5 h-5 text-emerald-400" />
          <h2 className="text-lg font-semibold text-slate-100">Failed Operations</h2>
        </div>
        <p className="text-slate-500 text-center py-8">No failed operations</p>
      </div>
    );
  }

  return (
    <>
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <h2 className="text-lg font-semibold text-slate-100">Failed Operations</h2>
            <span className="px-2 py-1 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-300">
              {failedTasks.length}
            </span>
          </div>
          <button
            onClick={() => fetchFailedTasks()}
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
          {failedTasks.map((task) => {
            const isExpanded = expandedTasks.has(task.id);
            return (
              <div
                key={task.id}
                className="bg-slate-800/50 border border-slate-700 rounded-lg overflow-hidden"
              >
                {/* Task Header */}
                <div className="p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="px-2 py-0.5 bg-red-500/20 border border-red-500/30 rounded text-xs font-medium text-red-300">
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
                        Failed: {new Date(task.updated_at || task.created_at || '').toLocaleString()}
                      </div>
                      {task.retry_count > 0 && (
                        <div className="text-xs text-amber-400 mt-1">
                          Retry attempts: {task.retry_count}
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleRetryClick(task)}
                        className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 rounded text-xs font-medium text-white transition-colors flex items-center gap-1.5"
                        title="Retry operation"
                      >
                        <RefreshCw className="w-3.5 h-3.5" />
                        Retry
                      </button>
                      <button
                        onClick={() => handleDeleteClick(task.id)}
                        className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors"
                        title="Remove from list"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => toggleExpanded(task.id)}
                        className="p-1.5 text-slate-400 hover:text-slate-300 transition-colors"
                        title={isExpanded ? 'Collapse' : 'Expand'}
                      >
                        {isExpanded ? (
                          <ChevronUp className="w-4 h-4" />
                        ) : (
                          <ChevronDown className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="px-4 pb-4 pt-2 border-t border-slate-700">
                    <div className="text-xs text-slate-400 font-semibold mb-2">Error Details:</div>
                    <div className="text-sm text-red-300 bg-red-950/30 border border-red-900/50 rounded p-3 font-mono whitespace-pre-wrap break-words">
                      {task.error_message || 'No error message available'}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Retry Confirmation Dialog */}
      {retryingTask && (
        <RetryConfirmDialog
          task={retryingTask}
          onClose={() => setRetryingTask(null)}
          onConfirm={() => {
            setRetryingTask(null);
            // Refresh failed tasks after a short delay to allow backend to process
            setTimeout(() => fetchFailedTasks(), 1000);
          }}
        />
      )}
    </>
  );
}
