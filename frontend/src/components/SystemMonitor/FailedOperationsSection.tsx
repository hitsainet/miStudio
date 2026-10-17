/**
 * Failed Operations Section
 *
 * Displays failed background operations with retry/delete controls.
 * Part of the System Monitor for manual operation management.
 *
 * Rows with can_retry=false are federated from other job tables (trainings,
 * extractions, labeling, pushes). They cannot be RETRIED here — that belongs to
 * their own panel — but they can be DISMISSED, which clears them from this list
 * without touching the job record.
 *
 * Until 2026-07-26 they could only be dismissed in theory: the UI said "manage
 * in its panel" while, for Neuronpedia pushes, no such control existed anywhere
 * (the only DELETE in that API targets neuronpedia_exports, a different table).
 * Four failures from March were unclearable.
 */

import { useEffect, useState } from 'react';
import { AlertTriangle, RefreshCw, Trash2, ChevronDown, ChevronUp } from 'lucide-react';
import { useTaskQueueStore } from '../../stores/taskQueueStore';
import { TaskQueueEntry } from '../../types/taskQueue';
import { RetryConfirmDialog } from './RetryConfirmDialog';

const TASK_TYPE_LABELS: Record<string, string> = {
  download: 'Download',
  training: 'Training',
  extraction: 'Extraction',
  tokenization: 'Tokenization',
  labeling: 'Labeling',
  neuronpedia_push: 'Neuronpedia Push',
};

const ENTITY_TYPE_LABELS: Record<string, string> = {
  model: 'Model',
  dataset: 'Dataset',
  training: 'Training',
  extraction: 'Extraction',
  labeling: 'Labeling',
  neuronpedia: 'Neuronpedia',
};

export function FailedOperationsSection() {
  const {
    failedTasks,
    failedLoading,
    failedError,
    fetchFailedTasks,
    deleteTask,
    dismissFailedTask,
    dismissAllFailedTasks,
  } = useTaskQueueStore();
  const [expandedTasks, setExpandedTasks] = useState<Set<string>>(new Set());
  const [retryingTask, setRetryingTask] = useState<TaskQueueEntry | null>(null);
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const [confirmingClearAll, setConfirmingClearAll] = useState(false);

  // Fetch failed tasks on mount and every 30 seconds (paused while tab hidden)
  useEffect(() => {
    fetchFailedTasks();
    const interval = setInterval(() => {
      if (document.visibilityState === 'visible') {
        fetchFailedTasks();
      }
    }, 30000);
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

  const handleDismissClick = async (task: TaskQueueEntry) => {
    // Same two-click inline confirmation as delete. Dismissing hides a real
    // failure, so it should not be a single stray click — but it is reversible
    // (DELETE on the same route restores it), hence no modal.
    if (confirmingDeleteId !== task.id) {
      setConfirmingDeleteId(task.id);
      return;
    }
    setConfirmingDeleteId(null);
    await dismissFailedTask(task.task_type, task.id);
  };

  const handleClearAllClick = async () => {
    if (!confirmingClearAll) {
      setConfirmingClearAll(true);
      return;
    }
    setConfirmingClearAll(false);
    await dismissAllFailedTasks();
  };

  const handleDeleteClick = async (taskId: string) => {
    // Two-click inline confirmation (no browser confirm())
    if (confirmingDeleteId !== taskId) {
      setConfirmingDeleteId(taskId);
      return;
    }
    setConfirmingDeleteId(null);
    try {
      await deleteTask(taskId);
    } catch (error) {
      console.error('Failed to delete task:', error);
    }
  };

  const getTaskTypeLabel = (type: string): string => TASK_TYPE_LABELS[type] || type;
  const getEntityTypeLabel = (type: string): string => ENTITY_TYPE_LABELS[type] || type;

  if (failedTasks.length === 0 && !failedLoading) {
    return (
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <AlertTriangle className="w-5 h-5 text-emerald-400" />
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Failed Operations</h2>
        </div>
        <p className="text-slate-500 text-center py-8">No failed operations</p>
      </div>
    );
  }

  return (
    <>
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Failed Operations</h2>
            <span className="px-2 py-1 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-300">
              {failedTasks.length}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {/* Clears only what is listed RIGHT NOW — not a standing rule, so a
                failure that happens later still surfaces. */}
            <button
              onClick={handleClearAllClick}
              onBlur={() => setConfirmingClearAll(false)}
              className={`px-2.5 py-1 rounded text-xs transition-colors ${
                confirmingClearAll
                  ? 'bg-red-600 hover:bg-red-700 text-white font-medium'
                  : 'text-slate-600 dark:text-slate-400 hover:text-red-400 hover:bg-red-500/10'
              }`}
              title={
                confirmingClearAll
                  ? 'Click again to clear the list'
                  : 'Clear all listed failures (job records are kept)'
              }
            >
              {confirmingClearAll ? 'Confirm clear all?' : 'Clear all'}
            </button>
            <button
              onClick={() => fetchFailedTasks()}
              disabled={failedLoading}
              className="text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors disabled:opacity-50"
              title="Refresh"
            >
              <RefreshCw className={`w-4 h-4 ${failedLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {failedError && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm">
            {failedError}
          </div>
        )}

        <div className="space-y-3">
          {failedTasks.map((task) => {
            const isExpanded = expandedTasks.has(task.id);
            const canManage = task.can_retry !== false;
            return (
              <div
                key={task.id}
                className="bg-slate-100 dark:bg-slate-800/50 border border-slate-300 dark:border-slate-700 rounded-lg overflow-hidden"
              >
                {/* Task Header */}
                <div className="p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="px-2 py-0.5 bg-red-500/20 border border-red-500/30 rounded text-xs font-medium text-red-300">
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
                        Failed: {new Date(task.completed_at || task.updated_at || task.created_at || '').toLocaleString()}
                      </div>
                      {task.retry_count > 0 && (
                        <div className="text-xs text-amber-400 mt-1">
                          Retry attempts: {task.retry_count}
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2">
                      {canManage ? (
                        <>
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
                            onBlur={() => setConfirmingDeleteId(null)}
                            className={`px-2 py-1.5 rounded text-xs transition-colors ${
                              confirmingDeleteId === task.id
                                ? 'bg-red-600 hover:bg-red-700 text-white font-medium'
                                : 'text-slate-600 dark:text-slate-400 hover:text-red-400 hover:bg-red-500/10'
                            }`}
                            title={
                              confirmingDeleteId === task.id
                                ? 'Click again to confirm removal'
                                : 'Remove from list'
                            }
                          >
                            {confirmingDeleteId === task.id ? (
                              'Confirm?'
                            ) : (
                              <Trash2 className="w-4 h-4" />
                            )}
                          </button>
                        </>
                      ) : (
                        <button
                          onClick={() => handleDismissClick(task)}
                          onBlur={() => setConfirmingDeleteId(null)}
                          className={`px-2 py-1.5 rounded text-xs transition-colors ${
                            confirmingDeleteId === task.id
                              ? 'bg-red-600 hover:bg-red-700 text-white font-medium'
                              : 'text-slate-600 dark:text-slate-400 hover:text-red-400 hover:bg-red-500/10'
                          }`}
                          title={
                            confirmingDeleteId === task.id
                              ? 'Click again to clear it from this list'
                              : 'Clear from this list (the job record is kept; retry from its own panel)'
                          }
                        >
                          {confirmingDeleteId === task.id ? (
                            'Confirm?'
                          ) : (
                            <Trash2 className="w-4 h-4" />
                          )}
                        </button>
                      )}
                      <button
                        onClick={() => toggleExpanded(task.id)}
                        className="p-1.5 text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
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
                  <div className="px-4 pb-4 pt-2 border-t border-slate-300 dark:border-slate-700">
                    <div className="text-xs text-slate-600 dark:text-slate-400 font-semibold mb-2">Error Details:</div>
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
