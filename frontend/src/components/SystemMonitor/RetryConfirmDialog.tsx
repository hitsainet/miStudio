/**
 * Retry Confirmation Dialog
 *
 * Modal dialog for confirming retry of failed operations.
 * Shows operation details and allows parameter overrides.
 */

import { useState } from 'react';
import { X, AlertTriangle, RefreshCw } from 'lucide-react';
import { useTaskQueueStore } from '../../stores/taskQueueStore';
import { TaskQueueEntry } from '../../types/taskQueue';

interface RetryConfirmDialogProps {
  task: TaskQueueEntry;
  onClose: () => void;
  onConfirm: () => void;
}

export function RetryConfirmDialog({ task, onClose, onConfirm }: RetryConfirmDialogProps) {
  const { retryTask } = useTaskQueueStore();
  const [retrying, setRetrying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRetry = async () => {
    try {
      setRetrying(true);
      setError(null);
      await retryTask(task.id);
      onConfirm();
    } catch (err) {
      console.error('Retry failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to retry operation');
      setRetrying(false);
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

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <RefreshCw className="w-6 h-6 text-emerald-400" />
            <div>
              <h2 className="text-xl font-semibold text-slate-100">Confirm Retry</h2>
              <p className="text-sm text-slate-400 mt-1">Retry this failed operation</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={retrying}
            className="text-slate-400 hover:text-slate-300 transition-colors disabled:opacity-50"
            aria-label="Close"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Operation Details */}
          <div>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Operation Details</h3>
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 space-y-3">
              <div>
                <div className="text-xs text-slate-500">Type</div>
                <div className="flex items-center gap-2 mt-1">
                  <span className="px-2 py-0.5 bg-emerald-500/20 border border-emerald-500/30 rounded text-xs font-medium text-emerald-300">
                    {getTaskTypeLabel(task.task_type)}
                  </span>
                  <span className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                    {getEntityTypeLabel(task.entity_type)}
                  </span>
                </div>
              </div>

              <div>
                <div className="text-xs text-slate-500">Name</div>
                <div className="text-sm text-slate-100 mt-1 font-medium">
                  {task.entity_info?.name || task.entity_id}
                </div>
              </div>

              {task.entity_info?.repo_id && (
                <div>
                  <div className="text-xs text-slate-500">Repository</div>
                  <div className="text-sm text-slate-300 mt-1 font-mono">
                    {task.entity_info.repo_id}
                  </div>
                </div>
              )}

              {task.retry_count > 0 && (
                <div>
                  <div className="text-xs text-slate-500">Previous Retry Attempts</div>
                  <div className="text-sm text-amber-400 mt-1 font-medium">
                    {task.retry_count}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Error Information */}
          <div>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Previous Error</h3>
            <div className="bg-red-950/30 border border-red-900/50 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-red-300 font-mono whitespace-pre-wrap break-words">
                    {task.error_message || 'No error message available'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Warning */}
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-sm font-medium text-amber-300">Retry Attempt</div>
                <div className="text-sm text-amber-200/80 mt-1">
                  This will attempt the operation again with the same parameters. If the operation
                  failed due to a temporary issue (network, memory, etc.), it may succeed on retry.
                </div>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
              <div className="text-sm text-red-300">{error}</div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-800 p-4 bg-slate-900/50">
          <div className="flex items-center justify-end gap-3">
            <button
              onClick={onClose}
              disabled={retrying}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors text-slate-300 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={handleRetry}
              disabled={retrying}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg transition-colors text-white font-medium flex items-center gap-2"
            >
              {retrying ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Retrying...
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4" />
                  Retry Operation
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
