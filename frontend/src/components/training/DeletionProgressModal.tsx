/**
 * Deletion Progress Modal Component
 *
 * Displays real-time progress tracking during training job deletion.
 * Shows a task checklist with spinner (in-progress) and checkmark (completed) indicators.
 */

import React from 'react';
import { X, Loader2, CheckCircle2 } from 'lucide-react';

interface DeletionTask {
  id: string;
  label: string;
  status: 'pending' | 'in_progress' | 'completed';
  message?: string;
  count?: number;
}

interface DeletionProgressModalProps {
  isOpen: boolean;
  onClose: () => void;
  trainingId: string;
  tasks: DeletionTask[];
}

const DeletionProgressModal: React.FC<DeletionProgressModalProps> = ({
  isOpen,
  onClose,
  trainingId,
  tasks,
}) => {
  if (!isOpen) return null;

  const allCompleted = tasks.every(task => task.status === 'completed');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-slate-800 border border-slate-700 rounded-lg shadow-2xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
          <h2 className="text-lg font-semibold text-slate-100">
            Deleting Training
          </h2>
          {allCompleted && (
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-slate-200 transition-colors"
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Content */}
        <div className="px-6 py-4">
          <p className="text-sm text-slate-400 mb-4">
            Training ID: <span className="text-slate-300 font-mono">{trainingId}</span>
          </p>

          {/* Task Checklist */}
          <div className="space-y-3">
            {tasks.map((task) => (
              <div
                key={task.id}
                className="flex items-start gap-3 p-3 rounded-lg bg-slate-900/50"
              >
                {/* Status Icon */}
                <div className="flex-shrink-0 mt-0.5">
                  {task.status === 'in_progress' && (
                    <Loader2 className="w-5 h-5 text-emerald-400 animate-spin" />
                  )}
                  {task.status === 'completed' && (
                    <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  )}
                  {task.status === 'pending' && (
                    <div className="w-5 h-5 rounded-full border-2 border-slate-600" />
                  )}
                </div>

                {/* Task Info */}
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium ${
                    task.status === 'completed'
                      ? 'text-slate-300'
                      : task.status === 'in_progress'
                      ? 'text-emerald-300'
                      : 'text-slate-500'
                  }`}>
                    {task.label}
                  </p>
                  {task.message && (
                    <p className="text-xs text-slate-400 mt-1">
                      {task.message}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        {allCompleted && (
          <div className="px-6 py-4 border-t border-slate-700">
            <button
              onClick={onClose}
              className="w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-colors font-medium"
            >
              Done
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DeletionProgressModal;
