/**
 * ExtractionJobCard Component
 *
 * Displays an individual extraction job with status, progress, and actions.
 */

import React from 'react';
import { Zap, Loader, CheckCircle, XCircle, Trash2, Clock } from 'lucide-react';
import type { ExtractionStatusResponse } from '../../types/features';
import { formatDistanceToNow } from 'date-fns';

interface ExtractionJobCardProps {
  extraction: ExtractionStatusResponse;
  onCancel?: () => void;
  onDelete?: () => void;
  onRetry?: () => void;
}

export const ExtractionJobCard: React.FC<ExtractionJobCardProps> = ({
  extraction,
  onCancel,
  onDelete,
  onRetry,
}) => {
  const isActive = extraction.status === 'queued' || extraction.status === 'extracting';
  const isCompleted = extraction.status === 'completed';
  const isFailed = extraction.status === 'failed';

  const getStatusBadge = () => {
    const baseClasses = 'px-3 py-1 rounded-full text-sm font-medium';

    switch (extraction.status) {
      case 'completed':
        return (
          <span className={`${baseClasses} bg-emerald-900/30 text-emerald-400 flex items-center gap-1`}>
            <CheckCircle className="w-4 h-4" />
            Completed
          </span>
        );
      case 'extracting':
        return (
          <span className={`${baseClasses} bg-blue-900/30 text-blue-400 flex items-center gap-1`}>
            <Loader className="w-4 h-4 animate-spin" />
            Extracting
          </span>
        );
      case 'queued':
        return (
          <span className={`${baseClasses} bg-yellow-900/30 text-yellow-400 flex items-center gap-1`}>
            <Clock className="w-4 h-4" />
            Queued
          </span>
        );
      case 'failed':
        return (
          <span className={`${baseClasses} bg-red-900/30 text-red-400 flex items-center gap-1`}>
            <XCircle className="w-4 h-4" />
            Failed
          </span>
        );
      default:
        return (
          <span className={`${baseClasses} bg-slate-800 text-slate-300`}>
            Unknown
          </span>
        );
    }
  };

  const progress = (extraction.progress || 0) * 100;

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 hover:border-slate-700 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-start gap-3 flex-1">
          <Zap className="w-6 h-6 text-emerald-400 flex-shrink-0 mt-1" />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-semibold text-lg text-slate-100 truncate">
                {extraction.model_name || 'Unknown Model'} - {extraction.dataset_name || 'Unknown Dataset'}
              </h3>
              {getStatusBadge()}
            </div>
            <p className="text-sm text-slate-400">
              Started {formatDistanceToNow(new Date(extraction.created_at), { addSuffix: true })}
            </p>
            {extraction.completed_at && (
              <p className="text-sm text-slate-400">
                Completed {formatDistanceToNow(new Date(extraction.completed_at), { addSuffix: true })}
              </p>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 flex-shrink-0 ml-4">
          {isActive && onCancel && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('Are you sure you want to cancel this extraction?')) {
                  onCancel();
                }
              }}
              className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
              title="Cancel extraction"
            >
              <XCircle className="w-5 h-5" />
            </button>
          )}
          {(isCompleted || isFailed) && onDelete && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('Are you sure you want to delete this extraction?')) {
                  onDelete();
                }
              }}
              className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
              title="Delete extraction"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar for Active Extractions */}
      {isActive && extraction.progress !== null && extraction.progress !== undefined && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-slate-400">
              {extraction.features_extracted !== null && extraction.total_features !== null
                ? `${extraction.features_extracted.toLocaleString()} / ${extraction.total_features.toLocaleString()} features`
                : 'Processing...'}
            </span>
            <span className="text-emerald-400 font-medium">
              {progress.toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Statistics for Completed */}
      {isCompleted && extraction.statistics && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="bg-slate-800/50 rounded p-3">
            <div className="text-xs text-slate-400 mb-1">Features Found</div>
            <div className="text-lg font-bold text-emerald-400">
              {extraction.statistics.total_features?.toLocaleString() || 'N/A'}
            </div>
          </div>
          <div className="bg-slate-800/50 rounded p-3">
            <div className="text-xs text-slate-400 mb-1">Interpretable</div>
            <div className="text-lg font-bold text-blue-400">
              {extraction.statistics.interpretable_percentage !== undefined
                ? `${extraction.statistics.interpretable_percentage.toFixed(1)}%`
                : 'N/A'}
            </div>
          </div>
          <div className="bg-slate-800/50 rounded p-3">
            <div className="text-xs text-slate-400 mb-1">Activation Rate</div>
            <div className="text-lg font-bold text-purple-400">
              {extraction.statistics.avg_activation_frequency !== undefined
                ? `${(extraction.statistics.avg_activation_frequency * 100).toFixed(2)}%`
                : 'N/A'}
            </div>
          </div>
        </div>
      )}

      {/* Error Message for Failed */}
      {isFailed && extraction.error_message && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          <div className="flex items-start gap-2">
            <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <div className="font-medium mb-1">Extraction Failed</div>
              <div className="text-red-300">{extraction.error_message}</div>
            </div>
          </div>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-3 w-full px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded transition-colors text-sm"
            >
              Retry Extraction
            </button>
          )}
        </div>
      )}

      {/* Configuration Details */}
      <div className="mt-4 pt-4 border-t border-slate-800">
        <details className="text-sm">
          <summary className="cursor-pointer text-slate-400 hover:text-slate-300 transition-colors">
            Configuration
          </summary>
          <div className="mt-2 space-y-1 text-slate-400">
            <div>Evaluation Samples: <span className="text-white">{extraction.config.evaluation_samples?.toLocaleString() || 'N/A'}</span></div>
            <div>Top-K Examples: <span className="text-white">{extraction.config.top_k_examples || 'N/A'}</span></div>
            <div className="text-xs text-slate-500 mt-2">ID: {extraction.id}</div>
          </div>
        </details>
      </div>
    </div>
  );
};
