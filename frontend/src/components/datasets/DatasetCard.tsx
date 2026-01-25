/**
 * DatasetCard component for displaying individual dataset information.
 *
 * This component renders a card with dataset details, status, and progress.
 */

import React from 'react';
import { Database, CheckCircle, Loader, Activity, Trash2, Settings, X } from 'lucide-react';
import { Dataset, DatasetTokenizationProgress } from '../../types/dataset';
import { StatusBadge } from '../common/StatusBadge';
import { ProgressBar } from '../common/ProgressBar';
import { TokenizationProgressDisplay } from './TokenizationProgressDisplay';
import { formatFileSize } from '../../utils/formatters';
import { COMPONENTS } from '../../config/brand';

interface DatasetCardProps {
  dataset: Dataset;
  tokenizationProgress?: DatasetTokenizationProgress; // Optional tokenization progress
  onClick?: () => void;
  onDelete?: (id: string) => void;
  onCancel?: (id: string) => void;
}

export function DatasetCard({ dataset, tokenizationProgress, onClick, onDelete, onCancel }: DatasetCardProps) {
  // Log props
  console.log(`[DatasetCard] Rendering ${dataset.name}:`, {
    status: dataset.status,
    progress: dataset.progress,
    hasTokenizationProgress: !!tokenizationProgress,
    tokenizationProgress: tokenizationProgress ? {
      stage: tokenizationProgress.stage,
      progress: tokenizationProgress.progress,
      samples: `${tokenizationProgress.samples_processed}/${tokenizationProgress.total_samples}`,
    } : null,
  });

  // Normalize status to string for consistent comparisons
  const statusString = String(dataset.status).toLowerCase();

  // Allow clicking during processing (tokenization) so user can view/manage tokenization jobs
  const isClickable = statusString === 'ready' || statusString === 'error' || statusString === 'processing';
  const showProgress = statusString === 'downloading' || statusString === 'processing';
  const isTokenized = dataset.metadata?.tokenization !== undefined;
  const isActive = statusString === 'downloading' || statusString === 'processing';

  // Status icon mapping
  const StatusIcon = React.useMemo(() => {
    if (statusString === 'ready') {
      return CheckCircle;
    }
    if (statusString === 'downloading') {
      return Loader;
    }
    if (statusString === 'processing' || statusString === 'error') {
      return Activity;
    }
    return Database;
  }, [statusString]);

  const iconClassName = React.useMemo(() => {
    if (statusString === 'downloading') {
      return 'animate-spin';
    }
    return '';
  }, [statusString]);

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click
    if (window.confirm(`Are you sure you want to delete "${dataset.name}"? This will remove all downloaded files.`)) {
      onDelete?.(dataset.id);
    }
  };

  const handleCancel = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click
    if (window.confirm(`Are you sure you want to cancel this operation? Partial files will be deleted.`)) {
      onCancel?.(dataset.id);
    }
  };

  return (
    <div
      className={`${COMPONENTS.card.base} p-4 transition-all ${
        isClickable
          ? 'cursor-pointer hover:bg-slate-900/70 hover:border-slate-700'
          : 'cursor-default'
      }`}
      onClick={isClickable ? onClick : undefined}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0">
          <Database className="w-6 h-6 text-slate-400" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <h3 className="text-base font-semibold text-slate-100 truncate">
                {dataset.name}
              </h3>
              <p className="text-sm text-slate-400 mt-1 truncate">
                Source: {dataset.source}
                {dataset.hf_repo_id && ` • ${dataset.hf_repo_id}`}
              </p>
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              <StatusIcon className={`w-5 h-5 text-slate-400 ${iconClassName}`} />
              <StatusBadge status={dataset.status} />
              {isActive && onCancel && (
                <button
                  onClick={handleCancel}
                  className={`ml-2 p-1.5 rounded ${COMPONENTS.button.ghost}`}
                  title="Cancel operation"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
              {!isActive && onDelete && (
                <button
                  onClick={handleDelete}
                  className={`ml-2 p-1.5 rounded ${COMPONENTS.button.ghost}`}
                  title="Delete dataset"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {dataset.size_bytes !== undefined && dataset.size_bytes > 0 && (
            <p className="text-sm text-slate-400 mt-2">
              Size: {formatFileSize(dataset.size_bytes)}
            </p>
          )}

          {dataset.num_samples !== undefined && dataset.num_samples > 0 && (
            <p className="text-sm text-slate-400 mt-1">
              Samples: {dataset.num_samples.toLocaleString()}
            </p>
          )}

          {isTokenized && (
            <div className="inline-flex items-center gap-1.5 mt-2 px-2 py-1 bg-emerald-500/10 border border-emerald-500/30 rounded text-xs text-emerald-400">
              <Settings className="w-3 h-3" />
              <span>Tokenized</span>
            </div>
          )}

          {showProgress && (
            <div className="mt-3">
              {tokenizationProgress ? (
                <TokenizationProgressDisplay progress={tokenizationProgress} />
              ) : (() => {
                // Fallback: Check for active tokenization progress from dataset
                const activeTokenization = dataset.tokenizations?.find(
                  t => t.status === 'processing' || t.status === 'queued'
                );
                if (activeTokenization && activeTokenization.progress !== undefined && activeTokenization.progress > 0) {
                  // Create a minimal progress object for display
                  return (
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs text-slate-400">
                        <span>Tokenizing</span>
                        <span>{activeTokenization.progress.toFixed(1)}%</span>
                      </div>
                      <ProgressBar progress={activeTokenization.progress} />
                    </div>
                  );
                }
                // Final fallback to dataset progress
                return dataset.progress !== undefined ? (
                  <ProgressBar progress={dataset.progress} />
                ) : null;
              })()}
            </div>
          )}

          {dataset.error_message && (
            <div className="mt-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">
              {dataset.error_message}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
