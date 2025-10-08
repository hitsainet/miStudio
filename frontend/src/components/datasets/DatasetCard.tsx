/**
 * DatasetCard component for displaying individual dataset information.
 *
 * This component renders a card with dataset details, status, and progress.
 */

import React from 'react';
import { Database, CheckCircle, Loader, Activity, Trash2 } from 'lucide-react';
import { Dataset, DatasetStatus } from '../../types/dataset';
import { StatusBadge } from '../common/StatusBadge';
import { ProgressBar } from '../common/ProgressBar';
import { formatFileSize } from '../../utils/formatters';

interface DatasetCardProps {
  dataset: Dataset;
  onClick?: () => void;
  onDelete?: (id: string) => void;
}

export function DatasetCard({ dataset, onClick, onDelete }: DatasetCardProps) {
  const isClickable = dataset.status === DatasetStatus.READY;
  const showProgress =
    dataset.status === DatasetStatus.DOWNLOADING ||
    dataset.status === DatasetStatus.PROCESSING;

  // Status icon mapping
  const StatusIcon = React.useMemo(() => {
    switch (dataset.status) {
      case DatasetStatus.READY:
        return CheckCircle;
      case DatasetStatus.DOWNLOADING:
        return Loader;
      case DatasetStatus.PROCESSING:
      case DatasetStatus.ERROR:
        return Activity;
      default:
        return Database;
    }
  }, [dataset.status]);

  const iconClassName = React.useMemo(() => {
    if (dataset.status === DatasetStatus.DOWNLOADING) {
      return 'animate-spin';
    }
    return '';
  }, [dataset.status]);

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click
    if (window.confirm(`Are you sure you want to delete "${dataset.name}"? This will remove all downloaded files.`)) {
      onDelete?.(dataset.id);
    }
  };

  return (
    <div
      className={`bg-slate-900/50 border border-slate-800 rounded-lg p-6 transition-all ${
        isClickable
          ? 'cursor-pointer hover:bg-slate-900/70 hover:border-slate-700'
          : 'cursor-default'
      }`}
      onClick={isClickable ? onClick : undefined}
    >
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0">
          <Database className="w-8 h-8 text-slate-400" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-semibold text-slate-100 truncate">
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
              {onDelete && (
                <button
                  onClick={handleDelete}
                  className="ml-2 p-1.5 hover:bg-red-500/10 rounded transition-colors group"
                  title="Delete dataset"
                >
                  <Trash2 className="w-4 h-4 text-slate-500 group-hover:text-red-400 transition-colors" />
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

          {showProgress && dataset.progress !== undefined && (
            <div className="mt-4">
              <ProgressBar progress={dataset.progress} />
            </div>
          )}

          {dataset.error_message && (
            <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">
              {dataset.error_message}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
