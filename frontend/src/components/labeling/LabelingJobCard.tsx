/**
 * LabelingJobCard Component
 *
 * Displays an individual labeling job with status, progress, and actions.
 */

import React from 'react';
import { Tag, Loader, CheckCircle, XCircle, Trash2, Clock, Ban } from 'lucide-react';
import type { LabelingJob } from '../../types/labeling';
import { LabelingStatus } from '../../types/labeling';
import { format, intervalToDuration } from 'date-fns';
import { COMPONENTS } from '../../config/brand';

interface LabelingJobCardProps {
  job: LabelingJob;
  onCancel?: () => void;
  onDelete?: () => void;
}

export const LabelingJobCard: React.FC<LabelingJobCardProps> = ({
  job,
  onCancel,
  onDelete,
}) => {
  const isActive = job.status === LabelingStatus.QUEUED || job.status === LabelingStatus.LABELING;
  const isCompleted = job.status === LabelingStatus.COMPLETED;
  const isFailed = job.status === LabelingStatus.FAILED;
  const isCancelled = job.status === LabelingStatus.CANCELLED;

  // Calculate progress percentage
  const totalFeatures = job.total_features || 0;
  const progress = totalFeatures > 0
    ? (job.features_labeled / totalFeatures) * 100
    : 0;

  // Get elapsed time
  const getElapsedTime = () => {
    const start = new Date(job.created_at);
    const end = job.completed_at ? new Date(job.completed_at) : new Date();

    const duration = intervalToDuration({ start, end });

    const parts = [];
    if (duration.hours) parts.push(`${duration.hours}h`);
    if (duration.minutes) parts.push(`${duration.minutes}m`);
    if (duration.seconds) parts.push(`${duration.seconds}s`);

    return parts.length > 0 ? parts.join(' ') : '0s';
  };

  // Status icon and color
  const getStatusDisplay = () => {
    switch (job.status) {
      case LabelingStatus.QUEUED:
        return {
          icon: <Clock className="w-5 h-5" />,
          color: 'text-yellow-400',
          bg: 'bg-yellow-500/10',
          border: 'border-yellow-500/30',
          label: 'Queued',
        };
      case LabelingStatus.LABELING:
        return {
          icon: <Loader className="w-5 h-5 animate-spin" />,
          color: 'text-blue-400',
          bg: 'bg-blue-500/10',
          border: 'border-blue-500/30',
          label: 'Labeling',
        };
      case LabelingStatus.COMPLETED:
        return {
          icon: <CheckCircle className="w-5 h-5" />,
          color: 'text-emerald-400',
          bg: 'bg-emerald-500/10',
          border: 'border-emerald-500/30',
          label: 'Completed',
        };
      case LabelingStatus.FAILED:
        return {
          icon: <XCircle className="w-5 h-5" />,
          color: 'text-red-400',
          bg: 'bg-red-500/10',
          border: 'border-red-500/30',
          label: 'Failed',
        };
      case LabelingStatus.CANCELLED:
        return {
          icon: <Ban className="w-5 h-5" />,
          color: 'text-slate-400',
          bg: 'bg-slate-500/10',
          border: 'border-slate-500/30',
          label: 'Cancelled',
        };
      default:
        return {
          icon: <Tag className="w-5 h-5" />,
          color: 'text-slate-400',
          bg: 'bg-slate-500/10',
          border: 'border-slate-500/30',
          label: 'Unknown',
        };
    }
  };

  const statusDisplay = getStatusDisplay();

  return (
    <div className={`${COMPONENTS.card.base} p-6 border ${statusDisplay.border}`}>
      <div className="flex items-start justify-between">
        {/* Left: Status and Info */}
        <div className="flex items-start gap-4 flex-1">
          {/* Status Icon */}
          <div className={`p-3 rounded-lg ${statusDisplay.bg} ${statusDisplay.color}`}>
            {statusDisplay.icon}
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            {/* Job ID and Status */}
            <div className="flex items-center gap-3 mb-2">
              <h3 className={`text-lg font-semibold ${COMPONENTS.text.primary}`}>
                Labeling Job
              </h3>
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${statusDisplay.bg} ${statusDisplay.color}`}>
                {statusDisplay.label}
              </span>
            </div>

            {/* Job Details */}
            <div className={`text-sm ${COMPONENTS.text.secondary} space-y-1`}>
              <p>
                <span className="font-medium">Extraction:</span> {job.extraction_job_id}
              </p>
              <p>
                <span className="font-medium">Method:</span>{' '}
                {job.labeling_method === 'openai'
                  ? 'OpenAI (requires api-key)'
                  : `Local LLM (${job.local_model || 'meta-llama/Llama-3.2-1B'})`}
              </p>
              <p>
                <span className="font-medium">Features:</span>{' '}
                {job.features_labeled.toLocaleString()} / {totalFeatures.toLocaleString()}
                {isCompleted && ' ✓'}
              </p>
              <p>Started: {format(new Date(job.created_at), 'MMM d, yyyy • h:mm:ss a')}</p>
              {job.completed_at && (
                <>
                  <p>Completed: {format(new Date(job.completed_at), 'MMM d, yyyy • h:mm:ss a')}</p>
                  <p className="text-emerald-400 font-medium">Elapsed: {getElapsedTime()}</p>
                </>
              )}
              {isActive && (
                <p className="text-blue-400 font-medium">Elapsed: {getElapsedTime()}</p>
              )}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 flex-shrink-0 ml-4">
          {isActive && onCancel && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('Are you sure you want to cancel this labeling job?')) {
                  onCancel();
                }
              }}
              className={`p-2 rounded-lg ${COMPONENTS.button.ghost}`}
              title="Cancel labeling"
            >
              <XCircle className="w-5 h-5" />
            </button>
          )}
          {(isCompleted || isFailed || isCancelled) && onDelete && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('Are you sure you want to delete this labeling job?')) {
                  onDelete();
                }
              }}
              className={`p-2 rounded-lg ${COMPONENTS.button.ghost}`}
              title="Delete labeling job"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar for Active Jobs */}
      {isActive && (
        <div className="mt-4">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className={COMPONENTS.text.secondary}>
              {job.features_labeled.toLocaleString()} / {totalFeatures.toLocaleString()} features
            </span>
            <span className="text-emerald-400 font-medium">
              {progress.toFixed(1)}%
            </span>
          </div>
          <div className={`h-2 ${COMPONENTS.card.base} rounded-full overflow-hidden`}>
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error Message */}
      {isFailed && job.error_message && (
        <div className="mt-4 p-3 bg-red-900/20 border border-red-800 rounded-lg">
          <p className="text-sm text-red-200">
            <span className="font-medium">Error:</span> {job.error_message}
          </p>
        </div>
      )}
    </div>
  );
};
