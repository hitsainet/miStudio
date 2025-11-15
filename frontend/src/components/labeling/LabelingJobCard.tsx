/**
 * LabelingJobCard Component
 *
 * Displays an individual labeling job with status, progress, and actions.
 */

import React, { useEffect, useState } from 'react';
import { Tag, Loader, CheckCircle, XCircle, Trash2, Clock, Ban } from 'lucide-react';
import type { LabelingJob } from '../../types/labeling';
import { LabelingStatus } from '../../types/labeling';
import { format, intervalToDuration } from 'date-fns';
import { COMPONENTS } from '../../config/brand';
import { useLabelingPromptTemplatesStore } from '../../stores/labelingPromptTemplatesStore';
import { LabelingResultsWindow } from './LabelingResultsWindow';

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
  const [templateName, setTemplateName] = useState<string>('Default Template');
  const { templates, fetchTemplate } = useLabelingPromptTemplatesStore();

  const isActive = job.status === LabelingStatus.QUEUED || job.status === LabelingStatus.LABELING;
  const isCompleted = job.status === LabelingStatus.COMPLETED;
  const isFailed = job.status === LabelingStatus.FAILED;
  const isCancelled = job.status === LabelingStatus.CANCELLED;

  // Fetch template name if template ID is present
  useEffect(() => {
    if (job.prompt_template_id) {
      // First check if we already have it in the store
      const existingTemplate = templates.find(t => t.id === job.prompt_template_id);
      if (existingTemplate) {
        setTemplateName(existingTemplate.name);
      } else {
        // Fetch from API if not in store
        fetchTemplate(job.prompt_template_id)
          .then(() => {
            const template = templates.find(t => t.id === job.prompt_template_id);
            if (template) {
              setTemplateName(template.name);
            }
          })
          .catch(() => {
            setTemplateName('Unknown Template');
          });
      }
    }
  }, [job.prompt_template_id, templates]);

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
    <div className={`${COMPONENTS.card.base} p-4 border ${statusDisplay.border}`}>
      {/* Compact Header: Status, Title, Info, and Actions - Single Row */}
      <div className="flex items-center justify-between mb-3">
        {/* Left: Status Icon + Title + Key Info */}
        <div className="flex items-center gap-3 flex-1">
          <div className={`p-1.5 rounded-lg ${statusDisplay.bg} ${statusDisplay.color}`}>
            {statusDisplay.icon}
          </div>
          <div className="flex items-center gap-4 flex-1">
            <span className={`text-sm font-semibold ${COMPONENTS.text.primary}`}>
              Labeling Job
            </span>
            <span className={`text-xs px-2 py-0.5 rounded ${statusDisplay.bg} ${statusDisplay.color}`}>
              {statusDisplay.label}
            </span>
            <span className={`text-xs ${COMPONENTS.text.secondary}`}>
              {job.extraction_job_id}
            </span>
            <span className={`text-xs ${COMPONENTS.text.secondary}`}>
              {job.labeling_method === 'openai'
                ? 'OpenAI'
                : job.labeling_method === 'openai_compatible'
                ? job.openai_compatible_model || 'Ollama'
                : 'Local LLM'}
            </span>
            <span className={`text-xs ${COMPONENTS.text.secondary}`}>
              Started: {format(new Date(job.created_at), 'MMM d, h:mm a')}
            </span>
            {isActive && (
              <>
                <span className="text-xs text-blue-400 font-medium">
                  Elapsed: {getElapsedTime()}
                </span>
                <span className={`text-xs ${COMPONENTS.text.secondary}`}>
                  {job.features_labeled.toLocaleString()} / {totalFeatures.toLocaleString()}
                </span>
                <span className="text-xs text-emerald-400 font-medium">
                  {progress.toFixed(1)}%
                </span>
              </>
            )}
            {isCompleted && job.completed_at && (
              <span className="text-xs text-emerald-400 font-medium">
                Completed in {getElapsedTime()}
              </span>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {isActive && onCancel && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('Are you sure you want to cancel this labeling job?')) {
                  onCancel();
                }
              }}
              className={`p-1.5 rounded-lg ${COMPONENTS.button.ghost}`}
              title="Cancel labeling"
            >
              <XCircle className="w-4 h-4" />
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
              className={`p-1.5 rounded-lg ${COMPONENTS.button.ghost}`}
              title="Delete labeling job"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar for Active Jobs */}
      {isActive && (
        <div className="mb-3">
          <div className={`h-1.5 ${COMPONENTS.card.base} rounded-full overflow-hidden`}>
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Real-time Results Window - Full Width at Bottom */}
      {isActive && (
        <div>
          <LabelingResultsWindow labelingJobId={job.id} />
        </div>
      )}

      {/* Single Column Layout for Completed/Failed/Cancelled Jobs */}
      {!isActive && (
        <div className={`text-sm ${COMPONENTS.text.secondary} space-y-1`}>
          <p>
            <span className="font-medium">Extraction:</span> {job.extraction_job_id}
          </p>
          <p>
            <span className="font-medium">Method:</span>{' '}
            {job.labeling_method === 'openai'
              ? 'OpenAI'
              : job.labeling_method === 'openai_compatible'
              ? `Local LLM (${job.openai_compatible_model || 'Ollama'})`
              : `Local LLM (${job.local_model || 'meta-llama/Llama-3.2-1B'})`}
          </p>
          <p>
            <span className="font-medium">Template:</span> {templateName}
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
