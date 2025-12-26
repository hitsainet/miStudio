/**
 * ModelCard - Card component displaying model information and status.
 *
 * Features:
 * - Model name, parameters, quantization, and memory info
 * - Real-time progress bars for download/quantization
 * - Status indicators (ready, downloading, quantizing, error)
 * - Action buttons (extract activations, view history, delete)
 * - Click to view architecture details
 * - Elapsed time display during extraction
 */

import { useState, useEffect } from 'react';
import { Cpu, CheckCircle, Loader, Activity, AlertCircle, Trash2, History, Clock } from 'lucide-react';
import { Model, ModelStatus } from '../../types/model';
import { COMPONENTS } from '../../config/brand';

interface ModelCardProps {
  model: Model;
  onClick: () => void;
  onExtract: () => void;
  onViewExtractions?: () => void;
  onDeleteExtractions?: () => void;
  onDelete: (id: string) => void;
  onCancel: (id: string) => void;
}

export function ModelCard({ model, onClick, onExtract, onViewExtractions, onDelete, onCancel }: ModelCardProps) {
  const [elapsedTime, setElapsedTime] = useState<number>(0);

  const isActive = model.status === ModelStatus.DOWNLOADING ||
                   model.status === ModelStatus.LOADING ||
                   model.status === ModelStatus.QUANTIZING;

  const isReady = model.status === ModelStatus.READY;
  const isError = model.status === ModelStatus.ERROR;
  const isExtracting = model.extraction_status && model.extraction_status !== 'complete' && model.extraction_status !== 'error' && model.extraction_status !== 'failed';

  // Track elapsed time during extraction
  useEffect(() => {
    if (isExtracting && model.extraction_started_at) {
      // Calculate initial elapsed time
      const initialElapsed = Math.floor((Date.now() - model.extraction_started_at) / 1000);
      setElapsedTime(initialElapsed);

      // Update every second
      const timer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - model.extraction_started_at!) / 1000);
        setElapsedTime(elapsed);
      }, 1000);

      return () => clearInterval(timer);
    }
    setElapsedTime(0);
    return undefined;
  }, [isExtracting, model.extraction_started_at]);

  const formatElapsedTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins > 0) {
      return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
  };

  const formatParams = (count: number): string => {
    if (count >= 1_000_000_000) {
      return `${(count / 1_000_000_000).toFixed(1)}B`;
    } else if (count >= 1_000_000) {
      return `${Math.round(count / 1_000_000)}M`;
    } else if (count >= 1_000) {
      return `${Math.round(count / 1_000)}K`;
    }
    return count.toString();
  };

  const formatMemory = (bytes?: number): string => {
    if (!bytes) return 'N/A';
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${Math.round(mb)} MB`;
  };

  const getStatusIcon = () => {
    switch (model.status) {
      case ModelStatus.READY:
        return <CheckCircle className="w-5 h-5 text-emerald-400" />;
      case ModelStatus.DOWNLOADING:
        return <Loader className="w-5 h-5 text-blue-400 animate-spin" />;
      case ModelStatus.LOADING:
      case ModelStatus.QUANTIZING:
        return <Activity className="w-5 h-5 text-yellow-400 animate-pulse" />;
      case ModelStatus.ERROR:
        return <AlertCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Cpu className="w-5 h-5 text-slate-400" />;
    }
  };

  const getStatusBadge = () => {
    const baseClasses = 'px-3 py-1 rounded-full text-sm font-medium';

    switch (model.status) {
      case ModelStatus.READY:
        return (
          <span className={`${baseClasses} bg-emerald-900/30 text-emerald-400`}>
            Ready
          </span>
        );
      case ModelStatus.DOWNLOADING:
        return (
          <span className={`${baseClasses} bg-blue-900/30 text-blue-400`}>
            Downloading
          </span>
        );
      case ModelStatus.LOADING:
        return (
          <span className={`${baseClasses} bg-yellow-900/30 text-yellow-400`}>
            Loading
          </span>
        );
      case ModelStatus.QUANTIZING:
        return (
          <span className={`${baseClasses} bg-purple-900/30 text-purple-400`}>
            Quantizing
          </span>
        );
      case ModelStatus.ERROR:
        return (
          <span className={`${baseClasses} bg-red-900/30 text-red-400`}>
            Error
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

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    // Always delete the model when trash can is clicked
    if (confirm(`Are you sure you want to delete ${model.name}? This will also delete all associated extractions.`)) {
      onDelete(model.id);
    }
  };

  const handleCancel = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Are you sure you want to cancel this download? Partial files will be deleted.`)) {
      onCancel(model.id);
    }
  };

  const handleExtract = (e: React.MouseEvent) => {
    e.stopPropagation();
    onExtract();
  };

  return (
    <div className={`${COMPONENTS.card.base} p-4 hover:border-slate-700 transition-colors`}>
      <div className="flex items-center justify-between">
        {/* Model Info */}
        <div
          className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity flex-1"
          onClick={onClick}
        >
          <Cpu className="w-6 h-6 text-purple-400 flex-shrink-0" />
          <div className="min-w-0">
            <h3 className="font-semibold text-base text-slate-100 truncate">{model.name}</h3>
            <p className="text-sm text-slate-400 mt-0.5">
              {formatParams(model.params_count)} params • {model.quantization} quantization
              {model.memory_required_bytes && ` • ${formatMemory(model.memory_required_bytes)} memory`}
            </p>
            {model.repo_id && (
              <p className="text-xs text-slate-500 mt-1 font-mono truncate">
                {model.repo_id}
              </p>
            )}
          </div>
        </div>

        {/* Actions & Status */}
        <div className="flex items-center gap-3 flex-shrink-0">
          {isReady && (
            <>
              <button
                type="button"
                onClick={handleExtract}
                className={`text-sm flex items-center gap-2 ${COMPONENTS.button.primary}`}
              >
                <Loader
                  className="w-4 h-4"
                  style={{
                    color: model.has_completed_extractions ? '#10b981' : '#9ca3af'
                  }}
                />
                Extract Activations
              </button>
              {onViewExtractions && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onViewExtractions();
                  }}
                  className={`p-2 rounded-lg ${COMPONENTS.button.ghost}`}
                  title="View extraction history"
                >
                  <History
                    className="w-5 h-5"
                    style={{
                      color: model.has_completed_extractions ? '#10b981' : '#9ca3af'
                    }}
                  />
                </button>
              )}
            </>
          )}

          {isActive && (
            <button
              type="button"
              onClick={handleCancel}
              className={`p-2 rounded-lg ${COMPONENTS.button.ghost}`}
              title="Cancel download"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          )}

          {!isActive && (
            <button
              type="button"
              onClick={handleDelete}
              className={`p-2 rounded-lg ${COMPONENTS.button.ghost}`}
              title="Delete model"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}

          {getStatusIcon()}
          {getStatusBadge()}
        </div>
      </div>

      {/* Progress Bar for Active Downloads/Processing */}
      {isActive && model.progress !== undefined && (
        <div className="mt-3 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">
              {model.status === ModelStatus.DOWNLOADING && 'Download Progress'}
              {model.status === ModelStatus.LOADING && 'Loading Model'}
              {model.status === ModelStatus.QUANTIZING && 'Quantizing Model'}
            </span>
            <span className="text-emerald-400 font-medium font-mono">
              {model.progress > 0 ? `${model.progress.toFixed(1)}%` : 'Starting...'}
            </span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            {model.progress > 0 ? (
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-500"
                style={{ width: `${model.progress}%` }}
              />
            ) : (
              <div className="h-full bg-gradient-to-r from-purple-500 to-purple-400 animate-pulse" />
            )}
          </div>
        </div>
      )}

      {/* Extraction Progress Bar */}
      {isExtracting && (
        <div className="mt-3 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400 flex items-center gap-2">
              {model.extraction_status === 'starting' && 'Starting Extraction'}
              {model.extraction_status === 'loading' && 'Loading Model'}
              {model.extraction_status === 'extracting' && 'Extracting Activations'}
              {model.extraction_status === 'saving' && 'Saving Results'}
              {/* Elapsed time indicator */}
              {elapsedTime > 0 && (
                <span className="flex items-center gap-1 text-slate-500">
                  <Clock className="w-3 h-3" />
                  {formatElapsedTime(elapsedTime)}
                </span>
              )}
            </span>
            <span className="text-emerald-400 font-medium font-mono">
              {model.extraction_progress !== undefined && model.extraction_progress > 0
                ? `${model.extraction_progress.toFixed(1)}%`
                : 'Starting...'}
            </span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            {model.extraction_progress !== undefined && model.extraction_progress > 0 ? (
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500"
                style={{ width: `${model.extraction_progress}%` }}
              />
            ) : (
              <div className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 animate-pulse" />
            )}
          </div>
          {model.extraction_message && (
            <p className="text-xs text-slate-500">{model.extraction_message}</p>
          )}
        </div>
      )}

      {/* Extraction Complete Message */}
      {model.extraction_status === 'complete' && (
        <div className="mt-3 p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400 text-sm">
          <div className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>Extraction completed: {model.extraction_message || 'Activations saved successfully'}</span>
          </div>
        </div>
      )}

      {/* Error Message */}
      {isError && model.error_message && (
        <div className="mt-3 p-2 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>{model.error_message}</span>
          </div>
        </div>
      )}
    </div>
  );
}
