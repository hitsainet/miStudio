/**
 * SAECard - Card component displaying SAE information and status.
 *
 * Features:
 * - SAE name, model, layer, and feature count info
 * - Real-time progress bars for download/conversion
 * - Status indicators (ready, downloading, converting, error)
 * - Source badge (HuggingFace, trained, file)
 * - Action buttons (use in steering, delete, upload)
 */

import {
  Brain,
  CheckCircle,
  Loader,
  Activity,
  AlertCircle,
  Trash2,
  Upload,
  Play,
  HardDrive,
  Cloud,
  Layers,
  FileJson,
} from 'lucide-react';
import { SAE, SAESource, SAEStatus } from '../../types/sae';
import { COMPONENTS } from '../../config/brand';

interface SAECardProps {
  sae: SAE;
  onSelect?: () => void;
  onUseSteering?: () => void;
  onUpload?: () => void;
  onExport?: () => void;
  onDelete: (id: string) => void;
  onCancel?: (id: string) => void;
  isSelected?: boolean;
}

export function SAECard({
  sae,
  onSelect,
  onUseSteering,
  onUpload,
  onExport,
  onDelete,
  onCancel,
  isSelected = false,
}: SAECardProps) {
  const isActive =
    sae.status === SAEStatus.DOWNLOADING ||
    sae.status === SAEStatus.CONVERTING ||
    sae.status === SAEStatus.PENDING;

  const isReady = sae.status === SAEStatus.READY;
  const isError = sae.status === SAEStatus.ERROR;

  const formatFeatureCount = (count: number): string => {
    if (count >= 1_000_000) {
      return `${(count / 1_000_000).toFixed(1)}M`;
    } else if (count >= 1_000) {
      return `${(count / 1_000).toFixed(0)}K`;
    }
    return count.toString();
  };

  const formatFileSize = (bytes?: number | null): string => {
    if (!bytes) return 'N/A';
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${Math.round(mb)} MB`;
  };

  const getStatusIcon = () => {
    switch (sae.status) {
      case SAEStatus.READY:
        return <CheckCircle className="w-5 h-5 text-emerald-400" />;
      case SAEStatus.DOWNLOADING:
        return <Loader className="w-5 h-5 text-blue-400 animate-spin" />;
      case SAEStatus.CONVERTING:
        return <Activity className="w-5 h-5 text-yellow-400 animate-pulse" />;
      case SAEStatus.PENDING:
        return <Loader className="w-5 h-5 text-slate-400 animate-spin" />;
      case SAEStatus.ERROR:
        return <AlertCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Brain className="w-5 h-5 text-slate-400" />;
    }
  };

  const getStatusBadge = () => {
    const baseClasses = 'px-3 py-1 rounded-full text-sm font-medium';

    switch (sae.status) {
      case SAEStatus.READY:
        return <span className={`${baseClasses} bg-emerald-900/30 text-emerald-400`}>Ready</span>;
      case SAEStatus.DOWNLOADING:
        return <span className={`${baseClasses} bg-blue-900/30 text-blue-400`}>Downloading</span>;
      case SAEStatus.CONVERTING:
        return <span className={`${baseClasses} bg-yellow-900/30 text-yellow-400`}>Converting</span>;
      case SAEStatus.PENDING:
        return <span className={`${baseClasses} bg-slate-800 text-slate-300`}>Pending</span>;
      case SAEStatus.ERROR:
        return <span className={`${baseClasses} bg-red-900/30 text-red-400`}>Error</span>;
      default:
        return <span className={`${baseClasses} bg-slate-800 text-slate-300`}>Unknown</span>;
    }
  };

  const getSourceBadge = () => {
    const baseClasses = 'px-2 py-0.5 rounded text-xs font-medium flex items-center gap-1';

    switch (sae.source) {
      case SAESource.HUGGINGFACE:
        return (
          <span className={`${baseClasses} bg-yellow-900/30 text-yellow-400`}>
            <Cloud className="w-3 h-3" />
            HuggingFace
          </span>
        );
      case SAESource.TRAINED:
        return (
          <span className={`${baseClasses} bg-purple-900/30 text-purple-400`}>
            <Layers className="w-3 h-3" />
            Trained
          </span>
        );
      case SAESource.LOCAL:
        return (
          <span className={`${baseClasses} bg-slate-700 text-slate-300`}>
            <HardDrive className="w-3 h-3" />
            Local File
          </span>
        );
      default:
        return null;
    }
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Are you sure you want to delete ${sae.name}? This will delete the local files.`)) {
      onDelete(sae.id);
    }
  };

  const handleCancel = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Are you sure you want to cancel this download?`)) {
      onCancel?.(sae.id);
    }
  };

  const handleUseSteering = (e: React.MouseEvent) => {
    e.stopPropagation();
    onUseSteering?.();
  };

  const handleUpload = (e: React.MouseEvent) => {
    e.stopPropagation();
    onUpload?.();
  };

  const handleExport = (e: React.MouseEvent) => {
    e.stopPropagation();
    onExport?.();
  };

  return (
    <div
      className={`${COMPONENTS.card.base} p-4 transition-colors ${
        isSelected ? 'border-emerald-500 bg-emerald-500/5' : 'hover:border-slate-700'
      } ${onSelect ? 'cursor-pointer' : ''}`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        {/* SAE Info */}
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <Brain className="w-6 h-6 text-purple-400 flex-shrink-0" />
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-base text-slate-100 truncate">{sae.name}</h3>
              {getSourceBadge()}
            </div>
            <p className="text-sm text-slate-400 mt-0.5">
              {sae.n_features ? formatFeatureCount(sae.n_features) : '?'} features
              {sae.layer !== undefined && sae.layer !== null && ` • Layer ${sae.layer}`}
              {sae.model_name && ` • ${sae.model_name}`}
              {sae.file_size_bytes && ` • ${formatFileSize(sae.file_size_bytes)}`}
            </p>
            {sae.hf_repo_id && (
              <p className="text-xs text-slate-500 mt-1 font-mono truncate">{sae.hf_repo_id}</p>
            )}
            {sae.description && (
              <p className="text-xs text-slate-500 mt-1 truncate">{sae.description}</p>
            )}
          </div>
        </div>

        {/* Actions & Status */}
        <div className="flex items-center gap-3 flex-shrink-0">
          {isReady && (
            <>
              {onUseSteering && (
                <button
                  type="button"
                  onClick={handleUseSteering}
                  className={`text-sm flex items-center gap-2 ${COMPONENTS.button.primary}`}
                >
                  <Play className="w-4 h-4" />
                  Use in Steering
                </button>
              )}
              {onUpload && sae.source !== SAESource.HUGGINGFACE && (
                <button
                  type="button"
                  onClick={handleUpload}
                  className={`p-2 rounded-lg ${COMPONENTS.button.ghost}`}
                  title="Upload to HuggingFace"
                >
                  <Upload className="w-5 h-5" />
                </button>
              )}
              {onExport && (
                <button
                  type="button"
                  onClick={handleExport}
                  className={`p-2 rounded-lg ${COMPONENTS.button.ghost}`}
                  title="Export to Neuronpedia"
                >
                  <FileJson className="w-5 h-5" />
                </button>
              )}
            </>
          )}

          {isActive && onCancel && (
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
              title="Delete SAE"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}

          {getStatusIcon()}
          {getStatusBadge()}
        </div>
      </div>

      {/* Progress Bar for Active Downloads/Conversions */}
      {isActive && sae.progress !== undefined && (
        <div className="mt-3 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">
              {sae.status === SAEStatus.DOWNLOADING && 'Download Progress'}
              {sae.status === SAEStatus.CONVERTING && 'Converting SAE'}
              {sae.status === SAEStatus.PENDING && 'Pending'}
            </span>
            <span className="text-emerald-400 font-medium font-mono">
              {sae.progress > 0 ? `${sae.progress.toFixed(1)}%` : 'Starting...'}
            </span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            {sae.progress > 0 ? (
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-500"
                style={{ width: `${sae.progress}%` }}
              />
            ) : (
              <div className="h-full bg-gradient-to-r from-purple-500 to-purple-400 animate-pulse" />
            )}
          </div>
        </div>
      )}

      {/* Error Message */}
      {isError && sae.error_message && (
        <div className="mt-3 p-2 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>{sae.error_message}</span>
          </div>
        </div>
      )}
    </div>
  );
}
