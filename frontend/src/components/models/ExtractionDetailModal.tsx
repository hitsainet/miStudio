/**
 * ExtractionDetailModal - Display detailed statistics for a selected extraction.
 *
 * Features:
 * - Comprehensive statistics view
 * - Layer-by-layer breakdown
 * - Activation ranges and sparsity
 * - Dataset and configuration info
 * - Real-time progress tracking via WebSocket (reads from modelsStore)
 */

import { Activity, Calendar, Layers, Database, HardDrive, TrendingUp, BarChart3, Loader2 } from 'lucide-react';
import { Extraction } from './ExtractionListModal';
import { useModelsStore } from '../../stores/modelsStore';

interface ExtractionDetailModalProps {
  extraction: Extraction;
  onClose: () => void;
  onBack?: () => void;  // Optional: return to list modal
}

export function ExtractionDetailModal({
  extraction,
  onClose,
  onBack
}: ExtractionDetailModalProps) {
  // Read live progress from modelsStore for real-time updates via WebSocket
  const models = useModelsStore((state) => state.models);
  const model = models.find((m) => m.id === extraction.model_id);

  // Check if this extraction is currently active (has live progress data)
  const isActiveExtraction = model?.extraction_id === extraction.extraction_id;
  const liveProgress = isActiveExtraction ? model?.extraction_progress : undefined;
  const liveStatus = isActiveExtraction ? model?.extraction_status : undefined;
  const liveMessage = isActiveExtraction ? model?.extraction_message : undefined;

  // Determine effective status and progress (live data takes precedence)
  const effectiveStatus = liveStatus || extraction.status;
  const effectiveProgress = liveProgress ?? extraction.progress ?? 0;
  const isInProgress = ['extracting', 'processing', 'saving', 'pending'].includes(effectiveStatus.toLowerCase());

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getSamplesProcessed = () => {
    return extraction.samples_processed ?? extraction.num_samples_processed ?? 0;
  };

  const getStatusBadge = (status: string) => {
    const baseClasses = 'px-3 py-1 rounded text-sm font-medium';
    const statusLower = status.toLowerCase();

    switch (statusLower) {
      case 'completed':
        return <span className={`${baseClasses} bg-emerald-900/30 text-emerald-400`}>Completed</span>;
      case 'failed':
        return <span className={`${baseClasses} bg-red-900/30 text-red-400`}>Failed</span>;
      case 'cancelled':
        return <span className={`${baseClasses} bg-yellow-900/30 text-yellow-400`}>Cancelled</span>;
      case 'extracting':
      case 'processing':
      case 'saving':
      case 'pending':
        return (
          <span className={`${baseClasses} bg-blue-900/30 text-blue-400 flex items-center gap-2`}>
            <Loader2 className="w-3 h-3 animate-spin" />
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </span>
        );
      default:
        return <span className={`${baseClasses} bg-blue-900/30 text-blue-400`}>{status}</span>;
    }
  };

  const formatBytes = (mb: number) => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(2)} GB`;
    }
    return `${mb.toFixed(2)} MB`;
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 mb-2">
              <Activity className="w-6 h-6 text-emerald-400 flex-shrink-0" />
              <h2 className="text-xl font-mono text-slate-100 truncate">
                {extraction.extraction_id}
              </h2>
              {getStatusBadge(effectiveStatus)}
            </div>
            <p className="text-sm text-slate-400">
              {isInProgress ? 'Extraction in progress...' : 'Detailed extraction statistics and configuration'}
            </p>
          </div>
          <div className="flex items-center gap-2 ml-4">
            {onBack && (
              <button
                type="button"
                onClick={onBack}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded transition-colors text-slate-300"
              >
                ← Back to List
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="text-slate-400 hover:text-slate-300 transition-colors"
              aria-label="Close"
            >
              <span className="text-2xl">×</span>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Progress Section (shown when extraction is in progress) */}
          {isInProgress && (
            <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
                  <span className="text-blue-400 font-medium">
                    {liveMessage || `${effectiveStatus.charAt(0).toUpperCase() + effectiveStatus.slice(1)}...`}
                  </span>
                </div>
                <span className="text-blue-300 font-mono text-sm">
                  {effectiveProgress.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-blue-500 to-emerald-500 h-3 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${Math.min(effectiveProgress, 100)}%` }}
                />
              </div>
              <div className="flex items-center justify-between mt-2 text-xs text-slate-400">
                <span>Started: {formatDate(extraction.created_at)}</span>
                {extraction.max_samples && (
                  <span>Target: {extraction.max_samples.toLocaleString()} samples</span>
                )}
              </div>
            </div>
          )}

          {/* Overview Section */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                <Calendar className="w-4 h-4" />
                <span>Created</span>
              </div>
              <div className="text-slate-100 font-medium">
                {formatDate(extraction.created_at)}
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                <Layers className="w-4 h-4" />
                <span>Layers</span>
              </div>
              <div className="text-slate-100 font-medium text-2xl">
                {extraction.layer_indices.length}
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                <Database className="w-4 h-4" />
                <span>Samples</span>
              </div>
              <div className="text-slate-100 font-medium text-2xl">
                {getSamplesProcessed().toLocaleString()}
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                <HardDrive className="w-4 h-4" />
                <span>Total Size</span>
              </div>
              <div className="text-slate-100 font-medium text-2xl">
                {extraction.statistics && Object.keys(extraction.statistics).length > 0
                  ? formatBytes(
                      Object.values(extraction.statistics).reduce((sum, stat) => sum + stat.size_mb, 0)
                    )
                  : 'N/A'}
              </div>
            </div>
          </div>

          {/* Configuration Section */}
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
            <h3 className="text-lg font-semibold text-slate-100 mb-3 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-emerald-400" />
              Configuration
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
              {extraction.architecture && (
                <div>
                  <div className="text-slate-400">Architecture</div>
                  <div className="text-slate-100 font-medium">{extraction.architecture}</div>
                </div>
              )}
              {extraction.quantization && (
                <div>
                  <div className="text-slate-400">Quantization</div>
                  <div className="text-slate-100 font-medium">{extraction.quantization}</div>
                </div>
              )}
              <div>
                <div className="text-slate-400">Batch Size</div>
                <div className="text-slate-100 font-medium">{extraction.batch_size || 'N/A'}</div>
              </div>
              <div>
                <div className="text-slate-400">Max Samples</div>
                <div className="text-slate-100 font-medium">{extraction.max_samples.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-slate-400">Hook Types</div>
                <div className="text-slate-100 font-medium">{extraction.hook_types.join(', ')}</div>
              </div>
              <div>
                <div className="text-slate-400">Layer Indices</div>
                <div className="text-slate-100 font-medium">{extraction.layer_indices.join(', ')}</div>
              </div>
            </div>
            {extraction.dataset_path && (
              <div className="mt-4 pt-4 border-t border-slate-700">
                <div className="text-slate-400 text-sm mb-1">Dataset Path</div>
                <div className="text-slate-100 font-mono text-xs bg-slate-900 p-2 rounded">
                  {extraction.dataset_path}
                </div>
              </div>
            )}
          </div>

          {/* Error Message (if failed) */}
          {extraction.error_message && (
            <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-red-400 mb-2">Error Details</h3>
              <div className="text-red-300 text-sm font-mono">
                {extraction.error_message}
              </div>
            </div>
          )}

          {/* Layer Statistics */}
          {extraction.statistics && Object.keys(extraction.statistics).length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-emerald-400" />
                Layer Statistics
              </h3>
              {Object.entries(extraction.statistics).map(([layerName, stats]) => (
                <div
                  key={layerName}
                  className="bg-slate-800/50 rounded-lg p-4 border border-slate-700"
                >
                  <h4 className="text-md font-semibold text-emerald-400 mb-3">
                    {layerName}
                  </h4>

                  {/* Shape Info */}
                  <div className="mb-4">
                    <div className="text-slate-400 text-sm mb-1">Shape</div>
                    <div className="text-slate-100 font-mono text-sm">
                      [{stats.shape.join(', ')}]
                    </div>
                  </div>

                  {/* Statistics Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <div className="text-slate-400 mb-1">Mean Magnitude</div>
                      <div className="text-slate-100 font-medium">
                        {stats.mean_magnitude.toFixed(6)}
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-400 mb-1">Max Activation</div>
                      <div className="text-slate-100 font-medium">
                        {stats.max_activation.toFixed(6)}
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-400 mb-1">Min Activation</div>
                      <div className="text-slate-100 font-medium">
                        {stats.min_activation.toExponential(3)}
                      </div>
                    </div>
                    {stats.std_activation !== null && (
                      <div>
                        <div className="text-slate-400 mb-1">Std Deviation</div>
                        <div className="text-slate-100 font-medium">
                          {stats.std_activation.toFixed(6)}
                        </div>
                      </div>
                    )}
                    <div>
                      <div className="text-slate-400 mb-1">Sparsity</div>
                      <div className="text-slate-100 font-medium">
                        {stats.sparsity_percent.toFixed(2)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-400 mb-1">File Size</div>
                      <div className="text-slate-100 font-medium">
                        {formatBytes(stats.size_mb)}
                      </div>
                    </div>
                  </div>

                  {/* Activation Range Visualization */}
                  <div className="mt-4">
                    <div className="text-slate-400 text-sm mb-2">Activation Range</div>
                    <div className="relative h-8 bg-slate-900 rounded overflow-hidden">
                      <div
                        className="absolute inset-y-0 bg-gradient-to-r from-blue-500/30 to-emerald-500/30"
                        style={{
                          left: '0%',
                          width: '100%'
                        }}
                      ></div>
                      <div className="absolute inset-0 flex items-center justify-between px-2 text-xs text-slate-300">
                        <span>{stats.min_activation.toExponential(2)}</span>
                        <span>{stats.max_activation.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Sparsity Bar */}
                  <div className="mt-4">
                    <div className="flex items-center justify-between text-sm mb-2">
                      <span className="text-slate-400">Sparsity Level</span>
                      <span className="text-slate-100">{stats.sparsity_percent.toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-2">
                      <div
                        className="bg-emerald-500 h-2 rounded-full"
                        style={{ width: `${Math.min(stats.sparsity_percent, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Saved Files */}
          {extraction.saved_files && extraction.saved_files.length > 0 && (
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <h3 className="text-lg font-semibold text-slate-100 mb-3 flex items-center gap-2">
                <HardDrive className="w-5 h-5 text-emerald-400" />
                Saved Files ({extraction.saved_files.length})
              </h3>
              <div className="space-y-2">
                {extraction.saved_files.map((file, idx) => (
                  <div
                    key={idx}
                    className="bg-slate-900 p-3 rounded font-mono text-sm text-slate-100"
                  >
                    {file}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Timing Info */}
          {extraction.completed_at && (
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <h3 className="text-lg font-semibold text-slate-100 mb-3">Timing</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-slate-400 mb-1">Started</div>
                  <div className="text-slate-100">{formatDate(extraction.created_at)}</div>
                </div>
                <div>
                  <div className="text-slate-400 mb-1">Completed</div>
                  <div className="text-slate-100">{formatDate(extraction.completed_at)}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-800 p-4 bg-slate-900/50">
          <div className="flex items-center justify-between">
            <div className="text-sm text-slate-400">
              Extraction ID: <span className="font-mono text-slate-300">{extraction.extraction_id}</span>
            </div>
            <div className="flex gap-2">
              {onBack && (
                <button
                  onClick={onBack}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded transition-colors text-slate-300"
                >
                  Back to List
                </button>
              )}
              <button
                onClick={onClose}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded transition-colors text-white"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
