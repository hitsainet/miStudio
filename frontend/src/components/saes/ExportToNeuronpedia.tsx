/**
 * ExportToNeuronpedia - Modal component for exporting SAEs to Neuronpedia format.
 *
 * Features:
 * - Configuration form for export options
 * - Real-time progress tracking during export
 * - Download completed export archives
 * - Export history for the SAE
 */

import { useState, useEffect } from 'react';
import {
  X,
  Download,
  Settings,
  FileJson,
  BarChart3,
  Hash,
  MessageSquare,
  Loader,
  CheckCircle,
  AlertCircle,
  ExternalLink,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { useNeuronpediaExportStore, selectIsExportInProgress, selectIsExportComplete, selectProgress, selectCurrentStage } from '../../stores/neuronpediaExportStore';
import { SAE } from '../../types/sae';
import { COMPONENTS } from '../../config/brand';
import { formatFileSize, formatDuration } from '../../api/neuronpedia';

interface ExportToNeuronpediaProps {
  sae: SAE;
  isOpen: boolean;
  onClose: () => void;
}

export function ExportToNeuronpedia({ sae, isOpen, onClose }: ExportToNeuronpediaProps) {
  const {
    config,
    currentJob,
    exportJobs,
    isExporting,
    error,
    setConfig,
    startExport,
    cancelExport,
    downloadExport,
    closeExportDialog,
  } = useNeuronpediaExportStore();

  const isInProgress = useNeuronpediaExportStore(selectIsExportInProgress);
  const isComplete = useNeuronpediaExportStore(selectIsExportComplete);
  const progress = useNeuronpediaExportStore(selectProgress);
  const currentStage = useNeuronpediaExportStore(selectCurrentStage);

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [view, setView] = useState<'config' | 'progress' | 'complete'>('config');

  // Update view based on job status
  useEffect(() => {
    if (!currentJob) {
      setView('config');
    } else if (isInProgress) {
      setView('progress');
    } else if (isComplete) {
      setView('complete');
    }
  }, [currentJob, isInProgress, isComplete]);

  const handleClose = () => {
    closeExportDialog();
    onClose();
  };

  const handleStartExport = async () => {
    await startExport();
  };

  const handleDownload = async () => {
    if (currentJob) {
      await downloadExport(currentJob.id, `${sae.name}-neuronpedia-export.zip`);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={handleClose} />

      {/* Modal */}
      <div className="relative w-full max-w-2xl mx-4 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <FileJson className="w-6 h-6 text-purple-400" />
            <div>
              <h2 className="text-lg font-semibold text-slate-100">Export to Neuronpedia</h2>
              <p className="text-sm text-slate-400">{sae.name}</p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="p-2 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {error && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm flex items-center gap-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {typeof error === 'string' ? error : JSON.stringify(error)}
            </div>
          )}

          {/* Config View */}
          {view === 'config' && (
            <div className="space-y-6">
              {/* SAE Info */}
              <div className="p-4 bg-slate-800/50 rounded-lg">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-slate-500">Features:</span>
                    <span className="ml-2 text-slate-200">{sae.n_features?.toLocaleString() ?? 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Layer:</span>
                    <span className="ml-2 text-slate-200">{sae.layer ?? 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Model:</span>
                    <span className="ml-2 text-slate-200">{sae.model_name ?? 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Source:</span>
                    <span className="ml-2 text-slate-200">{sae.source}</span>
                  </div>
                </div>
              </div>

              {/* Feature Selection */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Feature Selection
                </label>
                <select
                  value={config.featureSelection}
                  onChange={(e) => setConfig({ featureSelection: e.target.value as 'all' | 'extracted' | 'custom' })}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100"
                >
                  <option value="all">All Features</option>
                  <option value="extracted">Features with Extracted Activations</option>
                  <option value="custom">Custom Selection</option>
                </select>
              </div>

              {/* Dashboard Data Options */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-slate-300">Dashboard Data</h3>

                <label className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg cursor-pointer hover:bg-slate-800 transition-colors">
                  <input
                    type="checkbox"
                    checked={config.includeLogitLens}
                    onChange={(e) => setConfig({ includeLogitLens: e.target.checked })}
                    className="w-4 h-4 text-emerald-500 bg-slate-700 border-slate-600 rounded focus:ring-emerald-500"
                  />
                  <Hash className="w-4 h-4 text-blue-400" />
                  <div className="flex-1">
                    <span className="text-slate-200">Logit Lens Data</span>
                    <p className="text-xs text-slate-500">Top positive/negative tokens for each feature</p>
                  </div>
                </label>

                <label className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg cursor-pointer hover:bg-slate-800 transition-colors">
                  <input
                    type="checkbox"
                    checked={config.includeHistograms}
                    onChange={(e) => setConfig({ includeHistograms: e.target.checked })}
                    className="w-4 h-4 text-emerald-500 bg-slate-700 border-slate-600 rounded focus:ring-emerald-500"
                  />
                  <BarChart3 className="w-4 h-4 text-green-400" />
                  <div className="flex-1">
                    <span className="text-slate-200">Activation Histograms</span>
                    <p className="text-xs text-slate-500">Distribution of activation values</p>
                  </div>
                </label>

                <label className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg cursor-pointer hover:bg-slate-800 transition-colors">
                  <input
                    type="checkbox"
                    checked={config.includeTopTokens}
                    onChange={(e) => setConfig({ includeTopTokens: e.target.checked })}
                    className="w-4 h-4 text-emerald-500 bg-slate-700 border-slate-600 rounded focus:ring-emerald-500"
                  />
                  <Hash className="w-4 h-4 text-orange-400" />
                  <div className="flex-1">
                    <span className="text-slate-200">Top Activating Tokens</span>
                    <p className="text-xs text-slate-500">Aggregated tokens across examples</p>
                  </div>
                </label>

                <label className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg cursor-pointer hover:bg-slate-800 transition-colors">
                  <input
                    type="checkbox"
                    checked={config.includeExplanations}
                    onChange={(e) => setConfig({ includeExplanations: e.target.checked })}
                    className="w-4 h-4 text-emerald-500 bg-slate-700 border-slate-600 rounded focus:ring-emerald-500"
                  />
                  <MessageSquare className="w-4 h-4 text-purple-400" />
                  <div className="flex-1">
                    <span className="text-slate-200">Feature Explanations</span>
                    <p className="text-xs text-slate-500">Labels and descriptions</p>
                  </div>
                </label>

                <label className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg cursor-pointer hover:bg-slate-800 transition-colors">
                  <input
                    type="checkbox"
                    checked={config.includeSaelensFormat}
                    onChange={(e) => setConfig({ includeSaelensFormat: e.target.checked })}
                    className="w-4 h-4 text-emerald-500 bg-slate-700 border-slate-600 rounded focus:ring-emerald-500"
                  />
                  <Settings className="w-4 h-4 text-yellow-400" />
                  <div className="flex-1">
                    <span className="text-slate-200">SAELens Format</span>
                    <p className="text-xs text-slate-500">Include cfg.json and weights for SAELens compatibility</p>
                  </div>
                </label>
              </div>

              {/* Advanced Options */}
              <div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-sm text-slate-400 hover:text-slate-300 transition-colors"
                >
                  {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  Advanced Options
                </button>

                {showAdvanced && (
                  <div className="mt-3 p-4 bg-slate-800/50 rounded-lg space-y-4">
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">
                        Logit Lens K (top tokens per feature)
                      </label>
                      <input
                        type="number"
                        min={10}
                        max={50}
                        value={config.logitLensK}
                        onChange={(e) => setConfig({ logitLensK: parseInt(e.target.value) || 20 })}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">
                        Histogram Bins
                      </label>
                      <input
                        type="number"
                        min={20}
                        max={100}
                        value={config.histogramBins}
                        onChange={(e) => setConfig({ histogramBins: parseInt(e.target.value) || 50 })}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">
                        Top Tokens K (per feature)
                      </label>
                      <input
                        type="number"
                        min={20}
                        max={100}
                        value={config.topTokensK}
                        onChange={(e) => setConfig({ topTokensK: parseInt(e.target.value) || 50 })}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Previous Exports */}
              {exportJobs.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-slate-300 mb-2">Previous Exports</h3>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {exportJobs.slice(0, 5).map((job) => (
                      <div
                        key={job.id}
                        className="flex items-center justify-between p-2 bg-slate-800/50 rounded-lg text-sm"
                      >
                        <div className="flex items-center gap-2">
                          {job.status === 'completed' && <CheckCircle className="w-4 h-4 text-emerald-400" />}
                          {job.status === 'failed' && <AlertCircle className="w-4 h-4 text-red-400" />}
                          {['pending', 'computing', 'packaging'].includes(job.status) && (
                            <Loader className="w-4 h-4 text-blue-400 animate-spin" />
                          )}
                          <span className="text-slate-300">
                            {new Date(job.createdAt).toLocaleDateString()}
                          </span>
                          <span className="text-slate-500">
                            {job.featureCount?.toLocaleString()} features
                          </span>
                        </div>
                        {job.status === 'completed' && job.fileSizeBytes && (
                          <button
                            onClick={() => downloadExport(job.id)}
                            className="text-emerald-400 hover:text-emerald-300"
                          >
                            <Download className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Progress View */}
          {view === 'progress' && currentJob && (
            <div className="space-y-6">
              <div className="text-center py-4">
                <Loader className="w-12 h-12 text-purple-400 animate-spin mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-100">Exporting to Neuronpedia</h3>
                <p className="text-slate-400">{currentStage || 'Initializing...'}</p>
              </div>

              {/* Progress Bar */}
              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-slate-400">Overall Progress</span>
                  <span className="text-emerald-400 font-medium">{Math.round(progress)}%</span>
                </div>
                <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-emerald-500 transition-all duration-500"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>

              {/* Stage List */}
              <div className="space-y-2 text-sm">
                {['Computing logit lens', 'Computing histograms', 'Computing top tokens', 'Generating JSON files', 'Creating archive'].map((stage, idx) => {
                  const stageProgress = progress / 100;
                  const stageThreshold = idx * 0.2;
                  const isComplete = stageProgress > stageThreshold + 0.2;
                  const isCurrent = stageProgress > stageThreshold && stageProgress <= stageThreshold + 0.2;

                  return (
                    <div key={stage} className="flex items-center gap-3">
                      {isComplete && <CheckCircle className="w-4 h-4 text-emerald-400" />}
                      {isCurrent && <Loader className="w-4 h-4 text-blue-400 animate-spin" />}
                      {!isComplete && !isCurrent && <div className="w-4 h-4 rounded-full border border-slate-600" />}
                      <span className={isComplete ? 'text-slate-300' : isCurrent ? 'text-slate-200' : 'text-slate-500'}>
                        {stage}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Complete View */}
          {view === 'complete' && currentJob && (
            <div className="space-y-6">
              <div className="text-center py-4">
                <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-100">Export Complete!</h3>
                <p className="text-slate-400">Your SAE data is ready for Neuronpedia</p>
              </div>

              {/* Export Summary */}
              <div className="p-4 bg-slate-800/50 rounded-lg">
                <h4 className="text-sm font-medium text-slate-300 mb-3">Export Summary</h4>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-slate-500">Features Exported:</span>
                    <span className="ml-2 text-slate-200">{currentJob.featureCount?.toLocaleString()}</span>
                  </div>
                  {currentJob.fileSizeBytes && (
                    <div>
                      <span className="text-slate-500">Archive Size:</span>
                      <span className="ml-2 text-slate-200">{formatFileSize(currentJob.fileSizeBytes)}</span>
                    </div>
                  )}
                  {currentJob.startedAt && currentJob.completedAt && (
                    <div>
                      <span className="text-slate-500">Duration:</span>
                      <span className="ml-2 text-slate-200">
                        {formatDuration(
                          (new Date(currentJob.completedAt).getTime() - new Date(currentJob.startedAt).getTime()) / 1000
                        )}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Next Steps */}
              <div className="p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <h4 className="text-sm font-medium text-purple-300 mb-2">Next Steps</h4>
                <ol className="text-sm text-slate-300 space-y-2 list-decimal list-inside">
                  <li>Download the export archive</li>
                  <li>Visit Neuronpedia's upload form</li>
                  <li>Upload the archive and fill in metadata</li>
                  <li>Coordinate with the Neuronpedia team for hosting</li>
                </ol>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col gap-3">
                <button
                  onClick={handleDownload}
                  className={`w-full flex items-center justify-center gap-2 ${COMPONENTS.button.primary}`}
                >
                  <Download className="w-5 h-5" />
                  Download Archive
                </button>
                <a
                  href="https://www.neuronpedia.org/contribute"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`w-full flex items-center justify-center gap-2 ${COMPONENTS.button.ghost}`}
                >
                  <ExternalLink className="w-4 h-4" />
                  Open Neuronpedia Upload Form
                </a>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-slate-800">
          {view === 'config' && (
            <>
              <button onClick={handleClose} className={COMPONENTS.button.ghost}>
                Cancel
              </button>
              <button
                onClick={handleStartExport}
                disabled={isExporting}
                className={`flex items-center gap-2 ${COMPONENTS.button.primary}`}
              >
                {isExporting ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <FileJson className="w-4 h-4" />
                    Start Export
                  </>
                )}
              </button>
            </>
          )}

          {view === 'progress' && (
            <>
              <button onClick={cancelExport} className={COMPONENTS.button.ghost}>
                Cancel Export
              </button>
              <button onClick={handleClose} className={COMPONENTS.button.ghost}>
                Run in Background
              </button>
            </>
          )}

          {view === 'complete' && (
            <button onClick={handleClose} className={COMPONENTS.button.ghost}>
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
