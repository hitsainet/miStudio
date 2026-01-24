/**
 * PushToNeuronpedia - Modal component for pushing SAE features to a local Neuronpedia instance.
 *
 * Features:
 * - Configuration form for push options
 * - Real-time status checking for Neuronpedia connection
 * - Real-time progress display with WebSocket updates
 * - Progress bar, feature counts, elapsed time, and ETA
 * - Success view with link to browse features in Neuronpedia
 */

import { useState, useEffect } from 'react';
import {
  X,
  Send,
  Settings,
  MessageSquare,
  Activity,
  Loader,
  CheckCircle,
  AlertCircle,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Globe,
  Database,
  Clock,
  Zap,
} from 'lucide-react';
import { SAE } from '../../types/sae';
import { COMPONENTS } from '../../config/brand';
import {
  LocalPushConfig,
  LocalNeuronpediaStatus,
  getLocalStatus,
  pushToLocal,
} from '../../api/neuronpedia';
import { formatDuration } from '../../api/neuronpedia';
import { useNeuronpediaPushWebSocket } from '../../hooks/useNeuronpediaPushWebSocket';

interface PushToNeuronpediaProps {
  sae: SAE;
  isOpen: boolean;
  onClose: () => void;
}

export function PushToNeuronpedia({ sae, isOpen, onClose }: PushToNeuronpediaProps) {
  const [status, setStatus] = useState<LocalNeuronpediaStatus | null>(null);
  const [isCheckingStatus, setIsCheckingStatus] = useState(true);
  const [isPushing, setIsPushing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [view, setView] = useState<'config' | 'pushing' | 'complete'>('config');
  const [pushJobId, setPushJobId] = useState<string | null>(null);

  // WebSocket hook for real-time progress
  const { progress, isComplete, error: wsError, reset: resetWs } = useNeuronpediaPushWebSocket(pushJobId);

  // Configuration state
  const [config, setConfig] = useState<LocalPushConfig>({
    includeActivations: true,
    includeExplanations: true,
    maxActivationsPerFeature: 20,
  });

  // Check Neuronpedia connection status on mount
  useEffect(() => {
    if (isOpen) {
      checkStatus();
    }
  }, [isOpen]);

  // Handle WebSocket completion/error
  useEffect(() => {
    if (isComplete && progress) {
      if (progress.status === 'completed') {
        setView('complete');
        setIsPushing(false);
      } else if (progress.status === 'failed') {
        setError(wsError || progress.error || 'Push failed');
        setView('config');
        setIsPushing(false);
      }
    }
  }, [isComplete, progress, wsError]);

  const checkStatus = async () => {
    setIsCheckingStatus(true);
    setError(null);
    try {
      const statusResult = await getLocalStatus();
      setStatus(statusResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check Neuronpedia status');
    } finally {
      setIsCheckingStatus(false);
    }
  };

  const handlePush = async () => {
    setIsPushing(true);
    setError(null);
    setView('pushing');
    resetWs();

    try {
      const pushResponse = await pushToLocal(sae.id, config);
      setPushJobId(pushResponse.pushJobId);
      // The WebSocket hook will now receive progress updates
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Push failed');
      setView('config');
      setIsPushing(false);
    }
  };

  const handleClose = () => {
    setPushJobId(null);
    resetWs();
    setError(null);
    setView('config');
    onClose();
  };

  if (!isOpen) return null;

  const isReady = status?.configured && status?.connected;

  // Calculate progress percentage
  const progressPercent = progress?.progress ?? 0;
  const featuresTotal = progress?.total_features ?? sae.n_features ?? 0;
  const featuresPushed = progress?.features_pushed ?? 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={handleClose} />

      {/* Modal */}
      <div className="relative w-full max-w-2xl mx-4 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <Send className="w-6 h-6 text-emerald-400" />
            <div>
              <h2 className="text-lg font-semibold text-slate-100">Push to Neuronpedia</h2>
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
              {error}
            </div>
          )}

          {/* Config View */}
          {view === 'config' && (
            <div className="space-y-6">
              {/* Connection Status */}
              <div className="p-4 bg-slate-800/50 rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-slate-300 flex items-center gap-2">
                    <Database className="w-4 h-4" />
                    Local Neuronpedia Status
                  </h3>
                  <button
                    onClick={checkStatus}
                    disabled={isCheckingStatus}
                    className="text-xs text-slate-400 hover:text-slate-300 flex items-center gap-1"
                  >
                    {isCheckingStatus ? (
                      <Loader className="w-3 h-3 animate-spin" />
                    ) : (
                      'Refresh'
                    )}
                  </button>
                </div>

                {isCheckingStatus ? (
                  <div className="flex items-center gap-2 text-slate-400">
                    <Loader className="w-4 h-4 animate-spin" />
                    Checking connection...
                  </div>
                ) : status ? (
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      {status.configured ? (
                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-red-400" />
                      )}
                      <span className={status.configured ? 'text-emerald-400' : 'text-red-400'}>
                        {status.configured ? 'Configured' : 'Not Configured'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {status.connected ? (
                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-red-400" />
                      )}
                      <span className={status.connected ? 'text-emerald-400' : 'text-red-400'}>
                        {status.connected ? 'Connected' : 'Not Connected'}
                      </span>
                    </div>
                    {status.publicUrl && (
                      <div className="flex items-center gap-2 text-slate-400">
                        <Globe className="w-4 h-4" />
                        <a
                          href={status.publicUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hover:text-emerald-400 underline"
                        >
                          {status.publicUrl}
                        </a>
                      </div>
                    )}
                    {status.error && (
                      <div className="text-red-400 text-xs mt-2">{status.error}</div>
                    )}
                  </div>
                ) : (
                  <div className="text-red-400">Failed to check status</div>
                )}
              </div>

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

              {/* Push Options */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-slate-300">Push Options</h3>

                <label className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg cursor-pointer hover:bg-slate-800 transition-colors">
                  <input
                    type="checkbox"
                    checked={config.includeActivations}
                    onChange={(e) => setConfig({ ...config, includeActivations: e.target.checked })}
                    className="w-4 h-4 text-emerald-500 bg-slate-700 border-slate-600 rounded focus:ring-emerald-500"
                  />
                  <Activity className="w-4 h-4 text-blue-400" />
                  <div className="flex-1">
                    <span className="text-slate-200">Include Activations</span>
                    <p className="text-xs text-slate-500">Push activation examples for each feature</p>
                  </div>
                </label>

                <label className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg cursor-pointer hover:bg-slate-800 transition-colors">
                  <input
                    type="checkbox"
                    checked={config.includeExplanations}
                    onChange={(e) => setConfig({ ...config, includeExplanations: e.target.checked })}
                    className="w-4 h-4 text-emerald-500 bg-slate-700 border-slate-600 rounded focus:ring-emerald-500"
                  />
                  <MessageSquare className="w-4 h-4 text-purple-400" />
                  <div className="flex-1">
                    <span className="text-slate-200">Include Explanations</span>
                    <p className="text-xs text-slate-500">Push feature labels and descriptions</p>
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
                  <Settings className="w-4 h-4" />
                  Advanced Options
                </button>

                {showAdvanced && (
                  <div className="mt-3 p-4 bg-slate-800/50 rounded-lg space-y-4">
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">
                        Max Activations per Feature
                      </label>
                      <input
                        type="number"
                        min={1}
                        max={100}
                        value={config.maxActivationsPerFeature}
                        onChange={(e) => setConfig({ ...config, maxActivationsPerFeature: parseInt(e.target.value) || 20 })}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100"
                      />
                      <p className="text-xs text-slate-500 mt-1">
                        Limit the number of activation examples per feature (1-100)
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Warning if not configured */}
              {!isReady && !isCheckingStatus && (
                <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <h4 className="text-sm font-medium text-yellow-300 mb-2 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    Configuration Required
                  </h4>
                  <p className="text-sm text-slate-300">
                    Local Neuronpedia is not configured or not connected. Please ensure:
                  </p>
                  <ul className="text-sm text-slate-400 mt-2 list-disc list-inside space-y-1">
                    <li>NEURONPEDIA_LOCAL_DB_URL is set in the backend configuration</li>
                    <li>The Neuronpedia PostgreSQL database is accessible</li>
                    <li>The Neuronpedia webapp is running</li>
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Pushing View with Real-time Progress */}
          {view === 'pushing' && (
            <div className="space-y-6">
              {/* Progress Header */}
              <div className="text-center py-4">
                <Loader className="w-12 h-12 text-emerald-400 animate-spin mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-100">Pushing to Neuronpedia</h3>
                <p className="text-slate-400 mt-1">
                  {progress?.message || 'Initializing push...'}
                </p>
              </div>

              {/* Progress Bar */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Progress</span>
                  <span className="text-emerald-400 font-medium">{Math.round(progressPercent)}%</span>
                </div>
                <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300 ease-out"
                    style={{ width: `${Math.min(progressPercent, 100)}%` }}
                  />
                </div>
              </div>

              {/* Statistics Grid */}
              <div className="grid grid-cols-2 gap-4">
                {/* Features */}
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                    <Zap className="w-4 h-4" />
                    Features
                  </div>
                  <div className="text-xl font-semibold text-slate-100">
                    {featuresPushed.toLocaleString()} / {featuresTotal.toLocaleString()}
                  </div>
                </div>

                {/* Activations */}
                {config.includeActivations && (
                  <div className="p-4 bg-slate-800/50 rounded-lg">
                    <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                      <Activity className="w-4 h-4" />
                      Activations
                    </div>
                    <div className="text-xl font-semibold text-slate-100">
                      {(progress?.activations_pushed ?? 0).toLocaleString()}
                    </div>
                  </div>
                )}

                {/* Elapsed Time */}
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                    <Clock className="w-4 h-4" />
                    Elapsed
                  </div>
                  <div className="text-xl font-semibold text-slate-100">
                    {progress?.elapsed_seconds ? formatDuration(progress.elapsed_seconds) : '-'}
                  </div>
                </div>

                {/* ETA */}
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                    <Clock className="w-4 h-4" />
                    Remaining
                  </div>
                  <div className="text-xl font-semibold text-slate-100">
                    {progress?.eta_seconds ? formatDuration(progress.eta_seconds) : '-'}
                  </div>
                </div>
              </div>

              {/* Stage Indicator */}
              <div className="p-4 bg-slate-800/50 rounded-lg">
                <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                  Current Stage
                </div>
                <div className="text-sm text-slate-200 capitalize">
                  {(progress?.stage || 'initializing').replace(/_/g, ' ')}
                </div>
              </div>
            </div>
          )}

          {/* Complete View */}
          {view === 'complete' && progress && (
            <div className="space-y-6">
              <div className="text-center py-4">
                <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-100">Push Complete!</h3>
                <p className="text-slate-400">Your features are now available in Neuronpedia</p>
              </div>

              {/* Push Summary */}
              <div className="p-4 bg-slate-800/50 rounded-lg">
                <h4 className="text-sm font-medium text-slate-300 mb-3">Push Summary</h4>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-slate-500">Features Pushed:</span>
                    <span className="ml-2 text-slate-200">{(progress.features_pushed ?? 0).toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Activations Created:</span>
                    <span className="ml-2 text-slate-200">{(progress.activations_pushed ?? 0).toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Explanations Created:</span>
                    <span className="ml-2 text-slate-200">{(progress.explanations_pushed ?? 0).toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Total Time:</span>
                    <span className="ml-2 text-slate-200">
                      {progress.elapsed_seconds ? formatDuration(progress.elapsed_seconds) : '-'}
                    </span>
                  </div>
                </div>
              </div>

              {/* View in Neuronpedia */}
              {status?.publicUrl && (
                <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                  <h4 className="text-sm font-medium text-emerald-300 mb-2">View in Neuronpedia</h4>
                  <p className="text-sm text-slate-300 mb-3">
                    Your features are now available for browsing in the local Neuronpedia instance.
                  </p>
                  <a
                    href={status.publicUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`inline-flex items-center gap-2 ${COMPONENTS.button.primary}`}
                  >
                    <ExternalLink className="w-4 h-4" />
                    Open in Neuronpedia
                  </a>
                </div>
              )}
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
                onClick={handlePush}
                disabled={!isReady || isPushing}
                className={`flex items-center gap-2 ${COMPONENTS.button.primary} ${
                  (!isReady || isPushing) ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isPushing ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    Push to Neuronpedia
                  </>
                )}
              </button>
            </>
          )}

          {view === 'pushing' && (
            <button onClick={handleClose} className={COMPONENTS.button.ghost}>
              Run in Background
            </button>
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
