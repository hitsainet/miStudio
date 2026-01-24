/**
 * PushToNeuronpedia - Modal component for pushing SAE features to a local Neuronpedia instance.
 *
 * Features:
 * - Configuration form for push options
 * - Real-time status checking for Neuronpedia connection
 * - Progress display during push operation
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
} from 'lucide-react';
import { SAE } from '../../types/sae';
import { COMPONENTS } from '../../config/brand';
import {
  LocalPushConfig,
  LocalPushResult,
  LocalNeuronpediaStatus,
  getLocalStatus,
  pushToLocal,
} from '../../api/neuronpedia';

interface PushToNeuronpediaProps {
  sae: SAE;
  isOpen: boolean;
  onClose: () => void;
}

export function PushToNeuronpedia({ sae, isOpen, onClose }: PushToNeuronpediaProps) {
  const [status, setStatus] = useState<LocalNeuronpediaStatus | null>(null);
  const [isCheckingStatus, setIsCheckingStatus] = useState(true);
  const [isPushing, setIsPushing] = useState(false);
  const [result, setResult] = useState<LocalPushResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [view, setView] = useState<'config' | 'pushing' | 'complete'>('config');

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

    try {
      const pushResult = await pushToLocal(sae.id, config);
      setResult(pushResult);
      setView('complete');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Push failed');
      setView('config');
    } finally {
      setIsPushing(false);
    }
  };

  const handleClose = () => {
    setResult(null);
    setError(null);
    setView('config');
    onClose();
  };

  if (!isOpen) return null;

  const isReady = status?.configured && status?.connected;

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

          {/* Pushing View */}
          {view === 'pushing' && (
            <div className="space-y-6">
              <div className="text-center py-8">
                <Loader className="w-16 h-16 text-emerald-400 animate-spin mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-100">Pushing to Neuronpedia</h3>
                <p className="text-slate-400 mt-2">
                  This may take a while depending on the number of features...
                </p>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-3 text-slate-300">
                  <Loader className="w-4 h-4 text-blue-400 animate-spin" />
                  Creating model and source records...
                </div>
                <div className="flex items-center gap-3 text-slate-500">
                  <div className="w-4 h-4 rounded-full border border-slate-600" />
                  Creating neurons for each feature...
                </div>
                {config.includeActivations && (
                  <div className="flex items-center gap-3 text-slate-500">
                    <div className="w-4 h-4 rounded-full border border-slate-600" />
                    Creating activation examples...
                  </div>
                )}
                {config.includeExplanations && (
                  <div className="flex items-center gap-3 text-slate-500">
                    <div className="w-4 h-4 rounded-full border border-slate-600" />
                    Creating explanations...
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Complete View */}
          {view === 'complete' && result && (
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
                    <span className="text-slate-500">Neurons Created:</span>
                    <span className="ml-2 text-slate-200">{result.neuronsCreated.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Activations Created:</span>
                    <span className="ml-2 text-slate-200">{result.activationsCreated.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Explanations Created:</span>
                    <span className="ml-2 text-slate-200">{result.explanationsCreated.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Source ID:</span>
                    <span className="ml-2 text-slate-200 font-mono text-xs">{result.sourceId}</span>
                  </div>
                </div>
              </div>

              {/* View in Neuronpedia */}
              {result.neuronpediaUrl && (
                <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                  <h4 className="text-sm font-medium text-emerald-300 mb-2">View in Neuronpedia</h4>
                  <p className="text-sm text-slate-300 mb-3">
                    Your features are now available for browsing in the local Neuronpedia instance.
                  </p>
                  <a
                    href={result.neuronpediaUrl}
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
                    Pushing...
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
