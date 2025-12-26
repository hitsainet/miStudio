/**
 * TokenizationsList - Display and manage tokenizations for a dataset.
 *
 * Shows all available tokenizations for a dataset, their status, and provides
 * controls to create or delete tokenizations.
 */

import { useEffect, useState } from 'react';
import { CheckCircle, Loader, AlertCircle, Plus, Trash2, Hash, X, Clock } from 'lucide-react';
import { TokenizationStatus, TokenFilterMode } from '../../types/dataset';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useModelsStore } from '../../stores/modelsStore';
import { COMPONENTS } from '../../config/brand';
import { useTokenizationWebSocket } from '../../hooks/useTokenizationWebSocket';
import { TokenizationProgressDisplay } from './TokenizationProgressDisplay';

// Helper to format seconds to human-readable time
const formatElapsedTime = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
  }
};

// Fallback progress component for when WebSocket updates aren't available
function FallbackProgress({ progress, createdAt }: { progress: number; createdAt?: string }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!createdAt) return;

    const startTime = new Date(createdAt).getTime();

    const updateElapsed = () => {
      const now = Date.now();
      setElapsed((now - startTime) / 1000);
    };

    updateElapsed();
    const interval = setInterval(updateElapsed, 1000);

    return () => clearInterval(interval);
  }, [createdAt]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
        <span className="text-blue-400">Processing</span>
        <span className="font-medium text-emerald-400">{progress.toFixed(1)}%</span>
      </div>
      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-emerald-500 transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
      {createdAt && (
        <div className="flex items-center gap-1.5 text-xs">
          <Clock className="w-3 h-3 text-slate-500" />
          <span className="text-slate-500">Elapsed:</span>
          <span className="text-slate-300 font-medium">{formatElapsedTime(elapsed)}</span>
        </div>
      )}
    </div>
  );
}

interface TokenizationsListProps {
  datasetId: string;
}

export function TokenizationsList({ datasetId }: TokenizationsListProps) {
  const { tokenizations, tokenizationProgress, fetchTokenizations, deleteTokenization, cancelTokenization, createTokenization } = useDatasetsStore();
  const { models, fetchModels } = useModelsStore();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  // Filtering configuration state
  const [filterEnabled, setFilterEnabled] = useState(false);
  const [filterMode, setFilterMode] = useState<TokenFilterMode>('conservative');
  const [junkRatioThreshold, setJunkRatioThreshold] = useState(0.7);
  const [removeAllPunctuation, setRemoveAllPunctuation] = useState(false);
  const [customFilterChars, setCustomFilterChars] = useState('');

  const datasetTokenizations = tokenizations[datasetId] || [];

  // Set up WebSocket subscriptions for active tokenizations
  const activeTokenizations = datasetTokenizations
    .filter(t => t.status === TokenizationStatus.PROCESSING || t.status === TokenizationStatus.QUEUED)
    .map(t => ({ datasetId, tokenizationId: t.id }));

  useTokenizationWebSocket(activeTokenizations);

  useEffect(() => {
    fetchTokenizations(datasetId);
    fetchModels();
  }, [datasetId, fetchTokenizations, fetchModels]);

  const handleDelete = async (modelId: string, tokenizerName: string) => {
    if (window.confirm(`Delete tokenization with ${tokenizerName}?`)) {
      try {
        await deleteTokenization(datasetId, modelId);
      } catch (error) {
        console.error('Failed to delete tokenization:', error);
      }
    }
  };

  const handleCancel = async (modelId: string, tokenizerName: string) => {
    if (window.confirm(`Cancel tokenization with ${tokenizerName}?`)) {
      try {
        await cancelTokenization(datasetId, modelId);
      } catch (error) {
        console.error('Failed to cancel tokenization:', error);
      }
    }
  };

  const handleCreate = async () => {
    if (!selectedModelId) return;

    setIsCreating(true);
    try {
      await createTokenization(datasetId, selectedModelId, {
        max_length: 512,
        stride: 0,
        padding: 'max_length',
        truncation: 'longest_first',
        add_special_tokens: true,
        return_attention_mask: true,
        enable_cleaning: true,
        // Filter configuration
        tokenization_filter_enabled: filterEnabled,
        tokenization_filter_mode: filterMode,
        tokenization_junk_ratio_threshold: junkRatioThreshold,
        remove_all_punctuation: removeAllPunctuation,
        custom_filter_chars: customFilterChars || undefined,
      });
      setShowCreateForm(false);
      setSelectedModelId('');
      // Reset filter settings
      setFilterEnabled(false);
      setFilterMode('conservative');
      setJunkRatioThreshold(0.7);
      setRemoveAllPunctuation(false);
      setCustomFilterChars('');
    } catch (error) {
      console.error('Failed to create tokenization:', error);
    } finally {
      setIsCreating(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case TokenizationStatus.READY:
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case TokenizationStatus.PROCESSING:
      case TokenizationStatus.QUEUED:
        return <Loader className="w-4 h-4 text-blue-400 animate-spin" />;
      case TokenizationStatus.ERROR:
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case TokenizationStatus.READY:
        return 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400';
      case TokenizationStatus.PROCESSING:
      case TokenizationStatus.QUEUED:
        return 'bg-blue-500/10 border-blue-500/30 text-blue-400';
      case TokenizationStatus.ERROR:
        return 'bg-red-500/10 border-red-500/30 text-red-400';
      default:
        return 'bg-slate-500/10 border-slate-500/30 text-slate-400';
    }
  };

  const availableModels = models.filter(
    (model) => !datasetTokenizations.some((t) => t.model_id === model.id)
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-100">
          Tokenizations ({datasetTokenizations.length})
        </h3>
        {!showCreateForm && (
          <button
            onClick={() => setShowCreateForm(true)}
            className={`${COMPONENTS.button.secondary} text-sm px-3 py-1.5`}
            disabled={availableModels.length === 0}
          >
            <Plus className="w-4 h-4 mr-1" />
            Add Tokenization
          </button>
        )}
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <div className={`${COMPONENTS.card.base} p-4 space-y-3`}>
          <h4 className="text-sm font-medium text-slate-100">Create New Tokenization</h4>
          <div className="space-y-2">
            <label className="text-xs text-slate-400">Select Model</label>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100 text-sm"
            >
              <option value="">-- Select a model --</option>
              {availableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>

          {/* Filtering Settings Section */}
          <div className="border-t border-slate-700 pt-3 mt-3 space-y-3">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="filter-enabled"
                checked={filterEnabled}
                onChange={(e) => setFilterEnabled(e.target.checked)}
                className="w-4 h-4 bg-slate-900 border-slate-700 rounded text-emerald-500 focus:ring-emerald-500"
              />
              <label htmlFor="filter-enabled" className="text-sm font-medium text-slate-100">
                Enable Sample Filtering
              </label>
              <span className="text-xs text-slate-500 ml-auto">
                Removes samples with too many junk tokens
              </span>
            </div>

            {filterEnabled && (
              <div className="ml-6 space-y-3 bg-slate-900/50 p-3 rounded border border-slate-700/50">
                {/* Filter Mode */}
                <div className="space-y-2">
                  <label className="text-xs font-medium text-slate-300">Filter Mode</label>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="minimal"
                        checked={filterMode === 'minimal'}
                        onChange={(e) => setFilterMode(e.target.value as TokenFilterMode)}
                        className="w-3.5 h-3.5 text-emerald-500 bg-slate-900 border-slate-700 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-200">Minimal</span>
                      <span className="text-xs text-slate-500">- Only control chars</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="conservative"
                        checked={filterMode === 'conservative'}
                        onChange={(e) => setFilterMode(e.target.value as TokenFilterMode)}
                        className="w-3.5 h-3.5 text-emerald-500 bg-slate-900 border-slate-700 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-200">Conservative</span>
                      <span className="text-xs text-slate-500">- + Whitespace tokens</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="standard"
                        checked={filterMode === 'standard'}
                        onChange={(e) => setFilterMode(e.target.value as TokenFilterMode)}
                        className="w-3.5 h-3.5 text-emerald-500 bg-slate-900 border-slate-700 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-200">Standard</span>
                      <span className="text-xs text-slate-500">- + Pure punctuation</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="aggressive"
                        checked={filterMode === 'aggressive'}
                        onChange={(e) => setFilterMode(e.target.value as TokenFilterMode)}
                        className="w-3.5 h-3.5 text-emerald-500 bg-slate-900 border-slate-700 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-200">Aggressive</span>
                      <span className="text-xs text-slate-500">- + Short tokens</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="strict"
                        checked={filterMode === 'strict'}
                        onChange={(e) => setFilterMode(e.target.value as TokenFilterMode)}
                        className="w-3.5 h-3.5 text-emerald-500 bg-slate-900 border-slate-700 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-200">Strict</span>
                      <span className="text-xs text-slate-500">- + ALL punctuation</span>
                    </label>
                  </div>
                </div>

                {/* Remove All Punctuation */}
                <div className="flex items-center gap-2 pt-2 border-t border-slate-700/50">
                  <input
                    type="checkbox"
                    id="remove-all-punctuation"
                    checked={removeAllPunctuation}
                    onChange={(e) => setRemoveAllPunctuation(e.target.checked)}
                    className="w-4 h-4 bg-slate-900 border-slate-700 rounded text-emerald-500 focus:ring-emerald-500"
                  />
                  <label htmlFor="remove-all-punctuation" className="text-sm text-slate-200">
                    Remove ALL Punctuation
                  </label>
                </div>
                <p className="text-xs text-slate-500 ml-6">
                  Removes every punctuation character, even within words (overrides mode)
                </p>

                {/* Custom Filter Characters */}
                <div className="space-y-2 pt-2 border-t border-slate-700/50">
                  <label className="text-xs font-medium text-slate-300">Custom Characters to Filter</label>
                  <input
                    type="text"
                    autoComplete="off"
                    value={customFilterChars}
                    onChange={(e) => setCustomFilterChars(e.target.value)}
                    placeholder="e.g., ~@#$%"
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100 text-sm placeholder-slate-500"
                  />
                  <p className="text-xs text-slate-500">
                    Additional characters to remove from tokens (e.g., ~@#$%)
                  </p>
                </div>

                {/* Junk Ratio Threshold */}
                <div className="space-y-2 pt-2 border-t border-slate-700/50">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-slate-300">Junk Ratio Threshold</label>
                    <span className="text-xs text-emerald-400 font-mono">{(junkRatioThreshold * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={junkRatioThreshold * 100}
                    onChange={(e) => setJunkRatioThreshold(parseInt(e.target.value) / 100)}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                  <p className="text-xs text-slate-500">
                    Skip samples if &gt;{(junkRatioThreshold * 100).toFixed(0)}% of tokens are junk
                  </p>
                </div>

                {/* Warning */}
                <div className="flex items-start gap-2 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded">
                  <AlertCircle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-yellow-300">
                    Filtering is permanent. Filtered samples will not be included in the tokenized dataset.
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleCreate}
              disabled={!selectedModelId || isCreating}
              className={`${COMPONENTS.button.primary} text-sm px-4 py-2 flex-1`}
            >
              {isCreating ? 'Creating...' : 'Create'}
            </button>
            <button
              onClick={() => {
                setShowCreateForm(false);
                setSelectedModelId('');
              }}
              className={`${COMPONENTS.button.secondary} text-sm px-4 py-2`}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Empty State */}
      {datasetTokenizations.length === 0 && !showCreateForm && (
        <div className="text-center py-8 text-slate-400 text-sm">
          <Hash className="w-8 h-8 mx-auto mb-2 text-slate-600" />
          <p>No tokenizations yet</p>
          <p className="text-xs text-slate-500 mt-1">
            Create a tokenization to use this dataset for training
          </p>
        </div>
      )}

      {/* Tokenizations List */}
      {datasetTokenizations.length > 0 && (
        <div className="space-y-2">
          {datasetTokenizations.map((tokenization) => (
            <div
              key={tokenization.id}
              className={`${COMPONENTS.card.base} p-3 space-y-2`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(tokenization.status)}
                    <span className="text-sm font-medium text-slate-100 truncate">
                      {tokenization.tokenizer_repo_id}
                    </span>
                  </div>
                  <div className={`inline-flex items-center gap-1.5 mt-1.5 px-2 py-0.5 border rounded text-xs ${getStatusColor(tokenization.status)}`}>
                    {tokenization.status.toUpperCase()}
                  </div>
                </div>

                {/* Action buttons */}
                {(tokenization.status === TokenizationStatus.READY || tokenization.status === TokenizationStatus.ERROR) && (
                  <button
                    onClick={() => handleDelete(tokenization.model_id, tokenization.tokenizer_repo_id)}
                    className={`${COMPONENTS.button.ghost} p-1.5`}
                    title="Delete tokenization"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
                {(tokenization.status === TokenizationStatus.PROCESSING || tokenization.status === TokenizationStatus.QUEUED) && (
                  <button
                    onClick={() => handleCancel(tokenization.model_id, tokenization.tokenizer_repo_id)}
                    className={`${COMPONENTS.button.ghost} p-1.5 text-red-400 hover:text-red-300`}
                    title="Cancel tokenization"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>

              {/* Stats */}
              {tokenization.status === TokenizationStatus.READY && (
                <div className="grid grid-cols-3 gap-3 pt-2 border-t border-slate-700/50">
                  <div>
                    <p className="text-xs text-slate-500">Vocab Size</p>
                    <p className="text-sm text-slate-300 font-medium">
                      {tokenization.vocab_size?.toLocaleString() || 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Total Tokens</p>
                    <p className="text-sm text-slate-300 font-medium">
                      {tokenization.num_tokens ? (tokenization.num_tokens / 1e9).toFixed(2) + 'B' : 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Avg Length</p>
                    <p className="text-sm text-slate-300 font-medium">
                      {tokenization.avg_seq_length?.toFixed(1) || 'N/A'}
                    </p>
                  </div>
                </div>
              )}

              {/* Progress */}
              {(tokenization.status === TokenizationStatus.PROCESSING || tokenization.status === TokenizationStatus.QUEUED) && (
                <div className="pt-2 border-t border-slate-700/50">
                  {tokenizationProgress[tokenization.id] ? (
                    <TokenizationProgressDisplay progress={tokenizationProgress[tokenization.id]} />
                  ) : tokenization.progress !== undefined ? (
                    <FallbackProgress
                      progress={tokenization.progress}
                      createdAt={tokenization.created_at}
                    />
                  ) : null}
                </div>
              )}

              {/* Error Message */}
              {tokenization.error_message && (
                <div className="pt-2 text-xs text-red-400">
                  {tokenization.error_message}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
