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
import { API_BASE_URL } from '../../config/api';
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
      <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-400 mb-1">
        <span className="text-blue-400">Processing</span>
        <span className="font-medium text-emerald-400">{progress.toFixed(1)}%</span>
      </div>
      <div className="w-full h-2 bg-white dark:bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-emerald-500 transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
      {createdAt && (
        <div className="flex items-center gap-1.5 text-xs">
          <Clock className="w-3 h-3 text-slate-500" />
          <span className="text-slate-500">Elapsed:</span>
          <span className="text-slate-700 dark:text-slate-300 font-medium">{formatElapsedTime(elapsed)}</span>
        </div>
      )}
    </div>
  );
}

export type ChatFormat = 'auto' | 'chat_template' | 'plain' | 'none';

export interface TokenizationFormValues {
  maxLength: number;
  textColumn: string;
  chatFormat: ChatFormat;
  packSequences: boolean;
  enableCleaning: boolean;
  filterEnabled: boolean;
  filterMode: TokenFilterMode;
  junkRatioThreshold: number;
  removeAllPunctuation: boolean;
  customFilterChars: string;
}

/**
 * Build the /tokenize payload from the form.
 *
 * WHY THIS IS A FUNCTION AND NOT INLINE IN `handleCreate`. The three controls
 * this arc added — `text_column`, `chat_format`, `pack_sequences` — existed on
 * the backend schema and were sent by nothing, so an operator clicking Tokenize
 * got the old defaults. That is the same defect shape as round 4's `target_l0`:
 * a control made visible whose value never reached the request. A rendered
 * input proves nothing; only the payload does, and a payload built by a pure
 * function can be asserted directly instead of scraped out of JSX.
 *
 * `text_column` is omitted rather than sent empty, because omitted means
 * "auto-detect" to the backend while "" would be a column name that does not
 * exist. Auto-detection is what tokenized Bloomberg on `Headline`, so the
 * operator must be able to override it — but leaving the field blank has to go
 * on meaning the old behaviour, not an error.
 */
export function buildTokenizationPayload(values: TokenizationFormValues) {
  const textColumn = values.textColumn.trim();
  return {
    max_length: values.maxLength,
    stride: 0,
    padding: 'max_length' as const,
    truncation: 'longest_first' as const,
    add_special_tokens: true,
    return_attention_mask: true,
    // Off unless the operator asks: cleaning rewrites the corpus, and with it
    // on every newline and every chat turn marker was stripped (2026-09-12).
    enable_cleaning: values.enableCleaning,
    ...(textColumn ? { text_column: textColumn } : {}),
    chat_format: values.chatFormat,
    pack_sequences: values.packSequences,
    // Filter configuration
    tokenization_filter_enabled: values.filterEnabled,
    tokenization_filter_mode: values.filterMode,
    tokenization_junk_ratio_threshold: values.junkRatioThreshold,
    remove_all_punctuation: values.removeAllPunctuation,
    custom_filter_chars: values.customFilterChars || undefined,
  };
}

interface TokenizationsListProps {
  datasetId: string;
}

export function TokenizationsList({ datasetId }: TokenizationsListProps) {
  const { tokenizations, tokenizationProgress, fetchTokenizations, deleteTokenization, cancelTokenization, createTokenization, error: storeError } = useDatasetsStore();
  const [actionError, setActionError] = useState<string | null>(null);
  const { models, fetchModels } = useModelsStore();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [maxLength, setMaxLength] = useState(512);
  const [isCreating, setIsCreating] = useState(false);

  // Filtering configuration state
  const [filterEnabled, setFilterEnabled] = useState(false);
  const [filterMode, setFilterMode] = useState<TokenFilterMode>('conservative');
  const [junkRatioThreshold, setJunkRatioThreshold] = useState(0.7);
  const [removeAllPunctuation, setRemoveAllPunctuation] = useState(false);
  const [customFilterChars, setCustomFilterChars] = useState('');

  // Token-stream shape: which column, how conversations are rendered, packing.
  const [textColumn, setTextColumn] = useState('');
  const [chatFormat, setChatFormat] = useState<ChatFormat>('auto');
  const [packSequences, setPackSequences] = useState(false);
  const [enableCleaning, setEnableCleaning] = useState(false);
  const [availableColumns, setAvailableColumns] = useState<string[] | null>(null);

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

  // Column names for the override below. The dataset row carries no schema
  // metadata (`metadata.schema` is null on every row in production), so the
  // only place the real column names exist is a sample. One row is enough.
  // If this fails the override degrades to a free-text input rather than
  // disappearing — an operator who knows the column must still be able to
  // name it.
  useEffect(() => {
    if (!showCreateForm || availableColumns !== null) return;
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/api/v1/datasets/${datasetId}/samples?limit=1`
        );
        if (!response.ok) throw new Error(String(response.status));
        const body = await response.json();
        const first = body?.data?.[0]?.data;
        if (!cancelled && first && typeof first === 'object') {
          setAvailableColumns(Object.keys(first));
        } else if (!cancelled) {
          setAvailableColumns([]);
        }
      } catch {
        if (!cancelled) setAvailableColumns([]);
      }
    })();
    return () => { cancelled = true; };
  }, [showCreateForm, availableColumns, datasetId]);

  const handleDelete = async (tokenizationId: string, tokenizerName: string, maxLength: number) => {
    if (window.confirm(`Delete tokenization ${tokenizerName} (${maxLength} tokens)?`)) {
      setActionError(null);
      try {
        await deleteTokenization(datasetId, tokenizationId);
      } catch (error) {
        console.error('Failed to delete tokenization:', error);
        setActionError(
          error instanceof Error ? error.message : 'Failed to delete tokenization'
        );
      }
    }
  };

  const handleCancel = async (tokenizationId: string, tokenizerName: string, maxLength: number) => {
    if (window.confirm(`Cancel tokenization ${tokenizerName} (${maxLength} tokens)?`)) {
      setActionError(null);
      try {
        await cancelTokenization(datasetId, tokenizationId);
      } catch (error) {
        console.error('Failed to cancel tokenization:', error);
        setActionError(
          error instanceof Error ? error.message : 'Failed to cancel tokenization'
        );
      }
    }
  };

  const handleCreate = async () => {
    if (!selectedModelId) return;

    setIsCreating(true);
    try {
      await createTokenization(datasetId, selectedModelId, buildTokenizationPayload({
        maxLength,
        textColumn,
        chatFormat,
        packSequences,
        enableCleaning,
        filterEnabled,
        filterMode,
        junkRatioThreshold,
        removeAllPunctuation,
        customFilterChars,
      }));
      setShowCreateForm(false);
      setSelectedModelId('');
      setMaxLength(512);
      setTextColumn('');
      setChatFormat('auto');
      setPackSequences(false);
      setEnableCleaning(false);
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

  // All models are available - same model can have multiple tokenizations with different max_lengths
  const availableModels = models;

  const bannerError = actionError || storeError;

  return (
    <div className="space-y-4">
      {bannerError && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400"
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span className="flex-1">{bannerError}</span>
          <button
            type="button"
            onClick={() => setActionError(null)}
            className="text-red-400/70 hover:text-red-300"
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
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
          <h4 className="text-sm font-medium text-slate-900 dark:text-slate-100">Create New Tokenization</h4>
          <div className="space-y-2">
            <label className="text-xs text-slate-600 dark:text-slate-400" htmlFor="tokenizer-model">Select Model</label>
            <select
              id="tokenizer-model"
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-slate-100 text-sm"
            >
              <option value="">-- Select a model --</option>
              {availableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-xs text-slate-600 dark:text-slate-400" htmlFor="max-sequence-length">Max Sequence Length</label>
            <input
              id="max-sequence-length"
              type="number"
              value={maxLength}
              onChange={(e) => setMaxLength(Math.max(1, Math.min(8192, parseInt(e.target.value) || 512)))}
              min={1}
              max={8192}
              className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-slate-100 text-sm"
            />
            <p className="text-xs text-slate-500">
              Maximum tokens per sample (1-8192). Common values: 128, 256, 512, 1024, 2048
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-xs text-slate-600 dark:text-slate-400" htmlFor="text-column">
              Text Column
            </label>
            {availableColumns && availableColumns.length > 0 ? (
              <select
                id="text-column"
                value={textColumn}
                onChange={(e) => setTextColumn(e.target.value)}
                className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-slate-100 text-sm"
              >
                <option value="">Auto-detect</option>
                {availableColumns.map((column) => (
                  <option key={column} value={column}>{column}</option>
                ))}
              </select>
            ) : (
              <input
                id="text-column"
                type="text"
                value={textColumn}
                onChange={(e) => setTextColumn(e.target.value)}
                placeholder="Auto-detect"
                className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-slate-100 text-sm"
              />
            )}
            <p className="text-xs text-slate-500">
              Auto-detect takes the first text-like column. That is how this
              dataset&apos;s Bloomberg corpus was tokenized on{' '}
              <span className="font-mono">Headline</span> (17 tokens) while{' '}
              <span className="font-mono">Article</span> (464) went unused.
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-xs text-slate-600 dark:text-slate-400" htmlFor="chat-format">
              Conversation Rendering
            </label>
            <select
              id="chat-format"
              value={chatFormat}
              onChange={(e) => setChatFormat(e.target.value as ChatFormat)}
              className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-slate-100 text-sm"
            >
              <option value="auto">Auto — use the tokenizer&apos;s template when it has one</option>
              <option value="chat_template">Chat template — always</option>
              <option value="plain">Plain — concatenate content, no role markers</option>
              <option value="none">None — treat as ordinary text</option>
            </select>
            <p className="text-xs text-slate-500">
              Only affects datasets with a conversation column. Do not wrap raw
              web text in a chat template — it is a distribution the model
              rarely sees.
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="pack-sequences"
                checked={packSequences}
                onChange={(e) => setPackSequences(e.target.checked)}
                className="w-4 h-4 bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 rounded text-emerald-500 focus:ring-emerald-500"
              />
              <label htmlFor="pack-sequences" className="text-sm font-medium text-slate-900 dark:text-slate-100">
                Pack Sequences
              </label>
            </div>
            <p className="text-xs text-slate-500">
              Concatenate documents into full {maxLength.toLocaleString()}-token
              blocks instead of padding each one. On short-document corpora this
              is the difference between ~3% and ~99% real-token occupancy.
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="enable-cleaning"
                checked={enableCleaning}
                onChange={(e) => setEnableCleaning(e.target.checked)}
                className="w-4 h-4 bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 rounded text-emerald-500 focus:ring-emerald-500"
              />
              <label htmlFor="enable-cleaning" className="text-sm font-medium text-slate-900 dark:text-slate-100">
                Clean Text
              </label>
            </div>
            <p className="text-xs text-slate-500">
              Rewrites text before tokenizing: strips HTML tags and URLs and
              collapses repeated spaces. Off by default, because the SAE should
              see the text the model actually reads. Leave it off for code and chat.
            </p>
          </div>

          {/* Filtering Settings Section */}
          <div className="border-t border-slate-300 dark:border-slate-700 pt-3 mt-3 space-y-3">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="filter-enabled"
                checked={filterEnabled}
                onChange={(e) => setFilterEnabled(e.target.checked)}
                className="w-4 h-4 bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 rounded text-emerald-500 focus:ring-emerald-500"
              />
              <label htmlFor="filter-enabled" className="text-sm font-medium text-slate-900 dark:text-slate-100">
                Enable Sample Filtering
              </label>
              <span className="text-xs text-slate-500 ml-auto">
                Removes samples with too many junk tokens
              </span>
            </div>

            {filterEnabled && (
              <div className="ml-6 space-y-3 bg-slate-100 dark:bg-slate-900/50 p-3 rounded border border-slate-300 dark:border-slate-700/50">
                {/* Filter Mode */}
                <div className="space-y-2">
                  <label className="text-xs font-medium text-slate-700 dark:text-slate-300">Filter Mode</label>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="minimal"
                        checked={filterMode === 'minimal'}
                        onChange={(e) => setFilterMode(e.target.value as TokenFilterMode)}
                        className="w-3.5 h-3.5 text-emerald-500 bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-800 dark:text-slate-200">Minimal</span>
                      <span className="text-xs text-slate-500">- Only control chars</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="conservative"
                        checked={filterMode === 'conservative'}
                        onChange={(e) => setFilterMode(e.target.value as TokenFilterMode)}
                        className="w-3.5 h-3.5 text-emerald-500 bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-800 dark:text-slate-200">Conservative</span>
                      <span className="text-xs text-slate-500">- + Whitespace tokens</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="standard"
                        checked={filterMode === 'standard'}
                        onChange={(e) => setFilterMode(e.target.value as TokenFilterMode)}
                        className="w-3.5 h-3.5 text-emerald-500 bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-800 dark:text-slate-200">Standard</span>
                      <span className="text-xs text-slate-500">- + Pure punctuation</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="aggressive"
                        checked={filterMode === 'aggressive'}
                        onChange={(e) => setFilterMode(e.target.value as TokenFilterMode)}
                        className="w-3.5 h-3.5 text-emerald-500 bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-800 dark:text-slate-200">Aggressive</span>
                      <span className="text-xs text-slate-500">- + Short tokens</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="strict"
                        checked={filterMode === 'strict'}
                        onChange={(e) => setFilterMode(e.target.value as TokenFilterMode)}
                        className="w-3.5 h-3.5 text-emerald-500 bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-800 dark:text-slate-200">Strict</span>
                      <span className="text-xs text-slate-500">- + ALL punctuation</span>
                    </label>
                  </div>
                </div>

                {/* Remove All Punctuation */}
                <div className="flex items-center gap-2 pt-2 border-t border-slate-300 dark:border-slate-700/50">
                  <input
                    type="checkbox"
                    id="remove-all-punctuation"
                    checked={removeAllPunctuation}
                    onChange={(e) => setRemoveAllPunctuation(e.target.checked)}
                    className="w-4 h-4 bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 rounded text-emerald-500 focus:ring-emerald-500"
                  />
                  <label htmlFor="remove-all-punctuation" className="text-sm text-slate-800 dark:text-slate-200">
                    Remove ALL Punctuation
                  </label>
                </div>
                <p className="text-xs text-slate-500 ml-6">
                  Removes every punctuation character, even within words (overrides mode)
                </p>

                {/* Custom Filter Characters */}
                <div className="space-y-2 pt-2 border-t border-slate-300 dark:border-slate-700/50">
                  <label className="text-xs font-medium text-slate-700 dark:text-slate-300">Custom Characters to Filter</label>
                  <input
                    type="text"
                    autoComplete="off"
                    value={customFilterChars}
                    onChange={(e) => setCustomFilterChars(e.target.value)}
                    placeholder="e.g., ~@#$%"
                    className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-slate-100 text-sm placeholder-slate-400 dark:placeholder-slate-500"
                  />
                  <p className="text-xs text-slate-500">
                    Additional characters to remove from tokens (e.g., ~@#$%)
                  </p>
                </div>

                {/* Junk Ratio Threshold */}
                <div className="space-y-2 pt-2 border-t border-slate-300 dark:border-slate-700/50">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-slate-700 dark:text-slate-300">Junk Ratio Threshold</label>
                    <span className="text-xs text-emerald-400 font-mono">{(junkRatioThreshold * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={junkRatioThreshold * 100}
                    onChange={(e) => setJunkRatioThreshold(parseInt(e.target.value) / 100)}
                    className="w-full h-2 bg-slate-100 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                  <p className="text-xs text-slate-500">
                    Skip samples if &gt;{(junkRatioThreshold * 100).toFixed(0)}% of tokens are junk
                  </p>
                </div>

                {/* Warning */}
                <div className="flex items-start gap-2 p-2 bg-cyan-500/10 border border-cyan-500/30 rounded">
                  <AlertCircle className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-cyan-300">
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
        <div className="text-center py-8 text-slate-600 dark:text-slate-400 text-sm">
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
                    <span className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                      {tokenization.tokenizer_repo_id}
                    </span>
                    <span className="text-xs text-slate-500 bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded">
                      {tokenization.max_length || 512} tokens
                    </span>
                  </div>
                  <div className={`inline-flex items-center gap-1.5 mt-1.5 px-2 py-0.5 border rounded text-xs ${getStatusColor(tokenization.status)}`}>
                    {tokenization.status.toUpperCase()}
                  </div>
                </div>

                {/* Action buttons */}
                {(tokenization.status === TokenizationStatus.READY || tokenization.status === TokenizationStatus.ERROR) && (
                  <button
                    onClick={() => handleDelete(tokenization.id, tokenization.tokenizer_repo_id, tokenization.max_length || 512)}
                    className={`${COMPONENTS.button.ghost} p-1.5`}
                    title="Delete tokenization"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
                {(tokenization.status === TokenizationStatus.PROCESSING || tokenization.status === TokenizationStatus.QUEUED) && (
                  <button
                    onClick={() => handleCancel(tokenization.id, tokenization.tokenizer_repo_id, tokenization.max_length || 512)}
                    className={`${COMPONENTS.button.ghost} p-1.5 text-red-400 hover:text-red-300`}
                    title="Cancel tokenization"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>

              {/* Stats */}
              {tokenization.status === TokenizationStatus.READY && (
                <div className="grid grid-cols-4 gap-3 pt-2 border-t border-slate-300 dark:border-slate-700/50">
                  <div>
                    <p className="text-xs text-slate-500">Max Length</p>
                    <p className="text-sm text-slate-700 dark:text-slate-300 font-medium">
                      {tokenization.max_length?.toLocaleString() || '512'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Vocab Size</p>
                    <p className="text-sm text-slate-700 dark:text-slate-300 font-medium">
                      {tokenization.vocab_size?.toLocaleString() || 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Total Tokens</p>
                    <p className="text-sm text-slate-700 dark:text-slate-300 font-medium">
                      {tokenization.num_tokens ? (tokenization.num_tokens / 1e9).toFixed(2) + 'B' : 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Avg Length</p>
                    <p className="text-sm text-slate-700 dark:text-slate-300 font-medium">
                      {tokenization.avg_seq_length?.toFixed(1) || 'N/A'}
                    </p>
                  </div>
                </div>
              )}

              {/* Progress */}
              {(tokenization.status === TokenizationStatus.PROCESSING || tokenization.status === TokenizationStatus.QUEUED) && (
                <div className="pt-2 border-t border-slate-300 dark:border-slate-700/50">
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
