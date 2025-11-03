/**
 * DatasetDetailModal - Full-screen modal for dataset details.
 *
 * This component displays detailed information about a dataset with tabs
 * for samples, statistics, and tokenization settings.
 */

import { useState, useEffect } from 'react';
import { X, FileText, BarChart, Zap, ChevronLeft, ChevronRight, AlertCircle, Trash2 } from 'lucide-react';
import { Dataset } from '../../types/dataset';
import { formatFileSize, formatDateTime } from '../../utils/formatters';
import { StatusBadge } from '../common/StatusBadge';
import { ProgressBar } from '../common/ProgressBar';
import { useWebSocket } from '../../hooks/useWebSocket';
import { API_BASE_URL } from '../../config/api';
import { TokenizationPreview } from './TokenizationPreview';
import { useModelsStore } from '../../stores/modelsStore';

interface DatasetDetailModalProps {
  dataset: Dataset;
  onClose: () => void;
  onDatasetUpdate?: (updatedDataset: Dataset) => void;
}

type TabType = 'overview' | 'samples' | 'statistics' | 'tokenization';

export function DatasetDetailModal({ dataset, onClose, onDatasetUpdate }: DatasetDetailModalProps) {
  const [activeTab, setActiveTab] = useState<TabType>('overview');

  const tabs = [
    { id: 'overview' as TabType, label: 'Overview', icon: FileText },
    { id: 'samples' as TabType, label: 'Samples', icon: FileText },
    { id: 'tokenization' as TabType, label: 'Tokenization', icon: Zap },
    { id: 'statistics' as TabType, label: 'Statistics', icon: BarChart },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-6xl w-full mx-4 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex-1 min-w-0">
            <h2 className="text-2xl font-semibold text-slate-100 truncate">
              {dataset.name}
            </h2>
            <div className="flex items-center gap-4 mt-2">
              <StatusBadge status={dataset.status} />
              <span className="text-sm text-slate-400">
                Source: {dataset.source}
              </span>
              {dataset.num_samples && (
                <span className="text-sm text-slate-400">
                  {dataset.num_samples.toLocaleString()} samples
                </span>
              )}
              {dataset.size_bytes && (
                <span className="text-sm text-slate-400">
                  {formatFileSize(dataset.size_bytes)}
                </span>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="flex-shrink-0 ml-4 p-2 hover:bg-slate-800 rounded transition-colors"
            aria-label="Close"
          >
            <X className="w-6 h-6 text-slate-400" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-800">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-4 border-b-2 transition-colors ${
                  isActive
                    ? 'border-emerald-500 text-emerald-400'
                    : 'border-transparent text-slate-400 hover:text-slate-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="font-medium">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === 'overview' && <OverviewTab dataset={dataset} />}
          {activeTab === 'samples' && <SamplesTab dataset={dataset} />}
          {activeTab === 'tokenization' && <TokenizationTab dataset={dataset} onDatasetUpdate={onDatasetUpdate} />}
          {activeTab === 'statistics' && <StatisticsTab dataset={dataset} />}
        </div>
      </div>
    </div>
  );
}

// Overview Tab Component
function OverviewTab({ dataset }: { dataset: Dataset }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-6">
        <InfoCard label="Dataset ID" value={dataset.id} />
        <InfoCard label="Source" value={dataset.source} />
        {dataset.hf_repo_id && (
          <InfoCard label="HuggingFace Repository" value={dataset.hf_repo_id} />
        )}
        <InfoCard label="Status" value={<StatusBadge status={dataset.status} />} />
        <InfoCard
          label="Created"
          value={formatDateTime(dataset.created_at)}
        />
        <InfoCard
          label="Last Updated"
          value={formatDateTime(dataset.updated_at)}
        />
        {dataset.raw_path && <InfoCard label="Raw Path" value={dataset.raw_path} />}
        {dataset.tokenized_path && (
          <InfoCard label="Tokenized Path" value={dataset.tokenized_path} />
        )}
      </div>

      {dataset.num_samples !== undefined && (
        <div className="bg-slate-800/50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-slate-100 mb-4">Dataset Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Samples" value={dataset.num_samples.toLocaleString()} />
            {dataset.metadata?.tokenization?.num_tokens && (
              <StatCard label="Tokens" value={dataset.metadata.tokenization.num_tokens.toLocaleString()} />
            )}
            {dataset.metadata?.tokenization?.avg_seq_length && (
              <StatCard
                label="Avg Length"
                value={dataset.metadata.tokenization.avg_seq_length.toFixed(1)}
              />
            )}
            {dataset.size_bytes && (
              <StatCard label="Size" value={formatFileSize(dataset.size_bytes)} />
            )}
          </div>
        </div>
      )}

      {dataset.error_message && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
          <h3 className="text-red-400 font-semibold mb-2">Error</h3>
          <p className="text-red-400 text-sm">{dataset.error_message}</p>
        </div>
      )}
    </div>
  );
}

// Samples Tab Component
function SamplesTab({ dataset }: { dataset: Dataset }) {
  const [samples, setSamples] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pagination, setPagination] = useState<any>(null);
  const limit = 20;

  useEffect(() => {
    const fetchSamples = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(
          `${API_BASE_URL}/api/v1/datasets/${dataset.id}/samples?page=${page}&limit=${limit}`
        );

        if (!response.ok) {
          throw new Error('Failed to fetch samples');
        }

        const data = await response.json();
        setSamples(data.data);
        setPagination(data.pagination);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load samples');
      } finally {
        setLoading(false);
      }
    };

    if (dataset.status === 'ready') {
      fetchSamples();
    }
  }, [dataset.id, dataset.status, page]);

  if (dataset.status !== 'ready') {
    return (
      <div className="text-center py-12">
        <FileText className="w-12 h-12 text-slate-600 mx-auto mb-4" />
        <p className="text-slate-400 text-lg">Dataset not ready</p>
        <p className="text-slate-500 mt-2">
          Samples can be viewed once the dataset is in "ready" status
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 border-t-emerald-500 mb-4"></div>
        <p className="text-slate-400">Loading samples...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg inline-block">
          <p className="text-red-400">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Pagination Header */}
      {pagination && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-slate-400">
            Showing {((page - 1) * limit) + 1} - {Math.min(page * limit, pagination.total)} of {pagination.total.toLocaleString()} samples
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={!pagination.has_prev}
              className="p-2 rounded hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-4 h-4 text-slate-400" />
            </button>
            <span className="text-sm text-slate-400">
              Page {page} of {pagination.total_pages.toLocaleString()}
            </span>
            <button
              onClick={() => setPage(p => p + 1)}
              disabled={!pagination.has_next}
              className="p-2 rounded hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight className="w-4 h-4 text-slate-400" />
            </button>
          </div>
        </div>
      )}

      {/* Samples List */}
      <div className="space-y-3">
        {samples.map((sample) => (
          <div
            key={sample.index}
            className="bg-slate-800/50 rounded-lg p-4 border border-slate-700"
          >
            <div className="flex items-start justify-between mb-2">
              <span className="text-xs font-mono text-slate-500">Sample #{sample.index}</span>
            </div>
            <div className="space-y-2">
              {Object.entries(sample.data).map(([key, value]) => (
                <div key={key}>
                  <span className="text-xs font-medium text-emerald-400">{key}:</span>
                  <pre className="text-sm text-slate-300 mt-1 whitespace-pre-wrap font-mono">
                    {typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Pagination Footer */}
      {pagination && pagination.total_pages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-4">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={!pagination.has_prev}
            className="px-4 py-2 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm text-slate-300"
          >
            Previous
          </button>
          <button
            onClick={() => setPage(p => p + 1)}
            disabled={!pagination.has_next}
            className="px-4 py-2 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm text-slate-300"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

// Statistics Tab Component
function StatisticsTab({ dataset }: { dataset: Dataset }) {
  // Extract tokenization statistics from metadata
  const tokenizationStats = dataset.metadata?.tokenization;

  // Check if tokenization stats exist and have required fields
  const hasCompleteStats = tokenizationStats &&
    tokenizationStats.num_tokens !== undefined &&
    tokenizationStats.avg_seq_length !== undefined &&
    tokenizationStats.min_seq_length !== undefined &&
    tokenizationStats.max_seq_length !== undefined;

  if (!tokenizationStats) {
    return (
      <div className="text-center py-12">
        <BarChart className="w-12 h-12 text-slate-600 mx-auto mb-4" />
        <p className="text-slate-400 text-lg">No tokenization statistics available</p>
        <p className="text-slate-500 mt-2">
          Tokenize this dataset to view detailed statistics
        </p>
      </div>
    );
  }

  if (!hasCompleteStats) {
    return (
      <div className="text-center py-12">
        <BarChart className="w-12 h-12 text-slate-600 mx-auto mb-4" />
        <p className="text-slate-400 text-lg">Incomplete tokenization statistics</p>
        <p className="text-slate-500 mt-2">
          This dataset was tokenized with an older version. Please re-tokenize to view complete statistics.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Token Statistics */}
      <div className="bg-slate-800/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-slate-100 mb-4">Token Statistics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <ColoredStatCard
            label="Total Samples"
            value={dataset.num_samples?.toLocaleString() ?? 'N/A'}
            color="emerald"
          />
          <ColoredStatCard
            label="Total Tokens"
            value={tokenizationStats.num_tokens!.toLocaleString()}
            color="blue"
          />
          <ColoredStatCard
            label="Avg Tokens/Sample"
            value={tokenizationStats.avg_seq_length!.toFixed(1)}
            color="purple"
          />
          {tokenizationStats.vocab_size !== undefined && (
            <ColoredStatCard
              label="Unique Tokens"
              value={tokenizationStats.vocab_size.toLocaleString()}
              color="yellow"
            />
          )}
        </div>
      </div>

      {/* Sequence Length Distribution Visualization */}
      <div className="bg-slate-800/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-slate-100 mb-4">Sequence Length Distribution</h3>
        <div className="space-y-4">
          {/* Bucketed histogram - Horizontal bars */}
          {tokenizationStats.length_distribution ? (
            <div className="space-y-3">
              {Object.entries(tokenizationStats.length_distribution).map(([range, count]) => {
                const maxCount = Math.max(...Object.values(tokenizationStats.length_distribution!));
                const widthPercent = maxCount > 0 ? (count / maxCount) * 100 : 0;

                return (
                  <div key={range} className="flex items-center gap-3">
                    <div className="w-20 text-xs text-slate-400 text-right">{range}</div>
                    <div className="flex-1 h-8 bg-slate-700 rounded overflow-hidden relative">
                      <div
                        className="h-full bg-emerald-500/60 flex items-center justify-end pr-2 transition-all"
                        style={{
                          width: `${widthPercent}%`,
                          minWidth: count > 0 ? '24px' : '0px'
                        }}
                      >
                        {count > 0 && (
                          <span className="text-xs font-mono text-white">{count.toLocaleString()}</span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            // Fallback to simple Min/Avg/Max bars if length_distribution not available
            <div className="flex items-end justify-between h-48 gap-2">
              <div className="flex-1 flex flex-col items-center">
                <div
                  className="w-full bg-emerald-500/30 border-t-2 border-emerald-500"
                  style={{
                    height: `${(tokenizationStats.min_seq_length / tokenizationStats.max_seq_length) * 100}%`,
                    minHeight: '4px'
                  }}
                ></div>
                <span className="text-xs text-slate-400 mt-2">Min</span>
                <span className="text-xs font-mono text-slate-500">{tokenizationStats.min_seq_length}</span>
              </div>
              <div className="flex-1 flex flex-col items-center">
                <div
                  className="w-full bg-emerald-500/50 border-t-2 border-emerald-400"
                  style={{
                    height: `${(tokenizationStats.avg_seq_length / tokenizationStats.max_seq_length) * 100}%`,
                    minHeight: '4px'
                  }}
                ></div>
                <span className="text-xs text-slate-400 mt-2">Avg</span>
                <span className="text-xs font-mono text-slate-500">{tokenizationStats.avg_seq_length.toFixed(1)}</span>
              </div>
              <div className="flex-1 flex flex-col items-center">
                <div
                  className="w-full bg-emerald-500/70 border-t-2 border-emerald-300"
                  style={{ height: '100%', minHeight: '4px' }}
                ></div>
                <span className="text-xs text-slate-400 mt-2">Max</span>
                <span className="text-xs font-mono text-slate-500">{tokenizationStats.max_seq_length}</span>
              </div>
            </div>
          )}

          {/* Distribution summary with median */}
          <div className="mt-4 pt-4 border-t border-slate-700">
            <p className="text-sm text-slate-400 mb-2">Sequence Length Summary:</p>
            <div className="flex items-center gap-2 text-sm text-slate-200 font-mono">
              <span>Min: {tokenizationStats.min_seq_length}</span>
              <span className="text-slate-600">•</span>
              {tokenizationStats.median_seq_length !== undefined ? (
                <>
                  <span>Median: {tokenizationStats.median_seq_length.toFixed(1)}</span>
                  <span className="text-slate-600">•</span>
                </>
              ) : null}
              <span>Max: {tokenizationStats.max_seq_length}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Split Distribution */}
      {tokenizationStats.split_distribution && (
        <div className="bg-slate-800/50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-slate-100 mb-4">Split Distribution</h3>
          <div className="grid grid-cols-3 gap-4">
            {Object.entries(tokenizationStats.split_distribution).map(([splitName, count]) => (
              <div key={splitName} className="text-center">
                <div className="text-sm text-slate-400 mb-1 capitalize">{splitName}</div>
                <div className="text-2xl font-bold text-emerald-400">{count.toLocaleString()}</div>
                <div className="text-xs text-slate-500 mt-1">
                  {((count / dataset.num_samples!) * 100).toFixed(1)}% of total
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Tokenization Tab Component
function TokenizationTab({ dataset, onDatasetUpdate }: { dataset: Dataset; onDatasetUpdate?: (updatedDataset: Dataset) => void }) {
  const { models, fetchModels } = useModelsStore();
  const [tokenizerName, setTokenizerName] = useState('gpt2');
  const [maxLength, setMaxLength] = useState(512);
  const [stride, setStride] = useState(0);
  const [paddingStrategy, setPaddingStrategy] = useState<'max_length' | 'longest' | 'do_not_pad'>('max_length');
  const [truncationStrategy, setTruncationStrategy] = useState<'longest_first' | 'only_first' | 'only_second' | 'do_not_truncate'>('longest_first');
  const [addSpecialTokens, setAddSpecialTokens] = useState(true);
  const [returnAttentionMask, setReturnAttentionMask] = useState(true);
  const [enableCleaning, setEnableCleaning] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [progress, setProgress] = useState(dataset.progress || 0);
  const [progressMessage, setProgressMessage] = useState('');

  // Fetch models on mount
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Check if dataset is already tokenized
  const isTokenized = dataset.metadata?.tokenization !== undefined;
  const isProcessing = dataset.status === 'processing';
  const isError = dataset.status === 'error';
  // Allow tokenization for both READY and ERROR status (ERROR allows retry)
  const canTokenize = (dataset.status === 'ready' || dataset.status === 'error') && !isProcessing;

  // Initialize progress from dataset when component mounts or dataset changes
  useEffect(() => {
    if (dataset.progress !== undefined) {
      setProgress(dataset.progress);
    }
  }, [dataset.progress]);

  // WebSocket for real-time progress updates
  const { subscribe, unsubscribe } = useWebSocket();

  // Subscribe to progress updates when processing
  useEffect(() => {
    if (isProcessing && dataset.id) {
      const channel = `datasets/${dataset.id}/progress`;

      // Handler for 'progress' events
      const handleProgress = (data: any) => {
        console.log('Progress update received:', data);

        if (data.progress !== undefined) {
          setProgress(data.progress);
        }

        if (data.message) {
          setProgressMessage(data.message);
        }

        // Handle status changes in progress events
        if (data.status === 'ready' || data.event === 'completed') {
          setProgress(100);
          setProgressMessage('Tokenization complete!');
        }
      };

      // Handler for 'completed' events
      const handleCompleted = async (data: any) => {
        console.log('Tokenization completed:', data);
        setProgress(100);
        setProgressMessage(data.message || 'Tokenization complete!');

        // Refresh the dataset from API to get latest data including statistics
        if (onDatasetUpdate) {
          setTimeout(async () => {
            try {
              // Fetch the updated dataset
              const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${dataset.id}`);
              if (response.ok) {
                const updatedDataset = await response.json();
                console.log('Refreshed dataset after tokenization:', updatedDataset);
                onDatasetUpdate(updatedDataset);
              }
            } catch (error) {
              console.error('Failed to refresh dataset:', error);
            }
          }, 1500); // Give backend time to commit
        }
      };

      // Handler for 'error' events
      const handleError = (data: any) => {
        console.error('Tokenization error:', data);
        setError(data.message || 'Tokenization failed');
        setProgress(0);
      };

      // Subscribe to the channel (joins the Socket.IO room)
      // This is required for the client to receive events emitted to this room
      subscribe(channel, () => {
        console.log(`Joined channel: ${channel}`);
      });

      // Now subscribe to the actual event types
      // Backend emits: sio.emit('progress', data, room=channel)
      // Only clients in 'channel' room will receive these events
      subscribe('progress', handleProgress);
      subscribe('completed', handleCompleted);
      subscribe('error', handleError);

      return () => {
        // Unsubscribe from event handlers
        unsubscribe('progress', handleProgress);
        unsubscribe('completed', handleCompleted);
        unsubscribe('error', handleError);
        // Leave the room
        unsubscribe(channel);
      };
    }
    return undefined;
  }, [isProcessing, dataset.id, subscribe, unsubscribe, onDatasetUpdate]);

  // Build tokenizer options from available models + common tokenizers
  const commonTokenizers = [
    // Static common tokenizers
    { value: 'gpt2', label: 'GPT-2 (default)', description: 'OpenAI GPT-2 tokenizer', category: 'Common' },
    { value: 'gpt2-medium', label: 'GPT-2 Medium', description: '345M parameter model', category: 'Common' },
    { value: 'gpt2-large', label: 'GPT-2 Large', description: '774M parameter model', category: 'Common' },
    { value: 'bert-base-uncased', label: 'BERT Base Uncased', description: 'Google BERT base model', category: 'Common' },
    { value: 'bert-base-cased', label: 'BERT Base Cased', description: 'Case-sensitive BERT', category: 'Common' },
    { value: 'roberta-base', label: 'RoBERTa Base', description: 'Facebook RoBERTa model', category: 'Common' },
    { value: 'EleutherAI/gpt-neo-125M', label: 'GPT-Neo 125M', description: 'EleutherAI GPT-Neo', category: 'Common' },
    { value: 'EleutherAI/gpt-j-6B', label: 'GPT-J 6B', description: 'EleutherAI GPT-J', category: 'Common' },
  ];

  // Add models from the system as tokenizer options
  const modelTokenizers = models
    .filter((m) => m.status === 'ready')
    .map((m) => ({
      value: m.repo_id,
      label: m.name,
      description: `${m.architecture} - vocab: ${m.architecture_config?.vocab_size?.toLocaleString() || 'unknown'}`,
      category: 'Available Models',
    }));

  const handleTokenize = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setIsSubmitting(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/datasets/${dataset.id}/tokenize`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            tokenizer_name: tokenizerName,
            max_length: maxLength,
            stride: stride,
            padding: paddingStrategy,
            truncation: truncationStrategy,
            add_special_tokens: addSpecialTokens,
            return_attention_mask: returnAttentionMask,
            enable_cleaning: enableCleaning,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start tokenization');
      }

      const updatedDataset = await response.json();

      // Notify parent component of the update
      onDatasetUpdate?.(updatedDataset);

      // Initialize progress state
      setProgress(updatedDataset.progress || 0);
      setProgressMessage('Tokenization started...');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to tokenize dataset';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClearTokenization = async () => {
    if (!confirm('Clear tokenization data? This will remove tokenized files but keep the raw dataset. You can re-tokenize afterwards.')) {
      return;
    }

    setError(null);
    setSuccess(null);
    setIsSubmitting(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/datasets/${dataset.id}/tokenization`,
        {
          method: 'DELETE',
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to clear tokenization');
      }

      const updatedDataset = await response.json();

      // Notify parent component of the update
      onDatasetUpdate?.(updatedDataset);

      setSuccess('Tokenization cleared successfully');
      setProgress(0);
      setProgressMessage('');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to clear tokenization';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isProcessing) {
    return (
      <div className="text-center py-12 max-w-2xl mx-auto">
        <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-slate-700 border-t-emerald-500 mb-6"></div>
        <p className="text-slate-400 text-lg mb-4">Tokenization in progress...</p>

        {/* Progress Bar */}
        <div className="mb-4">
          <ProgressBar progress={progress} showPercentage={true} />
        </div>

        {/* Progress Message */}
        {progressMessage && (
          <p className="text-slate-500 text-sm mt-3">
            {progressMessage}
          </p>
        )}

        <p className="text-slate-500 text-xs mt-6">
          You can close this modal - tokenization will continue in the background
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Current Tokenization Status */}
      {isTokenized && (
        <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-6">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-emerald-500/20 rounded">
              <Zap className="w-5 h-5 text-emerald-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-emerald-400 mb-2">
                Dataset Already Tokenized
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-slate-400">Tokenizer:</span>
                  <span className="text-slate-200 ml-2 font-mono">
                    {dataset.metadata?.tokenization?.tokenizer_name ?? 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">Max Length:</span>
                  <span className="text-slate-200 ml-2 font-mono">
                    {dataset.metadata?.tokenization?.max_length ?? 'N/A'}
                  </span>
                </div>
                {dataset.metadata?.tokenization?.num_tokens !== undefined && (
                  <div>
                    <span className="text-slate-400">Total Tokens:</span>
                    <span className="text-slate-200 ml-2 font-mono">
                      {dataset.metadata.tokenization.num_tokens.toLocaleString()}
                    </span>
                  </div>
                )}
                {dataset.metadata?.tokenization?.avg_seq_length !== undefined && (
                  <div>
                    <span className="text-slate-400">Avg Length:</span>
                    <span className="text-slate-200 ml-2 font-mono">
                      {dataset.metadata.tokenization.avg_seq_length.toFixed(1)}
                    </span>
                  </div>
                )}
              </div>
              <p className="text-slate-400 text-sm mt-3">
                You can re-tokenize with different settings below to overwrite the current tokenization.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Error Status */}
      {isError && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-red-500/20 rounded">
              <AlertCircle className="w-5 h-5 text-red-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-red-400 mb-2">
                Tokenization Failed
              </h3>
              {dataset.error_message && (
                <p className="text-slate-300 text-sm mb-3">
                  {dataset.error_message}
                </p>
              )}
              <p className="text-slate-400 text-sm">
                You can retry with the same or different settings below, or clear the error to start fresh.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Clear Tokenization Button - Show for tokenized or error datasets */}
      {(isTokenized || isError) && (
        <div className="flex gap-3">
          <button
            type="button"
            onClick={handleClearTokenization}
            disabled={isSubmitting}
            className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed text-slate-200 font-medium rounded transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Clear Tokenization
          </button>
          {isTokenized && (
            <p className="text-sm text-slate-500 flex items-center">
              This will remove tokenized files but keep the raw dataset
            </p>
          )}
        </div>
      )}

      {/* Success/Error Messages */}
      {success && (
        <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-4">
          <p className="text-emerald-400 text-sm">{success}</p>
        </div>
      )}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Tokenization Form */}
      <form onSubmit={handleTokenize} className="bg-slate-800/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-slate-100 mb-4">
          {isTokenized ? 'Re-tokenize Dataset' : isError ? 'Retry Tokenization' : 'Tokenize Dataset'}
        </h3>

        <div className="space-y-6">
          {/* Tokenizer Selection */}
          <div>
            <label htmlFor="tokenizer" className="block text-sm font-medium text-slate-300 mb-2">
              Tokenizer Model
            </label>
            <select
              id="tokenizer"
              value={tokenizerName}
              onChange={(e) => setTokenizerName(e.target.value)}
              disabled={!canTokenize || isSubmitting}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {modelTokenizers.length > 0 && (
                <optgroup label="Available Models in System">
                  {modelTokenizers.map((tokenizer) => (
                    <option key={tokenizer.value} value={tokenizer.value}>
                      {tokenizer.label} - {tokenizer.description}
                    </option>
                  ))}
                </optgroup>
              )}
              <optgroup label="Common Tokenizers">
                {commonTokenizers.map((tokenizer) => (
                  <option key={tokenizer.value} value={tokenizer.value}>
                    {tokenizer.label} - {tokenizer.description}
                  </option>
                ))}
              </optgroup>
            </select>
            <p className="text-xs text-slate-500 mt-2">
              Select a tokenizer. Models from your system are shown first, followed by common tokenizers.
            </p>
          </div>

          {/* Custom Tokenizer Input */}
          <div>
            <label htmlFor="custom-tokenizer" className="block text-sm font-medium text-slate-300 mb-2">
              Or Enter Custom Tokenizer Name
            </label>
            <input
              id="custom-tokenizer"
              type="text"
              value={tokenizerName}
              onChange={(e) => setTokenizerName(e.target.value)}
              placeholder="e.g., EleutherAI/gpt-neo-2.7B"
              disabled={!canTokenize || isSubmitting}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <p className="text-xs text-slate-500 mt-2">
              Any valid HuggingFace tokenizer identifier (username/model-name)
            </p>
          </div>

          {/* Max Length */}
          <div>
            <label htmlFor="max-length" className="block text-sm font-medium text-slate-300 mb-2">
              Maximum Sequence Length
            </label>
            <div className="flex items-center gap-4">
              <input
                id="max-length"
                type="range"
                min="128"
                max="2048"
                step="128"
                value={maxLength}
                onChange={(e) => setMaxLength(Number(e.target.value))}
                disabled={!canTokenize || isSubmitting}
                className="flex-1"
              />
              <input
                type="number"
                min="1"
                max="8192"
                value={maxLength}
                onChange={(e) => setMaxLength(Number(e.target.value))}
                disabled={!canTokenize || isSubmitting}
                className="w-24 px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Maximum number of tokens per sequence. Longer sequences will be truncated. Range: 1-8192 tokens.
            </p>
          </div>

          {/* Padding Strategy */}
          <div>
            <label htmlFor="padding-strategy" className="block text-sm font-medium text-slate-300 mb-2">
              Padding Strategy
            </label>
            <select
              id="padding-strategy"
              value={paddingStrategy}
              onChange={(e) => setPaddingStrategy(e.target.value as 'max_length' | 'longest' | 'do_not_pad')}
              disabled={!canTokenize || isSubmitting}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <option value="max_length">Max Length - Pad all sequences to max_length</option>
              <option value="longest">Longest - Pad to longest sequence in batch</option>
              <option value="do_not_pad">Do Not Pad - No padding applied</option>
            </select>
            <p className="text-xs text-slate-500 mt-2">
              Padding strategy determines how sequences shorter than max_length are handled. "Max Length" pads all to max_length, "Longest" pads to longest in batch (dynamic), "Do Not Pad" disables padding.
            </p>
          </div>

          {/* Truncation Strategy */}
          <div>
            <label htmlFor="truncation-strategy" className="block text-sm font-medium text-slate-300 mb-2">
              Truncation Strategy
            </label>
            <select
              id="truncation-strategy"
              value={truncationStrategy}
              onChange={(e) => setTruncationStrategy(e.target.value as 'longest_first' | 'only_first' | 'only_second' | 'do_not_truncate')}
              disabled={!canTokenize || isSubmitting}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <option value="longest_first">Longest First - Truncate longest sequence first (default)</option>
              <option value="only_first">Only First - Truncate only first sequence</option>
              <option value="only_second">Only Second - Truncate only second sequence</option>
              <option value="do_not_truncate">Do Not Truncate - Disable truncation</option>
            </select>
            <p className="text-xs text-slate-500 mt-2">
              Controls truncation for sequences exceeding max_length. Useful for Q&A pairs or multi-sequence inputs.
            </p>
          </div>

          {/* Stride */}
          <div>
            <label htmlFor="stride" className="block text-sm font-medium text-slate-300 mb-2">
              Stride (Sliding Window Overlap)
            </label>
            <div className="flex items-center gap-4">
              <input
                id="stride"
                type="range"
                min="0"
                max={Math.floor(maxLength / 2)}
                step="32"
                value={stride}
                onChange={(e) => setStride(Number(e.target.value))}
                disabled={!canTokenize || isSubmitting}
                className="flex-1"
              />
              <input
                type="number"
                min="0"
                max={maxLength}
                value={stride}
                onChange={(e) => setStride(Math.min(Number(e.target.value), maxLength))}
                disabled={!canTokenize || isSubmitting}
                className="w-24 px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Number of overlapping tokens between consecutive sequences. 0 = no overlap. Useful for preserving context across boundaries.
            </p>
          </div>

          {/* Special Tokens Toggle */}
          <div>
            <label className="flex items-center justify-between cursor-pointer">
              <div>
                <span className="block text-sm font-medium text-slate-300 mb-1">
                  Add Special Tokens
                </span>
                <p className="text-xs text-slate-500">
                  Include BOS, EOS, PAD tokens - Recommended for most models
                </p>
              </div>
              <button
                type="button"
                onClick={() => setAddSpecialTokens(!addSpecialTokens)}
                disabled={!canTokenize || isSubmitting}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed ${
                  addSpecialTokens ? 'bg-emerald-600' : 'bg-slate-700'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    addSpecialTokens ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </label>
          </div>

          {/* Attention Mask Toggle */}
          <div>
            <label className="flex items-center justify-between cursor-pointer">
              <div>
                <span className="block text-sm font-medium text-slate-300 mb-1">
                  Return Attention Mask
                </span>
                <p className="text-xs text-slate-500">
                  Generate attention masks - Set to False to save memory if model doesn't use them
                </p>
              </div>
              <button
                type="button"
                onClick={() => setReturnAttentionMask(!returnAttentionMask)}
                disabled={!canTokenize || isSubmitting}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed ${
                  returnAttentionMask ? 'bg-emerald-600' : 'bg-slate-700'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    returnAttentionMask ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </label>
          </div>

          {/* Text Cleaning Toggle */}
          <div>
            <label className="flex items-center justify-between cursor-pointer">
              <div>
                <span className="block text-sm font-medium text-slate-300 mb-1">
                  Enable Text Cleaning
                </span>
                <p className="text-xs text-slate-500">
                  Remove HTML tags, control characters, excessive punctuation, normalize Unicode - Recommended for better feature quality
                </p>
              </div>
              <button
                type="button"
                onClick={() => setEnableCleaning(!enableCleaning)}
                disabled={!canTokenize || isSubmitting}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed ${
                  enableCleaning ? 'bg-emerald-600' : 'bg-slate-700'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    enableCleaning ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </label>
          </div>
        </div>
      </form>

      {/* Tokenization Preview */}
      <TokenizationPreview
        tokenizerName={tokenizerName}
        maxLength={maxLength}
        paddingStrategy={paddingStrategy}
        truncationStrategy={truncationStrategy}
        disabled={!canTokenize || isSubmitting}
      />

      {/* Tokenization Form - Action Section */}
      <form onSubmit={handleTokenize} className="bg-slate-800/50 rounded-lg p-6">
        <div className="space-y-6">
          {/* Error/Success Messages */}
          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {success && (
            <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
              <p className="text-emerald-400 text-sm">{success}</p>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={!canTokenize || isSubmitting || !tokenizerName}
            className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-medium rounded transition-colors"
          >
            <Zap className="w-5 h-5" />
            {isSubmitting ? 'Starting Tokenization...' : isTokenized ? 'Re-tokenize Dataset' : 'Start Tokenization'}
          </button>

          {!canTokenize && !isProcessing && (
            <p className="text-sm text-amber-400 text-center">
              Dataset must be in "ready" status to tokenize
            </p>
          )}
        </div>
      </form>

      {/* Information Panel */}
      <div className="bg-slate-800/30 border border-slate-700 rounded-lg p-6">
        <h4 className="text-sm font-semibold text-slate-300 mb-3">About Tokenization</h4>
        <div className="space-y-2 text-sm text-slate-400">
          <p>
            <strong className="text-slate-300">Tokenization</strong> converts text into numerical tokens that models can process.
          </p>
          <p>
            <strong className="text-slate-300">Tokenizer Selection:</strong> Different models use different tokenizers. Choose one compatible with your intended model.
          </p>
          <p>
            <strong className="text-slate-300">Max Length:</strong> Determines computational and memory requirements. Longer sequences = more compute.
          </p>
          <p>
            <strong className="text-slate-300">Stride:</strong> Enables sliding window approach for long documents, preserving context across chunk boundaries.
          </p>
        </div>
      </div>
    </div>
  );
}

// Helper Components
function InfoCard({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="text-sm text-slate-500 mb-1">{label}</dt>
      <dd className="text-slate-200 font-mono text-sm break-all">{value}</dd>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <div className="text-2xl font-bold text-emerald-400">{value}</div>
      <div className="text-sm text-slate-400 mt-1">{label}</div>
    </div>
  );
}

function ColoredStatCard({ label, value, color }: { label: string; value: string; color: 'emerald' | 'blue' | 'purple' | 'yellow' }) {
  const colorClasses = {
    emerald: 'text-emerald-400',
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    yellow: 'text-yellow-400',
  };

  return (
    <div className="text-center">
      <div className={`text-2xl font-bold ${colorClasses[color]}`}>{value}</div>
      <div className="text-sm text-slate-400 mt-1">{label}</div>
    </div>
  );
}
