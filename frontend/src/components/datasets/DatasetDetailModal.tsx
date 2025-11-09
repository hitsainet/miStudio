/**
 * DatasetDetailModal - Full-screen modal for dataset details.
 *
 * This component displays detailed information about a dataset with tabs
 * for samples, statistics, and tokenization settings.
 */

import { useState, useEffect } from 'react';
import { X, FileText, BarChart, Zap, ChevronLeft, ChevronRight } from 'lucide-react';
import { Dataset } from '../../types/dataset';
import { formatFileSize, formatDateTime } from '../../utils/formatters';
import { StatusBadge } from '../common/StatusBadge';
import { API_BASE_URL } from '../../config/api';
import { TokenizationsList } from './TokenizationsList';

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
  const [tokenizations, setTokenizations] = useState<any[]>([]);
  const [selectedTokenizationId, setSelectedTokenizationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTokenizations = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${dataset.id}/tokenizations`);
        if (!response.ok) throw new Error('Failed to fetch tokenizations');
        const result = await response.json();
        const readyTokenizations = result.data.filter((t: any) => t.status === 'ready');
        setTokenizations(readyTokenizations);
        if (readyTokenizations.length > 0 && !selectedTokenizationId) {
          setSelectedTokenizationId(readyTokenizations[0].id);
        }
      } catch (err) {
        console.error('Failed to load tokenizations:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchTokenizations();
  }, [dataset.id, selectedTokenizationId]);

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 border-t-emerald-500 mb-4"></div>
        <p className="text-slate-400">Loading tokenization statistics...</p>
      </div>
    );
  }

  if (tokenizations.length === 0) {
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

  const selectedTokenization = tokenizations.find(t => t.id === selectedTokenizationId);
  if (!selectedTokenization) return null;

  const tokenizationStats = {
    num_tokens: selectedTokenization.num_tokens,
    avg_seq_length: selectedTokenization.avg_seq_length,
    vocab_size: selectedTokenization.vocab_size,
    min_seq_length: selectedTokenization.avg_seq_length, // Fallback if not available
    max_seq_length: selectedTokenization.avg_seq_length, // Fallback if not available
  };

  return (
    <div className="space-y-6">
      {/* Tokenization Selector */}
      {tokenizations.length > 1 && (
        <div className="bg-slate-800/50 rounded-lg p-4">
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Select Tokenization
          </label>
          <select
            value={selectedTokenizationId || ''}
            onChange={(e) => setSelectedTokenizationId(e.target.value)}
            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500"
          >
            {tokenizations.map((tok) => (
              <option key={tok.id} value={tok.id}>
                {tok.tokenizer_repo_id} ({tok.vocab_size?.toLocaleString()} tokens)
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Token Statistics */}
      <div className="bg-slate-800/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-slate-100 mb-4">
          Token Statistics
          {tokenizations.length === 1 && (
            <span className="text-sm font-normal text-slate-400 ml-2">
              ({selectedTokenization.tokenizer_repo_id})
            </span>
          )}
        </h3>
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

      {/* Sequence Length Information */}
      <div className="bg-slate-800/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-slate-100 mb-4">Sequence Length</h3>
        <div className="text-center">
          <div className="text-4xl font-bold text-emerald-400 mb-2">
            {tokenizationStats.avg_seq_length.toFixed(1)}
          </div>
          <div className="text-sm text-slate-400">
            Average tokens per sample
          </div>
        </div>
      </div>
    </div>
  );
}
// Tokenization Tab Component
function TokenizationTab({ dataset }: { dataset: Dataset; onDatasetUpdate?: (updatedDataset: Dataset) => void }) {
  return (
    <div className="space-y-6">
      <div className="bg-slate-800/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-slate-100 mb-2">Tokenizations</h3>
        <p className="text-sm text-slate-400 mb-6">
          Manage different tokenizations of this dataset. Each tokenization uses a specific model's tokenizer
          and can be used independently for training.
        </p>
        <TokenizationsList datasetId={dataset.id} />
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
