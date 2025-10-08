/**
 * DatasetDetailModal - Full-screen modal for dataset details.
 *
 * This component displays detailed information about a dataset with tabs
 * for samples, statistics, and tokenization settings.
 */

import React, { useState } from 'react';
import { X, FileText, BarChart, Settings } from 'lucide-react';
import { Dataset } from '../../types/dataset';
import { formatFileSize, formatDateTime } from '../../utils/formatters';
import { StatusBadge } from '../common/StatusBadge';

interface DatasetDetailModalProps {
  dataset: Dataset;
  onClose: () => void;
}

type TabType = 'overview' | 'samples' | 'statistics' | 'tokenization';

export function DatasetDetailModal({ dataset, onClose }: DatasetDetailModalProps) {
  const [activeTab, setActiveTab] = useState<TabType>('overview');

  const tabs = [
    { id: 'overview' as TabType, label: 'Overview', icon: FileText },
    { id: 'samples' as TabType, label: 'Samples', icon: FileText },
    { id: 'statistics' as TabType, label: 'Statistics', icon: BarChart },
    { id: 'tokenization' as TabType, label: 'Tokenization', icon: Settings },
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
          {activeTab === 'statistics' && <StatisticsTab dataset={dataset} />}
          {activeTab === 'tokenization' && <TokenizationTab dataset={dataset} />}
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
            {dataset.num_tokens && (
              <StatCard label="Tokens" value={dataset.num_tokens.toLocaleString()} />
            )}
            {dataset.avg_seq_length && (
              <StatCard
                label="Avg Length"
                value={dataset.avg_seq_length.toFixed(1)}
              />
            )}
            {dataset.vocab_size && (
              <StatCard label="Vocab Size" value={dataset.vocab_size.toLocaleString()} />
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

// Samples Tab Component (Placeholder)
function SamplesTab({ dataset }: { dataset: Dataset }) {
  return (
    <div className="text-center py-12">
      <FileText className="w-12 h-12 text-slate-600 mx-auto mb-4" />
      <p className="text-slate-400 text-lg">Sample browser coming in next phase</p>
      <p className="text-slate-500 mt-2">
        Will display paginated dataset samples with search and filtering
      </p>
    </div>
  );
}

// Statistics Tab Component (Placeholder)
function StatisticsTab({ dataset }: { dataset: Dataset }) {
  return (
    <div className="text-center py-12">
      <BarChart className="w-12 h-12 text-slate-600 mx-auto mb-4" />
      <p className="text-slate-400 text-lg">Statistics visualizations coming in next phase</p>
      <p className="text-slate-500 mt-2">
        Will display token distribution histograms and detailed analytics
      </p>
    </div>
  );
}

// Tokenization Tab Component (Placeholder)
function TokenizationTab({ dataset }: { dataset: Dataset }) {
  return (
    <div className="text-center py-12">
      <Settings className="w-12 h-12 text-slate-600 mx-auto mb-4" />
      <p className="text-slate-400 text-lg">Tokenization settings coming in next phase</p>
      <p className="text-slate-500 mt-2">
        Will allow configuration of tokenization parameters and execution
      </p>
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
