/**
 * TokenizationsList - Display and manage tokenizations for a dataset.
 *
 * Shows all available tokenizations for a dataset, their status, and provides
 * controls to create or delete tokenizations.
 */

import { useEffect, useState } from 'react';
import { CheckCircle, Loader, AlertCircle, Plus, Trash2, Hash, X } from 'lucide-react';
import { TokenizationStatus } from '../../types/dataset';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useModelsStore } from '../../stores/modelsStore';
import { COMPONENTS } from '../../config/brand';

interface TokenizationsListProps {
  datasetId: string;
}

export function TokenizationsList({ datasetId }: TokenizationsListProps) {
  const { tokenizations, fetchTokenizations, deleteTokenization, cancelTokenization, createTokenization } = useDatasetsStore();
  const { models, fetchModels } = useModelsStore();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const datasetTokenizations = tokenizations[datasetId] || [];

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
      });
      setShowCreateForm(false);
      setSelectedModelId('');
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
                {tokenization.status === TokenizationStatus.READY && (
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
              {(tokenization.status === TokenizationStatus.PROCESSING || tokenization.status === TokenizationStatus.QUEUED) && tokenization.progress !== undefined && (
                <div className="pt-2">
                  <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                    <span>Progress</span>
                    <span>{tokenization.progress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-all duration-300"
                      style={{ width: `${tokenization.progress}%` }}
                    />
                  </div>
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
