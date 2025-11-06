/**
 * FeaturesPanel Component
 *
 * Main panel for feature discovery and browsing.
 * Shows different views based on extraction status:
 * - Configuration: Start extraction (if not started)
 * - Progress: Extraction in progress
 * - Browser: Feature list with search/filter/sort (if completed)
 *
 * Matches Mock UI specification (lines 2359-2800+).
 */

import React, { useEffect, useState } from 'react';
import { Zap, Search, ArrowUpDown, Star, XCircle, Trash2 } from 'lucide-react';
import { useFeaturesStore } from '../../stores/featuresStore';
import type { Training } from '../../types/training';
import type { FeatureSearchRequest } from '../../types/features';
import { TokenHighlightCompact } from './TokenHighlight';
import { FeatureDetailModal } from './FeatureDetailModal';
import { ResourceConfigPanel } from './ResourceConfigPanel';

interface FeaturesPanelProps {
  training: Training;
}

/**
 * FeaturesPanel Component.
 */
export const FeaturesPanel: React.FC<FeaturesPanelProps> = ({ training }) => {
  const {
    extractionStatus,
    featuresByTraining,
    featureListMetadata,
    searchFilters,
    isLoadingExtraction,
    isLoadingFeatures,
    extractionError,
    featuresError,
    startExtraction,
    cancelExtraction,
    deleteExtraction,
    getExtractionStatus,
    fetchFeatures,
    toggleFavorite,
    setSearchFilters,
  } = useFeaturesStore();

  // Local state for extraction config
  const [evaluationSamples, setEvaluationSamples] = useState(10000);
  const [topKExamples, setTopKExamples] = useState(100);
  const [labelingMethod, setLabelingMethod] = useState<'pattern' | 'local' | 'openai'>('pattern');
  const [localLabelingModel, setLocalLabelingModel] = useState<'phi3' | 'llama' | 'qwen'>('phi3');
  const [openaiApiKey, setOpenaiApiKey] = useState('');
  const [openaiModel, setOpenaiModel] = useState<'gpt4-mini' | 'gpt4' | 'gpt35'>('gpt4-mini');
  const [resourceConfig, setResourceConfig] = useState<{
    batch_size?: number;
    num_workers?: number;
    db_commit_batch?: number;
  }>({});

  // Local state for search
  const [searchQuery, setSearchQuery] = useState('');
  const [searchDebounceTimer, setSearchDebounceTimer] = useState<number | null>(null);

  // Local state for feature detail modal
  const [selectedFeatureId, setSelectedFeatureId] = useState<string | null>(null);

  const status = extractionStatus[training.id];
  const features = featuresByTraining[training.id] || [];
  const metadata = featureListMetadata[training.id];
  const filters = searchFilters[training.id] || { sort_by: 'activation_freq', sort_order: 'desc', limit: 50, offset: 0 };

  // Load extraction status on mount
  useEffect(() => {
    getExtractionStatus(training.id);
  }, [training.id]);

  // Load features if extraction is completed
  useEffect(() => {
    if (status?.status === 'completed') {
      fetchFeatures(training.id, filters);
    }
  }, [status?.status, training.id]);

  /**
   * Handle start extraction button click.
   */
  const handleStartExtraction = async () => {
    try {
      await startExtraction(training.id, {
        evaluation_samples: evaluationSamples,
        top_k_examples: topKExamples,
        labeling_method: labelingMethod,
        local_labeling_model: labelingMethod === 'local' ? localLabelingModel : undefined,
        openai_api_key: labelingMethod === 'openai' && openaiApiKey ? openaiApiKey : undefined,
        openai_model: labelingMethod === 'openai' ? openaiModel : undefined,
        ...resourceConfig,
      });
    } catch (error) {
      console.error('Failed to start extraction:', error);
      // Refresh extraction status to show current state (e.g., existing failed extraction)
      await getExtractionStatus(training.id);
    }
  };

  /**
   * Handle search input change with debouncing.
   */
  const handleSearchChange = (value: string) => {
    setSearchQuery(value);

    // Clear existing timer
    if (searchDebounceTimer) {
      clearTimeout(searchDebounceTimer);
    }

    // Set new timer for 300ms debounce
    const timer = setTimeout(() => {
      const newFilters: FeatureSearchRequest = {
        ...filters,
        search: value || null,
        offset: 0, // Reset to first page on new search
      };
      setSearchFilters(training.id, newFilters);
      fetchFeatures(training.id, newFilters);
    }, 300);

    setSearchDebounceTimer(timer);
  };

  /**
   * Handle sort change.
   */
  const handleSortChange = (sortBy: 'activation_freq' | 'interpretability' | 'feature_id') => {
    const newFilters: FeatureSearchRequest = {
      ...filters,
      sort_by: sortBy,
    };
    setSearchFilters(training.id, newFilters);
    fetchFeatures(training.id, newFilters);
  };

  /**
   * Handle sort order toggle.
   */
  const handleSortOrderToggle = () => {
    const newFilters: FeatureSearchRequest = {
      ...filters,
      sort_order: filters.sort_order === 'desc' ? 'asc' : 'desc',
    };
    setSearchFilters(training.id, newFilters);
    fetchFeatures(training.id, newFilters);
  };

  /**
   * Handle favorite filter toggle.
   */
  const handleFavoriteFilterToggle = () => {
    const newFilters: FeatureSearchRequest = {
      ...filters,
      is_favorite: filters.is_favorite === true ? null : true,
      offset: 0,
    };
    setSearchFilters(training.id, newFilters);
    fetchFeatures(training.id, newFilters);
  };

  /**
   * Handle favorite toggle for a feature.
   */
  const handleToggleFavorite = async (featureId: string, currentFavorite: boolean, event: React.MouseEvent) => {
    event.stopPropagation();
    try {
      await toggleFavorite(featureId, !currentFavorite);
    } catch (error) {
      console.error('Failed to toggle favorite:', error);
    }
  };

  /**
   * Handle pagination.
   */
  const handlePreviousPage = () => {
    if (!metadata || filters.offset === 0) return;

    const newFilters: FeatureSearchRequest = {
      ...filters,
      offset: Math.max(0, filters.offset! - filters.limit!),
    };
    setSearchFilters(training.id, newFilters);
    fetchFeatures(training.id, newFilters);
  };

  const handleNextPage = () => {
    if (!metadata || filters.offset! + filters.limit! >= metadata.total) return;

    const newFilters: FeatureSearchRequest = {
      ...filters,
      offset: filters.offset! + filters.limit!,
    };
    setSearchFilters(training.id, newFilters);
    fetchFeatures(training.id, newFilters);
  };

  // Show extraction configuration if not started
  if (!status || status.status === 'queued') {
    return (
      <div className="space-y-4">
        <div className="bg-slate-900/50 rounded-lg p-6 border border-slate-800">
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-white mb-2">Extract Features</h3>
            <p className="text-sm text-slate-400">
              Training complete. Extract interpretable features from the trained encoder.
            </p>
          </div>

          {/* Configuration Inputs */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Evaluation Samples</label>
              <input
                type="number"
                value={evaluationSamples}
                onChange={(e) => setEvaluationSamples(Number(e.target.value))}
                min={1000}
                max={100000}
                step={1000}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Top-K Examples per Feature</label>
              <input
                type="number"
                value={topKExamples}
                onChange={(e) => setTopKExamples(Number(e.target.value))}
                min={10}
                max={1000}
                step={10}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
              />
            </div>
          </div>

          {/* Labeling Configuration */}
          <div className="mb-4 p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
            <h4 className="text-sm font-semibold text-slate-300 mb-3">Feature Labeling</h4>

            {/* Labeling Method Selector */}
            <div className="mb-3">
              <label className="block text-xs text-slate-400 mb-1">Labeling Method</label>
              <select
                value={labelingMethod}
                onChange={(e) => setLabelingMethod(e.target.value as 'pattern' | 'local' | 'openai')}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
              >
                <option value="pattern">Pattern Matching (fast, simple patterns)</option>
                <option value="local">Local LLM (slow, high quality, zero cost)</option>
                <option value="openai">OpenAI API (fast, high quality, costs money)</option>
              </select>
              <p className="text-xs text-slate-500 mt-1">
                {labelingMethod === 'pattern' && 'Uses 8 hardcoded patterns for quick labeling'}
                {labelingMethod === 'local' && 'Uses Phi-3-mini locally (~5.5 hours for 16K features, ~2GB VRAM)'}
                {labelingMethod === 'openai' && 'Uses GPT-4o-mini API (~55 minutes for 16K features, ~$1.64 cost)'}
              </p>
            </div>

            {/* Local Model Selector (shown when labeling_method=local) */}
            {labelingMethod === 'local' && (
              <div className="mb-3">
                <label className="block text-xs text-slate-400 mb-1">Local Model</label>
                <select
                  value={localLabelingModel}
                  onChange={(e) => setLocalLabelingModel(e.target.value as 'phi3' | 'llama' | 'qwen')}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                >
                  <option value="phi3">Phi-3-mini (recommended, 3.8B params)</option>
                  <option value="llama">Llama 3.2 3B</option>
                  <option value="qwen">Qwen 2.5 3B</option>
                </select>
              </div>
            )}

            {/* OpenAI Configuration (shown when labeling_method=openai) */}
            {labelingMethod === 'openai' && (
              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">OpenAI API Key</label>
                  <input
                    type="password"
                    value={openaiApiKey}
                    onChange={(e) => setOpenaiApiKey(e.target.value)}
                    placeholder="sk-..."
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Leave blank to use OPENAI_API_KEY environment variable
                  </p>
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">OpenAI Model</label>
                  <select
                    value={openaiModel}
                    onChange={(e) => setOpenaiModel(e.target.value as 'gpt4-mini' | 'gpt4' | 'gpt35')}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                  >
                    <option value="gpt4-mini">GPT-4o-mini (recommended, $0.0001/feature)</option>
                    <option value="gpt4">GPT-4 Turbo (higher quality, more expensive)</option>
                    <option value="gpt35">GPT-3.5 Turbo (faster, lower quality)</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Resource Configuration Panel */}
          <div className="mb-4">
            <ResourceConfigPanel
              trainingId={training.id}
              evaluationSamples={evaluationSamples}
              topKExamples={topKExamples}
              onConfigChange={setResourceConfig}
            />
          </div>

          {/* Extract Button */}
          <button
            onClick={handleStartExtraction}
            disabled={isLoadingExtraction}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded transition-colors"
          >
            <Zap className="w-5 h-5" />
            {isLoadingExtraction ? 'Starting Extraction...' : 'Extract Features'}
          </button>

          {extractionError && (
            <div className="mt-4 p-3 bg-red-900/20 border border-red-700 rounded text-red-400 text-sm">
              {extractionError}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Show progress if extracting
  if (status.status === 'extracting') {
    const progress = (status.progress || 0) * 100;

    return (
      <div className="space-y-4">
        <div className="bg-slate-900/50 rounded-lg p-6 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Extracting Features...</h3>
            <button
              onClick={async () => {
                if (window.confirm('Are you sure you want to cancel this extraction?')) {
                  try {
                    await cancelExtraction(training.id);
                  } catch (error) {
                    console.error('Failed to cancel extraction:', error);
                  }
                }
              }}
              className="flex items-center gap-1 px-3 py-1 text-sm bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-700 rounded transition-colors"
            >
              <XCircle className="w-4 h-4" />
              Cancel
            </button>
          </div>

          {/* Progress Bar */}
          <div className="mb-4">
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {/* Progress Info */}
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">
              Processing activation patterns...
            </span>
            <span className="text-emerald-400 font-medium">
              {progress.toFixed(1)}%
            </span>
          </div>

          {status.features_extracted !== null && status.total_features !== null && (
            <div className="mt-2 text-sm text-slate-400">
              {status.features_extracted} / {status.total_features} features
            </div>
          )}
        </div>
      </div>
    );
  }

  // Show error if failed
  if (status.status === 'failed') {
    return (
      <div className="space-y-4">
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-red-400 mb-2">Extraction Failed</h3>
          <p className="text-sm text-red-300">
            {status.error_message || 'An error occurred during feature extraction'}
          </p>
          <div className="flex gap-2 mt-4">
            <button
              onClick={() => getExtractionStatus(training.id)}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded transition-colors"
            >
              Retry
            </button>
            <button
              onClick={async () => {
                if (window.confirm('Are you sure you want to delete this extraction?')) {
                  try {
                    await deleteExtraction(status.id, training.id);
                  } catch (error) {
                    console.error('Failed to delete extraction:', error);
                  }
                }
              }}
              className="flex items-center gap-1 px-4 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-700 rounded transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Delete
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Show feature browser if completed
  if (status.status === 'completed' && metadata) {
    const { statistics } = metadata;

    return (
      <div className="space-y-4">
        {/* Statistics Cards */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-800/50 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Features Found</div>
            <div className="text-2xl font-bold text-emerald-400">
              {statistics.total_features.toLocaleString()}
            </div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Interpretable</div>
            <div className="text-2xl font-bold text-blue-400">
              {statistics.interpretable_percentage.toFixed(1)}%
            </div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Activation Rate</div>
            <div className="text-2xl font-bold text-purple-400">
              {(statistics.avg_activation_frequency * 100).toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex gap-3">
          {/* Search Input */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => handleSearchChange(e.target.value)}
              placeholder="Search features..."
              className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
          </div>

          {/* Sort Dropdown */}
          <select
            value={filters.sort_by || 'activation_freq'}
            onChange={(e) => handleSortChange(e.target.value as any)}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
          >
            <option value="activation_freq">Activation Freq</option>
            <option value="interpretability">Interpretability</option>
            <option value="feature_id">Feature ID</option>
          </select>

          {/* Sort Order Toggle */}
          <button
            onClick={handleSortOrderToggle}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white hover:bg-slate-700 transition-colors"
            title={filters.sort_order === 'desc' ? 'Descending' : 'Ascending'}
          >
            <ArrowUpDown
              className={`w-4 h-4 transition-transform ${filters.sort_order === 'asc' ? 'rotate-180' : ''}`}
            />
          </button>

          {/* Favorites Filter */}
          <button
            onClick={handleFavoriteFilterToggle}
            className={`px-3 py-2 border rounded transition-colors ${
              filters.is_favorite
                ? 'bg-yellow-600/20 border-yellow-600 text-yellow-400'
                : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'
            }`}
            title="Filter favorites"
          >
            <Star className={`w-4 h-4 ${filters.is_favorite ? 'fill-current' : ''}`} />
          </button>
        </div>

        {/* Feature Table */}
        <div className="bg-slate-800/50 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-800">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">ID</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Label</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Example Context</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Activation Freq</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Interpretability</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Of Interest</th>
                </tr>
              </thead>
              <tbody>
                {isLoadingFeatures ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-slate-400">
                      Loading features...
                    </td>
                  </tr>
                ) : features.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-slate-400">
                      No features match your search
                    </td>
                  </tr>
                ) : (
                  features.map((feature) => (
                    <tr
                      key={feature.id}
                      onClick={() => setSelectedFeatureId(feature.id)}
                      className="border-t border-slate-700 hover:bg-slate-800/30 cursor-pointer"
                    >
                      <td className="px-4 py-3">
                        <span className="font-mono text-sm text-slate-400">
                          #{feature.neuron_index}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-sm text-white">{feature.name}</span>
                      </td>
                      <td className="px-4 py-3">
                        {feature.example_context && (
                          <TokenHighlightCompact
                            tokens={feature.example_context.tokens}
                            activations={feature.example_context.activations}
                            maxActivation={feature.example_context.max_activation}
                            maxTokens={15}
                          />
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-sm text-emerald-400">
                          {(feature.activation_frequency * 100).toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-sm text-blue-400">
                          {(feature.interpretability_score * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={(e) => handleToggleFavorite(feature.id, feature.is_favorite, e)}
                          className="p-1 hover:bg-slate-700 rounded transition-colors"
                        >
                          <Star
                            className={`w-4 h-4 ${
                              feature.is_favorite
                                ? 'fill-emerald-500 text-emerald-500'
                                : 'text-slate-500'
                            }`}
                          />
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Pagination */}
        {features.length > 0 && (
          <div className="flex items-center justify-between border-t border-slate-700 pt-4">
            <div className="text-sm text-slate-400">
              Showing {features.length} of {metadata.total} features
            </div>
            <div className="flex gap-2">
              <button
                onClick={handlePreviousPage}
                disabled={filters.offset === 0}
                className="px-3 py-1 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:text-slate-600 text-sm text-white rounded transition-colors"
              >
                Previous
              </button>
              <button
                onClick={handleNextPage}
                disabled={filters.offset! + filters.limit! >= metadata.total}
                className="px-3 py-1 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:text-slate-600 text-sm text-white rounded transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        )}

        {featuresError && (
          <div className="mt-4 p-3 bg-red-900/20 border border-red-700 rounded text-red-400 text-sm">
            {featuresError}
          </div>
        )}

        {/* Feature Detail Modal */}
        {selectedFeatureId && (
          <FeatureDetailModal
            featureId={selectedFeatureId}
            trainingId={training.id}
            onClose={() => setSelectedFeatureId(null)}
          />
        )}
      </div>
    );
  }

  return null;
};
