/**
 * ExtractionJobCard Component
 *
 * Displays an individual extraction job with status, progress, and actions.
 * Can be expanded to show discovered features.
 */

import React, { useState, useEffect } from 'react';
import { Zap, Loader, CheckCircle, XCircle, Trash2, Clock, ChevronDown, ChevronUp, Search, ArrowUpDown, Star } from 'lucide-react';
import type { ExtractionStatusResponse, FeatureSearchRequest } from '../../types/features';
import { formatDistanceToNow, formatDuration, intervalToDuration } from 'date-fns';
import { useFeaturesStore } from '../../stores/featuresStore';
import { TokenHighlightCompact } from './TokenHighlight';
import { FeatureDetailModal } from './FeatureDetailModal';
import { COMPONENTS } from '../../config/brand';

interface ExtractionJobCardProps {
  extraction: ExtractionStatusResponse;
  onCancel?: () => void;
  onDelete?: () => void;
  onRetry?: () => void;
}

export const ExtractionJobCard: React.FC<ExtractionJobCardProps> = ({
  extraction,
  onCancel,
  onDelete,
  onRetry,
}) => {
  const isActive = extraction.status === 'queued' || extraction.status === 'extracting';
  const isCompleted = extraction.status === 'completed';
  const isFailed = extraction.status === 'failed';

  // Expandable features list state
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchDebounceTimer, setSearchDebounceTimer] = useState<number | null>(null);
  const [selectedFeatureId, setSelectedFeatureId] = useState<string | null>(null);

  // Get features store methods and state
  const {
    featuresByExtraction,
    featureListMetadata,
    searchFilters,
    isLoadingFeatures,
    featuresError,
    fetchExtractionFeatures,
    toggleFavorite,
    setSearchFilters,
  } = useFeaturesStore();

  const features = featuresByExtraction[extraction.id] || [];
  const metadata = featureListMetadata[extraction.id];
  const filters = searchFilters[extraction.id] || { sort_by: 'activation_freq', sort_order: 'desc', limit: 50, offset: 0 };

  // Load features when expanded and completed
  useEffect(() => {
    if (isExpanded && isCompleted && !features.length) {
      fetchExtractionFeatures(extraction.id, filters);
    }
  }, [isExpanded, isCompleted, extraction.id]);

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
      setSearchFilters(extraction.id, newFilters);
      fetchExtractionFeatures(extraction.id, newFilters);
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
    setSearchFilters(extraction.id, newFilters);
    fetchExtractionFeatures(extraction.id, newFilters);
  };

  /**
   * Handle sort order toggle.
   */
  const handleSortOrderToggle = () => {
    const newFilters: FeatureSearchRequest = {
      ...filters,
      sort_order: filters.sort_order === 'desc' ? 'asc' : 'desc',
    };
    setSearchFilters(extraction.id, newFilters);
    fetchExtractionFeatures(extraction.id, newFilters);
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
    setSearchFilters(extraction.id, newFilters);
    fetchExtractionFeatures(extraction.id, newFilters);
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
    setSearchFilters(extraction.id, newFilters);
    fetchExtractionFeatures(extraction.id, newFilters);
  };

  const handleNextPage = () => {
    if (!metadata || filters.offset! + filters.limit! >= metadata.total) return;

    const newFilters: FeatureSearchRequest = {
      ...filters,
      offset: filters.offset! + filters.limit!,
    };
    setSearchFilters(extraction.id, newFilters);
    fetchExtractionFeatures(extraction.id, newFilters);
  };

  // Calculate elapsed time
  const getElapsedTime = () => {
    const startTime = new Date(extraction.created_at);
    const endTime = extraction.completed_at ? new Date(extraction.completed_at) : new Date();

    const duration = intervalToDuration({ start: startTime, end: endTime });

    return formatDuration(duration, {
      format: ['hours', 'minutes', 'seconds'],
      zero: false,
      delimiter: ', '
    }) || 'Less than a second';
  };

  const getStatusBadge = () => {
    const baseClasses = 'px-3 py-1 rounded-full text-sm font-medium';

    switch (extraction.status) {
      case 'completed':
        return (
          <span className={`${baseClasses} bg-emerald-900/30 text-emerald-400 flex items-center gap-1`}>
            <CheckCircle className="w-4 h-4" />
            Completed
          </span>
        );
      case 'extracting':
        return (
          <span className={`${baseClasses} bg-blue-900/30 text-blue-400 flex items-center gap-1`}>
            <Loader className="w-4 h-4 animate-spin" />
            Extracting
          </span>
        );
      case 'queued':
        return (
          <span className={`${baseClasses} bg-yellow-900/30 text-yellow-400 flex items-center gap-1`}>
            <Clock className="w-4 h-4" />
            Queued
          </span>
        );
      case 'failed':
        return (
          <span className={`${baseClasses} bg-red-900/30 text-red-400 flex items-center gap-1`}>
            <XCircle className="w-4 h-4" />
            Failed
          </span>
        );
      default:
        return (
          <span className={`${baseClasses} bg-slate-800 text-slate-300`}>
            Unknown
          </span>
        );
    }
  };

  const progress = (extraction.progress || 0) * 100;

  return (
    <div className={`${COMPONENTS.card.base} p-6 hover:border-slate-700 transition-colors`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-start gap-3 flex-1">
          <Zap className="w-6 h-6 text-emerald-400 flex-shrink-0 mt-1" />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-semibold text-lg text-slate-100 truncate">
                {extraction.model_name || 'Unknown Model'} - {extraction.dataset_name || 'Unknown Dataset'}
              </h3>
              {getStatusBadge()}
              {/* Expand/Collapse button for completed extractions */}
              {isCompleted && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className={`p-1 rounded ml-auto ${COMPONENTS.button.ghost}`}
                  title={isExpanded ? 'Collapse features' : 'Expand features'}
                >
                  {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                </button>
              )}
            </div>
            <div className="text-sm text-slate-400 space-y-1">
              <p>Started {formatDistanceToNow(new Date(extraction.created_at), { addSuffix: true })}</p>
              {extraction.completed_at && (
                <>
                  <p>Completed {formatDistanceToNow(new Date(extraction.completed_at), { addSuffix: true })}</p>
                  <p className="text-emerald-400 font-medium">Elapsed: {getElapsedTime()}</p>
                </>
              )}
              {isActive && (
                <p className="text-blue-400 font-medium">Elapsed: {getElapsedTime()}</p>
              )}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 flex-shrink-0 ml-4">
          {isActive && onCancel && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('Are you sure you want to cancel this extraction?')) {
                  onCancel();
                }
              }}
              className={`p-2 rounded-lg ${COMPONENTS.button.ghost}`}
              title="Cancel extraction"
            >
              <XCircle className="w-5 h-5" />
            </button>
          )}
          {(isCompleted || isFailed) && onDelete && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('Are you sure you want to delete this extraction?')) {
                  onDelete();
                }
              }}
              className={`p-2 rounded-lg ${COMPONENTS.button.ghost}`}
              title="Delete extraction"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar for Active Extractions */}
      {isActive && extraction.progress !== null && extraction.progress !== undefined && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-slate-400">
              {extraction.features_extracted !== null && extraction.total_features !== null
                ? `${extraction.features_extracted.toLocaleString()} / ${extraction.total_features.toLocaleString()} features`
                : 'Processing...'}
            </span>
            <span className="text-emerald-400 font-medium">
              {progress.toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Statistics for Completed */}
      {isCompleted && extraction.statistics && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="bg-slate-800/50 rounded p-3">
            <div className="text-xs text-slate-400 mb-1">Features Found</div>
            <div className="text-lg font-bold text-emerald-400">
              {extraction.statistics.total_features?.toLocaleString() || 'N/A'}
            </div>
          </div>
          <div className="bg-slate-800/50 rounded p-3">
            <div className="text-xs text-slate-400 mb-1">Interpretable</div>
            <div className="text-lg font-bold text-blue-400">
              {extraction.statistics.interpretable_count !== undefined && extraction.statistics.total_features
                ? `${((extraction.statistics.interpretable_count / extraction.statistics.total_features) * 100).toFixed(1)}%`
                : 'N/A'}
            </div>
          </div>
          <div className="bg-slate-800/50 rounded p-3">
            <div className="text-xs text-slate-400 mb-1">Activation Rate</div>
            <div className="text-lg font-bold text-purple-400">
              {extraction.statistics.avg_activation_frequency !== undefined
                ? `${(extraction.statistics.avg_activation_frequency * 100).toFixed(2)}%`
                : 'N/A'}
            </div>
          </div>
        </div>
      )}

      {/* Error Message for Failed */}
      {isFailed && extraction.error_message && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          <div className="flex items-start gap-2">
            <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <div className="font-medium mb-1">Extraction Failed</div>
              <div className="text-red-300">{extraction.error_message}</div>
            </div>
          </div>
          {onRetry && (
            <button
              onClick={onRetry}
              className={`mt-3 w-full text-sm ${COMPONENTS.button.secondary}`}
            >
              Retry Extraction
            </button>
          )}
        </div>
      )}

      {/* Expandable Features List (only for completed extractions) */}
      {isExpanded && isCompleted && (
        <div className="mt-6 pt-6 border-t border-slate-800 space-y-4">
          <h4 className="text-lg font-semibold text-white flex items-center gap-2">
            <Zap className="w-5 h-5 text-emerald-400" />
            Discovered Features
            {metadata && (
              <span className="text-sm text-slate-400 font-normal">
                ({metadata.total.toLocaleString()} total)
              </span>
            )}
          </h4>

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
              className={`border border-slate-700 ${COMPONENTS.button.secondary}`}
              title={filters.sort_order === 'desc' ? 'Descending' : 'Ascending'}
            >
              <ArrowUpDown
                className={`w-4 h-4 transition-transform ${filters.sort_order === 'asc' ? 'rotate-180' : ''}`}
              />
            </button>

            {/* Favorites Filter */}
            <button
              onClick={handleFavoriteFilterToggle}
              className={`border rounded transition-colors ${
                filters.is_favorite
                  ? 'bg-yellow-600/20 border-yellow-600 text-yellow-400 px-3 py-2'
                  : COMPONENTS.button.secondary
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
                    <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Actions</th>
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
                            className={`p-1 rounded ${COMPONENTS.button.ghost}`}
                          >
                            <Star
                              className={`w-4 h-4 ${
                                feature.is_favorite
                                  ? 'fill-yellow-400 text-yellow-400'
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
          {features.length > 0 && metadata && (
            <div className="flex items-center justify-between border-t border-slate-700 pt-4">
              <div className="text-sm text-slate-400">
                Showing {features.length} of {metadata.total} features
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handlePreviousPage}
                  disabled={filters.offset === 0}
                  className={`px-3 py-1 text-sm ${COMPONENTS.button.secondary}`}
                >
                  Previous
                </button>
                <button
                  onClick={handleNextPage}
                  disabled={filters.offset! + filters.limit! >= metadata.total}
                  className={`px-3 py-1 text-sm ${COMPONENTS.button.secondary}`}
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
        </div>
      )}

      {/* Configuration Details */}
      <div className="mt-4 pt-4 border-t border-slate-800">
        <details className="text-sm">
          <summary className="cursor-pointer text-slate-400 hover:text-slate-300 transition-colors">
            Configuration
          </summary>
          <div className="mt-2 space-y-1 text-slate-400">
            <div>Evaluation Samples: <span className="text-white">{extraction.config.evaluation_samples?.toLocaleString() || 'N/A'}</span></div>
            <div>Top-K Examples: <span className="text-white">{extraction.config.top_k_examples || 'N/A'}</span></div>
            <div className="text-xs text-slate-500 mt-2">ID: {extraction.id}</div>
          </div>
        </details>
      </div>

      {/* Feature Detail Modal */}
      {selectedFeatureId && (
        <FeatureDetailModal
          featureId={selectedFeatureId}
          trainingId={extraction.training_id}
          onClose={() => setSelectedFeatureId(null)}
        />
      )}
    </div>
  );
};
