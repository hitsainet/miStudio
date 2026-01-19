/**
 * ExtractionJobCard Component
 *
 * Displays an individual extraction job with status, progress, and actions.
 * Can be expanded to show discovered features.
 *
 * IMPORTANT: This component contains the EXCLUSIVE features browser for the application.
 * The features browser is ONLY accessible from the Extractions tab via this component.
 * Users must navigate to the Extractions tab to browse, search, filter, and analyze
 * discovered features from their extraction jobs.
 *
 * This ensures a single, centralized location for all feature viewing and management,
 * eliminating redundancy and providing a consistent user experience.
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Zap, Loader, CheckCircle, XCircle, Trash2, Clock, ChevronDown, ChevronUp, Search, ArrowUpDown, Star, ArrowUp, ArrowDown, RefreshCw, List, Layers, Activity, Copy, Check, Brain, StopCircle, Play, RotateCcw } from 'lucide-react';
import type { ExtractionStatusResponse, FeatureSearchRequest } from '../../types/features';
import { triggerNlpAnalysis, cancelNlpAnalysis, resetNlpAnalysis } from '../../api/models';
import { format, intervalToDuration } from 'date-fns';
import { useFeaturesStore } from '../../stores/featuresStore';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { TokenHighlightCompact } from './TokenHighlight';
import { FeatureDetailModal } from './FeatureDetailModal';
import { StartLabelingButton } from '../labeling/StartLabelingButton';
import { COMPONENTS } from '../../config/brand';
import { formatL0Absolute, formatActivation, getActivationFrequencyColor, getMaxActivationColor } from '../../utils/formatters';

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
  const isActive = extraction.status === 'queued' || extraction.status === 'extracting' || extraction.status === 'finalizing';
  const isCompleted = extraction.status === 'completed';
  const isFailed = extraction.status === 'failed';
  const isDeleting = extraction.status === 'deleting';

  // Expandable features list state
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchDebounceTimer, setSearchDebounceTimer] = useState<number | null>(null);
  const [selectedFeatureId, setSelectedFeatureId] = useState<string | null>(null);

  // Range filter state (local inputs with debounced API calls)
  const [rangeFilterInputs, setRangeFilterInputs] = useState({
    min_activation_freq: '',
    max_activation_freq: '',
    min_max_activation: '',
    max_max_activation: '',
  });
  const [rangeDebounceTimer, setRangeDebounceTimer] = useState<number | null>(null);

  // Sort state (client-side sorting)
  type SortColumn = 'id' | 'label' | 'category' | 'description' | 'activation_frequency' | 'max_activation' | 'is_favorite';
  const [sortColumn, setSortColumn] = useState<SortColumn>('id');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  // Sort scope: 'page' = sort only current page, 'all' = sort entire dataset (Note: 'all' not yet implemented - requires backend support)
  const [sortScope, setSortScope] = useState<'page' | 'all'>('page');

  // Live metrics state (similar to TrainingCard)
  const [showMetrics, setShowMetrics] = useState(false);
  const [logsCopied, setLogsCopied] = useState(false);

  // NLP processing state
  const [isNlpProcessing, setIsNlpProcessing] = useState(false);
  const [isNlpCancelling, setIsNlpCancelling] = useState(false);
  const [isNlpResetting, setIsNlpResetting] = useState(false);
  const [nlpError, setNlpError] = useState<string | null>(null);

  // Metrics history for live logs (keep last 20 entries)
  interface MetricsEntry {
    timestamp: string;
    batch: number;
    totalBatches: number;
    samplesProcessed: number;
    totalSamples: number;
    samplesPerSecond: number;
    etaSeconds: number;
    featuresInHeap: number;
    heapExamplesCount: number;
    progress: number;
  }
  const [metricsHistory, setMetricsHistory] = useState<MetricsEntry[]>([]);
  const lastBatchRef = useRef<number>(-1);

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
    updateExtractionById,
  } = useFeaturesStore();

  // Get trainings store methods and state
  const { trainings, fetchTraining } = useTrainingsStore();

  // Helper to get model name
  const getModelDisplayName = (): string => {
    if (extraction.model_name) return extraction.model_name;
    return 'Unknown Model';
  };

  const features = featuresByExtraction[extraction.id] || [];
  const metadata = featureListMetadata[extraction.id];
  const filters = searchFilters[extraction.id] || { sort_by: 'feature_id', sort_order: 'asc', limit: 50, offset: 0 };

  // Client-side sorting logic (only used when sortScope is 'page')
  const sortedFeatures = useMemo(() => {
    if (features.length === 0) return [];

    // If sorting dataset-wide, don't apply client-side sorting
    // (backend will handle it)
    if (sortScope === 'all') {
      return features;
    }

    // Apply client-side sorting for page-only scope
    const sorted = [...features].sort((a, b) => {
      let aVal: any;
      let bVal: any;

      switch (sortColumn) {
        case 'id':
          aVal = a.neuron_index;
          bVal = b.neuron_index;
          break;
        case 'label':
          aVal = (a.name || '').toLowerCase();
          bVal = (b.name || '').toLowerCase();
          break;
        case 'category':
          aVal = (a.category || '').toLowerCase();
          bVal = (b.category || '').toLowerCase();
          break;
        case 'description':
          aVal = (a.description || '').toLowerCase();
          bVal = (b.description || '').toLowerCase();
          break;
        case 'activation_frequency':
          aVal = a.activation_frequency;
          bVal = b.activation_frequency;
          break;
        case 'max_activation':
          aVal = a.max_activation;
          bVal = b.max_activation;
          break;
        case 'is_favorite':
          // Favorites first (true > false), then by neuron_index
          if (a.is_favorite === b.is_favorite) {
            return a.neuron_index - b.neuron_index;
          }
          aVal = a.is_favorite ? 1 : 0;
          bVal = b.is_favorite ? 1 : 0;
          break;
        default:
          return 0;
      }

      // Compare values
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
      } else {
        // String comparison
        if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1;
        if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1;
        return 0;
      }
    });

    return sorted;
  }, [features, sortColumn, sortOrder, sortScope]);

  // Map client-side sort columns to backend sort field names
  // Backend supports: "activation_freq", "max_activation", "feature_id"
  const mapSortColumnToBackend = (column: SortColumn): 'activation_freq' | 'max_activation' | 'feature_id' => {
    switch (column) {
      case 'id':
        return 'feature_id';
      case 'activation_frequency':
        return 'activation_freq';
      case 'max_activation':
        return 'max_activation';
      // Backend doesn't support sorting by these fields - fall back to feature_id
      case 'label':
      case 'category':
      case 'description':
      case 'is_favorite':
      default:
        return 'feature_id';
    }
  };

  // Handle sort column click
  const handleSort = (column: SortColumn) => {
    if (sortScope === 'page') {
      // Page-only sorting: Toggle client-side sort state
      if (sortColumn === column) {
        setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
      } else {
        setSortColumn(column);
        setSortOrder('asc');
      }
    } else {
      // Dataset-wide sorting: Update backend filters
      const newOrder = (sortColumn === column && filters.sort_order === 'asc') ? 'desc' : 'asc';
      const newFilters: FeatureSearchRequest = {
        ...filters,
        sort_by: mapSortColumnToBackend(column),
        sort_order: newOrder,
        offset: 0, // Reset to first page when changing sort
      };

      // Update local state to show visual feedback
      setSortColumn(column);
      setSortOrder(newOrder);

      // Fetch with new sort order from backend
      setSearchFilters(extraction.id, newFilters);
      fetchExtractionFeatures(extraction.id, newFilters);
    }
  };

  // Find the associated training job
  const training = trainings.find((t) => t.id === extraction.training_id);

  // Sortable Column Header Component
  const SortableColumnHeader: React.FC<{ column: SortColumn; label: string; className?: string }> = ({
    column,
    label,
    className = '',
  }) => {
    const isActive = sortColumn === column;
    const Icon = isActive ? (sortOrder === 'asc' ? ArrowUp : ArrowDown) : ArrowUpDown;

    return (
      <th
        className={`px-4 py-3 text-left text-xs font-medium ${COMPONENTS.text.secondary} uppercase cursor-pointer hover:bg-slate-800/30 transition-colors ${className}`}
        onClick={() => handleSort(column)}
      >
        <div className="flex items-center gap-2">
          <span>{label}</span>
          <Icon
            className={`w-3 h-3 ${
              isActive ? 'text-emerald-400' : 'text-slate-600'
            } transition-colors`}
          />
        </div>
      </th>
    );
  };

  // Load features when expanded and completed
  useEffect(() => {
    if (isExpanded && isCompleted && !features.length) {
      fetchExtractionFeatures(extraction.id, filters);
    }
  }, [isExpanded, isCompleted, extraction.id]);

  // Load training details if not already loaded
  useEffect(() => {
    if (extraction.training_id && !training) {
      fetchTraining(extraction.training_id).catch((error) => {
        console.error('Failed to fetch training details:', error);
      });
    }
  }, [extraction.training_id, training, fetchTraining]);

  // Update metrics history when extraction receives WebSocket updates
  useEffect(() => {
    // Only update if we have detailed metrics from WebSocket
    if (
      isActive &&
      extraction.current_batch !== undefined &&
      extraction.current_batch !== lastBatchRef.current
    ) {
      const newEntry: MetricsEntry = {
        timestamp: new Date().toISOString(),
        batch: extraction.current_batch || 0,
        totalBatches: extraction.total_batches || 0,
        samplesProcessed: extraction.samples_processed || 0,
        totalSamples: extraction.total_samples || 0,
        samplesPerSecond: extraction.samples_per_second || 0,
        etaSeconds: extraction.eta_seconds || 0,
        featuresInHeap: extraction.features_in_heap || 0,
        heapExamplesCount: extraction.heap_examples_count || 0,
        progress: (extraction.progress || 0) * 100,
      };

      lastBatchRef.current = extraction.current_batch || 0;

      setMetricsHistory((prev) => {
        const updated = [...prev, newEntry];
        // Keep only last 20 entries
        return updated.slice(-20);
      });
    }
  }, [
    isActive,
    extraction.current_batch,
    extraction.samples_processed,
    extraction.samples_per_second,
  ]);

  /**
   * Format ETA seconds to human-readable string.
   */
  const formatEta = (seconds: number): string => {
    if (seconds <= 0) return '—';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  /**
   * Handle copy extraction logs to clipboard.
   */
  const handleCopyLogs = async () => {
    if (metricsHistory.length === 0) return;

    const logLines = metricsHistory.map((entry) => {
      const time = new Date(entry.timestamp).toLocaleTimeString();
      return `[${time}] batch=${entry.batch}/${entry.totalBatches}, samples=${entry.samplesProcessed.toLocaleString()}/${entry.totalSamples.toLocaleString()}, speed=${entry.samplesPerSecond.toFixed(1)}/sec, ETA=${formatEta(entry.etaSeconds)}, examples=${entry.heapExamplesCount.toLocaleString()}, progress=${entry.progress.toFixed(1)}%`;
    });

    const logText = logLines.join('\n');

    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(logText);
      } else {
        // Fallback for HTTP
        const textarea = document.createElement('textarea');
        textarea.value = logText;
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      }
      setLogsCopied(true);
      setTimeout(() => setLogsCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy logs:', error);
    }
  };

  /**
   * Handle triggering NLP analysis for this extraction.
   */
  const handleProcessNlp = async () => {
    if (isNlpProcessing || extraction.nlp_status === 'processing') return;

    setIsNlpProcessing(true);
    setNlpError(null);

    try {
      await triggerNlpAnalysis(extraction.id);
      // Optimistically update the store to trigger WebSocket subscription
      // The WebSocket will then provide real-time progress updates
      updateExtractionById(extraction.id, { nlp_status: 'processing', nlp_progress: 0 });
    } catch (error: any) {
      console.error('Failed to trigger NLP analysis:', error);
      setNlpError(error.message || 'Failed to start NLP analysis');
    } finally {
      setIsNlpProcessing(false);
    }
  };

  /**
   * Handle cancelling NLP analysis for this extraction.
   * Preserves progress already made - features that have been analyzed keep their results.
   */
  const handleCancelNlp = async () => {
    if (isNlpCancelling || extraction.nlp_status !== 'processing') return;

    setIsNlpCancelling(true);
    setNlpError(null);

    try {
      await cancelNlpAnalysis(extraction.id);
      // Optimistically update the store to reflect cancelled status
      // Backend immediately sets status to cancelled, no need to wait for Celery task
      updateExtractionById(extraction.id, { nlp_status: 'cancelled' });
    } catch (error: any) {
      console.error('Failed to cancel NLP analysis:', error);
      setNlpError(error.message || 'Failed to cancel NLP analysis');
    } finally {
      setIsNlpCancelling(false);
    }
  };

  /**
   * Handle resuming NLP analysis for this extraction.
   * Skips features that already have NLP analysis.
   */
  const handleResumeNlp = async () => {
    if (isNlpProcessing) return;

    setIsNlpProcessing(true);
    setNlpError(null);

    try {
      // First reset status to allow restarting
      await resetNlpAnalysis(extraction.id, { clear_feature_analysis: false });
      // Then trigger analysis - it will skip already processed features
      await triggerNlpAnalysis(extraction.id, { force_reprocess: false });
      // Optimistically update the store to trigger WebSocket subscription
      updateExtractionById(extraction.id, { nlp_status: 'processing' });
    } catch (error: any) {
      console.error('Failed to resume NLP analysis:', error);
      setNlpError(error.message || 'Failed to resume NLP analysis');
    } finally {
      setIsNlpProcessing(false);
    }
  };

  /**
   * Handle restarting NLP analysis from scratch.
   * Clears all existing NLP analysis from features and starts fresh.
   */
  const handleRestartNlp = async () => {
    if (isNlpResetting) return;

    setIsNlpResetting(true);
    setNlpError(null);

    try {
      // Reset with clear_feature_analysis=true to clear all feature NLP data
      await resetNlpAnalysis(extraction.id, { clear_feature_analysis: true });
      // Then trigger fresh analysis
      await triggerNlpAnalysis(extraction.id, { force_reprocess: true });
      // Optimistically update the store to trigger WebSocket subscription
      updateExtractionById(extraction.id, { nlp_status: 'processing', nlp_progress: 0 });
    } catch (error: any) {
      console.error('Failed to restart NLP analysis:', error);
      setNlpError(error.message || 'Failed to restart NLP analysis');
    } finally {
      setIsNlpResetting(false);
    }
  };

  /**
   * Get NLP status badge based on current status.
   */
  const getNlpStatusBadge = () => {
    const status = extraction.nlp_status;
    const baseClasses = 'px-2 py-0.5 rounded-full text-xs font-medium';

    switch (status) {
      case 'completed':
        return (
          <span className={`${baseClasses} bg-cyan-900/30 text-cyan-400 flex items-center gap-1`} title="NLP analysis completed">
            <Brain className="w-3 h-3" />
            NLP
          </span>
        );
      case 'processing':
        return (
          <span className={`${baseClasses} bg-cyan-900/30 text-cyan-400 flex items-center gap-1`} title="NLP analysis in progress">
            <Brain className="w-3 h-3" />
            <Loader className="w-3 h-3 animate-spin" />
            {extraction.nlp_progress !== null && extraction.nlp_progress !== undefined
              ? `${(extraction.nlp_progress * 100).toFixed(0)}%`
              : 'Processing'}
          </span>
        );
      case 'cancelled':
        return (
          <span className={`${baseClasses} bg-yellow-900/30 text-yellow-400 flex items-center gap-1`} title={`NLP analysis paused at ${extraction.nlp_progress !== null && extraction.nlp_progress !== undefined ? `${(extraction.nlp_progress * 100).toFixed(0)}%` : '?'}. Progress preserved.`}>
            <Brain className="w-3 h-3" />
            <StopCircle className="w-3 h-3" />
            {extraction.nlp_progress !== null && extraction.nlp_progress !== undefined
              ? `${(extraction.nlp_progress * 100).toFixed(0)}%`
              : 'Paused'}
          </span>
        );
      case 'failed':
        return (
          <span className={`${baseClasses} bg-red-900/30 text-red-400 flex items-center gap-1`} title={extraction.nlp_error_message || 'NLP analysis failed'}>
            <Brain className="w-3 h-3" />
            <XCircle className="w-3 h-3" />
          </span>
        );
      case 'pending':
        return (
          <span className={`${baseClasses} bg-slate-800/50 text-slate-400 flex items-center gap-1`} title="NLP analysis pending">
            <Brain className="w-3 h-3" />
            Pending
          </span>
        );
      default:
        return null; // No badge if no NLP status
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
      setSearchFilters(extraction.id, newFilters);
      fetchExtractionFeatures(extraction.id, newFilters);
    }, 300);

    setSearchDebounceTimer(timer);
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
   * Handle range filter input change with debouncing.
   */
  const handleRangeFilterChange = (field: keyof typeof rangeFilterInputs, value: string) => {
    // Update local input state immediately
    setRangeFilterInputs(prev => ({ ...prev, [field]: value }));

    // Clear existing timer
    if (rangeDebounceTimer) {
      clearTimeout(rangeDebounceTimer);
    }

    // Set new timer for 500ms debounce
    const timer = window.setTimeout(() => {
      const numValue = value === '' ? null : parseFloat(value);
      const newFilters: FeatureSearchRequest = {
        ...filters,
        [field]: numValue,
        offset: 0, // Reset pagination
      };
      setSearchFilters(extraction.id, newFilters);
      fetchExtractionFeatures(extraction.id, newFilters);
    }, 500);

    setRangeDebounceTimer(timer);
  };

  /**
   * Clear all range filters.
   */
  const handleClearRangeFilters = () => {
    setRangeFilterInputs({
      min_activation_freq: '',
      max_activation_freq: '',
      min_max_activation: '',
      max_max_activation: '',
    });

    const newFilters: FeatureSearchRequest = {
      ...filters,
      min_activation_freq: null,
      max_activation_freq: null,
      min_max_activation: null,
      max_max_activation: null,
      offset: 0,
    };
    setSearchFilters(extraction.id, newFilters);
    fetchExtractionFeatures(extraction.id, newFilters);
  };

  /**
   * Check if any range filters are active.
   */
  const hasActiveRangeFilters = filters.min_activation_freq != null ||
    filters.max_activation_freq != null ||
    filters.min_max_activation != null ||
    filters.max_max_activation != null;

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

  /**
   * Handle refresh - reload current page of features.
   */
  const handleRefresh = () => {
    fetchExtractionFeatures(extraction.id, filters);
  };

  /**
   * Handle "Go to" page/feature navigation.
   */
  const [goToFeatureInput, setGoToFeatureInput] = useState('');
  const [goToPageInput, setGoToPageInput] = useState('');

  const handleGoToFeature = () => {
    if (!metadata || !goToFeatureInput.trim()) return;

    const featureNum = parseInt(goToFeatureInput.trim(), 10);
    if (isNaN(featureNum) || featureNum < 1) {
      setGoToFeatureInput('');
      return;
    }

    // Navigate to specific feature number (1-indexed)
    let newOffset: number;
    if (featureNum > metadata.total) {
      // Invalid feature number, go to last page
      newOffset = Math.max(0, metadata.total - filters.limit!);
    } else {
      newOffset = (featureNum - 1);
    }

    const newFilters: FeatureSearchRequest = {
      ...filters,
      offset: Math.max(0, Math.min(newOffset, metadata.total - 1)),
    };
    setSearchFilters(extraction.id, newFilters);
    fetchExtractionFeatures(extraction.id, newFilters);
    setGoToFeatureInput('');
  };

  const handleGoToPage = () => {
    if (!metadata || !goToPageInput.trim()) return;

    const pageNum = parseInt(goToPageInput.trim(), 10);
    const totalPages = Math.ceil(metadata.total / filters.limit!);

    if (isNaN(pageNum) || pageNum < 1 || pageNum > totalPages) {
      setGoToPageInput('');
      return;
    }

    // Navigate to specific page number (1-indexed)
    const newOffset = (pageNum - 1) * filters.limit!;

    const newFilters: FeatureSearchRequest = {
      ...filters,
      offset: newOffset,
    };
    setSearchFilters(extraction.id, newFilters);
    fetchExtractionFeatures(extraction.id, newFilters);
    setGoToPageInput('');
  };

  const handleGoToFeatureKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleGoToFeature();
    }
  };

  const handleGoToPageKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleGoToPage();
    }
  };

  // Calculate elapsed time
  const getElapsedTime = () => {
    const startTime = new Date(extraction.created_at);
    const endTime = extraction.completed_at ? new Date(extraction.completed_at) : new Date();

    const duration = intervalToDuration({ start: startTime, end: endTime });

    const parts = [];
    if (duration.hours) parts.push(`${duration.hours} ${duration.hours === 1 ? 'hour' : 'hours'}`);
    if (duration.minutes) parts.push(`${duration.minutes} ${duration.minutes === 1 ? 'minute' : 'minutes'}`);
    if (duration.seconds !== undefined) parts.push(`${duration.seconds} ${duration.seconds === 1 ? 'second' : 'seconds'}`);

    return parts.length > 0 ? parts.join(', ') : 'Less than a second';
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
      case 'finalizing':
        return (
          <span className={`${baseClasses} bg-amber-900/30 text-amber-400 flex items-center gap-1`}>
            <Loader className="w-4 h-4 animate-spin" />
            Saving
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
      case 'deleting':
        return (
          <span className={`${baseClasses} bg-orange-900/30 text-orange-400 flex items-center gap-1`}>
            <Loader className="w-4 h-4 animate-spin" />
            Deleting
          </span>
        );
      default:
        return (
          <span className={`${baseClasses} bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300`}>
            Unknown
          </span>
        );
    }
  };

  const progress = (extraction.progress || 0) * 100;
  const deletionProgress = (extraction.deletion_progress || 0) * 100;

  return (
    <div className={`${COMPONENTS.card.base} p-6 ${COMPONENTS.border.hover} transition-colors`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-start gap-3 flex-1">
          <Zap className="w-6 h-6 text-emerald-400 flex-shrink-0 mt-1" />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-semibold text-lg text-slate-900 dark:text-slate-100 truncate">
                {extraction.source_type === 'external_sae' && extraction.sae_name
                  ? `${extraction.sae_name} (${getModelDisplayName()})`
                  : `${getModelDisplayName()} - ${extraction.dataset_name || 'Unknown Dataset'}`}
              </h3>
              {extraction.source_type === 'external_sae' && (
                <span className="px-2 py-0.5 text-xs font-medium bg-purple-900/30 text-purple-400 rounded">SAE</span>
              )}
              {extraction.batch_id && (
                <span
                  className="px-2 py-0.5 text-xs font-medium bg-blue-900/30 text-blue-400 rounded"
                  title={`Batch: ${extraction.batch_id}`}
                >
                  {extraction.batch_position}/{extraction.batch_total}
                </span>
              )}
              {getStatusBadge()}
              {isCompleted && getNlpStatusBadge()}
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
            <div className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
              <p>Started: {format(new Date(extraction.created_at), 'MMM d, yyyy • h:mm:ss a')}</p>
              {extraction.completed_at && (
                <>
                  <p>Completed: {format(new Date(extraction.completed_at), 'MMM d, yyyy • h:mm:ss a')}</p>
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
          {/* NLP Controls - show different buttons based on NLP status */}
          {isCompleted && (
            <>
              {/* Start button - only show when NLP hasn't started or is null */}
              {(extraction.nlp_status === null || extraction.nlp_status === undefined || extraction.nlp_status === 'pending') && (
                <button
                  type="button"
                  onClick={handleProcessNlp}
                  disabled={isNlpProcessing}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    isNlpProcessing
                      ? 'bg-cyan-900/30 text-cyan-400 cursor-not-allowed'
                      : 'bg-cyan-600 hover:bg-cyan-500 text-white'
                  }`}
                  title="Run NLP analysis on feature activation examples"
                >
                  {isNlpProcessing ? (
                    <>
                      <Loader className="w-4 h-4 animate-spin" />
                      Starting...
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4" />
                      Process NLP
                    </>
                  )}
                </button>
              )}

              {/* Processing status with Cancel button */}
              {extraction.nlp_status === 'processing' && (
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium bg-cyan-900/30 text-cyan-400">
                    <Brain className="w-4 h-4" />
                    <Loader className="w-4 h-4 animate-spin" />
                    NLP {extraction.nlp_progress !== null && extraction.nlp_progress !== undefined
                      ? `${(extraction.nlp_progress * 100).toFixed(0)}%`
                      : '...'}
                  </div>
                  <button
                    type="button"
                    onClick={handleCancelNlp}
                    disabled={isNlpCancelling}
                    className={`flex items-center gap-1 px-2 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                      isNlpCancelling
                        ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                        : 'bg-yellow-600/20 border border-yellow-600 text-yellow-400 hover:bg-yellow-600/30'
                    }`}
                    title="Pause NLP analysis (progress will be preserved)"
                  >
                    {isNlpCancelling ? (
                      <Loader className="w-4 h-4 animate-spin" />
                    ) : (
                      <StopCircle className="w-4 h-4" />
                    )}
                  </button>
                </div>
              )}

              {/* Cancelled status - Resume and Restart buttons */}
              {extraction.nlp_status === 'cancelled' && (
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleResumeNlp}
                    disabled={isNlpProcessing}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                      isNlpProcessing
                        ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                        : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                    }`}
                    title="Resume NLP analysis from where it left off"
                  >
                    {isNlpProcessing ? (
                      <>
                        <Loader className="w-4 h-4 animate-spin" />
                        Resuming...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Resume
                      </>
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (window.confirm('This will clear all existing NLP analysis and start from scratch. Continue?')) {
                        handleRestartNlp();
                      }
                    }}
                    disabled={isNlpResetting}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                      isNlpResetting
                        ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                        : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
                    }`}
                    title="Restart NLP analysis from scratch (clears existing progress)"
                  >
                    {isNlpResetting ? (
                      <>
                        <Loader className="w-4 h-4 animate-spin" />
                        Restarting...
                      </>
                    ) : (
                      <>
                        <RotateCcw className="w-4 h-4" />
                        Restart
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Completed - option to reprocess */}
              {extraction.nlp_status === 'completed' && (
                <button
                  type="button"
                  onClick={() => {
                    if (window.confirm('This will clear all existing NLP analysis and reprocess from scratch. Continue?')) {
                      handleRestartNlp();
                    }
                  }}
                  disabled={isNlpResetting}
                  className={`flex items-center gap-1.5 px-2 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    isNlpResetting
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                      : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700 hover:text-slate-300'
                  }`}
                  title="Reprocess NLP analysis from scratch"
                >
                  {isNlpResetting ? (
                    <Loader className="w-3 h-3 animate-spin" />
                  ) : (
                    <RotateCcw className="w-3 h-3" />
                  )}
                </button>
              )}
            </>
          )}
          {isCompleted && (
            <StartLabelingButton
              extractionId={extraction.id}
              onSuccess={() => {
                // Optionally refresh features list if expanded
                if (isExpanded) {
                  fetchExtractionFeatures(extraction.id, filters);
                }
              }}
            />
          )}
          {(isCompleted || isFailed) && !isDeleting && onDelete && (
            <button
              type="button"
              onClick={() => {
                console.log(`[ExtractionJobCard] Delete button clicked for extraction ${extraction.id}`);
                console.log(`[ExtractionJobCard] Extraction status: ${extraction.status}, NLP status: ${extraction.nlp_status}`);
                if (window.confirm('Are you sure you want to delete this extraction?')) {
                  console.log(`[ExtractionJobCard] User confirmed deletion`);
                  onDelete();
                } else {
                  console.log(`[ExtractionJobCard] User cancelled deletion`);
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
        <div className="mb-4 space-y-3">
          {/* Progress info row */}
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-slate-600 dark:text-slate-400">
              {extraction.status_message
                ? extraction.status_message
                : extraction.samples_processed != null && extraction.total_samples != null
                  ? `${extraction.samples_processed.toLocaleString()} / ${extraction.total_samples.toLocaleString()} samples`
                  : extraction.features_in_heap != null
                    ? `${extraction.features_in_heap.toLocaleString()} active features found`
                    : 'Processing...'}
            </span>
            <span className="text-emerald-400 font-medium">
              {progress.toFixed(1)}%
            </span>
          </div>
          {/* Elapsed time and ETA row */}
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-1.5 text-blue-400">
              <Clock className="w-4 h-4" />
              <span>Elapsed: {getElapsedTime()}</span>
            </div>
            {extraction.eta_seconds !== undefined && extraction.eta_seconds > 0 && (
              <div className="text-purple-400">
                ETA: {formatEta(extraction.eta_seconds)}
              </div>
            )}
          </div>
          <div className={COMPONENTS.progress.container}>
            <div
              className={`h-full transition-all duration-300 ${
                extraction.status_message
                  ? 'bg-gradient-to-r from-amber-500 to-amber-400 animate-pulse'
                  : 'bg-gradient-to-r from-emerald-500 to-emerald-400'
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* Live Metrics Toggle Button */}
          <button
            type="button"
            onClick={() => setShowMetrics(!showMetrics)}
            className={`flex items-center justify-center gap-2 w-full rounded-lg ${COMPONENTS.button.secondary}`}
          >
            <Activity className="w-4 h-4" />
            <span>{showMetrics ? 'Hide' : 'Show'} Live Metrics</span>
          </button>

          {/* Live Metrics Section */}
          {showMetrics && (
            <div className="border-t border-slate-700 pt-3 mt-3">
              {/* Metrics Summary Grid */}
              <div className="grid grid-cols-4 gap-2 mb-3">
                {/* Batch Progress */}
                <div className="bg-slate-800/50 rounded-lg p-2">
                  <div className="text-xs text-slate-400 mb-1">Batch</div>
                  <div className="text-lg font-semibold text-slate-200">
                    {extraction.current_batch !== undefined
                      ? `${extraction.current_batch}/${extraction.total_batches || '?'}`
                      : '—'}
                  </div>
                </div>

                {/* Samples Processed */}
                <div className="bg-slate-800/50 rounded-lg p-2">
                  <div className="text-xs text-slate-400 mb-1">Samples</div>
                  <div className="text-lg font-semibold text-emerald-400">
                    {extraction.samples_processed !== undefined
                      ? extraction.samples_processed.toLocaleString()
                      : '—'}
                  </div>
                  {extraction.total_samples && (
                    <div className="text-xs text-slate-500">
                      of {extraction.total_samples.toLocaleString()}
                    </div>
                  )}
                </div>

                {/* Speed */}
                <div className="bg-slate-800/50 rounded-lg p-2">
                  <div className="text-xs text-slate-400 mb-1">Speed</div>
                  <div className="text-lg font-semibold text-blue-400">
                    {extraction.samples_per_second !== undefined
                      ? `${extraction.samples_per_second.toFixed(1)}/s`
                      : '—'}
                  </div>
                </div>

                {/* ETA */}
                <div className="bg-slate-800/50 rounded-lg p-2">
                  <div className="text-xs text-slate-400 mb-1">ETA</div>
                  <div className="text-lg font-semibold text-purple-400">
                    {extraction.eta_seconds !== undefined
                      ? formatEta(extraction.eta_seconds)
                      : '—'}
                  </div>
                </div>
              </div>

              {/* Two Column Layout: Logs left, Metrics right */}
              <div className="flex gap-3">
                {/* Extraction Logs - Left Column (60%) */}
                <div className="w-3/5 bg-slate-950 rounded-lg p-3 font-mono text-xs">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-slate-400">Extraction Logs</span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleCopyLogs}
                        disabled={metricsHistory.length === 0}
                        className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        title={logsCopied ? 'Copied!' : 'Copy logs to clipboard'}
                      >
                        {logsCopied ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                      <span className="text-emerald-400 text-xs">Live</span>
                    </div>
                  </div>
                  <div className="h-[268px] overflow-y-auto space-y-1">
                    {metricsHistory.length > 0 ? (
                      [...metricsHistory].reverse().map((entry, i) => {
                        const time = new Date(entry.timestamp).toLocaleTimeString();
                        return (
                          <div key={`${entry.batch}-${i}`} className="text-slate-300">
                            <span className="text-slate-500">[{time}]</span>{' '}
                            batch={entry.batch ?? 0}/{entry.totalBatches ?? '?'},
                            samples={(entry.samplesProcessed ?? 0).toLocaleString()},
                            speed={(entry.samplesPerSecond ?? 0).toFixed(1)}/s,
                            ETA={formatEta(entry.etaSeconds ?? 0)},
                            examples={(entry.heapExamplesCount ?? 0).toLocaleString()}
                          </div>
                        );
                      })
                    ) : (
                      <div className="text-slate-500 text-center py-4">
                        Waiting for extraction metrics...
                      </div>
                    )}
                  </div>
                </div>

                {/* Metrics Charts - Right Column (40%) */}
                <div className="w-2/5 flex flex-col gap-2">
                  {/* Speed Chart */}
                  <div className="bg-slate-800/30 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-slate-400">Speed (samples/s)</span>
                      <span className="text-xs text-blue-400 font-mono">
                        {metricsHistory.length > 0
                          ? `${metricsHistory[metricsHistory.length - 1].samplesPerSecond.toFixed(1)}/s`
                          : '—'}
                      </span>
                    </div>
                    <div className="h-16">
                      {metricsHistory.length > 1 ? (
                        <svg viewBox="0 0 100 40" className="w-full h-full" preserveAspectRatio="none">
                          <line x1="0" y1="10" x2="100" y2="10" stroke="#334155" strokeWidth="0.5" />
                          <line x1="0" y1="20" x2="100" y2="20" stroke="#334155" strokeWidth="0.5" />
                          <line x1="0" y1="30" x2="100" y2="30" stroke="#334155" strokeWidth="0.5" />
                          {(() => {
                            const data = metricsHistory.map(e => e.samplesPerSecond);
                            const maxVal = Math.max(...data, 1);
                            const minVal = Math.min(...data, 0);
                            const range = maxVal - minVal || 1;
                            const points = data.map((val, i) => {
                              const x = (i / (data.length - 1)) * 100;
                              const y = 40 - ((val - minVal) / range) * 36 - 2;
                              return `${x},${y}`;
                            }).join(' ');
                            return (
                              <>
                                <polyline fill="none" stroke="#3b82f6" strokeWidth="1.5" points={points} />
                                <circle
                                  cx={(data.length - 1) / (data.length - 1) * 100}
                                  cy={40 - ((data[data.length - 1] - minVal) / range) * 36 - 2}
                                  r="2"
                                  fill="#3b82f6"
                                />
                              </>
                            );
                          })()}
                        </svg>
                      ) : (
                        <div className="h-full flex items-center justify-center text-slate-500 text-xs">
                          Waiting...
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Examples Collected Chart */}
                  <div className="bg-slate-800/30 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-slate-400">Examples Collected</span>
                      <span className="text-xs text-emerald-400 font-mono">
                        {metricsHistory.length > 0
                          ? metricsHistory[metricsHistory.length - 1].heapExamplesCount.toLocaleString()
                          : '—'}
                      </span>
                    </div>
                    <div className="h-16">
                      {metricsHistory.length > 1 ? (
                        <svg viewBox="0 0 100 40" className="w-full h-full" preserveAspectRatio="none">
                          <line x1="0" y1="10" x2="100" y2="10" stroke="#334155" strokeWidth="0.5" />
                          <line x1="0" y1="20" x2="100" y2="20" stroke="#334155" strokeWidth="0.5" />
                          <line x1="0" y1="30" x2="100" y2="30" stroke="#334155" strokeWidth="0.5" />
                          {(() => {
                            const data = metricsHistory.map(e => e.heapExamplesCount);
                            const maxVal = Math.max(...data, 1);
                            const minVal = Math.min(...data, 0);
                            const range = maxVal - minVal || 1;
                            const points = data.map((val, i) => {
                              const x = (i / (data.length - 1)) * 100;
                              const y = 40 - ((val - minVal) / range) * 36 - 2;
                              return `${x},${y}`;
                            }).join(' ');
                            return (
                              <>
                                <polyline fill="none" stroke="#10b981" strokeWidth="1.5" points={points} />
                                <circle
                                  cx={(data.length - 1) / (data.length - 1) * 100}
                                  cy={40 - ((data[data.length - 1] - minVal) / range) * 36 - 2}
                                  r="2"
                                  fill="#10b981"
                                />
                              </>
                            );
                          })()}
                        </svg>
                      ) : (
                        <div className="h-full flex items-center justify-center text-slate-500 text-xs">
                          Waiting...
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Collection Rate Chart (examples/sec) */}
                  <div className="bg-slate-800/30 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-slate-400">Collection Rate</span>
                      <span className="text-xs text-purple-400 font-mono">
                        {(() => {
                          if (metricsHistory.length < 2) return '—';
                          const last = metricsHistory[metricsHistory.length - 1];
                          const prev = metricsHistory[metricsHistory.length - 2];
                          const timeDelta = (new Date(last.timestamp).getTime() - new Date(prev.timestamp).getTime()) / 1000;
                          const examplesDelta = last.heapExamplesCount - prev.heapExamplesCount;
                          const rate = timeDelta > 0 ? examplesDelta / timeDelta : 0;
                          return `${rate.toFixed(0)}/s`;
                        })()}
                      </span>
                    </div>
                    <div className="h-16">
                      {metricsHistory.length > 2 ? (
                        <svg viewBox="0 0 100 40" className="w-full h-full" preserveAspectRatio="none">
                          <line x1="0" y1="10" x2="100" y2="10" stroke="#334155" strokeWidth="0.5" />
                          <line x1="0" y1="20" x2="100" y2="20" stroke="#334155" strokeWidth="0.5" />
                          <line x1="0" y1="30" x2="100" y2="30" stroke="#334155" strokeWidth="0.5" />
                          {(() => {
                            // Calculate collection rates between consecutive entries
                            const rates: number[] = [];
                            for (let i = 1; i < metricsHistory.length; i++) {
                              const curr = metricsHistory[i];
                              const prev = metricsHistory[i - 1];
                              const timeDelta = (new Date(curr.timestamp).getTime() - new Date(prev.timestamp).getTime()) / 1000;
                              const examplesDelta = curr.heapExamplesCount - prev.heapExamplesCount;
                              rates.push(timeDelta > 0 ? examplesDelta / timeDelta : 0);
                            }
                            if (rates.length === 0) return null;
                            const maxVal = Math.max(...rates, 1);
                            const minVal = Math.min(...rates, 0);
                            const range = maxVal - minVal || 1;
                            const points = rates.map((val, i) => {
                              const x = (i / (rates.length - 1 || 1)) * 100;
                              const y = 40 - ((val - minVal) / range) * 36 - 2;
                              return `${x},${y}`;
                            }).join(' ');
                            return (
                              <>
                                <polyline fill="none" stroke="#a855f7" strokeWidth="1.5" points={points} />
                                <circle
                                  cx={100}
                                  cy={40 - ((rates[rates.length - 1] - minVal) / range) * 36 - 2}
                                  r="2"
                                  fill="#a855f7"
                                />
                              </>
                            );
                          })()}
                        </svg>
                      ) : (
                        <div className="h-full flex items-center justify-center text-slate-500 text-xs">
                          Waiting...
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Deletion Progress Bar */}
      {isDeleting && (
        <div className="mb-4 space-y-2">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-orange-400">
              {extraction.status_message ||
                (extraction.deletion_features_deleted != null && extraction.deletion_total_features != null
                  ? `Deleting ${extraction.deletion_features_deleted.toLocaleString()} / ${extraction.deletion_total_features.toLocaleString()} features...`
                  : 'Deleting extraction...')}
            </span>
            <span className="text-orange-400 font-medium">
              {deletionProgress.toFixed(1)}%
            </span>
          </div>
          <div className={COMPONENTS.progress.container}>
            <div
              className="h-full transition-all duration-300 bg-gradient-to-r from-orange-500 to-orange-400"
              style={{ width: `${deletionProgress}%` }}
            />
          </div>
          <p className="text-xs text-slate-500">
            This may take a while for large extractions with many features.
          </p>
        </div>
      )}

      {/* Statistics for Completed */}
      {isCompleted && extraction.statistics && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className={COMPONENTS.stat.container}>
            <div className={COMPONENTS.stat.label}>Features Found</div>
            <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400">
              {extraction.statistics.total_features?.toLocaleString() || 'N/A'}
            </div>
          </div>
          <div className={COMPONENTS.stat.container}>
            <div className={COMPONENTS.stat.label}>Interpretable</div>
            <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
              {extraction.statistics.interpretable_count !== undefined && extraction.statistics.total_features
                ? `${((extraction.statistics.interpretable_count / extraction.statistics.total_features) * 100).toFixed(1)}%`
                : 'N/A'}
            </div>
          </div>
          <div className={COMPONENTS.stat.container}>
            <div className={COMPONENTS.stat.label}>Activation Rate</div>
            <div className="text-lg font-bold text-purple-600 dark:text-purple-400">
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

      {/* NLP Analysis Error Message */}
      {(extraction.nlp_status === 'failed' && extraction.nlp_error_message) || nlpError ? (
        <div className="mt-4 p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400 text-sm">
          <div className="flex items-start gap-2">
            <Brain className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <div className="font-medium mb-1">NLP Analysis Failed</div>
              <div className="text-cyan-300">{extraction.nlp_error_message || nlpError}</div>
            </div>
          </div>
          <div className="mt-3 flex gap-2">
            <button
              onClick={() => {
                setNlpError(null);
                handleResumeNlp();
              }}
              disabled={isNlpProcessing}
              className={`flex-1 flex items-center justify-center gap-1.5 text-sm font-medium py-2 rounded-lg transition-colors ${
                isNlpProcessing
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-emerald-600 hover:bg-emerald-500 text-white'
              }`}
            >
              {isNlpProcessing ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  Resuming...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Resume from {extraction.nlp_progress !== null && extraction.nlp_progress !== undefined ? `${(extraction.nlp_progress * 100).toFixed(0)}%` : 'last point'}
                </>
              )}
            </button>
            <button
              onClick={() => {
                if (window.confirm('This will clear all existing NLP analysis and restart from scratch. Continue?')) {
                  setNlpError(null);
                  handleRestartNlp();
                }
              }}
              disabled={isNlpResetting}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                isNlpResetting
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
              }`}
            >
              {isNlpResetting ? (
                <Loader className="w-4 h-4 animate-spin" />
              ) : (
                <RotateCcw className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      ) : null}

      {/* Expandable Features List (only for completed extractions) */}
      {isExpanded && isCompleted && (
        <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-800 space-y-4">
          <h4 className={`text-lg font-semibold ${COMPONENTS.text.heading} flex items-center gap-2`}>
            <Zap className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
            Discovered Features
            {metadata && (
              <span className={`text-sm ${COMPONENTS.text.secondary} font-normal`}>
                ({metadata.total.toLocaleString()} total)
              </span>
            )}
          </h4>

          {/* Search and Filters */}
          <div className="flex gap-3">
            {/* Search Input */}
            <div className="flex-1 relative">
              <Search className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 ${COMPONENTS.text.secondary}`} />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => handleSearchChange(e.target.value)}
                placeholder="Search features..."
                className={`w-full pl-10 pr-4 py-2 bg-slate-100 dark:bg-slate-800 border ${COMPONENTS.border.default} rounded ${COMPONENTS.text.primary} placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-emerald-600 dark:focus:border-emerald-400`}
              />
            </div>

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

            {/* Sort Scope Toggle */}
            <button
              onClick={() => setSortScope(sortScope === 'page' ? 'all' : 'page')}
              className={`border rounded transition-colors ${
                sortScope === 'all'
                  ? 'bg-emerald-600/20 border-emerald-600 text-emerald-400 px-3 py-2'
                  : COMPONENTS.button.secondary
              }`}
              title={sortScope === 'page' ? 'Sorting current page only (click to sort all)' : 'Sorting entire dataset (click to sort page only)'}
            >
              {sortScope === 'page' ? (
                <List className="w-4 h-4" />
              ) : (
                <Layers className="w-4 h-4" />
              )}
            </button>
          </div>

          {/* Range Filters */}
          <div className="flex flex-wrap items-center gap-4 mb-3 p-2 bg-slate-800/50 rounded border border-slate-700/50">
            {/* Activation Frequency Range */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400 whitespace-nowrap">Act. Freq:</span>
              <input
                type="number"
                min={0}
                max={100}
                step={0.1}
                placeholder="Min"
                value={rangeFilterInputs.min_activation_freq}
                onChange={(e) => handleRangeFilterChange('min_activation_freq', e.target.value)}
                className="w-16 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-100 text-xs focus:outline-none focus:border-emerald-500"
              />
              <span className="text-slate-500">-</span>
              <input
                type="number"
                min={0}
                max={100}
                step={0.1}
                placeholder="Max"
                value={rangeFilterInputs.max_activation_freq}
                onChange={(e) => handleRangeFilterChange('max_activation_freq', e.target.value)}
                className="w-16 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-100 text-xs focus:outline-none focus:border-emerald-500"
              />
            </div>

            {/* Max Activation Range */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400 whitespace-nowrap">Max Act:</span>
              <input
                type="number"
                min={0}
                step={0.1}
                placeholder="Min"
                value={rangeFilterInputs.min_max_activation}
                onChange={(e) => handleRangeFilterChange('min_max_activation', e.target.value)}
                className="w-16 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-100 text-xs focus:outline-none focus:border-emerald-500"
              />
              <span className="text-slate-500">-</span>
              <input
                type="number"
                min={0}
                step={0.1}
                placeholder="Max"
                value={rangeFilterInputs.max_max_activation}
                onChange={(e) => handleRangeFilterChange('max_max_activation', e.target.value)}
                className="w-16 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-100 text-xs focus:outline-none focus:border-emerald-500"
              />
            </div>

            {/* Clear Filters Button */}
            {hasActiveRangeFilters && (
              <button
                type="button"
                onClick={handleClearRangeFilters}
                className="px-2 py-1 text-xs text-slate-400 hover:text-red-400 hover:bg-slate-700 rounded transition-colors"
                title="Clear range filters"
              >
                Clear
              </button>
            )}
          </div>

          {/* Top Navigation Controls */}
          {features.length > 0 && metadata && (
            <div className="flex items-center justify-between mb-3">
              <div className={`text-sm ${COMPONENTS.text.secondary}`}>
                Showing {filters.offset! + 1}-{Math.min(filters.offset! + filters.limit!, metadata.total)} of {metadata.total} features
              </div>
              <div className="flex items-center gap-3">
                {/* Go to Feature input */}
                <div className="flex items-center gap-2">
                  <label htmlFor={`goto-feature-top-${extraction.id}`} className={`text-sm ${COMPONENTS.text.secondary}`}>
                    Go to Feature:
                  </label>
                  <input
                    id={`goto-feature-top-${extraction.id}`}
                    type="number"
                    min="1"
                    value={goToFeatureInput}
                    onChange={(e) => {
                      setGoToFeatureInput(e.target.value);
                      setGoToPageInput(''); // Clear the other input
                    }}
                    onKeyPress={handleGoToFeatureKeyPress}
                    placeholder="#"
                    className="w-20 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                  />
                  <button
                    onClick={handleGoToFeature}
                    disabled={!goToFeatureInput.trim()}
                    className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 text-sm text-white rounded transition-colors"
                  >
                    Go
                  </button>
                </div>
                {/* Go to Page input */}
                <div className="flex items-center gap-2">
                  <label htmlFor={`goto-page-top-${extraction.id}`} className={`text-sm ${COMPONENTS.text.secondary}`}>
                    Page:
                  </label>
                  <input
                    id={`goto-page-top-${extraction.id}`}
                    type="number"
                    min="1"
                    max={Math.ceil(metadata.total / filters.limit!)}
                    value={goToPageInput}
                    onChange={(e) => {
                      setGoToPageInput(e.target.value);
                      setGoToFeatureInput(''); // Clear the other input
                    }}
                    onKeyPress={handleGoToPageKeyPress}
                    placeholder={`of ${Math.ceil(metadata.total / filters.limit!)}`}
                    className="w-24 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                  />
                  <button
                    onClick={handleGoToPage}
                    disabled={!goToPageInput.trim()}
                    className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 text-sm text-white rounded transition-colors"
                  >
                    Go
                  </button>
                </div>
                {/* Refresh button */}
                <button
                  onClick={handleRefresh}
                  className={`px-3 py-1 text-sm ${COMPONENTS.button.secondary} flex items-center gap-1.5`}
                  title="Refresh features"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Refresh
                </button>
                {/* Previous/Next buttons */}
                <div className="flex gap-2 border-l border-slate-700 pl-3">
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
            </div>
          )}

          {/* Feature Table */}
          <div className={`${COMPONENTS.surface.card} rounded-lg overflow-hidden`}>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className={COMPONENTS.surface.card}>
                  <tr>
                    <SortableColumnHeader column="id" label="ID" />
                    <SortableColumnHeader column="label" label="Label" />
                    <SortableColumnHeader column="category" label="Category" />
                    <SortableColumnHeader column="description" label="Description" />
                    <th className={`px-4 py-3 text-left text-xs font-medium ${COMPONENTS.text.secondary} uppercase`}>Example Context</th>
                    <SortableColumnHeader column="activation_frequency" label="Activation Freq" />
                    <SortableColumnHeader column="max_activation" label="Max Activation" />
                    <SortableColumnHeader column="is_favorite" label="Tag" />
                  </tr>
                </thead>
                <tbody>
                  {isLoadingFeatures ? (
                    <tr>
                      <td colSpan={8} className={`px-4 py-8 text-center ${COMPONENTS.text.secondary}`}>
                        Loading features...
                      </td>
                    </tr>
                  ) : features.length === 0 ? (
                    <tr>
                      <td colSpan={8} className={`px-4 py-8 text-center ${COMPONENTS.text.secondary}`}>
                        No features match your search
                      </td>
                    </tr>
                  ) : (
                    sortedFeatures.map((feature) => (
                      <tr
                        key={feature.id}
                        onClick={() => setSelectedFeatureId(feature.id)}
                        className={`border-t ${COMPONENTS.border.default} ${COMPONENTS.surface.hover} cursor-pointer`}
                      >
                        <td className="px-4 py-3">
                          <span className={`font-mono text-sm ${COMPONENTS.text.secondary}`}>
                            #{feature.neuron_index}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`text-sm ${COMPONENTS.text.primary}`}>{feature.name}</span>
                        </td>
                        <td className="px-4 py-3">
                          {feature.category ? (
                            <span className="text-xs text-slate-300 bg-slate-700/50 px-2 py-1 rounded">
                              {feature.category}
                            </span>
                          ) : (
                            <span className="text-xs text-slate-500 italic">—</span>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          {feature.description ? (
                            <span className="text-xs text-slate-400 line-clamp-2">
                              {feature.description}
                            </span>
                          ) : (
                            <span className="text-xs text-slate-500 italic">—</span>
                          )}
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
                          {(() => {
                            const freqColor = getActivationFrequencyColor(feature.activation_frequency);
                            return (
                              <span
                                className={`text-sm ${freqColor.textClass} px-2 py-0.5 rounded ${freqColor.bgClass}`}
                                title={freqColor.label}
                              >
                                {(feature.activation_frequency * 100).toFixed(2)}%
                              </span>
                            );
                          })()}
                        </td>
                        <td className="px-4 py-3">
                          {(() => {
                            const maxActColor = getMaxActivationColor(feature.max_activation);
                            return (
                              <span
                                className={`text-sm ${maxActColor.textClass} px-2 py-0.5 rounded ${maxActColor.bgClass}`}
                                title={maxActColor.label}
                              >
                                {formatActivation(feature.max_activation)}
                              </span>
                            );
                          })()}
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

          {/* Bottom Navigation Controls */}
          {features.length > 0 && metadata && (
            <div className={`flex items-center justify-between border-t ${COMPONENTS.border.default} pt-4`}>
              <div className={`text-sm ${COMPONENTS.text.secondary}`}>
                Showing {filters.offset! + 1}-{Math.min(filters.offset! + filters.limit!, metadata.total)} of {metadata.total} features
              </div>
              <div className="flex items-center gap-3">
                {/* Go to Feature input */}
                <div className="flex items-center gap-2">
                  <label htmlFor={`goto-feature-bottom-${extraction.id}`} className={`text-sm ${COMPONENTS.text.secondary}`}>
                    Go to Feature:
                  </label>
                  <input
                    id={`goto-feature-bottom-${extraction.id}`}
                    type="number"
                    min="1"
                    value={goToFeatureInput}
                    onChange={(e) => {
                      setGoToFeatureInput(e.target.value);
                      setGoToPageInput(''); // Clear the other input
                    }}
                    onKeyPress={handleGoToFeatureKeyPress}
                    placeholder="#"
                    className="w-20 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                  />
                  <button
                    onClick={handleGoToFeature}
                    disabled={!goToFeatureInput.trim()}
                    className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 text-sm text-white rounded transition-colors"
                  >
                    Go
                  </button>
                </div>
                {/* Go to Page input */}
                <div className="flex items-center gap-2">
                  <label htmlFor={`goto-page-bottom-${extraction.id}`} className={`text-sm ${COMPONENTS.text.secondary}`}>
                    Page:
                  </label>
                  <input
                    id={`goto-page-bottom-${extraction.id}`}
                    type="number"
                    min="1"
                    max={Math.ceil(metadata.total / filters.limit!)}
                    value={goToPageInput}
                    onChange={(e) => {
                      setGoToPageInput(e.target.value);
                      setGoToFeatureInput(''); // Clear the other input
                    }}
                    onKeyPress={handleGoToPageKeyPress}
                    placeholder={`of ${Math.ceil(metadata.total / filters.limit!)}`}
                    className="w-24 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                  />
                  <button
                    onClick={handleGoToPage}
                    disabled={!goToPageInput.trim()}
                    className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 text-sm text-white rounded transition-colors"
                  >
                    Go
                  </button>
                </div>
                {/* Refresh button */}
                <button
                  onClick={handleRefresh}
                  className={`px-3 py-1 text-sm ${COMPONENTS.button.secondary} flex items-center gap-1.5`}
                  title="Refresh features"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Refresh
                </button>
                {/* Previous/Next buttons */}
                <div className="flex gap-2 border-l border-slate-700 pl-3">
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
            </div>
          )}

          {featuresError && (
            <div className="mt-4 p-3 bg-red-500/10 dark:bg-red-500/20 border border-red-500/30 dark:border-red-500/40 rounded text-red-400 dark:text-red-300 text-sm">
              {featuresError}
            </div>
          )}
        </div>
      )}

      {/* Job Details */}
      <div className={`mt-4 pt-4 border-t ${COMPONENTS.border.default}`}>
        <details className="text-sm">
          <summary className={`cursor-pointer ${COMPONENTS.text.secondary} hover:text-slate-300 dark:hover:text-slate-200 transition-colors`}>
            Job Details
          </summary>
          <div className={`mt-2 ${COMPONENTS.text.secondary}`}>
            {/* Combined compact grid layout */}
            <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-sm">
              {/* Training Information */}
              {training && (
                <>
                  <div className="col-span-3 text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mt-1 mb-1">Training Job</div>
                  <div className="col-span-3">Training ID: <span className={`${COMPONENTS.text.primary} font-mono text-xs`}>{training.id}</span></div>
                  <div>Architecture: <span className={`${COMPONENTS.text.primary} capitalize`}>{training.hyperparameters.architecture_type}</span></div>
                  <div>Hidden Dim: <span className={COMPONENTS.text.primary}>{training.hyperparameters.hidden_dim.toLocaleString()}</span></div>
                  <div>Latent Dim: <span className={COMPONENTS.text.primary}>{training.hyperparameters.latent_dim.toLocaleString()}</span></div>
                  <div>Expansion: <span className={COMPONENTS.text.primary}>{(training.hyperparameters.latent_dim / training.hyperparameters.hidden_dim).toFixed(1)}x</span></div>
                  <div>Layer(s): <span className={COMPONENTS.text.primary}>{training.hyperparameters.training_layers.join(', ')}</span></div>
                  <div>L1 Alpha: <span className={COMPONENTS.text.primary}>{training.hyperparameters.l1_alpha}</span></div>
                  <div>Learning Rate: <span className={COMPONENTS.text.primary}>{training.hyperparameters.learning_rate}</span></div>
                  <div>Batch Size: <span className={COMPONENTS.text.primary}>{training.hyperparameters.batch_size}</span></div>
                  <div>Total Steps: <span className={COMPONENTS.text.primary}>{training.hyperparameters.total_steps.toLocaleString()}</span></div>
                  {training.hyperparameters.target_l0 !== undefined && (
                    <div>
                      Target L0:{' '}
                      <span className={COMPONENTS.text.primary} title={`${(training.hyperparameters.target_l0 * 100).toFixed(1)}% of ${training.hyperparameters.latent_dim} features`}>
                        {formatL0Absolute(training.hyperparameters.target_l0, training.hyperparameters.latent_dim)}
                      </span>
                      <span className="text-slate-500 ml-1">({(training.hyperparameters.target_l0 * 100).toFixed(1)}%)</span>
                    </div>
                  )}
                  {training.current_l0_sparsity !== null && training.current_l0_sparsity !== undefined && (
                    <div>
                      Actual L0:{' '}
                      <span className={COMPONENTS.text.primary} title={`${(training.current_l0_sparsity * 100).toFixed(1)}% of ${training.hyperparameters.latent_dim} features`}>
                        {formatL0Absolute(training.current_l0_sparsity, training.hyperparameters.latent_dim)}
                      </span>
                      <span className="text-slate-500 ml-1">({(training.current_l0_sparsity * 100).toFixed(1)}%)</span>
                    </div>
                  )}
                </>
              )}

              {/* Extraction Configuration */}
              <div className="col-span-3 text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mt-2 mb-1">Extraction</div>
              <div className="col-span-3">Extraction ID: <span className={`${COMPONENTS.text.primary} font-mono text-xs`}>{extraction.id}</span></div>
              <div>Eval Samples: <span className={COMPONENTS.text.primary}>{extraction.config.evaluation_samples?.toLocaleString() || 'N/A'}</span></div>
              <div className="col-span-2">Top-K Examples: <span className={COMPONENTS.text.primary}>{extraction.config.top_k_examples || 'N/A'}</span></div>

              {/* Context Window Configuration */}
              <div className="col-span-3 text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mt-2 mb-1">Context Window</div>
              <div>Prefix Tokens: <span className={COMPONENTS.text.primary}>{extraction.context_prefix_tokens ?? 25}</span></div>
              <div>Prime Token: <span className="text-emerald-400">1</span></div>
              <div>Suffix Tokens: <span className={COMPONENTS.text.primary}>{extraction.context_suffix_tokens ?? 25}</span></div>

              {/* Token Filtering Configuration */}
              <div className="col-span-3 text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mt-2 mb-1">Token Filtering</div>
              <div className="col-span-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                <div className="flex items-center gap-1.5">
                  <span className={extraction.filter_special !== false ? 'text-emerald-400' : 'text-slate-500'}>
                    {extraction.filter_special !== false ? '✓' : '○'}
                  </span>
                  <span className={extraction.filter_special !== false ? 'text-slate-300' : 'text-slate-500'}>
                    Special tokens
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className={extraction.filter_single_char !== false ? 'text-emerald-400' : 'text-slate-500'}>
                    {extraction.filter_single_char !== false ? '✓' : '○'}
                  </span>
                  <span className={extraction.filter_single_char !== false ? 'text-slate-300' : 'text-slate-500'}>
                    Single characters
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className={extraction.filter_numbers !== false ? 'text-emerald-400' : 'text-slate-500'}>
                    {extraction.filter_numbers !== false ? '✓' : '○'}
                  </span>
                  <span className={extraction.filter_numbers !== false ? 'text-slate-300' : 'text-slate-500'}>
                    Numbers
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className={extraction.filter_fragments !== false ? 'text-emerald-400' : 'text-slate-500'}>
                    {extraction.filter_fragments !== false ? '✓' : '○'}
                  </span>
                  <span className={extraction.filter_fragments !== false ? 'text-slate-300' : 'text-slate-500'}>
                    Word fragments
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className={extraction.filter_stop_words === true ? 'text-emerald-400' : 'text-slate-500'}>
                    {extraction.filter_stop_words === true ? '✓' : '○'}
                  </span>
                  <span className={extraction.filter_stop_words === true ? 'text-slate-300' : 'text-slate-500'}>
                    Stop words
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className={extraction.filter_punctuation !== false ? 'text-emerald-400' : 'text-slate-500'}>
                    {extraction.filter_punctuation !== false ? '✓' : '○'}
                  </span>
                  <span className={extraction.filter_punctuation !== false ? 'text-slate-300' : 'text-slate-500'}>
                    Punctuation
                  </span>
                </div>
              </div>
            </div>
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
