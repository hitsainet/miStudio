/**
 * ResourceConfigPanel Component
 *
 * Interactive panel for configuring resource usage for feature extraction jobs.
 * Provides real-time resource estimation, recommendations, and validation.
 *
 * Features:
 * - Adjustable sliders for batch_size, num_workers, db_commit_batch
 * - Real-time RAM/GPU/duration estimates
 * - Visual progress bars with color-coded warnings
 * - Auto-optimize button to apply recommended settings
 * - Warnings and errors displayed inline
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Settings, Zap, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import { getResourceEstimate } from '../../api/system';
import type { ResourceEstimateResponse } from '../../types/system';

interface ResourceConfigPanelProps {
  trainingId: string;
  evaluationSamples: number;
  topKExamples: number;
  onConfigChange: (config: {
    batch_size?: number;
    num_workers?: number;
    db_commit_batch?: number;
  }) => void;
}

export const ResourceConfigPanel: React.FC<ResourceConfigPanelProps> = ({
  trainingId,
  evaluationSamples,
  topKExamples,
  onConfigChange,
}) => {
  // State for resource settings
  const [batchSize, setBatchSize] = useState<number | null>(null);
  const [numWorkers, setNumWorkers] = useState<number | null>(null);
  const [dbCommitBatch, setDbCommitBatch] = useState<number | null>(null);

  // State for resource estimation
  const [estimate, setEstimate] = useState<ResourceEstimateResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Debounce timer for API calls
  const [debounceTimer, setDebounceTimer] = useState<number | null>(null);

  /**
   * Fetch resource estimate from backend.
   */
  const fetchEstimate = useCallback(
    async (batch?: number | null, workers?: number | null, commit?: number | null) => {
      // Validate parameters before making API call
      if (evaluationSamples < 1000 || evaluationSamples > 100000) {
        setError('Evaluation samples must be between 1,000 and 100,000');
        setIsLoading(false);
        return;
      }
      if (topKExamples < 10 || topKExamples > 1000) {
        setError('Top-K examples must be between 10 and 1,000');
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        const response = await getResourceEstimate({
          training_id: trainingId,
          evaluation_samples: evaluationSamples,
          top_k_examples: topKExamples,
          ...(batch !== null && batch !== undefined && { batch_size: batch }),
          ...(workers !== null && workers !== undefined && { num_workers: workers }),
          ...(commit !== null && commit !== undefined && { db_commit_batch: commit }),
        });

        setEstimate(response);

        // Initialize sliders with recommended values if not set
        if (batchSize === null) {
          setBatchSize(response.recommended_settings.batch_size);
        }
        if (numWorkers === null) {
          setNumWorkers(response.recommended_settings.num_workers);
        }
        if (dbCommitBatch === null) {
          setDbCommitBatch(response.recommended_settings.db_commit_batch);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch resource estimate');
      } finally {
        setIsLoading(false);
      }
    },
    [trainingId, evaluationSamples, topKExamples, batchSize, numWorkers, dbCommitBatch]
  );

  // Initial fetch
  useEffect(() => {
    fetchEstimate();
  }, [trainingId, evaluationSamples, topKExamples]);

  /**
   * Debounced update when sliders change.
   */
  const debouncedFetch = useCallback(
    (batch: number | null, workers: number | null, commit: number | null) => {
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }

      const timer = setTimeout(() => {
        fetchEstimate(batch, workers, commit);
      }, 500);

      setDebounceTimer(timer);
    },
    [debounceTimer, fetchEstimate]
  );

  /**
   * Handle batch size change.
   */
  const handleBatchSizeChange = (value: number) => {
    setBatchSize(value);
    debouncedFetch(value, numWorkers, dbCommitBatch);
    onConfigChange({
      batch_size: value,
      num_workers: numWorkers ?? undefined,
      db_commit_batch: dbCommitBatch ?? undefined,
    });
  };

  /**
   * Handle num workers change.
   */
  const handleNumWorkersChange = (value: number) => {
    setNumWorkers(value);
    debouncedFetch(batchSize, value, dbCommitBatch);
    onConfigChange({
      batch_size: batchSize ?? undefined,
      num_workers: value,
      db_commit_batch: dbCommitBatch ?? undefined,
    });
  };

  /**
   * Handle db commit batch change.
   */
  const handleDbCommitBatchChange = (value: number) => {
    setDbCommitBatch(value);
    debouncedFetch(batchSize, numWorkers, value);
    onConfigChange({
      batch_size: batchSize ?? undefined,
      num_workers: numWorkers ?? undefined,
      db_commit_batch: value,
    });
  };

  /**
   * Auto-optimize: Apply recommended settings.
   */
  const handleAutoOptimize = () => {
    if (!estimate) return;

    setBatchSize(estimate.recommended_settings.batch_size);
    setNumWorkers(estimate.recommended_settings.num_workers);
    setDbCommitBatch(estimate.recommended_settings.db_commit_batch);

    fetchEstimate(
      estimate.recommended_settings.batch_size,
      estimate.recommended_settings.num_workers,
      estimate.recommended_settings.db_commit_batch
    );

    onConfigChange({
      batch_size: estimate.recommended_settings.batch_size,
      num_workers: estimate.recommended_settings.num_workers,
      db_commit_batch: estimate.recommended_settings.db_commit_batch,
    });
  };

  /**
   * Reset to defaults (use recommended).
   */
  const handleReset = () => {
    setBatchSize(null);
    setNumWorkers(null);
    setDbCommitBatch(null);
    fetchEstimate(null, null, null);
    onConfigChange({});
  };

  /**
   * Get color for resource usage percentage.
   */
  const getUsageColor = (usedGb: number, availableGb: number): string => {
    const percent = (usedGb / availableGb) * 100;
    if (percent < 70) return 'bg-emerald-500';
    if (percent < 90) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  /**
   * Format duration for display.
   */
  const formatDuration = (minutes: number): string => {
    if (minutes < 60) {
      return `~${Math.round(minutes)} min`;
    }
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `~${hours}h ${mins}m`;
  };

  if (error) {
    return (
      <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
        <div className="flex items-start space-x-2 text-red-400">
          <XCircle size={20} className="mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-medium">Failed to load resource configuration</p>
            <p className="text-sm text-slate-400 mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!estimate) {
    return (
      <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
        <div className="flex items-center space-x-2 text-slate-400">
          <div className="animate-spin rounded-full h-5 w-5 border-2 border-slate-400 border-t-transparent" />
          <span>Loading resource configuration...</span>
        </div>
      </div>
    );
  }

  const { system_resources, recommended_settings, current_settings: _current_settings, resource_estimates } = estimate;
  const hasErrors = resource_estimates.errors.length > 0;
  const hasWarnings = resource_estimates.warnings.length > 0;

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Settings size={18} className="text-slate-400" />
          <h3 className="font-medium text-slate-200">Resource Configuration</h3>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleAutoOptimize}
            className="px-3 py-1 text-sm bg-emerald-600 hover:bg-emerald-700 text-white rounded transition-colors flex items-center space-x-1"
            disabled={isLoading}
          >
            <Zap size={14} />
            <span>Auto-Optimize</span>
          </button>
          <button
            onClick={handleReset}
            className="px-3 py-1 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition-colors"
            disabled={isLoading}
          >
            Reset
          </button>
        </div>
      </div>

      {/* System Resources Summary */}
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div className="bg-slate-900/50 rounded p-2">
          <p className="text-slate-400 text-xs">CPU Cores</p>
          <p className="text-slate-100 font-medium">{system_resources.cpu_cores}</p>
        </div>
        <div className="bg-slate-900/50 rounded p-2">
          <p className="text-slate-400 text-xs">Available RAM</p>
          <p className="text-slate-100 font-medium">
            {system_resources.available_ram_gb.toFixed(1)} GB
          </p>
        </div>
        {system_resources.gpu_available && (
          <div className="bg-slate-900/50 rounded p-2">
            <p className="text-slate-400 text-xs">Available GPU</p>
            <p className="text-slate-100 font-medium">
              {system_resources.gpu_memory_available_gb?.toFixed(1)} GB
            </p>
          </div>
        )}
      </div>

      {/* Processing Settings Sliders */}
      <div className="space-y-3">
        {/* Batch Size */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-sm text-slate-300">Batch Size</label>
            <span className="text-sm text-slate-400">
              {batchSize} {batchSize !== recommended_settings.batch_size && (
                <span className="text-xs text-slate-500">
                  (recommended: {recommended_settings.batch_size})
                </span>
              )}
            </span>
          </div>
          <input
            type="range"
            min="8"
            max="256"
            step="8"
            value={batchSize ?? recommended_settings.batch_size}
            onChange={(e) => handleBatchSizeChange(parseInt(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            disabled={isLoading}
          />
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>8</span>
            <span>256</span>
          </div>
        </div>

        {/* CPU Workers */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-sm text-slate-300">CPU Workers</label>
            <span className="text-sm text-slate-400">
              {numWorkers} {numWorkers !== recommended_settings.num_workers && (
                <span className="text-xs text-slate-500">
                  (recommended: {recommended_settings.num_workers})
                </span>
              )}
            </span>
          </div>
          <input
            type="range"
            min="1"
            max={system_resources.cpu_cores}
            step="1"
            value={numWorkers ?? recommended_settings.num_workers}
            onChange={(e) => handleNumWorkersChange(parseInt(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            disabled={isLoading}
          />
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>1</span>
            <span>{system_resources.cpu_cores}</span>
          </div>
        </div>

        {/* DB Commit Batch */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-sm text-slate-300">DB Commit Batch</label>
            <span className="text-sm text-slate-400">
              {dbCommitBatch} {dbCommitBatch !== recommended_settings.db_commit_batch && (
                <span className="text-xs text-slate-500">
                  (recommended: {recommended_settings.db_commit_batch})
                </span>
              )}
            </span>
          </div>
          <input
            type="range"
            min="500"
            max="5000"
            step="500"
            value={dbCommitBatch ?? recommended_settings.db_commit_batch}
            onChange={(e) => handleDbCommitBatchChange(parseInt(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            disabled={isLoading}
          />
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>500</span>
            <span>5000</span>
          </div>
        </div>
      </div>

      {/* Estimated Resource Usage */}
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-slate-300">Estimated Resource Usage</h4>

        {/* RAM Usage */}
        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-slate-400">RAM</span>
            <span className="text-slate-300">
              {resource_estimates.estimated_ram_gb.toFixed(1)} GB /{' '}
              {system_resources.available_ram_gb.toFixed(1)} GB available
            </span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-2">
            <div
              className={`${getUsageColor(
                resource_estimates.estimated_ram_gb,
                system_resources.available_ram_gb
              )} h-2 rounded-full transition-all duration-300`}
              style={{
                width: `${Math.min(
                  (resource_estimates.estimated_ram_gb / system_resources.available_ram_gb) * 100,
                  100
                )}%`,
              }}
            />
          </div>
        </div>

        {/* GPU Usage */}
        {system_resources.gpu_available && resource_estimates.estimated_gpu_gb !== null && (
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-slate-400">GPU Memory</span>
              <span className="text-slate-300">
                {resource_estimates.estimated_gpu_gb.toFixed(1)} GB /{' '}
                {system_resources.gpu_memory_available_gb?.toFixed(1)} GB available
              </span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className={`${getUsageColor(
                  resource_estimates.estimated_gpu_gb,
                  system_resources.gpu_memory_available_gb || 1
                )} h-2 rounded-full transition-all duration-300`}
                style={{
                  width: `${Math.min(
                    (resource_estimates.estimated_gpu_gb /
                      (system_resources.gpu_memory_available_gb || 1)) *
                      100,
                    100
                  )}%`,
                }}
              />
            </div>
          </div>
        )}

        {/* Duration */}
        <div className="flex items-center justify-between text-sm pt-1">
          <span className="text-slate-400">Estimated Time</span>
          <span className="text-slate-200 font-medium">
            {formatDuration(resource_estimates.estimated_duration_minutes)}
          </span>
        </div>
      </div>

      {/* Warnings and Errors */}
      {(hasWarnings || hasErrors) && (
        <div className="space-y-2">
          {hasErrors && (
            <div className="bg-red-900/20 border border-red-800 rounded p-2 space-y-1">
              {resource_estimates.errors.map((error, idx) => (
                <div key={idx} className="flex items-start space-x-2 text-red-400 text-sm">
                  <XCircle size={16} className="mt-0.5 flex-shrink-0" />
                  <span>{error}</span>
                </div>
              ))}
            </div>
          )}
          {hasWarnings && (
            <div className="bg-yellow-900/20 border border-yellow-800 rounded p-2 space-y-1">
              {resource_estimates.warnings.map((warning, idx) => (
                <div key={idx} className="flex items-start space-x-2 text-yellow-400 text-sm">
                  <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
                  <span>{warning}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Success indicator when no errors/warnings */}
      {!hasErrors && !hasWarnings && (
        <div className="flex items-center space-x-2 text-emerald-400 text-sm">
          <CheckCircle size={16} />
          <span>Configuration looks good!</span>
        </div>
      )}

      {/* Loading indicator */}
      {isLoading && (
        <div className="flex items-center justify-center py-2">
          <div className="animate-spin rounded-full h-5 w-5 border-2 border-slate-400 border-t-transparent" />
        </div>
      )}
    </div>
  );
};
