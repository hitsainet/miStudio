/**
 * ExtractionsPanel Component
 *
 * Panel for viewing and managing all feature extraction jobs.
 * Shows extractions in a grid with status, progress, and actions.
 */

import React, { useEffect, useState } from 'react';
import { Zap, Filter } from 'lucide-react';
import { useFeaturesStore } from '../../stores/featuresStore';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { ExtractionJobCard } from '../features/ExtractionJobCard';

export const ExtractionsPanel: React.FC = () => {
  const {
    allExtractions,
    extractionsMetadata,
    isLoadingExtractions,
    extractionsError,
    fetchAllExtractions,
    cancelExtraction,
    deleteExtraction,
  } = useFeaturesStore();

  const { trainings, fetchTrainings } = useTrainingsStore();

  const [statusFilter, setStatusFilter] = useState<string[]>([]);

  // Load trainings and extractions on mount
  useEffect(() => {
    fetchTrainings();
    fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined);
  }, []);

  // Reload extractions when status filter changes
  useEffect(() => {
    fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined);
  }, [statusFilter]);

  // Helper to get training name from training ID
  const getTrainingName = (trainingId: string): string => {
    const training = trainings.find((t) => t.id === trainingId);
    if (!training) return trainingId;

    // Build a descriptive name from training metadata
    const dataset = training.dataset_name || 'Unknown Dataset';
    const model = training.model_name || 'Unknown Model';
    return `${model} - ${dataset}`;
  };

  // Toggle status filter
  const toggleStatusFilter = (status: string) => {
    setStatusFilter((prev) => {
      if (prev.includes(status)) {
        return prev.filter((s) => s !== status);
      } else {
        return [...prev, status];
      }
    });
  };

  return (
    <div className="max-w-[80%] mx-auto px-6 py-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <Zap className="w-8 h-8 text-emerald-400" />
          <div>
            <h2 className="text-2xl font-bold text-white">Feature Extractions</h2>
            <p className="text-sm text-slate-400">View and manage all feature extraction jobs</p>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="mb-6 bg-slate-900/50 rounded-lg p-4 border border-slate-800">
        <div className="flex items-center gap-3">
          <Filter className="w-5 h-5 text-slate-400" />
          <span className="text-sm font-medium text-slate-300">Filter by status:</span>
          <div className="flex gap-2">
            {['queued', 'extracting', 'completed', 'failed'].map((status) => (
              <button
                key={status}
                onClick={() => toggleStatusFilter(status)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  statusFilter.includes(status)
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
              >
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </button>
            ))}
          </div>
          {statusFilter.length > 0 && (
            <button
              onClick={() => setStatusFilter([])}
              className="ml-auto text-sm text-slate-400 hover:text-slate-300"
            >
              Clear filters
            </button>
          )}
        </div>
      </div>

      {/* Loading State */}
      {isLoadingExtractions && (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-400"></div>
          <p className="mt-4 text-slate-400">Loading extractions...</p>
        </div>
      )}

      {/* Error State */}
      {extractionsError && (
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-6 text-center">
          <p className="text-red-400">{extractionsError}</p>
          <button
            onClick={() => fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined)}
            className="mt-4 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* Extractions Grid */}
      {!isLoadingExtractions && !extractionsError && (
        <>
          {allExtractions.length === 0 ? (
            <div className="text-center py-12">
              <Zap className="w-16 h-16 text-slate-700 mx-auto mb-4" />
              <p className="text-slate-400 text-lg mb-2">No extraction jobs found</p>
              <p className="text-slate-500 text-sm">
                {statusFilter.length > 0
                  ? 'Try removing some filters'
                  : 'Start a feature extraction from a completed training'}
              </p>
            </div>
          ) : (
            <>
              {/* Stats */}
              {extractionsMetadata && (
                <div className="mb-4 text-sm text-slate-400">
                  Showing {allExtractions.length} of {extractionsMetadata.total} extraction{extractionsMetadata.total !== 1 ? 's' : ''}
                </div>
              )}

              {/* Grid */}
              <div className="grid grid-cols-1 gap-4">
                {allExtractions.map((extraction) => (
                  <ExtractionJobCard
                    key={extraction.id}
                    extraction={extraction}
                    trainingName={getTrainingName(extraction.training_id)}
                    onCancel={async () => {
                      await cancelExtraction(extraction.training_id);
                      // Refresh list after cancellation
                      fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined);
                    }}
                    onDelete={async () => {
                      await deleteExtraction(extraction.id, extraction.training_id);
                      // Refresh list after deletion
                      fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined);
                    }}
                  />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
};
