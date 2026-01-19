/**
 * ExtractionsPanel Component
 *
 * Panel for viewing and managing all feature extraction jobs.
 * Shows extractions in a grid with status, progress, and actions.
 * Supports starting extraction from both trainings and external SAEs.
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Zap, Filter, Plus } from 'lucide-react';
import { useFeaturesStore } from '../../stores/featuresStore';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { ExtractionJobCard } from '../features/ExtractionJobCard';
import { StartExtractionModal } from '../extraction/StartExtractionModal';
import { cancelSAEExtraction } from '../../api/saes';
import { useExtractionWebSocket } from '../../hooks/useExtractionWebSocket';
import { useNlpWebSocket } from '../../hooks/useNlpWebSocket';
import { COMPONENTS } from '../../config/brand';

export const ExtractionsPanel: React.FC = () => {
  const {
    allExtractions,
    extractionsMetadata,
    isLoadingExtractions,
    extractionsError,
    fetchAllExtractions,
    deleteExtraction,
  } = useFeaturesStore();

  const { fetchTrainings } = useTrainingsStore();

  const [statusFilter, setStatusFilter] = useState<string[]>([]);
  const [showStartModal, setShowStartModal] = useState(false);

  // Get IDs of all extractions for WebSocket subscriptions
  // We subscribe to ALL extractions (not just active ones) to receive deletion notifications
  // for completed/failed extractions as well
  const allExtractionIds = useMemo(() => {
    return allExtractions.map(e => e.id);
  }, [allExtractions]);

  // Subscribe to WebSocket updates for all extractions (progress updates + deletion events)
  useExtractionWebSocket(allExtractionIds);

  // Get IDs of extractions with NLP pending or processing for WebSocket subscriptions
  // Include 'pending' to catch updates when NLP starts after auto-trigger
  const nlpActiveExtractionIds = useMemo(() => {
    return allExtractions
      .filter(e => e.nlp_status === 'pending' || e.nlp_status === 'processing')
      .map(e => e.id);
  }, [allExtractions]);

  // Subscribe to WebSocket updates for NLP processing
  useNlpWebSocket(nlpActiveExtractionIds);

  // Load trainings and extractions on mount
  useEffect(() => {
    fetchTrainings();
    fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined);
  }, []);

  // Reload extractions when status filter changes
  useEffect(() => {
    fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined);
  }, [statusFilter]);

  // Note: Extraction already includes model_name and dataset_name, so we don't need to look up training

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

  /**
   * Handle modal close with list refresh.
   */
  const handleModalClose = () => {
    setShowStartModal(false);
    // Refresh extractions list after modal closes
    fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined);
  };

  return (
    <div className="max-w-[80%] mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className={`text-2xl font-semibold ${COMPONENTS.text.primary} mb-2`}>Extractions</h1>
          <p className={COMPONENTS.text.secondary}>
            View and manage all feature extraction jobs
          </p>
        </div>
        <button
          onClick={() => setShowStartModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-colors"
        >
          <Plus className="w-5 h-5" />
          Start Extraction
        </button>
      </div>

      {/* Filters */}
      <div className={`mb-6 ${COMPONENTS.card.base} p-4`}>
        <div className="flex items-center gap-3">
          <Filter className={`w-5 h-5 ${COMPONENTS.text.secondary}`} />
          <span className={`text-sm font-medium ${COMPONENTS.text.primary}`}>Filter by status:</span>
          <div className="flex gap-2">
            {['queued', 'extracting', 'completed', 'failed'].map((status) => (
              <button
                key={status}
                onClick={() => toggleStatusFilter(status)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  statusFilter.includes(status)
                    ? 'bg-emerald-600 dark:bg-emerald-400 text-white'
                    : `${COMPONENTS.button.secondary}`
                }`}
              >
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </button>
            ))}
          </div>
          {statusFilter.length > 0 && (
            <button
              onClick={() => setStatusFilter([])}
              className={`ml-auto text-sm ${COMPONENTS.text.secondary} hover:text-slate-300 dark:hover:text-slate-200`}
            >
              Clear filters
            </button>
          )}
        </div>
      </div>

      {/* Loading State */}
      {isLoadingExtractions && (
        <div className="text-center py-12">
          <div className={COMPONENTS.spinner}></div>
          <p className={`mt-4 ${COMPONENTS.text.secondary}`}>Loading extractions...</p>
        </div>
      )}

      {/* Error State */}
      {extractionsError && (
        <div className="bg-red-500/10 dark:bg-red-500/20 border border-red-500/30 dark:border-red-500/40 rounded-lg p-6 text-center">
          <p className="text-red-400 dark:text-red-300">{extractionsError}</p>
          <button
            onClick={() => fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined)}
            className={`mt-4 ${COMPONENTS.button.secondary}`}
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
              <Zap className={`w-16 h-16 ${COMPONENTS.text.muted} mx-auto mb-4`} />
              <p className={`${COMPONENTS.text.secondary} text-lg mb-2`}>No extraction jobs found</p>
              <p className={`${COMPONENTS.text.muted} text-sm`}>
                {statusFilter.length > 0
                  ? 'Try removing some filters'
                  : 'Start a feature extraction from a completed training'}
              </p>
            </div>
          ) : (
            <>
              {/* Stats */}
              {extractionsMetadata && (
                <div className={`mb-4 text-sm ${COMPONENTS.text.secondary}`}>
                  Showing {allExtractions.length} of {extractionsMetadata.total} extraction{extractionsMetadata.total !== 1 ? 's' : ''}
                </div>
              )}

              {/* Grid */}
              <div className="grid grid-cols-1 gap-4">
                {allExtractions.map((extraction) => (
                  <ExtractionJobCard
                    key={extraction.id}
                    extraction={extraction}
                    onCancel={async () => {
                      // Cancel SAE extraction
                      if (extraction.external_sae_id) {
                        await cancelSAEExtraction(extraction.external_sae_id);
                      }
                      // Refresh list after cancellation
                      fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined);
                    }}
                    onDelete={async () => {
                      try {
                        console.log(`[Extractions] Deleting extraction ${extraction.id}...`);
                        await deleteExtraction(extraction.id);
                        console.log(`[Extractions] Extraction ${extraction.id} deleted successfully`);
                        // Refresh list after deletion
                        fetchAllExtractions(statusFilter.length > 0 ? statusFilter : undefined);
                      } catch (error: any) {
                        console.error(`[Extractions] Failed to delete extraction ${extraction.id}:`, error);
                        // Error is already set in the store, but let's alert the user
                        alert(`Failed to delete extraction: ${error?.message || 'Unknown error'}`);
                      }
                    }}
                  />
                ))}
              </div>
            </>
          )}
        </>
      )}

      {/* Start Extraction Modal */}
      <StartExtractionModal
        isOpen={showStartModal}
        onClose={handleModalClose}
      />
    </div>
  );
};
