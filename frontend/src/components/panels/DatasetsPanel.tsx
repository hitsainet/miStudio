/**
 * DatasetsPanel - Main dataset management panel.
 *
 * This component provides the primary interface for managing datasets,
 * including viewing, downloading, and monitoring dataset operations.
 */

import { useEffect, useState, useMemo } from 'react';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useTokenizationWebSocket } from '../../hooks/useTokenizationWebSocket';
import { DownloadForm } from '../datasets/DownloadForm';
import { DatasetCard } from '../datasets/DatasetCard';
import { DatasetDetailModal } from '../datasets/DatasetDetailModal';
import { Dataset, TokenizationStatus } from '../../types/dataset';

export function DatasetsPanel() {
  const {
    datasets,
    loading,
    error,
    tokenizationProgress,
    fetchDatasets,
    downloadDataset,
    deleteDataset,
    cancelDownload
  } = useDatasetsStore();
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);

  // Get active tokenizations from datasets for WebSocket subscriptions
  const activeTokenizations = useMemo(() => {
    const result: Array<{ datasetId: string; tokenizationId: string }> = [];

    console.log('[DatasetsPanel] Computing activeTokenizations, datasets:', datasets.length);
    datasets.forEach(dataset => {
      // Check if dataset has tokenizations
      const tokenizations = dataset.tokenizations || [];
      console.log(`[DatasetsPanel] Dataset ${dataset.name} has ${tokenizations.length} tokenizations`);
      tokenizations.forEach(tokenization => {
        console.log(`[DatasetsPanel] - Tokenization ${tokenization.id} status: ${tokenization.status}`);
        // Subscribe to processing or queued tokenizations
        if (
          tokenization.status === TokenizationStatus.PROCESSING ||
          tokenization.status === TokenizationStatus.QUEUED
        ) {
          console.log(`[DatasetsPanel] - Adding to active list`);
          result.push({
            datasetId: dataset.id,
            tokenizationId: tokenization.id,
          });
        }
      });
    });

    console.log('[DatasetsPanel] Found', result.length, 'active tokenizations');
    return result;
  }, [datasets]);

  // Subscribe to WebSocket updates for active tokenizations
  useTokenizationWebSocket(activeTokenizations);

  // Fetch datasets on mount
  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  const handleDownload = async (repoId: string, accessToken?: string, split?: string, config?: string) => {
    await downloadDataset(repoId, accessToken, split, config);
  };

  const handleDatasetClick = (dataset: Dataset) => {
    setSelectedDataset(dataset);
  };

  const handleDatasetUpdate = (updatedDataset: Dataset) => {
    // Update the selected dataset with the new data
    setSelectedDataset(updatedDataset);
    // Optionally, refresh all datasets to keep the list in sync
    fetchDatasets();
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteDataset(id);
    } catch (error) {
      console.error('Failed to delete dataset:', error);
    }
  };

  const handleCancel = async (id: string) => {
    try {
      await cancelDownload(id);
    } catch (error) {
      console.error('Failed to cancel dataset operation:', error);
    }
  };

  return (
    <div className="">
      <div className="max-w-[90%] mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">Datasets</h1>
          <p className="text-slate-600 dark:text-slate-400">
            Manage training datasets from HuggingFace or local sources
          </p>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {/* Download Form */}
        <div className="mb-6">
          <DownloadForm onDownload={handleDownload} />
        </div>

        {/* Loading state */}
        {loading && datasets.length === 0 && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 border-t-emerald-500"></div>
            <p className="text-slate-400 mt-4">Loading datasets...</p>
          </div>
        )}

        {/* Empty state */}
        {!loading && datasets.length === 0 && (
          <div className="text-center py-12">
            <p className="text-slate-400 text-lg">No datasets yet</p>
            <p className="text-slate-500 mt-2">Download a dataset from HuggingFace to get started</p>
          </div>
        )}

        {/* Datasets grid */}
        {datasets.length > 0 && (
          <div>
            <h2 className="text-base font-semibold text-slate-100 mb-3">
              Your Datasets ({datasets.length})
            </h2>
            <div className="grid gap-3 md:grid-cols-2">
              {[...datasets]
                .sort((a, b) => {
                  // Active operations first (downloading, processing)
                  const aActive = ['downloading', 'processing'].includes(a.status);
                  const bActive = ['downloading', 'processing'].includes(b.status);

                  if (aActive && !bActive) return -1;
                  if (!aActive && bActive) return 1;

                  // Then by creation time (newest first)
                  return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
                })
                .map((dataset) => {
                  // Find the active tokenization for this dataset
                  const activeTokenization = dataset.tokenizations?.find(
                    t => t.status === TokenizationStatus.PROCESSING || t.status === TokenizationStatus.QUEUED
                  );

                  console.log(`[DatasetsPanel] Dataset ${dataset.name}:`, {
                    hasTokenizations: !!dataset.tokenizations,
                    tokenizationsCount: dataset.tokenizations?.length,
                    activeTokenization: activeTokenization?.id,
                    activeTokenizationStatus: activeTokenization?.status,
                  });

                  // Get the progress data for the active tokenization
                  const progress = activeTokenization ? tokenizationProgress[activeTokenization.id] : undefined;

                  console.log(`[DatasetsPanel] Progress for ${dataset.name}:`, progress ? {
                    tokenizationId: progress.tokenization_id,
                    stage: progress.stage,
                    progress: progress.progress,
                    samples: `${progress.samples_processed}/${progress.total_samples}`,
                  } : 'No progress data');

                  return (
                    <DatasetCard
                      key={dataset.id}
                      dataset={dataset}
                      tokenizationProgress={progress}
                      onClick={() => handleDatasetClick(dataset)}
                      onDelete={handleDelete}
                      onCancel={handleCancel}
                    />
                  );
                })}
            </div>
          </div>
        )}

        {/* Dataset Detail Modal */}
        {selectedDataset && (
          <DatasetDetailModal
            dataset={selectedDataset}
            onClose={() => setSelectedDataset(null)}
            onDatasetUpdate={handleDatasetUpdate}
          />
        )}
      </div>
    </div>
  );
}
