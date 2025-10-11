/**
 * DatasetsPanel - Main dataset management panel.
 *
 * This component provides the primary interface for managing datasets,
 * including viewing, downloading, and monitoring dataset operations.
 */

import { useEffect, useState } from 'react';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { DownloadForm } from '../datasets/DownloadForm';
import { DatasetCard } from '../datasets/DatasetCard';
import { DatasetDetailModal } from '../datasets/DatasetDetailModal';
import { Dataset } from '../../types/dataset';

export function DatasetsPanel() {
  const { datasets, loading, error, fetchDatasets, downloadDataset, deleteDataset } = useDatasetsStore();
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);

  // WebSocket progress updates are now handled globally in App.tsx via useGlobalDatasetProgress()

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

  return (
    <div className="min-h-screen bg-slate-950">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-slate-100 mb-2">Datasets</h1>
          <p className="text-slate-400">
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
        <div className="mb-8">
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
            <h2 className="text-lg font-semibold text-slate-100 mb-4">
              Your Datasets ({datasets.length})
            </h2>
            <div className="grid gap-4 md:grid-cols-2">
              {datasets.map((dataset) => (
                <DatasetCard
                  key={dataset.id}
                  dataset={dataset}
                  onClick={() => handleDatasetClick(dataset)}
                  onDelete={handleDelete}
                />
              ))}
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
