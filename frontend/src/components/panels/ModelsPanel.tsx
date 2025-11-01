/**
 * ModelsPanel - Main model management panel.
 *
 * This component provides the primary interface for managing PyTorch models,
 * including downloading from HuggingFace, quantization, and activation extraction.
 *
 * Features:
 * - Download models from HuggingFace with quantization options
 * - Real-time progress tracking via WebSocket
 * - Model architecture visualization
 * - Activation extraction configuration
 * - Support for gated models via access tokens
 */

import { useEffect, useState } from 'react';
import { useModelsStore } from '../../stores/modelsStore';
import { useAllModelsProgress } from '../../hooks/useModelProgress';
import { Model } from '../../types/model';
import { ModelDownloadForm } from '../models/ModelDownloadForm';
import { ModelCard } from '../models/ModelCard';
import { ModelArchitectureViewer } from '../models/ModelArchitectureViewer';
import { ActivationExtractionConfig } from '../models/ActivationExtractionConfig';
import { ExtractionListModal } from '../models/ExtractionListModal';
import { ExtractionDetailModal } from '../models/ExtractionDetailModal';
import { DeleteExtractionsModal } from '../models/DeleteExtractionsModal';
import { Extraction } from '../models/ExtractionListModal';

export function ModelsPanel() {
  const { models, loading, error, fetchModels, downloadModel, deleteModel, cancelDownload, extractActivations } = useModelsStore();
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [showArchitectureViewer, setShowArchitectureViewer] = useState(false);
  const [showExtractionConfig, setShowExtractionConfig] = useState(false);
  const [showExtractionList, setShowExtractionList] = useState(false);
  const [showExtractionDetail, setShowExtractionDetail] = useState(false);
  const [showDeleteExtractions, setShowDeleteExtractions] = useState(false);
  const [extractionModel, setExtractionModel] = useState<Model | null>(null);
  const [historyModel, setHistoryModel] = useState<Model | null>(null);
  const [deleteExtractionsModel, setDeleteExtractionsModel] = useState<Model | null>(null);
  const [selectedExtraction, setSelectedExtraction] = useState<Extraction | null>(null);

  // Subscribe to WebSocket updates for all active models
  useAllModelsProgress();

  // Fetch models on mount
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleDownload = async (repoId: string, quantization: string, accessToken?: string, trustRemoteCode?: boolean) => {
    await downloadModel(repoId, quantization as any, accessToken, trustRemoteCode);
  };

  const handleModelClick = (model: Model) => {
    setSelectedModel(model);
    setShowArchitectureViewer(true);
  };

  // Get the latest model data from store when displaying architecture viewer
  const latestSelectedModel = selectedModel
    ? models.find(m => m.id === selectedModel.id) || selectedModel
    : null;

  const handleExtractActivations = (model: Model) => {
    setExtractionModel(model);
    setShowExtractionConfig(true);
  };

  const handleViewExtractions = (model: Model) => {
    setHistoryModel(model);
    setShowExtractionList(true);
  };

  const handleSelectExtraction = (extraction: Extraction) => {
    setSelectedExtraction(extraction);
    setShowExtractionList(false);
    setShowExtractionDetail(true);
  };

  const handleBackToList = () => {
    setShowExtractionDetail(false);
    setShowExtractionList(true);
  };

  const handleCloseExtractionModals = () => {
    setShowExtractionList(false);
    setShowExtractionDetail(false);
    setHistoryModel(null);
    setSelectedExtraction(null);
  };

  const handleDeleteExtractions = (model: Model) => {
    setDeleteExtractionsModel(model);
    setShowDeleteExtractions(true);
  };

  const handleDeleteExtractionsSubmit = async (extractionIds: string[]) => {
    if (!deleteExtractionsModel) return;

    try {
      // Import the API function
      const { deleteExtractions } = await import('../../api/models');

      // Call the API to delete extractions
      const response = await deleteExtractions(deleteExtractionsModel.id, extractionIds);

      console.log('[ModelsPanel] Deletion response:', response);

      // Show success message with details
      if (response.failed_count > 0) {
        alert(
          `Deleted ${response.deleted_count} extraction(s).\n` +
          `Failed to delete ${response.failed_count} extraction(s).\n\n` +
          `${response.message}`
        );
      } else {
        alert(`Successfully deleted ${response.deleted_count} extraction(s)!`);
      }

      // Refresh models to update the UI
      await fetchModels();

    } catch (error) {
      console.error('[ModelsPanel] Failed to delete extractions:', error);
      alert(
        'Failed to delete extractions.\n\n' +
        `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteModel(id);
    } catch (error) {
      console.error('[ModelsPanel] Failed to delete model:', error);
    }
  };

  const handleCancel = async (id: string) => {
    try {
      await cancelDownload(id);
    } catch (error) {
      console.error('[ModelsPanel] Failed to cancel download:', error);
    }
  };

  // Adapter function to match ActivationExtractionConfig's expected signature
  const handleExtractActivationsSubmit = async (modelId: string, config: any) => {
    await extractActivations(
      modelId,
      config.dataset_id,
      config.layer_indices,
      config.hook_types,
      config.max_samples,
      config.batch_size
    );
  };

  return (
    <div className="">
      <div className="max-w-[80%] mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100 mb-2">Models</h1>
          <p className="text-slate-600 dark:text-slate-400">
            Download and manage PyTorch models with quantization support
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
          <ModelDownloadForm onDownload={handleDownload} />
        </div>

        {/* Loading state */}
        {loading && models.length === 0 && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 border-t-emerald-500"></div>
            <p className="text-slate-400 mt-4">Loading models...</p>
          </div>
        )}

        {/* Empty state */}
        {!loading && models.length === 0 && (
          <div className="text-center py-12">
            <p className="text-slate-400 text-lg">No models yet</p>
            <p className="text-slate-500 mt-2">Download a model from HuggingFace to get started</p>
          </div>
        )}

        {/* Models grid */}
        {models.length > 0 && (
          <div>
            <h2 className="text-lg font-semibold text-slate-100 mb-4">
              Your Models ({models.length})
            </h2>
            <div className="grid gap-4">
              {[...models]
                .sort((a, b) => {
                  // Active downloads first (downloading, loading, quantizing)
                  const aActive = ['downloading', 'loading', 'quantizing'].includes(a.status);
                  const bActive = ['downloading', 'loading', 'quantizing'].includes(b.status);

                  if (aActive && !bActive) return -1;
                  if (!aActive && bActive) return 1;

                  // Then by creation time (newest first)
                  return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
                })
                .map((model) => (
                  <ModelCard
                    key={model.id}
                    model={model}
                    onClick={() => handleModelClick(model)}
                    onExtract={() => handleExtractActivations(model)}
                    onViewExtractions={() => handleViewExtractions(model)}
                    onDeleteExtractions={() => handleDeleteExtractions(model)}
                    onDelete={handleDelete}
                    onCancel={handleCancel}
                  />
                ))}
            </div>
          </div>
        )}

        {/* Model Architecture Viewer Modal */}
        {showArchitectureViewer && latestSelectedModel && (
          <ModelArchitectureViewer
            model={latestSelectedModel}
            onClose={() => {
              setShowArchitectureViewer(false);
              setSelectedModel(null);
            }}
          />
        )}

        {/* Activation Extraction Config Modal */}
        {showExtractionConfig && extractionModel && (
          <ActivationExtractionConfig
            model={extractionModel}
            onClose={() => {
              setShowExtractionConfig(false);
              setExtractionModel(null);
            }}
            onExtract={handleExtractActivationsSubmit}
          />
        )}

        {/* Extraction List Modal (Step 1) */}
        {showExtractionList && historyModel && (
          <ExtractionListModal
            model={historyModel}
            onClose={handleCloseExtractionModals}
            onSelectExtraction={handleSelectExtraction}
          />
        )}

        {/* Extraction Detail Modal (Step 2) */}
        {showExtractionDetail && selectedExtraction && (
          <ExtractionDetailModal
            extraction={selectedExtraction}
            onClose={handleCloseExtractionModals}
            onBack={handleBackToList}
          />
        )}

        {/* Delete Extractions Modal */}
        {showDeleteExtractions && deleteExtractionsModel && (
          <DeleteExtractionsModal
            model={deleteExtractionsModel}
            onClose={() => {
              setShowDeleteExtractions(false);
              setDeleteExtractionsModel(null);
            }}
            onDelete={handleDeleteExtractionsSubmit}
          />
        )}
      </div>
    </div>
  );
}
