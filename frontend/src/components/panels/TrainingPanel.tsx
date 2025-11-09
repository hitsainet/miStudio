/**
 * Training Panel Component
 *
 * Main panel for SAE Training feature. Allows users to configure and launch
 * new SAE training jobs, view active training jobs, and monitor progress.
 *
 * Mock UI Reference: Lines 1628-1842
 * TID Reference: Lines 280-357
 *
 * Features:
 * - Training configuration form (model, dataset, encoder type)
 * - Advanced hyperparameters section (collapsible)
 * - Training jobs list with real-time progress
 * - Status filtering (All/Running/Completed/Failed)
 * - WebSocket integration for live updates
 */

import React, { useEffect, useState, useMemo } from 'react';
import {
  Play,
  ChevronDown,
  ChevronUp,
  Activity,
  CheckCircle,
  XCircle,
  Loader,
  Trash2,
  AlertTriangle,
} from 'lucide-react';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { useModelsStore } from '../../stores/modelsStore';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useTrainingWebSocket } from '../../hooks/useTrainingWebSocket';
import { useWebSocketContext } from '../../contexts/WebSocketContext';
import { TrainingStatus, SAEArchitectureType } from '../../types/training';
import type { TrainingCreateRequest } from '../../types/training';
import { TrainingCard } from '../training/TrainingCard';
import { estimateMultilayerTrainingMemory, formatMemorySize } from '../../utils/memoryEstimation';
import { HyperparameterLabel, HyperparameterTooltip } from '../common/HyperparameterTooltip';
import { calculateOptimalL1Alpha, validateSparsityConfig } from '../../utils/hyperparameterOptimization';
import { COMPONENTS } from '../../config/brand';

export const TrainingPanel: React.FC = () => {
  // Store state
  const {
    trainings,
    config,
    updateConfig,
    fetchTrainings,
    createTraining,
    deleteTraining,
    statusFilter,
    setStatusFilter,
    statusCounts,
    isLoading,
    error,
  } = useTrainingsStore();

  const { models, fetchModels } = useModelsStore();
  const { datasets, fetchDatasets } = useDatasetsStore();

  // WebSocket connection status
  const { isConnected } = useWebSocketContext();

  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [selectedTrainingIds, setSelectedTrainingIds] = useState<Set<string>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);

  // Memory estimation
  const memoryEstimate = useMemo(() => {
    return estimateMultilayerTrainingMemory(
      config.hidden_dim,
      config.latent_dim,
      config.batch_size,
      config.training_layers?.length || 1
    );
  }, [config.hidden_dim, config.latent_dim, config.batch_size, config.training_layers]);

  // Fetch data on mount
  useEffect(() => {
    fetchModels();
    fetchDatasets();
    fetchTrainings();
  }, [fetchModels, fetchDatasets, fetchTrainings]);

  // Subscribe to WebSocket updates for all trainings
  useTrainingWebSocket(trainings.map((t) => t.id));

  // Filter ready models and datasets
  const readyModels = models.filter((m) => m.status === 'ready');
  const readyDatasets = datasets.filter((d) => d.status === 'ready');

  // Get selected model and its layer count
  const selectedModel = models.find((m) => m.id === config.model_id);
  const numLayers = selectedModel?.architecture_config?.num_hidden_layers || 0;

  // Auto-select layer 0 when model is first selected
  useEffect(() => {
    if (config.model_id && numLayers > 0 && (!config.training_layers || config.training_layers.length === 0)) {
      updateConfig({ training_layers: [0] });
    }
  }, [config.model_id, numLayers]);

  // Autodiscover hidden dimension from extraction metadata
  useEffect(() => {
    const autodiscoverHiddenDim = async () => {
      if (!config.extraction_id || config.extraction_id.trim() === '') return;
      if (!config.model_id) return;

      try {
        // Extract model_id from extraction_id format: ext_m_{uuid}_{timestamp}
        const extractionModelId = config.extraction_id.match(/ext_(m_[^_]+)/)?.[1];
        if (!extractionModelId || extractionModelId !== config.model_id) {
          console.warn('Extraction ID model mismatch');
          return;
        }

        // Fetch extraction metadata
        const response = await fetch(`/api/v1/models/${config.model_id}/extractions`);
        if (!response.ok) return;

        const data = await response.json();
        const extraction = data.extractions?.find((e: any) => e.extraction_id === config.extraction_id);

        if (extraction?.statistics) {
          // Find first layer with statistics and extract hidden_dim from shape
          const layerNames = Object.keys(extraction.statistics);
          if (layerNames.length > 0) {
            const firstLayerStats = extraction.statistics[layerNames[0]];
            if (firstLayerStats?.shape && Array.isArray(firstLayerStats.shape) && firstLayerStats.shape.length === 3) {
              // Shape format: [n_samples, seq_len, hidden_dim]
              const hiddenDim = firstLayerStats.shape[2];

              // Only update if different from current value
              if (hiddenDim !== config.hidden_dim) {
                console.log(`[TrainingPanel] Autodiscovered hidden_dim=${hiddenDim} from extraction ${config.extraction_id}`);
                updateConfig({ hidden_dim: hiddenDim });
              }
            }
          }
        }
      } catch (error) {
        console.error('Failed to autodiscover hidden dimension:', error);
      }
    };

    autodiscoverHiddenDim();
  }, [config.extraction_id, config.model_id]);

  // Check for tokenizer/model vocabulary mismatch
  const selectedDataset = datasets.find((d) => d.id === config.dataset_id);
  const vocabMismatch = useMemo(() => {
    if (!selectedModel || !selectedDataset) return null;

    const datasetTokenizerName = selectedDataset.metadata?.tokenization?.tokenizer_name;
    const datasetVocabSize = selectedDataset.metadata?.tokenization?.vocab_size;
    const modelVocabSize = selectedModel.architecture_config?.vocab_size;

    if (!datasetVocabSize || !modelVocabSize) return null;

    const vocabDiff = Math.abs(datasetVocabSize - modelVocabSize);
    const vocabRatio = vocabDiff / modelVocabSize;

    if (vocabRatio > 0.1) {
      return {
        datasetTokenizer: datasetTokenizerName || 'unknown',
        datasetVocabSize,
        modelVocabSize,
        difference: vocabDiff,
        ratio: vocabRatio,
      };
    }

    return null;
  }, [selectedModel, selectedDataset]);

  // Validation
  const isFormValid = config.model_id && config.dataset_id && config.training_layers && config.training_layers.length > 0 && !vocabMismatch;

  // Selection handlers
  const handleToggleSelection = (trainingId: string) => {
    setSelectedTrainingIds((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(trainingId)) {
        newSet.delete(trainingId);
      } else {
        newSet.add(trainingId);
      }
      return newSet;
    });
  };

  const handleSelectAll = () => {
    if (selectedTrainingIds.size === trainings.length) {
      setSelectedTrainingIds(new Set());
    } else {
      setSelectedTrainingIds(new Set(trainings.map((t) => t.id)));
    }
  };

  const handleDeleteSelected = async () => {
    if (selectedTrainingIds.size === 0) return;

    const count = selectedTrainingIds.size;
    if (!confirm(`Are you sure you want to delete ${count} training job${count > 1 ? 's' : ''}? This will remove all associated data and cannot be undone.`)) {
      return;
    }

    setIsDeleting(true);
    try {
      // Delete all selected trainings in parallel
      await Promise.all(
        Array.from(selectedTrainingIds).map((id) => deleteTraining(id))
      );
      // Clear selection after successful deletion
      setSelectedTrainingIds(new Set());
    } catch (error) {
      console.error('Failed to delete selected trainings:', error);
    } finally {
      setIsDeleting(false);
    }
  };

  // Handle start training
  const handleStartTraining = async () => {
    if (!isFormValid) return;

    setIsStarting(true);
    try {
      const request: TrainingCreateRequest = {
        model_id: config.model_id,
        dataset_id: config.dataset_id,
        ...(config.extraction_id && config.extraction_id.trim() !== '' && { extraction_id: config.extraction_id }),
        hyperparameters: {
          hidden_dim: config.hidden_dim,
          latent_dim: config.latent_dim,
          architecture_type: config.architecture_type,
          training_layers: config.training_layers || [0],
          l1_alpha: config.l1_alpha,
          target_l0: config.target_l0,
          learning_rate: config.learning_rate,
          batch_size: config.batch_size,
          total_steps: config.total_steps,
          warmup_steps: config.warmup_steps,
          weight_decay: config.weight_decay,
          grad_clip_norm: config.grad_clip_norm,
          checkpoint_interval: config.checkpoint_interval,
          log_interval: config.log_interval,
          dead_neuron_threshold: config.dead_neuron_threshold,
          resample_dead_neurons: config.resample_dead_neurons,
        },
      };

      await createTraining(request);
      // Don't reset config - keep selections so user can easily start another training
      // Only collapse advanced configuration section
      setShowAdvanced(false);
    } catch (err) {
      console.error('Failed to start training:', err);
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="">
      <div className="max-w-[80%] mx-auto px-6 py-8 space-y-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100 mb-2">SAE Training</h1>
              <p className="text-slate-600 dark:text-slate-400">
                Configure and launch sparse autoencoder training jobs
              </p>
            </div>
            {/* WebSocket Connection Status */}
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400'} animate-pulse`} />
              <span className={`text-xs font-medium ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
                {isConnected ? 'Live' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
        {/* Configuration Section */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-slate-100 mb-4">
            Training Configuration
          </h3>

          {/* Basic Configuration */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            {/* Dataset Selection - First in flow */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Dataset
              </label>
              <select
                value={config.dataset_id}
                onChange={(e) => updateConfig({ dataset_id: e.target.value })}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
              >
                <option value="">Select dataset...</option>
                {readyDatasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Model Selection - Second in flow */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Model
              </label>
              <select
                value={config.model_id}
                onChange={(e) => updateConfig({ model_id: e.target.value })}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
              >
                <option value="">Select model...</option>
                {readyModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Architecture Type Selection - Third in flow */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                SAE Architecture
              </label>
              <select
                value={config.architecture_type}
                onChange={(e) =>
                  updateConfig({ architecture_type: e.target.value as SAEArchitectureType })
                }
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
              >
                <option value={SAEArchitectureType.STANDARD}>Standard</option>
                <option value={SAEArchitectureType.SKIP}>Skip Connection</option>
                <option value={SAEArchitectureType.TRANSCODER}>Transcoder</option>
              </select>
            </div>

          </div>

          {/* Vocabulary Mismatch Warning */}
          {vocabMismatch && (
            <div className="mt-4 p-4 bg-amber-900/20 border border-amber-600/50 rounded-lg">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <h4 className="text-sm font-semibold text-amber-500 mb-2">
                    Tokenizer/Model Vocabulary Mismatch
                  </h4>
                  <div className="text-sm text-slate-300 space-y-1">
                    <p>
                      The selected dataset was tokenized with <span className="font-mono text-amber-400">{vocabMismatch.datasetTokenizer}</span> (vocab: {vocabMismatch.datasetVocabSize.toLocaleString()}),
                      but the selected model uses a vocabulary of {vocabMismatch.modelVocabSize.toLocaleString()} tokens.
                    </p>
                    <p className="text-amber-400 font-medium mt-2">
                      This will cause "index out of bounds" errors during training or feature extraction.
                    </p>
                    <p className="mt-2">
                      Please re-tokenize the dataset using the model's tokenizer (<span className="font-mono text-emerald-400">{selectedModel?.repo_id}</span>)
                      in the Datasets panel before starting training.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Optional: Use Cached Activations */}
          <div className="mt-4">
            <div className="flex items-start gap-3">
              <input
                type="checkbox"
                id="use-cached-activations"
                checked={!!config.extraction_id}
                onChange={(e) => {
                  if (e.target.checked) {
                    // Set a placeholder to enable the checkbox, user will fill in the actual ID
                    updateConfig({ extraction_id: '' });
                  } else {
                    updateConfig({ extraction_id: undefined });
                  }
                }}
                className="mt-1 w-4 h-4 bg-slate-800 border border-slate-700 rounded focus:ring-2 focus:ring-emerald-500"
              />
              <div className="flex-1">
                <label htmlFor="use-cached-activations" className="block text-sm font-medium text-slate-300 mb-1 cursor-pointer">
                  Use Cached Activations (10-20x faster training)
                </label>
                <p className="text-xs text-slate-500 mb-2">
                  Use pre-extracted activations from an extraction job instead of extracting on-the-fly during training. This dramatically speeds up training but requires a completed extraction for the selected model and dataset.
                </p>
                {(config.extraction_id !== undefined) && (
                  <input
                    type="text"
                    id="extraction-id"
                    value={config.extraction_id || ''}
                    onChange={(e) => updateConfig({ extraction_id: e.target.value || undefined })}
                    placeholder="Enter extraction ID (e.g., ext_m_6d64e8d9_20251027_023246)"
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 text-sm focus:outline-none focus:border-emerald-500 transition-colors font-mono"
                  />
                )}
              </div>
            </div>
          </div>

          {/* Training Layers Selection */}
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-slate-300">
                Select Layers ({config.training_layers?.length || 0} selected)
              </label>
              {numLayers > 0 && (
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      const allLayers = Array.from({ length: numLayers }, (_, i) => i);
                      updateConfig({ training_layers: allLayers });
                    }}
                    className={`px-3 py-1 text-xs ${COMPONENTS.button.secondary}`}
                  >
                    Select All
                  </button>
                  <button
                    type="button"
                    onClick={() => updateConfig({ training_layers: [] })}
                    className={`px-3 py-1 text-xs ${COMPONENTS.button.secondary}`}
                  >
                    Deselect All
                  </button>
                </div>
              )}
            </div>
            {numLayers > 0 ? (
              <div className="grid grid-cols-6 gap-2">
                {Array.from({ length: numLayers }, (_, i) => i).map((layerIdx) => {
                  const isSelected = config.training_layers?.includes(layerIdx) || false;
                  return (
                    <button
                      key={layerIdx}
                      type="button"
                      onClick={() => {
                        const currentLayers = config.training_layers || [];
                        if (isSelected) {
                          updateConfig({
                            training_layers: currentLayers.filter((l) => l !== layerIdx),
                          });
                        } else {
                          updateConfig({
                            training_layers: [...currentLayers, layerIdx].sort((a, b) => a - b),
                          });
                        }
                      }}
                      className={`px-3 py-2 text-sm font-medium rounded transition-colors ${
                        isSelected
                          ? 'bg-emerald-600 text-white hover:bg-emerald-700'
                          : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }`}
                    >
                      L{layerIdx}
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="px-4 py-8 text-center text-slate-500 bg-slate-800/50 rounded-lg border border-slate-700">
                {config.model_id
                  ? 'Loading model architecture...'
                  : 'Select a model to choose training layers'}
              </div>
            )}
          </div>

          {/* Memory Estimation Display */}
          <div className="mt-4 p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
            <div className="flex items-start justify-between">
              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-2">
                  Estimated GPU Memory
                </h4>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-400">Total:</span>
                    <span className={`text-sm font-mono font-semibold ${
                      memoryEstimate.fits_in_6gb ? 'text-emerald-400' : 'text-orange-400'
                    }`}>
                      {formatMemorySize(memoryEstimate.total_gb)}
                    </span>
                    {config.training_layers && config.training_layers.length > 1 && (
                      <>
                        <span className="text-xs text-slate-500">×</span>
                        <span className="text-xs text-slate-400">
                          {formatMemorySize(memoryEstimate.per_layer_gb)} per layer
                        </span>
                      </>
                    )}
                  </div>
                  <div className="text-xs text-slate-500">
                    {config.training_layers?.length || 1} layer{(config.training_layers?.length || 1) !== 1 ? 's' : ''}
                    {' • '}
                    {config.hidden_dim}d hidden
                    {' • '}
                    {config.latent_dim}d latent
                    {' • '}
                    batch {config.batch_size}
                  </div>
                </div>
              </div>
              {!memoryEstimate.fits_in_6gb && (
                <AlertTriangle size={20} className="text-orange-400 flex-shrink-0" />
              )}
            </div>
            {memoryEstimate.warning && (
              <div className="mt-3 p-3 bg-orange-900/20 border border-orange-900/50 rounded-md">
                <div className="flex items-start gap-2">
                  <AlertTriangle size={16} className="text-orange-400 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-orange-200">{memoryEstimate.warning}</p>
                </div>
              </div>
            )}
          </div>

          {/* Advanced Configuration Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
          >
            {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            Advanced Configuration
          </button>

          {/* Advanced Hyperparameters */}
          {showAdvanced && (
            <div className="mt-4 pt-4 border-t border-slate-700">
              <div className="grid grid-cols-2 gap-4">
                {/* Hidden Dimension */}
                <div>
                  <HyperparameterLabel
                    paramName="hidden_dim"
                    label="Hidden Dimension"
                    htmlFor="hidden-dim"
                    className="mb-2"
                  />
                  <input
                    id="hidden-dim"
                    type="number"
                    value={config.hidden_dim}
                    onChange={(e) => updateConfig({ hidden_dim: parseInt(e.target.value) })}
                    min={64}
                    max={8192}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  {config.extraction_id && (
                    <p className="mt-1 text-xs text-emerald-400 flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" />
                      Auto-detected from extraction activations
                    </p>
                  )}
                </div>

                {/* Latent Dimension */}
                <div>
                  <HyperparameterLabel
                    paramName="latent_dim"
                    label="Latent Dimension"
                    htmlFor="latent-dim"
                    className="mb-2"
                  />
                  <input
                    id="latent-dim"
                    type="number"
                    value={config.latent_dim}
                    onChange={(e) => updateConfig({ latent_dim: parseInt(e.target.value) })}
                    min={512}
                    max={65536}
                    step={512}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* L1 Alpha with Auto-Calculate */}
                <div>
                  <HyperparameterLabel
                    paramName="l1_alpha"
                    label="L1 Sparsity Coefficient"
                    htmlFor="l1-alpha"
                    className="mb-2"
                  />
                  <div className="flex gap-2">
                    <input
                      id="l1-alpha"
                      type="number"
                      value={config.l1_alpha}
                      onChange={(e) => updateConfig({ l1_alpha: parseFloat(e.target.value) })}
                      min={0.00001}
                      max={10.0}
                      step={0.00001}
                      className="flex-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        const optimal = calculateOptimalL1Alpha(config.latent_dim, config.target_l0 ?? 0.05);
                        updateConfig({ l1_alpha: optimal });
                      }}
                      className="px-3 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-sm rounded-md transition-colors whitespace-nowrap"
                      title={`Calculate optimal L1 alpha for ${config.latent_dim} latent dimensions`}
                    >
                      Auto
                    </button>
                  </div>
                  {/* Sparsity Warnings */}
                  {(() => {
                    const warnings = validateSparsityConfig(config.l1_alpha, config.latent_dim, config.target_l0 ?? 0.05);
                    return warnings.length > 0 ? (
                      <div className="mt-2 space-y-1">
                        {warnings.map((warning, idx) => (
                          <div key={idx} className="flex items-start gap-2 text-xs text-yellow-400">
                            <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                            <span>{warning}</span>
                          </div>
                        ))}
                      </div>
                    ) : null;
                  })()}
                </div>

                {/* Target L0 */}
                <div>
                  <HyperparameterLabel
                    paramName="target_l0"
                    label="Target L0 Sparsity"
                    htmlFor="target-l0"
                    className="mb-2"
                  />
                  <input
                    id="target-l0"
                    type="number"
                    value={config.target_l0 ?? 0.05}
                    onChange={(e) => updateConfig({ target_l0: parseFloat(e.target.value) })}
                    min={0.001}
                    max={1.0}
                    step={0.001}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Top-K Sparsity */}
                <div>
                  <HyperparameterLabel
                    paramName="top_k_sparsity"
                    label="Top-K Sparsity % (Hard)"
                    htmlFor="top-k-sparsity"
                    className="mb-2"
                  />
                  <input
                    id="top-k-sparsity"
                    type="number"
                    value={config.top_k_sparsity ?? ''}
                    onChange={(e) => updateConfig({ top_k_sparsity: e.target.value ? parseFloat(e.target.value) : undefined })}
                    min={0.1}
                    max={100}
                    step={0.1}
                    placeholder="Optional (e.g., 5 for 5%)"
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors placeholder:text-slate-500"
                  />
                  <p className="mt-1 text-xs text-slate-400">
                    Guarantees exact sparsity by keeping only top-K neurons. Leave empty for L1 penalty (soft sparsity).
                  </p>
                </div>

                {/* Normalize Activations */}
                <div>
                  <HyperparameterLabel
                    paramName="normalize_activations"
                    label="Activation Normalization"
                    htmlFor="normalize-activations"
                    className="mb-2"
                  />
                  <select
                    id="normalize-activations"
                    value={config.normalize_activations ?? 'constant_norm_rescale'}
                    onChange={(e) => updateConfig({ normalize_activations: e.target.value })}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  >
                    <option value="constant_norm_rescale">Constant Norm Rescale (SAELens)</option>
                    <option value="none">None</option>
                  </select>
                </div>

                {/* Learning Rate */}
                <div>
                  <HyperparameterLabel
                    paramName="learning_rate"
                    label="Learning Rate"
                    htmlFor="learning-rate"
                    className="mb-2"
                  />
                  <input
                    id="learning-rate"
                    type="number"
                    value={config.learning_rate}
                    onChange={(e) => updateConfig({ learning_rate: parseFloat(e.target.value) })}
                    min={0.00001}
                    max={0.01}
                    step={0.00001}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Batch Size */}
                <div>
                  <HyperparameterLabel
                    paramName="batch_size"
                    label="Batch Size"
                    htmlFor="batch-size"
                    className="mb-2"
                  />
                  <input
                    id="batch-size"
                    type="number"
                    value={config.batch_size}
                    onChange={(e) => updateConfig({ batch_size: parseInt(e.target.value) })}
                    min={1}
                    max={512}
                    step={32}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Total Steps */}
                <div>
                  <HyperparameterLabel
                    paramName="total_steps"
                    label="Total Steps"
                    htmlFor="total-steps"
                    className="mb-2"
                  />
                  <input
                    id="total-steps"
                    type="number"
                    value={config.total_steps}
                    onChange={(e) => updateConfig({ total_steps: parseInt(e.target.value) })}
                    min={1000}
                    max={1000000}
                    step={1000}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Warmup Steps */}
                <div>
                  <HyperparameterLabel
                    paramName="warmup_steps"
                    label="Warmup Steps"
                    htmlFor="warmup-steps"
                    className="mb-2"
                  />
                  <input
                    id="warmup-steps"
                    type="number"
                    value={config.warmup_steps ?? 0}
                    onChange={(e) => updateConfig({ warmup_steps: parseInt(e.target.value) })}
                    min={0}
                    max={100000}
                    step={100}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Weight Decay */}
                <div>
                  <HyperparameterLabel
                    paramName="weight_decay"
                    label="Weight Decay"
                    htmlFor="weight-decay"
                    className="mb-2"
                  />
                  <input
                    id="weight-decay"
                    type="number"
                    value={config.weight_decay ?? 0.01}
                    onChange={(e) => updateConfig({ weight_decay: parseFloat(e.target.value) })}
                    min={0}
                    max={0.1}
                    step={0.001}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Gradient Clipping */}
                <div>
                  <HyperparameterLabel
                    paramName="grad_clip_norm"
                    label="Gradient Clipping"
                    htmlFor="grad-clip-norm"
                    className="mb-2"
                  />
                  <input
                    id="grad-clip-norm"
                    type="number"
                    value={config.grad_clip_norm ?? 1.0}
                    onChange={(e) => updateConfig({ grad_clip_norm: parseFloat(e.target.value) })}
                    min={0}
                    max={10}
                    step={0.1}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Checkpoint Interval */}
                <div>
                  <HyperparameterLabel
                    paramName="checkpoint_interval"
                    label="Checkpoint Interval"
                    htmlFor="checkpoint-interval"
                    className="mb-2"
                  />
                  <input
                    id="checkpoint-interval"
                    type="number"
                    value={config.checkpoint_interval ?? 1000}
                    onChange={(e) =>
                      updateConfig({ checkpoint_interval: parseInt(e.target.value) })
                    }
                    min={100}
                    max={10000}
                    step={100}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Log Interval */}
                <div>
                  <HyperparameterLabel
                    paramName="log_interval"
                    label="Log Interval"
                    htmlFor="log-interval"
                    className="mb-2"
                  />
                  <input
                    id="log-interval"
                    type="number"
                    value={config.log_interval ?? 100}
                    onChange={(e) => updateConfig({ log_interval: parseInt(e.target.value) })}
                    min={10}
                    max={1000}
                    step={10}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Dead Neuron Threshold */}
                <div>
                  <HyperparameterLabel
                    paramName="dead_neuron_threshold"
                    label="Dead Neuron Threshold"
                    htmlFor="dead-neuron-threshold"
                    className="mb-2"
                  />
                  <input
                    id="dead-neuron-threshold"
                    type="number"
                    value={config.dead_neuron_threshold ?? 10000}
                    onChange={(e) =>
                      updateConfig({ dead_neuron_threshold: parseInt(e.target.value) })
                    }
                    min={1000}
                    max={100000}
                    step={1000}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Resample Interval */}
                <div>
                  <HyperparameterLabel
                    paramName="resample_interval"
                    label="Resample Interval"
                    htmlFor="resample-interval"
                    className="mb-2"
                  />
                  <input
                    id="resample-interval"
                    type="number"
                    value={config.resample_interval ?? 5000}
                    onChange={(e) =>
                      updateConfig({ resample_interval: parseInt(e.target.value) })
                    }
                    min={1000}
                    max={50000}
                    step={1000}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Resample Dead Neurons */}
                <div className="col-span-2">
                  <div className="flex items-center gap-2">
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={config.resample_dead_neurons ?? true}
                        onChange={(e) =>
                          updateConfig({ resample_dead_neurons: e.target.checked })
                        }
                        className="w-4 h-4 rounded bg-slate-800 border-slate-700 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                      />
                      <span className="text-sm font-medium text-slate-300">
                        Resample dead neurons during training
                      </span>
                    </label>
                    <HyperparameterTooltip paramName="resample_dead_neurons" />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Start Training Button */}
          <div className="mt-6 pt-4 border-t border-slate-700">
            <button
              onClick={handleStartTraining}
              disabled={!isFormValid || isStarting}
              className={`w-full flex items-center justify-center gap-2 py-3 ${COMPONENTS.button.primary}`}
            >
              {isStarting ? (
                <>
                  <Loader size={20} className="animate-spin" />
                  Starting Training...
                </>
              ) : (
                <>
                  <Play size={20} />
                  Start Training
                </>
              )}
            </button>
          </div>
        </div>

        {/* Training Jobs Section */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-semibold text-slate-100">Training Jobs</h3>
              {trainings.length > 0 && (
                <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer hover:text-slate-300">
                  <input
                    type="checkbox"
                    checked={selectedTrainingIds.size === trainings.length && trainings.length > 0}
                    onChange={handleSelectAll}
                    className="w-4 h-4 rounded bg-slate-800 border-slate-700 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                  />
                  Select All
                </label>
              )}
            </div>
            <div className="flex items-center gap-3">
              {selectedTrainingIds.size > 0 && (
                <button
                  onClick={handleDeleteSelected}
                  disabled={isDeleting}
                  className={`flex items-center gap-2 text-sm ${COMPONENTS.button.danger}`}
                >
                  {isDeleting ? (
                    <Loader size={16} className="animate-spin" />
                  ) : (
                    <Trash2 size={16} />
                  )}
                  Delete Selected ({selectedTrainingIds.size})
                </button>
              )}
              <span className="text-sm text-slate-600 dark:text-slate-400">{statusCounts.all} total</span>
            </div>
          </div>

          {/* Status Filter Tabs */}
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setStatusFilter('all')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                statusFilter === 'all'
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              All ({statusCounts.all})
            </button>
            <button
              onClick={() => setStatusFilter(TrainingStatus.RUNNING)}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                statusFilter === TrainingStatus.RUNNING
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              <Activity size={16} className="animate-pulse" />
              Running ({statusCounts.running})
            </button>
            <button
              onClick={() => setStatusFilter(TrainingStatus.COMPLETED)}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                statusFilter === TrainingStatus.COMPLETED
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              <CheckCircle size={16} />
              Completed ({statusCounts.completed})
            </button>
            <button
              onClick={() => setStatusFilter(TrainingStatus.FAILED)}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                statusFilter === TrainingStatus.FAILED
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              <XCircle size={16} />
              Failed ({statusCounts.failed})
            </button>
          </div>

          {/* Training Jobs List */}
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader size={32} className="animate-spin text-emerald-500" />
            </div>
          ) : trainings.length === 0 ? (
            <div className="text-center py-12">
              <Activity size={48} className="mx-auto text-slate-600 mb-4" />
              <p className="text-slate-400 mb-2">No training jobs yet</p>
              <p className="text-sm text-slate-500">
                Configure a training job above to get started
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {[...trainings]
                .sort((a, b) => {
                  // Running jobs first
                  if (a.status === TrainingStatus.RUNNING && b.status !== TrainingStatus.RUNNING) return -1;
                  if (a.status !== TrainingStatus.RUNNING && b.status === TrainingStatus.RUNNING) return 1;
                  // Then by creation time (newest first)
                  return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
                })
                .map((training) => (
                  <TrainingCard
                    key={training.id}
                    training={training}
                    isSelected={selectedTrainingIds.has(training.id)}
                    onToggleSelect={handleToggleSelection}
                    models={models}
                    datasets={datasets}
                  />
                ))}
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-4 bg-red-900/20 border border-red-900/50 rounded-lg">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
