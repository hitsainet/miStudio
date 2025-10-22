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

  // Validation
  const isFormValid = config.model_id && config.dataset_id && config.training_layers && config.training_layers.length > 0;

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
        extraction_id: config.extraction_id,
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

  // Status filter stats
  const statusCounts = {
    all: trainings.length,
    running: trainings.filter((t) => t.status === TrainingStatus.RUNNING).length,
    completed: trainings.filter((t) => t.status === TrainingStatus.COMPLETED).length,
    failed: trainings.filter((t) => t.status === TrainingStatus.FAILED).length,
  };

  return (
    <div className="min-h-screen bg-slate-950">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold text-slate-100 mb-2">SAE Training</h1>
              <p className="text-slate-400">
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
                    className="px-3 py-1 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition-colors"
                  >
                    Select All
                  </button>
                  <button
                    type="button"
                    onClick={() => updateConfig({ training_layers: [] })}
                    className="px-3 py-1 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition-colors"
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
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Hidden Dimension
                  </label>
                  <input
                    type="number"
                    value={config.hidden_dim}
                    onChange={(e) => updateConfig({ hidden_dim: parseInt(e.target.value) })}
                    min={64}
                    max={8192}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Input/output size (e.g., 768 for GPT-2)
                  </p>
                </div>

                {/* Latent Dimension */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Latent Dimension
                  </label>
                  <input
                    type="number"
                    value={config.latent_dim}
                    onChange={(e) => updateConfig({ latent_dim: parseInt(e.target.value) })}
                    min={512}
                    max={65536}
                    step={512}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    SAE width (typically 8-16x hidden_dim)
                  </p>
                </div>

                {/* L1 Alpha */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    L1 Sparsity Coefficient
                  </label>
                  <input
                    type="number"
                    value={config.l1_alpha}
                    onChange={(e) => updateConfig({ l1_alpha: parseFloat(e.target.value) })}
                    min={0.00001}
                    max={0.1}
                    step={0.00001}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">L1 penalty weight</p>
                </div>

                {/* Target L0 */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Target L0 Sparsity
                  </label>
                  <input
                    type="number"
                    value={config.target_l0 ?? 0.05}
                    onChange={(e) => updateConfig({ target_l0: parseFloat(e.target.value) })}
                    min={0.001}
                    max={1.0}
                    step={0.001}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Target activation rate (0-1)
                  </p>
                </div>

                {/* Learning Rate */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    value={config.learning_rate}
                    onChange={(e) => updateConfig({ learning_rate: parseFloat(e.target.value) })}
                    min={0.00001}
                    max={0.01}
                    step={0.00001}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">Initial learning rate</p>
                </div>

                {/* Batch Size */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    value={config.batch_size}
                    onChange={(e) => updateConfig({ batch_size: parseInt(e.target.value) })}
                    min={1}
                    max={512}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">Training batch size</p>
                </div>

                {/* Total Steps */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Total Steps
                  </label>
                  <input
                    type="number"
                    value={config.total_steps}
                    onChange={(e) => updateConfig({ total_steps: parseInt(e.target.value) })}
                    min={1000}
                    max={1000000}
                    step={1000}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">Total training steps</p>
                </div>

                {/* Warmup Steps */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Warmup Steps
                  </label>
                  <input
                    type="number"
                    value={config.warmup_steps ?? 0}
                    onChange={(e) => updateConfig({ warmup_steps: parseInt(e.target.value) })}
                    min={0}
                    max={100000}
                    step={100}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">Linear warmup steps</p>
                </div>

                {/* Weight Decay */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Weight Decay
                  </label>
                  <input
                    type="number"
                    value={config.weight_decay ?? 0.01}
                    onChange={(e) => updateConfig({ weight_decay: parseFloat(e.target.value) })}
                    min={0}
                    max={0.1}
                    step={0.001}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">L2 regularization</p>
                </div>

                {/* Gradient Clipping */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Gradient Clipping
                  </label>
                  <input
                    type="number"
                    value={config.grad_clip_norm ?? 1.0}
                    onChange={(e) => updateConfig({ grad_clip_norm: parseFloat(e.target.value) })}
                    min={0}
                    max={10}
                    step={0.1}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">Max gradient norm</p>
                </div>

                {/* Checkpoint Interval */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Checkpoint Interval
                  </label>
                  <input
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
                  <p className="text-xs text-slate-500 mt-1">Save checkpoint every N steps</p>
                </div>

                {/* Log Interval */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Log Interval
                  </label>
                  <input
                    type="number"
                    value={config.log_interval ?? 100}
                    onChange={(e) => updateConfig({ log_interval: parseInt(e.target.value) })}
                    min={10}
                    max={1000}
                    step={10}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <p className="text-xs text-slate-500 mt-1">Log metrics every N steps</p>
                </div>

                {/* Dead Neuron Threshold */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Dead Neuron Threshold
                  </label>
                  <input
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
                  <p className="text-xs text-slate-500 mt-1">
                    Steps before neuron considered dead
                  </p>
                </div>

                {/* Resample Dead Neurons */}
                <div className="col-span-2">
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
                </div>
              </div>
            </div>
          )}

          {/* Start Training Button */}
          <div className="mt-6 pt-4 border-t border-slate-700">
            <button
              onClick={handleStartTraining}
              disabled={!isFormValid || isStarting}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium rounded-md transition-colors"
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
                  className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-medium rounded-md transition-colors"
                >
                  {isDeleting ? (
                    <Loader size={16} className="animate-spin" />
                  ) : (
                    <Trash2 size={16} />
                  )}
                  Delete Selected ({selectedTrainingIds.size})
                </button>
              )}
              <span className="text-sm text-slate-400">{statusCounts.all} total</span>
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
