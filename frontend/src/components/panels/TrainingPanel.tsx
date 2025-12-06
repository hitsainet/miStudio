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
  Save,
  X,
} from 'lucide-react';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { useModelsStore } from '../../stores/modelsStore';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useTrainingTemplatesStore } from '../../stores/trainingTemplatesStore';
import { useTrainingWebSocket } from '../../hooks/useTrainingWebSocket';
import { useDeletionProgressWebSocket } from '../../hooks/useDeletionProgressWebSocket';
import { useWebSocketContext } from '../../contexts/WebSocketContext';
import { TrainingStatus, SAEArchitectureType } from '../../types/training';
import type { TrainingCreateRequest } from '../../types/training';
import { TrainingCard } from '../training/TrainingCard';
import { TemplateSelector } from '../training/TemplateSelector';
import DeletionProgressModal from '../training/DeletionProgressModal';
import { estimateMultilayerTrainingMemory, formatMemorySize } from '../../utils/memoryEstimation';
import { HyperparameterLabel, HyperparameterTooltip } from '../common/HyperparameterTooltip';
import { calculateOptimalL1Alpha, validateSparsityConfig } from '../../utils/hyperparameterOptimization';
import { COMPONENTS } from '../../config/brand';
import type { TrainingTemplate } from '../../types/trainingTemplate';

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
  const { createTemplate } = useTrainingTemplatesStore();

  // WebSocket connection status
  const { isConnected } = useWebSocketContext();

  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [selectedTrainingIds, setSelectedTrainingIds] = useState<Set<string>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);
  const [latentMultiplier, setLatentMultiplier] = useState(8);
  const [availableExtractions, setAvailableExtractions] = useState<any[]>([]);
  const [isLoadingExtractions, setIsLoadingExtractions] = useState(false);
  const [showSaveTemplateModal, setShowSaveTemplateModal] = useState(false);
  const [templateName, setTemplateName] = useState('');
  const [templateDescription, setTemplateDescription] = useState('');
  const [isSavingTemplate, setIsSavingTemplate] = useState(false);
  const [saveTemplateError, setSaveTemplateError] = useState<string | null>(null);

  // Deletion progress state
  const [deletingTrainingId, setDeletingTrainingId] = useState<string | null>(null);
  const [deletionTasks, setDeletionTasks] = useState<Array<{
    id: string;
    label: string;
    status: 'pending' | 'in_progress' | 'completed';
    message?: string;
    count?: number;
  }>>([
    { id: 'database', label: 'Database Records', status: 'pending' },
    { id: 'files', label: 'Training Files', status: 'pending' },
  ]);

  // Memory estimation
  const memoryEstimate = useMemo(() => {
    return estimateMultilayerTrainingMemory(
      config.hidden_dim,
      config.latent_dim,
      config.batch_size,
      config.training_layers?.length || 1
    );
  }, [config.hidden_dim, config.latent_dim, config.batch_size, config.training_layers]);

  // Generate default template name and description based on current config
  const generateTemplateDefaults = useMemo(() => {
    const model = models.find(m => m.id === config.model_id);
    const dataset = datasets.find(d => d.id === config.dataset_id);

    // Extract short model name (e.g., "microsoft/Phi-4-mini-instruct" -> "Phi-4-mini")
    const modelName = model?.name || 'Model';
    const modelShort = modelName
      .replace(/^.*\//, '') // Remove org prefix
      .split('-')
      .slice(0, 3)
      .join('-')
      .replace(/-instruct$/, '') // Remove common suffixes
      .replace(/-chat$/, '');

    // Extract short dataset name
    const datasetName = dataset?.name || 'Dataset';
    const datasetShort = datasetName
      .replace(/^.*\//, '') // Remove org prefix
      .replace(/_/g, '-')
      .slice(0, 20); // Truncate long names

    // Architecture short name
    const archMap: Record<string, string> = {
      'standard': 'Std',
      'skip': 'Skip',
      'transcoder': 'Trans',
      'jumprelu': 'JumpReLU',
    };
    const archShort = archMap[config.architecture_type] || config.architecture_type;

    // Layers formatting
    const layers = config.training_layers || [0];
    const layersStr = layers.length === 1
      ? `L${layers[0]}`
      : layers.length <= 3
        ? `L${layers.join('-')}`
        : `L${layers[0]}-${layers[layers.length - 1]}`;

    const name = `${modelShort}_${datasetShort}_${archShort}_${layersStr}`;

    // Detailed description
    const multiplier = Math.round(config.latent_dim / config.hidden_dim);
    const stepsK = config.total_steps >= 1000
      ? `${Math.round(config.total_steps / 1000)}k`
      : String(config.total_steps);
    const layerList = layers.join(', ');

    const descParts = [
      `Hidden: ${config.hidden_dim} → Latent: ${config.latent_dim} (${multiplier}x)`,
      `L1: ${config.l1_alpha}`,
      `LR: ${config.learning_rate}`,
      `Batch: ${config.batch_size}`,
      `Steps: ${stepsK}`,
      `Layers: ${layerList}`,
    ];

    // Add JumpReLU-specific parameters if applicable
    if (config.architecture_type === SAEArchitectureType.JUMPRELU) {
      const sparsityCoeff = (config as any).sparsity_coeff ?? 0.0006;
      const initialThreshold = (config as any).initial_threshold ?? 0.001;
      const bandwidth = (config as any).bandwidth ?? 0.001;
      descParts.push(`SparsityCoeff: ${sparsityCoeff}`);
      descParts.push(`Thresh: ${initialThreshold}`);
      descParts.push(`BW: ${bandwidth}`);
    }

    // Add target L0 if set
    if (config.target_l0) {
      descParts.push(`L0: ${config.target_l0}`);
    }

    // Add top-k if set
    if ((config as any).top_k_sparsity) {
      descParts.push(`TopK: ${(config as any).top_k_sparsity}%`);
    }

    const description = descParts.join(' | ');

    return { name, description };
  }, [config, models, datasets]);

  // Fetch data on mount
  useEffect(() => {
    fetchModels();
    fetchDatasets();
    fetchTrainings();
  }, [fetchModels, fetchDatasets, fetchTrainings]);

  // Fetch available extractions when model is selected
  useEffect(() => {
    const fetchExtractions = async () => {
      if (!config.model_id) {
        setAvailableExtractions([]);
        return;
      }

      setIsLoadingExtractions(true);
      try {
        const response = await fetch(`/api/v1/models/${config.model_id}/extractions`);
        if (response.ok) {
          const data = await response.json();
          // Filter to only completed extractions
          const completedExtractions = (data.extractions || []).filter(
            (ext: any) => ext.status === 'completed'
          );
          setAvailableExtractions(completedExtractions);
        } else {
          setAvailableExtractions([]);
        }
      } catch (error) {
        console.error('Failed to fetch extractions:', error);
        setAvailableExtractions([]);
      } finally {
        setIsLoadingExtractions(false);
      }
    };

    fetchExtractions();
  }, [config.model_id]);

  // Subscribe to WebSocket updates for all trainings
  useTrainingWebSocket(trainings.map((t) => t.id));

  // Handle deletion progress updates via WebSocket
  const handleDeletionTaskUpdate = React.useCallback((update: {
    training_id: string;
    task: string;
    status: 'in_progress' | 'completed';
    message?: string;
    count?: number;
  }) => {
    console.log('[TrainingPanel] 📥 Received deletion task update:', update);
    setDeletionTasks((prevTasks) => {
      const updated = prevTasks.map((task) => {
        const matches = task.id === update.task;
        console.log(`[TrainingPanel] Task "${task.id}" matches "${update.task}":`, matches);
        return matches
          ? {
              ...task,
              status: update.status,
              message: update.message,
              count: update.count,
            }
          : task;
      });
      console.log('[TrainingPanel] Updated deletion tasks:', updated);
      return updated;
    });
  }, []);

  // Subscribe to deletion progress WebSocket for the training being deleted
  useDeletionProgressWebSocket(deletingTrainingId, handleDeletionTaskUpdate);

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

  // Autodiscover hidden dimension and training layers from extraction metadata
  useEffect(() => {
    const autodiscoverFromExtraction = async () => {
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

        if (extraction) {
          const updates: any = {};

          // Autodiscover training_layers from extraction layer_indices
          if (extraction.layer_indices && Array.isArray(extraction.layer_indices)) {
            const extractionLayers = extraction.layer_indices.sort((a: number, b: number) => a - b);
            const currentLayers = config.training_layers || [];

            // Check if current layers differ from extraction layers
            const layersDiffer = extractionLayers.length !== currentLayers.length ||
              extractionLayers.some((layer: number, i: number) => layer !== currentLayers[i]);

            if (layersDiffer) {
              console.log(`[TrainingPanel] Autodiscovered training_layers=${JSON.stringify(extractionLayers)} from extraction ${config.extraction_id}`);
              updates.training_layers = extractionLayers;
            }
          }

          // Autodiscover hidden_dim from extraction statistics
          if (extraction.statistics) {
            const layerNames = Object.keys(extraction.statistics);
            if (layerNames.length > 0) {
              const firstLayerStats = extraction.statistics[layerNames[0]];
              if (firstLayerStats?.shape && Array.isArray(firstLayerStats.shape) && firstLayerStats.shape.length === 3) {
                // Shape format: [n_samples, seq_len, hidden_dim]
                const hiddenDim = firstLayerStats.shape[2];

                // Only update if different from current value
                if (hiddenDim !== config.hidden_dim) {
                  console.log(`[TrainingPanel] Autodiscovered hidden_dim=${hiddenDim} from extraction ${config.extraction_id}`);
                  updates.hidden_dim = hiddenDim;
                }
              }
            }
          }

          // Apply all updates at once if any
          if (Object.keys(updates).length > 0) {
            updateConfig(updates);
          }
        }
      } catch (error) {
        console.error('Failed to autodiscover from extraction:', error);
      }
    };

    autodiscoverFromExtraction();
  }, [config.extraction_id, config.model_id]);

  // Update latent_dim when hidden_dim or latent multiplier changes
  useEffect(() => {
    const calculatedLatentDim = config.hidden_dim * latentMultiplier;
    if (calculatedLatentDim !== config.latent_dim) {
      updateConfig({ latent_dim: calculatedLatentDim });
    }
  }, [config.hidden_dim, latentMultiplier]);

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

    // Track the first training for progress modal
    const firstTrainingId = Array.from(selectedTrainingIds)[0];

    // Reset deletion tasks to pending
    setDeletionTasks((tasks) => tasks.map(task => ({ ...task, status: 'pending', message: undefined, count: undefined })));

    // Show modal for first training
    setDeletingTrainingId(firstTrainingId);

    setIsDeleting(true);
    try {
      // Delete all selected trainings in parallel
      await Promise.all(
        Array.from(selectedTrainingIds).map((id) => deleteTraining(id))
      );

      // Mark database deletion as completed (happens synchronously in the API)
      // Mark files task as in_progress (Celery task is now queued and will run shortly)
      setDeletionTasks((tasks) =>
        tasks.map((task) => {
          if (task.id === 'database') {
            return { ...task, status: 'completed', message: 'Database records deleted' };
          } else if (task.id === 'files') {
            return { ...task, status: 'in_progress', message: 'Deleting training files...' };
          }
          return task;
        })
      );

      // Clear selection after successful deletion
      setSelectedTrainingIds(new Set());
    } catch (error) {
      console.error('Failed to delete selected trainings:', error);
    } finally {
      setIsDeleting(false);
    }
  };

  // Handle template load
  const handleTemplateLoad = (template: TrainingTemplate) => {
    console.log('[TrainingPanel] Loading template:', template);

    // Map template data to training config
    const updates: any = {
      ...template.hyperparameters,
      architecture_type: template.encoder_type as SAEArchitectureType,
    };

    // Update latent multiplier if both dimensions are available
    if (template.hyperparameters.hidden_dim && template.hyperparameters.latent_dim) {
      const multiplier = Math.round(template.hyperparameters.latent_dim / template.hyperparameters.hidden_dim);
      setLatentMultiplier(multiplier);
    }

    updateConfig(updates);
    console.log('[TrainingPanel] Template loaded successfully');
  };

  // Handle save template
  const handleSaveTemplate = async () => {
    if (!templateName.trim()) {
      setSaveTemplateError('Please enter a template name');
      return;
    }

    if (!config.model_id || !config.dataset_id) {
      setSaveTemplateError('Please select a model and dataset first');
      return;
    }

    setIsSavingTemplate(true);
    setSaveTemplateError(null);

    try {
      // Create template from current config
      await createTemplate({
        name: templateName.trim(),
        description: templateDescription.trim() || undefined,
        model_id: config.model_id,
        dataset_id: config.dataset_id,
        encoder_type: config.architecture_type as any,
        hyperparameters: {
          hidden_dim: config.hidden_dim,
          latent_dim: config.latent_dim,
          architecture_type: config.architecture_type,
          training_layers: config.training_layers,
          l1_alpha: config.l1_alpha,
          learning_rate: config.learning_rate,
          batch_size: config.batch_size,
          total_steps: config.total_steps,
          warmup_steps: config.warmup_steps,
          target_l0: config.target_l0,
          weight_decay: config.weight_decay,
          grad_clip_norm: config.grad_clip_norm,
          checkpoint_interval: config.checkpoint_interval,
          // JumpReLU-specific parameters
          ...(config.architecture_type === SAEArchitectureType.JUMPRELU && {
            initial_threshold: (config as any).initial_threshold,
            bandwidth: (config as any).bandwidth,
            sparsity_coeff: (config as any).sparsity_coeff,
            normalize_decoder: (config as any).normalize_decoder,
          }),
        },
        is_favorite: false,
      });

      // Success - close modal and reset form
      setShowSaveTemplateModal(false);
      setTemplateName('');
      setTemplateDescription('');
      alert('Template saved successfully!');
    } catch (error) {
      console.error('Failed to save template:', error);
      setSaveTemplateError(error instanceof Error ? error.message : 'Failed to save template');
    } finally {
      setIsSavingTemplate(false);
    }
  };

  // Handle start training
  const handleStartTraining = async () => {
    if (!isFormValid) {
      console.error('[TrainingPanel] Form validation failed:', {
        model_id: config.model_id,
        dataset_id: config.dataset_id,
        hidden_dim: config.hidden_dim,
        latent_dim: config.latent_dim,
        isFormValid,
      });
      alert('Please fill in all required fields (Model, Dataset, Hidden Dim, Latent Dim)');
      return;
    }

    console.log('[TrainingPanel] Starting training with config:', config);
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
          top_k_sparsity: (config as any).top_k_sparsity,
          // JumpReLU-specific parameters (only sent when JumpReLU is selected)
          ...(config.architecture_type === SAEArchitectureType.JUMPRELU && {
            initial_threshold: (config as any).initial_threshold,
            bandwidth: (config as any).bandwidth,
            sparsity_coeff: (config as any).sparsity_coeff,
            normalize_decoder: (config as any).normalize_decoder,
          }),
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

      console.log('[TrainingPanel] Sending training request:', request);
      await createTraining(request);
      console.log('[TrainingPanel] Training created successfully');
      // Don't reset config - keep selections so user can easily start another training
      // Only collapse advanced configuration section
      setShowAdvanced(false);
    } catch (err) {
      console.error('[TrainingPanel] Failed to start training:', err);
      alert(`Failed to start training: ${err instanceof Error ? err.message : String(err)}`);
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
                <option value={SAEArchitectureType.JUMPRELU}>JumpReLU (Gemma Scope)</option>
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

          {/* Optional: Use Cached Activations & Template Selector */}
          <div className="mt-4 grid grid-cols-2 gap-6">
            {/* Use Cached Activations */}
            <div>
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
                    {config.extraction_id && availableExtractions.length > 0 && (() => {
                      const selectedExtraction = availableExtractions.find(ext => ext.extraction_id === config.extraction_id);
                      if (selectedExtraction?.layer_indices && selectedExtraction.layer_indices.length > 0) {
                        const layerLabels = selectedExtraction.layer_indices.map((idx: number) => `L${idx}`).join(', ');
                        return <span className="text-emerald-400"> Cached layers: {layerLabels}</span>;
                      }
                      return null;
                    })()}
                  </p>
                  {(config.extraction_id !== undefined) && (
                    <>
                      {availableExtractions.length > 0 ? (
                        <select
                          id="extraction-id"
                          value={config.extraction_id || ''}
                          onChange={(e) => updateConfig({ extraction_id: e.target.value || undefined })}
                          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 text-sm focus:outline-none focus:border-emerald-500 transition-colors"
                        >
                          <option value="">Select an extraction...</option>
                          {availableExtractions.map((extraction) => {
                            const layerCount = extraction.layer_indices?.length || 0;
                            const sampleCount = extraction.num_samples_processed || extraction.samples_processed || 0;
                            const createdDateTime = extraction.created_at
                              ? new Date(extraction.created_at).toLocaleString()
                              : 'Unknown date/time';

                            // Look up model name from models store
                            const extractionModel = models.find(m => m.id === extraction.model_id);
                            const modelName = extractionModel?.name || extraction.model_id || 'Unknown model';

                            return (
                              <option key={extraction.extraction_id} value={extraction.extraction_id}>
                                {extraction.extraction_id} | {modelName} | {layerCount} layer{layerCount !== 1 ? 's' : ''}, {sampleCount.toLocaleString()} samples | {createdDateTime}
                              </option>
                            );
                          })}
                        </select>
                      ) : isLoadingExtractions ? (
                        <div className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 text-sm flex items-center gap-2">
                          <Loader size={14} className="animate-spin" />
                          <span>Loading available extractions...</span>
                        </div>
                      ) : config.model_id ? (
                        <div className="text-sm text-slate-400 italic">
                          No completed extractions available for this model. Please complete an extraction first in the Extractions panel.
                        </div>
                      ) : (
                        <div className="text-sm text-slate-400 italic">
                          Select a model first to see available extractions.
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Template Selector - Only show when model and dataset are selected */}
            {config.model_id && config.dataset_id ? (
              <TemplateSelector
                modelId={config.model_id}
                datasetId={config.dataset_id}
                onTemplateLoad={handleTemplateLoad}
              />
            ) : (
              <div></div>
            )}
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

                {/* Latent Dimension Multiplier */}
                <div>
                  <HyperparameterLabel
                    paramName="latent_dim"
                    label="Latent Dimension Multiplier"
                    htmlFor="latent-dim"
                    className="mb-2"
                  />
                  <div className="flex items-center gap-3">
                    <input
                      id="latent-dim"
                      type="number"
                      value={latentMultiplier}
                      onChange={(e) => setLatentMultiplier(parseInt(e.target.value) || 1)}
                      min={1}
                      max={32}
                      step={1}
                      className="w-24 px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                    />
                    <span className="text-slate-400 text-sm font-mono">
                      × {config.hidden_dim} = <span className="text-emerald-400">{config.latent_dim}</span>
                    </span>
                  </div>
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
                    value={(config as any).top_k_sparsity ?? ''}
                    onChange={(e) => updateConfig({ top_k_sparsity: e.target.value ? parseFloat(e.target.value) : undefined } as any)}
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

                {/* JumpReLU-specific parameters - only shown when JumpReLU is selected */}
                {config.architecture_type === SAEArchitectureType.JUMPRELU && (
                  <>
                    {/* JumpReLU Section Header */}
                    <div className="col-span-2 mt-4 mb-2">
                      <div className="flex items-center gap-2 pb-2 border-b border-slate-700">
                        <span className="text-sm font-semibold text-emerald-400">JumpReLU Parameters</span>
                        <span className="text-xs text-slate-500">(Gemma Scope Architecture)</span>
                      </div>
                    </div>

                    {/* L0 Sparsity Coefficient */}
                    <div>
                      <HyperparameterLabel
                        paramName="sparsity_coeff"
                        label="L0 Sparsity Coefficient"
                        htmlFor="sparsity-coeff"
                        className="mb-2"
                      />
                      <input
                        id="sparsity-coeff"
                        type="number"
                        value={(config as any).sparsity_coeff ?? 0.0006}
                        onChange={(e) => updateConfig({ sparsity_coeff: parseFloat(e.target.value) } as any)}
                        min={0.00001}
                        max={0.1}
                        step={0.0001}
                        className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                      />
                      <p className="mt-1 text-xs text-slate-400">
                        L0 penalty coefficient (default: 6e-4 per Gemma Scope). Replaces L1 for JumpReLU.
                      </p>
                    </div>

                    {/* Initial Threshold */}
                    <div>
                      <HyperparameterLabel
                        paramName="initial_threshold"
                        label="Initial Threshold"
                        htmlFor="initial-threshold"
                        className="mb-2"
                      />
                      <input
                        id="initial-threshold"
                        type="number"
                        value={(config as any).initial_threshold ?? 0.001}
                        onChange={(e) => updateConfig({ initial_threshold: parseFloat(e.target.value) } as any)}
                        min={0.00001}
                        max={1.0}
                        step={0.0001}
                        className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                      />
                      <p className="mt-1 text-xs text-slate-400">
                        Starting threshold for JumpReLU activation. Each feature learns its own optimal threshold.
                      </p>
                    </div>

                    {/* KDE Bandwidth */}
                    <div>
                      <HyperparameterLabel
                        paramName="bandwidth"
                        label="KDE Bandwidth (ε)"
                        htmlFor="bandwidth"
                        className="mb-2"
                      />
                      <input
                        id="bandwidth"
                        type="number"
                        value={(config as any).bandwidth ?? 0.001}
                        onChange={(e) => updateConfig({ bandwidth: parseFloat(e.target.value) } as any)}
                        min={0.00001}
                        max={0.1}
                        step={0.0001}
                        className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                      />
                      <p className="mt-1 text-xs text-slate-400">
                        Smoothness of STE gradient estimation. Default: 0.001.
                      </p>
                    </div>

                    {/* Normalize Decoder */}
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={(config as any).normalize_decoder ?? true}
                            onChange={(e) => updateConfig({ normalize_decoder: e.target.checked } as any)}
                            className="w-4 h-4 rounded bg-slate-800 border-slate-700 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                          />
                          <span className="text-sm font-medium text-slate-300">
                            Normalize Decoder Columns
                          </span>
                        </label>
                        <HyperparameterTooltip paramName="normalize_decoder" />
                      </div>
                      <p className="text-xs text-slate-400">
                        Required for JumpReLU. Normalizes decoder columns to unit norm after each step.
                      </p>
                    </div>
                  </>
                )}

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

          {/* Action Buttons */}
          <div className="mt-6 pt-4 border-t border-slate-700 flex gap-3">
            <button
              onClick={() => {
                // Pre-populate template name and description from current config
                const defaults = generateTemplateDefaults;
                setTemplateName(defaults.name);
                setTemplateDescription(defaults.description);
                setShowSaveTemplateModal(true);
              }}
              disabled={!config.model_id || !config.dataset_id}
              className="flex items-center justify-center gap-2 py-3 px-4 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed text-slate-100 disabled:text-slate-500 rounded-md transition-colors"
              title="Save current configuration as a template"
            >
              <Save size={20} />
              Save as Template
            </button>
            <button
              onClick={handleStartTraining}
              disabled={!isFormValid || isStarting}
              className={`flex-1 flex items-center justify-center gap-2 py-3 ${COMPONENTS.button.primary}`}
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

      {/* Deletion Progress Modal */}
      <DeletionProgressModal
        isOpen={!!deletingTrainingId}
        onClose={() => setDeletingTrainingId(null)}
        trainingId={deletingTrainingId || ''}
        tasks={deletionTasks}
      />

      {/* Save Template Modal */}
      {showSaveTemplateModal && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
          onClick={() => {
            setShowSaveTemplateModal(false);
            setSaveTemplateError(null);
            setTemplateName('');
            setTemplateDescription('');
          }}
        >
          <div
            className="bg-slate-900 rounded-lg max-w-lg w-full p-6 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-slate-100">Save as Template</h2>
              <button
                onClick={() => {
                  setShowSaveTemplateModal(false);
                  setSaveTemplateError(null);
                  setTemplateName('');
                  setTemplateDescription('');
                }}
                className="p-1 hover:bg-slate-800 rounded transition-colors"
                title="Close modal"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>

            <p className="text-sm text-slate-400 mb-4">
              Save your current training configuration as a reusable template.
            </p>

            {/* Error message */}
            {saveTemplateError && (
              <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-md">
                <p className="text-sm text-red-400">{saveTemplateError}</p>
              </div>
            )}

            {/* Form */}
            <div className="space-y-4">
              {/* Template Name */}
              <div>
                <label htmlFor="template-name" className="block text-sm font-medium text-slate-300 mb-1">
                  Template Name <span className="text-red-400">*</span>
                </label>
                <input
                  id="template-name"
                  type="text"
                  value={templateName}
                  onChange={(e) => setTemplateName(e.target.value)}
                  placeholder="e.g., TinyLlama_OpenWebText_Standard_L1-0.0001"
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 text-sm focus:outline-none focus:border-emerald-500 transition-colors"
                  disabled={isSavingTemplate}
                />
              </div>

              {/* Template Description */}
              <div>
                <label htmlFor="template-description" className="block text-sm font-medium text-slate-300 mb-1">
                  Description (Optional)
                </label>
                <textarea
                  id="template-description"
                  value={templateDescription}
                  onChange={(e) => setTemplateDescription(e.target.value)}
                  placeholder="e.g., L1: 0.0001 | LR: 0.00027 | Dict: 2048→8192 (4x) | Steps: 50k"
                  rows={3}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 text-sm focus:outline-none focus:border-emerald-500 transition-colors resize-none"
                  disabled={isSavingTemplate}
                />
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowSaveTemplateModal(false);
                    setSaveTemplateError(null);
                    setTemplateName('');
                    setTemplateDescription('');
                  }}
                  disabled={isSavingTemplate}
                  className="flex-1 px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-300 disabled:text-slate-600 rounded-md transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveTemplate}
                  disabled={!templateName.trim() || isSavingTemplate}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:cursor-not-allowed text-white disabled:text-slate-600 rounded-md transition-colors"
                >
                  {isSavingTemplate ? (
                    <>
                      <Loader size={16} className="animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save size={16} />
                      Save Template
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
