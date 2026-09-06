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

import React, { useEffect, useState, useMemo, useCallback } from 'react';
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
  AlertCircle,
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
import { getFrameworkConfig, getFrameworkOptions, isFieldVisible } from '../../config/frameworkConfigs';
import { TrainingCard } from '../training/TrainingCard';
import { TemplateSelector } from '../training/TemplateSelector';
import DeletionProgressModal from '../training/DeletionProgressModal';
import { estimateMultilayerTrainingMemory, formatMemorySize } from '../../utils/memoryEstimation';
import { HyperparameterLabel, HyperparameterTooltip } from '../common/HyperparameterTooltip';
import { calculateOptimalL1Alpha, validateSparsityConfig } from '../../utils/hyperparameterOptimization';
import { COMPONENTS } from '../../config/brand';
import { formatLayerIndices } from '../../utils/formatters';
import type { TrainingTemplate } from '../../types/trainingTemplate';
import { fireAndForget } from '../../utils/fireAndForget';

export const TrainingPanel: React.FC = () => {
  // Store state
  const {
    trainings,
    config,
    updateConfig,
    fetchTrainings,
    fetchTraining,
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
  const [extractionFetchError, setExtractionFetchError] = useState<string | null>(null);
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

  // Memory estimation - account for both layers and hook types
  const numSAEs = (config.training_layers?.length || 1) * (config.hook_types?.length || 1);
  const memoryEstimate = useMemo(() => {
    return estimateMultilayerTrainingMemory(
      config.hidden_dim,
      config.latent_dim,
      config.batch_size,
      numSAEs
    );
  }, [config.hidden_dim, config.latent_dim, config.batch_size, numSAEs]);

  // Generate default template name and description based on current config
  const generateTemplateDefaults = useMemo(() => {
    const model = models.find(m => m.id === config.model_id);
    // Use first selected dataset for template naming
    const firstDatasetId = config.dataset_ids?.[0];
    const dataset = firstDatasetId ? datasets.find(d => d.id === firstDatasetId) : undefined;

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
      'standard_saelens': 'SAELens',
      'standard_anthropic': 'Anthropic',
      'standard': 'Std',
      'skip': 'Skip',
      'transcoder': 'Trans',
      'jumprelu': 'JumpReLU',
      'topk': 'TopK',
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

    const fw = getFrameworkConfig(config.architecture_type);
    const descParts = [
      `Hidden: ${config.hidden_dim} → Latent: ${config.latent_dim} (${multiplier}x)`,
      `LR: ${config.learning_rate}`,
      `Batch: ${config.batch_size}`,
      `Steps: ${stepsK}`,
      `Layers: ${layerList}`,
    ];

    // Add framework-specific sparsity params
    if (fw.sparsityType === 'l1' && config.l1_alpha != null) {
      descParts.push(`L1: ${config.l1_alpha}`);
    } else if (fw.sparsityType === 'l0') {
      descParts.push(`SparsityCoeff: ${config.sparsity_coeff ?? 1e-3}`);
      descParts.push(`Thresh: ${config.initial_threshold ?? 0.5}`);
      descParts.push(`BW: ${config.bandwidth ?? 0.01}`);
    } else if (fw.sparsityType === 'topk') {
      descParts.push(`K: ${config.top_k ?? 64}`);
      descParts.push(`AuxAlpha: ${config.aux_loss_alpha ?? 1/32}`);
    }

    // Add target L0 if set
    if (config.target_l0 && fw.sparsityType !== 'topk') {
      descParts.push(`L0: ${config.target_l0}`);
    }

    const description = descParts.join(' | ');

    return { name, description };
  }, [config, models, datasets]);

  // Fetch data on mount
  useEffect(() => {
    fetchModels();
    fetchDatasets();
    fireAndForget(fetchTrainings());
  }, [fetchModels, fetchDatasets, fetchTrainings]);

  // Load the extraction list, and KEEP IT CURRENT.
  //
  // This used to run only when `config.model_id` changed, so an extraction that
  // finished while this panel was open never appeared. Reported 2026-08-27:
  // "There were never two extractions for the same model (different layers) in
  // the dropdown. I couldn't select the correct one." The API had both all
  // along — the panel was showing a list fetched before the second one existed,
  // and the only cures were navigating away and back, or reloading.
  const refreshExtractions = useCallback(async () => {
    const fetchExtractions = async () => {
      if (!config.model_id) {
        setAvailableExtractions([]);
        setExtractionFetchError(null);
        return;
      }

      setIsLoadingExtractions(true);
      setExtractionFetchError(null);
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
          // Surface non-OK response as a visible error rather than silently
          // clearing the dropdown — users need to know why the list is empty.
          const statusText = `${response.status} ${response.statusText}`;
          setAvailableExtractions([]);
          setExtractionFetchError(`Failed to load extractions (${statusText})`);
          console.error('Failed to fetch extractions:', statusText);
        }
      } catch (error) {
        setAvailableExtractions([]);
        setExtractionFetchError('Failed to load extractions — check network connection');
        console.error('Failed to fetch extractions:', error);
      } finally {
        setIsLoadingExtractions(false);
      }
    };

    await fetchExtractions();
  }, [config.model_id]);

  useEffect(() => {
    refreshExtractions();
  }, [refreshExtractions]);

  // An extraction finishing is the event that changes this list, and it happens
  // in another panel. Re-read when this tab regains focus, and when the selected
  // model's extraction settles.
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === 'visible') refreshExtractions();
    };
    document.addEventListener('visibilitychange', onVisible);
    window.addEventListener('focus', onVisible);
    return () => {
      document.removeEventListener('visibilitychange', onVisible);
      window.removeEventListener('focus', onVisible);
    };
  }, [refreshExtractions]);

  const selectedModelExtractionStatus = models.find(
    (m) => m.id === config.model_id,
  )?.extraction_status;

  const seenExtractionStatus = React.useRef<string | undefined | null>(null);
  useEffect(() => {
    // Fires when an in-flight extraction reaches a settled state (or clears).
    // Skip the first run: the effect above already fetched on mount, and firing
    // both would double every page load's requests.
    if (seenExtractionStatus.current === null) {
      seenExtractionStatus.current = selectedModelExtractionStatus;
      return;
    }
    if (seenExtractionStatus.current === selectedModelExtractionStatus) return;
    seenExtractionStatus.current = selectedModelExtractionStatus;
    refreshExtractions();
  }, [selectedModelExtractionStatus, refreshExtractions]);

  // Group available extractions by dataset_id for per-dataset pickers
  const extractionsPerDataset = useMemo(() => {
    const map: Record<string, any[]> = {};
    for (const ext of availableExtractions) {
      if (ext.dataset_id) {
        if (!map[ext.dataset_id]) map[ext.dataset_id] = [];
        map[ext.dataset_id].push(ext);
      }
    }
    return map;
  }, [availableExtractions]);

  // Check which datasets are missing extractions (when cached activations enabled)
  const datasetsMissingExtractions = useMemo(() => {
    if (!config.extraction_ids || !config.dataset_ids) return [];
    return config.dataset_ids.filter(dsId => {
      const dsExtractions = extractionsPerDataset[dsId] || [];
      return dsExtractions.length === 0;
    });
  }, [config.extraction_ids, config.dataset_ids, extractionsPerDataset]);

  // Clear stale extraction_ids when datasets change
  useEffect(() => {
    if (config.extraction_ids && config.extraction_ids.length > 0) {
      const stillValid = config.extraction_ids.filter(eid =>
        availableExtractions.some((ext: any) => ext.extraction_id === eid)
      );
      if (stillValid.length !== config.extraction_ids.length) {
        updateConfig({ extraction_ids: stillValid.length > 0 ? stillValid : [] });
      }
    }
  }, [availableExtractions, config.extraction_ids, updateConfig]);

  // Subscribe to WebSocket updates for all trainings
  useTrainingWebSocket(trainings.map((t) => t.id));

  // Stable content-based key: only changes when a training's id or status changes.
  // Memoised so the inline .map().join() doesn't create a new value on every render,
  // which would cause the polling interval to be torn down and recreated constantly.
  const trainingStatusKey = useMemo(
    () => trainings.map((t) => `${t.id}:${t.status}`).join(','),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [trainings.map((t) => `${t.id}:${t.status}`).join(',')]
  );

  // Polling fallback for running trainings (in case WebSocket isn't working)
  useEffect(() => {
    const runningTrainingIds = trainings
      .filter((t) => t.status === TrainingStatus.RUNNING || t.status === TrainingStatus.INITIALIZING)
      .map((t) => t.id);

    if (runningTrainingIds.length === 0) {
      return;
    }

    // Poll every 5 seconds - silent update to avoid UI flicker/collapse
    const pollInterval = setInterval(() => {
      runningTrainingIds.forEach((id) => {
        fireAndForget(fetchTraining(id, true)); // silent=true prevents loading state changes
      });
    }, 5000);

    return () => clearInterval(pollInterval);
  }, [trainingStatusKey, fetchTraining]);

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

  // Autodiscover hidden dimension and training layers from first selected extraction
  useEffect(() => {
    if (!config.extraction_ids || config.extraction_ids.length === 0) return;
    if (!config.model_id) return;

    // Use the first extraction for autodiscovery (all should share layers/hidden_dim)
    const firstExtId = config.extraction_ids[0];
    const extraction = availableExtractions.find((e: any) => e.extraction_id === firstExtId);
    if (!extraction) return;

    const updates: any = {};

    // Autodiscover training_layers from extraction layer_indices
    if (extraction.layer_indices && Array.isArray(extraction.layer_indices)) {
      const extractionLayers = extraction.layer_indices.sort((a: number, b: number) => a - b);
      const currentLayers = config.training_layers || [];
      const layersDiffer = extractionLayers.length !== currentLayers.length ||
        extractionLayers.some((layer: number, i: number) => layer !== currentLayers[i]);
      if (layersDiffer) {
        updates.training_layers = extractionLayers;
      }
    }

    // Autodiscover hidden_dim from extraction statistics
    if (extraction.statistics) {
      const layerNames = Object.keys(extraction.statistics);
      if (layerNames.length > 0) {
        const firstLayerStats = extraction.statistics[layerNames[0]];
        if (firstLayerStats?.shape && Array.isArray(firstLayerStats.shape) && firstLayerStats.shape.length === 3) {
          const hiddenDim = firstLayerStats.shape[2];
          if (hiddenDim !== config.hidden_dim) {
            updates.hidden_dim = hiddenDim;
          }
        }
      }
    }

    if (Object.keys(updates).length > 0) {
      updateConfig(updates);
    }
  }, [config.extraction_ids, config.model_id, availableExtractions]);

  // Update latent_dim when hidden_dim or latent multiplier changes
  useEffect(() => {
    const calculatedLatentDim = config.hidden_dim * latentMultiplier;
    if (calculatedLatentDim !== config.latent_dim) {
      updateConfig({ latent_dim: calculatedLatentDim });
    }
  }, [config.hidden_dim, latentMultiplier]);

  // Check for tokenizer/model vocabulary mismatch (check first dataset)
  const selectedDatasets = (config.dataset_ids || []).map(id => datasets.find(d => d.id === id)).filter(Boolean);
  const firstSelectedDataset = selectedDatasets[0];
  const vocabMismatch = useMemo(() => {
    if (!selectedModel || !firstSelectedDataset) return null;

    const datasetTokenizerName = firstSelectedDataset.metadata?.tokenization?.tokenizer_name;
    const datasetVocabSize = firstSelectedDataset.metadata?.tokenization?.vocab_size;
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
  }, [selectedModel, firstSelectedDataset]);

  // Validation - if cached activations enabled, all datasets with available extractions must have one selected
  const allExtractionsSelected = !config.extraction_ids || (
    config.dataset_ids?.every(dsId => {
      const dsExtractions = extractionsPerDataset[dsId] || [];
      return dsExtractions.length === 0 || config.extraction_ids!.some(eid =>
        dsExtractions.some((ext: any) => ext.extraction_id === eid)
      );
    }) ?? true
  );
  // EVERY selected extraction must contain the layers being trained.
  //
  // training_layers is autodiscovered from the FIRST selected extraction, and
  // the rest were submitted unchecked. The server does validate — correctly,
  // with a clear message — but only after creating a training row, so one
  // mistake produced three identical failed records 15 seconds apart
  // (2026-08-27). Catch it in the form, name the offender, and let the user
  // pick a different extraction instead.
  const extractionLayerMismatches = useMemo(() => {
    const wanted = config.training_layers || [];
    if (!config.extraction_ids || config.extraction_ids.length === 0) return [];
    if (wanted.length === 0) return [];

    return config.extraction_ids.flatMap((eid) => {
      const ext: any = availableExtractions.find(
        (e: any) => e.extraction_id === eid,
      );
      if (!ext) return [];
      const available: number[] = Array.isArray(ext.layer_indices)
        ? ext.layer_indices
        : [];
      const missing = wanted.filter((l) => !available.includes(l));
      if (missing.length === 0) return [];

      const ds = datasets.find((d) => String(d.id) === String(ext.dataset_id));
      return [{
        extractionId: eid,
        datasetName: ds?.name || String(ext.dataset_id ?? 'unknown dataset'),
        has: available,
        missing,
      }];
    });
  }, [config.extraction_ids, config.training_layers, availableExtractions, datasets]);

  const isFormValid = config.model_id && config.dataset_ids && config.dataset_ids.length > 0 && config.training_layers && config.training_layers.length > 0 && !vocabMismatch && allExtractionsSelected && extractionLayerMismatches.length === 0;

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

    // IMPORTANT: Wait for React to re-render and WebSocket to subscribe
    // This prevents a race condition where the Celery task completes before
    // the frontend is listening for deletion progress events
    await new Promise(resolve => setTimeout(resolve, 150));

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

    if (!config.model_id || !config.dataset_ids || config.dataset_ids.length === 0) {
      setSaveTemplateError('Please select a model and at least one dataset first');
      return;
    }

    setIsSavingTemplate(true);
    setSaveTemplateError(null);

    try {
      // Create template from current config
      const fwConfig = getFrameworkConfig(config.architecture_type);
      await createTemplate({
        name: templateName.trim(),
        description: templateDescription.trim() || undefined,
        model_id: config.model_id,
        dataset_ids: config.dataset_ids,
        encoder_type: config.architecture_type as any,
        hyperparameters: {
          hidden_dim: config.hidden_dim,
          latent_dim: config.latent_dim,
          architecture_type: config.architecture_type,
          training_layers: config.training_layers,
          learning_rate: config.learning_rate,
          batch_size: config.batch_size,
          total_steps: config.total_steps,
          warmup_steps: config.warmup_steps,
          weight_decay: config.weight_decay,
          grad_clip_norm: config.grad_clip_norm,
          checkpoint_interval: config.checkpoint_interval,
          normalize_activations: config.normalize_activations,
          // L1 frameworks
          ...(fwConfig.sparsityType === 'l1' && {
            l1_alpha: config.l1_alpha,
            target_l0: config.target_l0,
            sparsity_warmup_steps: config.sparsity_warmup_steps,
            normalize_decoder: config.normalize_decoder,
            resample_dead_neurons: config.resample_dead_neurons,
          }),
          // JumpReLU
          ...(fwConfig.sparsityType === 'l0' && {
            sparsity_coeff: config.sparsity_coeff,
            initial_threshold: config.initial_threshold,
            bandwidth: config.bandwidth,
            normalize_decoder: config.normalize_decoder,
            sparsity_warmup_steps: config.sparsity_warmup_steps,
          }),
          // TopK
          ...(fwConfig.sparsityType === 'topk' && {
            top_k: config.top_k,
            aux_k: config.aux_k,
            aux_loss_alpha: config.aux_loss_alpha,
            adam_epsilon: config.adam_epsilon,
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
        dataset_ids: config.dataset_ids,
        hidden_dim: config.hidden_dim,
        latent_dim: config.latent_dim,
        isFormValid,
      });
      alert('Please fill in all required fields (Model, at least one Dataset, Hidden Dim, Latent Dim)');
      return;
    }

    console.log('[TrainingPanel] Starting training with config:', config);
    setIsStarting(true);
    try {
      const fwConfig = getFrameworkConfig(config.architecture_type);
      const request: TrainingCreateRequest = {
        model_id: config.model_id,
        dataset_ids: config.dataset_ids,
        ...(config.extraction_ids && config.extraction_ids.length > 0 && { extraction_ids: config.extraction_ids }),
        hyperparameters: {
          hidden_dim: config.hidden_dim,
          latent_dim: config.latent_dim,
          architecture_type: config.architecture_type,
          training_layers: config.training_layers || [0],
          hook_types: config.hook_types || ['residual'],
          normalize_activations: config.normalize_activations,
          learning_rate: config.learning_rate,
          batch_size: config.batch_size,
          total_steps: config.total_steps,
          warmup_steps: config.warmup_steps,
          sparsity_warmup_steps: config.sparsity_warmup_steps,
          weight_decay: config.weight_decay,
          grad_clip_norm: config.grad_clip_norm,
          checkpoint_interval: config.checkpoint_interval,
          log_interval: config.log_interval,
          dead_neuron_threshold: config.dead_neuron_threshold,
          // L1 frameworks
          ...(fwConfig.sparsityType === 'l1' && {
            l1_alpha: config.l1_alpha,
            target_l0: config.target_l0,
            normalize_decoder: config.normalize_decoder,
            resample_dead_neurons: config.resample_dead_neurons,
          }),
          // JumpReLU
          ...(fwConfig.sparsityType === 'l0' && {
            sparsity_coeff: config.sparsity_coeff,
            initial_threshold: config.initial_threshold,
            bandwidth: config.bandwidth,
            normalize_decoder: config.normalize_decoder,
          }),
          // TopK
          ...(fwConfig.sparsityType === 'topk' && {
            top_k: config.top_k,
            aux_k: config.aux_k,
            aux_loss_alpha: config.aux_loss_alpha,
            adam_epsilon: config.adam_epsilon,
          }),
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
      <div className="px-6 py-8 space-y-6">
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
        <div className="bg-white dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">
            Training Configuration
          </h3>

          {/* Basic Configuration */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            {/* Dataset Selection - Multi-select for combining datasets */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                  Datasets ({config.dataset_ids?.length || 0} selected)
                </label>
                {readyDatasets.length > 0 && (
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => updateConfig({ dataset_ids: readyDatasets.map(d => d.id) })}
                      className={`px-3 py-1 text-xs ${COMPONENTS.button.secondary}`}
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      onClick={() => updateConfig({ dataset_ids: [] })}
                      className={`px-3 py-1 text-xs ${COMPONENTS.button.secondary}`}
                    >
                      Deselect All
                    </button>
                  </div>
                )}
              </div>
              {readyDatasets.length > 0 ? (
                <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                  {readyDatasets.map((dataset) => {
                    const isSelected = config.dataset_ids?.includes(dataset.id) || false;
                    return (
                      <button
                        key={dataset.id}
                        type="button"
                        onClick={() => {
                          const currentDatasets = config.dataset_ids || [];
                          if (isSelected) {
                            updateConfig({
                              dataset_ids: currentDatasets.filter((id) => id !== dataset.id),
                            });
                          } else {
                            updateConfig({
                              dataset_ids: [...currentDatasets, dataset.id],
                            });
                          }
                        }}
                        className={`px-3 py-2 text-sm text-left rounded transition-colors truncate ${
                          isSelected
                            ? 'bg-emerald-600 text-white hover:bg-emerald-700'
                            : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
                        }`}
                        title={dataset.name}
                      >
                        {dataset.name}
                      </button>
                    );
                  })}
                </div>
              ) : (
                <p className="text-sm text-slate-500">No datasets ready for training</p>
              )}
            </div>

            {/* Model Selection - Second in flow */}
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Model
              </label>
              <select
                value={config.model_id}
                onChange={(e) => updateConfig({ model_id: e.target.value })}
                className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
              >
                <option value="">Select model...</option>
                {readyModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Training Framework Selection - Third in flow */}
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Training Framework
              </label>
              <select
                value={config.architecture_type}
                onChange={(e) =>
                  updateConfig({ architecture_type: e.target.value as SAEArchitectureType })
                }
                className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
              >
                {getFrameworkOptions().map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-slate-500">
                {getFrameworkConfig(config.architecture_type).description}
              </p>
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
                  <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
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
                  checked={!!config.extraction_ids}
                  onChange={(e) => {
                    if (e.target.checked) {
                      updateConfig({ extraction_ids: [] });
                    } else {
                      updateConfig({ extraction_ids: undefined });
                    }
                  }}
                  className="mt-1 w-4 h-4 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded focus:ring-2 focus:ring-emerald-500"
                />
                <div className="flex-1">
                  <label htmlFor="use-cached-activations" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1 cursor-pointer">
                    Use Cached Activations (10-20x faster training)
                  </label>
                  <div className="flex items-center gap-2 mb-2">
                    <p className="text-xs text-slate-500 flex-1">
                      Select a cached extraction for each dataset. Requires a completed extraction per dataset for the selected model.
                    </p>
                    {/* An extraction finishing elsewhere used to be invisible here
                        until the panel was remounted. The automatic refreshes
                        should cover it; this is the escape hatch when they do
                        not. */}
                    <button
                      type="button"
                      onClick={() => refreshExtractions()}
                      disabled={isLoadingExtractions || !config.model_id}
                      title="Re-read the extraction list"
                      className="text-xs px-2 py-0.5 rounded border border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-50"
                    >
                      {isLoadingExtractions ? 'Refreshing…' : 'Refresh'}
                    </button>
                  </div>
                  {config.extraction_ids !== undefined && (
                    <>
                      {extractionFetchError && !isLoadingExtractions && (
                        <div className="flex items-center gap-2 text-xs text-red-400 mb-2 px-1">
                          <AlertCircle size={13} className="shrink-0" />
                          <span>{extractionFetchError}</span>
                        </div>
                      )}
                      {isLoadingExtractions ? (
                        <div className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 text-sm flex items-center gap-2">
                          <Loader size={14} className="animate-spin" />
                          <span>Loading available extractions...</span>
                        </div>
                      ) : !config.model_id || !config.dataset_ids || config.dataset_ids.length === 0 ? (
                        <div className="text-sm text-slate-600 dark:text-slate-400 italic">
                          Select a model and datasets first.
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {config.dataset_ids.map(dsId => {
                            const ds = datasets.find(d => d.id === dsId);
                            const dsName = ds?.name || dsId;
                            const dsExtractions = extractionsPerDataset[dsId] || [];
                            const selectedExtId = config.extraction_ids?.find(eid =>
                              dsExtractions.some((ext: any) => ext.extraction_id === eid)
                            );

                            return (
                              <div key={dsId} className="flex items-center gap-2">
                                <div className="w-4 flex-shrink-0">
                                  {selectedExtId ? (
                                    <CheckCircle className="w-4 h-4 text-emerald-500" />
                                  ) : dsExtractions.length > 0 ? (
                                    <AlertTriangle className="w-4 h-4 text-amber-500" />
                                  ) : (
                                    <XCircle className="w-4 h-4 text-slate-600" />
                                  )}
                                </div>
                                <span className="text-xs text-slate-600 dark:text-slate-400 w-36 truncate flex-shrink-0" title={dsName}>{dsName}</span>
                                {dsExtractions.length > 0 ? (
                                  <select
                                    value={selectedExtId || ''}
                                    onChange={(e) => {
                                      const newExtId = e.target.value;
                                      const currentIds = (config.extraction_ids || []).filter(eid =>
                                        !dsExtractions.some((ext: any) => ext.extraction_id === eid)
                                      );
                                      if (newExtId) currentIds.push(newExtId);
                                      updateConfig({ extraction_ids: currentIds });
                                    }}
                                    className={`flex-1 px-2 py-1 bg-white dark:bg-slate-800 border rounded text-slate-900 dark:text-slate-100 text-xs focus:outline-none focus:border-emerald-500 ${
                                      !selectedExtId ? 'border-amber-600/50' : 'border-slate-300 dark:border-slate-700'
                                    }`}
                                  >
                                    <option value="">Select...</option>
                                    {dsExtractions.map((ext: any) => {
                                      // Name the LAYERS, not just how many. Two
                                      // extractions of one dataset — layer 45, and
                                      // layers 44+46 — both read "…L" and could not
                                      // be told apart (2026-08-27).
                                      const layers = formatLayerIndices(ext.layer_indices);
                                      const sampleCount = ext.num_samples_processed || ext.samples_processed || 0;
                                      return (
                                        <option key={ext.extraction_id} value={ext.extraction_id}>
                                          {layers} · {sampleCount.toLocaleString()} samples · {ext.created_at ? new Date(ext.created_at).toLocaleDateString() : '?'}
                                        </option>
                                      );
                                    })}
                                  </select>
                                ) : (
                                  <span className="flex-1 text-xs text-slate-600 italic">No extraction available</span>
                                )}
                              </div>
                            );
                          })}
                          {extractionLayerMismatches.length > 0 && (
                            <div
                              role="alert"
                              className="mt-1 p-2 bg-red-900/20 border border-red-600/40 rounded-md flex gap-2"
                            >
                              <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                              <div className="text-xs text-red-400 space-y-1">
                                <p>
                                  Training layers {formatLayerIndices(config.training_layers)} are
                                  not present in every selected extraction. Choose an extraction
                                  that covers them, or change the training layers.
                                </p>
                                <ul className="space-y-0.5">
                                  {extractionLayerMismatches.map((m) => (
                                    <li key={m.extractionId}>
                                      <span className="font-medium">{m.datasetName}</span>
                                      {' — selected extraction has '}
                                      {formatLayerIndices(m.has)}
                                      {', missing '}
                                      {formatLayerIndices(m.missing)}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          )}
                          {datasetsMissingExtractions.length > 0 && (
                            <div className="mt-1 p-2 bg-amber-900/20 border border-amber-600/40 rounded-md flex gap-2">
                              <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
                              <p className="text-xs text-amber-400">
                                {datasetsMissingExtractions.length} dataset{datasetsMissingExtractions.length > 1 ? 's have' : ' has'} no cached extractions.
                                Run extractions in the Extractions panel, or uncheck to train on-the-fly.
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Template Selector - Only show when model and datasets are selected */}
            {config.model_id && config.dataset_ids && config.dataset_ids.length > 0 ? (
              <TemplateSelector
                modelId={config.model_id}
                datasetId={config.dataset_ids[0]}
                onTemplateLoad={handleTemplateLoad}
              />
            ) : (
              <div></div>
            )}
          </div>

          {/* Training Layers Selection */}
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
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
                          : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
                      }`}
                    >
                      L{layerIdx}
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="px-4 py-8 text-center text-slate-500 bg-slate-100 dark:bg-slate-800/50 rounded-lg border border-slate-300 dark:border-slate-700">
                {config.model_id
                  ? 'Loading model architecture...'
                  : 'Select a model to choose training layers'}
              </div>
            )}
          </div>

          {/* Hook Types Selection */}
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                Hook Types ({config.hook_types?.length || 0} selected)
              </label>
              <div className="text-xs text-slate-500">
                SAEs trained per layer × hook type
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {(['residual', 'mlp', 'attention'] as const).map((hookType) => {
                const isSelected = config.hook_types?.includes(hookType) || false;
                const displayNames: Record<string, string> = {
                  'residual': 'Residual Stream',
                  'mlp': 'MLP Output',
                  'attention': 'Attention Output',
                };
                return (
                  <button
                    key={hookType}
                    type="button"
                    onClick={() => {
                      const currentTypes = config.hook_types || ['residual'];
                      if (isSelected) {
                        // Prevent deselecting the last hook type
                        if (currentTypes.length > 1) {
                          updateConfig({
                            hook_types: currentTypes.filter((t) => t !== hookType) as any,
                          });
                        }
                      } else {
                        updateConfig({
                          hook_types: [...currentTypes, hookType] as any,
                        });
                      }
                    }}
                    className={`px-3 py-2 text-sm font-medium rounded transition-colors ${
                      isSelected
                        ? 'bg-emerald-600 text-white hover:bg-emerald-700'
                        : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
                    }`}
                  >
                    {displayNames[hookType]}
                  </button>
                );
              })}
            </div>
            {config.hook_types && config.hook_types.length > 1 && config.training_layers && config.training_layers.length > 0 && (
              <div className="mt-2 text-xs text-amber-500">
                ⚠️ Training {config.training_layers.length} layers × {config.hook_types.length} hook types = {config.training_layers.length * config.hook_types.length} SAEs
              </div>
            )}
          </div>

          {/* Memory Estimation Display */}
          <div className="mt-4 p-4 bg-slate-100 dark:bg-slate-800/50 border border-slate-300 dark:border-slate-700 rounded-lg">
            <div className="flex items-start justify-between">
              <div>
                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Estimated GPU Memory
                </h4>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-600 dark:text-slate-400">Total:</span>
                    <span className={`text-sm font-mono font-semibold ${
                      memoryEstimate.fits_in_6gb ? 'text-emerald-400' : 'text-orange-400'
                    }`}>
                      {formatMemorySize(memoryEstimate.total_gb)}
                    </span>
                    {numSAEs > 1 && (
                      <>
                        <span className="text-xs text-slate-500">×</span>
                        <span className="text-xs text-slate-600 dark:text-slate-400">
                          {formatMemorySize(memoryEstimate.per_layer_gb)} per SAE
                        </span>
                      </>
                    )}
                  </div>
                  <div className="text-xs text-slate-500">
                    {numSAEs} SAE{numSAEs !== 1 ? 's' : ''}
                    {numSAEs > 1 && (
                      <span className="text-slate-600">
                        {' '}({config.training_layers?.length || 1}L × {config.hook_types?.length || 1}H)
                      </span>
                    )}
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
            <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700">
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
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  {config.extraction_ids && config.extraction_ids.length > 0 && (
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
                      className="w-24 px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                    />
                    <span className="text-slate-600 dark:text-slate-400 text-sm font-mono">
                      × {config.hidden_dim} = <span className="text-emerald-400">{config.latent_dim}</span>
                    </span>
                  </div>
                </div>

                {/* === Framework-Specific Sparsity Section === */}

                {/* L1 Alpha — visible for L1 frameworks */}
                {isFieldVisible(config.architecture_type, 'l1_alpha') && (
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
                        value={config.l1_alpha ?? 5e-4}
                        onChange={(e) => updateConfig({ l1_alpha: parseFloat(e.target.value) })}
                        min={0.00001}
                        max={100.0}
                        step={config.architecture_type === SAEArchitectureType.STANDARD_ANTHROPIC ? 0.1 : 0.00001}
                        className="flex-1 px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                      />
                      {config.architecture_type !== SAEArchitectureType.STANDARD_ANTHROPIC && (
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
                      )}
                    </div>
                    {/* Sparsity Warnings */}
                    {config.l1_alpha != null && (() => {
                      const warnings = validateSparsityConfig(config.l1_alpha, config.latent_dim, config.target_l0 ?? 0.05);
                      return warnings.length > 0 ? (
                        <div className="mt-2 space-y-1">
                          {warnings.map((warning, idx) => (
                            <div key={idx} className="flex items-start gap-2 text-xs text-cyan-400">
                              <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>{warning}</span>
                            </div>
                          ))}
                        </div>
                      ) : null;
                    })()}
                    {config.architecture_type === SAEArchitectureType.STANDARD_ANTHROPIC && (
                      <p className="mt-1 text-xs text-slate-500">
                        Anthropic normalization uses L1 ~5.0 (not ~5e-4). Scale is different from SAELens.
                      </p>
                    )}
                  </div>
                )}

                {/* Target L0 — visible for L1 frameworks */}
                {isFieldVisible(config.architecture_type, 'target_l0') && (
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
                      className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                    />
                  </div>
                )}

                {/* TopK Parameters — visible for TopK framework */}
                {isFieldVisible(config.architecture_type, 'top_k') && (
                  <>
                    <div className="col-span-2 mt-4 mb-2">
                      <div className="flex items-center gap-2 pb-2 border-b border-slate-200 dark:border-slate-700">
                        <span className="text-sm font-semibold text-emerald-400">TopK Parameters</span>
                        <span className="text-xs text-slate-500">(Gao et al. 2024)</span>
                      </div>
                    </div>

                    {/* Top K */}
                    <div>
                      <HyperparameterLabel
                        paramName="top_k"
                        label="K (Active Features)"
                        htmlFor="top-k"
                        className="mb-2"
                      />
                      <div className="flex items-center gap-3">
                        <input
                          id="top-k"
                          type="number"
                          value={config.top_k ?? 64}
                          onChange={(e) => updateConfig({ top_k: parseInt(e.target.value) || 64 })}
                          min={1}
                          max={config.latent_dim}
                          step={1}
                          className="w-32 px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                        />
                        {config.top_k && config.latent_dim > 0 && (
                          <span className="text-slate-600 dark:text-slate-400 text-sm font-mono">
                            of {config.latent_dim.toLocaleString()} features{' '}
                            <span className="text-emerald-400">({((config.top_k / config.latent_dim) * 100).toFixed(2)}%)</span>
                          </span>
                        )}
                      </div>
                      {config.top_k && config.top_k > config.latent_dim * 0.10 && (
                        <p className="mt-1 text-xs text-cyan-400 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                          K is &gt;10% of features — quite dense. Typical range: 32-256.
                        </p>
                      )}
                    </div>

                    {/* Aux K */}
                    <div>
                      <HyperparameterLabel
                        paramName="aux_k"
                        label="Aux K (Dead Features)"
                        htmlFor="aux-k"
                        className="mb-2"
                      />
                      <input
                        id="aux-k"
                        type="number"
                        value={config.aux_k ?? ''}
                        onChange={(e) => updateConfig({ aux_k: e.target.value ? parseInt(e.target.value) : undefined })}
                        min={1}
                        max={config.latent_dim}
                        step={1}
                        placeholder={`Default: ${config.top_k ?? 64}`}
                        className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors placeholder:text-slate-400 dark:placeholder:text-slate-500"
                      />
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                        Number of dead features for auxiliary loss. Defaults to K.
                      </p>
                    </div>

                    {/* Aux Loss Alpha */}
                    <div>
                      <HyperparameterLabel
                        paramName="aux_loss_alpha"
                        label="Aux Loss Coefficient"
                        htmlFor="aux-loss-alpha"
                        className="mb-2"
                      />
                      <input
                        id="aux-loss-alpha"
                        type="number"
                        value={config.aux_loss_alpha ?? 1/32}
                        onChange={(e) => updateConfig({ aux_loss_alpha: parseFloat(e.target.value) })}
                        min={0}
                        max={1.0}
                        step={0.001}
                        className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                      />
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                        Coefficient for auxiliary dead feature loss. Default: 1/32 ≈ 0.03125.
                      </p>
                    </div>

                    {/* Adam Epsilon */}
                    <div>
                      <HyperparameterLabel
                        paramName="adam_epsilon"
                        label="Adam Epsilon"
                        htmlFor="adam-epsilon"
                        className="mb-2"
                      />
                      <input
                        id="adam-epsilon"
                        type="text"
                        value={config.adam_epsilon ?? 6.25e-10}
                        onChange={(e) => updateConfig({ adam_epsilon: parseFloat(e.target.value) || 6.25e-10 })}
                        className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                      />
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                        TopK uses very small epsilon (6.25e-10) per Gao et al. Default Adam: 1e-8.
                      </p>
                    </div>
                  </>
                )}

                {/* JumpReLU Parameters — visible for JumpReLU framework */}
                {isFieldVisible(config.architecture_type, 'sparsity_coeff') && (
                  <>
                    <div className="col-span-2 mt-4 mb-2">
                      <div className="flex items-center gap-2 pb-2 border-b border-slate-200 dark:border-slate-700">
                        <span className="text-sm font-semibold text-emerald-400">JumpReLU Parameters</span>
                        <span className="text-xs text-slate-500">(Rajamanoharan et al. 2024)</span>
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
                        value={config.sparsity_coeff ?? 1e-3}
                        onChange={(e) => updateConfig({ sparsity_coeff: parseFloat(e.target.value) })}
                        min={0.000001}
                        max={10.0}
                        step={0.00001}
                        className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                      />
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                        L0 penalty coefficient (λ) applied to raw feature count. Default: 1e-3 (Gemma Scope: 6e-4).
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
                        value={config.initial_threshold ?? 0.5}
                        onChange={(e) => updateConfig({ initial_threshold: parseFloat(e.target.value) })}
                        min={0.001}
                        max={5.0}
                        step={0.1}
                        className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                      />
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                        Starting threshold for JumpReLU activation. Should match pre-activation magnitude (~0.5).
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
                        value={config.bandwidth ?? 0.01}
                        onChange={(e) => updateConfig({ bandwidth: parseFloat(e.target.value) })}
                        min={0.001}
                        max={0.5}
                        step={0.001}
                        className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                      />
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                        Smoothness of STE gradient estimation. Wider = more features get gradient. Default: 0.01.
                      </p>
                    </div>
                  </>
                )}

                {/* Normalize Decoder — visible for L1 and JumpReLU frameworks */}
                {isFieldVisible(config.architecture_type, 'normalize_decoder') && (
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={config.normalize_decoder ?? true}
                          onChange={(e) => updateConfig({ normalize_decoder: e.target.checked })}
                          className="w-4 h-4 rounded bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                        />
                        <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                          Normalize Decoder Columns
                        </span>
                      </label>
                      <HyperparameterTooltip paramName="normalize_decoder" />
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                      Normalizes decoder columns to unit norm after each step.
                    </p>
                  </div>
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
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  >
                    <option value="constant_norm_rescale">Constant Norm Rescale (SAELens)</option>
                    <option value="anthropic_rescale">Anthropic Rescale (E[||x||²]=d_model)</option>
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
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
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
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
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
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
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
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Sparsity Warmup Steps — hidden for TopK */}
                {isFieldVisible(config.architecture_type, 'sparsity_warmup_steps') && (
                  <div>
                    <HyperparameterLabel
                      paramName="sparsity_warmup_steps"
                      label="Sparsity Warmup Steps"
                      htmlFor="sparsity-warmup-steps"
                      className="mb-2"
                    />
                    <input
                      id="sparsity-warmup-steps"
                      type="number"
                      value={config.sparsity_warmup_steps ?? 5000}
                      onChange={(e) => updateConfig({ sparsity_warmup_steps: parseInt(e.target.value) })}
                      min={0}
                      max={100000}
                      step={1000}
                      className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                    />
                    <p className="text-xs text-slate-500 mt-1">Ramps sparsity penalty from 0 to full. Prevents dead neurons.</p>
                  </div>
                )}

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
                    value={config.weight_decay ?? 0.0}
                    onChange={(e) => updateConfig({ weight_decay: parseFloat(e.target.value) })}
                    min={0}
                    max={0.1}
                    step={0.001}
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
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
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
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
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
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
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>

                {/* Dead Neuron Threshold — hidden for TopK */}
                {isFieldVisible(config.architecture_type, 'dead_neuron_threshold') && (
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
                      className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                    />
                  </div>
                )}

                {/* Resample Interval — hidden for TopK/JumpReLU */}
                {isFieldVisible(config.architecture_type, 'resample_interval') && (
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
                      className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                    />
                  </div>
                )}

                {/* Resample Dead Neurons — hidden for TopK/JumpReLU */}
                {isFieldVisible(config.architecture_type, 'resample_dead_neurons') && (
                  <div className="col-span-2">
                    <div className="flex items-center gap-2">
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={config.resample_dead_neurons ?? true}
                          onChange={(e) =>
                            updateConfig({ resample_dead_neurons: e.target.checked })
                          }
                          className="w-4 h-4 rounded bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                        />
                        <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                          Resample dead neurons during training
                        </span>
                      </label>
                      <HyperparameterTooltip paramName="resample_dead_neurons" />
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-6 pt-4 border-t border-slate-200 dark:border-slate-700 flex gap-3">
            <button
              onClick={() => {
                // Pre-populate template name and description from current config
                const defaults = generateTemplateDefaults;
                setTemplateName(defaults.name);
                setTemplateDescription(defaults.description);
                setShowSaveTemplateModal(true);
              }}
              disabled={!config.model_id || !config.dataset_ids || config.dataset_ids.length === 0}
              className="flex items-center justify-center gap-2 py-3 px-4 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed text-slate-900 dark:text-slate-100 disabled:text-slate-500 rounded-md transition-colors"
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
        <div className="bg-white dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Training Jobs</h3>
              {trainings.length > 0 && (
                <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 cursor-pointer hover:text-slate-700 dark:hover:text-slate-300">
                  <input
                    type="checkbox"
                    checked={selectedTrainingIds.size === trainings.length && trainings.length > 0}
                    onChange={handleSelectAll}
                    className="w-4 h-4 rounded bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
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
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              All ({statusCounts.all})
            </button>
            <button
              onClick={() => setStatusFilter(TrainingStatus.RUNNING)}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                statusFilter === TrainingStatus.RUNNING
                  ? 'bg-emerald-600 text-white'
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
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
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
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
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
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
              <p className="text-slate-600 dark:text-slate-400 mb-2">No training jobs yet</p>
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
            className="bg-white dark:bg-slate-900 rounded-lg max-w-lg w-full p-6 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100">Save as Template</h2>
              <button
                onClick={() => {
                  setShowSaveTemplateModal(false);
                  setSaveTemplateError(null);
                  setTemplateName('');
                  setTemplateDescription('');
                }}
                className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
                title="Close modal"
              >
                <X className="w-5 h-5 text-slate-600 dark:text-slate-400" />
              </button>
            </div>

            <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
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
                <label htmlFor="template-name" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Template Name <span className="text-red-400">*</span>
                </label>
                <input
                  id="template-name"
                  type="text"
                  value={templateName}
                  onChange={(e) => setTemplateName(e.target.value)}
                  placeholder="e.g., TinyLlama_OpenWebText_Standard_L1-0.0001"
                  className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 text-sm focus:outline-none focus:border-emerald-500 transition-colors"
                  disabled={isSavingTemplate}
                />
              </div>

              {/* Template Description */}
              <div>
                <label htmlFor="template-description" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Description (Optional)
                </label>
                <textarea
                  id="template-description"
                  value={templateDescription}
                  onChange={(e) => setTemplateDescription(e.target.value)}
                  placeholder="e.g., L1: 0.0001 | LR: 0.00027 | Dict: 2048→8192 (4x) | Steps: 50k"
                  rows={3}
                  className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-slate-900 dark:text-slate-100 text-sm focus:outline-none focus:border-emerald-500 transition-colors resize-none"
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
                  className="flex-1 px-4 py-2 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-700 dark:text-slate-300 disabled:text-slate-600 rounded-md transition-colors"
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
