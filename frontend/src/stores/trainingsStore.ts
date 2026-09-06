/**
 * Training Store
 *
 * Zustand store for SAE Training feature state management.
 * Manages training jobs, configurations, and real-time updates via WebSocket.
 *
 * Backend API Contract:
 * - GET /api/v1/trainings - List all training jobs
 * - GET /api/v1/trainings/:id - Get training job details
 * - POST /api/v1/trainings - Create and start new training job
 * - POST /api/v1/trainings/:id/control - Control training (pause/resume/stop)
 * - GET /api/v1/trainings/:id/metrics - Get training metrics
 * - GET /api/v1/trainings/:id/checkpoints - List checkpoints
 *
 * WebSocket Events:
 * - trainings/{training_id}/progress - Training progress updates
 * - trainings/{training_id}/checkpoints - Checkpoint creation events
 */

import { create } from 'zustand';
import axios from 'axios';
import type {
  Training,
  TrainingStatus,
  TrainingCreateRequest,
  TrainingListResponse,
  TrainingControlRequest,
  TrainingControlResponse,
  Checkpoint,
  CheckpointListResponse,
  CheckpointPrunePreview,
} from '../types/training';
import { SAEArchitectureType } from '../types/training';
import { getFrameworkConfig } from '../config/frameworkConfigs';

/**
 * Training configuration form state.
 * Used for collecting user inputs before creating a training job.
 */
export interface TrainingConfig {
  // Target configuration
  model_id: string;
  dataset_ids: string[];  // Supports multiple datasets
  extraction_ids?: string[];

  // SAE Architecture
  hidden_dim: number;
  latent_dim: number;
  architecture_type: SAEArchitectureType;

  // Layer configuration
  training_layers: number[];
  // Hook types (default: ['residual']). Multiple types create separate SAEs per layer/hook combination.
  hook_types: ('residual' | 'mlp' | 'attention')[];

  // Sparsity (L1-based frameworks)
  l1_alpha?: number;
  target_l0?: number;
  /** @deprecated Use top_k instead */
  top_k_sparsity?: number;
  normalize_activations?: string;

  // TopK-specific parameters (Gao et al. 2024)
  top_k?: number;
  aux_k?: number;
  aux_loss_alpha?: number;
  adam_epsilon?: number;

  // JumpReLU-specific parameters (Rajamanoharan et al. 2024)
  initial_threshold?: number;
  bandwidth?: number;
  sparsity_coeff?: number;
  normalize_decoder?: boolean;

  // Training
  learning_rate: number;
  batch_size: number;
  total_steps: number;
  warmup_steps?: number;
  sparsity_warmup_steps?: number;

  // Optimization
  weight_decay?: number;
  grad_clip_norm?: number;

  // Checkpointing
  checkpoint_interval?: number;
  log_interval?: number;

  // Dead neuron handling
  dead_neuron_threshold?: number;
  resample_dead_neurons?: boolean;
  resample_interval?: number;
}

/**
 * Training store state.
 */
interface TrainingStoreState {
  // Training jobs list
  trainings: Training[];

  // Selected training for detail view
  selectedTraining: Training | null;

  // Training configuration form
  config: TrainingConfig;

  // UI state
  isLoading: boolean;
  error: string | null;

  // Pagination
  currentPage: number;
  totalPages: number;
  totalTrainings: number;

  // Filters
  statusFilter: TrainingStatus | 'all';
  modelFilter: string | null;
  datasetFilter: string | null;

  // Status counts (from backend)
  statusCounts: {
    all: number;
    running: number;
    completed: number;
    failed: number;
  };

  // Checkpoints cache: trainingId -> Checkpoint[]
  checkpoints: Record<string, Checkpoint[]>;
}

/**
 * Training store actions.
 */
interface TrainingStoreActions {
  // Training CRUD operations
  fetchTrainings: (page?: number, limit?: number) => Promise<void>;
  fetchTraining: (trainingId: string, silent?: boolean) => Promise<void>;
  createTraining: (config: TrainingCreateRequest) => Promise<Training>;
  deleteTraining: (trainingId: string) => Promise<void>;
  retryTraining: (trainingId: string) => Promise<Training>;

  // Training control operations
  pauseTraining: (trainingId: string) => Promise<TrainingControlResponse>;
  resumeTraining: (trainingId: string) => Promise<TrainingControlResponse>;
  stopTraining: (trainingId: string) => Promise<TrainingControlResponse>;
  /** Stop a run AND write community_format from its newest checkpoint. */
  stopAndFinalizeTraining: (trainingId: string) => Promise<TrainingControlResponse>;
  /** Finalize an already-stopped run so its SAE becomes importable. */
  finalizeTraining: (
    trainingId: string,
    checkpointStep?: number,
    opts?: { allowFailed?: boolean; force?: boolean }
  ) => Promise<void>;
  /** Report which checkpoints the retention policy would delete (read-only). */
  previewCheckpointPrune: (trainingId: string) => Promise<CheckpointPrunePreview>;
  /** Run the retention policy against one training now. */
  pruneCheckpoints: (trainingId: string) => Promise<void>;

  // Checkpoint operations
  fetchCheckpoints: (trainingId: string) => Promise<Checkpoint[]>;
  saveCheckpoint: (trainingId: string) => Promise<Checkpoint>;
  deleteCheckpoint: (
    trainingId: string,
    checkpointId: string,
    allowBest?: boolean
  ) => Promise<void>;

  // Configuration management
  updateConfig: (updates: Partial<TrainingConfig>) => void;
  resetConfig: () => void;
  setConfigFromTraining: (training: Training) => void;

  // Real-time updates (WebSocket handler)
  updateTrainingStatus: (trainingId: string, updates: Partial<Training>) => void;
  addCheckpoint: (trainingId: string, checkpoint: Checkpoint) => void;

  // UI state management
  setSelectedTraining: (training: Training | null) => void;
  setStatusFilter: (status: TrainingStatus | 'all') => void;
  setModelFilter: (modelId: string | null) => void;
  setDatasetFilter: (datasetId: string | null) => void;
  clearError: () => void;
}

type TrainingStore = TrainingStoreState & TrainingStoreActions;

/**
 * Default training configuration.
 * Provides sensible defaults for SAE training.
 */
const defaultConfig: TrainingConfig = {
  model_id: '',
  dataset_ids: [],
  extraction_ids: undefined,

  // SAE Architecture - typical values for 768-dim transformer hidden states
  hidden_dim: 768,
  latent_dim: 8192,
  architecture_type: SAEArchitectureType.STANDARD_SAELENS,

  // Layer configuration
  training_layers: [0],
  hook_types: ['residual'],

  // Sparsity — defaults match Standard (SAELens) framework
  l1_alpha: 5e-4,
  target_l0: 0.05,
  normalize_activations: 'constant_norm_rescale',

  // JumpReLU-specific (applied when framework switches to jumprelu)
  initial_threshold: 0.5,
  bandwidth: 0.01,
  sparsity_coeff: 1e-3,
  normalize_decoder: true,

  // TopK-specific (applied when framework switches to topk)
  top_k: undefined,
  aux_k: undefined,
  aux_loss_alpha: undefined,
  adam_epsilon: undefined,

  // Training
  learning_rate: 4e-4,
  batch_size: 2048,
  total_steps: 50000,
  warmup_steps: 2000,
  sparsity_warmup_steps: 5000,

  // Optimization
  weight_decay: 0.0,
  grad_clip_norm: 1.0,

  // Checkpointing
  checkpoint_interval: 2000,
  log_interval: 100,

  // Dead neuron handling
  dead_neuron_threshold: 10000,
  resample_dead_neurons: true,
  resample_interval: 5000,
};

/**
 * API base URL for training endpoints.
 */
const API_BASE_URL = '/api/v1/trainings';

/**
 * Training store using Zustand.
 *
 * This store manages the state for the SAE Training feature, including:
 * - Training job list and filtering
 * - Training configuration form
 * - Real-time updates via WebSocket
 * - Training control operations (pause/resume/stop)
 *
 * Usage:
 *   const { trainings, fetchTrainings, createTraining } = useTrainingsStore();
 *
 *   useEffect(() => {
 *     fetchTrainings();
 *   }, []);
 *
 *   const handleStart = async () => {
 *     const training = await createTraining({
 *       model_id: 'gpt2',
 *       dataset_ids: ['my_dataset', 'another_dataset'],
 *       hyperparameters: {...}
 *     });
 *   };
 */
export const useTrainingsStore = create<TrainingStore>((set, get) => ({
  // Initial state
  trainings: [],
  selectedTraining: null,
  config: { ...defaultConfig },
  isLoading: false,
  error: null,
  currentPage: 1,
  totalPages: 1,
  totalTrainings: 0,
  statusFilter: 'all',
  modelFilter: null,
  datasetFilter: null,
  statusCounts: {
    all: 0,
    running: 0,
    completed: 0,
    failed: 0,
  },
  checkpoints: {},

  /**
   * Fetch list of training jobs.
   *
   * @param page - Page number (1-indexed)
   * @param limit - Number of items per page
   */
  fetchTrainings: async (page = 1, limit = 50) => {
    set({ isLoading: true, error: null });

    try {
      const { statusFilter, modelFilter, datasetFilter } = get();

      // Build query parameters
      const params: Record<string, any> = {
        page,
        limit,
      };

      if (statusFilter !== 'all') {
        params.status = statusFilter;
      }
      if (modelFilter) {
        params.model_id = modelFilter;
      }
      if (datasetFilter) {
        params.dataset_id = datasetFilter;
      }

      // Fetch trainings
      const response = await axios.get<TrainingListResponse>(API_BASE_URL, { params });

      set({
        trainings: response.data.data,
        currentPage: response.data.pagination.page,
        totalPages: response.data.pagination.total_pages,
        totalTrainings: response.data.pagination.total,
        statusCounts: response.data.status_counts,
        isLoading: false,
      });
    } catch (error: any) {
      set({
        error: error.response?.data?.message || 'Failed to fetch trainings',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Fetch a single training job by ID.
   *
   * @param trainingId - Training job ID
   */
  fetchTraining: async (trainingId: string, silent = false) => {
    // Only set loading state if not silent (for polling updates)
    if (!silent) {
      set({ isLoading: true, error: null });
    }

    try {
      const response = await axios.get<Training>(`${API_BASE_URL}/${trainingId}`);

      // Update the training in the list (silently - no loading state change)
      set((state) => ({
        trainings: state.trainings.map((t) => (t.id === trainingId ? response.data : t)),
        selectedTraining: state.selectedTraining?.id === trainingId ? response.data : state.selectedTraining,
        ...(silent ? {} : { isLoading: false }),
      }));
    } catch (error: any) {
      if (!silent) {
        set({
          error: error.response?.data?.message || 'Failed to fetch training',
          isLoading: false,
        });
      }
      throw error;
    }
  },

  /**
   * Create and start a new training job.
   *
   * @param config - Training configuration
   * @returns Created training job
   */
  createTraining: async (config: TrainingCreateRequest) => {
    set({ isLoading: true, error: null });

    try {
      const response = await axios.post<Training>(API_BASE_URL, config);

      // Add the new training to the list
      set((state) => ({
        trainings: [response.data, ...state.trainings],
        totalTrainings: state.totalTrainings + 1,
        isLoading: false,
      }));

      return response.data;
    } catch (error: any) {
      set({
        error: error.response?.data?.message || 'Failed to create training',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Delete a training job.
   *
   * @param trainingId - Training job ID
   */
  deleteTraining: async (trainingId: string) => {
    set({ isLoading: true, error: null });

    try {
      await axios.delete(`${API_BASE_URL}/${trainingId}`);

      // Remove the training from the list
      set((state) => ({
        trainings: state.trainings.filter((t) => t.id !== trainingId),
        totalTrainings: state.totalTrainings - 1,
        selectedTraining:
          state.selectedTraining?.id === trainingId ? null : state.selectedTraining,
        isLoading: false,
      }));
    } catch (error: any) {
      set({
        error: error.response?.data?.message || 'Failed to delete training',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Retry a failed training job by creating a new training with the same configuration.
   *
   * @param trainingId - Failed training job ID
   * @returns New training job
   */
  retryTraining: async (trainingId: string) => {
    set({ isLoading: true, error: null });

    try {
      // Get the failed training to extract its configuration
      const state = get();
      const failedTraining = state.trainings.find((t) => t.id === trainingId);

      if (!failedTraining) {
        throw new Error('Training not found');
      }

      // Create new training request with same configuration
      const retryRequest: TrainingCreateRequest = {
        model_id: failedTraining.model_id,
        dataset_ids: failedTraining.dataset_ids,
        extraction_ids: failedTraining.extraction_ids || (failedTraining.extraction_id ? [failedTraining.extraction_id] : undefined),
        hyperparameters: failedTraining.hyperparameters,
      };

      // Create the new training
      const response = await axios.post<Training>(API_BASE_URL, retryRequest);

      // Add the new training to the list
      set((state) => ({
        trainings: [response.data, ...state.trainings],
        totalTrainings: state.totalTrainings + 1,
        isLoading: false,
      }));

      return response.data;
    } catch (error: any) {
      set({
        error: error.response?.data?.message || 'Failed to retry training',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Pause a running training job.
   *
   * @param trainingId - Training job ID
   * @returns Control response
   */
  pauseTraining: async (trainingId: string) => {
    set({ isLoading: true, error: null });

    try {
      const request: TrainingControlRequest = { action: 'pause' };
      const response = await axios.post<TrainingControlResponse>(
        `${API_BASE_URL}/${trainingId}/control`,
        request
      );

      // Update the training status in the store
      get().updateTrainingStatus(trainingId, { status: response.data.status });

      set({ isLoading: false });
      return response.data;
    } catch (error: any) {
      set({
        error: error.response?.data?.message || 'Failed to pause training',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Resume a paused training job.
   *
   * @param trainingId - Training job ID
   * @returns Control response
   */
  resumeTraining: async (trainingId: string) => {
    set({ isLoading: true, error: null });

    try {
      const request: TrainingControlRequest = { action: 'resume' };
      const response = await axios.post<TrainingControlResponse>(
        `${API_BASE_URL}/${trainingId}/control`,
        request
      );

      // Update the training status in the store
      get().updateTrainingStatus(trainingId, { status: response.data.status });

      set({ isLoading: false });
      return response.data;
    } catch (error: any) {
      set({
        error: error.response?.data?.message || 'Failed to resume training',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Stop (cancel) a training job.
   *
   * @param trainingId - Training job ID
   * @returns Control response
   */
  stopTraining: async (trainingId: string) => {
    set({ isLoading: true, error: null });

    try {
      const request: TrainingControlRequest = { action: 'stop' };
      const response = await axios.post<TrainingControlResponse>(
        `${API_BASE_URL}/${trainingId}/control`,
        request
      );

      // Update the training status in the store
      get().updateTrainingStatus(trainingId, { status: response.data.status });

      set({ isLoading: false });
      return response.data;
    } catch (error: any) {
      set({
        error: error.response?.data?.message || 'Failed to stop training',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Stop a training and immediately finalize it from the newest checkpoint.
   *
   * Plain 'stop' leaves usable checkpoints that nothing downstream can read,
   * because cancelling skips the training loop's community_format export.
   */
  stopAndFinalizeTraining: async (trainingId: string) => {
    set({ isLoading: true, error: null });

    try {
      const request: TrainingControlRequest = { action: 'stop_and_finalize' };
      const response = await axios.post<TrainingControlResponse>(
        `${API_BASE_URL}/${trainingId}/control`,
        request
      );

      get().updateTrainingStatus(trainingId, { status: response.data.status });

      set({ isLoading: false });
      return response.data;
    } catch (error: any) {
      set({
        error:
          error.response?.data?.detail ||
          error.response?.data?.message ||
          'Failed to stop and finalize training',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Preview what pruning would delete for a training. Never deletes anything.
   */
  previewCheckpointPrune: async (trainingId: string) => {
    const response = await axios.get(
      `${API_BASE_URL}/${trainingId}/checkpoints/prune-preview`
    );
    return response.data.data as CheckpointPrunePreview;
  },

  /**
   * Prune this training's checkpoints now (honours dry-run and every guard).
   */
  pruneCheckpoints: async (trainingId: string) => {
    await axios.post(`${API_BASE_URL}/${trainingId}/checkpoints/prune`);
  },

  /**
   * Finalize an already-stopped training from a checkpoint.
   *
   * This is the rescue path for runs stopped before finalize existed: their
   * checkpoints are intact but no community_format was ever written.
   */
  finalizeTraining: async (
    trainingId: string,
    checkpointStep?: number,
    opts?: { allowFailed?: boolean; force?: boolean }
  ) => {
    set({ isLoading: true, error: null });

    try {
      // The API refuses a FAILED run without allow_failed and an already
      // COMPLETED run without force. Both must be sendable, or the 409 tells
      // the user to do something the product gives them no way to do.
      const params = new URLSearchParams();
      if (checkpointStep !== undefined) {
        params.set('checkpoint_step', String(checkpointStep));
      }
      if (opts?.allowFailed) params.set('allow_failed', 'true');
      if (opts?.force) params.set('force', 'true');
      const query = params.toString() ? `?${params.toString()}` : '';
      await axios.post(`${API_BASE_URL}/${trainingId}/finalize${query}`);
      set({ isLoading: false });
    } catch (error: any) {
      set({
        error:
          error.response?.data?.detail ||
          error.response?.data?.message ||
          'Failed to finalize training',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Fetch checkpoints for a training job.
   *
   * @param trainingId - Training job ID
   * @returns List of checkpoints
   */
  fetchCheckpoints: async (trainingId: string) => {
    try {
      const response = await axios.get<CheckpointListResponse>(
        `${API_BASE_URL}/${trainingId}/checkpoints`
      );
      const checkpointList = response.data.data;
      set((state) => ({
        checkpoints: { ...state.checkpoints, [trainingId]: checkpointList },
      }));
      return checkpointList;
    } catch (error: any) {
      console.error('Failed to fetch checkpoints:', error);
      throw error;
    }
  },

  addCheckpoint: (trainingId: string, checkpoint: Checkpoint) => {
    set((state) => {
      const existing = state.checkpoints[trainingId] ?? [];
      return {
        checkpoints: {
          ...state.checkpoints,
          [trainingId]: [...existing, checkpoint],
        },
      };
    });
  },

  /**
   * Save a checkpoint for a training job.
   *
   * @param trainingId - Training job ID
   * @returns Created checkpoint
   */
  saveCheckpoint: async (trainingId: string) => {
    try {
      const response = await axios.post<Checkpoint>(
        `${API_BASE_URL}/${trainingId}/checkpoints`
      );
      return response.data;
    } catch (error: any) {
      console.error('Failed to save checkpoint:', error);
      throw error;
    }
  },

  /**
   * Delete a checkpoint.
   *
   * @param trainingId - Training job ID
   * @param checkpointId - Checkpoint ID
   */
  deleteCheckpoint: async (
    trainingId: string,
    checkpointId: string,
    allowBest?: boolean
  ) => {
    try {
      // The API 409s on the best checkpoint unless deletion is explicit; without
      // sending this the UI can never honour its own error message.
      const query = allowBest ? '?allow_best=true' : '';
      await axios.delete(
        `${API_BASE_URL}/${trainingId}/checkpoints/${checkpointId}${query}`
      );
    } catch (error: any) {
      console.error('Failed to delete checkpoint:', error);
      throw error;
    }
  },

  /**
   * Update training configuration form.
   * When architecture_type changes, automatically applies framework defaults
   * for learning_rate, sparsity params, and normalization.
   *
   * @param updates - Partial configuration updates
   */
  updateConfig: (updates: Partial<TrainingConfig>) => {
    set((state) => {
      const newConfig = { ...state.config, ...updates };

      // When architecture_type changes, apply framework-specific defaults
      if (updates.architecture_type && updates.architecture_type !== state.config.architecture_type) {
        const fw = getFrameworkConfig(updates.architecture_type);
        const fwDefaults = fw.defaults;

        // Apply framework defaults for optimizer and sparsity params
        newConfig.learning_rate = fwDefaults.learning_rate;
        newConfig.normalize_activations = fwDefaults.normalize_activations;
        newConfig.sparsity_warmup_steps = fwDefaults.sparsity_warmup_steps;

        if (fwDefaults.normalize_decoder !== undefined) {
          newConfig.normalize_decoder = fwDefaults.normalize_decoder;
        }
        if (fwDefaults.resample_dead_neurons !== undefined) {
          newConfig.resample_dead_neurons = fwDefaults.resample_dead_neurons;
        }

        // Apply sparsity-type-specific defaults
        if (fw.sparsityType === 'l1') {
          newConfig.l1_alpha = fwDefaults.l1_alpha;
          // Clear non-L1 fields
          newConfig.top_k = undefined;
          newConfig.aux_k = undefined;
          newConfig.aux_loss_alpha = undefined;
          newConfig.adam_epsilon = undefined;
          newConfig.sparsity_coeff = undefined;
          newConfig.initial_threshold = undefined;
          newConfig.bandwidth = undefined;
        } else if (fw.sparsityType === 'l0') {
          newConfig.sparsity_coeff = fwDefaults.sparsity_coeff;
          newConfig.initial_threshold = fwDefaults.initial_threshold;
          newConfig.bandwidth = fwDefaults.bandwidth;
          // Clear non-L0 fields
          newConfig.l1_alpha = undefined;
          newConfig.top_k = undefined;
          newConfig.aux_k = undefined;
          newConfig.aux_loss_alpha = undefined;
          newConfig.adam_epsilon = undefined;
        } else if (fw.sparsityType === 'topk') {
          newConfig.top_k = fwDefaults.top_k;
          newConfig.aux_loss_alpha = fwDefaults.aux_loss_alpha;
          newConfig.adam_epsilon = fwDefaults.adam_epsilon;
          // Clear non-TopK fields
          newConfig.l1_alpha = undefined;
          newConfig.sparsity_coeff = undefined;
          newConfig.initial_threshold = undefined;
          newConfig.bandwidth = undefined;
        }
      }

      return { config: newConfig };
    });
  },

  /**
   * Reset training configuration to defaults.
   */
  resetConfig: () => {
    set({ config: { ...defaultConfig } });
  },

  /**
   * Set configuration from an existing training job.
   * Useful for re-running or modifying previous training configs.
   *
   * @param training - Training job to copy config from
   */
  setConfigFromTraining: (training: Training) => {
    set({
      config: {
        model_id: training.model_id,
        dataset_ids: training.dataset_ids,
        extraction_ids: training.extraction_ids || (training.extraction_id ? [training.extraction_id] : undefined),
        ...training.hyperparameters,
        // Ensure hook_types has a default value for backward compatibility
        hook_types: training.hyperparameters.hook_types || ['residual'],
      },
    });
  },

  /**
   * Update training status in real-time (WebSocket handler).
   *
   * This function is called when WebSocket events are received
   * to update the training job state without refetching from the API.
   *
   * @param trainingId - Training job ID
   * @param updates - Partial training updates
   */
  updateTrainingStatus: (trainingId: string, updates: Partial<Training>) => {
    set((state) => ({
      trainings: state.trainings.map((t) =>
        t.id === trainingId ? { ...t, ...updates } : t
      ),
      selectedTraining:
        state.selectedTraining?.id === trainingId
          ? { ...state.selectedTraining, ...updates }
          : state.selectedTraining,
    }));
  },

  /**
   * Set the selected training for detail view.
   *
   * @param training - Training job to select
   */
  setSelectedTraining: (training: Training | null) => {
    set({ selectedTraining: training });
  },

  /**
   * Set status filter for training list.
   *
   * @param status - Status to filter by, or 'all'
   */
  setStatusFilter: (status: TrainingStatus | 'all') => {
    set({ statusFilter: status });
    // Defer the fetch to the next microtask so any batched synchronous set()
    // calls in the same tick all complete before fetchTrainings() reads state.
    // This prevents the filter value from being stale when multiple filters
    // are updated in rapid succession.
    Promise.resolve().then(() => get().fetchTrainings(1));
  },

  /**
   * Set model filter for training list.
   *
   * @param modelId - Model ID to filter by
   */
  setModelFilter: (modelId: string | null) => {
    set({ modelFilter: modelId });
    Promise.resolve().then(() => get().fetchTrainings(1));
  },

  /**
   * Set dataset filter for training list.
   *
   * @param datasetId - Dataset ID to filter by
   */
  setDatasetFilter: (datasetId: string | null) => {
    set({ datasetFilter: datasetId });
    Promise.resolve().then(() => get().fetchTrainings(1));
  },

  /**
   * Clear error message.
   */
  clearError: () => {
    set({ error: null });
  },
}));
