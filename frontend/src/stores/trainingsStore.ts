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
} from '../types/training';
import { SAEArchitectureType } from '../types/training';

/**
 * Training configuration form state.
 * Used for collecting user inputs before creating a training job.
 */
export interface TrainingConfig {
  // Target configuration
  model_id: string;
  dataset_id: string;
  extraction_id?: string;

  // SAE Architecture
  hidden_dim: number;
  latent_dim: number;
  architecture_type: SAEArchitectureType;

  // Layer configuration
  training_layers: number[];

  // Sparsity
  l1_alpha: number;
  target_l0?: number;

  // Training
  learning_rate: number;
  batch_size: number;
  total_steps: number;
  warmup_steps?: number;

  // Optimization
  weight_decay?: number;
  grad_clip_norm?: number;

  // Checkpointing
  checkpoint_interval?: number;
  log_interval?: number;

  // Dead neuron handling
  dead_neuron_threshold?: number;
  resample_dead_neurons?: boolean;
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
}

/**
 * Training store actions.
 */
interface TrainingStoreActions {
  // Training CRUD operations
  fetchTrainings: (page?: number, limit?: number) => Promise<void>;
  fetchTraining: (trainingId: string) => Promise<void>;
  createTraining: (config: TrainingCreateRequest) => Promise<Training>;
  deleteTraining: (trainingId: string) => Promise<void>;
  retryTraining: (trainingId: string) => Promise<Training>;

  // Training control operations
  pauseTraining: (trainingId: string) => Promise<TrainingControlResponse>;
  resumeTraining: (trainingId: string) => Promise<TrainingControlResponse>;
  stopTraining: (trainingId: string) => Promise<TrainingControlResponse>;

  // Checkpoint operations
  fetchCheckpoints: (trainingId: string) => Promise<Checkpoint[]>;
  saveCheckpoint: (trainingId: string) => Promise<Checkpoint>;
  deleteCheckpoint: (trainingId: string, checkpointId: string) => Promise<void>;

  // Configuration management
  updateConfig: (updates: Partial<TrainingConfig>) => void;
  resetConfig: () => void;
  setConfigFromTraining: (training: Training) => void;

  // Real-time updates (WebSocket handler)
  updateTrainingStatus: (trainingId: string, updates: Partial<Training>) => void;

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
  dataset_id: '',
  extraction_id: undefined,

  // SAE Architecture - typical values for 768-dim transformer hidden states
  hidden_dim: 768,
  latent_dim: 8192, // 8-16x expansion ratio
  architecture_type: SAEArchitectureType.STANDARD,

  // Layer configuration - default to layer 0 (single-layer training)
  training_layers: [0],

  // Sparsity - moderate L1 penalty (recommended: 10/sqrt(latent_dim))
  l1_alpha: 0.003,
  target_l0: 0.05, // 5% activation rate

  // Training - conservative defaults
  learning_rate: 0.0003,
  batch_size: 32,
  total_steps: 100000,
  warmup_steps: 10000,

  // Optimization
  weight_decay: 0.01,
  grad_clip_norm: 1.0,

  // Checkpointing
  checkpoint_interval: 1000,
  log_interval: 100,

  // Dead neuron handling
  dead_neuron_threshold: 10000,
  resample_dead_neurons: true,
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
 *       dataset_id: 'my_dataset',
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
  fetchTraining: async (trainingId: string) => {
    set({ isLoading: true, error: null });

    try {
      const response = await axios.get<Training>(`${API_BASE_URL}/${trainingId}`);

      // Update the training in the list
      set((state) => ({
        trainings: state.trainings.map((t) => (t.id === trainingId ? response.data : t)),
        selectedTraining: response.data,
        isLoading: false,
      }));
    } catch (error: any) {
      set({
        error: error.response?.data?.message || 'Failed to fetch training',
        isLoading: false,
      });
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
        dataset_id: failedTraining.dataset_id,
        extraction_id: failedTraining.extraction_id || undefined,
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
      return response.data.data;
    } catch (error: any) {
      console.error('Failed to fetch checkpoints:', error);
      throw error;
    }
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
  deleteCheckpoint: async (trainingId: string, checkpointId: string) => {
    try {
      await axios.delete(`${API_BASE_URL}/${trainingId}/checkpoints/${checkpointId}`);
    } catch (error: any) {
      console.error('Failed to delete checkpoint:', error);
      throw error;
    }
  },

  /**
   * Update training configuration form.
   *
   * @param updates - Partial configuration updates
   */
  updateConfig: (updates: Partial<TrainingConfig>) => {
    set((state) => ({
      config: { ...state.config, ...updates },
    }));
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
        dataset_id: training.dataset_id,
        extraction_id: training.extraction_id || undefined,
        ...training.hyperparameters,
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
    // Refetch trainings with new filter
    get().fetchTrainings(1);
  },

  /**
   * Set model filter for training list.
   *
   * @param modelId - Model ID to filter by
   */
  setModelFilter: (modelId: string | null) => {
    set({ modelFilter: modelId });
    // Refetch trainings with new filter
    get().fetchTrainings(1);
  },

  /**
   * Set dataset filter for training list.
   *
   * @param datasetId - Dataset ID to filter by
   */
  setDatasetFilter: (datasetId: string | null) => {
    set({ datasetFilter: datasetId });
    // Refetch trainings with new filter
    get().fetchTrainings(1);
  },

  /**
   * Clear error message.
   */
  clearError: () => {
    set({ error: null });
  },
}));
